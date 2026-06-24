import os
import uuid
import json
import threading
from datetime import datetime
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

from src.planning_agent import planner_agent, executor_agent_step
from src.trace_logger import Tracer, set_active_tracer
from src.run_diagram import render_run_architecture

import html, textwrap

# === Load env vars ===
load_dotenv()

# Default to a local SQLite database so the project runs without an external
# Postgres server. Override DATABASE_URL in .env to use Postgres.
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./tasks.db"

# Fix for Heroku's postgres:// URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


# === DB setup ===
Base = declarative_base()
_engine_kwargs = {"echo": False, "future": True}
if DATABASE_URL.startswith("sqlite"):
    # SQLite connections are shared across the request and worker threads.
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine)


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    result = Column(Text)


try:
    Base.metadata.drop_all(bind=engine)
except Exception as e:
    print(f"\u274c DB creation failed: {e}")

try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"\u274c DB creation failed: {e}")

# === FastAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

task_progress = {}
task_tracers = {}
task_arch = {}


class PromptRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/api", response_class=JSONResponse)
def health_check(request: Request):
    return {"status": "ok"}


@app.post("/generate_report")
def generate_report(req: PromptRequest):
    task_id = str(uuid.uuid4())
    db = SessionLocal()
    db.add(Task(id=task_id, prompt=req.prompt, status="running"))
    db.commit()
    db.close()

    tracer = Tracer(task_id=task_id, prompt=req.prompt)
    task_tracers[task_id] = tracer
    set_active_tracer(tracer)

    task_progress[task_id] = {"steps": []}

    try:
        initial_plan_steps = planner_agent(req.prompt)
    except Exception as plan_exc:
        # Surface planning failures (e.g. missing API key) in the UI and trace
        # instead of returning an opaque 500.
        tracer.finalize("error", error=f"Planner failed: {plan_exc}")
        try:
            tracer.write()
            task_arch[task_id] = render_run_architecture(tracer)
        except Exception as side_exc:
            print(f"Failed to persist planner-error trace: {side_exc}")
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "error"
            task.updated_at = datetime.utcnow()
            db.commit()
        db.close()
        return {"task_id": task_id}

    tracer.set_plan(initial_plan_steps)
    for step_title in initial_plan_steps:
        task_progress[task_id]["steps"].append(
            {
                "title": step_title,
                "status": "pending",
                "description": "Awaiting execution",
                "substeps": [],
            }
        )

    thread = threading.Thread(
        target=run_agent_workflow, args=(task_id, req.prompt, initial_plan_steps)
    )
    thread.start()
    return {"task_id": task_id}


@app.get("/task_progress/{task_id}")
def get_task_progress(task_id: str):
    return task_progress.get(task_id, {"steps": []})


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": task.status,
        "result": json.loads(task.result) if task.result else None,
    }


@app.get("/task_trace/{task_id}")
def get_task_trace(task_id: str):
    """Live, machine-readable trace: plan, per-step events, tool I/O and telemetry."""
    tracer = task_tracers.get(task_id)
    if tracer is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return tracer.to_json()


@app.get("/task_trace_txt/{task_id}", response_class=PlainTextResponse)
def get_task_trace_txt(task_id: str):
    """Human-readable .txt analysis for the run (same content saved under runs/)."""
    tracer = task_tracers.get(task_id)
    if tracer is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return tracer.render_txt()


@app.get("/task_architecture/{task_id}")
def get_task_architecture(task_id: str):
    """PNG architecture diagram generated from what this run actually used."""
    path = task_arch.get(task_id)
    if not path or not os.path.exists(path):
        tracer = task_tracers.get(task_id)
        if tracer is not None:
            try:
                path = render_run_architecture(tracer)
                task_arch[task_id] = path
            except Exception:
                path = None
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Architecture not available yet")
    return FileResponse(path, media_type="image/png")


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.join(_BASE_DIR, "runs")
_BENCH_DIR = os.path.join(_BASE_DIR, "benchmark")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Telemetry + benchmark dashboard across all stored runs."""
    path = os.path.join(_BASE_DIR, "templates", "dashboard.html")
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/runs")
def list_runs():
    """Per-run metrics read from every trace_*.json under runs/."""
    out = []
    if os.path.isdir(_RUNS_DIR):
        for name in os.listdir(_RUNS_DIR):
            if name.startswith("trace_") and name.endswith(".json"):
                try:
                    with open(os.path.join(_RUNS_DIR, name), "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                metric = dict(data.get("metrics") or {})
                metric["file"] = name
                metric["finished_at"] = data.get("finished_at")
                task8 = (data.get("task_id") or "")[:8]
                metric["arch_url"] = f"/runs_file/arch_{task8}.png"
                metric["txt_url"] = f"/runs_file/{name[:-5]}.txt"
                ev = data.get("evaluation") or {}
                rep = ev.get("report") or {}
                metric["where_it_went_wrong"] = rep.get("where_it_went_wrong", "")
                out.append(metric)
    out.sort(key=lambda r: r.get("finished_at") or r.get("file", ""), reverse=True)
    return {"count": len(out), "runs": out}


@app.get("/benchmark")
def get_benchmark():
    """Aggregate benchmark results (written by scripts/benchmark.py)."""
    path = os.path.join(_BENCH_DIR, "benchmark_results.json")
    if not os.path.exists(path):
        return {"available": False}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["available"] = True
    return data


@app.get("/runs_file/{filename}")
def runs_file(filename: str):
    """Serve a saved trace/architecture file from runs/ (path-traversal safe)."""
    safe = os.path.basename(filename)
    path = os.path.join(_RUNS_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="not found")
    media = "image/png" if safe.endswith(".png") else "text/plain"
    return FileResponse(path, media_type=media)


def format_history(history):
    return "\n\n".join(
        f"🔹 {title}\n{desc}\n\n📝 Output:\n{output}" for title, desc, output in history
    )


def run_agent_workflow(task_id: str, prompt: str, initial_plan_steps: list):
    steps_data = task_progress[task_id]["steps"]
    execution_history = []
    tracer = task_tracers.get(task_id)
    if tracer is not None:
        # Re-bind the tracer in this worker thread (ContextVars do not cross threads).
        set_active_tracer(tracer)

    def update_step_status(index, status, description="", substep=None):
        if index < len(steps_data):
            steps_data[index]["status"] = status
            if description:
                steps_data[index]["description"] = description
            if substep:
                steps_data[index]["substeps"].append(substep)
            steps_data[index]["updated_at"] = datetime.utcnow().isoformat()

    try:
        for i, plan_step_title in enumerate(initial_plan_steps):
            update_step_status(i, "running", f"Executing: {plan_step_title}")
            if tracer is not None:
                tracer.begin_step(i, plan_step_title)

            actual_step_description, agent_name, output = executor_agent_step(
                plan_step_title, execution_history, prompt
            )

            execution_history.append([plan_step_title, actual_step_description, output])
            if tracer is not None:
                tracer.end_step(i, status="done", output=output, agent=agent_name)

            def esc(s: str) -> str:
                return html.escape(s or "")

            def nl2br(s: str) -> str:
                return esc(s).replace("\n", "<br>")

            # ...
            update_step_status(
                i,
                "done",
                f"Completed: {plan_step_title}",
                {
                    "title": f"Called {agent_name}",
                    "content": f"""
<div style='border:1px solid #ccc; border-radius:8px; padding:10px; margin:8px 0; background:#fff;'>
  <div style='font-weight:bold; color:#2563eb;'>📘 User Prompt</div>
  <div style='white-space:pre-wrap;'>{prompt}</div>

  <div style='font-weight:bold; color:#16a34a; margin-top:8px;'>📜 Previous Step</div>
  <pre style='white-space:pre-wrap; background:#f9fafb; padding:6px; border-radius:6px; margin:0;'>
{format_history(execution_history[-2:-1])}
  </pre>

  <div style='font-weight:bold; color:#f59e0b; margin-top:8px;'>🧹 Your next task</div>
  <div style='white-space:pre-wrap;'>{actual_step_description}</div>

  <div style='font-weight:bold; color:#10b981; margin-top:8px;'>✅ Output</div>
  <!-- ⚠️ NO <pre> AQUÍ -->
  <div style='white-space:pre-wrap;'>
{output}
  </div>
</div>
""".strip(),
                },
            )

        final_report_markdown = (
            execution_history[-1][-1] if execution_history else "No report generated."
        )

        result = {"html_report": final_report_markdown, "history": steps_data}

        if tracer is not None:
            tracer.set_final_report(final_report_markdown)
            tracer.finalize("done")
            if os.getenv("EVAL_ENABLED", "1") != "0":
                try:
                    from src.evaluation import evaluate_run

                    tracer.set_evaluation(evaluate_run(tracer))
                    print("Evaluation (LLM-as-judge) complete.")
                except Exception as eval_exc:
                    print(f"Evaluation failed: {eval_exc}")
            trace_path = tracer.write()
            print(f"Trace written to: {trace_path}")
            try:
                arch_path = render_run_architecture(tracer)
                task_arch[task_id] = arch_path
                print(f"Architecture image written to: {arch_path}")
            except Exception as arch_exc:
                print(f"Failed to render architecture: {arch_exc}")

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "done"
        task.result = json.dumps(result)
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()

    except Exception as e:
        print(f"Workflow error for task {task_id}: {e}")
        if tracer is not None:
            try:
                tracer.finalize("error", error=str(e))
                trace_path = tracer.write()
                print(f"Trace (error) written to: {trace_path}")
                try:
                    task_arch[task_id] = render_run_architecture(tracer)
                except Exception as arch_exc:
                    print(f"Failed to render architecture: {arch_exc}")
            except Exception as trace_exc:
                print(f"Failed to write trace: {trace_exc}")
        if steps_data:
            error_step_index = next(
                (i for i, s in enumerate(steps_data) if s["status"] == "running"),
                len(steps_data) - 1,
            )
            if error_step_index >= 0:
                update_step_status(
                    error_step_index,
                    "error",
                    f"Error during execution: {e}",
                    {"title": "Error", "content": str(e)},
                )

        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "error"
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()


if __name__ == "__main__":
    import uvicorn

    # Single entry point: run `python main.py` to start the whole app.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting Multi-Agent Research Pipeline on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)