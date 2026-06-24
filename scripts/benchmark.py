"""
Multi-run benchmark for the research pipeline.

Runs a set of research prompts through the SAME pipeline used by the web app
(planner -> executor -> research/writer/editor + tools), evaluates each run with
the LLM-as-judge, and aggregates real-company metrics:

  average cost, average latency, P95 latency, average token usage,
  tool success rate, error rate, average sources retrieved,
  retrieval precision / recall(proxy) / F1 / MRR, citation coverage,
  report quality (overall + instruction-following + faithfulness).

Outputs:
  benchmark/benchmark_results.json   (machine-readable aggregate + per-run)
  README.md                          (updates the BENCHMARK section in place)
  runs/trace_*.json/.txt, runs/arch_*.png   (one per run, same as the web app)

Usage (from the multi_agent folder):
  ./venv/Scripts/python.exe scripts/benchmark.py [N]
N defaults to 5; the full prompt set has 14 prompts (so 10-20 is supported).
"""

import json
import math
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.planning_agent import planner_agent, executor_agent_step  # noqa: E402
from src.trace_logger import Tracer, set_active_tracer, clear_active_tracer  # noqa: E402
from src.run_diagram import render_run_architecture  # noqa: E402
from src.evaluation import evaluate_run  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(BASE_DIR, "benchmark")
README_PATH = os.path.join(BASE_DIR, "README.md")

PROMPTS = [
    "Retrieval-augmented generation for enterprise search in 2024-2025: chunking strategies and reranking trade-offs.",
    "Vector database indexing methods (HNSW vs IVF-PQ): recall, build cost and query latency trade-offs.",
    "Mixture-of-Experts large language models: routing methods and inference efficiency advances 2023-2025.",
    "Long-context transformers: attention optimizations and their effect on inference latency.",
    "Speculative decoding for LLM inference: speedups and quality trade-offs.",
    "LLM-as-a-judge evaluation: reliability, known biases and mitigation strategies.",
    "Quantization of LLMs (GPTQ, AWQ, GGUF): accuracy versus memory and latency trade-offs.",
    "Agentic tool-use frameworks for LLMs: planning and reliability in 2024-2025.",
    "Graph neural networks for recommendation systems: recent scalable approaches.",
    "Diffusion models for high-resolution image synthesis: sampling speed improvements.",
    "Parameter-efficient fine-tuning (LoRA / QLoRA): when it matches full fine-tuning.",
    "Knowledge distillation for small language models: recent recipes and limits.",
    "Federated learning privacy guarantees: differential privacy in practice.",
    "RLHF alternatives (DPO, ORPO): trade-offs versus PPO-based alignment.",
]


def run_one(prompt: str, idx: int, total: int) -> dict:
    task_id = str(uuid.uuid4())
    tracer = Tracer(task_id=task_id, prompt=prompt)
    set_active_tracer(tracer)
    print(f"\n[{idx}/{total}] {prompt[:80]}")
    try:
        plan = planner_agent(prompt)
        tracer.set_plan(plan)
        history = []
        for i, step in enumerate(plan):
            tracer.begin_step(i, step)
            _, agent, out = executor_agent_step(step, history, prompt)
            history.append([step, agent, out])
            tracer.end_step(i, "done", out, agent)
        tracer.set_final_report(history[-1][-1] if history else "")
        tracer.finalize("done")
        try:
            tracer.set_evaluation(evaluate_run(tracer))
        except Exception as exc:  # noqa: BLE001
            print(f"   evaluation failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        tracer.finalize("error", error=str(exc))
        print(f"   run failed: {exc}")

    try:
        tracer.write()
        render_run_architecture(tracer)
    except Exception as exc:  # noqa: BLE001
        print(f"   persist failed: {exc}")
    clear_active_tracer()

    m = tracer.metrics()
    print(
        f"   done: tokens={m['total_tokens']} cost=${m['cost_usd']:.4f} "
        f"latency={m['wall_time_s']}s tool_ok={m['tool_success_rate']} "
        f"sources={m['sources_retrieved']} quality={m['quality_overall']}"
    )
    return m


def _nums(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def _percentile(vals, p):
    vals = sorted(v for v in vals if isinstance(v, (int, float)))
    if not vals:
        return None
    k = max(0, math.ceil(p * len(vals)) - 1)
    return round(vals[k], 4)


def aggregate(rows: list) -> dict:
    n = len(rows)
    errors = sum(1 for r in rows if r.get("status") != "done")
    return {
        "avg_cost_usd": _avg(_nums(rows, "cost_usd")),
        "avg_latency_s": _avg(_nums(rows, "wall_time_s")),
        "p50_latency_s": _percentile(_nums(rows, "wall_time_s"), 0.50),
        "p95_latency_s": _percentile(_nums(rows, "wall_time_s"), 0.95),
        "p99_latency_s": _percentile(_nums(rows, "wall_time_s"), 0.99),
        "avg_total_tokens": _avg(_nums(rows, "total_tokens")),
        "avg_prompt_tokens": _avg(_nums(rows, "prompt_tokens")),
        "avg_completion_tokens": _avg(_nums(rows, "completion_tokens")),
        "tool_success_rate": _avg(_nums(rows, "tool_success_rate")),
        "error_rate": round(errors / n, 4) if n else 0.0,
        "avg_sources_retrieved": _avg(_nums(rows, "sources_retrieved")),
        "avg_citation_coverage": _avg(_nums(rows, "citation_coverage")),
        "avg_quality_overall": _avg(_nums(rows, "quality_overall")),
        "avg_instruction_following": _avg(_nums(rows, "instruction_following")),
        "avg_faithfulness": _avg(_nums(rows, "faithfulness")),
        "avg_retrieval_precision": _avg(_nums(rows, "retrieval_precision")),
        "avg_retrieval_recall": _avg(_nums(rows, "retrieval_recall")),
        "avg_retrieval_f1": _avg(_nums(rows, "retrieval_f1")),
        "avg_retrieval_mrr": _avg(_nums(rows, "retrieval_mrr")),
    }


def _fmt(v, nd=2, pct=False):
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.0f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(agg: dict, rows: list, n: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"_Last run: {ts} · {n} research queries · judge = gpt-4.1-mini_\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Average cost (USD) | ${_fmt(agg['avg_cost_usd'], 4)} |")
    lines.append(f"| Average latency | {_fmt(agg['avg_latency_s'], 1)} s |")
    lines.append(f"| P50 latency | {_fmt(agg['p50_latency_s'], 1)} s |")
    lines.append(f"| P95 latency | {_fmt(agg['p95_latency_s'], 1)} s |")
    lines.append(f"| P99 latency | {_fmt(agg['p99_latency_s'], 1)} s |")
    lines.append(f"| Average total tokens | {_fmt(agg['avg_total_tokens'], 0)} |")
    lines.append(f"| Average prompt tokens | {_fmt(agg['avg_prompt_tokens'], 0)} |")
    lines.append(f"| Tool success rate | {_fmt(agg['tool_success_rate'], pct=True)} |")
    lines.append(f"| Error rate | {_fmt(agg['error_rate'], pct=True)} |")
    lines.append(f"| Average sources retrieved | {_fmt(agg['avg_sources_retrieved'], 1)} |")
    lines.append(f"| Average citation coverage | {_fmt(agg['avg_citation_coverage'], pct=True)} |")
    lines.append(f"| Retrieval precision (LLM-judged) | {_fmt(agg['avg_retrieval_precision'], 2)} |")
    lines.append(f"| Retrieval recall (coverage proxy) | {_fmt(agg['avg_retrieval_recall'], 2)} |")
    lines.append(f"| Retrieval F1 | {_fmt(agg['avg_retrieval_f1'], 2)} |")
    lines.append(f"| Retrieval MRR | {_fmt(agg['avg_retrieval_mrr'], 2)} |")
    lines.append(f"| Report quality (0-100) | {_fmt(agg['avg_quality_overall'], 1)} |")
    lines.append(f"| Instruction following (0-100) | {_fmt(agg['avg_instruction_following'], 1)} |")
    lines.append(f"| Faithfulness (0-100) | {_fmt(agg['avg_faithfulness'], 1)} |")
    lines.append("")
    lines.append("### Per-run")
    lines.append("")
    lines.append("| # | Query | Tokens | Cost $ | Latency s | Tool ok | Src | Cite cov | Quality | Prec | Rec* | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r.get('prompt','')[:48]} | {r.get('total_tokens','')} | "
            f"{_fmt(r.get('cost_usd'),4)} | {_fmt(r.get('wall_time_s'),1)} | "
            f"{_fmt(r.get('tool_success_rate'),pct=True)} | {r.get('sources_retrieved','')} | "
            f"{_fmt(r.get('citation_coverage'),pct=True)} | {_fmt(r.get('quality_overall'),0)} | "
            f"{_fmt(r.get('retrieval_precision'),2)} | {_fmt(r.get('retrieval_recall'),2)} | "
            f"{_fmt(r.get('retrieval_f1'),2)} |"
        )
    lines.append("")
    lines.append("\\* recall is an LLM coverage proxy (no labelled ground-truth set).")
    return "\n".join(lines)


def update_readme(section_md: str) -> None:
    start, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
    block = f"{start}\n{section_md}\n{end}"
    if os.path.exists(README_PATH):
        with open(README_PATH, "r", encoding="utf-8") as fh:
            content = fh.read()
        if start in content and end in content:
            pre = content.split(start)[0]
            post = content.split(end)[1]
            content = pre + block + post
        else:
            content = content.rstrip() + "\n\n## Benchmark results\n\n" + block + "\n"
        with open(README_PATH, "w", encoding="utf-8") as fh:
            fh.write(content)
    else:
        with open(README_PATH, "w", encoding="utf-8") as fh:
            fh.write("# Multi-Agent Research Pipeline\n\n## Benchmark results\n\n" + block + "\n")


def persist(rows: list, n: int) -> dict:
    """Write JSON + README after every run so partial progress is never lost."""
    agg = aggregate(rows)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_runs": len(rows),
        "n_target": n,
        "judge_model": "gpt-4.1-mini",
        "aggregate": agg,
        "runs": rows,
    }
    os.makedirs(BENCH_DIR, exist_ok=True)
    with open(os.path.join(BENCH_DIR, "benchmark_results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    section = render_markdown(agg, rows, len(rows))
    with open(os.path.join(BENCH_DIR, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("# Benchmark results\n\n" + section + "\n")
    update_readme(section)
    return agg


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n = max(1, min(n, len(PROMPTS)))
    os.makedirs(BENCH_DIR, exist_ok=True)
    print(f"Running benchmark over {n} prompts...")

    rows = []
    for i, prompt in enumerate(PROMPTS[:n], start=1):
        rows.append(run_one(prompt, i, n))
        agg = persist(rows, n)  # incremental save after each run
        print(f"   [saved {len(rows)}/{n}] avg_cost=${agg['avg_cost_usd']} "
              f"avg_latency={agg['avg_latency_s']}s tool_ok={agg['tool_success_rate']}")

    print("\n==== AGGREGATE ====")
    print(json.dumps(aggregate(rows), indent=2))
    print("\nWrote benchmark/benchmark_results.json and updated README.md")


if __name__ == "__main__":
    main()
