"""
Observability / tracing layer for the multi-agent research pipeline.

This module records, for every run, a complete step-by-step trace of:
  - how the planner breaks the user query into a plan,
  - which agent runs each step and what input/context it receives,
  - the model's tool-call decisions (which tool, with which arguments),
  - the actual tool inputs and outputs (Tavily / arXiv / Wikipedia),
  - the final output of each agent,
  - tokens (prompt / completion / total), estimated cost (USD),
    latency (seconds) and any errors.

The trace is written to a human-readable .txt file and a machine-readable
.json file under multi_agent/runs/.

Integration points:
  - traced_completion(client, ...) wraps client.chat.completions.create and
    records one LLM event (tokens, cost, latency, tool decisions, errors).
  - wrap_tool(fn) wraps a tool callable and records its input/output/latency.
  - A Tracer is made active per task via a ContextVar so nested agent and tool
    calls attach to the correct run and step automatically.

Design goals: never break the pipeline. All recording is best-effort and
wrapped in try/except; if tracing fails the underlying call still proceeds.
"""

from __future__ import annotations

import contextvars
import functools
import json
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Active-tracer context
# --------------------------------------------------------------------------- #
# A ContextVar does not propagate to brand-new threads automatically, so the
# web layer explicitly calls set_active_tracer() at the top of both the request
# thread (for the planner) and the worker thread (for execution).
_active_tracer: "contextvars.ContextVar[Optional[Tracer]]" = contextvars.ContextVar(
    "active_tracer", default=None
)


def set_active_tracer(tracer: "Optional[Tracer]") -> None:
    _active_tracer.set(tracer)


def get_active_tracer() -> "Optional[Tracer]":
    return _active_tracer.get()


def clear_active_tracer() -> None:
    _active_tracer.set(None)


# --------------------------------------------------------------------------- #
# Pricing (USD per 1,000,000 tokens). Estimates - edit to match your account.
# --------------------------------------------------------------------------- #
PRICING_AS_OF = "2025 list-price estimates"
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o3": {"input": 2.00, "output": 8.00},
}

# Field length caps in the .txt render (the .json keeps full content).
TXT_INPUT_CHARS = 4000
TXT_OUTPUT_CHARS = 8000
TXT_TOOL_OUTPUT_CHARS = 6000

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")


def _normalize_model(model: str) -> str:
    return model.split(":", 1)[1] if ":" in model else model


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
    """Return (cost_usd, is_known_model).

    Cached prompt tokens (OpenAI prompt caching) are billed at a discount
    (~25% of the input price for the GPT-4.1 / o-series families), so pricing
    them separately gives the *correct*, lower cost instead of over-charging
    every repeated-context token at the full input rate.
    """
    name = _normalize_model(model)
    price = MODEL_PRICING.get(name)
    if price is None:
        for key, value in MODEL_PRICING.items():
            if name.startswith(key):
                price = value
                break
    if price is None:
        return 0.0, False
    cached_price = price.get("cached", price["input"] * 0.25)
    cached_tokens = max(0, min(cached_tokens or 0, prompt_tokens))
    non_cached = prompt_tokens - cached_tokens
    cost = (
        non_cached / 1_000_000.0 * price["input"]
        + cached_tokens / 1_000_000.0 * cached_price
        + completion_tokens / 1_000_000.0 * price["output"]
    )
    return cost, True


# --------------------------------------------------------------------------- #
# Extraction helpers (work against raw OpenAI responses returned by aisuite)
# --------------------------------------------------------------------------- #
def _extract_usage(response: Any) -> Dict[str, int]:
    """Sum token usage across the final response and any tool-loop turns."""
    responses = [response] + list(getattr(response, "intermediate_responses", []) or [])
    prompt = completion = total = api_calls = cached = 0
    for resp in responses:
        usage = getattr(resp, "usage", None)
        if not usage:
            continue
        prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
        completion += int(getattr(usage, "completion_tokens", 0) or 0)
        total += int(getattr(usage, "total_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached += int(getattr(details, "cached_tokens", 0) or 0)
        api_calls += 1
    if total == 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
        "api_calls": api_calls,
    }


def _extract_tool_decisions(response: Any) -> List[Dict[str, Any]]:
    """Recover each turn's tool-call decisions from the tool-loop responses."""
    decisions: List[Dict[str, Any]] = []
    turns = list(getattr(response, "intermediate_responses", []) or []) + [response]
    for turn_index, resp in enumerate(turns, start=1):
        try:
            message = resp.choices[0].message
        except Exception:
            continue
        for call in getattr(message, "tool_calls", None) or []:
            try:
                name = call.function.name
                arguments = call.function.arguments
            except Exception:
                name = getattr(call, "name", "?")
                arguments = "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass
            decisions.append({"turn": turn_index, "tool": name, "arguments": arguments})
    return decisions


def _safe_messages(messages: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, dict):
            role = message.get("role", "?")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "?")
            content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = _to_text(content)
        out.append({"role": str(role), "content": content})
    return out


def _to_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _kv(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return str(arguments)


def _trunc(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars; full content in .json]"


def _result_items(result: Any) -> List[Any]:
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


def _tool_failed(event: Dict[str, Any]) -> bool:
    """A tool call failed if it raised OR returned only error payloads.

    Tools like Tavily/arXiv/Wikipedia do not raise on failure; they return
    ``[{"error": ...}]``. A naive error count therefore reads 0 even when a
    tool failed, so we inspect the returned payload here.
    """
    if event.get("error"):
        return True
    items = _result_items(event.get("result"))
    if not items:
        return True
    dict_items = [it for it in items if isinstance(it, dict)]
    if dict_items and len(dict_items) == len(items):
        return all(it.get("error") for it in dict_items)
    return False


def _extract_sources_from_result(tool: str, result: Any) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for it in _result_items(result):
        if not isinstance(it, dict) or it.get("error"):
            continue
        url = (it.get("url") or it.get("link_pdf") or "").strip()
        title = (it.get("title") or "").strip()
        snippet = (it.get("content") or it.get("summary") or "")[:400]
        if url or title:
            sources.append(
                {"origin": tool, "title": title, "url": url, "snippet": snippet}
            )
    return sources


def _url_key(u: str) -> str:
    """Normalise a URL for matching: collapse arXiv abs/pdf + version to an ID."""
    u = (u or "").strip().lower()
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
    if m:
        return "arxiv:" + m.group(1)
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u.rstrip("/.,);]")


def compute_citation_metrics(sources: List[Dict[str, Any]], report: str) -> Dict[str, Any]:
    """How many retrieved sources are actually cited in the final report."""
    report = report or ""
    report_urls = re.findall(r"https?://[^\s)\"'<>\]]+", report)
    report_keys = {_url_key(u) for u in report_urls if u}
    cited = 0
    for s in sources:
        k = _url_key(s.get("url", ""))
        if not k:
            continue
        if k in report_keys or any(k in rk or rk in k for rk in report_keys):
            cited += 1
    n = len(sources)
    inline = len(set(re.findall(r"\[(\d{1,3})\]", report)))
    return {
        "sources_retrieved": n,
        "sources_cited": cited,
        "citation_coverage": round(cited / n, 4) if n else 0.0,
        "inline_citation_markers": inline,
        "urls_in_report": len(report_urls),
    }


# --------------------------------------------------------------------------- #
# Tracer
# --------------------------------------------------------------------------- #
class Tracer:
    """Collects and renders a full trace for a single pipeline run."""

    def __init__(self, task_id: str, prompt: str):
        self.task_id = task_id
        self.run_id = task_id
        self.prompt = prompt
        self.started_at = datetime.now()
        self.finished_at: Optional[datetime] = None
        self.status = "running"
        self.error: Optional[str] = None
        self.plan: List[str] = []
        self.final_report: str = ""
        self.evaluation: Dict[str, Any] = {}

        self._lock = threading.Lock()
        self._current_index: Optional[int] = None
        self.planner_events: List[Dict[str, Any]] = []
        self.steps: Dict[int, Dict[str, Any]] = {}
        self.loose_events: List[Dict[str, Any]] = []
        self.llm_events: List[Dict[str, Any]] = []
        self.tool_events: List[Dict[str, Any]] = []
        self.txt_path: Optional[str] = None
        self.json_path: Optional[str] = None

    # -- run-level metadata -------------------------------------------------- #
    def set_plan(self, plan: List[str]) -> None:
        with self._lock:
            self.plan = list(plan or [])

    def set_final_report(self, report: str) -> None:
        with self._lock:
            self.final_report = report or ""

    def finalize(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.finished_at = datetime.now()

    # -- steps --------------------------------------------------------------- #
    def begin_step(self, index: int, title: str, agent: Optional[str] = None) -> None:
        with self._lock:
            self.steps[index] = {
                "index": index,
                "title": title,
                "agent": agent,
                "status": "running",
                "output": "",
                "events": [],
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._current_index = index

    def end_step(
        self,
        index: int,
        status: str = "done",
        output: str = "",
        agent: Optional[str] = None,
    ) -> None:
        with self._lock:
            step = self.steps.get(index)
            if step is not None:
                step["status"] = status
                step["output"] = output or ""
                if agent:
                    step["agent"] = agent
                step["ended_at"] = datetime.now().isoformat(timespec="seconds")
            self._current_index = None

    # -- event recording ----------------------------------------------------- #
    def _append(self, event: Dict[str, Any]) -> None:
        phase = event.get("phase")
        with self._lock:
            if event["type"] == "llm":
                self.llm_events.append(event)
            elif event["type"] == "tool":
                self.tool_events.append(event)

            if phase == "planner":
                self.planner_events.append(event)
            elif self._current_index is not None and self._current_index in self.steps:
                self.steps[self._current_index]["events"].append(event)
            else:
                self.loose_events.append(event)

    def record_llm(
        self,
        *,
        agent: str,
        phase: str,
        model: str,
        messages: Any,
        response: Any = None,
        error: Optional[str] = None,
        latency_s: float = 0.0,
    ) -> None:
        usage = (
            _extract_usage(response)
            if response is not None
            else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0, "api_calls": 0}
        )
        cost, known = estimate_cost(
            model, usage["prompt_tokens"], usage["completion_tokens"], usage.get("cached_tokens", 0)
        )

        content = ""
        decisions: List[Dict[str, Any]] = []
        finish_reason = None
        if response is not None:
            try:
                content = response.choices[0].message.content or ""
            except Exception:
                content = ""
            try:
                finish_reason = response.choices[0].finish_reason
            except Exception:
                finish_reason = None
            decisions = _extract_tool_decisions(response)

        self._append(
            {
                "type": "llm",
                "agent": agent,
                "phase": phase,
                "model": model,
                "messages": _safe_messages(messages),
                "content": content,
                "tool_decisions": decisions,
                "usage": usage,
                "cost_usd": cost,
                "cost_known": known,
                "latency_s": round(latency_s, 4),
                "finish_reason": finish_reason,
                "error": error,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    def record_tool(
        self,
        *,
        tool: str,
        arguments: Any,
        result: Any = None,
        error: Optional[str] = None,
        latency_s: float = 0.0,
    ) -> None:
        self._append(
            {
                "type": "tool",
                "tool": tool,
                "arguments": arguments,
                "result": result,
                "error": error,
                "latency_s": round(latency_s, 4),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    # -- aggregation --------------------------------------------------------- #
    def totals(self) -> Dict[str, Any]:
        prompt = sum(e["usage"]["prompt_tokens"] for e in self.llm_events)
        completion = sum(e["usage"]["completion_tokens"] for e in self.llm_events)
        total = sum(e["usage"]["total_tokens"] for e in self.llm_events)
        cost = sum(e["cost_usd"] for e in self.llm_events)
        llm_latency = sum(e["latency_s"] for e in self.llm_events)
        tool_latency = sum(e["latency_s"] for e in self.tool_events)
        errors = sum(1 for e in self.llm_events if e["error"]) + sum(
            1 for e in self.tool_events if e["error"]
        )
        n_tools = len(self.tool_events)
        tool_result_errors = sum(1 for e in self.tool_events if _tool_failed(e))
        return {
            "llm_calls": len(self.llm_events),
            "tool_calls": n_tools,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cost_usd": cost,
            "llm_latency_s": llm_latency,
            "tool_latency_s": tool_latency,
            "errors": errors,
            "tool_result_errors": tool_result_errors,
            "tool_success_rate": round(1 - tool_result_errors / n_tools, 4) if n_tools else 1.0,
        }

    def _per_agent(self) -> Dict[str, Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = {}
        for event in self.llm_events:
            row = agg.setdefault(
                event["agent"],
                {"calls": 0, "tokens": 0, "cost": 0.0, "latency": 0.0},
            )
            row["calls"] += 1
            row["tokens"] += event["usage"]["total_tokens"]
            row["cost"] += event["cost_usd"]
            row["latency"] += event["latency_s"]
        return agg

    def _per_tool(self) -> Dict[str, Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = {}
        for event in self.tool_events:
            row = agg.setdefault(
                event["tool"], {"calls": 0, "latency": 0.0, "errors": 0}
            )
            row["calls"] += 1
            row["latency"] += event["latency_s"]
            if _tool_failed(event):
                row["errors"] += 1
        return agg

    # -- benchmark metrics --------------------------------------------------- #
    def sources_retrieved(self) -> List[Dict[str, str]]:
        seen = set()
        out: List[Dict[str, str]] = []
        for event in self.tool_events:
            for src in _extract_sources_from_result(event.get("tool", ""), event.get("result")):
                key = (src.get("url") or src.get("title") or "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(src)
        return out

    def tool_outcomes(self) -> Dict[str, Any]:
        total = len(self.tool_events)
        failed = sum(1 for e in self.tool_events if _tool_failed(e))
        success = total - failed
        return {
            "tool_calls": total,
            "tool_success": success,
            "tool_failed": failed,
            "tool_success_rate": round(success / total, 4) if total else 1.0,
            "tool_error_rate": round(failed / total, 4) if total else 0.0,
        }

    def citation_metrics(self) -> Dict[str, Any]:
        return compute_citation_metrics(self.sources_retrieved(), self.final_report or "")

    def set_evaluation(self, evaluation: Dict[str, Any]) -> None:
        with self._lock:
            self.evaluation = evaluation or {}

    def metrics(self) -> Dict[str, Any]:
        t = self.totals()
        wall = (
            (self.finished_at - self.started_at).total_seconds()
            if self.finished_at
            else (datetime.now() - self.started_at).total_seconds()
        )
        cit = self.citation_metrics()
        outcomes = self.tool_outcomes()
        ev = self.evaluation or {}
        rep = ev.get("report", {}) or {}
        ret = ev.get("retrieval", {}) or {}
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "status": self.status,
            "wall_time_s": round(wall, 2),
            "llm_calls": t["llm_calls"],
            "tool_calls": t["tool_calls"],
            "prompt_tokens": t["prompt_tokens"],
            "completion_tokens": t["completion_tokens"],
            "total_tokens": t["total_tokens"],
            "cost_usd": round(t["cost_usd"], 6),
            "llm_latency_s": round(t["llm_latency_s"], 2),
            "tool_latency_s": round(t["tool_latency_s"], 2),
            "errors_raised": t["errors"],
            "tool_success_rate": outcomes["tool_success_rate"],
            "tool_error_rate": outcomes["tool_error_rate"],
            "sources_retrieved": cit["sources_retrieved"],
            "sources_cited": cit["sources_cited"],
            "citation_coverage": cit["citation_coverage"],
            "quality_overall": rep.get("overall"),
            "quality_grade": rep.get("grade"),
            "instruction_following": rep.get("instruction_following"),
            "faithfulness": (rep.get("scores", {}) or {}).get("factual_grounding"),
            "retrieval_precision": ret.get("precision"),
            "retrieval_recall": ret.get("recall"),
            "retrieval_f1": ret.get("f1"),
            "retrieval_mrr": ret.get("mrr"),
            "retrieval_hit_at_3": ret.get("hit_at_3"),
        }

    # -- rendering ----------------------------------------------------------- #
    def render_txt(self) -> str:
        lines: List[str] = []
        bar = "=" * 78
        thin = "-" * 78
        totals = self.totals()
        wall = (
            (self.finished_at - self.started_at).total_seconds()
            if self.finished_at
            else (datetime.now() - self.started_at).total_seconds()
        )

        lines.append(bar)
        lines.append("MULTI-AGENT RESEARCH PIPELINE - EXECUTION TRACE")
        lines.append(bar)
        lines.append(f"Task ID      : {self.task_id}")
        lines.append(f"User prompt  : {self.prompt}")
        lines.append(f"Started      : {self.started_at.isoformat(timespec='seconds')}")
        lines.append(
            f"Finished     : {self.finished_at.isoformat(timespec='seconds') if self.finished_at else '(in progress)'}"
        )
        lines.append(f"Wall time    : {wall:.2f} s")
        lines.append(f"Status       : {self.status}")
        if self.error:
            lines.append(f"Error        : {self.error}")
        lines.append("")
        lines.append("PIPELINE OVERVIEW")
        lines.append(
            "  User prompt -> Planner (o4-mini) -> ordered plan -> Executor loop:"
        )
        lines.append(
            "     each step -> Research agent (Tavily/arXiv/Wikipedia) | Writer | Editor"
        )
        lines.append(
            "  -> final Markdown report. The Tracer records tokens, cost, latency, errors."
        )
        lines.append("")

        # Summary
        lines.append(thin)
        lines.append("SUMMARY (TOTALS)")
        lines.append(thin)
        lines.append(f"LLM calls            : {totals['llm_calls']}")
        lines.append(f"Tool calls           : {totals['tool_calls']}")
        lines.append(f"Prompt tokens        : {totals['prompt_tokens']}")
        lines.append(f"Completion tokens    : {totals['completion_tokens']}")
        lines.append(f"Total tokens         : {totals['total_tokens']}")
        lines.append(f"Estimated cost (USD) : ${totals['cost_usd']:.6f}")
        lines.append(f"Total LLM latency    : {totals['llm_latency_s']:.3f} s")
        lines.append(f"Total tool latency   : {totals['tool_latency_s']:.3f} s")
        lines.append(f"Errors               : {totals['errors']}")
        lines.append("")
        lines.append("Per-agent breakdown:")
        for agent, row in self._per_agent().items():
            lines.append(
                f"  {agent:<16} calls={row['calls']:<3} tokens={row['tokens']:<8} "
                f"cost=${row['cost']:.6f}  latency={row['latency']:.3f}s"
            )
        per_tool = self._per_tool()
        if per_tool:
            lines.append("")
            lines.append("Per-tool breakdown:")
            for tool, row in per_tool.items():
                lines.append(
                    f"  {tool:<22} calls={row['calls']:<3} "
                    f"latency={row['latency']:.3f}s  errors={row['errors']}"
                )
        lines.append("")

        # Planning section
        lines.append(bar)
        lines.append("STEP 0 - PLANNING  (agent: planner)")
        lines.append(bar)
        if self.planner_events:
            for event in self.planner_events:
                self._render_llm_input(lines, event)
                lines.append("[LLM DECISION] Raw planner output:")
                lines.append(_trunc(event["content"], TXT_OUTPUT_CHARS))
                lines.append("")
                self._render_llm_telemetry(lines, event)
        else:
            lines.append("(planner produced no recorded LLM event)")
        if self.plan:
            lines.append("")
            lines.append(f"[PARSED PLAN] {len(self.plan)} step(s):")
            for i, step in enumerate(self.plan, start=1):
                lines.append(f"  {i}. {step}")
        lines.append("")

        # Execution steps
        for index in sorted(self.steps.keys()):
            step = self.steps[index]
            llm_events = [e for e in step["events"] if e["type"] == "llm"]
            tool_events = [e for e in step["events"] if e["type"] == "tool"]
            agent = step.get("agent") or (llm_events[0]["agent"] if llm_events else "?")
            model = llm_events[0]["model"] if llm_events else "-"

            lines.append(bar)
            lines.append(f"STEP {index + 1} - {step['title']}")
            lines.append(f"   assigned agent: {agent} / model: {model} / status: {step['status']}")
            lines.append(bar)

            if llm_events:
                self._render_llm_input(lines, llm_events[0], label="[INPUT] Context/task given to the agent:")
                for event in llm_events:
                    decisions = event["tool_decisions"]
                    if decisions:
                        lines.append(
                            f"[REASONING / TOOL DECISIONS] model issued {len(decisions)} tool call(s):"
                        )
                        for dec in decisions:
                            lines.append(
                                f"  turn {dec['turn']}: -> {dec['tool']}({_kv(dec['arguments'])})"
                            )
                        lines.append("")
            else:
                lines.append("[INPUT] (no LLM event recorded for this step)")
                lines.append("")

            if tool_events:
                lines.append("[TOOL EXECUTION] actual tool inputs and outputs:")
                for n, event in enumerate(tool_events, start=1):
                    lines.append(f"  ({n}) {event['tool']}")
                    lines.append(f"      INPUT : {_kv(event['arguments'])}")
                    if event["error"]:
                        lines.append(
                            f"      ERROR (after {event['latency_s']:.3f}s): {event['error']}"
                        )
                    else:
                        result_text = _trunc(_to_text(event["result"]), TXT_TOOL_OUTPUT_CHARS)
                        indented = "\n".join("        " + ln for ln in result_text.splitlines())
                        lines.append(f"      OUTPUT (latency {event['latency_s']:.3f}s):")
                        lines.append(indented if indented else "        (empty)")
                lines.append("")

            lines.append("[OUTPUT] Final output of this step:")
            lines.append(_trunc(step["output"], TXT_OUTPUT_CHARS))
            lines.append("")

            for event in llm_events:
                self._render_llm_telemetry(lines, event)
            step_tool_latency = sum(e["latency_s"] for e in tool_events)
            if tool_events:
                lines.append(f"[TELEMETRY] step tool latency: {step_tool_latency:.3f}s")
            lines.append("")

        # Final report
        lines.append(bar)
        lines.append("FINAL REPORT (output of the last step)")
        lines.append(bar)
        lines.append(_trunc(self.final_report, TXT_OUTPUT_CHARS * 2))
        lines.append("")

        # Benchmark metrics + evaluation
        m = self.metrics()
        cit = self.citation_metrics()
        outcomes = self.tool_outcomes()
        tt = self.totals()
        lines.append(bar)
        lines.append("BENCHMARK METRICS")
        lines.append(bar)
        lines.append(f"Run ID              : {self.run_id}")
        lines.append(f"Wall time (s)       : {m['wall_time_s']}")
        lines.append(
            f"Tokens              : total {m['total_tokens']} "
            f"(prompt {m['prompt_tokens']} / completion {m['completion_tokens']})"
        )
        lines.append(f"Cost (USD)          : ${m['cost_usd']:.6f}")
        lines.append(
            f"Tool calls          : {outcomes['tool_calls']}  "
            f"success_rate={outcomes['tool_success_rate']}  error_rate={outcomes['tool_error_rate']}"
        )
        lines.append(
            f"Sources retrieved   : {cit['sources_retrieved']}  cited={cit['sources_cited']}  "
            f"citation_coverage={cit['citation_coverage']}"
        )
        lines.append("")
        lines.append("HOW ERRORS ARE COUNTED (why a naive error count can read 0):")
        lines.append("  errors_raised    = exceptions thrown by an LLM/tool call (hard failures).")
        lines.append("  tool_result_errors = tool calls whose RESULT was an error payload, e.g. arXiv")
        lines.append("                     returned [{'error': ...}]. These do NOT raise an exception,")
        lines.append("                     so they are invisible to a raised-exception counter.")
        lines.append(
            f"  -> errors_raised={tt['errors']}  tool_result_errors={tt['tool_result_errors']}"
        )
        lines.append("")
        ev = self.evaluation or {}
        rep = ev.get("report") or {}
        ret = ev.get("retrieval") or {}
        if rep or ret:
            lines.append(f"LLM-AS-JUDGE EVALUATION (judge: {ev.get('judge_model', '')})")
            if rep:
                lines.append(f"  Report overall    : {rep.get('overall')} (grade {rep.get('grade')})")
                for k, v in (rep.get("scores", {}) or {}).items():
                    lines.append(f"    - {k:<22}: {v}")
                if rep.get("where_it_went_wrong"):
                    lines.append("  Where it went wrong:")
                    lines.append("    " + _trunc(rep.get("where_it_went_wrong", ""), 1400))
                for w in (rep.get("weaknesses", []) or [])[:6]:
                    lines.append(f"    weakness: {w}")
            if ret:
                lines.append("  Retrieval (LLM-judged):")
                lines.append(f"    - precision : {ret.get('precision')}")
                lines.append(f"    - recall*   : {ret.get('recall')}  (*coverage proxy, no labelled set)")
                lines.append(f"    - f1        : {ret.get('f1')}")
                lines.append(f"    - relevant  : {ret.get('relevant')}/{ret.get('retrieved')}")
        else:
            lines.append("LLM-AS-JUDGE EVALUATION: (not run)")
        lines.append("")

        # Footer / pricing
        lines.append(thin)
        lines.append(f"PRICING TABLE (USD per 1,000,000 tokens) - {PRICING_AS_OF}")
        for model, price in MODEL_PRICING.items():
            lines.append(
                f"  {model:<14} input ${price['input']:.2f}  output ${price['output']:.2f}"
            )
        lines.append("  Edit MODEL_PRICING in src/trace_logger.py to match your account.")
        lines.append(thin)
        lines.append("END OF TRACE")
        return "\n".join(lines)

    def _render_llm_input(
        self, lines: List[str], event: Dict[str, Any], label: str = "[INPUT] Prompt sent to the model:"
    ) -> None:
        lines.append(label)
        for message in event["messages"]:
            content = _trunc(message["content"], TXT_INPUT_CHARS)
            lines.append(f"  <{message['role']}>")
            for ln in content.splitlines() or [""]:
                lines.append("    " + ln)
        lines.append("")

    def _render_llm_telemetry(self, lines: List[str], event: Dict[str, Any]) -> None:
        usage = event["usage"]
        known = "" if event["cost_known"] else " (model not in pricing table -> $0)"
        lines.append(
            "[TELEMETRY] "
            f"prompt={usage['prompt_tokens']} completion={usage['completion_tokens']} "
            f"total={usage['total_tokens']} cached={usage.get('cached_tokens', 0)} tokens | "
            f"api_calls={usage['api_calls']} | "
            f"cost=${event['cost_usd']:.6f}{known} | latency={event['latency_s']:.3f}s | "
            f"finish={event['finish_reason']}"
        )
        if event["error"]:
            lines.append(f"[TELEMETRY] error: {event['error']}")
        lines.append("")

    def to_json(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "totals": self.totals(),
            "metrics": self.metrics(),
            "tool_outcomes": self.tool_outcomes(),
            "citation_metrics": self.citation_metrics(),
            "sources_retrieved": self.sources_retrieved(),
            "evaluation": self.evaluation,
            "per_agent": self._per_agent(),
            "per_tool": self._per_tool(),
            "plan": self.plan,
            "planner_events": self.planner_events,
            "steps": [self.steps[i] for i in sorted(self.steps.keys())],
            "loose_events": self.loose_events,
            "final_report": self.final_report,
            "pricing": MODEL_PRICING,
        }

    def _slug(self) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", (self.prompt or "run")).strip("-").lower()
        return (slug[:24] or "run")

    def write(self, runs_dir: Optional[str] = None) -> str:
        runs_dir = runs_dir or RUNS_DIR
        os.makedirs(runs_dir, exist_ok=True)
        base = f"trace_{self.started_at.strftime('%Y%m%d_%H%M%S')}_{self._slug()}_{self.task_id[:8]}"
        self.txt_path = os.path.join(runs_dir, base + ".txt")
        self.json_path = os.path.join(runs_dir, base + ".json")
        with open(self.txt_path, "w", encoding="utf-8") as handle:
            handle.write(self.render_txt())
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, default=str, ensure_ascii=False, indent=2)
        return self.txt_path


# --------------------------------------------------------------------------- #
# Public helpers used by the agents
# --------------------------------------------------------------------------- #
def traced_completion(client, *, agent: str, phase: str, model: str, messages, **kwargs):
    """
    Wrap client.chat.completions.create, record one LLM event on the active
    tracer (tokens, cost, latency, tool decisions, errors) and return the raw
    response so existing call sites keep working unchanged.
    """
    tracer = get_active_tracer()
    start = time.perf_counter()
    response = None
    error = None
    try:
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response
    except Exception as exc:  # noqa: BLE001 - re-raised after recording
        error = str(exc)
        raise
    finally:
        latency = time.perf_counter() - start
        if tracer is not None:
            try:
                tracer.record_llm(
                    agent=agent,
                    phase=phase,
                    model=model,
                    messages=messages,
                    response=response,
                    error=error,
                    latency_s=latency,
                )
            except Exception:
                pass


def wrap_tool(fn):
    """
    Wrap a tool callable so each invocation is recorded (input args, output,
    latency, errors). functools.wraps preserves the original name, docstring and
    signature, which aisuite relies on to build the tool schema.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        tracer = get_active_tracer()
        start = time.perf_counter()
        result = None
        error = None
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:  # noqa: BLE001 - re-raised after recording
            error = str(exc)
            raise
        finally:
            latency = time.perf_counter() - start
            if tracer is not None:
                try:
                    arguments = dict(kwargs) if kwargs else (list(args) if args else {})
                    tracer.record_tool(
                        tool=getattr(fn, "__name__", "tool"),
                        arguments=arguments,
                        result=result,
                        error=error,
                        latency_s=latency,
                    )
                except Exception:
                    pass

    return wrapper
