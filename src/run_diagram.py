"""
Dynamic, per-run architecture diagram.

Unlike a fixed/predefined diagram, this renders an image FROM A SINGLE RUN's
trace: only the agents that actually executed, the tools they actually called
(with call counts, latency and errors), and per-step tokens / cost / latency.

A fresh PNG is produced for every task after it finishes, so the picture always
reflects what really happened, including which tool calls were made.

Thread-safe: uses the matplotlib object-oriented API (Figure / Agg canvas)
instead of the global pyplot state machine, because runs execute on worker
threads.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------- #
# Palette (clean, light theme)
# --------------------------------------------------------------------------- #
AGENT_COLORS = {
    "planner_agent": "#7c3aed",   # violet
    "research_agent": "#ea580c",  # orange
    "writer_agent": "#16a34a",    # green
    "editor_agent": "#0ea5e9",    # sky
}
DEFAULT_AGENT_COLOR = "#ea580c"
TOOL_COLOR = "#0891b2"   # cyan
ENTRY_COLOR = "#2563eb"  # blue
REPORT_COLOR = "#15803d" # dark green
ERROR_COLOR = "#dc2626"  # red

INK = "#0f172a"
SUBTLE = "#475569"
HAIR = "#cbd5e1"
CARD_BG = "#f8fafc"
CARD_EDGE = "#e2e8f0"


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def _node(ax, cx, top, w, h, title, subtitle, color):
    """Rounded node anchored by its TOP edge centre (cx, top)."""
    x = cx - w / 2.0
    y = top - h
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0, facecolor=color, alpha=0.97,
        )
    )
    if subtitle:
        ax.text(cx, y + h * 0.63, title, ha="center", va="center",
                fontsize=10.5, color="white", weight="bold")
        ax.text(cx, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=8.2, color="white", alpha=0.95)
    else:
        ax.text(cx, y + h / 2.0, title, ha="center", va="center",
                fontsize=10.5, color="white", weight="bold")
    return {"cx": cx, "cy": y + h / 2.0, "x": x, "y": y, "w": w, "h": h,
            "top": top, "bottom": y}


def _chip(ax, cx, cy, w, h, line1, line2, color):
    """Small rounded chip centred at (cx, cy)."""
    x = cx - w / 2.0
    y = cy - h / 2.0
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=0, facecolor=color, alpha=0.96,
        )
    )
    ax.text(cx, y + h * 0.64, line1, ha="center", va="center",
            fontsize=8.0, color="white", weight="bold")
    if line2:
        ax.text(cx, y + h * 0.28, line2, ha="center", va="center",
                fontsize=7.0, color="white", alpha=0.95)
    return {"cx": cx, "cy": cy, "x": x, "y": y, "w": w, "h": h}


def _arrow(ax, start, end, color="#94a3b8", lw=1.8, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=lw, color=color, linestyle=ls,
            shrinkA=3, shrinkB=3, joinstyle="round",
        )
    )


def _stat(ax, x, y, w, h, label, value, accent):
    """Header metric card with a coloured accent bar."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, edgecolor=CARD_EDGE, facecolor="white",
        )
    )
    ax.add_patch(Rectangle((x + 0.07, y + 0.14), 0.11, h - 0.28, color=accent))
    ax.text(x + 0.32, y + h - 0.30, label, fontsize=8.2, color=SUBTLE,
            ha="left", va="center")
    ax.text(x + 0.32, y + 0.30, value, fontsize=13, color=INK, weight="bold",
            ha="left", va="center")


def _agent_color(agent: str) -> str:
    return AGENT_COLORS.get(agent, DEFAULT_AGENT_COLOR)


def _step_summary(step: Dict[str, Any]):
    events = step.get("events", [])
    llm = [e for e in events if e.get("type") == "llm"]
    tools = [e for e in events if e.get("type") == "tool"]
    tokens = sum(e["usage"]["total_tokens"] for e in llm)
    cost = sum(e["cost_usd"] for e in llm)
    latency = sum(e["latency_s"] for e in llm) + sum(e["latency_s"] for e in tools)
    tool_counts: Dict[str, Dict[str, Any]] = {}
    for event in tools:
        row = tool_counts.setdefault(event["tool"], {"calls": 0, "lat": 0.0, "err": 0})
        row["calls"] += 1
        row["lat"] += event["latency_s"]
        row["err"] += 1 if event.get("error") else 0
    return tokens, cost, latency, tool_counts


def _short(text: str, limit: int = 46) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _tool_label(tool: str) -> str:
    return tool.replace("_search_tool", "").replace("_tool", "")


def render_run_architecture(tracer, out_dir: Optional[str] = None) -> str:
    """Render the run's architecture to runs/arch_<task_id>.png and return path."""
    from src.trace_logger import RUNS_DIR  # local import avoids import cycles

    out_dir = out_dir or RUNS_DIR
    os.makedirs(out_dir, exist_ok=True)

    data = tracer.to_json()
    steps: List[Dict[str, Any]] = data.get("steps", [])
    totals = data.get("totals", {})
    per_agent = data.get("per_agent", {})
    per_tool = data.get("per_tool", {})
    planner_events = data.get("planner_events", [])

    # ---- Geometry -------------------------------------------------------- #
    W = 16.0
    HEADER = 2.9
    ROW = 1.55
    FOOTER = 1.05
    n_rows = 2 + len(steps) + 1            # user + planner + steps + final report
    H = HEADER + n_rows * ROW + FOOTER

    FX = 3.3                                # flow column centre
    NW, NH = 4.7, 1.05                      # node width / height
    CHIP_X0 = FX + NW / 2.0 + 0.55          # tool-chip lane (left edge)
    CW, CH = 1.95, 0.82                     # chip size
    PX, PW = 10.7, 5.0                      # summary panel x / width

    fig = Figure(figsize=(W, H), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor("white")

    # ---- Header ---------------------------------------------------------- #
    ax.text(0.6, H - 0.55, "Run Architecture", fontsize=21, weight="bold",
            color=INK, ha="left", va="center")
    status = data.get("status", "")
    status_color = {"done": REPORT_COLOR, "error": ERROR_COLOR,
                    "running": "#d97706"}.get(status, "#64748b")
    ax.text(0.62, H - 1.12, f"\u25cf {status}", fontsize=10.5, color=status_color,
            ha="left", va="center", weight="bold")
    ax.text(1.7, H - 1.12, _short(data.get("prompt", ""), 116), fontsize=10.5,
            color=SUBTLE, ha="left", va="center", style="italic")

    total_latency = totals.get("llm_latency_s", 0.0) + totals.get("tool_latency_s", 0.0)
    stats = [
        ("LLM calls", str(totals.get("llm_calls", 0)), ENTRY_COLOR),
        ("Tool calls", str(totals.get("tool_calls", 0)), TOOL_COLOR),
        ("Tokens", f"{totals.get('total_tokens', 0):,}", "#7c3aed"),
        ("Cost (USD)", f"${totals.get('cost_usd', 0.0):.4f}", REPORT_COLOR),
        ("Latency", f"{total_latency:.1f}s", "#d97706"),
        ("Errors", str(totals.get("errors", 0)),
         ERROR_COLOR if totals.get("errors", 0) else "#64748b"),
    ]
    sc_w, sc_h, sc_gap = 2.28, 0.95, 0.18
    sc_x = 0.6
    sc_y = H - 2.55
    for label, value, accent in stats:
        _stat(ax, sc_x, sc_y, sc_w, sc_h, label, value, accent)
        sc_x += sc_w + sc_gap

    # ---- Flow column ----------------------------------------------------- #
    top = H - HEADER - 0.05

    user = _node(ax, FX, top, NW, NH, "User prompt", None, ENTRY_COLOR)
    prev = user

    p_tokens = sum(e["usage"]["total_tokens"] for e in planner_events)
    p_cost = sum(e["cost_usd"] for e in planner_events)
    p_lat = sum(e["latency_s"] for e in planner_events)
    p_model = (planner_events[0]["model"].split(":")[-1] if planner_events else "o4-mini")
    top -= ROW
    planner = _node(
        ax, FX, top, NW, NH,
        f"Planner  \u00b7  {p_model}",
        f"{len(data.get('plan', []))} steps   {p_tokens} tok   ${p_cost:.4f}   {p_lat:.1f}s",
        AGENT_COLORS["planner_agent"],
    )
    _arrow(ax, (prev["cx"], prev["bottom"]), (planner["cx"], planner["top"]))
    prev = planner

    for step in steps:
        top -= ROW
        agent = step.get("agent") or "?"
        tokens, cost, latency, tool_counts = _step_summary(step)
        has_err = any(v["err"] for v in tool_counts.values())
        title = f"Step {step.get('index', 0) + 1}  \u00b7  {_tool_label(agent)}"
        subtitle = f"{tokens} tok   ${cost:.4f}   {latency:.1f}s"
        node = _node(ax, FX, top, NW, NH, title, subtitle,
                     ERROR_COLOR if has_err else _agent_color(agent))
        _arrow(ax, (prev["cx"], prev["bottom"]), (node["cx"], node["top"]))
        prev = node

        # tool chips to the right of the step (2 per row, never reach the panel)
        for i, (tool, info) in enumerate(tool_counts.items()):
            col, row_i = i % 2, i // 2
            cx = CHIP_X0 + col * (CW + 0.2) + CW / 2.0
            cy = node["cy"] - row_i * (CH + 0.14)
            err = f"  err {info['err']}" if info["err"] else ""
            chip = _chip(
                ax, cx, cy, CW, CH,
                _tool_label(tool),
                f"x{info['calls']}  {info['lat']:.1f}s{err}",
                ERROR_COLOR if info["err"] else TOOL_COLOR,
            )
            _arrow(ax, (node["x"] + node["w"], node["cy"]),
                   (chip["x"], chip["cy"]), color=TOOL_COLOR, lw=1.3)

    top -= ROW
    final = _node(ax, FX, top, NW, NH, "Final report",
                  "Markdown + references", REPORT_COLOR)
    _arrow(ax, (prev["cx"], prev["bottom"]), (final["cx"], final["top"]))

    # ---- Summary panel (right) ------------------------------------------ #
    panel_top = H - HEADER - 0.05
    panel_bottom = FOOTER + 0.15
    ax.add_patch(
        FancyBboxPatch(
            (PX, panel_bottom), PW, panel_top - panel_bottom,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.1, edgecolor=CARD_EDGE, facecolor=CARD_BG,
        )
    )
    ax.text(PX + 0.3, panel_top - 0.4, "RUN SUMMARY", fontsize=11,
            weight="bold", color=INK, ha="left", va="center")
    ax.plot([PX + 0.3, PX + PW - 0.3], [panel_top - 0.7, panel_top - 0.7],
            color=HAIR, lw=1.0)

    rows = [
        ("LLM calls", str(totals.get("llm_calls", 0))),
        ("Tool calls", str(totals.get("tool_calls", 0))),
        ("Prompt tokens", f"{totals.get('prompt_tokens', 0):,}"),
        ("Completion tokens", f"{totals.get('completion_tokens', 0):,}"),
        ("Total tokens", f"{totals.get('total_tokens', 0):,}"),
        ("Estimated cost", f"${totals.get('cost_usd', 0.0):.6f}"),
        ("LLM latency", f"{totals.get('llm_latency_s', 0.0):.2f}s"),
        ("Tool latency", f"{totals.get('tool_latency_s', 0.0):.2f}s"),
        ("Errors", str(totals.get("errors", 0))),
    ]
    ry = panel_top - 1.05
    for label, value in rows:
        ax.text(PX + 0.3, ry, label, fontsize=9, color=SUBTLE, ha="left", va="center")
        ax.text(PX + PW - 0.3, ry, value, fontsize=9, color=INK, ha="right",
                va="center", family="monospace", weight="bold")
        ry -= 0.34

    if per_agent:
        ry -= 0.12
        ax.text(PX + 0.3, ry, "Per agent", fontsize=9.5, color=INK,
                weight="bold", ha="left", va="center")
        ry -= 0.34
        for agent, r in per_agent.items():
            ax.add_patch(Rectangle((PX + 0.32, ry - 0.08), 0.16, 0.16,
                                   color=_agent_color(agent)))
            ax.text(PX + 0.6, ry, _short(agent, 26), fontsize=8.4, color=SUBTLE,
                    ha="left", va="center")
            ax.text(PX + PW - 0.3, ry,
                    f"{r['calls']}\u00d7  {r['tokens']:,} tok  ${r['cost']:.4f}",
                    fontsize=8.2, color=INK, ha="right", va="center",
                    family="monospace")
            ry -= 0.32

    if per_tool:
        ry -= 0.12
        ax.text(PX + 0.3, ry, "Tools used", fontsize=9.5, color=INK,
                weight="bold", ha="left", va="center")
        ry -= 0.34
        for tool, r in per_tool.items():
            color = ERROR_COLOR if r.get("errors") else TOOL_COLOR
            ax.add_patch(Rectangle((PX + 0.32, ry - 0.08), 0.16, 0.16, color=color))
            ax.text(PX + 0.6, ry, _short(_tool_label(tool), 26), fontsize=8.4,
                    color=SUBTLE, ha="left", va="center")
            ax.text(PX + PW - 0.3, ry,
                    f"x{r['calls']}  {r['latency']:.1f}s  err {r['errors']}",
                    fontsize=8.2, color=INK, ha="right", va="center",
                    family="monospace")
            ry -= 0.32

    # ---- Footer legend --------------------------------------------------- #
    legend = [
        ("User", ENTRY_COLOR),
        ("Planner", AGENT_COLORS["planner_agent"]),
        ("Research", AGENT_COLORS["research_agent"]),
        ("Writer", AGENT_COLORS["writer_agent"]),
        ("Editor", AGENT_COLORS["editor_agent"]),
        ("Tool call", TOOL_COLOR),
        ("Final report", REPORT_COLOR),
    ]
    lx = 0.6
    ly = 0.45
    for name, color in legend:
        ax.add_patch(FancyBboxPatch(
            (lx, ly), 0.3, 0.3, boxstyle="round,pad=0.0,rounding_size=0.06",
            linewidth=0, facecolor=color))
        ax.text(lx + 0.42, ly + 0.15, name, fontsize=9, color=SUBTLE,
                ha="left", va="center")
        lx += 0.42 + 0.135 * len(name) + 0.55

    out_path = os.path.join(out_dir, f"arch_{tracer.task_id[:8]}.png")
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight",
                pad_inches=0.25)
    return out_path
