import json
import os
import re
from typing import List
from datetime import datetime
from aisuite import Client
from .agent import (
    research_agent,
    writer_agent,
    editor_agent,
)
from .trace_logger import traced_completion, get_active_tracer

client = Client()


def clean_json_block(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip("` \n")


from typing import List
import json, ast


def planner_agent(topic: str, model: str = os.getenv("PLANNER_MODEL", "openai:gpt-4.1-mini")) -> List[str]:
    prompt = f"""
You are a planning agent responsible for organizing a research workflow using multiple intelligent agents.

🧠 Available agents:
- Research agent: MUST begin with a broad **web search using Tavily** to identify only **relevant** and **authoritative** items (e.g., high-impact venues, seminal works, surveys, or recent comprehensive sources). The output of this step MUST capture for each candidate: title, authors, year, venue/source, URL, and (if available) DOI.
- Research agent: AFTER the Tavily step, perform a **targeted arXiv search** ONLY for the candidates discovered in the web step (match by title/author/DOI). If an arXiv preprint/version exists, record its arXiv URL and version info. Do NOT run a generic arXiv search detached from the Tavily results.
- Writer agent: drafts based on research findings.
- Editor agent: reviews, reflects on, and improves drafts.

🎯 Produce a clear step-by-step research plan **as a valid Python list of strings** (no markdown, no explanations). 
Each step must be atomic, actionable, and assigned to one of the agents.
Keep it LEAN: maximum of 5 steps, with at most one analysis/synthesis step between
retrieval and the final report. Fewer, higher-value steps reduce latency and cost.

🚫 DO NOT include steps like “create CSV”, “set up repo”, “install packages”.
✅ Focus on meaningful research tasks (search, extract, rank, draft, revise).
✅ The FIRST step MUST be exactly: 
"Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
✅ The SECOND step MUST be exactly:
"Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."

🔚 The FINAL step MUST instruct the writer agent to generate a comprehensive Markdown report that:
- Uses all findings and outputs from previous steps
- Includes inline citations (e.g., [1], (Wikipedia/arXiv))
- Includes a References section with clickable links for all citations
- Preserves earlier sources
- Is detailed and self-contained

Topic: "{topic}"
"""

    # Reasoning models (o-series) require temperature=1; others get a low temp
    # for stable, focused plans.
    _temp = 1 if any(x in model for x in ("o4", "o3", "o1")) else 0.2
    response = traced_completion(
        client,
        agent="planner_agent",
        phase="planner",
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=_temp,
    )

    raw = response.choices[0].message.content.strip()

    # --- robust parsing: JSON -> ast -> fallback ---
    def _coerce_to_list(s: str) -> List[str]:
        # try strict JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except json.JSONDecodeError:
            pass
        # try Python literal list
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except Exception:
            pass
        # try to extract code fence if present
        if s.startswith("```") and s.endswith("```"):
            inner = s.strip("`")
            try:
                obj = ast.literal_eval(inner)
                if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                    return obj[:7]
            except Exception:
                pass
        return []

    steps = _coerce_to_list(raw)

    # enforce ordering & minimal contract
    required_first = "Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
    required_second = "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."
    final_required = "Writer agent: Generate the final comprehensive Markdown report with inline citations and a complete References section with clickable links."

    def _ensure_contract(steps_list: List[str]) -> List[str]:
        if not steps_list:
            return [
                required_first,
                required_second,
                "Research agent: Synthesize and rank the most relevant findings; deduplicate by title/DOI and drop low-relevance sources.",
                "Editor agent: Check coverage and citation completeness against the retrieved sources.",
                final_required,
            ]
        # inject/replace first two if missing or out of order
        steps_list = [s for s in steps_list if isinstance(s, str)]
        if not steps_list or steps_list[0] != required_first:
            steps_list = [required_first] + steps_list
        if len(steps_list) < 2 or steps_list[1] != required_second:
            # remove any generic arxiv step that is not tied to Tavily results
            steps_list = (
                [steps_list[0]]
                + [required_second]
                + [
                    s
                    for s in steps_list[1:]
                    if "arXiv" not in s or "For each collected item" in s
                ]
            )
        # ensure final step requirement present
        if final_required not in steps_list:
            steps_list.append(final_required)
        # cap to 5 (lean plan -> fewer sources, lower latency)
        return steps_list[:5]

    steps = _ensure_contract(steps)

    return steps


def executor_agent_step(step_title: str, history: list, prompt: str):
    """
    Executes a step of the executor agent.
    Returns:
        - step_title (str)
        - agent_name (str)
        - output (str)
    """

    # Build a TOKEN-BUDGETED context. Passing every prior step's full output to
    # every later step is the main driver of prompt-token bloat, so cap each
    # item and keep only the most recent items within a fixed character budget.
    PER_ITEM_CHARS = 1400
    CONTEXT_BUDGET = 7000

    def _label(desc_, agent_):
        d = desc_.lower()
        if "draft" in d or agent_ == "writer_agent":
            return "Draft"
        if "feedback" in d or agent_ == "editor_agent":
            return "Feedback"
        if "research" in d or agent_ == "research_agent":
            return "Research"
        return f"Output by {agent_}"

    blocks = []
    used = 0
    for i in range(len(history) - 1, -1, -1):
        desc, agent, output = history[i]
        snippet = (output or "").strip()
        if len(snippet) > PER_ITEM_CHARS:
            snippet = snippet[:PER_ITEM_CHARS] + " ...[truncated]"
        block = f"\n[{_label(desc, agent)} - Step {i + 1}]\n{snippet}\n"
        if blocks and used + len(block) > CONTEXT_BUDGET:
            blocks.append("\n[earlier steps omitted to keep context small]\n")
            break
        blocks.append(block)
        used += len(block)
    blocks.reverse()
    context = (
        f"User prompt:\n{prompt}\n\nHistory so far (most recent kept within budget):\n"
        + "".join(blocks)
    )

    enriched_task = f"""{context}

Your next task:
{step_title}
"""

    # Select the agent. The planner prefixes every step with the responsible
    # agent ("Research agent:", "Writer agent:", "Editor agent:"), so honour
    # that explicit assignment first. Only fall back to keyword heuristics when
    # no prefix is present, otherwise an editor step that mentions "draft" is
    # misrouted to the writer and the editing pass never runs.
    step_lower = step_title.lower()
    if step_lower.startswith("research agent"):
        agent_kind = "research"
    elif step_lower.startswith("editor agent"):
        agent_kind = "editor"
    elif step_lower.startswith("writer agent"):
        agent_kind = "writer"
    elif "research" in step_lower:
        agent_kind = "research"
    elif "revise" in step_lower or "edit" in step_lower or "feedback" in step_lower:
        agent_kind = "editor"
    elif "draft" in step_lower or "write" in step_lower:
        agent_kind = "writer"
    else:
        raise ValueError(f"Unknown step type: {step_title}")

    # For writer/editor steps, inject an explicit numbered list of the ACTUAL
    # retrieved sources so the model cites real URLs by [n] instead of pulling
    # canonical or hallucinated references from memory. This is the main lever
    # for citation coverage and faithfulness.
    if agent_kind in ("writer", "editor"):
        tracer = get_active_tracer()
        if tracer is not None:
            srcs = tracer.sources_retrieved()[:12]
            if srcs:
                listing = "\n".join(
                    f"[{i}] {(s.get('title') or '')[:90]} {s.get('url', '')}".strip()
                    for i, s in enumerate(srcs, start=1)
                )
                enriched_task += (
                    "\n\nAVAILABLE SOURCES - cite ONLY these by [n] and copy their URLs "
                    "VERBATIM into the References section. Do not invent or substitute "
                    "references:\n" + listing
                )

    if agent_kind == "research":
        content, _ = research_agent(prompt=enriched_task)
        print("🔍 Research Agent Output:", content)
        return step_title, "research_agent", content
    if agent_kind == "editor":
        content, _ = editor_agent(prompt=enriched_task)
        return step_title, "editor_agent", content
    content, _ = writer_agent(prompt=enriched_task)
    return step_title, "writer_agent", content