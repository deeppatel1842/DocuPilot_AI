"""
LLM-as-judge evaluation for the research pipeline.

Provides two evaluators that run AFTER a pipeline finishes:

  judge_report(...)     - scores the final report on relevance, coherence,
                          coverage, citation accuracy, factual grounding and
                          instruction following, and explains where it went
                          wrong (a real "where did the AI do it wrong" critique).

  judge_retrieval(...)  - labels every retrieved source as relevant / not
                          relevant to the query and derives retrieval
                          precision, a coverage-based recall proxy and F1.

Important on precision / recall: we have no human-labelled ground-truth set of
"all relevant documents", so true recall is not computable. We therefore report
LLM-judged relevance precision and an LLM-estimated coverage recall (a proxy).
This is a standard practice for reference-free RAG evaluation; the README
documents how to plug in a labelled set for exact recall.

All calls are best-effort: if the judge fails or returns malformed JSON, the
evaluator returns a structured fallback so the pipeline never breaks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from aisuite import Client

client = Client()

JUDGE_MODEL = "openai:gpt-4.1-mini"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_json(raw: str) -> Optional[Any]:
    """Pull the first JSON object/array out of an LLM response."""
    if not raw:
        return None
    raw = raw.strip()
    # strip code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # find the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                continue
    return None


def _clamp(value: Any, lo: float = 0.0, hi: float = 100.0, default: float = 0.0) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except Exception:
        return default


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _call_judge(system: str, user: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Report quality judge
# --------------------------------------------------------------------------- #
REPORT_DIMENSIONS = [
    "relevance",
    "coherence",
    "coverage",
    "citation_accuracy",
    "factual_grounding",
    "instruction_following",
]

REPORT_SYSTEM = """
You are a strict, fair senior research editor acting as an automated evaluator.
Score a research report against the user's query and the sources that were
retrieved. Score each dimension from 0 to 100 (100 = excellent).

Dimensions:
- relevance: does the report directly answer the user's query?
- coherence: logical structure, flow, clear sections.
- coverage: breadth and depth across the key aspects of the query.
- citation_accuracy: claims are supported by inline citations and a matching
  References section; citations map to the retrieved sources.
- factual_grounding: claims are grounded in the provided sources with low
  hallucination/unsupported assertions.
- instruction_following: the report follows the required academic structure
  (Title, Abstract, Introduction, Body sections, Conclusion, References) and is
  in Markdown.

Return ONLY a JSON object with this exact shape:
{
  "relevance": int, "coherence": int, "coverage": int,
  "citation_accuracy": int, "factual_grounding": int,
  "instruction_following": int, "overall": int,
  "strengths": [str, ...],
  "weaknesses": [str, ...],
  "where_it_went_wrong": str,
  "verdict": str
}
"where_it_went_wrong" must be one concrete paragraph naming the biggest problem.
No prose outside the JSON.
""".strip()


def judge_report(
    prompt: str,
    report: str,
    sources: List[Dict[str, Any]],
    model: str = JUDGE_MODEL,
) -> Dict[str, Any]:
    src_lines = []
    for i, s in enumerate(sources[:25], start=1):
        src_lines.append(f"[{i}] {s.get('title', '')} ({s.get('url', '')}) — {s.get('origin', '')}")
    sources_block = "\n".join(src_lines) if src_lines else "(no sources retrieved)"

    user = (
        f"USER QUERY:\n{prompt}\n\n"
        f"RETRIEVED SOURCES ({len(sources)}):\n{sources_block}\n\n"
        f"REPORT TO EVALUATE (Markdown):\n{report[:14000]}"
    )

    try:
        raw = _call_judge(REPORT_SYSTEM, user, model)
        data = _extract_json(raw) or {}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "overall": None,
            "grade": None,
            "scores": {},
            "strengths": [],
            "weaknesses": [],
            "where_it_went_wrong": f"Judge call failed: {exc}",
            "verdict": "evaluation unavailable",
            "judge_model": model,
        }

    scores = {dim: int(_clamp(data.get(dim), default=0)) for dim in REPORT_DIMENSIONS}
    if data.get("overall") is not None:
        overall = _clamp(data.get("overall"))
    else:
        overall = sum(scores.values()) / len(scores) if scores else 0.0

    return {
        "ok": True,
        "scores": scores,
        "overall": round(overall, 1),
        "grade": _grade(overall),
        "instruction_following": scores.get("instruction_following"),
        "strengths": data.get("strengths", [])[:6],
        "weaknesses": data.get("weaknesses", [])[:6],
        "where_it_went_wrong": data.get("where_it_went_wrong", ""),
        "verdict": data.get("verdict", ""),
        "judge_model": model,
    }


# --------------------------------------------------------------------------- #
# Retrieval relevance judge -> precision / recall(proxy) / F1
# --------------------------------------------------------------------------- #
RETRIEVAL_SYSTEM = """
You evaluate the quality of retrieved sources for a research query.

1. For EACH source, decide if it is relevant to the query (1 = relevant,
   0 = not relevant) with a short reason.
2. Estimate "coverage_recall" from 0.0 to 1.0: how well the SET of relevant
   sources covers the important sub-topics the query implies (this is a proxy
   for recall since there is no labelled ground-truth set).

Return ONLY JSON of this shape:
{
  "labels": [{"index": int, "relevant": 0|1, "reason": str}, ...],
  "coverage_recall": float,
  "missing_aspects": [str, ...]
}
No prose outside the JSON.
""".strip()


def judge_retrieval(
    prompt: str,
    sources: List[Dict[str, Any]],
    model: str = JUDGE_MODEL,
) -> Dict[str, Any]:
    n = len(sources)
    if n == 0:
        return {
            "ok": True,
            "retrieved": 0,
            "relevant": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage_recall": 0.0,
            "labels": [],
            "missing_aspects": [],
            "judge_model": model,
            "note": "no sources retrieved",
        }

    src_lines = []
    for i, s in enumerate(sources[:25], start=1):
        snippet = (s.get("snippet") or "")[:300]
        src_lines.append(
            f"[{i}] title: {s.get('title', '')}\n    url: {s.get('url', '')}\n"
            f"    via: {s.get('origin', '')}\n    snippet: {snippet}"
        )
    user = f"QUERY:\n{prompt}\n\nSOURCES:\n" + "\n".join(src_lines)

    try:
        raw = _call_judge(RETRIEVAL_SYSTEM, user, model)
        data = _extract_json(raw) or {}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "retrieved": n,
            "relevant": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "labels": [],
            "judge_model": model,
        }

    labels = data.get("labels", []) or []
    relevant = 0
    norm_labels = []
    for lab in labels:
        try:
            rel = 1 if int(lab.get("relevant", 0)) == 1 else 0
        except Exception:
            rel = 0
        relevant += rel
        norm_labels.append(
            {"index": lab.get("index"), "relevant": rel, "reason": lab.get("reason", "")}
        )

    judged = len(norm_labels) or n
    precision = relevant / judged if judged else 0.0
    coverage_recall = _clamp(data.get("coverage_recall", 0.0), 0.0, 1.0, 0.0)
    f1 = (
        2 * precision * coverage_recall / (precision + coverage_recall)
        if (precision + coverage_recall) > 0
        else 0.0
    )

    # Ranking metrics (RAGAS/IR style) from the ordered relevance labels.
    ordered = sorted(norm_labels, key=lambda lab: (lab.get("index") or 10**6))
    rels = [lab["relevant"] for lab in ordered]
    first_rel = next((i + 1 for i, r in enumerate(rels) if r), 0)
    mrr = round(1.0 / first_rel, 4) if first_rel else 0.0
    hit_at_3 = 1 if any(rels[:3]) else 0
    hit_at_5 = 1 if any(rels[:5]) else 0

    return {
        "ok": True,
        "retrieved": n,
        "relevant": relevant,
        "precision": round(precision, 4),
        "recall": round(coverage_recall, 4),  # proxy
        "f1": round(f1, 4),
        "coverage_recall": round(coverage_recall, 4),
        "accuracy": round(precision, 4),  # binary-relevance accuracy == precision here
        "mrr": mrr,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "labels": norm_labels,
        "missing_aspects": data.get("missing_aspects", [])[:8],
        "judge_model": model,
        "note": "precision = LLM-judged relevance; recall = LLM coverage proxy (no labelled set)",
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_run(tracer, model: str = JUDGE_MODEL) -> Dict[str, Any]:
    """Run both judges against a finished tracer and return a combined dict."""
    prompt = tracer.prompt
    report = tracer.final_report or ""
    sources = tracer.sources_retrieved()

    report_eval = judge_report(prompt, report, sources, model=model)
    retrieval_eval = judge_retrieval(prompt, sources, model=model)

    return {
        "judge_model": model,
        "report": report_eval,
        "retrieval": retrieval_eval,
    }
