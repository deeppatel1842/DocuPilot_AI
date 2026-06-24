"""
Recompute citation coverage for already-saved runs (no API calls).

The first benchmark used a citation matcher that missed arXiv version suffixes
(e.g. 2410.19288v1 vs 2410.19288). This re-derives citation coverage from the
raw `sources_retrieved` + `final_report` stored in each runs/trace_*.json using
the arXiv-ID-aware matcher, then rewrites the per-run trace metrics,
benchmark/benchmark_results.json, benchmark/RESULTS.md and the README section.

Usage: ./venv/Scripts/python.exe scripts/recompute_metrics.py
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.trace_logger import compute_citation_metrics  # noqa: E402
from scripts.benchmark import aggregate, render_markdown, update_readme, BENCH_DIR  # noqa: E402

RUNS = os.path.join(BASE, "runs")
BENCH_JSON = os.path.join(BENCH_DIR, "benchmark_results.json")


def main():
    # 1) recompute + rewrite every trace_*.json, index by task_id
    index = {}
    if os.path.isdir(RUNS):
        for name in os.listdir(RUNS):
            if not (name.startswith("trace_") and name.endswith(".json")):
                continue
            path = os.path.join(RUNS, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            cm = compute_citation_metrics(
                data.get("sources_retrieved", []), data.get("final_report", "")
            )
            data["citation_metrics"] = cm
            if isinstance(data.get("metrics"), dict):
                data["metrics"]["citation_coverage"] = cm["citation_coverage"]
                data["metrics"]["sources_cited"] = cm["sources_cited"]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            tid = data.get("task_id") or data.get("run_id")
            if tid:
                index[tid] = cm

    # 2) patch benchmark rows + re-aggregate
    if not os.path.exists(BENCH_JSON):
        print("No benchmark_results.json to update.")
        return
    with open(BENCH_JSON, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    rows = bench.get("runs", [])
    patched = 0
    for row in rows:
        rid = row.get("run_id") or row.get("task_id")
        cm = index.get(rid)
        if cm:
            row["citation_coverage"] = cm["citation_coverage"]
            row["sources_cited"] = cm["sources_cited"]
            patched += 1
    agg = aggregate(rows)
    bench["aggregate"] = agg
    with open(BENCH_JSON, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=2, default=str)

    section = render_markdown(agg, rows, len(rows))
    with open(os.path.join(BENCH_DIR, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("# Benchmark results\n\n" + section + "\n")
    update_readme(section)

    print(f"Patched {patched}/{len(rows)} runs.")
    print(f"avg_citation_coverage -> {agg['avg_citation_coverage']}")


if __name__ == "__main__":
    main()
