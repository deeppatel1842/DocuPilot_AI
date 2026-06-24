# Benchmark results

_Last run: 2026-06-24 12:25 · 3 research queries · judge = gpt-4.1-mini_

| Metric | Value |
|---|---|
| Average cost (USD) | $0.0164 |
| Average latency | 95.4 s |
| P50 latency | 97.8 s |
| P95 latency | 107.0 s |
| P99 latency | 107.0 s |
| Average total tokens | 20640 |
| Average prompt tokens | 13871 |
| Tool success rate | 100% |
| Error rate | 0% |
| Average sources retrieved | 11.0 |
| Average citation coverage | 87% |
| Retrieval precision (LLM-judged) | 0.66 |
| Retrieval recall (coverage proxy) | 0.85 |
| Retrieval F1 | 0.74 |
| Retrieval MRR | 1.00 |
| Report quality (0-100) | 97.0 |
| Instruction following (0-100) | 96.7 |
| Faithfulness (0-100) | 98.3 |

### Per-run

| # | Query | Tokens | Cost $ | Latency s | Tool ok | Src | Cite cov | Quality | Prec | Rec* | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Retrieval-augmented generation for enterprise se | 19794 | 0.0161 | 97.8 | 100% | 12 | 100% | 97 | 0.75 | 0.90 | 0.82 |
| 2 | Vector database indexing methods (HNSW vs IVF-PQ | 17080 | 0.0140 | 81.5 | 100% | 8 | 100% | 97 | 0.62 | 0.80 | 0.70 |
| 3 | Mixture-of-Experts large language models: routin | 25047 | 0.0191 | 107.0 | 100% | 13 | 62% | 97 | 0.62 | 0.85 | 0.71 |

\* recall is an LLM coverage proxy (no labelled ground-truth set).
