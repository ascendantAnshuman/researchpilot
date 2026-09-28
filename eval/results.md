# ResearchPilot evaluation results

- **Run:** 2026-09-28 06:22 UTC
- **Pipeline model:** `gemini-3.5-flash-lite` · **Judge model:** `gemini-3.5-flash-lite`
- **Questions:** 13 completed of 16 (3 failed)
- **Papers per question:** 6.0 on average (5.3 read in full text)
- **Total cost:** pipeline $0.00 + judges $0.00

## Main result: Critic ablation

Both arms come from the same pipeline run (same papers, evidence and first draft); the *without Critic*
arm is the first draft finalised without verification. Δ = with Critic − without Critic, with a 95%
bootstrap confidence interval over questions.

| Metric | With Critic | Without Critic | Δ (95% CI) | Critic better / worse / tie |
|---|---|---|---|---|
| Citation accuracy | 54.5% | 44.6% | +9.8 pts (+3.3 pts, +16.1 pts) | 9 / 4 / 0 |
| Citation accuracy (incl. partial) | 88.0% | 82.3% | +5.7 pts (+0.8 pts, +11.1 pts) | 8 / 4 / 1 |
| Faithfulness (supported claims) | 85.0% | 73.8% | +11.3 pts (+4.1 pts, +18.3 pts) | 10 / 3 / 0 |
| Answer relevance (1-5) | 5.00 | 5.00 | +0.00 (+0.00, +0.00) | 0 / 0 / 13 |
| Latency per query | 261 s | 116 s | +145 s (+105 s, +190 s) | 0 / 13 / 0 |
| Pipeline cost per query | $0.000 | $0.000 | +$0.000 (+$0.000, +$0.000) | 0 / 0 / 13 |

On average the Critic triggered 0.92 revision(s) and removed 2.2 unsupported claim(s) per report.

## Latency per stage (mean seconds per question)

| Stage | With Critic | Without Critic |
|---|---|---|
| Planner | 7.1 | 7.1 |
| Search | 24.5 | 24.5 |
| Reader | 54.1 | 54.1 |
| Synthesizer | 55.0 | 30.1 |
| Critic | 119.8 | 0.0 |
| Finalize | 0.0 | 0.0 |

## Per-question results

| ID | Topic | Citation acc. (C / no C) | Faithfulness (C / no C) | Relevance (C / no C) | Latency | Cost |
|---|---|---|---|---|---|---|
| q01 | NLP / RAG | 67.4% / 69.6% | 94.4% / 100.0% | 5.00 / 5.00 | 461 s | $0.000 |
| q02 | NLP / fine-tuning | 54.8% / 41.5% | 89.5% / 66.7% | 5.00 / 5.00 | 178 s | $0.000 |
| q03 | NLP / summarization | 51.1% / 35.7% | 93.1% / 77.8% | 5.00 / 5.00 | 318 s | $0.000 |
| q04 | NLP / reasoning | 65.3% / 49.0% | 95.7% / 82.9% | 5.00 / 5.00 | 341 s | $0.000 |
| q05 | NLP / safety | 51.2% / 61.1% | 80.0% / 84.6% | 5.00 / 5.00 | 445 s | $0.000 |
| q06 | Computer vision | 48.6% / 55.3% | 65.4% / 78.9% | 5.00 / 5.00 | 162 s | $0.000 |
| q07 | Model compression | 43.6% / 47.6% | 82.9% / 73.3% | 5.00 / 5.00 | 241 s | $0.000 |
| q08 | Information retrieval | failed: Incomplete - some model calls failed, re-run with --resume:  | | | | |
| q09 | Efficient transformers | 46.4% / 28.1% | 82.8% / 55.0% | 5.00 / 5.00 | 188 s | $0.000 |
| q10 | Alignment | 63.6% / 41.1% | 90.0% / 58.8% | 5.00 / 5.00 | 234 s | $0.000 |
| q11 | Continual learning | failed: Incomplete - some model calls failed, re-run with --resume:  | | | | |
| q16 | Code generation | 40.5% / 39.5% | 76.7% / 55.0% | 5.00 / 5.00 | 213 s | $0.000 |
| q17 | Reinforcement learning | 58.3% / 34.5% | 88.6% / 70.0% | 5.00 / 5.00 | 158 s | $0.000 |
| q18 | Federated learning | 60.0% / 39.1% | 78.3% / 73.1% | 5.00 / 5.00 | 177 s | $0.000 |
| q19 | Model compression | 57.1% / 38.3% | 88.5% / 82.8% | 5.00 / 5.00 | 270 s | $0.000 |
| q20 | Multimodal learning | failed: Incomplete - some model calls failed, re-run with --resume:  | | | | |

## Metric definitions

- **Citation accuracy:** share of (sentence, cited passage) pairs where an LLM judge finds that the passage
  fully supports the sentence. The lenient variant also counts partial support.
- **Faithfulness:** share of atomic claims in the final report labelled *supported* when verified against
  the source passages with the Critic's claim-splitting and verification procedure (applied to both arms).
- **Answer relevance:** LLM-judge score from 1 (irrelevant) to 5 (answers the question completely).
- **Latency / cost:** wall-clock time and API cost ($0 on free tiers) of the pipeline for one question (judging excluded).
  For the *without Critic* arm these stop at the first draft.

## Threats to validity

- The judges are LLMs; when the judge model equals the pipeline model, the Critic-optimised report may be
  favoured by a judge that shares its preferences. Use `--judge-model` to cross-check with another model.
- Faithfulness reuses the Critic's verification procedure, so it measures consistency with that procedure;
  citation accuracy uses an independent pairwise prompt and is the less circular of the two.
- Search results (arXiv / Semantic Scholar) change over time and API rate limits can reduce the number of
  papers found, so absolute numbers vary between runs; the paired design keeps the comparison fair.
