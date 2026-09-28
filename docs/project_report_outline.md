# ResearchPilot – Project Report Outline

A chapter-by-chapter plan for the final-year project report. Each chapter lists what to write, which parts of
the repository to draw from, and which figures and tables to include. Suggested length: 40–60 pages.

---

## Front matter
- Title page, certificate, declaration, acknowledgements.
- **Abstract** (see Chapter 0), table of contents, list of figures, list of tables, list of abbreviations
  (LLM, RAG, SSE, JWT, API, PDF, CI).

## 0. Abstract (≈250 words)
Write it last. Cover, in order:
1. **Problem:** literature reviews are slow to write, and LLM-written summaries hallucinate facts and citations.
2. **Approach:** ResearchPilot, a multi-agent system (Planner, Search, Reader, Synthesizer, Critic) that grounds
   every sentence in retrieved passages from real papers and verifies each claim before the report is shown.
3. **Implementation:** LangGraph orchestration, LLM gateway (free Gemini tier, optional Claude), ChromaDB + MiniLM retrieval, FastAPI + Next.js web app
   with live progress streaming.
4. **Evaluation:** 25 research questions; citation accuracy, faithfulness, relevance, latency and cost; a paired
   ablation of the Critic agent.
5. **Key result:** the headline numbers from `eval/results.md` (with/without Critic and the difference).
   `python -m eval.deliverables` prints a ready-made results paragraph.

## 1. Introduction
- **1.1 Motivation:** the growth of publications (arXiv receives thousands of papers per month) and the cost of
  manual literature reviews.
- **1.2 Problem statement:** produce a structured, cited literature review for an arbitrary research question,
  where every claim can be traced to a source passage.
- **1.3 Objectives:** real paper search; full-text reading; grounded synthesis; claim verification; follow-up Q&A;
  export; measurable quality (evaluation and ablation).
- **1.4 Scope and assumptions:** open-access papers (arXiv / Semantic Scholar); English; CPU-only local embeddings.
- **1.5 Contributions:** (a) a verified multi-agent RAG pipeline with chunk-level citations; (b) a paired ablation
  design isolating the Critic's effect; (c) a complete, tested, deployable web application.
- **1.6 Report organisation.**

## 2. Literature Survey
Group the related work, and end each group with how ResearchPilot differs.

| Theme | Key works to discuss |
|---|---|
| Retrieval-augmented generation | Lewis et al. (2020) *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*; Karpukhin et al. (2020) *Dense Passage Retrieval* |
| Hallucination | Ji et al. (2023) *Survey of Hallucination in Natural Language Generation* |
| Self-verification and refinement | Madaan et al. (2023) *Self-Refine*; Shinn et al. (2023) *Reflexion*; Dhuliawala et al. (2023) *Chain-of-Verification*; Asai et al. (2023) *Self-RAG* |
| Citations and attribution | Gao et al. (2023) *Enabling LLMs to Generate Text with Citations (ALCE)*; Min et al. (2023) *FActScore* (atomic-claim evaluation) |
| LLM agents / multi-agent systems | Yao et al. (2023) *ReAct*; Wu et al. (2023) *AutoGen*; LangGraph documentation |
| Research assistants | Shao et al. (2024) *STORM* (Wikipedia-style articles); Lála et al. (2023) *PaperQA* |
| Evaluation | Es et al. (2023) *RAGAS*; Zheng et al. (2023) *Judging LLM-as-a-Judge with MT-Bench*; Efron & Tibshirani (1993) *An Introduction to the Bootstrap* |
| Embeddings | Reimers & Gurevych (2019) *Sentence-BERT*; Wang et al. (2020) *MiniLM* |

- End with a **comparison table**: rows are systems (single-prompt LLM, plain RAG, Self-RAG, PaperQA, STORM,
  ResearchPilot); columns are real paper search, full-text reading, chunk-level citations, claim verification,
  revision loop, web UI, open evaluation.
- Before submitting, confirm each reference's venue and year in Google Scholar, and format them consistently
  (e.g. IEEE style).

## 3. System Analysis and Requirements
- **Functional requirements:** authentication; start a research run; live progress; report with themed sections,
  comparison table and citations; verification panel; follow-up chat; Markdown/PDF export; history; delete.
- **Non-functional requirements:** runs on an 8–16 GB laptop without a GPU; failures are handled gracefully;
  secrets stay in `.env`; every table is scoped to its user; runs for free on the Gemini free tier (quota-aware
  model pool); tested (380+ automated tests).
- **Use-case diagram**, and a short feasibility and cost analysis (API cost per question from the evaluation).

## 4. System Architecture
- **Figure 4.1:** overall architecture (`docs/architecture.md`, section 1; render the Mermaid diagram to an image).
- **Figure 4.2:** LangGraph state machine (section 2), including the conditional Critic → Synthesizer loop and
  the early exits.
- **Figure 4.3:** sequence diagram of one research run over SSE (section 3).
- **Data model:** ER diagram of `users`, `research_sessions`, `reports`, `session_papers`, `chat_messages`
  (`backend/app/models.py`).
- **API design:** the endpoint table (`docs/architecture.md` → HTTP API).
- **Technology choices and why:** LangGraph (explicit, testable control flow); schema-constrained LLM outputs
  (validated JSON, Gemini `response_schema` / Claude structured outputs); free Gemini tier with a quota-aware
  model pool (cost $0); ChromaDB + MiniLM (free, local, CPU); FastAPI (async + SSE); Next.js.

## 5. Module Design (one subsection per module)
For each module give: its purpose, inputs and outputs, the algorithm, and the failure handling.
- **5.1 Planner:** 3–5 sub-queries; normalisation; heuristic fallback plan (`agents/planner.py`).
- **5.2 Search:** query building for arXiv (AND → OR fallback); Semantic Scholar; de-duplication by arXiv id,
  DOI and normalised title; citation enrichment; ranking formula
  `score = 0.5·relevance + 0.3·log-scaled citations + 0.2·recency`; circuit breaker for rate-limited sources
  (`agents/search.py`).
- **5.3 Reader:** PDF download with a size limit and a real-PDF check; section detection; references stripped;
  800/100-token sentence-aware chunking; windowed MiniLM embeddings; structured extraction; abstract fallback
  (`agents/reader.py`, `services/pdf_parser.py`, `services/vector_store.py`).
- **5.4 Synthesizer:** evidence selection (per-paper coverage, then by score); `[E#]` citation protocol;
  prompt caching of the evidence block for the revision (`agents/synthesizer.py`).
- **5.5 Critic:** atomic-claim extraction; evidence pooling per batch; supported / partial / unsupported
  labels; routing rule; removal policy (`agents/critic.py`, `agents/finalize.py`).
- **5.6 LLM gateway:** model capabilities, retries, concurrency limit, cost metering (`app/llm.py`).
- **5.7 Web layer:** background run manager, SSE protocol with history replay, JWT auth, chat RAG, export.

## 6. Implementation
- Development environment and tools; the repository layout (from the README).
- Important code excerpts, kept short and explained: the LangGraph wiring (`build_graph`), the citation
  renumbering (`CitationNumberer`), the SSE parser (`frontend/src/lib/sse.ts`).
- **Testing strategy:** unit tests with fakes (the fake LLM plants a false claim so the Critic is actually
  exercised); API tests including SSE; live integration tests; frontend unit tests; an end-to-end browser run.
  Include a table of test counts per area.
- **Deployment:** Docker images, docker-compose, Vercel + Railway (`docs/deployment.md`).
- **UI screenshots:** use screenshots from a real run (see `docs/screenshots/` for which screens to show).

## 7. Results and Ablation
- **7.1 Experimental setup:** 25 questions (`eval/questions.json`); pipeline and judge models; top-N papers;
  hardware; dates. Note that search results change over time.
- **7.2 Metrics:** definitions of citation accuracy (strict and lenient), faithfulness, answer relevance,
  latency and cost (copy them from `eval/results.md`).
- **7.3 Main results table:** the "Main result: Critic ablation" table from `eval/results.md`.
- **7.4 Ablation discussion:** why the paired design (same papers, same first draft) isolates the Critic.
  Report how many revisions happened and how many claims were removed, and interpret the confidence intervals.
- **7.5 Latency and cost analysis:** the per-stage latency table; which stage dominates, and the extra cost of
  the Critic.
- **7.6 Qualitative analysis:** 2–3 examples from `results.json` and the Verification tab: a claim the Critic
  caught, a partially supported claim, and a failure case.
- **7.7 Threats to validity:** LLM judges; the circularity of faithfulness; API rate limits; paper availability.
  Copy the list from `eval/results.md`.

## 8. Conclusion
Summarise what was built, the measured effect of the Critic, and what the results mean for trustworthy
AI-assisted research tools.

## 9. Future Work
- Use a different model family as an independent judge, plus a small human-labelled set to validate the judges.
- A job queue (Redis/Celery), PostgreSQL and pgvector, so the backend can run several workers.
- Figure and table extraction from PDFs; multilingual papers; more sources (PubMed, ACL Anthology).
- Streaming token-by-token chat answers; letting users flag bad citations to improve ranking.
- Several revision rounds with a cost budget; a learned claim-risk model to verify selectively.

## References and appendices
- References in IEEE format (see Chapter 2).
- **Appendix A:** `.env` configuration reference.
- **Appendix B:** API reference (`/docs` screenshots).
- **Appendix C:** full per-question results table from `eval/results.md`.
- **Appendix D:** test summary (output of `pytest` and `npm test`).
