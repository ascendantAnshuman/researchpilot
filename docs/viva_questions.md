# ResearchPilot – Likely Viva Questions

Short answers you can expand in your own words. File references point to where the answer lives in the code.

---

**1. Why build a multi-agent pipeline instead of one big prompt?**
Each step has a different job, input and failure mode, so splitting them makes every step simpler, testable
and replaceable. Search and PDF parsing are not LLM tasks at all. The graph also lets the program, not the model,
decide control flow, e.g. "revise only if a claim is unsupported, at most once". A single prompt cannot read 10
full papers, cite exact passages and check its own claims reliably. (`backend/app/agents/graph.py`)

**2. How does ResearchPilot reduce hallucination?**
In three ways. (a) *Retrieval grounding:* the Synthesizer only sees retrieved passages and must cite one for every
factual sentence. (b) *Verification:* the Critic splits the draft into atomic claims and checks each against the
passages, and unsupported claims trigger a revision. (c) *Enforcement:* claims still unsupported after the
revision are removed (or flagged). Citations to non-existent evidence are also stripped automatically.

**3. What exactly does the Critic do, and why only one revision?**
It extracts atomic claims (one fact each), gathers evidence for each (the passages the sentence cited plus the
two closest passages from a fresh vector search), and labels each claim supported, partial or unsupported, in
batches of eight. One revision is a cost/benefit choice: each extra round costs another Synthesizer call and
another full verification. The revision prompt reuses the cached evidence block, and anything still unsupported
is removed at the end anyway. `MAX_REVISIONS` makes it configurable. (`agents/critic.py`)

**4. Explain your ablation study. Why is it "paired"?**
For every question the pipeline runs once with the Critic. The Synthesizer's first draft, finalised without
verification, is the "without Critic" arm; the verified and revised report is the "with Critic" arm. Both use
the same papers, evidence and first draft, so the only difference is the Critic. Two independent runs would mix
in randomness from search results and different drafts. I report the mean paired difference with a 95%
bootstrap confidence interval, and win/tie/loss counts. (`eval/ablation.py`)

**5. How are the metrics computed, and what are their weaknesses?**
- *Citation accuracy:* an LLM judge checks every (sentence, cited passage) pair.
- *Faithfulness:* atomic claims verified against the sources (FActScore-style).
- *Relevance:* a 1–5 judge score.
- *Latency and cost:* from the pipeline's own metering.

Weaknesses: the judges are LLMs and can be wrong or biased toward their own style, so the judge model is
configurable (a different model reduces bias). Faithfulness uses the same procedure as the Critic, so citation
accuracy, which uses an independent pairwise prompt, is the less circular metric. Validating the judges against
a small human-labelled sample is the next step.

**6. Why all-MiniLM-L6-v2, and what is "windowed pooling"?**
MiniLM is small (~90 MB), free and fast on a CPU, which meets the "normal laptop, no GPU" requirement. But it
only reads 256 word-pieces, while my chunks are ~800 tokens, so a naive embedding would ignore two thirds of
every chunk. I split each chunk into model-sized windows, embed each window, average the vectors and re-normalise.
(`services/vector_store.py`)

**7. Why chunks of 800 tokens with 100 overlap?**
Large enough that a chunk contains a complete argument or result for Claude to read and cite; small enough that
~24 chunks from 10 papers fit comfortably in one prompt. The overlap keeps an idea from being cut in half at a
boundary. Chunks follow sentence boundaries and never cross section boundaries, so each citation points to one
section and page range. (`services/pdf_parser.py`)

**8. How do you rank papers?**
`score = 0.5·relevance + 0.3·citations + 0.2·recency`, where relevance is the embedding similarity between the
question and the title+abstract (min-max normalised), citations are log-scaled (so one famous paper cannot
dominate), and recency decays linearly over 10 years. Clearly off-topic papers (similarity < 0.2) are dropped.
Duplicates across arXiv and Semantic Scholar are merged by arXiv id, DOI or normalised title first.
(`agents/search.py`)

**9. How does the live progress work? Why SSE rather than WebSockets?**
The research runs as a background asyncio task; each agent emits events that are stored in the run's history
and pushed to every listener's queue. `GET /api/research/{id}/events` streams them as Server-Sent Events, first
replaying the history, so a refresh or reconnect loses nothing. SSE fits because data only flows server → browser
and it is plain HTTP. The browser reads the stream with `fetch` rather than `EventSource`, because `EventSource`
cannot send the `Authorization` header. (`services/run_manager.py`, `frontend/src/lib/sse.ts`)

**10. What happens when things fail? And what would you change to scale it?**
- *External APIs:* exponential backoff with jitter; `Retry-After` is honoured; a source that keeps failing (e.g.
  Semantic Scholar HTTP 429) is skipped for the rest of the run.
- *PDFs:* a broken, too-large or paywalled PDF falls back to the abstract.
- *LLM:* SDK retries, then clear errors; a failed Critic leaves the report marked unverified.
- *Server restarts:* unfinished runs are marked failed with a reason.
- *Security:* bcrypt passwords, JWT, per-user 404s, secrets only in `.env`, and a switch to close registration.

To scale: the run manager keeps state in one process, so it must run a single worker. I would move runs to a job
queue (Redis + Celery or Arq), events to Redis pub/sub, SQLite to PostgreSQL and ChromaDB to pgvector or a vector
database server. The API could then run several workers.
