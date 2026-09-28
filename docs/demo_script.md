# ResearchPilot – 3-Minute Demo Script

## Before the demo (preparation)
- A real run takes several minutes, so **run one question ~15 minutes before**, e.g. *"What are recent methods
  for reducing hallucination in RAG systems?"* You will show that finished report.
- Just before you start talking, **start a second question** so the live progress view is actually running
  when you open it.
- Have open: the app (logged in, dashboard), the finished report in a second tab, and `eval/results.md`.
- Zoom the browser to 110–125% so the room can read it. Close other tabs and notifications.
- **Backup:** the screenshots in `docs/screenshots/` and an exported PDF, in case the network fails.

---

## Script (≈3:00)

**0:00 – 0:20 · The problem**
> "Writing a literature review means finding papers, reading them and summarising them with correct citations.
> Chatbots can write summaries, but they invent facts and even citations. ResearchPilot is a team of AI agents
> that writes a cited review from real papers and fact-checks every claim before you see it."

**0:20 – 0:40 · Ask a question** *(dashboard)*
> "I type a research question. I can choose how many papers to use, and whether the Critic agent fact-checks the
> result."

Point at the example question and the "Fact-check with the Critic" option. Don't submit; the live run is
already going.

**0:40 – 1:20 · Live agents** *(open the running session from the sidebar)*
> "This run started a minute ago. Every agent reports live over Server-Sent Events. The **Planner** split the
> question into focused sub-queries. **Search** queried arXiv and Semantic Scholar, removed duplicates and ranked
> papers by relevance, citations and recency; here are the ones it picked. The **Reader** downloads each PDF,
> splits it into chunks, embeds them locally with MiniLM, and extracts the method, datasets and results. Next the
> **Synthesizer** writes, and the **Critic** verifies."

Point out a warning in the activity log if there is one, e.g. Semantic Scholar rate-limited:
> "When a source is rate-limited, the run continues with the other one."

**1:20 – 2:05 · The report** *(switch to the finished report tab)*
> "Here is the finished review: an executive summary, themed sections on methods, datasets, results and open
> problems, and a comparison table."

Click a citation such as [2]; it jumps to the reference.
> "Every sentence cites a specific passage from a specific paper."

Open **Verification**:
> "This is the key feature. The Critic split the draft into single claims and checked each against the sources:
> supported, partially supported, or unsupported, with the exact evidence quote. This claim was unsupported, so
> the Synthesizer revised the draft once, and whatever still couldn't be verified was removed."

Open **Papers** briefly:
> "For every paper, the Reader extracted the problem, method, datasets and limitations, and shows whether it
> read the full text or only the abstract."

**2:05 – 2:30 · Follow-up chat and export** *(Chat tab)*
Ask: *"Which datasets are used most often?"*
> "Follow-up questions are answered from the same papers, with the same reference numbers."

Click **PDF**:
> "And the report exports to PDF or Markdown."

**2:30 – 2:55 · Does the Critic actually help?** *(show `eval/results.md`)*
> "I evaluated the system on 25 research questions. For each one, I compared the report before and after the
> Critic on the same papers and the same draft: a paired ablation. With the Critic, faithfulness went from
> [X]% to [Y]% and citation accuracy from [A]% to [B]%, at a cost of about [N] extra seconds per question."

*(Read the real numbers from `eval/results.md`, or from the paragraph `python -m eval.deliverables` prints.)*

**2:55 – 3:00 · Close**
> "The whole system is tested, containerised with Docker, and deployed. Thank you, happy to take questions."

---

## Likely follow-up while you are demoing
- *"What if a paper is paywalled?"* → The Reader falls back to the abstract, and the Papers tab says "Abstract only".
- *"Is this expensive?"* → The report header shows the exact Claude API cost of the run.
- *"Can it make things up?"* → Show a removed or partially supported claim in the Verification tab, and explain the
  judges' limitations (see `docs/viva_questions.md`, Q5).
