"""Turn eval/results.json into résumé bullets and a results paragraph - using your real numbers.

    backend\\.venv\\Scripts\\python -m eval.deliverables                 # print to the terminal
    backend\\.venv\\Scripts\\python -m eval.deliverables --out docs/my_results.md

Run it after ``python -m eval.run_eval``. The wording adapts to the measured
result: if the Critic did not improve a metric, the text says so rather than
claiming an improvement.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval import EVAL_DIR
from eval.metrics import EvalRun

MIN_RELIABLE_QUESTIONS = 10


def _pct(value: float) -> str:
    """0.873 -> '87.3%'."""
    return f"{value * 100:.1f}%"


def _pts(value: float) -> str:
    """0.052 -> '+5.2 pts'."""
    return f"{value * 100:+.1f} pts"


def _metric(summary: dict[str, Any], key: str) -> dict[str, Any] | None:
    """A metric's summary block, or None if it could not be measured."""
    data = summary.get("metrics", {}).get(key)
    return data if data and data.get("n") else None


def _change(data: dict[str, Any], label: str) -> str:
    """Honest description of how a percentage metric changed with the Critic."""
    before, after, delta = data["without_critic_mean"], data["with_critic_mean"], data["delta_mean"]
    low, high = data["delta_ci95"]
    interval = f"95% CI {_pts(low)} to {_pts(high)}"
    if low > 0:
        return f"raised {label} from {_pct(before)} to {_pct(after)} ({_pts(delta)}, {interval})"
    if high < 0:
        return f"lowered {label} from {_pct(before)} to {_pct(after)} ({_pts(delta)}, {interval})"
    return f"changed {label} from {_pct(before)} to {_pct(after)} ({_pts(delta)}; not significant, {interval})"


def resume_bullets(run: EvalRun) -> list[str]:
    """Four résumé bullets built from the evaluation summary."""
    summary = run.summary
    questions = summary.get("questions_completed", 0)
    faithfulness = _metric(summary, "faithfulness")
    citations = _metric(summary, "citation_accuracy")
    relevance = _metric(summary, "relevance")
    latency = _metric(summary, "latency_seconds")
    cost = _metric(summary, "cost_usd")

    critic_effects = [text for text in (
        _change(faithfulness, "claim faithfulness") if faithfulness else None,
        _change(citations, "citation accuracy") if citations else None,
    ) if text]
    effect_text = " and ".join(critic_effects) if critic_effects else "was measured on faithfulness and citation accuracy"

    bullets = [
        "Built ResearchPilot, a multi-agent research copilot (LangGraph, FastAPI, Next.js, Claude API) in which "
        "Planner, Search, Reader, Synthesizer and Critic agents turn a research question into a cited, fact-checked "
        "literature review from arXiv and Semantic Scholar papers, with live agent progress streamed over SSE.",
        f"Designed a Critic agent that verifies every atomic claim against its source passages and triggers a revision; "
        f"in a paired ablation over {questions} research questions it {effect_text}.",
    ]
    pipeline_parts = [
        "Implemented retrieval-augmented generation over full-text PDFs (PyMuPDF, 800-token chunks, MiniLM embeddings "
        "with windowed pooling, ChromaDB) with chunk-level citations and graceful fallbacks for rate limits and "
        "unreadable PDFs"
    ]
    if latency and cost:
        pipeline_parts.append(
            f"averaging {summary.get('mean_papers', 0):.1f} papers per question, "
            f"{latency['with_critic_mean']:.0f} s end-to-end and ${cost['with_critic_mean']:.2f} per query"
        )
    bullets.append(", ".join(pipeline_parts) + ".")
    relevance_text = f" (answer relevance {relevance['with_critic_mean']:.2f}/5)" if relevance else ""
    bullets.append(
        "Built an automated evaluation harness with LLM-as-judge metrics for citation accuracy, faithfulness and "
        f"relevance{relevance_text} and bootstrap confidence intervals; shipped with 340+ automated tests "
        "(pytest, Vitest), Docker Compose and a Vercel + Railway deployment."
    )
    return bullets


def results_paragraph(run: EvalRun) -> str:
    """A paragraph for the 'Results & Ablation' chapter of the project report."""
    summary = run.summary
    questions = summary.get("questions_completed", 0)
    parts = [
        f"We evaluated ResearchPilot on {questions} research questions spanning machine learning and NLP topics, "
        f"using {run.pipeline_model} for the pipeline and {run.judge_model} as the judge. For each question the "
        "pipeline ran once; the Synthesizer's first draft, finalised without verification, formed the "
        "without-Critic arm, so both arms shared the same papers and evidence."
    ]
    for key, label in (("faithfulness", "claim faithfulness"), ("citation_accuracy", "citation accuracy")):
        data = _metric(summary, key)
        if data:
            parts.append(
                f"With the Critic, {label} was {_pct(data['with_critic_mean'])} versus {_pct(data['without_critic_mean'])} "
                f"without it ({_pts(data['delta_mean'])}, 95% bootstrap CI {_pts(data['delta_ci95'][0])} to "
                f"{_pts(data['delta_ci95'][1])}; better on {data['critic_better']}, worse on {data['critic_worse']}, "
                f"tied on {data['ties']} questions)."
            )
    relevance = _metric(summary, "relevance")
    if relevance:
        parts.append(
            f"Answer relevance was {relevance['with_critic_mean']:.2f}/5 with the Critic and "
            f"{relevance['without_critic_mean']:.2f}/5 without it."
        )
    latency, cost = _metric(summary, "latency_seconds"), _metric(summary, "cost_usd")
    if latency and cost:
        parts.append(
            f"The Critic added {latency['delta_mean']:.0f} s of latency and ${cost['delta_mean']:.3f} of API cost per "
            f"question on average (totals: {latency['with_critic_mean']:.0f} s and ${cost['with_critic_mean']:.3f}), "
            f"triggering {summary.get('mean_revisions', 0):.2f} revisions and removing "
            f"{summary.get('mean_claims_removed', 0):.1f} unsupported claims per report."
        )
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="python -m eval.deliverables", description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=EVAL_DIR / "results.json")
    parser.add_argument("--out", type=Path, help="Also write the text to this Markdown file.")
    args = parser.parse_args(argv)

    if not args.results.exists():
        sys.stderr.write(f"{args.results} not found. Run `python -m eval.run_eval` first.\n")
        return 1
    run = EvalRun.model_validate_json(args.results.read_text(encoding="utf-8"))
    questions = run.summary.get("questions_completed", 0)
    lines = ["# Résumé bullets (from your evaluation results)", ""]
    lines += [f"- {bullet}" for bullet in resume_bullets(run)]
    lines += ["", "# Results paragraph for the project report", "", results_paragraph(run), ""]
    if questions < MIN_RELIABLE_QUESTIONS:
        lines.append(
            f"> Note: only {questions} question(s) were evaluated. Run all 25 before quoting these numbers."
        )
    text = "\n".join(lines) + "\n"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.stdout.write(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
