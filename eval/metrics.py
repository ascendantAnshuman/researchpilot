"""Result records, paired statistics and the Markdown summary for the evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

import eval  # noqa: F401  (puts backend/ on sys.path)
from app.schemas import ResearchReport

from eval.judges import CitationScore, FaithfulnessScore

Arm = Literal["with_critic", "without_critic"]
STAGES = ("planner", "search", "reader", "synthesizer", "critic", "finalize")
BOOTSTRAP_SAMPLES = 5000


class ArmResult(BaseModel):
    """All measurements for one arm (with or without the Critic) of one question."""

    citation_accuracy: float | None = None
    citation_accuracy_lenient: float | None = None
    citations_checked: int = 0
    faithfulness: float | None = None
    claims_checked: int = 0
    claims_unsupported: int = 0
    relevance: float | None = None
    relevance_rationale: str | None = None
    latency_seconds: float = 0.0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    papers: int = 0
    papers_full_text: int = 0
    revisions: int = 0
    claims_removed: int = 0
    report_title: str = ""
    notes: list[str] = Field(default_factory=list)


class QuestionResult(BaseModel):
    """Both arms for one benchmark question."""

    id: str
    topic: str
    question: str
    with_critic: ArmResult | None = None
    without_critic: ArmResult | None = None
    judge_cost_usd: float = 0.0
    error: str | None = None


class EvalRun(BaseModel):
    """Everything written to ``eval/results.json``."""

    started_at: datetime
    finished_at: datetime | None = None
    pipeline_model: str
    judge_model: str
    settings: dict[str, Any] = Field(default_factory=dict)
    questions: list[QuestionResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


def arm_result(
    report: ResearchReport,
    citations: CitationScore | None,
    faithfulness: FaithfulnessScore | None,
    relevance: tuple[int, str] | None,
    notes: list[str],
) -> ArmResult:
    """Combine a report's own metrics with the judges' scores."""
    metrics = report.metrics
    return ArmResult(
        citation_accuracy=citations.accuracy if citations else None,
        citation_accuracy_lenient=citations.lenient_accuracy if citations else None,
        citations_checked=citations.pairs if citations else 0,
        faithfulness=faithfulness.faithfulness if faithfulness else None,
        claims_checked=faithfulness.claims if faithfulness else 0,
        claims_unsupported=faithfulness.unsupported if faithfulness else 0,
        relevance=float(relevance[0]) if relevance else None,
        relevance_rationale=relevance[1] if relevance else None,
        latency_seconds=metrics.total_seconds,
        stage_seconds={stage: metrics.stages[stage].seconds for stage in STAGES if stage in metrics.stages},
        cost_usd=metrics.total_cost_usd,
        input_tokens=metrics.total_input_tokens,
        output_tokens=metrics.total_output_tokens,
        papers=metrics.papers_found,
        papers_full_text=metrics.papers_full_text,
        revisions=metrics.revisions,
        claims_removed=sum(claim.action == "removed" for claim in report.verification),
        report_title=report.title,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """How a metric is named, formatted and compared."""

    key: str
    label: str
    kind: Literal["percent", "score", "seconds", "usd"]
    higher_is_better: bool


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("citation_accuracy", "Citation accuracy", "percent", True),
    MetricSpec("citation_accuracy_lenient", "Citation accuracy (incl. partial)", "percent", True),
    MetricSpec("faithfulness", "Faithfulness (supported claims)", "percent", True),
    MetricSpec("relevance", "Answer relevance (1-5)", "score", True),
    MetricSpec("latency_seconds", "Latency per query", "seconds", False),
    MetricSpec("cost_usd", "Pipeline cost per query", "usd", False),
)


def bootstrap_ci(deltas: np.ndarray, seed: int = 7) -> tuple[float, float]:
    """95% bootstrap confidence interval of the mean paired difference."""
    if deltas.size < 2:
        value = float(deltas.mean()) if deltas.size else 0.0
        return value, value
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(BOOTSTRAP_SAMPLES, deltas.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(results: list[QuestionResult]) -> dict[str, Any]:
    """Per-metric means for both arms, paired differences with 95% CIs, and win/tie/loss counts."""
    complete = [result for result in results if result.with_critic and result.without_critic]
    summary: dict[str, Any] = {
        "questions_total": len(results),
        "questions_completed": len(complete),
        "questions_failed": sum(result.error is not None for result in results),
        "metrics": {},
        "stage_seconds": {},
    }
    for spec in METRICS:
        pairs = [
            (getattr(result.with_critic, spec.key), getattr(result.without_critic, spec.key))
            for result in complete
        ]
        pairs = [(with_value, without_value) for with_value, without_value in pairs if with_value is not None and without_value is not None]
        if not pairs:
            summary["metrics"][spec.key] = {"n": 0}
            continue
        with_values = np.array([pair[0] for pair in pairs], dtype=float)
        without_values = np.array([pair[1] for pair in pairs], dtype=float)
        deltas = with_values - without_values
        better = deltas > 1e-9 if spec.higher_is_better else deltas < -1e-9
        worse = deltas < -1e-9 if spec.higher_is_better else deltas > 1e-9
        low, high = bootstrap_ci(deltas)
        summary["metrics"][spec.key] = {
            "n": len(pairs),
            "with_critic_mean": float(with_values.mean()),
            "without_critic_mean": float(without_values.mean()),
            "with_critic_std": float(with_values.std(ddof=1)) if len(pairs) > 1 else 0.0,
            "without_critic_std": float(without_values.std(ddof=1)) if len(pairs) > 1 else 0.0,
            "delta_mean": float(deltas.mean()),
            "delta_ci95": [low, high],
            "critic_better": int(better.sum()),
            "critic_worse": int(worse.sum()),
            "ties": int(len(pairs) - better.sum() - worse.sum()),
        }
    for arm in ("with_critic", "without_critic"):
        per_stage: dict[str, float] = {}
        for stage in STAGES:
            values = [getattr(result, arm).stage_seconds.get(stage, 0.0) for result in complete]
            per_stage[stage] = float(np.mean(values)) if values else 0.0
        summary["stage_seconds"][arm] = per_stage
    if complete:
        summary["mean_revisions"] = float(np.mean([result.with_critic.revisions for result in complete]))  # type: ignore[union-attr]
        summary["mean_claims_removed"] = float(np.mean([result.with_critic.claims_removed for result in complete]))  # type: ignore[union-attr]
        summary["mean_papers"] = float(np.mean([result.with_critic.papers for result in complete]))  # type: ignore[union-attr]
        summary["mean_papers_full_text"] = float(np.mean([result.with_critic.papers_full_text for result in complete]))  # type: ignore[union-attr]
        summary["total_pipeline_cost_usd"] = float(sum(result.with_critic.cost_usd for result in complete))  # type: ignore[union-attr]
    summary["total_judge_cost_usd"] = float(sum(result.judge_cost_usd for result in results))
    return summary


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value: float | None, kind: str) -> str:
    """Format a metric value for the table."""
    if value is None:
        return "n/a"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "score":
        return f"{value:.2f}"
    if kind == "seconds":
        return f"{value:.0f} s"
    return f"${value:.3f}"


def _fmt_delta(value: float, kind: str) -> str:
    """Format a signed difference (percentage points for percentages)."""
    sign = "+" if value >= 0 else "-"
    magnitude = abs(value)
    if kind == "percent":
        return f"{sign}{magnitude * 100:.1f} pts"
    if kind == "score":
        return f"{sign}{magnitude:.2f}"
    if kind == "seconds":
        return f"{sign}{magnitude:.0f} s"
    return f"{sign}${magnitude:.3f}"


def render_markdown(run: EvalRun) -> str:
    """The human-readable results summary (``eval/results.md``)."""
    summary = run.summary
    lines = [
        "# ResearchPilot evaluation results",
        "",
        f"- **Run:** {run.started_at:%Y-%m-%d %H:%M} UTC",
        f"- **Pipeline model:** `{run.pipeline_model}` · **Judge model:** `{run.judge_model}`",
        f"- **Questions:** {summary.get('questions_completed', 0)} completed of {summary.get('questions_total', 0)}"
        f" ({summary.get('questions_failed', 0)} failed)",
        f"- **Papers per question:** {summary.get('mean_papers', 0):.1f} on average"
        f" ({summary.get('mean_papers_full_text', 0):.1f} read in full text)",
        f"- **Total cost:** pipeline ${summary.get('total_pipeline_cost_usd', 0):.2f}"
        f" + judges ${summary.get('total_judge_cost_usd', 0):.2f}",
        "",
        "## Main result: Critic ablation",
        "",
        "Both arms come from the same pipeline run (same papers, evidence and first draft); the *without Critic*",
        "arm is the first draft finalised without verification. Δ = with Critic − without Critic, with a 95%",
        "bootstrap confidence interval over questions.",
        "",
        "| Metric | With Critic | Without Critic | Δ (95% CI) | Critic better / worse / tie |",
        "|---|---|---|---|---|",
    ]
    for spec in METRICS:
        data = summary.get("metrics", {}).get(spec.key, {"n": 0})
        if not data.get("n"):
            lines.append(f"| {spec.label} | n/a | n/a | n/a | n/a |")
            continue
        low, high = data["delta_ci95"]
        lines.append(
            f"| {spec.label} | {_fmt(data['with_critic_mean'], spec.kind)} | {_fmt(data['without_critic_mean'], spec.kind)}"
            f" | {_fmt_delta(data['delta_mean'], spec.kind)} ({_fmt_delta(low, spec.kind)}, {_fmt_delta(high, spec.kind)})"
            f" | {data['critic_better']} / {data['critic_worse']} / {data['ties']} |"
        )
    lines += [
        "",
        f"On average the Critic triggered {summary.get('mean_revisions', 0):.2f} revision(s) and removed"
        f" {summary.get('mean_claims_removed', 0):.1f} unsupported claim(s) per report.",
        "",
        "## Latency per stage (mean seconds per question)",
        "",
        "| Stage | With Critic | Without Critic |",
        "|---|---|---|",
    ]
    stage_seconds = summary.get("stage_seconds", {})
    for stage in STAGES:
        with_value = stage_seconds.get("with_critic", {}).get(stage, 0.0)
        without_value = stage_seconds.get("without_critic", {}).get(stage, 0.0)
        lines.append(f"| {stage.capitalize()} | {with_value:.1f} | {without_value:.1f} |")

    lines += [
        "",
        "## Per-question results",
        "",
        "| ID | Topic | Citation acc. (C / no C) | Faithfulness (C / no C) | Relevance (C / no C) | Latency | Cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in run.questions:
        if result.error or not result.with_critic or not result.without_critic:
            lines.append(f"| {result.id} | {result.topic} | failed: {(result.error or 'incomplete')[:60]} | | | | |")
            continue
        with_arm, without_arm = result.with_critic, result.without_critic
        lines.append(
            f"| {result.id} | {result.topic}"
            f" | {_fmt(with_arm.citation_accuracy, 'percent')} / {_fmt(without_arm.citation_accuracy, 'percent')}"
            f" | {_fmt(with_arm.faithfulness, 'percent')} / {_fmt(without_arm.faithfulness, 'percent')}"
            f" | {_fmt(with_arm.relevance, 'score')} / {_fmt(without_arm.relevance, 'score')}"
            f" | {_fmt(with_arm.latency_seconds, 'seconds')} | {_fmt(with_arm.cost_usd, 'usd')} |"
        )

    lines += [
        "",
        "## Metric definitions",
        "",
        "- **Citation accuracy:** share of (sentence, cited passage) pairs where an LLM judge finds that the passage",
        "  fully supports the sentence. The lenient variant also counts partial support.",
        "- **Faithfulness:** share of atomic claims in the final report labelled *supported* when verified against",
        "  the source passages with the Critic's claim-splitting and verification procedure (applied to both arms).",
        "- **Answer relevance:** LLM-judge score from 1 (irrelevant) to 5 (answers the question completely).",
        "- **Latency / cost:** wall-clock time and API cost ($0 on free tiers) of the pipeline for one question (judging excluded).",
        "  For the *without Critic* arm these stop at the first draft.",
        "",
        "## Threats to validity",
        "",
        "- The judges are LLMs; when the judge model equals the pipeline model, the Critic-optimised report may be",
        "  favoured by a judge that shares its preferences. Use `--judge-model` to cross-check with another model.",
        "- Faithfulness reuses the Critic's verification procedure, so it measures consistency with that procedure;",
        "  citation accuracy uses an independent pairwise prompt and is the less circular of the two.",
        "- Search results (arXiv / Semantic Scholar) change over time and API rate limits can reduce the number of",
        "  papers found, so absolute numbers vary between runs; the paired design keeps the comparison fair.",
        "",
    ]
    return "\n".join(lines)
