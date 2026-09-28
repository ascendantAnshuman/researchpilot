"""Paired Critic ablation from a single pipeline run.

The pipeline runs once with the Critic enabled. The moment the Synthesizer
produces its *first* draft, the state is snapshotted: that draft, finalised
without any verification, is the "without Critic" report. The run then
continues (Critic -> optional revision -> removal of unsupported claims) to
produce the "with Critic" report.

Both arms therefore share the same plan, papers, evidence and first draft, so
any difference between them is caused by the Critic alone - a controlled,
paired comparison at the cost of one pipeline run per question.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

import eval  # noqa: F401  (puts backend/ on sys.path)
from app.agents.finalize import build_report
from app.agents.graph import get_graph, initial_state
from app.agents.state import PipelineContext
from app.llm import UsageTracker
from app.schemas import ResearchReport, StageMetrics

logger = logging.getLogger(__name__)


@dataclass
class PairedRun:
    """The two reports compared by the ablation."""

    with_critic: ResearchReport
    without_critic: ResearchReport
    draft_available: bool


def _apply_timings(report: ResearchReport, stage_seconds: dict[str, float], total_seconds: float) -> None:
    """Copy per-stage and total wall-clock times onto a report's metrics."""
    for stage, seconds in stage_seconds.items():
        report.metrics.stages.setdefault(stage, StageMetrics()).seconds = round(seconds, 3)
    report.metrics.total_seconds = round(total_seconds, 3)


async def run_paired(question: str, context: PipelineContext) -> PairedRun:
    """Run the pipeline once and return both arms of the Critic ablation."""
    settings = context.settings.model_copy(update={"enable_critic": True})
    critic_context = dataclasses.replace(context, settings=settings)

    started = time.perf_counter()
    first_draft_state: dict[str, Any] | None = None
    usage_at_first_draft: dict[str, Any] = {}
    seconds_at_first_draft = 0.0
    capture_next_values = False
    final_state: dict[str, Any] = {}

    async for mode, payload in get_graph().astream(
        initial_state(question, settings), context=critic_context, stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            if first_draft_state is None and "synthesizer" in payload:
                capture_next_values = True
        elif mode == "values":
            final_state = payload
            if capture_next_values:
                first_draft_state = dict(payload)
                usage_at_first_draft = copy.deepcopy(context.llm.tracker.stages)
                seconds_at_first_draft = time.perf_counter() - started
                capture_next_values = False

    with_critic: ResearchReport | None = final_state.get("report")
    if with_critic is None:
        raise RuntimeError("The pipeline finished without producing a report")
    _apply_timings(with_critic, final_state.get("stage_seconds", {}), time.perf_counter() - started)

    if first_draft_state is None:
        # No draft was written (e.g. no papers found): both arms are the same "no review" report.
        without_critic = with_critic.model_copy(deep=True)
        without_critic.metrics.critic_enabled = False
        return PairedRun(with_critic, without_critic, draft_available=False)

    tracker = UsageTracker()
    tracker.stages = usage_at_first_draft
    no_critic_settings = settings.model_copy(update={"enable_critic": False})
    draft_state = {**first_draft_state, "verification": [], "revisions": 0}
    without_critic = build_report(draft_state, no_critic_settings, tracker)  # type: ignore[arg-type]
    _apply_timings(without_critic, first_draft_state.get("stage_seconds", {}), seconds_at_first_draft)
    logger.info(
        "Paired run done: with critic %d claims checked / %d revision(s); without critic %d sentences",
        with_critic.metrics.claims_total,
        with_critic.metrics.revisions,
        len(without_critic.sentences),
    )
    return PairedRun(with_critic, without_critic, draft_available=True)
