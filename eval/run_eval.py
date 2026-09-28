"""Evaluation runner: benchmark questions -> paired Critic ablation -> LLM-as-judge metrics.

Run from the project root (``researchpilot/``) with the backend's virtual environment::

    backend\\.venv\\Scripts\\python -m eval.run_eval --limit 2 --yes      # quick trial (Windows)
    backend\\.venv\\Scripts\\python -m eval.run_eval --yes                # all 25 questions
    backend\\.venv\\Scripts\\python -m eval.run_eval --resume --yes       # continue after an interruption

Outputs (next to this file unless ``--output-dir`` is given):
    results.json   every measurement for every question, plus the summary
    results.md     the summary tables for the report / viva

Each question costs one pipeline run plus judging, so start with ``--limit``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from eval import EVAL_DIR
from app.agents.graph import open_pipeline_context
from app.config import Settings, get_settings
from app.llm import LLM, QUOTAS, LLMClient, LLMConfigurationError, LLMError
from app.logging_config import setup_logging
from app.schemas import ResearchReport
from app.services.session_store import settings_snapshot
from eval.ablation import PairedRun, run_paired
from eval.judges import judge_citations, judge_faithfulness, judge_relevance
from eval.metrics import ArmResult, EvalRun, QuestionResult, arm_result, render_markdown, summarize

logger = logging.getLogger("eval")


class Question(BaseModel):
    """One benchmark question."""

    id: str
    topic: str
    question: str


def load_questions(path: Path) -> list[Question]:
    """Read the benchmark questions from JSON."""
    return TypeAdapter(list[Question]).validate_json(path.read_text(encoding="utf-8"))


def make_pipeline_llm(settings: Settings) -> LLM:
    """The Claude client used by the pipeline (a new one per question, so costs are per question)."""
    return LLMClient(settings)


def make_judge_llm(settings: Settings) -> LLM:
    """The Claude client used by the judges."""
    return LLMClient(settings)


async def evaluate_arm(report: ResearchReport, context: Any, judge: LLM, judge_settings: Settings) -> ArmResult:
    """Score one report with all judges; a failing judge leaves its metric empty."""
    notes: list[str] = []
    citations = faithfulness = None
    relevance: tuple[int, str] | None = None
    try:
        citations = await judge_citations(report, context.vector_store, judge)
    except LLMConfigurationError:
        raise
    except LLMError as error:
        notes.append(f"citation judge failed: {error}")
    try:
        faithfulness = await judge_faithfulness(report, context.vector_store, judge, judge_settings)
    except LLMConfigurationError:
        raise
    except LLMError as error:
        notes.append(f"faithfulness judge failed: {error}")
    try:
        verdict = await judge_relevance(report, judge, judge_settings.llm_effort)
        relevance = (verdict.score, verdict.rationale)
    except LLMConfigurationError:
        raise
    except LLMError as error:
        notes.append(f"relevance judge failed: {error}")
    return arm_result(report, citations, faithfulness, relevance, notes)


LLM_FAILURE_MARKERS: tuple[str, ...] = ("summary extraction failed", "Critic failed", "Revision failed", "judge failed")


def pool_exhausted(*settings_list: Settings) -> bool:
    """True when every model configured for the pipeline and the judges has used up today's quota."""
    models = {model for settings in settings_list for model in (settings.llm_model, *settings.llm_fallback_models)}
    return all(QUOTAS.is_exhausted(model) for model in models)


def llm_failures(paired: PairedRun, arms: Sequence[ArmResult]) -> list[str]:
    """Pipeline warnings or judge notes showing that some model call failed (so the data is incomplete)."""
    texts = [*paired.with_critic.warnings, *(note for arm in arms for note in arm.notes)]
    return [text for text in texts if any(marker in text for marker in LLM_FAILURE_MARKERS)]


async def evaluate_question(item: Question, settings: Settings, judge_settings: Settings) -> QuestionResult:
    """Run the paired pipeline for one question and judge both arms.

    A question whose data is incomplete because a model call failed (typically the free daily quota
    running out) is recorded as an error, so ``--resume`` evaluates it again later.
    """
    result = QuestionResult(id=item.id, topic=item.topic, question=item.question)
    judge = make_judge_llm(judge_settings)
    try:
        async with open_pipeline_context(settings, llm=make_pipeline_llm(settings)) as context:
            paired = await run_paired(item.question, context)
            result.with_critic = await evaluate_arm(paired.with_critic, context, judge, judge_settings)
            result.without_critic = await evaluate_arm(paired.without_critic, context, judge, judge_settings)
            if not paired.draft_available:
                result.with_critic.notes.append("no review was produced (no usable papers)")
            failures = llm_failures(paired, [result.with_critic, result.without_critic])
            if failures:
                result.error = f"Incomplete - some model calls failed, re-run with --resume: {failures[0][:200]}"
    except LLMConfigurationError:
        raise
    except Exception as error:  # One broken question must not stop a multi-hour evaluation.
        logger.exception("Question %s failed", item.id)
        result.error = f"{type(error).__name__}: {error}"
    result.judge_cost_usd = round(judge.tracker.total().cost_usd, 6)
    return result


def save(run: EvalRun, output_dir: Path) -> tuple[Path, Path]:
    """Write results.json and results.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run.summary = summarize(run.questions)
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "results.md"
    json_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(run), encoding="utf-8")
    return json_path, markdown_path


async def run_all(
    questions: Sequence[Question], run: EvalRun, settings: Settings, judge_settings: Settings, output_dir: Path
) -> EvalRun:
    """Evaluate the questions one by one, saving after each so progress is never lost."""
    for index, item in enumerate(questions, start=1):
        logger.info("[%d/%d] %s: %s", index, len(questions), item.id, item.question)
        result = await evaluate_question(item, settings, judge_settings)
        run.questions = [existing for existing in run.questions if existing.id != item.id] + [result]
        save(run, output_dir)
        if pool_exhausted(settings, judge_settings):
            logger.warning(
                "Every configured model has used up today's free quota. Stopping after %d of %d question(s); "
                "run the same command with --resume after midnight Pacific time to continue.",
                index,
                len(questions),
            )
            break
        if result.error:
            logger.warning("[%d/%d] %s failed: %s", index, len(questions), item.id, result.error)
        elif result.with_critic and result.without_critic:
            logger.info(
                "[%d/%d] %s done: faithfulness %s -> %s, citation accuracy %s -> %s (without -> with Critic)",
                index,
                len(questions),
                item.id,
                result.without_critic.faithfulness,
                result.with_critic.faithfulness,
                result.without_critic.citation_accuracy,
                result.with_critic.citation_accuracy,
            )
    run.finished_at = datetime.now(UTC)
    save(run, output_dir)
    return run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json", help="Questions JSON file.")
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR, help="Where results.json/results.md go.")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions.")
    parser.add_argument("--ids", help="Comma-separated question ids to evaluate, e.g. q01,q05.")
    parser.add_argument("--pipeline-model", help="Model for the pipeline (default: LLM_MODEL).")
    parser.add_argument("--judge-model", help="Model for the judges (default: the pipeline model).")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Ignore LLM_FALLBACK_MODEL so every call uses exactly the chosen models (consistent measurements).",
    )
    parser.add_argument("--top-n", type=int, help="Papers per question (default: TOP_N_PAPERS).")
    parser.add_argument("--resume", action="store_true", help="Skip questions already completed in results.json.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending API credit.")
    return parser.parse_args(argv)


def confirm(question_count: int, pipeline_model: str, judge_model: str) -> bool:
    """Ask before starting a run that costs money."""
    prompt = (
        f"\nAbout to evaluate {question_count} question(s): one full pipeline run plus judging per question.\n"
        f"Pipeline model: {pipeline_model} | Judge model: {judge_model}\n"
        "This calls the Claude API and costs real money. Continue? [y/N] "
    )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    base = get_settings()
    overrides: dict[str, Any] = {}
    if args.pipeline_model:
        overrides["llm_model"] = args.pipeline_model
    if args.top_n:
        overrides["top_n_papers"] = args.top_n
    if args.no_fallback:
        overrides["llm_fallback_model"] = None
    settings = base.model_copy(update=overrides) if overrides else base
    judge_settings = settings.model_copy(update={"llm_model": args.judge_model or settings.llm_model})
    setup_logging(settings.log_level)

    if not settings.llm_configured or not judge_settings.llm_configured:
        missing = settings.llm_key_name if not settings.llm_configured else judge_settings.llm_key_name
        logger.error("%s is not set. Add it to the .env file in the project root.", missing)
        return 2

    questions = load_questions(args.questions)
    if args.ids:
        wanted = {identifier.strip() for identifier in args.ids.split(",") if identifier.strip()}
        questions = [item for item in questions if item.id in wanted]
    if args.limit:
        questions = questions[: args.limit]

    results_path = args.output_dir / "results.json"
    if args.resume and results_path.exists():
        run = EvalRun.model_validate_json(results_path.read_text(encoding="utf-8"))
        done = {result.id for result in run.questions if result.error is None and result.with_critic}
        questions = [item for item in questions if item.id not in done]
        logger.info("Resuming: %d question(s) already done, %d to go", len(done), len(questions))
    else:
        run = EvalRun(
            started_at=datetime.now(UTC),
            pipeline_model=settings.llm_model,
            judge_model=judge_settings.llm_model,
            settings=settings_snapshot(settings),
        )

    if not questions:
        logger.info("Nothing to evaluate.")
        save(run, args.output_dir)
        return 0
    if not args.yes and not confirm(len(questions), settings.llm_model, judge_settings.llm_model):
        logger.info("Cancelled (use --yes to skip this question).")
        return 1

    try:
        run = asyncio.run(run_all(questions, run, settings, judge_settings, args.output_dir))
    except LLMConfigurationError as error:
        logger.error("Stopping: %s", error)
        return 2
    json_path, markdown_path = save(run, args.output_dir)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.stdout.write(f"\n{render_markdown(run).split('## Latency')[0]}\nFull results: {markdown_path}\nRaw data: {json_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
