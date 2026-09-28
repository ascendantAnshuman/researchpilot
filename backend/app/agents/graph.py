"""LangGraph orchestration of the five agents, plus a command-line entry point.

Graph::

    START -> planner -> search --(no papers)--------------------------> finalize -> END
                          |                                                ^
                          v                                                |
                        reader --(no readable text)------------------------+
                          |                                                |
                          v                                                |
                     synthesizer --(ENABLE_CRITIC=false)-------------------+
                          |   ^                                            |
                          v   | (unsupported claims, one revision)        |
                        critic ---------------------------------------------+

Command line (from the ``backend/`` folder)::

    python -m app.agents.graph "What are recent methods for reducing hallucination in RAG systems?"
    python -m app.agents.graph "..." --no-critic --top-n 6

The report is saved as Markdown and JSON under ``DATA_DIR/runs/``.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import re
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.critic import critic_node, route_after_critic
from app.agents.finalize import finalize_node
from app.agents.planner import planner_node
from app.agents.reader import reader_node
from app.agents.search import search_node
from app.agents.state import PipelineContext, ResearchState, usable_digests
from app.agents.synthesizer import synthesizer_node
from app.config import Settings, get_settings
from app.llm import LLM, LLMClient, LLMConfigurationError, LLMError
from app.logging_config import setup_logging
from app.schemas import ResearchReport, StageMetrics
from app.services.arxiv_client import ArxivClient
from app.services.http_utils import create_http_client
from app.services.semantic_scholar_client import SemanticScholarClient
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
MIN_QUESTION_CHARS = 10


def _after_search(state: ResearchState) -> str:
    """Skip to the end if nothing was found."""
    return "reader" if state.get("papers") else "finalize"


def _after_reader(state: ResearchState) -> str:
    """Skip writing if no paper had any readable text."""
    return "synthesizer" if usable_digests(state.get("digests", [])) else "finalize"


def _after_synthesizer(state: ResearchState) -> str:
    """Verify the draft unless the Critic is switched off (ablation)."""
    return "critic" if state.get("critic_enabled", True) else "finalize"


def build_graph() -> CompiledStateGraph:
    """Wire the agents into a LangGraph state machine."""
    builder = StateGraph(ResearchState, context_schema=PipelineContext)
    builder.add_node("planner", planner_node)
    builder.add_node("search", search_node)
    builder.add_node("reader", reader_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "search")
    builder.add_conditional_edges("search", _after_search, ["reader", "finalize"])
    builder.add_conditional_edges("reader", _after_reader, ["synthesizer", "finalize"])
    builder.add_conditional_edges("synthesizer", _after_synthesizer, ["critic", "finalize"])
    builder.add_conditional_edges("critic", route_after_critic, ["synthesizer", "finalize"])
    builder.add_edge("finalize", END)
    return builder.compile()


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """The compiled graph (built once per process)."""
    return build_graph()


@asynccontextmanager
async def open_pipeline_context(settings: Settings, llm: LLM | None = None) -> AsyncIterator[PipelineContext]:
    """Create the real clients for one run and close them afterwards."""
    async with create_http_client(settings) as http:
        yield PipelineContext(
            settings=settings,
            llm=llm or LLMClient(settings),
            vector_store=get_vector_store(settings),
            arxiv=ArxivClient(settings, http_client=http),
            semantic_scholar=SemanticScholarClient(settings, http_client=http),
            http=http,
        )


def initial_state(question: str, settings: Settings) -> ResearchState:
    """The state a run starts from."""
    return {
        "question": question,
        "critic_enabled": settings.enable_critic,
        "max_revisions": settings.max_revisions,
        "revisions": 0,
        "warnings": [],
        "stage_seconds": {},
    }


async def _execute(
    question: str, context: PipelineContext, on_event: EventHandler | None, started: float
) -> ResearchReport:
    """Stream the graph, forwarding progress events, and return the final report."""
    initial = initial_state(question, context.settings)
    final_state: dict[str, Any] = {}
    async for mode, payload in get_graph().astream(initial, context=context, stream_mode=["custom", "values"]):
        if mode == "custom" and on_event is not None:
            outcome = on_event(payload)
            if inspect.isawaitable(outcome):
                await outcome
        elif mode == "values":
            final_state = payload

    report: ResearchReport | None = final_state.get("report")
    if report is None:
        raise RuntimeError("The pipeline finished without producing a report")
    for stage, seconds in final_state.get("stage_seconds", {}).items():
        report.metrics.stages.setdefault(stage, StageMetrics()).seconds = round(seconds, 3)
    report.metrics.total_seconds = round(time.perf_counter() - started, 3)
    return report


async def run_research(
    question: str,
    *,
    settings: Settings | None = None,
    context: PipelineContext | None = None,
    on_event: EventHandler | None = None,
) -> ResearchReport:
    """Run the full pipeline for one question.

    Args:
        question: The research question.
        settings: Settings to use (defaults to the global settings).
        context: Pre-built dependencies (tests inject fakes); created from ``settings`` if omitted.
        on_event: Called (sync or async) with every progress event as a JSON-ready dict.

    Raises:
        ValueError: If the question is too short.
        LLMError: If Claude cannot be used (bad key, persistent outage, ...).
    """
    settings = context.settings if context is not None else (settings or get_settings())
    question = " ".join(question.split())
    if len(question) < MIN_QUESTION_CHARS:
        raise ValueError(f"Please ask a complete research question (at least {MIN_QUESTION_CHARS} characters).")
    started = time.perf_counter()
    if context is not None:
        return await _execute(question, context, on_event, started)
    async with open_pipeline_context(settings) as fresh_context:
        return await _execute(question, fresh_context, on_event, started)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _slug(text: str, length: int = 50) -> str:
    """File-name friendly version of ``text``."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:length].rstrip("-") or "report"


def save_report(report: ResearchReport, settings: Settings, output: Path | None = None) -> tuple[Path, Path]:
    """Write the report as Markdown and JSON; return both paths."""
    if output is None:
        runs_dir = settings.data_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        output = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(report.question)}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.markdown, encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output, json_path


def summarise(report: ResearchReport, markdown_path: Path, json_path: Path) -> str:
    """A short, human-readable run summary for the terminal."""
    metrics = report.metrics
    faithfulness = f"{metrics.faithfulness:.0%}" if metrics.faithfulness is not None else "n/a (critic off)"
    lines = [
        "",
        f"  {report.title}",
        "",
        f"  Papers used:    {metrics.papers_found} ({metrics.papers_full_text} full text)",
        f"  Claims checked: {metrics.claims_total} "
        f"({metrics.claims_supported} supported / {metrics.claims_partial} partial / {metrics.claims_unsupported} unsupported)",
        f"  Faithfulness:   {faithfulness}   Revisions: {metrics.revisions}",
        f"  Time:           {metrics.total_seconds:.0f}s   "
        f"Tokens: {metrics.total_input_tokens:,} in / {metrics.total_output_tokens:,} out   "
        f"Cost: ${metrics.total_cost_usd:.3f} ({metrics.llm_model})",
        "",
        f"  Markdown: {markdown_path}",
        f"  JSON:     {json_path}",
    ]
    if report.warnings:
        lines += ["", f"  {len(report.warnings)} warning(s):"] + [f"   - {warning}" for warning in report.warnings[:8]]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        prog="python -m app.agents.graph",
        description="Run the ResearchPilot multi-agent pipeline for one research question.",
    )
    parser.add_argument("question", help="The research question, in quotes.")
    parser.add_argument("--no-critic", action="store_true", help="Skip the Critic agent (ablation).")
    parser.add_argument("--top-n", type=int, help="Number of papers to keep (default: TOP_N_PAPERS).")
    parser.add_argument("--output", type=Path, help="Where to write the Markdown report (JSON is written next to it).")
    parser.add_argument("--print", dest="print_report", action="store_true", help="Also print the full report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if args.no_critic:
        overrides["enable_critic"] = False
    if args.top_n:
        overrides["top_n_papers"] = args.top_n
    if overrides:
        settings = settings.model_copy(update=overrides)
    setup_logging(settings.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if not settings.llm_configured:
        logger.error("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project root (see .env.example).")
        return 2
    try:
        report = asyncio.run(run_research(args.question, settings=settings))
    except (LLMConfigurationError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except LLMError as exc:
        logger.error("The research run failed: %s", exc)
        return 1

    markdown_path, json_path = save_report(report, settings, args.output)
    if args.print_report:
        sys.stdout.write(report.markdown + "\n")
    sys.stdout.write(summarise(report, markdown_path, json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
