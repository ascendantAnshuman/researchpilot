"""End-to-end tests of the LangGraph pipeline and the CLI, fully offline (fake LLM and search)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agents.graph import main, run_research
from app.agents.state import PipelineContext
from app.config import Settings, get_settings
from app.services.vector_store import VectorStore
from tests.builders import FALSE_SENTENCE_START, make_context
from tests.fakes import FakeSearchClient

QUESTION = "What are recent methods for reducing hallucination in RAG systems?"


def stages_seen(events: list[dict[str, Any]]) -> list[str]:
    """Distinct stages in the order their first event arrived."""
    return list(dict.fromkeys(event["stage"] for event in events))


async def test_full_pipeline_catches_and_removes_the_false_claim(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """Critic flags the false claim, the Synthesizer revises once, and the final report is fully supported."""
    context = make_context(test_settings, vector_store, http_client)
    events: list[dict[str, Any]] = []

    report = await run_research(QUESTION, context=context, on_event=events.append)

    assert stages_seen(events) == ["planner", "search", "reader", "synthesizer", "critic", "finalize"]
    assert FALSE_SENTENCE_START not in report.markdown
    assert report.metrics.revisions == 1
    assert report.metrics.faithfulness == 1.0
    assert all(claim.label == "supported" for claim in report.verification)
    assert len(context.llm.calls_for("synthesizer")) == 2
    assert report.references and report.comparison_table
    assert report.metrics.papers_found == 3
    assert report.metrics.total_cost_usd > 0
    assert report.metrics.stages["critic"].llm_calls == 4  # 2 passes x (split + verify)
    assert report.metrics.total_seconds > 0
    assert all(event["timestamp"] for event in events)


async def test_ablation_without_critic_keeps_the_false_claim(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """ENABLE_CRITIC=false skips verification - and the false claim survives (the point of the ablation)."""
    settings = test_settings.model_copy(update={"enable_critic": False})
    context = make_context(settings, vector_store, http_client)
    events: list[dict[str, Any]] = []

    report = await run_research(QUESTION, context=context, on_event=events.append)

    assert "critic" not in stages_seen(events)
    assert FALSE_SENTENCE_START in report.markdown
    assert report.verification == []
    assert report.metrics.faithfulness is None
    assert report.metrics.critic_enabled is False


async def test_without_revisions_unsupported_claims_are_removed_by_policy(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """With MAX_REVISIONS=0 the false claim is removed at the end and listed as removed."""
    settings = test_settings.model_copy(update={"max_revisions": 0})

    report = await run_research(QUESTION, context=make_context(settings, vector_store, http_client))

    assert report.metrics.revisions == 0
    assert FALSE_SENTENCE_START not in report.markdown
    removed = [claim for claim in report.verification if claim.action == "removed"]
    assert len(removed) == 1 and removed[0].label == "unsupported"
    assert report.metrics.faithfulness is not None and report.metrics.faithfulness < 1.0


async def test_no_papers_found_ends_gracefully(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """Empty search results skip straight to a report that explains the problem."""
    context = make_context(test_settings, vector_store, http_client, arxiv=FakeSearchClient([]))
    events: list[dict[str, Any]] = []

    report = await run_research(QUESTION, context=context, on_event=events.append)

    assert stages_seen(events) == ["planner", "search", "finalize"]
    assert report.title == "No literature review could be produced"
    assert [call["stage"] for call in context.llm.calls] == ["planner"]


async def test_async_event_handlers_are_awaited(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """on_event may be a coroutine function (the SSE endpoint in Phase 5 uses one)."""
    received: list[str] = []

    async def handler(event: dict[str, Any]) -> None:
        received.append(event["stage"])

    await run_research(QUESTION, context=make_context(test_settings, vector_store, http_client), on_event=handler)

    assert "finalize" in received


async def test_short_questions_are_rejected(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """A question that is too short fails fast with a clear message."""
    with pytest.raises(ValueError, match="complete research question"):
        await run_research("RAG?", context=make_context(test_settings, vector_store, http_client))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_refuses_to_run_without_api_key(test_settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    """Without ANTHROPIC_API_KEY the CLI explains what to do and exits with code 2."""
    assert main([QUESTION]) == 2
    assert "ANTHROPIC_API_KEY is not set" in caplog.text


def test_cli_runs_pipeline_and_saves_markdown_and_json(
    test_settings: Settings,
    vector_store: VectorStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI runs the graph, writes both files and prints a summary."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    get_settings.cache_clear()

    @asynccontextmanager
    async def fake_context(settings: Settings) -> AsyncIterator[PipelineContext]:
        async with httpx.AsyncClient() as http:
            yield make_context(settings, vector_store, http)

    monkeypatch.setattr("app.agents.graph.open_pipeline_context", fake_context)
    output = tmp_path / "report.md"

    exit_code = main([QUESTION, "--top-n", "2", "--output", str(output)])

    assert exit_code == 0
    assert output.read_text(encoding="utf-8").startswith("# Review: hallucination in RAG")
    assert output.with_suffix(".json").exists()
    summary = capsys.readouterr().out
    assert "Faithfulness:   100%" in summary
    assert "Papers used:    2" in summary
