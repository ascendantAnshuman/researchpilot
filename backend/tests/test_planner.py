"""Tests for the Planner agent (fake LLM)."""

from __future__ import annotations

import httpx

from app.agents.planner import heuristic_plan, keyword_query, normalise_plan, plan_research, planner_node
from app.agents.state import ResearchPlan, SubQuery
from app.config import Settings
from app.llm import LLMOutputError
from app.services.vector_store import VectorStore
from tests.builders import PLAN, fake_runtime, make_context
from tests.fakes import FakeLLM

QUESTION = "What are recent methods for reducing hallucination in RAG systems?"


def sub_query(text: str, keywords: list[str] | None = None) -> SubQuery:
    """Build a sub-query."""
    return SubQuery(query=text, keywords=keywords if keywords is not None else ["rag"], rationale="r")


def test_normalise_plan_removes_duplicates_and_caps_at_five() -> None:
    """Case-insensitive duplicates are dropped and at most five sub-queries remain."""
    plan = ResearchPlan(
        title="T",
        sub_queries=[sub_query("Query one"), sub_query("query ONE")] + [sub_query(f"Query {n}") for n in range(2, 8)],
    )

    normalised = normalise_plan(plan, QUESTION)

    assert [q.query for q in normalised.sub_queries] == ["Query one", "Query 2", "Query 3", "Query 4", "Query 5"]


def test_normalise_plan_pads_to_at_least_three() -> None:
    """Too few sub-queries are topped up with question-based fallbacks."""
    normalised = normalise_plan(ResearchPlan(title="T", sub_queries=[sub_query("dense retrieval")]), QUESTION)

    assert len(normalised.sub_queries) == 3
    assert normalised.sub_queries[0].query == "dense retrieval"
    assert "hallucination" in normalised.sub_queries[1].query.lower()


def test_missing_keywords_are_derived_from_the_query() -> None:
    """A sub-query without keywords still gets usable arXiv terms."""
    normalised = normalise_plan(
        ResearchPlan(title="T", sub_queries=[sub_query("hallucination in retrieval-augmented generation", [])]), QUESTION
    )

    assert normalised.sub_queries[0].keywords == ["hallucination", '"retrieval augmented"', "generation"]


def test_keywords_are_trimmed_deduplicated_and_capped() -> None:
    """At most five distinct keywords are kept."""
    keywords = [" rag ", "RAG", "a", "b", "c", "d", "e"]
    normalised = normalise_plan(ResearchPlan(title="T", sub_queries=[sub_query("q", keywords)]), QUESTION)

    assert normalised.sub_queries[0].keywords == ["rag", "a", "b", "c", "d"]


def test_keyword_query_prefers_keywords() -> None:
    """arXiv gets the keywords; without keywords it falls back to the query text."""
    assert keyword_query(sub_query("text", ['"dense retrieval"', "rag"])) == '"dense retrieval" rag'
    assert keyword_query(sub_query("plain text query", [])) == "plain text query"


def test_heuristic_plan_has_three_distinct_queries() -> None:
    """The no-LLM fallback plan is usable on its own."""
    plan = heuristic_plan(QUESTION)

    assert len({q.query for q in plan.sub_queries}) == 3
    assert all(q.keywords for q in plan.sub_queries)


async def test_plan_research_uses_lighter_effort(test_settings: Settings) -> None:
    """Planning is simple, so it runs at most at medium effort."""
    llm = FakeLLM({ResearchPlan: lambda call: PLAN})

    plan, warnings = await plan_research(QUESTION, llm, "high")

    assert plan.sub_queries == PLAN.sub_queries
    assert warnings == []
    assert llm.calls[0]["effort"] == "medium"
    assert QUESTION in llm.calls[0]["messages"][0]["content"]


async def test_plan_research_falls_back_when_output_is_unusable() -> None:
    """A malformed answer does not stop the run."""
    llm = FakeLLM({ResearchPlan: lambda call: LLMOutputError("bad json")})

    plan, warnings = await plan_research(QUESTION, llm, "high")

    assert len(plan.sub_queries) == 3
    assert "fallback" in warnings[0]


async def test_planner_node_returns_plan_and_streams_events(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> None:
    """The node stores the plan, records its duration and emits started/progress/completed."""
    runtime, events = fake_runtime(make_context(test_settings, vector_store, http_client))

    update = await planner_node({"question": QUESTION}, runtime)

    assert update["plan"].title == PLAN.title
    assert "planner" in update["stage_seconds"]
    assert [event["status"] for event in events] == ["started", "progress", "completed"]
    assert events[1]["data"]["sub_queries"] == [q.query for q in PLAN.sub_queries]
