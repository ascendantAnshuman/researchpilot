"""Planner agent: splits the research question into 3-5 focused sub-queries.

Each sub-query has a natural-language form (sent to Semantic Scholar) and a
keyword form (sent to arXiv). The plan is post-processed in code so the rest of
the pipeline can always rely on 3-5 distinct sub-queries, even if Claude returns
too few, too many or duplicates. If Claude's answer is unusable, a simple
heuristic plan is used instead so the run can continue.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.agents.state import (
    PipelineRuntime,
    ResearchPlan,
    ResearchState,
    SubQuery,
    emit,
    stage_node,
)
from app.config import Effort
from app.llm import LLM, LLMOutputError, LLMRefusalError, lighter_effort
from app.services.arxiv_client import extract_search_terms

logger = logging.getLogger(__name__)

MIN_SUB_QUERIES = 3
MAX_SUB_QUERIES = 5
MAX_KEYWORDS = 5

PLANNER_SYSTEM_PROMPT = """You are the Planner in ResearchPilot, a multi-agent assistant that writes cited literature reviews from academic papers.

Break the user's research question into 3 to 5 complementary sub-queries that together cover:
- the main methods or approaches,
- the datasets and benchmarks used to evaluate them,
- reported results and comparisons,
- open problems and limitations.

For every sub-query give:
- "query": a natural-language search query of 5-12 words, suitable for an academic search engine.
- "keywords": 2-5 search terms for arXiv. Wrap multi-word technical terms in double quotes, e.g. "retrieval augmented generation".
- "rationale": one short sentence on which part of the question it covers.

Prefer established technical vocabulary over casual wording. Do not repeat the same query twice."""

_FALLBACK_SUFFIXES: tuple[str, ...] = ("", "benchmark datasets and evaluation", "limitations and open problems")


def keyword_query(sub_query: SubQuery) -> str:
    """The arXiv query for a sub-query: its keywords, or its text if it has none."""
    return " ".join(sub_query.keywords) if sub_query.keywords else sub_query.query


def _clean_keywords(keywords: Sequence[str], fallback_text: str) -> list[str]:
    """Trim, de-duplicate and cap keywords; derive them from the query text if none are left."""
    cleaned: list[str] = []
    for keyword in keywords:
        text = " ".join(keyword.split()).strip()
        if text and text.lower() not in {existing.lower() for existing in cleaned}:
            cleaned.append(text)
    return cleaned[:MAX_KEYWORDS] or extract_search_terms(fallback_text)[:MAX_KEYWORDS]


def heuristic_plan(question: str) -> ResearchPlan:
    """A no-LLM plan: the question itself plus dataset- and limitation-focused variants."""
    base = " ".join(question.split()).rstrip("?")
    sub_queries = [
        SubQuery(
            query=f"{base} {suffix}".strip(),
            keywords=_clean_keywords([], f"{base} {suffix}"),
            rationale="Fallback query derived directly from the question.",
        )
        for suffix in _FALLBACK_SUFFIXES
    ]
    return ResearchPlan(title=base[:120] or "Literature review", sub_queries=sub_queries)


def normalise_plan(plan: ResearchPlan, question: str) -> ResearchPlan:
    """Guarantee 3-5 distinct, non-empty sub-queries with usable keywords."""
    seen: set[str] = set()
    sub_queries: list[SubQuery] = []
    for sub_query in plan.sub_queries:
        text = " ".join(sub_query.query.split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        sub_queries.append(
            SubQuery(
                query=text,
                keywords=_clean_keywords(sub_query.keywords, text),
                rationale=sub_query.rationale.strip(),
            )
        )
        if len(sub_queries) == MAX_SUB_QUERIES:
            break

    for filler in heuristic_plan(question).sub_queries:
        if len(sub_queries) >= MIN_SUB_QUERIES:
            break
        if filler.query.lower() not in seen:
            seen.add(filler.query.lower())
            sub_queries.append(filler)

    title = " ".join(plan.title.split()) or heuristic_plan(question).title
    return ResearchPlan(title=title, sub_queries=sub_queries)


async def plan_research(question: str, llm: LLM, effort: Effort) -> tuple[ResearchPlan, list[str]]:
    """Ask Claude for a plan; fall back to a heuristic plan if the answer is unusable.

    Returns:
        The normalised plan and any warnings.
    """
    try:
        plan = await llm.structured(
            stage="planner",
            system=PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Research question:\n{question.strip()}"}],
            output_model=ResearchPlan,
            effort=lighter_effort(effort),
        )
    except (LLMOutputError, LLMRefusalError) as exc:
        logger.warning("Planner fell back to a heuristic plan: %s", exc)
        return heuristic_plan(question), [f"Planner used a fallback plan ({exc})"]
    return normalise_plan(plan, question), []


@stage_node("planner")
async def planner_node(state: ResearchState, runtime: PipelineRuntime) -> dict[str, object]:
    """LangGraph node: create the research plan."""
    plan, warnings = await plan_research(state["question"], runtime.context.llm, runtime.context.settings.llm_effort)
    emit(
        runtime,
        "planner",
        f"Planned {len(plan.sub_queries)} sub-queries",
        sub_queries=[sub_query.query for sub_query in plan.sub_queries],
    )
    return {"plan": plan, "warnings": warnings}
