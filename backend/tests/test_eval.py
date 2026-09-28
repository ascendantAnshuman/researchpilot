"""Tests for the evaluation module (eval/): paired ablation, judges, statistics and the CLI - all offline."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import run_eval  # noqa: E402
from eval.ablation import run_paired  # noqa: E402
from eval.judges import (  # noqa: E402
    CitationVerdict,
    CitationVerdictList,
    RelevanceVerdict,
    judge_citations,
    judge_faithfulness,
    judge_relevance,
    strip_citations,
)
from eval.metrics import ArmResult, EvalRun, QuestionResult, render_markdown, summarize  # noqa: E402

from app.agents.state import ClaimList, JudgementList, PipelineContext  # noqa: E402
from app.config import Settings  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402
from tests.builders import FALSE_SENTENCE_START, claims_handler, judgement_handler, make_context  # noqa: E402
from tests.fakes import FakeLLM, prompt_text  # noqa: E402

QUESTION = "What are recent methods for reducing hallucination in RAG systems?"
_PAIR_RE = re.compile(r"^\[(P\d+)\]\nSENTENCE: (.+)$", re.MULTILINE)


def citation_handler(call: dict[str, Any]) -> CitationVerdictList:
    """Fake judge: every citation supports its sentence except the planted false one."""
    verdicts = [
        CitationVerdict(
            pair_id=pair_id,
            label="does_not_support" if sentence.startswith(FALSE_SENTENCE_START) else "supports",
            rationale="fake",
        )
        for pair_id, sentence in _PAIR_RE.findall(prompt_text(call))
    ]
    return CitationVerdictList(verdicts=verdicts)


def judge_llm(relevance: int = 4) -> FakeLLM:
    """A fake judge answering every schema the evaluation uses."""
    return FakeLLM(
        {
            CitationVerdictList: citation_handler,
            RelevanceVerdict: lambda call: RelevanceVerdict(score=relevance, rationale="fake"),
            ClaimList: claims_handler,
            JudgementList: judgement_handler,
        }
    )


@pytest.fixture()
async def context(
    test_settings: Settings, vector_store: VectorStore, http_client: httpx.AsyncClient
) -> PipelineContext:
    """Offline pipeline context."""
    return make_context(test_settings, vector_store, http_client)


# ---------------------------------------------------------------------------
# Paired ablation
# ---------------------------------------------------------------------------


async def test_paired_run_isolates_the_critic(context: PipelineContext) -> None:
    """The without-Critic arm is the first draft; the with-Critic arm is the verified, revised report."""
    paired = await run_paired(QUESTION, context)

    assert paired.draft_available
    assert FALSE_SENTENCE_START in paired.without_critic.markdown
    assert FALSE_SENTENCE_START not in paired.with_critic.markdown
    assert paired.with_critic.metrics.revisions == 1
    assert paired.without_critic.metrics.critic_enabled is False
    assert paired.without_critic.verification == []
    # Same papers, and the ablation arm is cheaper and faster because it stops at the first draft.
    assert [d.paper.paper_id for d in paired.with_critic.papers] == [d.paper.paper_id for d in paired.without_critic.papers]
    assert paired.without_critic.metrics.total_cost_usd < paired.with_critic.metrics.total_cost_usd
    assert paired.without_critic.metrics.total_seconds <= paired.with_critic.metrics.total_seconds
    assert paired.without_critic.metrics.stages["critic"].llm_calls == 0


async def test_paired_run_without_papers(test_settings: Settings, vector_store: VectorStore,
                                         http_client: httpx.AsyncClient) -> None:
    """If no draft is written, both arms are the same 'no review' report."""
    from tests.fakes import FakeSearchClient

    context = make_context(test_settings, vector_store, http_client, arxiv=FakeSearchClient([]))

    paired = await run_paired(QUESTION, context)

    assert not paired.draft_available
    assert paired.with_critic.title == paired.without_critic.title == "No literature review could be produced"


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


def test_strip_citations() -> None:
    """Citation markers are removed before judging."""
    assert strip_citations("Grounding helps [1, 3]. Maybe not [unverified]") == "Grounding helps. Maybe not"


async def test_citation_accuracy_penalises_the_false_sentence(context: PipelineContext) -> None:
    """Without the Critic one citation is wrong; with it, all citations support their sentences."""
    paired = await run_paired(QUESTION, context)
    judge = judge_llm()

    without = await judge_citations(paired.without_critic, context.vector_store, judge)
    with_critic = await judge_citations(paired.with_critic, context.vector_store, judge)

    assert without.pairs > with_critic.pairs > 0
    assert without.unsupported == 1
    assert without.accuracy is not None and without.accuracy < 1.0
    assert with_critic.accuracy == 1.0
    assert all(call["stage"] == "judge_citations" for call in judge.calls)


async def test_faithfulness_reuses_the_critic_procedure(context: PipelineContext, test_settings: Settings) -> None:
    """The final text is rebuilt with [E#] labels, split into claims and verified."""
    paired = await run_paired(QUESTION, context)

    without = await judge_faithfulness(paired.without_critic, context.vector_store, judge_llm(), test_settings)
    with_critic = await judge_faithfulness(paired.with_critic, context.vector_store, judge_llm(), test_settings)

    assert without.claims == with_critic.claims + 1
    assert without.unsupported == 1
    assert with_critic.faithfulness == 1.0
    assert without.faithfulness is not None and without.faithfulness < with_critic.faithfulness


async def test_relevance_score_is_clamped(context: PipelineContext) -> None:
    """Out-of-range judge scores are clamped to 1..5, and the question is shown to the judge."""
    paired = await run_paired(QUESTION, context)
    judge = judge_llm(relevance=9)

    verdict = await judge_relevance(paired.with_critic, judge)

    assert verdict.score == 5
    assert QUESTION in judge.calls[0]["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Statistics and Markdown
# ---------------------------------------------------------------------------


def question_result(index: int, with_faith: float, without_faith: float) -> QuestionResult:
    """A synthetic result for aggregation tests."""
    return QuestionResult(
        id=f"q{index:02d}",
        topic="test",
        question=f"Question {index}?",
        with_critic=ArmResult(faithfulness=with_faith, citation_accuracy=0.9, relevance=4, latency_seconds=100,
                              cost_usd=0.5, revisions=1, claims_removed=2, papers=10, papers_full_text=8,
                              stage_seconds={"critic": 30.0}),
        without_critic=ArmResult(faithfulness=without_faith, citation_accuracy=0.8, relevance=4, latency_seconds=60,
                                 cost_usd=0.3, papers=10, papers_full_text=8),
        judge_cost_usd=0.1,
    )


def test_summary_statistics() -> None:
    """Means, paired deltas, bootstrap CIs and win/loss counts are computed per metric."""
    results = [question_result(1, 0.9, 0.7), question_result(2, 0.8, 0.8), question_result(3, 0.95, 0.75)]
    results.append(QuestionResult(id="q04", topic="t", question="Broken?", error="RuntimeError: boom"))

    summary = summarize(results)

    faithfulness = summary["metrics"]["faithfulness"]
    assert summary["questions_completed"] == 3 and summary["questions_failed"] == 1
    assert faithfulness["n"] == 3
    assert faithfulness["delta_mean"] == pytest.approx((0.2 + 0.0 + 0.2) / 3)
    low, high = faithfulness["delta_ci95"]
    assert low <= faithfulness["delta_mean"] <= high
    assert (faithfulness["critic_better"], faithfulness["critic_worse"], faithfulness["ties"]) == (2, 0, 1)
    # Lower is better for latency: the slower Critic arm counts as "worse".
    assert summary["metrics"]["latency_seconds"]["critic_worse"] == 3
    assert summary["stage_seconds"]["with_critic"]["critic"] == 30.0
    assert summary["total_judge_cost_usd"] == pytest.approx(0.3)


def test_markdown_summary_contains_the_ablation_table() -> None:
    """results.md has the main table, stage latencies, per-question rows and definitions."""
    run = EvalRun(started_at=datetime(2026, 9, 28, tzinfo=UTC), pipeline_model="claude-sonnet-5",
                  judge_model="claude-opus-5", questions=[question_result(1, 0.9, 0.7)])
    run.summary = summarize(run.questions)

    markdown = render_markdown(run)

    assert "## Main result: Critic ablation" in markdown
    assert "| Faithfulness (supported claims) | 90.0% | 70.0% | +20.0 pts" in markdown
    assert "| Critic | 30.0 | 0.0 |" in markdown
    assert "| q01 | test |" in markdown
    assert "## Threats to validity" in markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture()
def offline_eval(keyed_settings: Settings, vector_store: VectorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the eval runner at the offline pipeline and the fake judge."""

    @asynccontextmanager
    async def fake_context(settings: Settings, llm: Any = None) -> AsyncIterator[PipelineContext]:
        async with httpx.AsyncClient() as http:
            yield make_context(settings, vector_store, http)

    monkeypatch.setattr(run_eval, "open_pipeline_context", fake_context)
    monkeypatch.setattr(run_eval, "make_judge_llm", lambda settings: judge_llm())
    monkeypatch.setattr(run_eval, "make_pipeline_llm", lambda settings: None)


def test_cli_writes_results_and_resumes(offline_eval: None, tmp_path: Path) -> None:
    """The CLI evaluates questions, writes both files, and --resume skips finished questions."""
    exit_code = run_eval.main(["--limit", "2", "--yes", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [question["id"] for question in data["questions"]] == ["q01", "q02"]
    faithfulness = data["summary"]["metrics"]["faithfulness"]
    assert faithfulness["with_critic_mean"] == 1.0
    assert faithfulness["without_critic_mean"] < 1.0
    assert "Critic ablation" in (tmp_path / "results.md").read_text(encoding="utf-8")

    assert run_eval.main(["--limit", "2", "--yes", "--resume", "--output-dir", str(tmp_path)]) == 0
    assert len(json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["questions"]) == 2


def test_cli_selects_questions_by_id(offline_eval: None, tmp_path: Path) -> None:
    """--ids evaluates only the named questions."""
    assert run_eval.main(["--ids", "q05,q07", "--yes", "--output-dir", str(tmp_path)]) == 0

    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [question["id"] for question in data["questions"]] == ["q05", "q07"]


def test_cli_asks_before_spending_money(offline_eval: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --yes the user must confirm; answering no cancels without running anything."""
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    assert run_eval.main(["--limit", "1", "--output-dir", str(tmp_path)]) == 1
    assert not (tmp_path / "results.json").exists()


def test_cli_requires_api_key(test_settings: Settings, tmp_path: Path) -> None:
    """Without ANTHROPIC_API_KEY the evaluation refuses to start."""
    assert run_eval.main(["--limit", "1", "--yes", "--output-dir", str(tmp_path)]) == 2


def test_deliverables_use_the_measured_numbers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Résumé bullets and the results paragraph quote the real summary, honestly worded."""
    from eval import deliverables

    improved = [question_result(i, 0.92, 0.70) for i in range(1, 13)]
    run = EvalRun(started_at=datetime(2026, 9, 28, tzinfo=UTC), pipeline_model="claude-sonnet-5",
                  judge_model="claude-opus-5", questions=improved)
    run.summary = summarize(run.questions)
    results = tmp_path / "results.json"
    results.write_text(run.model_dump_json(), encoding="utf-8")

    assert deliverables.main(["--results", str(results), "--out", str(tmp_path / "mine.md")]) == 0

    text = capsys.readouterr().out
    assert "raised claim faithfulness from 70.0% to 92.0% (+22.0 pts" in text
    assert "over 12 research questions" in text
    assert "better on 12, worse on 0" in text
    assert (tmp_path / "mine.md").read_text(encoding="utf-8") == text


def test_deliverables_do_not_overclaim(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """If the Critic's effect is not significant, the text says so."""
    from eval import deliverables

    mixed = [question_result(1, 0.9, 0.8), question_result(2, 0.7, 0.8)]
    run = EvalRun(started_at=datetime(2026, 9, 28, tzinfo=UTC), pipeline_model="m", judge_model="m", questions=mixed)
    run.summary = summarize(run.questions)
    results = tmp_path / "results.json"
    results.write_text(run.model_dump_json(), encoding="utf-8")

    deliverables.main(["--results", str(results)])

    text = capsys.readouterr().out
    assert "changed claim faithfulness" in text and "not significant" in text
    assert "Run all 25 before quoting these numbers" in text


def test_deliverables_need_results(tmp_path: Path) -> None:
    """Without results.json there is nothing to report."""
    from eval import deliverables

    assert deliverables.main(["--results", str(tmp_path / "missing.json")]) == 1


def test_benchmark_has_25_unique_questions() -> None:
    """The benchmark file matches the project brief."""
    questions = run_eval.load_questions(PROJECT_ROOT / "eval" / "questions.json")

    assert len(questions) == 25
    assert len({question.id for question in questions}) == 25
    assert all(question.question.endswith("?") for question in questions)
