"""API tests for follow-up chat (RAG over a session's papers, fake Claude)."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import LLMError, LLMUnavailableError
from app.services.vector_store import VectorStore
from tests.api_harness import Api, blocking_runner, build_api


def finished_run(api: Api, headers: dict[str, str]) -> str:
    """Start a run and wait for it to finish."""
    session_id = api.start(headers)
    api.events(session_id, headers)
    return session_id


def test_chat_answers_with_citations_matching_the_report(api: Api) -> None:
    """Answers cite [n] numbers that match the report's reference list."""
    headers = api.register()
    session_id = finished_run(api, headers)
    report = api.client.get(f"/api/reports/{session_id}", headers=headers).json()["report"]
    numbers_by_paper = {reference["paper_id"]: reference["number"] for reference in report["references"]}

    response = api.client.post(f"/api/chat/{session_id}", json={"message": "How is hallucination reduced?"},
                               headers=headers)

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert answer["role"] == "assistant"
    assert re.search(r"\[\d+\]", answer["content"])
    assert "[E" not in answer["content"]
    assert answer["citations"]
    for citation in answer["citations"]:
        if citation["paper_id"] in numbers_by_paper:
            assert citation["number"] == numbers_by_paper[citation["paper_id"]]
        assert citation["snippet"]
    assert response.json()["question"]["content"] == "How is hallucination reduced?"


def test_chat_history_is_stored_and_sent_back_to_claude(api: Api) -> None:
    """Earlier turns are saved and included as context for the next question."""
    headers = api.register()
    session_id = finished_run(api, headers)

    api.client.post(f"/api/chat/{session_id}", json={"message": "First question?"}, headers=headers)
    api.client.post(f"/api/chat/{session_id}", json={"message": "And a follow-up?"}, headers=headers)

    history = api.client.get(f"/api/chat/{session_id}", headers=headers).json()
    assert [(m["role"], m["content"][:15]) for m in history][:3] == [
        ("user", "First question?"),
        ("assistant", history[1]["content"][:15]),
        ("user", "And a follow-up"),
    ]
    second_call = api.chat_llm.calls_for("chat")[1]
    assert [message["role"] for message in second_call["messages"]] == ["user", "assistant", "user"]
    assert second_call["messages"][0]["content"] == "First question?"


def test_chat_retrieval_is_limited_to_the_session_papers(api: Api) -> None:
    """Passages shown to Claude only come from this session's papers."""
    headers = api.register()
    session_id = finished_run(api, headers)
    report = api.client.get(f"/api/reports/{session_id}", headers=headers).json()["report"]
    titles = {paper["paper"]["title"] for paper in report["papers"]}

    api.client.post(f"/api/chat/{session_id}", json={"message": "What benchmarks exist?"}, headers=headers)

    prompt = api.chat_llm.calls_for("chat")[0]["messages"][-1]["content"]
    shown_titles = set(re.findall(r"^\[E\d+\] (.+?) \(", prompt, re.MULTILINE))
    assert shown_titles and shown_titles <= titles


def test_chat_before_the_report_is_ready_is_a_conflict(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """Chat needs a finished report."""
    app, _ = build_api(keyed_settings, vector_store, runner=blocking_runner())
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        headers = api.register()
        session_id = api.start(headers)

        response = client.post(f"/api/chat/{session_id}", json={"message": "Anything?"}, headers=headers)

    assert response.status_code == 409


def test_chat_without_api_key_is_unavailable(client: TestClient, auth_headers: dict[str, str]) -> None:
    """Without ANTHROPIC_API_KEY the chat endpoint answers 503."""
    response = client.post("/api/chat/some-session", json={"message": "Hello?"}, headers=auth_headers)

    assert response.status_code == 503


def test_llm_failures_map_to_clear_http_errors(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """Outages give 503, other model errors 502, and nothing is stored."""
    outcomes = iter([LLMUnavailableError("overloaded"), LLMError("bad request")])
    app, _ = build_api(keyed_settings, vector_store, chat_handler=lambda call: next(outcomes))
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        headers = api.register()
        session_id = finished_run(api, headers)

        busy = client.post(f"/api/chat/{session_id}", json={"message": "Q1?"}, headers=headers)
        broken = client.post(f"/api/chat/{session_id}", json={"message": "Q2?"}, headers=headers)
        history = client.get(f"/api/chat/{session_id}", headers=headers).json()

    assert busy.status_code == 503
    assert broken.status_code == 502
    assert history == []


def test_chat_is_private_and_validated(api: Api) -> None:
    """Other users get 404; empty messages get 422."""
    owner = api.register()
    session_id = finished_run(api, owner)
    intruder = api.register("intruder@university.edu")

    assert api.client.post(f"/api/chat/{session_id}", json={"message": "Hi?"}, headers=intruder).status_code == 404
    assert api.client.get(f"/api/chat/{session_id}", headers=intruder).status_code == 404
    assert api.client.post(f"/api/chat/{session_id}", json={"message": "   "}, headers=owner).status_code == 422
