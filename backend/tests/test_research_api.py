"""API tests for starting research runs and streaming their progress over SSE (offline fake pipeline)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import Settings
from app.db import session_scope
from app.llm import LLMUnavailableError
from app.models import SessionPaper
from app.services.vector_store import VectorStore
from tests.api_harness import QUESTION, Api, blocking_runner, build_api, failing_runner


def test_starting_research_requires_login(api: Api) -> None:
    """Anonymous users cannot start runs."""
    assert api.client.post("/api/research", json={"question": QUESTION}).status_code == 401


def test_starting_research_without_api_key_is_refused(client: TestClient, auth_headers: dict[str, str]) -> None:
    """The server explains that Claude is not configured (HTTP 503)."""
    response = client.post("/api/research", json={"question": QUESTION}, headers=auth_headers)

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_question_is_validated(api: Api) -> None:
    """Too-short questions and out-of-range options are rejected with 422."""
    headers = api.register()

    assert api.client.post("/api/research", json={"question": "  RAG?  "}, headers=headers).status_code == 422
    too_many = {"question": QUESTION, "top_n": 99}
    assert api.client.post("/api/research", json=too_many, headers=headers).status_code == 422


def test_full_run_streams_every_stage_and_saves_the_report(api: Api) -> None:
    """POST starts a run; the SSE stream shows each stage live and ends with 'done'; results are stored."""
    headers = api.register()
    response = api.client.post("/api/research", json={"question": QUESTION}, headers=headers)
    assert response.status_code == 202
    body = response.json()
    session_id = body["session"]["id"]
    assert body["session"]["status"] == "pending"
    assert body["events_url"] == f"/api/research/{session_id}/events"

    events = api.events(session_id, headers)

    names = [name for name, _ in events]
    assert names[0] == "running"
    assert names[-1] == "done"
    stages = list(dict.fromkeys(data["stage"] for name, data in events if name == "progress"))
    assert stages == ["planner", "search", "reader", "synthesizer", "critic", "finalize"]
    assert events[-1][1]["faithfulness"] == 1.0

    status = api.client.get(f"/api/research/{session_id}", headers=headers).json()
    assert status["status"] == "completed"
    assert status["title"] == "Review: hallucination in RAG"
    assert status["metrics"]["revisions"] == 1
    with session_scope() as db:
        stored_papers = db.scalar(
            select(func.count()).select_from(SessionPaper).where(SessionPaper.session_id == session_id)
        )
    assert stored_papers == 3


def test_reconnecting_replays_the_whole_history(api: Api) -> None:
    """A second listener (e.g. after a page refresh) receives every event from the start."""
    headers = api.register()
    session_id = api.start(headers)
    first = api.events(session_id, headers)

    second = api.events(session_id, headers)

    assert [name for name, _ in second] == [name for name, _ in first]


def test_stream_of_a_finished_run_after_restart_comes_from_the_database(api: Api) -> None:
    """If the run is no longer in memory, the stream sends one final event built from the database."""
    headers = api.register()
    session_id = api.start(headers)
    api.events(session_id, headers)
    api.app.state.run_manager._runs.clear()  # simulate a server restart

    events = api.events(session_id, headers)

    assert len(events) == 1
    assert events[0][0] == "done"
    assert events[0][1]["title"] == "Review: hallucination in RAG"


def test_failed_run_reports_a_friendly_error(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """Pipeline errors end the stream with 'error' and are saved on the session."""
    app, _ = build_api(keyed_settings, vector_store, runner=failing_runner(LLMUnavailableError("429")))
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        headers = api.register()
        session_id = api.start(headers)

        events = api.events(session_id, headers)

        assert events[-1][0] == "error"
        assert "temporarily unavailable" in events[-1][1]["message"]
        status = client.get(f"/api/research/{session_id}", headers=headers).json()
        assert status["status"] == "failed"
        assert "temporarily unavailable" in status["error_message"]


def test_users_cannot_start_unlimited_runs(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """With MAX_ACTIVE_RUNS_PER_USER=2 the third simultaneous run gets 429."""
    app, _ = build_api(keyed_settings, vector_store, runner=blocking_runner())
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        headers = api.register()
        api.start(headers)
        api.start(headers)

        third = client.post("/api/research", json={"question": QUESTION}, headers=headers)

        assert third.status_code == 429
        other_user = api.register("other@university.edu")
        assert client.post("/api/research", json={"question": QUESTION}, headers=other_user).status_code == 202


def test_extra_runs_wait_in_a_queue(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """With MAX_CONCURRENT_RUNS=1 a second run announces that it is queued."""
    settings = keyed_settings.model_copy(update={"max_concurrent_runs": 1})
    app, _ = build_api(settings, vector_store, runner=blocking_runner())
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        api.start(api.register())
        second = api.start(api.register("second@university.edu"))

        deadline = time.monotonic() + 5
        names: list[str] = []
        while time.monotonic() < deadline and "queued" not in names:
            subscription = app.state.run_manager.subscribe(second)
            assert subscription is not None
            names = [item["event"] for item in subscription[0]]
            time.sleep(0.05)
        assert names == ["queued"]


def test_shutdown_marks_unfinished_runs_as_failed(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """Stopping the server cancels running pipelines and records why."""
    app, _ = build_api(keyed_settings, vector_store, runner=blocking_runner())
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        headers = api.register()
        session_id = api.start(headers)
        api.wait_for_status(session_id, headers, "running")

    with TestClient(app) as client:
        login = client.post("/api/auth/login", json={"email": "student@university.edu", "password": "CorrectHorse42"})
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        status = client.get(f"/api/research/{session_id}", headers=headers).json()
    assert status["status"] == "failed"
    assert "cancelled" in status["error_message"]


def test_sessions_are_private(api: Api) -> None:
    """Another user gets 404 for status and events of someone else's run."""
    owner = api.register()
    session_id = api.start(owner)
    api.events(session_id, owner)
    intruder = api.register("intruder@university.edu")

    assert api.client.get(f"/api/research/{session_id}", headers=intruder).status_code == 404
    assert api.client.get(f"/api/research/{session_id}/events", headers=intruder).status_code == 404


def test_per_run_options_are_applied(api: Api) -> None:
    """enable_critic=false and top_n are honoured and recorded with the session."""
    headers = api.register()
    session_id = api.start(headers, enable_critic=False, top_n=2)

    events = api.events(session_id, headers)

    assert "critic" not in {data.get("stage") for _, data in events}
    detail = api.client.get(f"/api/reports/{session_id}", headers=headers).json()
    assert detail["metrics"]["critic_enabled"] is False
    assert detail["metrics"]["papers_found"] == 2
    assert detail["report"]["metrics"]["faithfulness"] is None
