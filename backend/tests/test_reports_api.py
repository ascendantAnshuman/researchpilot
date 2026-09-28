"""API tests for the history list, report detail, deletion and export."""

from __future__ import annotations

import pymupdf
from fastapi.testclient import TestClient

from app.config import Settings
from app.services.vector_store import VectorStore
from tests.api_harness import QUESTION, Api, blocking_runner, build_api


def finished_run(api: Api, headers: dict[str, str], question: str = QUESTION) -> str:
    """Start a run and wait for it to finish."""
    session_id = api.start(headers, question=question)
    api.events(session_id, headers)
    return session_id


def test_history_lists_own_sessions_newest_first(api: Api) -> None:
    """The sidebar shows only the user's sessions, newest first."""
    headers = api.register()
    first = finished_run(api, headers, "First question about hallucination in RAG?")
    second = finished_run(api, headers, "Second question about retrieval augmented generation?")
    finished_run(api, api.register("someone@university.edu"))

    history = api.client.get("/api/reports", headers=headers).json()

    assert [item["id"] for item in history] == [second, first]
    assert history[0]["status"] == "completed"
    assert history[0]["metrics"]["faithfulness"] == 1.0


def test_report_detail_contains_everything_the_ui_needs(api: Api) -> None:
    """Report, papers, verification and metrics come back in one response."""
    headers = api.register()
    session_id = finished_run(api, headers)

    detail = api.client.get(f"/api/reports/{session_id}", headers=headers).json()

    report = detail["report"]
    assert report["title"] == "Review: hallucination in RAG"
    assert report["references"] and report["comparison_table"] and report["verification"]
    assert len(report["papers"]) == 3
    assert report["markdown"].startswith("# Review: hallucination in RAG")


def test_markdown_export(api: Api) -> None:
    """Markdown downloads with a sensible file name."""
    headers = api.register()
    session_id = finished_run(api, headers)

    response = api.client.get(f"/api/reports/{session_id}/export?format=markdown", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="review-hallucination-in-rag.md"'
    assert response.text.startswith("# Review: hallucination in RAG")


def test_pdf_export_contains_the_report(api: Api) -> None:
    """The PDF is valid and contains the title, verification and references."""
    headers = api.register()
    session_id = finished_run(api, headers)

    response = api.client.get(f"/api/reports/{session_id}/export?format=pdf", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    with pymupdf.open(stream=response.content, filetype="pdf") as document:
        text = " ".join(page.get_text() for page in document)
    assert "Review: hallucination in RAG" in text
    assert "Verification" in text
    assert "References" in text


def test_export_of_unfinished_run_is_a_conflict(keyed_settings: Settings, vector_store: VectorStore) -> None:
    """There is nothing to export until the report exists."""
    app, _ = build_api(keyed_settings, vector_store, runner=blocking_runner())
    with TestClient(app) as client:
        api = Api(client=client, app=app, chat_llm=None, store=vector_store)  # type: ignore[arg-type]
        headers = api.register()
        session_id = api.start(headers)

        response = client.get(f"/api/reports/{session_id}/export?format=pdf", headers=headers)
        delete = client.delete(f"/api/reports/{session_id}", headers=headers)

    assert response.status_code == 409
    assert delete.status_code == 409


def test_delete_removes_the_session(api: Api) -> None:
    """Deleting removes the session, its report and chat history."""
    headers = api.register()
    session_id = finished_run(api, headers)
    api.client.post(f"/api/chat/{session_id}", json={"message": "Which benchmark is used?"}, headers=headers)

    assert api.client.delete(f"/api/reports/{session_id}", headers=headers).status_code == 204

    assert api.client.get(f"/api/reports/{session_id}", headers=headers).status_code == 404
    assert api.client.get("/api/reports", headers=headers).json() == []


def test_reports_are_private(api: Api) -> None:
    """Other users get 404 for detail, export and delete."""
    owner = api.register()
    session_id = finished_run(api, owner)
    intruder = api.register("intruder@university.edu")

    assert api.client.get(f"/api/reports/{session_id}", headers=intruder).status_code == 404
    assert api.client.get(f"/api/reports/{session_id}/export", headers=intruder).status_code == 404
    assert api.client.delete(f"/api/reports/{session_id}", headers=intruder).status_code == 404
