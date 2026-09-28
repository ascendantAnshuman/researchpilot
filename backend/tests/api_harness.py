"""Helpers for API tests: an app wired to the offline fake pipeline, plus SSE and auth helpers."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.graph import EventHandler, run_research
from app.config import Settings
from app.deps import get_chat_llm, get_store
from app.main import create_app
from app.schemas import ResearchReport
from app.services.run_manager import Runner
from app.services.vector_store import VectorStore
from tests.builders import make_context
from tests.fakes import FakeLLM, prompt_text

QUESTION = "What are recent methods for reducing hallucination in RAG systems?"
PASSWORD = "CorrectHorse42"
_EVIDENCE_LABEL_RE = re.compile(r"^\[(E\d+)\]", re.MULTILINE)


def chat_answer(call: dict[str, Any]) -> str:
    """Fake chat reply citing the first and last passage it was given."""
    last_message = call["messages"][-1]["content"]
    labels = _EVIDENCE_LABEL_RE.findall(last_message)
    if not labels:
        return "The stored papers do not cover this."
    return f"Grounding answers in retrieved passages lowers hallucination [{labels[0]}]. " \
           f"Benchmarks measure it with human labels [{labels[-1]}]."


def fake_pipeline(store: VectorStore) -> Runner:
    """A runner executing the real graph with fake Claude and fake search."""

    async def runner(question: str, settings: Settings, on_event: EventHandler) -> ResearchReport:
        async with httpx.AsyncClient() as http:
            return await run_research(question, context=make_context(settings, store, http), on_event=on_event)

    return runner


def blocking_runner() -> Runner:
    """A runner that never finishes on its own (until the app shuts down and cancels it)."""

    async def runner(question: str, settings: Settings, on_event: EventHandler) -> ResearchReport:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return runner


def failing_runner(error: Exception) -> Runner:
    """A runner that emits one event and then fails with ``error``."""

    async def runner(question: str, settings: Settings, on_event: EventHandler) -> ResearchReport:
        result = on_event({"stage": "planner", "status": "started", "message": "Planner started", "data": {}})
        if asyncio.iscoroutine(result):
            await result
        raise error

    return runner


@dataclass
class Api:
    """A running test app and its collaborators."""

    client: TestClient
    app: FastAPI
    chat_llm: FakeLLM
    store: VectorStore

    def register(self, email: str = "student@university.edu") -> dict[str, str]:
        """Create a user and return its Authorization header."""
        response = self.client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        assert response.status_code == 201, response.text
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    def start(self, headers: dict[str, str], question: str = QUESTION, **options: Any) -> str:
        """Start a research run and return its session id."""
        response = self.client.post("/api/research", json={"question": question, **options}, headers=headers)
        assert response.status_code == 202, response.text
        return str(response.json()["session"]["id"])

    def events(self, session_id: str, headers: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
        """Read the whole SSE stream of a run as (event, data) pairs."""
        return read_sse(self.client, f"/api/research/{session_id}/events", headers)

    def wait_for_status(self, session_id: str, headers: dict[str, str], wanted: str, timeout: float = 10.0) -> dict:
        """Poll the status endpoint until the run reaches ``wanted``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.client.get(f"/api/research/{session_id}", headers=headers).json()
            if body["status"] == wanted:
                return body
            time.sleep(0.05)
        raise AssertionError(f"session {session_id} never reached {wanted}")


def read_sse(client: TestClient, url: str, headers: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
    """Parse a Server-Sent Events response into (event, data) pairs (pings are ignored)."""
    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream("GET", url, headers=headers) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        event_name = "message"
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((event_name, json.loads(line.split(":", 1)[1])))
    return events


def build_api(settings: Settings, store: VectorStore, runner: Runner | None = None,
              chat_handler: Callable[[dict[str, Any]], Any] = chat_answer) -> tuple[FastAPI, FakeLLM]:
    """Create an app wired to fakes (pipeline, vector store and chat LLM)."""
    chat_llm = FakeLLM({}, text_handler=chat_handler)
    app = create_app(settings, runner=runner or fake_pipeline(store))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_chat_llm] = lambda: chat_llm
    return app, chat_llm


def evidence_labels_in(call: dict[str, Any]) -> list[str]:
    """Evidence labels shown to the fake LLM in a call."""
    return _EVIDENCE_LABEL_RE.findall(prompt_text(call))
