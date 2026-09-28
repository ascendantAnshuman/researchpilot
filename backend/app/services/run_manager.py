"""Runs research pipelines in the background and fans their events out to SSE listeners.

``POST /api/research`` returns immediately; the pipeline runs as an asyncio task
here. Every progress event is

* appended to the run's history (so a browser that connects late, or reconnects,
  first receives everything it missed), and
* pushed to each connected listener's queue.

A run ends with exactly one terminal message: ``done`` (report saved) or
``error`` (reason saved on the session). At most ``MAX_CONCURRENT_RUNS``
pipelines execute at once; extra runs wait in a queue and say so.
Finished runs stay in memory for a while so late listeners still get the history.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agents.graph import EventHandler, run_research
from app.config import Settings
from app.llm import LLMConfigurationError, LLMError, LLMRefusalError, LLMUnavailableError
from app.schemas import ResearchReport
from app.services import session_store

logger = logging.getLogger(__name__)

Runner = Callable[[str, Settings, EventHandler], Awaitable[ResearchReport]]
Message = dict[str, Any]

TERMINAL_EVENTS: frozenset[str] = frozenset({"done", "error"})
FINISHED_RUN_TTL_SECONDS = 15 * 60


async def default_runner(question: str, settings: Settings, on_event: EventHandler) -> ResearchReport:
    """Run the real pipeline."""
    return await run_research(question, settings=settings, on_event=on_event)


def user_facing_error(error: BaseException) -> str:
    """A short explanation of why a run failed, safe to show in the browser."""
    if isinstance(error, LLMConfigurationError):
        return f"Claude is not configured correctly: {error}"
    if isinstance(error, LLMUnavailableError):
        return "Claude is temporarily unavailable (rate limit or outage). Please try again in a few minutes."
    if isinstance(error, LLMRefusalError):
        return "Claude declined to answer this question."
    if isinstance(error, LLMError):
        return f"The language model returned an error: {error}"
    if isinstance(error, ValueError):
        return str(error)
    return f"Unexpected error ({type(error).__name__}). Check the server logs for details."


def message(event: str, **data: Any) -> Message:
    """Build an SSE message: an event name plus a JSON-ready payload."""
    return {"event": event, "data": {**data, "timestamp": datetime.now(UTC).isoformat()}}


@dataclass
class RunState:
    """In-memory state of one run."""

    session_id: str
    history: list[Message] = field(default_factory=list)
    listeners: set[asyncio.Queue[Message]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None
    finished: bool = False


class RunManager:
    """Starts pipeline runs and streams their progress to any number of listeners."""

    def __init__(self, runner: Runner = default_runner, max_concurrent_runs: int = 2) -> None:
        """Create a manager that executes ``runner`` for each research session."""
        self._runner = runner
        self._max_concurrent_runs = max_concurrent_runs
        self._runs: dict[str, RunState] = {}
        self._slots: asyncio.Semaphore | None = None

    def is_active(self, session_id: str) -> bool:
        """Whether the run is currently queued or executing."""
        state = self._runs.get(session_id)
        return state is not None and not state.finished

    def start(self, session_id: str, question: str, settings: Settings) -> None:
        """Launch a run in the background (must be called from within the event loop)."""
        state = RunState(session_id)
        self._runs[session_id] = state
        state.task = asyncio.create_task(self._execute(state, question, settings), name=f"research-{session_id}")

    def subscribe(self, session_id: str) -> tuple[list[Message], asyncio.Queue[Message]] | None:
        """History so far plus a queue of future messages, or None if the run is unknown."""
        state = self._runs.get(session_id)
        if state is None:
            return None
        queue: asyncio.Queue[Message] = asyncio.Queue()
        state.listeners.add(queue)
        return list(state.history), queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[Message]) -> None:
        """Stop sending messages to ``queue``."""
        state = self._runs.get(session_id)
        if state is not None:
            state.listeners.discard(queue)

    async def shutdown(self) -> None:
        """Cancel all unfinished runs (their sessions are marked failed)."""
        tasks = [state.task for state in self._runs.values() if state.task is not None and not state.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Cancelled %d running research task(s)", len(tasks))

    def _publish(self, state: RunState, item: Message) -> None:
        """Record a message and deliver it to every listener."""
        state.history.append(item)
        for queue in list(state.listeners):
            queue.put_nowait(item)

    async def _on_pipeline_event(self, state: RunState, event: dict[str, Any]) -> None:
        """Forward a pipeline progress event and track the current stage in the database."""
        self._publish(state, {"event": "progress", "data": event})
        if event.get("status") == "started" and event.get("stage"):
            await asyncio.to_thread(session_store.update_stage, state.session_id, str(event["stage"]))

    async def _execute(self, state: RunState, question: str, settings: Settings) -> None:
        """Run the pipeline for one session, persist the outcome and publish a terminal message."""
        if self._slots is None:
            self._slots = asyncio.Semaphore(self._max_concurrent_runs)
        try:
            if self._slots.locked():
                self._publish(state, message("queued", message="Waiting for another research run to finish"))
            async with self._slots:
                await asyncio.to_thread(session_store.mark_running, state.session_id)
                self._publish(state, message("running", message="Research started"))
                report = await self._runner(
                    question, settings, lambda event: self._on_pipeline_event(state, event)
                )
                await asyncio.to_thread(session_store.save_report, state.session_id, report)
            self._publish(
                state,
                message(
                    "done",
                    session_id=state.session_id,
                    status="completed",
                    title=report.title,
                    faithfulness=report.metrics.faithfulness,
                    cost_usd=report.metrics.total_cost_usd,
                ),
            )
            logger.info("Research run %s completed", state.session_id)
        except asyncio.CancelledError:
            await asyncio.to_thread(
                session_store.mark_failed, state.session_id, "The run was cancelled because the server stopped."
            )
            self._publish(state, message("error", session_id=state.session_id, message="The run was cancelled."))
            raise
        except Exception as error:  # Any failure must end the run cleanly for the user.
            logger.exception("Research run %s failed", state.session_id)
            reason = user_facing_error(error)
            await asyncio.to_thread(session_store.mark_failed, state.session_id, reason)
            self._publish(state, message("error", session_id=state.session_id, message=reason))
        finally:
            state.finished = True
            asyncio.get_running_loop().call_later(FINISHED_RUN_TTL_SECONDS, self._runs.pop, state.session_id, None)
