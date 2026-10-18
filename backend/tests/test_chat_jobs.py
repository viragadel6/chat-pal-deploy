from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.services.chat_jobs import execute_chat_job, follow_chat_job
from tests.local_stack import ModelPlan

JOB_ID = "33333333-3333-4333-8333-333333333333"
CONVERSATION_ID = "44444444-4444-4444-8444-444444444444"


def seed_job(database: Any, **overrides: Any) -> dict[str, Any]:
    row = {
        "id": JOB_ID,
        "conversation_id": CONVERSATION_ID,
        "history": [{"role": "user", "content": "Írj egy verset."}],
        "status": "queued",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    row.update(overrides)
    database.store.table("chat_jobs").append(row)
    return row


async def test_execute_chat_job_runs_agent_and_stores_events(agent_stack):
    seed_job(agent_stack.database)
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "A vers elkészült."}}]],
            final_content="Elkészült a vers.",
        )
    )

    result = await execute_chat_job(JOB_ID)

    assert result == {"status": "completed"}
    job = agent_stack.database.store.table("chat_jobs")[0]
    assert job["status"] == "completed"
    assert job["response_content"] == "Elkészült a vers."
    assert job["lease_expires_at"] is None
    payloads = [event["payload"] for event in agent_stack.database.store.table("chat_job_events")]
    kinds = [payload.get("type") for payload in payloads]
    assert "terminal" in kinds
    assert "done" in kinds
    assert kinds[-1] == "done"
    terminal = next(payload for payload in payloads if payload.get("type") == "terminal")
    assert terminal["outcome"] == "completed"


async def test_follow_chat_job_replays_stored_events(agent_stack):
    seed_job(agent_stack.database)
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "Kész."}}]],
            final_content="Kész.",
        )
    )
    await execute_chat_job(JOB_ID)

    frames: list[str] = []
    async for frame in follow_chat_job(JOB_ID, 0, "http://127.0.0.1/chat"):
        frames.append(frame)

    assert frames
    assert all("data: " in frame for frame in frames)
    assert any('"type": "terminal"' in frame for frame in frames)
    assert '"type": "done"' in frames[-1]
    assert not any('"type": "done"' in frame for frame in frames[:-1])


async def test_follow_chat_job_resumes_after_event_cursor(agent_stack):
    seed_job(agent_stack.database)
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "Kész."}}]],
            final_content="Kész.",
        )
    )
    await execute_chat_job(JOB_ID)
    events = agent_stack.database.store.table("chat_job_events")
    cursor = int(events[0]["id"])

    frames: list[str] = []
    async for frame in follow_chat_job(JOB_ID, cursor, "http://127.0.0.1/chat"):
        frames.append(frame)

    assert frames
    identifiers = [int(frame.split("id: ", 1)[1].split("\n", 1)[0]) for frame in frames]
    assert all(identifier > cursor for identifier in identifiers)
    assert identifiers == sorted(identifiers)


async def test_execute_chat_job_cancels_stop_requested_job(agent_stack):
    seed_job(agent_stack.database, stop_requested=True)
    agent_stack.use_plan(ModelPlan(final_content="Nem fut le."))

    result = await execute_chat_job(JOB_ID)

    assert result == {"status": "cancelled"}
    job = agent_stack.database.store.table("chat_jobs")[0]
    assert job["status"] == "cancelled"
    assert not agent_stack.model_api.agent_requests()


async def test_execute_chat_job_skips_missing_job(agent_stack):
    result = await execute_chat_job("55555555-5555-4555-8555-555555555555")
    assert result == {"status": "missing"}


async def test_execute_chat_job_skips_finished_job(agent_stack):
    seed_job(agent_stack.database, status="completed")
    agent_stack.use_plan(ModelPlan(final_content="Nem fut le."))

    result = await execute_chat_job(JOB_ID)

    assert result == {"status": "completed"}
    assert not agent_stack.model_api.agent_requests()


async def test_execute_chat_job_respects_active_lease(agent_stack):
    from datetime import UTC, datetime, timedelta

    lease = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    seed_job(agent_stack.database, status="running", lease_expires_at=lease)
    agent_stack.use_plan(ModelPlan(final_content="Nem fut le."))

    result = await execute_chat_job(JOB_ID)

    assert result == {"status": "running"}
    assert not agent_stack.model_api.agent_requests()


async def test_execute_chat_job_claims_queued_job_once(agent_stack):
    seed_job(agent_stack.database)
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "Kész."}}]],
            final_content="Kész.",
        )
    )

    first = await execute_chat_job(JOB_ID)
    second = await execute_chat_job(JOB_ID)

    assert first == {"status": "completed"}
    assert second == {"status": "completed"}


async def test_execute_chat_job_reports_model_failure(agent_stack):
    from tests.local_stack import ModelReply

    seed_job(agent_stack.database)

    def policy(body: dict[str, Any]) -> ModelReply:
        return ModelReply(status_code=401, error={"error": "nope"})

    agent_stack.model_api.policy = policy

    result = await execute_chat_job(JOB_ID)

    assert result == {"status": "failed"}
    job = agent_stack.database.store.table("chat_jobs")[0]
    assert job["status"] == "failed"
    assert job["error"] == "A háttérmunka hibával leállt."
    payloads = [event["payload"] for event in agent_stack.database.store.table("chat_job_events")]
    assert any(payload.get("type") == "error" for payload in payloads)


@pytest.mark.asyncio
async def test_follow_chat_job_stops_on_terminal_event(agent_stack):
    seed_job(agent_stack.database)
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "Kész."}}]],
            final_content="Kész.",
        )
    )
    await execute_chat_job(JOB_ID)

    frames: list[str] = []
    async def consume() -> None:
        async for frame in follow_chat_job(JOB_ID, 0, "http://127.0.0.1/chat"):
            frames.append(frame)

    await asyncio.wait_for(consume(), timeout=20)

    assert frames
    assert '"type": "done"' in frames[-1]
