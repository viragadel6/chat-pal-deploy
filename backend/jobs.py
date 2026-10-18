from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from .agent import Agent
from .config import Settings
from .db import Supabase, eq, gt, is_null, utc_now
from .errors import AppError
from .sse import event_frame


class JobSink:
    def __init__(self, db: Supabase, job_id: str) -> None:
        self.db = db
        self.job_id = job_id
        self.content = ""
        self.reasoning = ""
        self.final_status = "failed"
        self._events = 0
        self._last_flush = 0.0

    async def send(self, event: dict[str, Any]) -> None:
        if event.get("type") == "delta":
            content = event.get("content")
            reasoning = event.get("reasoning_content")
            if isinstance(content, str):
                self.content += content
            if isinstance(reasoning, str):
                self.reasoning += reasoning
        if event.get("type") == "done":
            self.final_status = str(event.get("outcome", "failed"))
        await self.db.admin_insert("chat_job_events", {"job_id": self.job_id, "payload": event}, "minimal")
        self._events += 1
        now = time.monotonic()
        if event.get("type") != "delta" or now - self._last_flush >= 2:
            self._last_flush = now
            await self.flush()

    async def flush(self) -> None:
        await self.db.admin_update("chat_jobs", {"response_content": self.content, "response_reasoning": self.reasoning, "updated_at": utc_now()}, [eq("id", self.job_id)])


class JobService:
    def __init__(self, config: Settings, db: Supabase) -> None:
        self.config = config
        self.db = db
        self.tasks: set[asyncio.Task[Any]] = set()

    def _track(self, task: asyncio.Task[Any]) -> None:
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def dispatch(self, job_id: str, request_url: str) -> None:
        parsed = urlparse(request_url)
        if parsed.hostname in {"localhost", "127.0.0.1", None} or self.config.environment in {"development", "test"}:
            self._track(asyncio.create_task(self.execute(job_id)))
            return
        origin = f"{parsed.scheme}://{parsed.netloc}"
        try:
            await self.db.rpc("kick_chat_worker", {"_base_url": origin, "_job_id": job_id})
        except AppError:
            self._track(asyncio.create_task(self.execute(job_id)))

    async def revive(self, job_id: str, request_url: str) -> None:
        job = await self.db.admin_single("chat_jobs", "status,lease_expires_at,stop_requested", [eq("id", job_id)])
        if not job or job.get("stop_requested"):
            return
        expiry = self._parse_time(job.get("lease_expires_at"))
        stale = job.get("status") == "queued" or (job.get("status") == "running" and (expiry is None or expiry < datetime.now(timezone.utc)))
        if stale:
            await self.dispatch(job_id, request_url)

    def _parse_time(self, value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    async def _claim(self, job: dict[str, Any]) -> bool:
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self.config.worker_lease_seconds)
        filters = [eq("id", str(job["id"])), eq("status", str(job["status"]))]
        if job.get("lease_expires_at"):
            filters.append(eq("lease_expires_at", str(job["lease_expires_at"])))
        else:
            filters.append(is_null("lease_expires_at"))
        updated = await self.db.admin_update("chat_jobs", {"status": "running", "started_at": now.isoformat(), "updated_at": now.isoformat(), "lease_expires_at": expires.isoformat()}, filters, "id")
        return bool(updated)

    async def execute(self, job_id: str) -> dict[str, str]:
        job = await self.db.admin_single("chat_jobs", "id,history,status,lease_expires_at,stop_requested", [eq("id", job_id)])
        if not job:
            return {"status": "missing"}
        status = str(job.get("status", ""))
        if status in {"completed", "cancelled", "failed"}:
            return {"status": status}
        if job.get("stop_requested"):
            await self.db.admin_update("chat_jobs", {"status": "cancelled", "finished_at": utc_now(), "lease_expires_at": None, "updated_at": utc_now()}, [eq("id", job_id)])
            return {"status": "cancelled"}
        expiry = self._parse_time(job.get("lease_expires_at"))
        if status == "running" and expiry and expiry > datetime.now(timezone.utc):
            return {"status": "running"}
        if not await self._claim(job):
            return {"status": "running"}
        cancellation = asyncio.Event()
        sink = JobSink(self.db, job_id)
        final_status = "failed"
        final_error: str | None = "A háttérmunka hibával leállt."
        heartbeat = asyncio.create_task(self._heartbeat(job_id, cancellation))
        try:
            history = job.get("history")
            if not isinstance(history, list):
                raise AppError("JOB_PROTOCOL", "A munka előzménye érvénytelen.", 502)
            normalized = [{"role": item["role"], "content": item["content"]} for item in history if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str)]
            agent = Agent(self.config, self.db, job_id, normalized, sink, cancellation)
            outcome, reason, steps, calls = await agent.run()
            final_status = "completed" if outcome == "completed" else "cancelled" if outcome == "cancelled" else "failed"
            final_error = None if final_status != "failed" else reason
            await sink.send({"type": "terminal", "task_id": job_id, "outcome": outcome, "reason": reason, "steps": steps, "tool_calls": calls, "duration_ms": 0})
            await sink.send({"type": "done", "task_id": job_id, "outcome": outcome})
        except AppError as error:
            final_status = "cancelled" if error.code == "CANCELLED" else "failed"
            final_error = error.message
            if final_status == "failed":
                await sink.send({"type": "error", "task_id": job_id, "code": error.code, "message": error.message})
            await sink.send({"type": "terminal", "task_id": job_id, "outcome": "cancelled" if final_status == "cancelled" else "failed", "reason": error.message, "steps": 0, "tool_calls": 0, "duration_ms": 0})
            await sink.send({"type": "done", "task_id": job_id, "outcome": "cancelled" if final_status == "cancelled" else "failed"})
        except Exception:
            final_status = "failed"
            final_error = "A háttérmunka hibával leállt."
            await sink.send({"type": "error", "task_id": job_id, "code": "BACKGROUND_FAILURE", "message": final_error})
            await sink.send({"type": "terminal", "task_id": job_id, "outcome": "failed", "reason": final_error, "steps": 0, "tool_calls": 0, "duration_ms": 0})
            await sink.send({"type": "done", "task_id": job_id, "outcome": "failed"})
        finally:
            cancellation.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await sink.flush()
            await self.db.admin_update("chat_jobs", {"status": final_status, "response_content": sink.content, "response_reasoning": sink.reasoning, "error": final_error, "finished_at": utc_now(), "lease_expires_at": None, "updated_at": utc_now()}, [eq("id", job_id)])
        return {"status": final_status}

    async def _heartbeat(self, job_id: str, cancellation: asyncio.Event) -> None:
        try:
            while not cancellation.is_set():
                await asyncio.sleep(self.config.worker_heartbeat_seconds)
                rows = await self.db.admin_update("chat_jobs", {"lease_expires_at": (datetime.now(timezone.utc) + timedelta(seconds=self.config.worker_lease_seconds)).isoformat(), "updated_at": utc_now()}, [eq("id", job_id)], "stop_requested")
                if rows and rows[0].get("stop_requested"):
                    cancellation.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            cancellation.set()

    async def follow(self, job_id: str, after: int, request_url: str) -> AsyncIterator[bytes]:
        cursor = max(0, after)
        start = time.monotonic()
        last_heartbeat = 0.0
        last_revive = 0.0
        await self.revive(job_id, request_url)
        while time.monotonic() - start <= self.config.max_runtime_seconds:
            try:
                events = await self.db.admin_select("chat_job_events", "id,payload", [eq("job_id", job_id), gt("id", cursor)], order="id.asc", limit=200)
                terminal = False
                for item in events:
                    identifier = int(item["id"])
                    cursor = max(cursor, identifier)
                    payload = item.get("payload")
                    if not isinstance(payload, dict):
                        continue
                    yield event_frame(payload, identifier)
                    if payload.get("type") in {"done", "error"}:
                        terminal = True
                if terminal:
                    return
                if events:
                    continue
                job = await self.db.admin_single("chat_jobs", "status", [eq("id", job_id)])
                if not job or job.get("status") not in {"queued", "running"}:
                    remaining = await self.db.admin_select("chat_job_events", "id,payload", [eq("job_id", job_id), gt("id", cursor)], order="id.asc", limit=200)
                    for item in remaining:
                        cursor = max(cursor, int(item["id"]))
                        if isinstance(item.get("payload"), dict):
                            yield event_frame(item["payload"], int(item["id"]))
                    return
                now = time.monotonic()
                if now - last_revive >= 30:
                    last_revive = now
                    await self.revive(job_id, request_url)
                if now - last_heartbeat >= 15:
                    last_heartbeat = now
                    yield b": heartbeat\n\n"
                await asyncio.sleep(0.8)
            except asyncio.CancelledError:
                return
            except AppError as error:
                yield event_frame({"type": "error", "task_id": job_id, "code": error.code, "message": error.message})
                return
        yield event_frame({"type": "error", "task_id": job_id, "code": "RUNTIME_LIMIT", "message": "Lejárt a teljes feladathatáridő."})
