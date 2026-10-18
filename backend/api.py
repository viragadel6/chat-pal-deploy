from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import ValidationError

from .cdp import browser_command
from .config import Settings
from .db import Supabase, eq, gt, in_values, utc_now
from .errors import AppError
from .jobs import JobService
from .models import BrowserKeyInput, BrowserMouseInput, BrowserTextInput, BrowserWheelInput, ChatRequest
from .sse import event_frame

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")
MAX_BODY = 1048576
NO_STORE = {"Cache-Control": "private, no-store"}


class RateLimiter:
    def __init__(self) -> None:
        self.buckets: dict[str, tuple[int, float]] = {}
        self.lock = asyncio.Lock()

    async def check(self, name: str, key: str, limit: int, window: float) -> int | None:
        now = time.monotonic()
        async with self.lock:
            count, reset = self.buckets.get(f"{name}:{key}", (0, now + window))
            if reset <= now:
                count, reset = 0, now + window
            if count >= limit:
                return max(1, int(reset - now + 0.999))
            self.buckets[f"{name}:{key}"] = (count + 1, reset)
            if len(self.buckets) > 50000:
                self.buckets = {item: value for item, value in self.buckets.items() if value[1] > now}
        return None


class BrowserState:
    def __init__(self) -> None:
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.sequences: dict[str, int] = {}

    def lock(self, session_id: str) -> asyncio.Lock:
        return self.locks[session_id]

    def sequence(self, session_id: str, requested: int | None) -> int | None:
        previous = self.sequences.get(session_id, -1)
        value = previous + 1 if requested is None else requested
        if value <= previous:
            return None
        self.sequences[session_id] = value
        if len(self.sequences) > 10000:
            self.sequences.pop(next(iter(self.sequences)))
        return value


def json_response(data: Any, status: int = 200, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers={**NO_STORE, **(headers or {})})


def error_response(error: AppError) -> JSONResponse:
    return json_response({"error": {"code": error.code, "message": error.message}}, 400 if error.status_code == 499 else error.status_code)


def browser_error(message: str, status: int = 400, headers: dict[str, str] | None = None) -> JSONResponse:
    return json_response({"error": message}, status, headers)


async def read_json(request: Request, limit: int = MAX_BODY) -> Any:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("application/json"):
        raise AppError("INVALID_REQUEST", "application/json szükséges.", 415)
    declared = request.headers.get("content-length")
    if declared and (not declared.isdigit() or int(declared) > limit):
        raise AppError("REQUEST_TOO_LARGE", "A kérés túl nagy.", 413)
    body = await request.body()
    if len(body) > limit:
        raise AppError("REQUEST_TOO_LARGE", "A kérés túl nagy.", 413)
    try:
        return json.loads(body.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AppError("INVALID_REQUEST", "Hibás JSON vagy UTF-8 kérés.", 400) from error


def authorization_token(request: Request) -> str:
    value = request.headers.get("authorization", "")
    match = re.fullmatch(r"Bearer\s+([A-Za-z0-9._~-]+)", value.strip())
    if not match:
        raise AppError("UNAUTHORIZED", "Bejelentkezés szükséges.", 401)
    return match.group(1)


def request_origin(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-proto")
    scheme = forwarded if forwarded in {"http", "https"} else request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}"


def verify_origin(request: Request, config: Settings) -> None:
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site and fetch_site not in {"same-origin", "none"}:
        raise AppError("ORIGIN_DENIED", "Érvénytelen kérés eredete.", 403)
    allowed = set(config.origins)
    allowed.add(request_origin(request))
    candidate = request.headers.get("origin")
    if candidate:
        if candidate not in allowed:
            raise AppError("ORIGIN_DENIED", "Érvénytelen kérés eredete.", 403)
        return
    referer = request.headers.get("referer")
    if not referer:
        raise AppError("ORIGIN_DENIED", "Érvénytelen kérés eredete.", 403)
    try:
        origin = f"{urlparse(referer).scheme}://{urlparse(referer).netloc}"
    except Exception as error:
        raise AppError("ORIGIN_DENIED", "Érvénytelen kérés eredete.", 403) from error
    if origin not in allowed:
        raise AppError("ORIGIN_DENIED", "Érvénytelen kérés eredete.", 403)


def browser_input(value: Any) -> dict[str, Any]:
    try:
        if not isinstance(value, dict):
            raise ValueError("input")
        kind = value.get("kind")
        if kind == "mouse":
            parsed = BrowserMouseInput.model_validate(value)
            if not (0 <= parsed.x < 10000 and 0 <= parsed.y < 10000) or (parsed.type != "mouseMoved" and (parsed.button == "none" or parsed.clickCount < 1)):
                raise ValueError("mouse")
            return parsed.model_dump()
        if kind == "wheel":
            parsed = BrowserWheelInput.model_validate(value)
            if not (0 <= parsed.x < 10000 and 0 <= parsed.y < 10000):
                raise ValueError("wheel")
            return parsed.model_dump()
        if kind == "key":
            return BrowserKeyInput.model_validate(value).model_dump()
        if kind == "text":
            return BrowserTextInput.model_validate(value).model_dump()
    except ValidationError as error:
        raise AppError("INVALID_INPUT", "Érvénytelen bevitel.", 400) from error
    raise AppError("INVALID_INPUT", "Érvénytelen bevitel.", 400)


def viewport_valid(input_data: dict[str, Any], width: int, height: int) -> bool:
    if input_data["kind"] in {"mouse", "wheel"}:
        return 0 <= input_data["x"] < width and 0 <= input_data["y"] < height
    return True


def uuid_valid(value: str | None) -> bool:
    return isinstance(value, str) and bool(UUID_RE.fullmatch(value))


def app(config: Settings) -> FastAPI:
    instance = FastAPI(title="Chat Pal Backend", docs_url=None, redoc_url=None, openapi_url=None)
    database = Supabase(config)
    jobs = JobService(config, database)
    limiter = RateLimiter()
    browser_state = BrowserState()
    instance.state.config = config
    instance.state.db = database
    instance.state.jobs = jobs

    @instance.exception_handler(AppError)
    async def app_error_handler(_: Request, error: AppError) -> JSONResponse:
        return error_response(error)

    @instance.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return json_response({"error": {"code": "INVALID_REQUEST", "message": "Hibás kérés."}}, 400)

    @instance.on_event("shutdown")
    async def shutdown() -> None:
        for task in list(jobs.tasks):
            task.cancel()
        if jobs.tasks:
            await asyncio.gather(*jobs.tasks, return_exceptions=True)
        await database.close()

    @instance.get("/healthz")
    async def healthz() -> Response:
        return Response(status_code=204)

    @instance.post("/api/chat")
    async def create_chat(request: Request) -> Response:
        try:
            raw = await read_json(request)
            payload = ChatRequest.model_validate(raw)
            if any(not item.content.strip() for item in payload.messages):
                raise AppError("INVALID_REQUEST", "Az üzenet tartalma hibás vagy túl nagy.", 400)
            if payload.messages[-1].role != "user":
                raise AppError("INVALID_REQUEST", "A legutolsó üzenetnek felhasználói feladatnak kell lennie.", 400)
            conversation_id = payload.conversation_id if uuid_valid(payload.conversation_id) else str(uuid.uuid4())
            job_id = str(uuid.uuid4())
            messages = [item.model_dump() for item in payload.messages]
            first_user = next((item.content for item in payload.messages if item.role == "user"), "Beszélgetés")
            title = re.sub(r"\s+", " ", first_user).strip()[:80] or "Beszélgetés"
            now = utc_now()
            await database.admin_upsert("chat_conversations", {"id": conversation_id, "title": title, "messages": messages, "updated_at": now}, "id", "minimal")
            await database.admin_insert("chat_jobs", {"id": job_id, "conversation_id": conversation_id, "history": messages, "status": "queued", "created_at": now, "updated_at": now}, "minimal")
            await jobs.dispatch(job_id, str(request.url))
            async def stream() -> AsyncIterator[bytes]:
                yield event_frame({"type": "meta", "conversation_id": conversation_id, "job_id": job_id})
                async for chunk in jobs.follow(job_id, 0, str(request.url)):
                    yield chunk
            return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "private, no-store, no-cache, no-transform", "X-Accel-Buffering": "no", "X-Chat-Job-Id": job_id, "Connection": "keep-alive"})
        except ValidationError:
            return json_response({"error": {"code": "INVALID_REQUEST", "message": "Hiányzó vagy hibás üzenetlista."}}, 400)
        except AppError as error:
            return error_response(error)

    @instance.get("/api/chat/jobs/{job_id}/events")
    async def chat_events(job_id: str, request: Request, after: int = 0) -> Response:
        if not uuid_valid(job_id):
            return browser_error("Érvénytelen munkaazonosító.", 400)
        async def stream() -> AsyncIterator[bytes]:
            async for chunk in jobs.follow(job_id, max(0, after), str(request.url)):
                yield chunk
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "private, no-store, no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    @instance.post("/api/chat/jobs/{job_id}/stop")
    async def stop_chat(job_id: str) -> JSONResponse:
        if not uuid_valid(job_id):
            return browser_error("Érvénytelen munkaazonosító.", 400)
        job = await database.admin_single("chat_jobs", "id,status", [eq("id", job_id)])
        if not job:
            return browser_error("A munka nem található.", 404)
        if job["status"] == "queued":
            await database.admin_update("chat_jobs", {"status": "cancelled", "stop_requested": True, "finished_at": utc_now(), "updated_at": utc_now()}, [eq("id", job_id)])
        elif job["status"] == "running":
            await database.admin_update("chat_jobs", {"stop_requested": True, "updated_at": utc_now()}, [eq("id", job_id)])
        return json_response({"ok": True})

    @instance.post("/api/chat/jobs/{job_id}/redirect")
    async def redirect_chat(job_id: str, request: Request) -> JSONResponse:
        try:
            if not uuid_valid(job_id):
                return browser_error("Érvénytelen munkaazonosító.", 400)
            raw = await read_json(request, 65536)
            message = raw.get("message", "") if isinstance(raw, dict) else ""
            if not isinstance(message, str) or not message.strip():
                return browser_error("Üres üzenet nem küldhető.", 400)
            if len(message) > 32000:
                return browser_error("Az üzenet túl hosszú.", 400)
            job = await database.admin_single("chat_jobs", "id,status", [eq("id", job_id)])
            if not job:
                return browser_error("A munka nem található.", 404)
            if job["status"] not in {"queued", "running"}:
                return browser_error("A munka már lezárult, új irányt nem kaphat.", 409)
            await database.admin_insert("agent_operator_messages", {"task_id": job_id, "content": message.strip(), "consumed": False}, "minimal")
            return json_response({"ok": True})
        except AppError as error:
            return error_response(error)

    @instance.post("/api/public/chat-worker")
    async def public_worker(request: Request) -> JSONResponse:
        try:
            raw = await read_json(request, 16384)
            job_id = raw.get("job_id") if isinstance(raw, dict) else None
            if not uuid_valid(job_id):
                return browser_error("Érvénytelen munkaazonosító.", 400)
            result = await jobs.execute(job_id)
            return json_response(result)
        except AppError as error:
            return error_response(error)

    @instance.get("/api/conversations")
    async def conversations(ids: str = "") -> JSONResponse:
        values = [item.strip() for item in ids.split(",") if uuid_valid(item.strip())][:100]
        if not values:
            return json_response({"conversations": []})
        conversations_rows = await database.admin_select("chat_conversations", "id,title,messages,created_at,updated_at", [in_values("id", values)])
        jobs_rows = await database.admin_select("chat_jobs", "id,conversation_id,status,response_content,response_reasoning,error,created_at,updated_at", [in_values("conversation_id", values)], order="created_at.desc", limit=1000)
        latest: dict[str, dict[str, Any]] = {}
        for item in jobs_rows:
            latest.setdefault(str(item["conversation_id"]), item)
        event_ids: dict[str, int] = {}
        if jobs_rows:
            events = await database.admin_select("chat_job_events", "job_id,id", [in_values("job_id", [str(item["id"]) for item in jobs_rows])], order="id.desc", limit=1000)
            for item in events:
                event_ids.setdefault(str(item["job_id"]), int(item["id"]))
        result = []
        for conversation in conversations_rows:
            job = latest.get(str(conversation["id"]))
            result.append({"conversation_id": conversation["id"], "title": conversation["title"], "messages": conversation["messages"], "created_at": conversation["created_at"], "updated_at": conversation["updated_at"], "latest_job": {"job_id": job["id"], "conversation_id": job["conversation_id"], "status": job["status"], "response_content": job["response_content"], "response_reasoning": job["response_reasoning"], "error": job["error"], "last_event_id": event_ids.get(str(job["id"]), 0)} if job else None})
        return json_response({"conversations": result})

    @instance.post("/api/operator")
    async def operator(request: Request) -> JSONResponse:
        try:
            raw = await read_json(request, 16384)
            task_id = raw.get("task_id") if isinstance(raw, dict) else None
            content = raw.get("content") if isinstance(raw, dict) else None
            if not uuid_valid(task_id):
                return browser_error("Hibás task_id.", 400)
            if not isinstance(content, str) or not content.strip():
                return browser_error("Üres üzenet.", 400)
            await database.admin_insert("agent_operator_messages", {"task_id": task_id, "content": content.strip(), "consumed": False}, "minimal")
            return json_response({"accepted": True})
        except AppError as error:
            return error_response(error)

    @instance.post("/api/images/edits")
    async def image_edit(request: Request, image: UploadFile = File(...), prompt: str = "", quality: str = "medium", size: str = "1024x1024") -> Response:
        key = config.secret("flymyai_api_key")
        if not key:
            return json_response({"message": "A FlyMyAI kulcs nincs beállítva."}, 503)
        if not image.filename or not prompt.strip():
            return json_response({"message": "Adj meg egy kiinduló képet és képleírást."}, 400)
        content = await image.read()
        if not content:
            return json_response({"message": "A feltöltött kép üres."}, 400)
        client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=30), follow_redirects=False)
        files = {"image": (image.filename, content, image.content_type or "application/octet-stream")}
        data = {"prompt": prompt, "quality": quality, "size": size}
        try:
            upstream = await client.send(client.build_request("POST", "https://api.flymy.ai/api/v1/flymyai/gpt-image-2-5-sunburst_edit/predict", headers={"X-API-KEY": key}, files=files, data=data), stream=True)
        except httpx.HTTPError:
            await client.aclose()
            return json_response({"message": "A FlyMyAI nem érhető el; próbáld újra később."}, 502)
        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
        return StreamingResponse(stream(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "application/json"), headers={"Cache-Control": "no-store"})

    async def browser_auth(request: Request) -> tuple[str, str]:
        token = authorization_token(request)
        user_id = await database.user_id(token)
        return token, user_id

    async def session_access(token: str, session_id: str) -> None:
        if not await database.user_can_access_session(token, session_id):
            raise AppError("NOT_FOUND", "A böngészőnézet nem található.", 404)

    async def load_session(session_id: str) -> dict[str, Any]:
        row = await database.admin_single("chat_browser_sessions", "id,job_id,cdp_url,control_mode,status,agent_busy,viewport_width,viewport_height", [eq("id", session_id)])
        if not row:
            raise AppError("NOT_FOUND", "A böngészőnézet nem található.", 404)
        if row.get("control_mode") not in {"agent", "requested", "human"} or not isinstance(row.get("cdp_url"), str):
            raise AppError("DATABASE_PROTOCOL", "A böngészőmunkamenet érvénytelen.", 502)
        row["viewport_width"] = row["viewport_width"] if isinstance(row.get("viewport_width"), int) and 0 < row["viewport_width"] <= 10000 else 1280
        row["viewport_height"] = row["viewport_height"] if isinstance(row.get("viewport_height"), int) and 0 < row["viewport_height"] <= 10000 else 1029
        return row

    @instance.get("/api/browser")
    async def browser_get(request: Request, id: str | None = None, conversation: str | None = None, frame: int | None = None, limit: int = 50, offset: int = 0) -> JSONResponse:
        try:
            token, user_id = await browser_auth(request)
            if conversation is not None:
                retry = await limiter.check("list", user_id, 60, 60)
                if retry:
                    return browser_error("Túl sok kérés. Próbáld újra később.", 429, {"Retry-After": str(retry)})
                if not uuid_valid(conversation) or not 1 <= limit <= 100 or not 0 <= offset <= 100000:
                    return browser_error("Érvénytelen lapozási paraméter.", 400)
                if not await database.user_can_access_conversation(token, conversation):
                    return json_response({"sessions": [], "nextOffset": None})
                chat_jobs = await database.admin_select("chat_jobs", "id", [eq("conversation_id", conversation)], limit=1000)
                if not chat_jobs:
                    return json_response({"sessions": [], "nextOffset": None})
                rows = await database.admin_select("chat_browser_sessions", "id,control_mode,status,created_at", [in_values("job_id", [str(item["id"]) for item in chat_jobs])], order="created_at.desc", limit=limit + 1, offset=offset)
                more = len(rows) > limit
                return json_response({"sessions": rows[:limit], "nextOffset": offset + limit if more else None})
            if not uuid_valid(id):
                return browser_error("Érvénytelen böngészőnézet.", 400)
            retry = await limiter.check("frame" if frame is not None else "status", f"{user_id}:{id}", 180 if frame is not None else 120, 60)
            if retry:
                return browser_error("Túl sok kérés. Próbáld újra később.", 429, {"Retry-After": str(retry)})
            await session_access(token, id)
            session = await load_session(id)
            if frame is None:
                return json_response({"id": session["id"], "control_mode": session["control_mode"], "status": session["status"]})
            if session["status"] != "active":
                return json_response({"id": session["id"], "error": "A böngésző már leállt.", "control_mode": session["control_mode"], "status": session["status"]}, 409)
            result = await browser_command(session["cdp_url"], "frame")
            return json_response({**result, "id": session["id"], "control_mode": session["control_mode"], "status": session["status"]})
        except AppError as error:
            return browser_error(error.message, error.status_code)

    @instance.post("/api/browser")
    async def browser_post(request: Request) -> JSONResponse:
        try:
            verify_origin(request, config)
            token, user_id = await browser_auth(request)
            raw = await read_json(request, 16384)
            if not isinstance(raw, dict) or not uuid_valid(raw.get("id")):
                return browser_error("Érvénytelen kérés.", 400)
            session_id = raw["id"]
            await session_access(token, session_id)
            if "input" in raw:
                retry = await limiter.check("input", f"{user_id}:{session_id}", 1200, 60)
                if retry:
                    return browser_error("Túl sok kérés. Próbáld újra később.", 429, {"Retry-After": str(retry)})
                input_data = browser_input(raw["input"])
                requested_sequence = raw.get("seq")
                if requested_sequence is not None and (not isinstance(requested_sequence, int) or requested_sequence < 0):
                    return browser_error("Érvénytelen bevitel.", 400)
                async with browser_state.lock(session_id):
                    sequence = browser_state.sequence(session_id, requested_sequence)
                    if sequence is None:
                        return browser_error("Elavult vagy ismételt bevitel.", 409)
                    session = await load_session(session_id)
                    if session["status"] != "active":
                        return browser_error("A böngésző már leállt.", 409)
                    if session["control_mode"] != "human":
                        return browser_error("Előbb vedd át az irányítást.", 409)
                    if not viewport_valid(input_data, session["viewport_width"], session["viewport_height"]):
                        return browser_error("Érvénytelen bevitel.", 400)
                    result = await browser_command(session["cdp_url"], "input", input_data)
                    return json_response({"ok": True, "seq": sequence, "result": result})
            mode = raw.get("mode")
            if mode not in {"human", "agent"}:
                return browser_error("Érvénytelen irányítási kérés.", 400)
            retry = await limiter.check("mode", f"{user_id}:{session_id}", 20, 60)
            if retry:
                return browser_error("Túl sok kérés. Próbáld újra később.", 429, {"Retry-After": str(retry)})
            async with browser_state.lock(session_id):
                if mode == "human":
                    granted = await database.admin_update("chat_browser_sessions", {"control_mode": "human"}, [eq("id", session_id), eq("status", "active"), eq("agent_busy", False), ("control_mode", "in", "(agent,requested)")], "id")
                    if granted:
                        return json_response({"ok": True, "id": session_id, "control_mode": "human"})
                    requested = await database.admin_update("chat_browser_sessions", {"control_mode": "requested"}, [eq("id", session_id), eq("status", "active"), eq("control_mode", "agent")], "id")
                    if requested:
                        return json_response({"ok": True, "id": session_id, "control_mode": "requested"}, 202)
                    session = await load_session(session_id)
                    if session["status"] != "active":
                        return browser_error("A böngésző már leállt.", 409)
                    if session["control_mode"] in {"human", "requested"}:
                        return json_response({"ok": True, "id": session_id, "control_mode": session["control_mode"]}, 202 if session["control_mode"] == "requested" else 200)
                    return browser_error("Az irányítás átvétele nem sikerült.", 409)
                updated = await database.admin_update("chat_browser_sessions", {"control_mode": "agent"}, [eq("id", session_id), eq("status", "active"), ("control_mode", "in", "(human,requested)")], "id,job_id,control_mode")
                if not updated:
                    session = await load_session(session_id)
                    if session["status"] != "active":
                        return browser_error("A böngésző már leállt.", 409)
                    if session["control_mode"] == "agent":
                        return json_response({"ok": True, "id": session_id, "control_mode": "agent"})
                    return browser_error("Az irányítás visszaadása nem sikerült.", 409)
                row = updated[0]
                try:
                    await database.admin_insert("agent_operator_messages", {"task_id": row["job_id"], "content": "A felhasználó visszaadta a böngésző irányítását. Vizsgáld meg az aktuális oldalt, és onnan folytasd; ne ismételd meg a már elvégzett műveleteket.", "consumed": False}, "minimal")
                except AppError:
                    await database.admin_update("chat_browser_sessions", {"control_mode": "human"}, [eq("id", session_id), eq("status", "active"), eq("control_mode", "agent")])
                    return browser_error("Az ügynök értesítése sikertelen; az irányítás nálad maradt.", 502)
                return json_response({"ok": True, "id": session_id, "control_mode": "agent"})
        except AppError as error:
            return browser_error(error.message, error.status_code)

    @instance.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse("/chat.html", status_code=307)

    public_dir = Path(config.public_dir)
    if public_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        instance.mount("/", StaticFiles(directory=str(public_dir), html=True), name="public")

    return instance
