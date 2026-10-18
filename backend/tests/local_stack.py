from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


@dataclass
class ModelReply:
    events: list[dict[str, Any]] = field(default_factory=list)
    status_code: int = 200
    error: dict[str, Any] | None = None


ModelPolicy = Callable[[dict[str, Any]], ModelReply]


def tool_call_event(
    call_id: str, name: str, arguments: dict[str, Any] | str, index: int = 0
) -> dict[str, Any]:
    payload = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": payload,
                            },
                        }
                    ]
                }
            }
        ]
    }


def tool_calls_finish_event() -> dict[str, Any]:
    return {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


def content_event(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def reasoning_event(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"reasoning_content": text}}]}


def stop_event() -> dict[str, Any]:
    return {"choices": [{"delta": {}, "finish_reason": "stop"}]}


class ModelService:
    def __init__(self, policy: ModelPolicy) -> None:
        self.policy = policy
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.raw_payloads: list[str] | None = None

    def agent_requests(self) -> list[dict[str, Any]]:
        return [body for body in self.requests if body.get("tools")]

    def cognitive_requests(self) -> list[dict[str, Any]]:
        return [body for body in self.requests if not body.get("tools")]

    async def handle(self, request: Request) -> Response:
        body = await request.json()
        self.requests.append(body)
        self.headers.append({key.lower(): value for key, value in request.headers.items()})
        if self.raw_payloads is not None:
            return StreamingResponse(self._raw_stream(), media_type="text/event-stream")
        reply = self.policy(body)
        if reply.status_code != 200:
            return JSONResponse(
                reply.error or {"error": {"message": "A modell szolgáltató hibát adott."}},
                status_code=reply.status_code,
            )

        async def stream() -> Any:
            for event in reply.events:
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def _raw_stream(self) -> Any:
        for payload in self.raw_payloads or []:
            yield f"data: {payload}\n\n".encode()
        yield b"data: [DONE]\n\n"


class SandboxService:
    def __init__(self, root: str) -> None:
        self.root = root
        self.sessions: dict[str, dict[str, Any]] = {}
        self.commands: list[dict[str, Any]] = []
        self.app_urls: list[dict[str, Any]] = []

    async def provision(self, request: Request) -> Response:
        payload = await request.json()
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return JSONResponse({"error": "hiányzó session_id"}, status_code=400)
        self.sessions[session_id] = {
            "status": "running",
            "vm_lifetime_seconds": payload.get("vm_lifetime_seconds"),
            "egress_policy": payload.get("egress_policy"),
        }
        return JSONResponse({"session_id": session_id, "status": "running"})

    async def execute(self, request: Request) -> Response:
        payload = await request.json()
        command = str(payload.get("command") or "")
        session_id = str(payload.get("session_id") or "")
        timeout = int(payload.get("timeout") or 60)
        self.commands.append(
            {"session_id": session_id, "command": command, "timeout": timeout}
        )
        stdout, stderr, exit_code = await self._run(command, timeout)
        return JSONResponse(
            {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
        )

    async def kill(self, request: Request) -> Response:
        payload = await request.json()
        session_id = str(payload.get("session_id") or "")
        self.sessions.pop(session_id, None)
        return JSONResponse({})

    async def app_url(self, request: Request) -> Response:
        session_id = request.path_params.get("session_id") or ""
        port = request.query_params.get("port") or ""
        self.app_urls.append({"session_id": session_id, "port": port})
        return JSONResponse({"url": f"https://instavm.local.test/app/{port}"})

    async def _run(self, command: str, timeout: int) -> tuple[str, str, int]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
            "HOME": self.root,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "dumb",
            "TMPDIR": self.root,
        }
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=self.root,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=max(5, timeout) + 20
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return (
                stdout.decode("utf-8", "replace"),
                stderr.decode("utf-8", "replace") + "\nA parancs határidő lejárt.",
                124,
            )
        return (
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
            process.returncode if process.returncode is not None else 0,
        )


class _QuietServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@dataclass
class LocalService:
    app: Any
    base_url: str = ""
    server: _QuietServer | None = None
    task: asyncio.Task[None] | None = None

    async def start(self) -> "LocalService":
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        socket_handle = config.bind_socket()
        port = socket_handle.getsockname()[1]
        server = _QuietServer(config)
        task = asyncio.create_task(server.serve(sockets=[socket_handle]))
        for _ in range(1000):
            if server.started:
                break
            if task.done():
                await task
                raise RuntimeError("A helyi tesztkiszolgáló nem indult el.")
            await asyncio.sleep(0.01)
        else:
            server.should_exit = True
            raise RuntimeError("A helyi tesztkiszolgáló indítása időtúllépett.")
        self.server = server
        self.task = task
        self.base_url = f"http://127.0.0.1:{port}"
        return self

    async def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.task is not None:
            try:
                await asyncio.wait_for(self.task, timeout=15)
            except (TimeoutError, asyncio.CancelledError):
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        self.server = None
        self.task = None


async def start_model_service(policy: ModelPolicy) -> tuple[LocalService, ModelService]:
    service = ModelService(policy)
    app = Starlette(routes=[Route("/v1/chat/completions", service.handle, methods=["POST"])])
    local = await LocalService(app).start()
    return local, service


async def start_sandbox_service() -> tuple[LocalService, SandboxService]:
    root = tempfile.mkdtemp(prefix="chat-pal-vm-")
    service = SandboxService(root)
    app = Starlette(
        routes=[
            Route("/v1/vms", service.provision, methods=["POST"]),
            Route("/execute", service.execute, methods=["POST"]),
            Route("/kill", service.kill, methods=["POST"]),
            Route("/v1/sessions/app-url/{session_id}", service.app_url, methods=["GET"]),
        ]
    )
    local = await LocalService(app).start()
    return local, service


def remove_sandbox_root(service: SandboxService) -> None:
    shutil.rmtree(service.root, ignore_errors=True)


@dataclass
class AgentStack:
    settings: Any
    database: Any
    model_api: ModelService
    model_server: LocalService
    sandbox_api: SandboxService
    sandbox_server: LocalService

    def use_plan(self, plan: ModelPlan) -> None:
        self.model_api.policy = build_policy(plan)


@dataclass
class ModelPlan:
    steps: list[list[dict[str, Any]]] = field(default_factory=list)
    final_content: str = "Elkészült."
    strategy: dict[str, Any] = field(default_factory=dict)
    reflection: dict[str, Any] = field(default_factory=dict)
    skill_dsl: str = (
        "SKILL teszt_kepessseg\n"
        "WHEN egy teszt feladat fut\n"
        "REQUIRE a cél ismert\n"
        "STEP ellenőrizd a környezetet\n"
        "STEP hajtsd végre a lépést\n"
        "VERIFY a kimenet létezik\n"
        "RECOVER ismételd meg a lépést"
    )
    playbook: str = "Bevált eljárások: Ellenőrizd a környezetet."
    extra_text: str = "Rendben, folytatom."


def _strategy_payload(plan: ModelPlan) -> dict[str, Any]:
    strategy = dict(plan.strategy)
    strategy.setdefault("strategy", "A feladat következő lépéseinek megtervezése.")
    strategy.setdefault("subgoals", [])
    strategy.setdefault("next_steps", ["Ellenőrizd a környezetet."])
    strategy.setdefault("vector", [0.6] * 16)
    strategy.setdefault(
        "goal", {"state_pointers": [], "required_files": [], "file_contents": []}
    )
    return strategy


def _reflection_payload(plan: ModelPlan) -> dict[str, Any]:
    reflection = dict(plan.reflection)
    reflection.setdefault("failure_point", "A(z) eszközhívás nem adott ellenőrizhető eredményt.")
    reflection.setdefault("component", "executor")
    reflection.setdefault("root_cause", "A környezet nem a várt állapotban volt.")
    reflection.setdefault("pivot_action", "Ellenőrizd a kimenetet újra.")
    reflection.setdefault("patch", "Futtasd újra a parancsot, majd ellenőrizd a fájlt.")
    reflection.setdefault("corrective_action", "ls -l")
    return reflection


def build_policy(plan: ModelPlan) -> ModelPolicy:
    def policy(body: dict[str, Any]) -> ModelReply:
        messages = body.get("messages") or []
        system = ""
        for message in messages:
            if message.get("role") == "system":
                system = str(message.get("content") or "")
                break
        if not body.get("tools"):
            return ModelReply(
                events=[content_event(_cognitive_answer(system, plan)), stop_event()]
            )
        completed = len([message for message in messages if message.get("role") == "tool"])
        consumed = 0
        for step in plan.steps:
            if consumed >= completed:
                events: list[dict[str, Any]] = []
                for index, call in enumerate(step):
                    raw_arguments = call.get("raw_arguments")
                    arguments: dict[str, Any] | str = (
                        str(raw_arguments)
                        if raw_arguments is not None
                        else dict(call.get("arguments") or {})
                    )
                    events.append(
                        tool_call_event(
                            f"call-{consumed + index + 1}",
                            str(call.get("name") or ""),
                            arguments,
                            index=index,
                        )
                    )
                events.append(tool_calls_finish_event())
                return ModelReply(events=events)
            consumed += len(step)
        return ModelReply(
            events=[
                reasoning_event("A célfeltételek teljesülnek, lezárom a feladatot."),
                content_event(plan.final_content),
                stop_event(),
            ]
        )

    return policy


def _cognitive_answer(system: str, plan: ModelPlan) -> str:
    if "stratégiai tervező" in system:
        return json.dumps(_strategy_payload(plan), ensure_ascii=False)
    if "Hibaelemző" in system:
        return json.dumps(_reflection_payload(plan), ensure_ascii=False)
    if "Meta-ágens" in system:
        return plan.skill_dsl
    if "Tapasztalati tudáskonszolidátor" in system:
        return plan.playbook
    return plan.extra_text
