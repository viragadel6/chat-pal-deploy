from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any

from app.agent import runtime as runtime_module
from app.models import TerminalState
from tests.fakes import FakeResponse
from tests.local_stack import ModelPlan, ModelReply


@dataclass
class EventCollector:
    events: list[dict[str, Any]] = field(default_factory=list)
    enqueued: list[str] = field(default_factory=list)

    async def send(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def enqueue(self, text: str) -> None:
        self.enqueued.append(text)

    def types(self) -> list[str]:
        return [str(event.get("type")) for event in self.events]

    def inner_events(self, kind: str) -> list[dict[str, Any]]:
        return [
            event["event"]
            for event in self.events
            if event.get("type") == "agent_event"
            and isinstance(event.get("event"), dict)
            and event["event"].get("type") == kind
        ]

    def tool_results(self) -> list[dict[str, Any]]:
        return self.inner_events("tool_result")

    def notes(self) -> list[dict[str, Any]]:
        return self.inner_events("cognitive_note")

    def terminal(self) -> dict[str, Any]:
        return next(event for event in self.events if event.get("type") == "terminal")


async def run_agent(
    collector: EventCollector, task_id: str, prompt: str
) -> TerminalState:
    return await runtime_module.run_durable_chat(
        task_id,
        [{"role": "user", "content": prompt}],
        collector.send,
        collector.enqueue,
        asyncio.Event(),
    )


async def test_agent_completes_goal_with_state_update(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "A kimenet elkészült."}}]],
            final_content="Elkészült a feladat.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-complete", "Készítsd el a kimenetet.")

    assert outcome == TerminalState.COMPLETED
    assert "progress" in collector.types()
    assert "terminal" in collector.types()
    assert collector.types()[-1] == "done"
    terminal = collector.terminal()
    assert terminal["outcome"] == "completed"
    assert terminal["steps"] == 1
    assert terminal["tool_calls"] == 1
    assert "data: [DONE]\n\n" in collector.enqueued
    results = collector.tool_results()
    assert len(results) == 1
    assert results[0]["tool"] == "update_state"
    assert results[0]["status"] == "completed"
    assert "A kimenet elkészült." in results[0]["output"]


async def test_agent_executes_shell_command_in_sandbox(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [
                    {
                        "name": "run_shell",
                        "arguments": {"command": "printf 'hello from sandbox'"},
                    }
                ],
                [
                    {
                        "name": "update_state",
                        "arguments": {"fact": "A parancs kimenete: hello from sandbox"},
                    }
                ],
            ],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-shell", "Futtasd a parancsot.")

    assert outcome == TerminalState.COMPLETED
    results = collector.tool_results()
    assert results[0]["tool"] == "run_shell"
    assert results[0]["status"] == "completed"
    assert "hello from sandbox" in results[0]["output"]
    assert agent_stack.sandbox_api.commands
    encoded_script = base64.b64encode(b"printf 'hello from sandbox'").decode("ascii")
    assert any(
        encoded_script in str(entry.get("command"))
        for entry in agent_stack.sandbox_api.commands
    )


async def test_agent_executes_python_program(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [
                    {
                        "name": "run_python",
                        "arguments": {"code": "print('python says hi')"},
                    }
                ],
                [
                    {
                        "name": "update_state",
                        "arguments": {"fact": "A python program lefutott."},
                    }
                ],
            ],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-python", "Futtasd a python programot.")

    assert outcome == TerminalState.COMPLETED
    results = collector.tool_results()
    assert results[0]["tool"] == "run_python"
    assert results[0]["status"] == "completed"
    assert "python says hi" in results[0]["output"]


async def test_agent_writes_and_reads_file_in_sandbox(agent_stack):
    path = f"{agent_stack.sandbox_api.root}/artifact.txt"
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [
                    {
                        "name": "write_file",
                        "arguments": {"path": path, "content": "chat-pal artifact"},
                    },
                    {"name": "read_file", "arguments": {"path": path}},
                ],
                [
                    {
                        "name": "update_state",
                        "arguments": {"fact": "A fájl kiírva és visszaolvasva."},
                    }
                ],
            ],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-file", "Írj egy fájlt és olvasd vissza.")

    assert outcome == TerminalState.COMPLETED
    results = collector.tool_results()
    assert [entry["tool"] for entry in results] == [
        "write_file",
        "read_file",
        "update_state",
    ]
    assert all(entry["status"] == "completed" for entry in results)
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == "chat-pal artifact"
    encoded = "Y2hhdC1wYWwgYXJ0aWZhY3Q="
    assert encoded in results[1]["output"]


async def test_agent_verifies_required_file_before_completion(agent_stack):
    path = f"{agent_stack.sandbox_api.root}/verified.txt"
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [
                    {
                        "name": "write_file",
                        "arguments": {"path": path, "content": "igazolt"},
                    }
                ]
            ],
            strategy={
                "goal": {
                    "state_pointers": [],
                    "required_files": [path],
                    "file_contents": [],
                }
            },
            final_content="A fájl létezik, kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-verify", "Készítsd el a fájlt.")

    assert outcome == TerminalState.COMPLETED
    notes = [note for note in collector.notes() if note.get("label") == "ellenőrzés"]
    assert notes
    assert "Minden célfeltétel teljesült." in str(notes[0].get("detail"))


async def test_agent_exposes_service_port(agent_stack):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        agent_stack.use_plan(
            ModelPlan(
                steps=[
                    [{"name": "expose_port", "arguments": {"port": port}}],
                    [
                        {
                            "name": "update_state",
                            "arguments": {"fact": "A port URL kiállítva."},
                        }
                    ],
                ],
                final_content="A szolgáltatás elérhető.",
            )
        )
        collector = EventCollector()
        outcome = await run_agent(collector, "job-port", "Tedd elérhetővé a portot.")

        assert outcome == TerminalState.COMPLETED
        results = collector.tool_results()
        assert results[0]["tool"] == "expose_port"
        assert results[0]["status"] == "completed"
        assert f"https://instavm.local.test/app/{port}" in results[0]["output"]
    finally:
        server.close()
        await server.wait_closed()


async def test_agent_reflects_after_failed_tool_call(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [{"name": "run_shell", "arguments": {"command": "exit 3"}}],
                [
                    {
                        "name": "update_state",
                        "arguments": {"fact": "A hiba után újrapróbáltam."},
                    }
                ],
            ],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-reflect", "Futtasd a hibás parancsot.")

    assert outcome == TerminalState.COMPLETED
    results = collector.tool_results()
    assert results[0]["status"] == "error"
    reflections = [note for note in collector.notes() if note.get("label") == "reflexió"]
    assert reflections
    assert "Gyökérok" in str(reflections[0].get("detail"))


async def test_agent_applies_operator_message(agent_stack):
    consumed = {"count": 0}

    def operator_handler(query: Any) -> FakeResponse:
        consumed["count"] += 1
        if consumed["count"] == 1:
            return FakeResponse(
                [
                    {
                        "id": 1,
                        "task_id": "job-operator",
                        "content": "Adj neki több időt.",
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                ]
            )
        return FakeResponse([])

    agent_stack.database.register("agent_operator_messages", "PATCH", operator_handler)
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "Elkészült."}}]],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-operator", "Végezz el egy feladatot.")

    assert outcome == TerminalState.COMPLETED
    agent_requests = agent_stack.model_api.agent_requests()
    assert agent_requests
    contents = [
        str(message.get("content") or "")
        for message in agent_requests[-1].get("messages") or []
    ]
    assert any("Adj neki több időt." in content for content in contents)


async def test_agent_reports_configuration_failure(agent_stack, monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "")
    monkeypatch.setattr(agent_stack.settings, "orcarouter_api_key", "", raising=False)
    collector = EventCollector()
    outcome = await run_agent(collector, "job-config", "Indíts egy feladatot.")

    assert outcome == TerminalState.FAILED
    errors = [event for event in collector.events if event.get("type") == "error"]
    assert errors
    assert errors[0]["code"] == "CONFIGURATION"
    assert collector.types()[-1] == "done"
    assert "data: [DONE]\n\n" in collector.enqueued


async def test_agent_reports_model_auth_failure(agent_stack):
    def policy(body: dict[str, Any]) -> ModelReply:
        return ModelReply(status_code=401, error={"error": "nope"})

    agent_stack.model_api.policy = policy
    collector = EventCollector()
    outcome = await run_agent(collector, "job-auth", "Indíts egy feladatot.")

    assert outcome == TerminalState.FAILED
    errors = [event for event in collector.events if event.get("type") == "error"]
    assert errors
    assert errors[0]["code"] == "MODEL_AUTH"
    assert collector.terminal()["outcome"] == "failed"


async def test_agent_reports_model_protocol_failure(agent_stack):
    def policy(body: dict[str, Any]) -> ModelReply:
        return ModelReply(
            events=[
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "run_shell",
                                            "arguments": "{not json",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        )

    agent_stack.model_api.policy = policy
    collector = EventCollector()
    outcome = await run_agent(collector, "job-protocol", "Indíts egy feladatot.")

    assert outcome == TerminalState.FAILED
    errors = [event for event in collector.events if event.get("type") == "error"]
    assert errors
    assert errors[0]["code"] == "MODEL_INCOMPLETE"


async def test_agent_reports_invalid_tool_arguments_as_tool_error(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [{"name": "run_shell", "raw_arguments": "{not json"}],
                [
                    {
                        "name": "update_state",
                        "arguments": {"fact": "A hibás hívást rögzítettem."},
                    }
                ],
            ],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-arguments", "Hívj egy eszközt.")

    assert outcome == TerminalState.COMPLETED
    results = collector.tool_results()
    assert results[0]["tool"] == "run_shell"
    assert results[0]["status"] == "error"
    assert "INVALID_ARGUMENT" in results[0]["output"]
    assert "Eszközargumentum JSON hiba" in results[0]["output"]


async def test_agent_reports_missing_tool_argument_as_tool_error(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[
                [{"name": "run_shell", "arguments": {}}],
                [
                    {
                        "name": "update_state",
                        "arguments": {"fact": "A hiányzó argumentumot észleltem."},
                    }
                ],
            ],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    outcome = await run_agent(collector, "job-missing-argument", "Hívj egy eszközt.")

    assert outcome == TerminalState.COMPLETED
    results = collector.tool_results()
    assert results[0]["status"] == "error"
    assert "INVALID_ARGUMENT" in results[0]["output"]
    assert "Hiányzó argumentum: command" in results[0]["output"]


async def test_agent_reports_invalid_model_event_as_protocol_failure(agent_stack):
    agent_stack.model_api.raw_payloads = ["{not json}"]
    collector = EventCollector()
    outcome = await run_agent(collector, "job-raw-event", "Indíts egy feladatot.")

    assert outcome == TerminalState.FAILED
    errors = [event for event in collector.events if event.get("type") == "error"]
    assert errors
    assert errors[0]["code"] == "MODEL_PROTOCOL"


async def test_agent_sends_model_tools_and_messages(agent_stack):
    agent_stack.use_plan(
        ModelPlan(
            steps=[[{"name": "update_state", "arguments": {"fact": "Tény."}}]],
            final_content="Kész.",
        )
    )
    collector = EventCollector()
    await run_agent(collector, "job-contract", "Írj egy összefoglalót.")

    request = agent_stack.model_api.agent_requests()[0]
    assert request["stream"] is True
    assert request["tool_choice"] == "auto"
    names = {tool["function"]["name"] for tool in request.get("tools") or []}
    assert "run_shell" in names
    assert "update_state" in names
    assert "reason" in names
    messages = request.get("messages") or []
    assert messages[0]["role"] == "system"
    assert "Feladat indítása" in str(messages[0].get("content"))
    assert messages[-1]["role"] == "user"
    assert json.dumps(request, ensure_ascii=False)
