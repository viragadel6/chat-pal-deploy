from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import database as database_module
from app.main import create_app
from app.routers import browser as browser_router
from tests.fakes import FakeDatabase

SESSION_ID = "33333333-3333-4333-8333-333333333333"
USER_ID = "44444444-4444-4444-8444-444444444444"
ORIGIN = {"origin": "http://testserver"}


class BrowserHarness:
    def __init__(self) -> None:
        self.database = FakeDatabase(stateful=True)
        self.commands: list[tuple[str, str, dict[str, Any] | None]] = []
        self.sessions: list[dict[str, Any]] = []

    def seed_session(self, control_mode: str = "human", status: str = "active") -> None:
        self.database.store.table("chat_browser_sessions").append(
            {
                "id": SESSION_ID,
                "job_id": "55555555-5555-4555-8555-555555555555",
                "cdp_url": "ws://127.0.0.1:9222/devtools/browser/test",
                "control_mode": control_mode,
                "status": status,
                "agent_busy": False,
                "viewport_width": 1280,
                "viewport_height": 1029,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )

    async def browser_command(self, url: str, action: str, input_data: dict[str, Any] | None = None):
        self.commands.append((url, action, input_data))
        if action == "frame":
            return {"data": "aGVsbG8=", "url": "https://example.test/"}
        return {"ok": True}


@pytest.fixture()
def harness(monkeypatch):
    state = BrowserHarness()
    monkeypatch.setattr(database_module, "get_admin_database", lambda: state.database)

    async def authenticate(request):
        return USER_ID, None

    async def command(url: str, action: str, input_data: dict[str, Any] | None = None):
        return await state.browser_command(url, action, input_data)

    monkeypatch.setattr(browser_router, "_authenticate", authenticate)
    monkeypatch.setattr(browser_router, "browser_command", command)
    browser_router._last_sequences.clear()
    browser_router._auto_sequences.clear()
    app = create_app()
    client = TestClient(app, base_url="http://testserver")
    client.harness = state
    yield client


def test_browser_input_accepts_frontend_payload(harness):
    harness.harness.seed_session()
    response = harness.post(
        "/api/browser",
        json={
            "id": SESSION_ID,
            "input": {
                "kind": "mouse",
                "type": "mousePressed",
                "button": "left",
                "buttons": 1,
                "clickCount": 1,
                "x": 120.5,
                "y": 240.25,
            },
        },
        headers=ORIGIN,
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    url, action, input_data = harness.harness.commands[-1]
    assert action == "input"
    assert input_data == {
        "kind": "mouse",
        "type": "mousePressed",
        "x": 120.5,
        "y": 240.25,
        "button": "left",
        "clickCount": 1,
        "buttons": 1,
        "modifiers": 0,
    }


def test_browser_input_generates_monotonic_sequence(harness):
    harness.harness.seed_session()
    for _ in range(3):
        response = harness.post(
            "/api/browser",
            json={
                "id": SESSION_ID,
                "input": {"kind": "text", "text": "a"},
            },
            headers=ORIGIN,
        )
        assert response.status_code == 200
    assert browser_router._last_sequences[SESSION_ID] == 3


def test_browser_input_rejects_replayed_sequence(harness):
    harness.harness.seed_session()
    payload = {
        "id": SESSION_ID,
        "seq": 5,
        "input": {"kind": "text", "text": "a"},
    }
    first = harness.post("/api/browser", json=payload, headers=ORIGIN)
    assert first.status_code == 200
    second = harness.post("/api/browser", json=payload, headers=ORIGIN)
    assert second.status_code == 409
    assert second.json()["lastSeq"] == 5


def test_browser_input_requires_human_control(harness):
    harness.harness.seed_session(control_mode="agent")
    response = harness.post(
        "/api/browser",
        json={"id": SESSION_ID, "input": {"kind": "text", "text": "a"}},
        headers=ORIGIN,
    )
    assert response.status_code == 409
    assert response.json()["error"] == "Előbb vedd át az irányítást."


def test_browser_input_rejects_out_of_viewport_coordinates(harness):
    harness.harness.seed_session()
    response = harness.post(
        "/api/browser",
        json={
            "id": SESSION_ID,
            "input": {
                "kind": "mouse",
                "type": "mouseMoved",
                "button": "none",
                "buttons": 0,
                "clickCount": 0,
                "x": 5000,
                "y": 10,
            },
        },
        headers=ORIGIN,
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Érvénytelen bevitel."


def test_browser_input_rejects_unknown_modifier_count(harness):
    harness.harness.seed_session()
    response = harness.post(
        "/api/browser",
        json={
            "id": SESSION_ID,
            "input": {
                "kind": "key",
                "type": "keyDown",
                "key": "a",
                "code": "KeyA",
                "keyCode": 65,
                "modifiers": 99,
            },
        },
        headers=ORIGIN,
    )
    assert response.status_code == 400


def test_browser_takeover_grants_control(harness):
    harness.harness.seed_session(control_mode="agent")
    response = harness.post(
        "/api/browser", json={"id": SESSION_ID, "mode": "human"}, headers=ORIGIN
    )
    assert response.status_code == 200
    assert response.json()["control_mode"] == "human"
    stored = harness.harness.database.store.table("chat_browser_sessions")[0]
    assert stored["control_mode"] == "human"


def test_browser_session_status_returns_frame(harness):
    harness.harness.seed_session()
    response = harness.get(f"/api/browser?id={SESSION_ID}&frame=1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] == "aGVsbG8="
    assert payload["control_mode"] == "human"
    assert payload["status"] == "active"


def test_browser_conversation_list_returns_sessions(harness):
    harness.harness.seed_session()
    harness.harness.database.store.table("chat_jobs").append(
        {"id": "66666666-6666-4666-8666-666666666666", "conversation_id": SESSION_ID}
    )
    rows = harness.harness.database.store.table("chat_browser_sessions")

    def handle(query):
        from tests.fakes import FakeResponse

        return FakeResponse([dict(row) for row in rows])

    harness.harness.database.register("chat_browser_sessions", "GET", handle)
    response = harness.get(f"/api/browser?conversation={SESSION_ID}")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == SESSION_ID
