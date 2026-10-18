from __future__ import annotations

import pytest

from app import database as database_module
from app.main import create_app
from tests.fakes import FakeResponse

VALID_JOB_ID = "11111111-1111-4111-8111-111111111111"
VALID_CONVERSATION_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from tests.fakes import FakeDatabase

    fake_db = FakeDatabase()
    monkeypatch.setattr(database_module, "get_admin_database", lambda: fake_db)
    app = create_app()
    test_client = TestClient(app, base_url="http://testserver")
    test_client.fake_db = fake_db
    yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_rejects_non_json_content_type(client):
    response = client.post(
        "/api/chat", content=b"hello", headers={"content-type": "text/plain"}
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_chat_rejects_missing_user_message(client):
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "assistant", "content": "szia"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_chat_rejects_tool_history_from_client(client):
    response = client.post(
        "/api/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "x",
                    "tool_calls": [{"id": "1"}],
                }
            ]
        },
    )
    assert response.status_code == 400


def test_chat_creates_conversation_and_job(client, monkeypatch):
    stored: dict = {"conversation": None, "job": None}

    def conversation_handler(query):
        stored["conversation"] = query._payload
        return FakeResponse([{"id": stored["conversation"]["id"]}])

    def job_handler(query):
        stored["job"] = query._payload
        return FakeResponse([{"id": stored["job"]["id"]}])

    client.fake_db.register("chat_conversations", "POST", conversation_handler)
    client.fake_db.register("chat_jobs", "POST", job_handler)
    client.fake_db.register("chat_job_events", "GET", lambda query: FakeResponse([]))
    client.fake_db.register(
        "chat_jobs", "GET", lambda query: FakeResponse([{"status": "completed"}])
    )

    async def fake_dispatch(job_id: str, request_url: str) -> None:
        stored["dispatched"] = job_id

    monkeypatch.setattr("app.routers.chat.dispatch_chat_job", fake_dispatch)

    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Írj egy verset."}]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert stored["conversation"] is not None
    assert stored["conversation"]["title"] == "Írj egy verset."
    assert stored["job"] is not None
    assert stored["job"]["status"] == "queued"
    assert stored["dispatched"] == stored["job"]["id"]


def test_conversations_requires_ids(client):
    response = client.get("/api/conversations")
    assert response.status_code == 200
    assert response.json() == {"conversations": []}


def test_conversations_filters_invalid_ids(client):
    response = client.get("/api/conversations?ids=not-a-uuid")
    assert response.status_code == 200
    assert response.json() == {"conversations": []}


def test_conversations_returns_latest_job(client):
    client.fake_db.register(
        "chat_conversations",
        "GET",
        lambda query: FakeResponse(
            [
                {
                    "id": VALID_CONVERSATION_ID,
                    "title": "Teszt",
                    "messages": [],
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        ),
    )
    client.fake_db.register(
        "chat_jobs",
        "GET",
        lambda query: FakeResponse(
            [
                {
                    "id": VALID_JOB_ID,
                    "conversation_id": VALID_CONVERSATION_ID,
                    "status": "completed",
                    "response_content": "válasz",
                    "response_reasoning": "",
                    "error": None,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        ),
    )
    client.fake_db.register(
        "chat_job_events",
        "GET",
        lambda query: FakeResponse([{"job_id": VALID_JOB_ID, "id": 7}]),
    )
    response = client.get(f"/api/conversations?ids={VALID_CONVERSATION_ID}")
    assert response.status_code == 200
    payload = response.json()["conversations"][0]
    assert payload["conversation_id"] == VALID_CONVERSATION_ID
    assert payload["latest_job"]["job_id"] == VALID_JOB_ID
    assert payload["latest_job"]["last_event_id"] == 7


def test_operator_rejects_invalid_payload(client):
    response = client.post("/api/operator", json={"task_id": "nope", "content": "x"})
    assert response.status_code == 400


def test_operator_stores_message(client, monkeypatch):
    stored: dict = {}

    async def fake_post(task_id: str, content: str) -> None:
        stored["task_id"] = task_id
        stored["content"] = content

    monkeypatch.setattr("app.routers.operator.cog.post_operator_message", fake_post)
    response = client.post(
        "/api/operator", json={"task_id": VALID_JOB_ID, "content": "  irányítás  "}
    )
    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert stored == {"task_id": VALID_JOB_ID, "content": "irányítás"}


def test_stop_job_cancels_queued_job(client):
    updated: dict = {}

    def update_handler(query):
        updated["payload"] = query._payload
        return FakeResponse([{"id": VALID_JOB_ID}])

    client.fake_db.register("chat_jobs", "GET", lambda query: FakeResponse([{"id": VALID_JOB_ID, "status": "queued"}]))
    client.fake_db.register("chat_jobs", "PATCH", update_handler)
    response = client.post(f"/api/chat/jobs/{VALID_JOB_ID}/stop")
    assert response.status_code == 200
    assert updated["payload"]["status"] == "cancelled"
    assert updated["payload"]["stop_requested"] is True


def test_stop_job_marks_running_job(client):
    updated: dict = {}

    def update_handler(query):
        updated["payload"] = query._payload
        return FakeResponse([{"id": VALID_JOB_ID}])

    client.fake_db.register("chat_jobs", "GET", lambda query: FakeResponse([{"id": VALID_JOB_ID, "status": "running"}]))
    client.fake_db.register("chat_jobs", "PATCH", update_handler)
    response = client.post(f"/api/chat/jobs/{VALID_JOB_ID}/stop")
    assert response.status_code == 200
    assert updated["payload"]["stop_requested"] is True
    assert "status" not in updated["payload"]


def test_stop_job_rejects_invalid_identifier(client):
    response = client.post("/api/chat/jobs/not-a-uuid/stop")
    assert response.status_code == 400


def test_redirect_job_rejects_closed_job(client):
    client.fake_db.register(
        "chat_jobs", "GET", lambda query: FakeResponse([{"id": VALID_JOB_ID, "status": "completed"}])
    )
    response = client.post(
        f"/api/chat/jobs/{VALID_JOB_ID}/redirect", json={"message": "másik irány"}
    )
    assert response.status_code == 409


def test_redirect_job_stores_operator_message(client, monkeypatch):
    stored: dict = {}

    async def fake_post(task_id: str, content: str) -> None:
        stored["content"] = content

    monkeypatch.setattr("app.routers.chat_jobs.cog.post_operator_message", fake_post)
    client.fake_db.register(
        "chat_jobs", "GET", lambda query: FakeResponse([{"id": VALID_JOB_ID, "status": "running"}])
    )
    response = client.post(
        f"/api/chat/jobs/{VALID_JOB_ID}/redirect", json={"message": "másik irány"}
    )
    assert response.status_code == 200
    assert stored["content"] == "másik irány"


def test_browser_requires_authentication(client):
    response = client.get("/api/browser?id=" + VALID_JOB_ID)
    assert response.status_code == 401


def test_browser_rejects_cross_origin_post(client):
    response = client.post(
        "/api/browser",
        json={"id": VALID_JOB_ID, "mode": "human"},
        headers={"origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_image_edit_requires_api_key(client, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "flymyai_api_key", "", raising=False)
    response = client.post("/api/images/edits", files={"image": ("a.png", b"x", "image/png")}, data={"prompt": "x"})
    assert response.status_code == 503


def test_image_edit_validates_payload(client, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "flymyai_api_key", "fly-key", raising=False)
    response = client.post("/api/images/edits", files={"image": ("a.png", b"x", "image/png")}, data={"prompt": ""})
    assert response.status_code == 400
