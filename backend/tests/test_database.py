import json

import httpx
import pytest

from app.database import DatabaseError, SupabaseDatabase, encode_in_value, encode_value


def build_client(handler) -> SupabaseDatabase:
    database = SupabaseDatabase("https://example.supabase.co", "sb_secret_test_key")
    database._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return database


@pytest.mark.asyncio
async def test_select_builds_postgrest_query():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=[{"id": "a"}])

    database = build_client(handler)
    response = await (
        database.table("chat_jobs")
        .select("id", "status")
        .eq("status", "queued")
        .gt("id", 42)
        .order("created_at", desc=True)
        .limit(10)
        .execute()
    )
    assert response.rows == [{"id": "a"}]
    assert "select=id%2Cstatus" in captured["url"] or "select=id,status" in captured["url"]
    assert "status=eq.queued" in captured["url"]
    assert "id=gt.42" in captured["url"]
    assert "order=created_at.desc" in captured["url"]
    assert "limit=10" in captured["url"]
    assert captured["headers"]["apikey"] == "sb_secret_test_key"
    assert "authorization" not in captured["headers"]
    await database.close()


@pytest.mark.asyncio
async def test_legacy_jwt_key_sends_bearer_authorization():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=[])

    database = SupabaseDatabase("https://example.supabase.co", "legacy-jwt-token")
    database._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await database.table("chat_jobs").select("id").execute()
    assert captured["headers"]["authorization"] == "Bearer legacy-jwt-token"
    await database.close()


@pytest.mark.asyncio
async def test_update_returns_updated_rows():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.headers["prefer"] == "return=representation"
        body = json.loads(request.content)
        assert body == {"status": "cancelled"}
        return httpx.Response(200, json=[{"id": "job", "status": "cancelled"}])

    database = build_client(handler)
    response = await (
        database.table("chat_jobs").update({"status": "cancelled"}).eq("id", "job").execute()
    )
    assert response.rows[0]["status"] == "cancelled"
    await database.close()


@pytest.mark.asyncio
async def test_upsert_sets_on_conflict_and_resolution():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["prefer"] = request.headers.get("prefer", "")
        return httpx.Response(201, json=[{"id": "c1"}])

    database = build_client(handler)
    await database.table("chat_conversations").upsert(
        {"id": "c1", "title": "x"}, on_conflict="id"
    ).execute()
    assert "on_conflict=id" in captured["url"]
    assert "resolution=mergeDuplicates" in captured["prefer"]
    await database.close()


@pytest.mark.asyncio
async def test_rpc_posts_json_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=True)

    database = build_client(handler)
    response = await database.rpc(
        "claim_chat_browser_step", {"_job_id": "11111111-1111-4111-8111-111111111111"}
    ).execute()
    assert response.data is True
    assert captured["path"].endswith("/rpc/claim_chat_browser_step")
    assert captured["body"] == {"_job_id": "11111111-1111-4111-8111-111111111111"}
    await database.close()


@pytest.mark.asyncio
async def test_error_response_raises_database_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409, json={"message": "duplicate key", "code": "23505", "details": "id"}
        )

    database = build_client(handler)
    with pytest.raises(DatabaseError) as excinfo:
        await database.table("chat_jobs").insert({"id": "x"}).execute()
    assert excinfo.value.status_code == 409
    assert excinfo.value.code == "23505"
    await database.close()


@pytest.mark.asyncio
async def test_count_exact_reads_content_range():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[{"id": "a"}], headers={"content-range": "0-0/42"}
        )

    database = build_client(handler)
    total = await database.table("agent_tasks").select("id", count="exact", head=True).count_exact()
    assert total == 42
    await database.close()


def test_value_encoding_handles_special_characters():
    assert encode_value(None) == "null"
    assert encode_value(True) == "true"
    assert encode_value(12) == "12"
    assert encode_value("abc") == "abc"
    assert encode_in_value(["a", "b"]) == "(a,b)"
    assert encode_in_value(["a,b", "c"]) == '("a,b",c)'
    assert encode_value({"k": 1}) == '{"k":1}'
