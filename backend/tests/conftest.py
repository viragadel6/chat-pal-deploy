import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_test_key")
os.environ.setdefault("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test_key")
os.environ.setdefault("ORCAROUTER_API_KEY", "test-orcarouter-key-value")
os.environ.setdefault("INSTAVM_API_KEY", "test-instavm-key-value")
os.environ.setdefault("CHAT_DEPLOYMENT_MAX_REQUEST_MS", "22000000")


@pytest.fixture(autouse=True)
def clear_settings_cache():
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
async def agent_stack(monkeypatch):
    from app import database as database_module
    from app.config import get_settings
    from tests.fakes import FakeDatabase, FakeResponse
    from tests.local_stack import (
        AgentStack,
        ModelPlan,
        build_policy,
        remove_sandbox_root,
        start_model_service,
        start_sandbox_service,
    )

    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-test-key-value")
    monkeypatch.setenv("INSTAVM_API_KEY", "instavm-test-key-value")
    monkeypatch.setenv("CHAT_DEPLOYMENT_MAX_REQUEST_MS", "22000000")
    monkeypatch.setenv("CHAT_SANDBOX_ALLOWED_DOMAINS", "")
    monkeypatch.setenv("CHAT_BROWSER_ALLOWED_HOSTS", "")
    get_settings.cache_clear()

    database = FakeDatabase(stateful=True)
    database.register_rpc("claim_chat_browser_step", lambda payload: FakeResponse(data=True))
    monkeypatch.setattr(database_module, "get_admin_database", lambda: database)

    model_server, model_api = await start_model_service(build_policy(ModelPlan()))
    sandbox_server, sandbox_api = await start_sandbox_service()
    monkeypatch.setenv("ORCAROUTER_URL", f"{model_server.base_url}/v1/chat/completions")
    monkeypatch.setenv("INSTAVM_URL", sandbox_server.base_url)
    get_settings.cache_clear()
    try:
        yield AgentStack(
            settings=get_settings(),
            database=database,
            model_api=model_api,
            model_server=model_server,
            sandbox_api=sandbox_api,
            sandbox_server=sandbox_server,
        )
    finally:
        await model_server.stop()
        await sandbox_server.stop()
        remove_sandbox_root(sandbox_api)
