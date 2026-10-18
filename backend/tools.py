from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import random
import re
import socket
import time
import uuid
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .config import Settings
from .db import Supabase, eq
from .errors import AppError

INSTAVM_URL = "https://api.instavm.io"
BROWSER_USE_URL = "https://api.browser-use.com/api/v4/browsers"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MAX_FILE_BYTES = 131072
MAX_PAGE_BYTES = 6000
MAX_RESULT_BYTES = 12000
MAX_PROVIDER_BYTES = 2097152
MAX_FILE_CHUNK_BYTES = 65536
EXECUTE_WAIT_MAX_SECONDS = 600
CLEANUP_TIMEOUT = 12


def require_string(values: dict[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str):
        raise AppError("INVALID_ARGUMENT", f"Hibás szöveges argumentum: {name}", 400)
    return value


def require_int(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AppError("INVALID_ARGUMENT", f"Hibás egész szám argumentum: {name}", 400)
    return value


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


class Subagents:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=30), follow_redirects=False, http2=True)

    async def close(self) -> None:
        await self.client.aclose()

    async def _interact(self, model: str, prompt: str, cancellation: asyncio.Event) -> dict[str, Any]:
        key = self.config.secret("gemini_api_key")
        if not key:
            raise AppError("CONFIGURATION", "A GEMINI_API_KEY nincs beállítva.", 503)
        body = {"model": model, "input": prompt, "tools": [{"type": "code_execution"}, {"type": "google_search"}, {"type": "url_context"}], "generation_config": {"max_output_tokens": 65536, "thinking_level": "high"}}
        for attempt in range(3):
            if cancellation.is_set():
                raise AppError("CANCELLED", "A feladat megszakadt.", 499)
            try:
                response = await self.client.post(GEMINI_URL, headers={"x-goog-api-key": key}, json=body)
            except httpx.HTTPError as error:
                if attempt == 2:
                    raise AppError("SUBAGENT_FAILED", "A Gemini szolgáltatás nem érhető el.", 502) from error
                await asyncio.sleep(2 ** attempt + random.random())
                continue
            if response.is_success:
                if len(response.content) > MAX_PROVIDER_BYTES:
                    raise AppError("RESPONSE_TOO_LARGE", "A válasz meghaladja a bájthatárt.", 502)
                try:
                    data = response.json()
                except ValueError as error:
                    raise AppError("SUBAGENT_FAILED", "A Gemini érvénytelen választ adott.", 502) from error
                parts: list[str] = []
                self._collect_text(data, parts)
                answer = "\n".join(dict.fromkeys(part for part in parts if part.strip())).strip()
                if not answer:
                    raise AppError("SUBAGENT_FAILED", "A Gemini üres választ adott.", 502)
                return {"model": model, "text": answer}
            message = response.text[:500]
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2 and "quota" not in message.casefold():
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1) + random.random()
                await asyncio.sleep(delay)
                continue
            raise AppError("SUBAGENT_FAILED", f"Gemini HTTP {response.status_code}: {message}", 502)
        raise AppError("SUBAGENT_FAILED", "A Gemini újrapróbálkozási kerete elfogyott.", 502)

    def _collect_text(self, value: Any, result: list[str]) -> None:
        if isinstance(value, list):
            for item in value:
                self._collect_text(item, result)
        elif isinstance(value, dict):
            if value.get("type") == "text" and isinstance(value.get("text"), str):
                result.append(value["text"])
            for key in ("outputs", "steps", "content", "parts"):
                if key in value:
                    self._collect_text(value[key], result)

    async def delegate(self, task: str, context: str, cancellation: asyncio.Event) -> dict[str, Any]:
        try:
            response = await self._interact("models/gemini-3.8-flash", f"Önálló alügynök vagy. Hajtsd végre pontosan a részfeladatot; kódfuttatást, Google keresést és URL-olvasást használhatsz. Adj tömör, ellenőrzött eredményt magyarul, forrásokkal ha kerestél.\n\nKontextus:\n{context or 'nincs'}\n\nRészfeladat:\n{task}", cancellation)
            return {"status": "completed", "result": {"model": response["model"], "answer": response["text"]}}
        except AppError as error:
            return {"status": "error", "result": {"code": error.code, "message": error.message}}

    async def advisor(self, question: str, context: str, cancellation: asyncio.Event) -> dict[str, Any]:
        try:
            response = await self._interact("models/gemini-3.1-pro-preview", f"Szakértő tanácsadó vagy egy autonóm ügynök mellett. Elemezd a helyzetet, mutass rá hibákra és kockázatokra, és adj konkrét, sorrendbe állított javaslatot a továbblépésre magyarul.\n\nHelyzet:\n{context or 'nincs'}\n\nKérdés:\n{question}", cancellation)
            return {"status": "completed", "result": {"model": response["model"], "advice": response["text"]}}
        except AppError as error:
            return {"status": "error", "result": {"code": error.code, "message": error.message}}

    async def consensus(self, question: str, context: str, cancellation: asyncio.Event) -> dict[str, Any]:
        models = ["models/gemini-3.8-flash", "models/gemini-3.5-flash", "models/gemini-3.1-pro-preview"]
        prompt = f"Válaszolj önállóan és indokolva magyarul a kérdésre; ellenőrizd keresővel vagy kóddal, ahol lehet.\n\nKontextus:\n{context or 'nincs'}\n\nKérdés:\n{question}"
        settled = await asyncio.gather(*(self._interact(model, prompt, cancellation) for model in models), return_exceptions=True)
        answers = [entry for entry in settled if isinstance(entry, dict)]
        failures = [str(entry) for entry in settled if isinstance(entry, Exception)]
        if not answers:
            return {"status": "error", "result": {"code": "SUBAGENT_FAILED", "message": "Egyik modell sem válaszolt.", "failures": failures}}
        source = "\n\n".join(f"### {index + 1}. válasz ({entry['model']})\n{entry['text']}" for index, entry in enumerate(answers))
        try:
            synthesis = await self._interact("models/gemini-3.1-pro-preview", f"Több modell független válaszát kaptad. Hasonlítsd össze őket, jelezd az eltéréseket, majd adj megalapozott konszenzusválaszt és bizonyossági szintet magyarul.\n\nKérdés:\n{question}\n\n{source}", cancellation)
            return {"status": "completed", "result": {"consensus": synthesis["text"], "answers": [{"model": item["model"], "answer": item["text"]} for item in answers], "failures": failures}}
        except AppError as error:
            return {"status": "completed", "result": {"consensus": None, "synthesis_error": error.message, "answers": [{"model": item["model"], "answer": item["text"]} for item in answers], "failures": failures}}


class CloudBrowser:
    def __init__(self, config: Settings, db: Supabase) -> None:
        self.config = config
        self.db = db
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=20), follow_redirects=False, http2=True)
        self.owned: set[str] = set()

    async def close(self) -> None:
        await self.client.aclose()

    async def shutdown(self) -> None:
        await self.stop_all()
        await self.client.aclose()

    def _key(self) -> str:
        key = self.config.secret("browser_use_api_key")
        if not key:
            raise AppError("CONFIGURATION", "A BROWSER_USE_API_KEY nincs beállítva.", 503)
        return key

    async def _call(self, path: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = await self.client.request(method, f"{BROWSER_USE_URL}{path}", headers={"X-Browser-Use-API-Key": self._key()}, json=payload)
        except httpx.HTTPError as error:
            raise AppError("CLOUD_BROWSER_FAILED", "A Browser Use szolgáltatás nem érhető el.", 502) from error
        if not response.is_success:
            raise AppError("CLOUD_BROWSER_FAILED", f"Browser Use HTTP {response.status_code}: {response.text[:500]}", 502)
        if len(response.content) > MAX_PROVIDER_BYTES:
            raise AppError("RESPONSE_TOO_LARGE", "A válasz meghaladja a bájthatárt.", 502)
        return response.json() if response.content else {}

    async def start(self, country: str, timeout_minutes: int) -> dict[str, Any]:
        try:
            result = await self._call("", "POST", {"proxyCountryCode": country or "us", "timeout": min(240, max(1, timeout_minutes)), "allowResizing": True, "enableRecording": True, "browserScreenWidth": 1280, "browserScreenHeight": 1029})
            browser_id = result.get("id")
            cdp_url = result.get("cdpUrl")
            if not isinstance(browser_id, str) or not isinstance(cdp_url, str):
                raise AppError("CLOUD_BROWSER_FAILED", "A böngésző nem adott vissza CDP címet.", 502)
            self.owned.add(browser_id)
            return {"status": "completed", "result": {"browser_id": browser_id, "cdp_url": cdp_url, "live_url": result.get("liveUrl"), "usage": "Csatlakozz a VM-ből Playwrighttal a cdp_url címre. A munka végén hívd a cloud_browser_stop eszközt."}}
        except AppError as error:
            return {"status": "error", "result": {"code": error.code, "message": error.message}}

    async def stop(self, browser_id: str) -> dict[str, Any]:
        try:
            location = f"/{quote(browser_id, safe='')}"
            await self._call(location, "PATCH", {"action": "stop"})
            self.owned.discard(browser_id)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                state = await self._call(location, "GET")
                if isinstance(state.get("recordingUrl"), str):
                    return {"status": "completed", "result": {"browser_id": browser_id, "stopped": True, "recording_url": state["recordingUrl"]}}
                if state.get("recordingAvailable") is False:
                    return {"status": "completed", "result": {"browser_id": browser_id, "stopped": True, "recording_url": None}}
                await asyncio.sleep(2)
            return {"status": "completed", "result": {"browser_id": browser_id, "stopped": True, "recording_url": None, "message": "A felvétel még feldolgozás alatt van."}}
        except AppError as error:
            return {"status": "error", "result": {"code": error.code, "message": error.message}}

    async def stop_all(self) -> None:
        await asyncio.gather(*(self.stop(browser_id) for browser_id in list(self.owned)), return_exceptions=True)

    async def register(self, job_id: str, result: dict[str, Any]) -> str:
        rows = await self.db.admin_insert("chat_browser_sessions", {"job_id": job_id, "browser_id": result["browser_id"], "cdp_url": result["cdp_url"], "live_url": result.get("live_url"), "agent_busy": True})
        if not rows or not isinstance(rows[0].get("id"), str):
            raise AppError("DATABASE_FAILURE", "A böngészőnézet mentése sikertelen.", 502)
        return rows[0]["id"]

    async def mark_stopped(self, browser_id: str) -> None:
        await self.db.admin_update("chat_browser_sessions", {"status": "stopped", "control_mode": "agent", "agent_busy": False}, [eq("browser_id", browser_id)])

    async def set_agent_busy(self, job_id: str, busy: bool) -> None:
        await self.db.admin_update("chat_browser_sessions", {"agent_busy": busy}, [eq("job_id", job_id), eq("status", "active")])
        if not busy:
            await self.db.admin_update("chat_browser_sessions", {"control_mode": "human"}, [eq("job_id", job_id), eq("status", "active"), eq("control_mode", "requested")])

    async def wait_for_control(self, job_id: str, cancellation: asyncio.Event) -> None:
        while not cancellation.is_set():
            value = await self.db.rpc("claim_chat_browser_step", {"_job_id": job_id})
            if value is True:
                return
            await asyncio.sleep(0.7)
        raise AppError("CANCELLED", "A feladat megszakadt.", 499)


READ_PAGE_SCRIPT = """import base64,json,os,sys
p=sys.argv[1]
o=int(sys.argv[2])
with open(p,'rb') as f:
 f.seek(o)
 b=f.read({page})
 size=f.seek(0,2)
print(json.dumps({{'data_base64':base64.b64encode(b).decode('ascii'),'bytes_read':len(b),'next_offset':o+len(b),'eof':o+len(b)>=size,'total_bytes':size}}))"""


CAPTURE_SCRIPT = """import base64,json,os,sys
paths = sys.argv[1:]
def read(location):
 with open(location, "rb") as handle:
  data = handle.read(2001)
 size = os.path.getsize(location)
 if len(data) > 2000:
  data = data[:2000]
 try:
  text = data.decode("utf-8")
 except UnicodeDecodeError:
  text = ""
 return text, base64.b64encode(data).decode("ascii"), size, size > len(data)
left, left_encoded, left_size, left_cut = read(paths[0])
right, right_encoded, right_size, right_cut = read(paths[1])
print(json.dumps({"stdout": left, "stderr": right, "stdout_base64_preview": left_encoded, "stderr_base64_preview": right_encoded, "stdout_bytes": left_size, "stderr_bytes": right_size, "stdout_truncated": left_cut, "stderr_truncated": right_cut}))"""


class Sandbox:
    def __init__(self, config: Settings, cancellation: asyncio.Event) -> None:
        self.config = config
        self.cancellation = cancellation
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=20), follow_redirects=False, http2=True)
        self.session_id: str | None = None
        self.browser_id: str | None = None
        self.owned_sessions: set[str] = set()
        self.disposed = False
        self.created_at = time.monotonic()
        self.deadline = time.monotonic() + config.max_runtime_seconds

    def _remaining(self) -> None:
        if self.cancellation.is_set():
            raise AppError("CANCELLED", "A feladat megszakadt.", 499)
        if time.monotonic() >= self.deadline:
            raise AppError("RUNTIME_LIMIT", "A futásidő lejárt.", 408)

    async def close(self) -> None:
        if self.disposed:
            return
        self.disposed = True
        failures: list[str] = []
        if self.browser_id:
            try:
                await self._api(f"/v1/browser/sessions/{self.browser_id}", "DELETE", None, timeout=CLEANUP_TIMEOUT, force=True)
            except AppError:
                failures.append("browser")
        for session_id in self.owned_sessions:
            try:
                await self._api("/kill", "POST", {"session_id": session_id}, timeout=CLEANUP_TIMEOUT, force=True)
            except AppError:
                failures.append(f"vm:{session_id}")
        await self.client.aclose()
        if failures:
            raise AppError("CLEANUP_FAILED", f"Nem ellenőrzött erőforrás-felszabadítás: {', '.join(failures)}", 502)

    def _key(self) -> str:
        key = self.config.secret("instavm_api_key")
        if not key:
            raise AppError("SANDBOX_DISABLED", "Az InstaVM eszközök le vannak tiltva.", 503)
        return key

    async def _api(self, path: str, method: str, payload: dict[str, Any] | None, timeout: float = 45, force: bool = False) -> dict[str, Any]:
        if not force and self.cancellation.is_set():
            raise AppError("CANCELLED", "A feladat megszakadt.", 499)
        try:
            response = await self.client.request(method, f"{INSTAVM_URL}{path}", headers={"X-API-Key": self._key()}, json=payload, timeout=timeout)
        except httpx.TimeoutException as error:
            raise AppError("UNKNOWN_OUTCOME", "A sandbox műveletének kimenetele időtúllépés miatt ismeretlen.", 504) from error
        except httpx.HTTPError as error:
            raise AppError("UNKNOWN_OUTCOME", "A sandbox műveletének kimenetele hálózati hiba miatt ismeretlen.", 502) from error
        if not response.is_success:
            if response.status_code in {401, 403}:
                raise AppError("PROVIDER_AUTH", "A sandbox hitelesítése vagy jogosultsága sikertelen.", 502)
            if response.status_code == 429:
                raise AppError("PROVIDER_RATE_LIMIT", "A sandbox szolgáltató korlátozta a kéréseket.", 503)
            if 400 <= response.status_code < 500:
                raise AppError("PROVIDER_REJECTED", f"A sandbox elutasította a kérést: HTTP {response.status_code}", 502)
            raise AppError("PROVIDER_HTTP", f"Sandbox HTTP {response.status_code}", 502)
        if len(response.content) > MAX_PROVIDER_BYTES:
            raise AppError("RESPONSE_TOO_LARGE", "A sandbox válasza meghaladja a bájthatárt.", 502)
        try:
            data = response.json() if response.content else {}
        except ValueError as error:
            raise AppError("PROTOCOL_ERROR", "A sandbox nem JSON választ adott.", 502) from error
        if not isinstance(data, dict):
            raise AppError("PROTOCOL_ERROR", "A sandbox válasza nem objektum.", 502)
        return data

    async def _session(self) -> str:
        if self.disposed:
            raise AppError("SESSION_TERMINATED", "A VM már lezárult.", 409)
        if self.session_id:
            return self.session_id
        reserved = str(uuid.uuid4())
        domains = sorted(set(self.config.sandbox_domains) | {"releases.nixos.org", "cache.nixos.org", "github.com", "api.github.com", "codeload.github.com"})
        payload = await self._api("/v1/vms?wait=true", "POST", {"session_id": reserved, "vm_lifetime_seconds": min(22500, self.config.max_runtime_seconds + 900), "egress_policy": {"allow_http": False, "allow_https": True, "allow_package_managers": True, "allowed_domains": domains, "allowed_cidrs": []}}, timeout=90)
        identifier = payload.get("session_id")
        if not isinstance(identifier, str) or identifier != reserved or payload.get("status") not in {"running", "active"}:
            raise AppError("PROVIDER_PROTOCOL", "A VM létrehozási válasza érvénytelen.", 502)
        self.session_id = identifier
        self.owned_sessions.add(identifier)
        return identifier

    async def _raw(self, command: str, timeout: int) -> dict[str, Any]:
        session_id = await self._session()
        wait = min(EXECUTE_WAIT_MAX_SECONDS, max(1, int(timeout)))
        response = await self._api("/execute", "POST", {"session_id": session_id, "command": command, "language": "bash", "timeout": wait}, timeout=(wait + 10) * 1.0)
        if response.get("success") is False or response.get("error"):
            raise AppError("PROVIDER_EXECUTION", "Az InstaVM végrehajtási hibát jelentett.", 502)
        return response

    async def _checked(self, command: str, timeout: int = 60) -> dict[str, Any]:
        marker = f"__CHAT_EXIT_{uuid.uuid4().hex}__"
        raw = await self._raw(f"set +e\n(\n{command}\n)\nresult=$?\nprintf '\\n{marker}%s\\n' \"$result\"\nexit 0", timeout)
        stdout = raw.get("stdout", raw.get("output"))
        stderr = raw.get("stderr", "")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise AppError("PROTOCOL_ERROR", "A végrehajtási válasz nem tartalmaz szöveges kimenetet.", 502)
        marker_index = stdout.rfind(f"\n{marker}")
        if marker_index < 0:
            raise AppError("EXECUTION_UNKNOWN", "Nem érkezett végrehajtási zárójelzés.", 502)
        code_text = stdout[marker_index + len(marker) + 1 :].strip()
        if not code_text.isdigit():
            raise AppError("PROTOCOL_ERROR", "Érvénytelen végrehajtási kilépési kód.", 502)
        return {"stdout": stdout[:marker_index], "stderr": stderr, "exit_code": int(code_text)}

    async def _append_bytes(self, target: str, content: bytes) -> dict[str, int] | None:
        destination = shell_quote(target)
        for start in range(0, len(content), MAX_FILE_CHUNK_BYTES):
            encoded = base64.b64encode(content[start : start + MAX_FILE_CHUNK_BYTES]).decode("ascii")
            written = await self._checked(f"printf %s {shell_quote(encoded)} | base64 -d >> {destination}", 30)
            if written["exit_code"] != 0:
                return written
        return None

    async def _write_bytes(self, path: str, content: bytes) -> dict[str, Any]:
        if len(content) > MAX_FILE_BYTES:
            raise AppError("INVALID_ARGUMENT", "A fájl bájtban túl nagy.", 400)
        temporary = f"{path}.chat-{uuid.uuid4().hex}.tmp"
        initialized = await self._checked(f"mkdir -p -- $(dirname -- {shell_quote(path)}) && : > {shell_quote(temporary)}", 30)
        if initialized["exit_code"] != 0:
            return {"status": "error", "result": initialized}
        try:
            transfer = await self._append_bytes(temporary, content)
            if transfer is not None:
                return {"status": "error", "result": transfer}
            digest = hashlib.sha256(content).hexdigest()
            checked = await self._checked(f"test \"$(wc -c < {shell_quote(temporary)})\" -eq {len(content)} && test \"$(sha256sum -- {shell_quote(temporary)} | cut -d ' ' -f 1)\" = {shell_quote(digest)} && mv -f -- {shell_quote(temporary)} {shell_quote(path)}", 30)
            if checked["exit_code"] != 0:
                return {"status": "error", "result": checked}
            return {"status": "completed", "result": {"bytes_written": len(content), "sha256": digest, "path": path}}
        finally:
            await self._checked(f"rm -f -- {shell_quote(temporary)}", 20)

    async def background(self, script: str, interpreter: str, timeout_seconds: int) -> dict[str, Any]:
        identifier = uuid.uuid4().hex
        directory = f"/tmp/chat-job-{identifier}"
        path = shell_quote(directory)
        init = await self._checked(f"umask 077; mkdir -m 700 {path} && : > {path}/job", 30)
        if init["exit_code"] != 0:
            raise AppError("EXECUTION_FAILED", "A folyamatkönyvtár létrehozása sikertelen.", 502)
        transfer = await self._append_bytes(f"{directory}/job", script.encode("utf-8"))
        if transfer is not None:
            raise AppError("EXECUTION_FAILED", "A távoli program átvitele sikertelen.", 502)
        invocation = "bash -lc 'exec python3 \"$0\"' \"$1/job\"" if interpreter == "python" else "bash -l \"$1/job\""
        runner = "\n".join((
            "set -e",
            f"chmod 600 {path}/job",
            f"cat > {path}/runner.sh <<'CHAT_RUNNER'",
            "set +e",
            f"timeout --foreground --signal=TERM --kill-after=5s {timeout_seconds}s {invocation} > \"$1/stdout\" 2> \"$1/stderr\"",
            "rc=$?",
            "printf '%s\\n' \"$rc\" > \"$1/exit.tmp\"",
            "mv \"$1/exit.tmp\" \"$1/exit\"",
            "CHAT_RUNNER",
            f"chmod 700 {path}/runner.sh",
            f"setsid nohup bash {path}/runner.sh {path} </dev/null >/dev/null 2>&1 &",
            f"printf '%s\\n' \"$!\" > {path}/pid",
        ))
        launch = await self._checked(runner, 30)
        if launch["exit_code"] != 0:
            raise AppError("EXECUTION_FAILED", "A folyamat indítása sikertelen.", 502)
        limit = min(self.deadline, time.monotonic() + timeout_seconds + 10)
        settled = False
        try:
            while True:
                self._remaining()
                sample = await self._checked(f"if test -f {path}/exit; then printf 'FINISHED\\n'; cat {path}/exit; else printf 'RUNNING\\n'; fi", 20)
                if sample["exit_code"] != 0:
                    raise AppError("EXECUTION_UNKNOWN", "A háttérfolyamat státusza nem olvasható.", 502)
                if not sample["stdout"].startswith("FINISHED\n"):
                    if time.monotonic() >= limit:
                        raise AppError("EXECUTION_UNKNOWN", "A távoli folyamat nem zárult le ellenőrizhetően.", 504)
                    await asyncio.sleep(min(1.0, max(0.0, limit - time.monotonic())))
                    continue
                value = sample["stdout"][9:].strip()
                if not value.isdigit():
                    raise AppError("PROTOCOL_ERROR", "Hibás folyamatkilépési kód.", 502)
                status = int(value)
                logs = await self._checked(f"python3 -c {shell_quote(CAPTURE_SCRIPT)} {path}/stdout {path}/stderr", 25)
                if logs["exit_code"] != 0:
                    raise AppError("EXECUTION_UNKNOWN", "A folyamatnapló nem olvasható.", 502)
                try:
                    details = json.loads(logs["stdout"])
                except json.JSONDecodeError as error:
                    raise AppError("PROTOCOL_ERROR", "Hibás folyamatnapló.", 502) from error
                if not isinstance(details, dict) or any(key not in details for key in ("stdout", "stderr", "stdout_base64_preview", "stderr_base64_preview", "stdout_bytes", "stderr_bytes", "stdout_truncated", "stderr_truncated")):
                    raise AppError("PROTOCOL_ERROR", "Hiányos folyamatnapló.", 502)
                settled = True
                return {
                    "stdout": str(details["stdout"]),
                    "stderr": str(details["stderr"]),
                    "stdout_base64_preview": str(details["stdout_base64_preview"]),
                    "stderr_base64_preview": str(details["stderr_base64_preview"]),
                    "stdout_bytes": int(details["stdout_bytes"]),
                    "stderr_bytes": int(details["stderr_bytes"]),
                    "stdout_truncated": bool(details["stdout_truncated"]),
                    "stderr_truncated": bool(details["stderr_truncated"]),
                    "stdout_file": f"{directory}/stdout",
                    "stderr_file": f"{directory}/stderr",
                    "exit_code": status,
                    "timed_out": status in {124, 137},
                }
        finally:
            if not settled and self.session_id:
                terminate = f'if test -f {path}/pid; then pid=$(cat {path}/pid); if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; fi; fi'
                try:
                    await self._api("/execute", "POST", {"session_id": self.session_id, "command": terminate, "language": "bash", "timeout": 10}, timeout=CLEANUP_TIMEOUT, force=True)
                except AppError:
                    pass

    async def exec(self, script: str, interpreter: str, timeout: int) -> dict[str, Any]:
        if timeout < 1 or timeout > 1800:
            raise AppError("INVALID_ARGUMENT", "A végrehajtási határidő érvénytelen.", 400)
        result = await self.background(script, "python" if interpreter == "python" else "bash", timeout)
        return {"status": "completed" if result["exit_code"] == 0 else "error", "result": result}

    async def install(self, packages: list[str]) -> dict[str, Any]:
        if not packages or len(packages) > 24 or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.+-]{0,99}", item) for item in packages):
            raise AppError("INVALID_ARGUMENT", "Érvénytelen Nix csomaglista.", 400)
        revision = self.config.chat_nixpkgs_rev
        version = self.config.chat_nix_version
        checksum = self.config.chat_nix_installer_sha256
        if not re.fullmatch(r"[a-f0-9]{40}", revision) or not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
            raise AppError("CONFIGURATION", "A rögzített Nix telepítési konfiguráció hiányzik.", 503)
        references = " ".join(shell_quote(f"github:NixOS/nixpkgs/{revision}#{item}") for item in packages)
        installer = f"https://releases.nixos.org/nix/nix-{version}/install"
        script = f"set -euo pipefail\nif ! command -v nix >/dev/null 2>&1; then sudo install -d -m 755 -o \"$(id -u)\" -g \"$(id -g)\" /nix; curl --fail --location --show-error --silent {shell_quote(installer)} --output /tmp/chat-nix-installer.sh; printf '%s  %s\\n' {shell_quote(checksum.lower())} /tmp/chat-nix-installer.sh | sha256sum --check --status; sh /tmp/chat-nix-installer.sh --no-daemon; . \"$HOME/.nix-profile/etc/profile.d/nix.sh\"; fi\nnix --extra-experimental-features 'nix-command flakes' profile add {references}"
        return await self.exec(script, "bash", 1800)

    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        if "\x00" in path:
            raise AppError("INVALID_ARGUMENT", "A fájlútvonal NUL karaktert tartalmaz.", 400)
        return await self._write_bytes(path, content.encode("utf-8"))

    async def read_file(self, path: str, offset: int) -> dict[str, Any]:
        if offset < 0:
            raise AppError("INVALID_ARGUMENT", "A fájl eltolása nemnegatív egész kell legyen.", 400)
        script = READ_PAGE_SCRIPT.format(page=MAX_PAGE_BYTES)
        response = await self._checked(f"python3 -c {shell_quote(script)} {shell_quote(path)} {offset}", 30)
        if response["exit_code"] != 0:
            return {"status": "error", "result": response}
        try:
            result = json.loads(response["stdout"])
        except json.JSONDecodeError as error:
            raise AppError("PROTOCOL_ERROR", "Hibás fájllapozási válasz.", 502) from error
        if not isinstance(result, dict) or not isinstance(result.get("data_base64"), str) or not isinstance(result.get("bytes_read"), int) or not isinstance(result.get("next_offset"), int) or not isinstance(result.get("eof"), bool) or not isinstance(result.get("total_bytes"), int) or result["bytes_read"] > MAX_PAGE_BYTES or result["next_offset"] != offset + result["bytes_read"]:
            raise AppError("PROTOCOL_ERROR", "Érvénytelen fájllapozási válasz.", 502)
        return {"status": "completed", "result": result}

    async def expose_port(self, port: int) -> dict[str, Any]:
        if not 1 <= port <= 65535:
            raise AppError("INVALID_ARGUMENT", "A port érvénytelen.", 400)
        probe_program = 'import socket,sys;s=socket.socket();s.settimeout(3);s.connect(("127.0.0.1",int(sys.argv[1])));s.close()'
        probe = await self._checked(f"python3 -c {shell_quote(probe_program)} {port}", 15)
        if probe["exit_code"] != 0:
            return {"status": "error", "result": {"message": "A TCP port nem elérhető a VM-ben.", "port": port}}
        session = await self._session()
        result = await self._api(f"/v1/sessions/app-url/{session}?port={port}", "GET", None)
        url = result.get("url") or result.get("app_url")
        if not isinstance(url, str) or not re.match(r"^https://[^\s/]+", url):
            raise AppError("PROVIDER_PROTOCOL", "A szolgáltató nem küldött HTTPS port URL-t.", 502)
        return {"status": "completed", "result": {"url": url, "port": port, "retention": "A VM és az URL a feladat befejezéséig élhet.", "session_age_ms": round((time.monotonic() - self.created_at) * 1000)}}

    async def browse(self, url: str) -> dict[str, Any]:
        await validate_browser_url(url, self.config.browser_hosts)
        if not self.browser_id:
            created = await self._api("/v1/browser/sessions/", "POST", {"viewport_width": 1280, "viewport_height": 900})
            if not isinstance(created.get("session_id"), str):
                raise AppError("PROVIDER_PROTOCOL", "A böngésző munkamenete érvénytelen.", 502)
            self.browser_id = created["session_id"]
        navigation = await self._api("/v1/browser/interactions/navigate", "POST", {"url": url, "wait_timeout": 30000, "session_id": self.browser_id}, timeout=40)
        if navigation.get("error") or navigation.get("success") is False:
            raise AppError("BROWSER_NAVIGATION", "A böngésző navigációja sikertelen.", 502)
        result = await self._api("/v1/browser/interactions/content", "POST", {"session_id": self.browser_id, "include_interactive": False, "include_anchors": True}, timeout=40)
        if result.get("error") or result.get("success") is False or not isinstance(result.get("text", result.get("content")), str):
            raise AppError("BROWSER_CONTENT", "A böngésző nem adott olvasható tartalmat.", 502)
        serialized = json.dumps(result, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > MAX_RESULT_BYTES - 2000:
            location = f"/tmp/chat-browser-{uuid.uuid4().hex}.json"
            saved = await self._write_bytes(location, serialized.encode("utf-8"))
            if saved["status"] != "completed":
                return saved
            return {"status": "completed", "result": {"artifact_path": location, "retrieval": "read_file", "preview": serialized[:1000], "original_bytes": len(serialized.encode('utf-8'))}}
        return {"status": "completed", "result": result}


async def validate_browser_url(value: str, allowed_hosts: set[str]) -> None:
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or parsed.username or parsed.password or (parsed.port and parsed.port != 443) or hostname not in allowed_hosts:
        raise AppError("NETWORK_DENIED", "Ez a böngésző URL nem engedélyezett.", 403)
    try:
        entries = await asyncio.get_running_loop().run_in_executor(None, lambda: socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM))
    except socket.gaierror as error:
        raise AppError("DNS_RESOLUTION_FAILED", "A böngésző gazdagépe nem oldható fel.", 502) from error
    if not entries:
        raise AppError("NETWORK_DENIED", "A böngésző cím nem oldható fel.", 403)
    for entry in entries:
        address = ipaddress.ip_address(entry[4][0])
        if not address.is_global:
            raise AppError("NETWORK_DENIED", "Privát vagy fenntartott böngésző cím.", 403)
