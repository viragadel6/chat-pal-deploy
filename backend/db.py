from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from .config import Settings
from .errors import AppError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_opaque_key(value: str) -> bool:
    return value.startswith("sb_publishable_") or value.startswith("sb_secret_")


class Supabase:
    def __init__(self, config: Settings) -> None:
        self.config = config
        url = str(config.supabase_url).rstrip("/") if config.supabase_url else ""
        self.base_url = url
        self.service_key = config.secret("supabase_service_role_key")
        self.public_key = config.secret("supabase_publishable_key") or config.secret("supabase_anon_key")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=15), follow_redirects=False, http2=True)

    async def close(self) -> None:
        await self.client.aclose()

    def _base_headers(self, key: str, authorization: str | None = None) -> dict[str, str]:
        headers = {"apikey": key, "Accept": "application/json"}
        if authorization:
            headers["Authorization"] = authorization
        elif not is_opaque_key(key):
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _admin_headers(self) -> dict[str, str]:
        if not self.base_url or not self.service_key:
            raise AppError("CONFIGURATION", "A Supabase szerveroldali konfiguráció hiányzik.", 503)
        return self._base_headers(self.service_key)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | list[tuple[str, str]] | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
        key: str | None = None,
        authorization: str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        selected_key = key or self.service_key
        if not self.base_url or not selected_key:
            raise AppError("CONFIGURATION", "A Supabase konfiguráció hiányzik.", 503)
        all_headers = self._base_headers(selected_key, authorization)
        all_headers.update(headers or {})
        try:
            response = await self.client.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                json=body,
                headers=all_headers,
                timeout=timeout,
            )
        except httpx.HTTPError as error:
            raise AppError("DATABASE_UNAVAILABLE", "Az adatbázis-szolgáltatás nem érhető el.", 502) from error
        if response.status_code in {401, 403} and path.startswith("/auth/"):
            raise AppError("UNAUTHORIZED", "Bejelentkezés szükséges.", 401)
        if response.status_code >= 400:
            detail = response.text[:1000]
            try:
                parsed = response.json()
                detail = str(parsed.get("message") or parsed.get("hint") or parsed.get("details") or detail)
            except (ValueError, AttributeError):
                pass
            raise AppError("DATABASE_FAILURE", f"Adatbázis hiba: {detail}", 502)
        return response

    async def admin_select(
        self,
        table: str,
        select: str = "*",
        filters: list[tuple[str, str, str]] | None = None,
        order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = [("select", select)]
        for column, operator, value in filters or []:
            params.append((column, f"{operator}.{value}"))
        if order:
            params.append(("order", order))
        if limit is not None:
            params.append(("limit", str(limit)))
        if offset is not None:
            params.append(("offset", str(offset)))
        response = await self._request("GET", f"/rest/v1/{table}", params=params)
        data = response.json()
        if not isinstance(data, list):
            raise AppError("DATABASE_PROTOCOL", "Az adatbázis érvénytelen választ adott.", 502)
        return data

    async def admin_single(
        self,
        table: str,
        select: str = "*",
        filters: list[tuple[str, str, str]] | None = None,
    ) -> dict[str, Any] | None:
        rows = await self.admin_select(table, select, filters, limit=1)
        return rows[0] if rows else None

    async def admin_insert(self, table: str, values: dict[str, Any] | list[dict[str, Any]], returning: str = "representation") -> list[dict[str, Any]]:
        response = await self._request(
            "POST",
            f"/rest/v1/{table}",
            body=values,
            headers={"Content-Type": "application/json", "Prefer": f"return={returning}"},
        )
        if returning == "minimal" or not response.content:
            return []
        parsed = response.json()
        return parsed if isinstance(parsed, list) else [parsed]

    async def admin_upsert(
        self,
        table: str,
        values: dict[str, Any] | list[dict[str, Any]],
        conflict: str,
        returning: str = "representation",
    ) -> list[dict[str, Any]]:
        response = await self._request(
            "POST",
            f"/rest/v1/{table}",
            params={"on_conflict": conflict},
            body=values,
            headers={"Content-Type": "application/json", "Prefer": f"resolution=merge-duplicates,return={returning}"},
        )
        if returning == "minimal" or not response.content:
            return []
        parsed = response.json()
        return parsed if isinstance(parsed, list) else [parsed]

    async def admin_update(
        self,
        table: str,
        values: dict[str, Any],
        filters: list[tuple[str, str, str]],
        select: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = []
        for column, operator, value in filters:
            params.append((column, f"{operator}.{value}"))
        if select:
            params.append(("select", select))
        response = await self._request(
            "PATCH",
            f"/rest/v1/{table}",
            params=params,
            body=values,
            headers={"Content-Type": "application/json", "Prefer": "return=representation" if select else "return=minimal"},
        )
        if not select or not response.content:
            return []
        parsed = response.json()
        return parsed if isinstance(parsed, list) else [parsed]

    async def admin_count(self, table: str, filters: list[tuple[str, str, str]] | None = None) -> int:
        params: list[tuple[str, str]] = [("select", "id")]
        for column, operator, value in filters or []:
            params.append((column, f"{operator}.{value}"))
        response = await self._request("HEAD", f"/rest/v1/{table}", params=params, headers={"Range": "0-0", "Range-Unit": "items", "Prefer": "count=exact"})
        value = response.headers.get("content-range", "").rpartition("/")[2]
        if not value.isdigit():
            raise AppError("DATABASE_PROTOCOL", "Az adatbázis nem adta meg a sorok számát.", 502)
        return int(value)

    async def admin_delete(self, table: str, filters: list[tuple[str, str, str]], select: str | None = None) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = []
        for column, operator, value in filters:
            params.append((column, f"{operator}.{value}"))
        if select:
            params.append(("select", select))
        response = await self._request(
            "DELETE",
            f"/rest/v1/{table}",
            params=params,
            headers={"Prefer": "return=representation" if select else "return=minimal"},
        )
        if not select or not response.content:
            return []
        parsed = response.json()
        return parsed if isinstance(parsed, list) else [parsed]

    async def rpc(self, function: str, values: dict[str, Any]) -> Any:
        response = await self._request(
            "POST",
            f"/rest/v1/rpc/{function}",
            body=values,
            headers={"Content-Type": "application/json"},
        )
        if not response.content:
            return None
        return response.json()

    async def user_id(self, token: str) -> str:
        if not self.base_url or not self.public_key:
            raise AppError("CONFIGURATION", "A Supabase hitelesítési konfiguráció hiányzik.", 503)
        response = await self._request(
            "GET",
            "/auth/v1/user",
            key=self.public_key,
            authorization=f"Bearer {token}",
        )
        try:
            data = response.json()
            identifier = data.get("id")
        except (ValueError, AttributeError) as error:
            raise AppError("AUTH_FAILURE", "A hitelesítési szolgáltatás hibás választ adott.", 502) from error
        if not isinstance(identifier, str):
            raise AppError("UNAUTHORIZED", "Bejelentkezés szükséges.", 401)
        return identifier

    async def user_can_access_session(self, token: str, session_id: str) -> bool:
        if not self.base_url or not self.public_key:
            raise AppError("CONFIGURATION", "A Supabase hitelesítési konfiguráció hiányzik.", 503)
        response = await self._request(
            "GET",
            "/rest/v1/chat_browser_sessions",
            params={"select": "id", "id": f"eq.{session_id}", "limit": "1"},
            key=self.public_key,
            authorization=f"Bearer {token}",
        )
        data = response.json()
        return isinstance(data, list) and bool(data)

    async def user_can_access_conversation(self, token: str, conversation_id: str) -> bool:
        if not self.base_url or not self.public_key:
            raise AppError("CONFIGURATION", "A Supabase hitelesítési konfiguráció hiányzik.", 503)
        response = await self._request(
            "GET",
            "/rest/v1/chat_jobs",
            params={"select": "id", "conversation_id": f"eq.{conversation_id}", "limit": "1"},
            key=self.public_key,
            authorization=f"Bearer {token}",
        )
        data = response.json()
        return isinstance(data, list) and bool(data)


def eq(column: str, value: str | int | bool) -> tuple[str, str, str]:
    normalized = "true" if value is True else "false" if value is False else str(value)
    return column, "eq", normalized


def gt(column: str, value: str | int) -> tuple[str, str, str]:
    return column, "gt", str(value)


def lt(column: str, value: str | int | float) -> tuple[str, str, str]:
    return column, "lt", str(value)


def in_values(column: str, values: list[str]) -> tuple[str, str, str]:
    return column, "in", f"({','.join(values)})"


def is_null(column: str) -> tuple[str, str, str]:
    return column, "is", "null"


def not_null(column: str) -> tuple[str, str, str]:
    return column, "not.is", "null"
