from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import AppError

HOST_RE = re.compile(r"^[a-z0-9.-]+$")


def host_list(value: str, message: str) -> set[str]:
    hosts: set[str] = set()
    for item in value.split(","):
        host = item.strip().lower()
        if not host:
            continue
        if not HOST_RE.fullmatch(host) or host.startswith(".") or host.endswith(".") or ".." in host:
            raise AppError("CONFIGURATION", message, 503)
        hosts.add(host)
    return hosts


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 8000
    public_dir: str = "public"
    app_origin: str | None = None
    app_origins: str = ""
    supabase_url: HttpUrl | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_publishable_key: SecretStr | None = None
    supabase_anon_key: SecretStr | None = None
    orcarouter_api_key: SecretStr | None = None
    instavm_api_key: SecretStr | None = None
    browser_use_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    flymyai_api_key: SecretStr | None = None
    lovable_cron_secret: SecretStr | None = None
    lovable_cron_secret_previous: SecretStr | None = None
    chat_deployment_max_request_ms: int = 0
    chat_sandbox_allowed_domains: str = ""
    chat_browser_allowed_hosts: str = ""
    chat_nixpkgs_rev: str = ""
    chat_nix_version: str = ""
    chat_nix_installer_sha256: str = ""
    worker_lease_seconds: Annotated[int, Field(ge=30, le=600)] = 90
    worker_heartbeat_seconds: Annotated[int, Field(ge=5, le=120)] = 20
    max_runtime_seconds: Annotated[int, Field(ge=60, le=86400)] = 21600

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"development", "test", "staging", "production"}:
            raise ValueError("ENVIRONMENT must be development, test, staging, or production")
        return normalized

    @field_validator("app_origins", "chat_sandbox_allowed_domains", "chat_browser_allowed_hosts")
    @classmethod
    def normalize_csv(cls, value: str) -> str:
        return ",".join(part.strip() for part in value.split(",") if part.strip())

    def secret(self, name: str) -> str | None:
        value = getattr(self, name)
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return None

    @property
    def origins(self) -> set[str]:
        values = {item for item in self.app_origins.split(",") if item}
        if self.app_origin:
            values.add(self.app_origin.rstrip("/"))
        return values

    @property
    def sandbox_domains(self) -> set[str]:
        return host_list(self.chat_sandbox_allowed_domains, "Érvénytelen sandbox gazdagép engedélylista.")

    @property
    def browser_hosts(self) -> set[str]:
        return host_list(self.chat_browser_allowed_hosts, "Érvénytelen böngésző gazdagép engedélylista.")

    def validate_host_lists(self) -> tuple[set[str], set[str]]:
        return host_list(self.chat_sandbox_allowed_domains, "Érvénytelen sandbox gazdagép engedélylista."), host_list(self.chat_browser_allowed_hosts, "Érvénytelen böngésző gazdagép engedélylista.")


@lru_cache
def settings() -> Settings:
    return Settings()
