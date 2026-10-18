from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int = 500

    def __str__(self) -> str:
        return self.message


def require(condition: bool, code: str, message: str, status_code: int = 400) -> None:
    if not condition:
        raise AppError(code, message, status_code)
