from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import AppError

MAX_TEXT_LENGTH = 400000
MAX_HISTORY_LENGTH = 300


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class BrowserMouseInput(BaseModel):
    kind: Literal["mouse"]
    type: Literal["mousePressed", "mouseReleased", "mouseMoved"]
    x: float
    y: float
    button: Literal["none", "left", "middle", "right", "back", "forward"]
    clickCount: int = Field(ge=0, le=3)
    modifiers: int = Field(ge=0, le=15)


class BrowserWheelInput(BaseModel):
    kind: Literal["wheel"]
    type: Literal["mouseWheel"]
    x: float
    y: float
    deltaX: float = Field(ge=-10000, le=10000)
    deltaY: float = Field(ge=-10000, le=10000)
    modifiers: int = Field(ge=0, le=15)


class BrowserKeyInput(BaseModel):
    kind: Literal["key"]
    type: Literal["keyDown", "keyUp"]
    key: str = Field(min_length=1, max_length=40)
    code: str = Field(min_length=1, max_length=40)
    keyCode: int = Field(ge=0, le=255)
    modifiers: int = Field(ge=0, le=15)


class BrowserTextInput(BaseModel):
    kind: Literal["text"]
    text: str = Field(min_length=1, max_length=10000)


def invalid(message: str) -> AppError:
    return AppError("INVALID_REQUEST", message, 400)


def validate_history(payload: Any) -> list[ChatMessage]:
    if not isinstance(payload, dict):
        raise invalid("Hiányzó vagy hibás üzenetlista.")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= MAX_HISTORY_LENGTH:
        raise invalid("Hiányzó vagy hibás üzenetlista.")
    messages: list[ChatMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise invalid("Az üzenet szerepköre nem megengedett.")
        role = item.get("role")
        if not isinstance(role, str) or role not in ("user", "assistant"):
            raise invalid("Az üzenet szerepköre nem megengedett.")
        if any(key not in ("role", "content") for key in item):
            raise invalid("Kliens oldali utasítás- vagy eszköztörténet nem fogadható el.")
        content = item.get("content")
        if not isinstance(content, str) or len(content) > MAX_TEXT_LENGTH or not content.strip():
            raise invalid("Az üzenet tartalma hibás vagy túl nagy.")
        messages.append(ChatMessage(role=role, content=content))
    if messages[-1].role != "user":
        raise invalid("A legutolsó üzenetnek felhasználói feladatnak kell lennie.")
    return messages
