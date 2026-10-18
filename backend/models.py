from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=400000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    conversation_id: str | None = Field(default=None, max_length=36)
    messages: list[ChatMessage] = Field(min_length=1, max_length=300)


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: dict[str, str]


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    task_id: str
    step: int | None = None
    content: str | None = None
    reasoning_content: str | None = None
    event: dict[str, Any] | None = None
    code: str | None = None
    message: str | None = None
    outcome: str | None = None
    reason: str | None = None
    steps: int | None = None
    tool_calls: int | None = None
    duration_ms: int | None = None
    elapsed_ms: int | None = None


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
