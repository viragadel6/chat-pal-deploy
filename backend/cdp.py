from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets

from .errors import AppError


async def browser_command(url: str, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
    sequence = 0
    try:
        async with websockets.connect(url, open_timeout=15, close_timeout=5, max_size=2**22) as socket:
            async def receive() -> None:
                while True:
                    raw = await socket.recv()
                    message = json.loads(raw)
                    identifier = message.get("id")
                    future = pending.pop(identifier, None)
                    if future is None:
                        continue
                    if "error" in message:
                        future.set_exception(AppError("CDP_FAILURE", str(message["error"].get("message", "CDP hiba.")), 502))
                    else:
                        future.set_result(message.get("result") or {})

            receiver = asyncio.create_task(receive())

            async def command(method: str, params: dict[str, Any] | None = None, session_id: str | None = None) -> dict[str, Any]:
                nonlocal sequence
                sequence += 1
                loop = asyncio.get_running_loop()
                future: asyncio.Future[dict[str, Any]] = loop.create_future()
                pending[sequence] = future
                message: dict[str, Any] = {"id": sequence, "method": method, "params": params or {}}
                if session_id:
                    message["sessionId"] = session_id
                await socket.send(json.dumps(message, separators=(",", ":")))
                return await asyncio.wait_for(future, 12)

            try:
                targets = await command("Target.getTargets")
                pages = [target for target in targets.get("targetInfos", []) if target.get("type") == "page" and not str(target.get("url", "")).startswith("chrome-extension:")]
                if not pages:
                    raise AppError("CDP_NO_PAGE", "Nincs megnyitott böngészőlap.", 502)
                page = pages[0]
                attached = await command("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True})
                session_id = str(attached["sessionId"])
                if action == "frame":
                    response = await command("Page.captureScreenshot", {"format": "jpeg", "quality": 75, "captureBeyondViewport": False}, session_id)
                    response["url"] = page.get("url", "")
                    return response
                if payload is None:
                    raise AppError("INVALID_ARGUMENT", "Hiányzó böngészőművelet.", 400)
                kind = payload.get("kind")
                if kind == "text":
                    await command("Input.insertText", {"text": payload["text"]}, session_id)
                elif kind == "key":
                    await command("Input.dispatchKeyEvent", {"type": payload["type"], "key": payload["key"], "code": payload["code"], "windowsVirtualKeyCode": payload["keyCode"], "nativeVirtualKeyCode": payload["keyCode"], "modifiers": payload["modifiers"]}, session_id)
                else:
                    parameters = {key: value for key, value in payload.items() if key != "kind"}
                    if kind == "wheel":
                        parameters["type"] = "mouseWheel"
                    await command("Input.dispatchMouseEvent", parameters, session_id)
                return {"ok": True}
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    except AppError:
        raise
    except Exception as error:
        raise AppError("CDP_UNAVAILABLE", "A böngésző kapcsolata nem érhető el.", 502) from error
