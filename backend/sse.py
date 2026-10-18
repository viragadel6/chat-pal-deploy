from __future__ import annotations

import codecs
from collections.abc import AsyncIterator

from .errors import AppError


async def sse_data(chunks: AsyncIterator[bytes], limit: int = 1048576) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer = ""
    data: list[str] = []
    size = 0
    try:
        async for chunk in chunks:
            try:
                buffer += decoder.decode(chunk)
            except UnicodeDecodeError as error:
                raise AppError("MODEL_ENCODING", "Érvénytelen UTF-8 az adatfolyamban.", 502) from error
            while True:
                position = buffer.find("\n")
                if position < 0:
                    if len(buffer.encode("utf-8")) > limit:
                        raise AppError("STREAM_TOO_LARGE", "Túl hosszú SSE sor.", 502)
                    break
                line = buffer[:position]
                buffer = buffer[position + 1 :]
                if line.endswith("\r"):
                    line = line[:-1]
                if not line:
                    if data:
                        yield "\n".join(data)
                        data = []
                        size = 0
                    continue
                if line.startswith(":"):
                    continue
                field, separator, value = line.partition(":")
                if separator and value.startswith(" "):
                    value = value[1:]
                if field == "data":
                    data.append(value)
                    size += len(value.encode("utf-8")) + 1
                    if size > limit:
                        raise AppError("STREAM_TOO_LARGE", "Túl nagy SSE esemény.", 502)
        try:
            trailing = decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise AppError("MODEL_ENCODING", "Érvénytelen záró UTF-8 szekvencia.", 502) from error
        buffer += trailing
        if buffer:
            if buffer.startswith("data:"):
                data.append(buffer[5:].lstrip(" "))
        if data:
            yield "\n".join(data)
    finally:
        await chunks.aclose()


def event_frame(payload: dict, event_id: int | None = None) -> bytes:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}data: {__import__('json').dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")
