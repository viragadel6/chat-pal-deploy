import asyncio
import json

import httpx
import pytest

from app.agent.provider import (
    ToolCallBuilder,
    finalize_tool_call,
    parse_sse,
    receive_model,
    validate_tool_delta,
)
from app.errors import ProtocolFault


async def _chunks(items: list[bytes]):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_parse_sse_handles_utf8_crlf_comments_and_trailing_data():
    payload = (
        ': ping\r\n\r\ndata: {"text":"árvíz"}\r\n\r\ndata: [DONE]'.encode()
    )
    frames = [frame async for frame in parse_sse(_chunks([payload]), asyncio.Event(), lambda: None)]
    assert frames == ['{"text":"árvíz"}', "[DONE]"]


@pytest.mark.asyncio
async def test_parse_sse_yields_incrementally():
    body = _chunks([b'data: {"delta":"first"}\n\n'])
    iterator = parse_sse(body, asyncio.Event(), lambda: None)
    first = await iterator.__anext__()
    assert first == '{"delta":"first"}'


@pytest.mark.asyncio
async def test_parse_sse_rejects_oversized_event():
    oversized = b"data: " + b"x" * 1_100_000 + b"\n\n"
    with pytest.raises(ProtocolFault):
        async for _ in parse_sse(_chunks([oversized]), asyncio.Event(), lambda: None):
            pass


def test_validate_tool_delta_collects_fragments():
    parts: dict[int, ToolCallBuilder] = {}
    validate_tool_delta(
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_", "arguments": '{"com'},
                    }
                ]
            }
        },
        parts,
    )
    validate_tool_delta(
        {
            "delta": {
                "tool_calls": [
                    {"index": 0, "function": {"name": "shell", "arguments": 'mand":"ls"}'}}
                ]
            }
        },
        parts,
    )
    call = finalize_tool_call(parts[0])
    assert call.id == "call_1"
    assert call.function_name == "run_shell"
    assert json.loads(call.function_arguments) == {"command": "ls"}


def test_validate_tool_delta_rejects_out_of_range_index():
    with pytest.raises(ProtocolFault):
        validate_tool_delta(
            {"delta": {"tool_calls": [{"index": 99, "id": "x"}]}}, {}
        )


def test_finalize_tool_call_falls_back_to_snapshot_on_truncated_json():
    builder = ToolCallBuilder(id="call_2")
    builder.name = "run_shell"
    builder.name_parts = 2
    builder.name_snapshot = "run_shell"
    builder.name_snapshot_valid = True
    builder.arguments = '{"command":'
    builder.argument_parts = 2
    builder.argument_snapshot = '{"command":'
    builder.argument_snapshot_valid = True
    builder.argument_overflow = True
    call = finalize_tool_call(builder)
    assert call.function_name == "run_shell"
    assert call.function_arguments == '{"command":'


def _sse_response(chunks: list[str]) -> httpx.Response:
    body = b"".join(chunk.encode("utf-8") for chunk in chunks)
    return httpx.Response(
        200,
        content=body,
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.asyncio
async def test_receive_model_collects_content_and_tool_calls():
    events: list[dict] = []
    stream_response = _sse_response(
        [
            'data: {"choices":[{"delta":{"content":"Helló"}}]}\n\n',
            'data: {"choices":[{"delta":{"reasoning_content":"gondolkodom"}}]}\n\n',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run_shell","arguments":"{\\"command\\":\\"ls\\"}"}}]}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: stream_response))
    response = await client.send(client.build_request("GET", "https://example.test"), stream=True)
    from app.agent.provider import ModelStream

    stream = ModelStream(response=response, client=client, touch=lambda: None)
    result = await receive_model(
        stream,
        "task",
        0,
        lambda event: _append(events, event),
        asyncio.Event(),
    )
    assert result.content == "Helló"
    assert result.finish == "tool_calls"
    assert len(result.calls) == 1
    assert result.calls[0].function_name == "run_shell"
    assert any(event.get("reasoning_content") == "gondolkodom" for event in events)


async def _append(events: list[dict], event: dict) -> None:
    events.append(event)


@pytest.mark.asyncio
async def test_receive_model_rejects_incomplete_stream():
    stream_response = _sse_response(['data: {"choices":[{"delta":{"content":"x"}}]}\n\n'])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: stream_response))
    response = await client.send(client.build_request("GET", "https://example.test"), stream=True)
    from app.agent.provider import ModelStream

    stream = ModelStream(response=response, client=client, touch=lambda: None)
    with pytest.raises(ProtocolFault):
        await receive_model(stream, "task", 0, lambda event: _append([], event), asyncio.Event())
