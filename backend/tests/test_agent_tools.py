import json

import pytest

from app.agent.tools import (
    TOOL_SPECS,
    enabled_tool_names,
    provider_tools,
    validate_arguments,
)
from app.errors import RequestFault


def test_every_tool_spec_has_a_description_and_fields():
    for name, spec in TOOL_SPECS.items():
        assert spec.name == name
        assert spec.description.strip()
        assert spec.fields


def test_validate_arguments_rejects_unknown_keys():
    with pytest.raises(RequestFault):
        validate_arguments("run_shell", {"command": "ls", "unknown": 1})


def test_validate_arguments_requires_mandatory_fields():
    with pytest.raises(RequestFault):
        validate_arguments("run_shell", {})


def test_validate_arguments_applies_integer_defaults():
    args = validate_arguments("run_shell", {"command": "ls"})
    assert args["timeout"] == 180


def test_validate_arguments_enforces_string_bounds():
    with pytest.raises(RequestFault):
        validate_arguments("write_file", {"path": "/tmp/x", "content": "a" * 200_000})


def test_validate_arguments_rejects_null_bytes_in_paths():
    with pytest.raises(RequestFault):
        validate_arguments("write_file", {"path": "/tmp/\x00x", "content": "ok"})


def test_validate_arguments_allows_empty_file_content():
    args = validate_arguments("write_file", {"path": "/tmp/x", "content": ""})
    assert args["content"] == ""


def test_update_state_requires_any_value():
    with pytest.raises(RequestFault):
        validate_arguments("update_state", {"fact": "", "blocker": "  "})


def test_install_packages_validates_pattern():
    with pytest.raises(RequestFault):
        validate_arguments("install_packages", {"packages": ["bad package!"]})
    args = validate_arguments("install_packages", {"packages": ["python3", "jq"]})
    assert args["packages"] == ["python3", "jq"]


def test_expose_port_range_is_enforced():
    with pytest.raises(RequestFault):
        validate_arguments("expose_port", {"port": 0})
    with pytest.raises(RequestFault):
        validate_arguments("expose_port", {"port": 70000})


def test_provider_tools_schema_shape():
    tools = provider_tools(list(TOOL_SPECS.keys()))
    assert len(tools) == len(TOOL_SPECS)
    for tool in tools:
        assert tool["type"] == "function"
        function = tool["function"]
        assert function["name"] in TOOL_SPECS
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert isinstance(parameters["required"], list)
        assert isinstance(parameters["properties"], dict)


def test_provider_tools_update_state_requires_min_properties():
    tools = {tool["function"]["name"]: tool for tool in provider_tools(list(TOOL_SPECS.keys()))}
    assert tools["update_state"]["function"]["parameters"]["minProperties"] == 1


def test_enabled_tools_depend_on_sandbox_and_browser_configuration():
    without_sandbox = enabled_tool_names(None, 0)
    assert "run_shell" not in without_sandbox
    assert "browse_url" not in without_sandbox
    assert "update_plan" in without_sandbox
    with_sandbox = enabled_tool_names("key", 0)
    assert "run_shell" in with_sandbox
    assert "browse_url" not in with_sandbox
    with_browser = enabled_tool_names("key", 3)
    assert "browse_url" in with_browser


def test_provider_tool_serialization_is_json_safe():
    tools = provider_tools(list(TOOL_SPECS.keys()))
    assert json.dumps(tools, ensure_ascii=False)
