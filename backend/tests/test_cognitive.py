import asyncio
import json

import pytest

from app.agent import cognitive as cog
from app.errors import AgentFault


def test_gate_maps_vectors_to_decisions():
    favourable = {
        "confidence": 1.0,
        "progress": 1.0,
        "urgency": 1.0,
        "tool_need": 1.0,
        "reversibility": 1.0,
        "alignment": 1.0,
        "coherence": 1.0,
        "momentum": 1.0,
    }
    act_vector = [favourable.get(dim, 0.0) for dim in cog.VECTOR_DIMS]
    value, decision = cog.gate(act_vector)
    assert decision == "act"
    assert 0.0 <= value <= 1.0
    risky = {
        "risk": 1.0,
        "uncertainty": 1.0,
        "info_need": 1.0,
        "complexity": 1.0,
        "novelty": 1.0,
        "resource_cost": 1.0,
        "blocker_pressure": 1.0,
    }
    replan_vector = [risky.get(dim, 0.0) for dim in cog.VECTOR_DIMS]
    value, decision = cog.gate(replan_vector)
    assert decision == "replan"
    probe_vector = [0.0] * len(cog.VECTOR_DIMS)
    probe_vector[cog.VECTOR_DIMS.index("risk")] = 0.1
    value, decision = cog.gate(probe_vector)
    assert decision == "probe"


def test_gate_thresholds_match_documented_boundaries():
    import math

    value, decision = cog.gate([0.5] * len(cog.VECTOR_DIMS))
    z = cog.GATE_BIAS + 0.5 * sum(cog.GATE_WEIGHTS.values())
    expected = 1 / (1 + math.exp(-z))
    assert abs(value - expected) < 1e-9
    assert decision == ("act" if value >= 0.5 else "probe" if value >= 0.3 else "replan")


def test_empty_sigma_has_neutral_vector():
    sigma = cog.empty_sigma("feladat")
    assert len(sigma.vector) == len(cog.VECTOR_DIMS)
    assert sigma.decision in ("act", "probe", "replan")
    assert sigma.subgoals == []


def test_maintain_keeps_closed_count_and_trims():
    sigma = cog.empty_sigma("feladat")
    sigma.subgoals = [
        cog.Subgoal("a", "első", True),
        cog.Subgoal("b", "második", False),
    ]
    sigma.facts = [f"tény {index}" for index in range(30)]
    result = cog.maintain(sigma)
    assert result.closed_subgoals == 1
    assert len(result.facts) == 20
    assert [item.id for item in result.subgoals] == ["b"]


def test_extract_json_handles_fenced_and_bare_payloads():
    assert cog.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert cog.extract_json('előtte {"a": [1, 2]} utána') == {"a": [1, 2]}
    assert cog.extract_json("nincs json") is None


def test_apply_state_update_records_facts_blockers_and_subgoals():
    sigma = cog.empty_sigma("feladat")
    sigma.subgoals = [cog.Subgoal("s1", "részcél", False)]
    updated = cog.apply_state_update(
        sigma,
        {"fact": "elvégeztük", "blocker": "hiányzik kulcs", "complete_subgoal": "s1"},
    )
    assert "elvégeztük" in updated.facts
    assert "hiányzik kulcs" in updated.blockers
    assert updated.closed_subgoals == 1
    assert [item.id for item in updated.subgoals] == []
    cleared = cog.apply_state_update(updated, {"resolve_blocker": "HIÁNYZIK KULCS"})
    assert cleared.blockers == []


def test_resolve_pointer_supports_json_pointer():
    payload = {"a": {"b": [10, 20]}}
    assert cog.resolve_pointer(payload, "/a/b/1") == 20
    assert cog.resolve_pointer(payload, "/a/missing") is None
    assert cog.resolve_pointer(payload, "bad") is None


def test_deep_equal_compares_structures():
    assert cog.deep_equal({"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]})
    assert not cog.deep_equal({"a": 1}, {"a": 2})


def test_embed_is_normalized():
    vector = cog.embed("python script futtatása")
    assert len(vector) == cog.EMBED_DIM
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6


def test_cosine_detects_similarity():
    left = cog.embed("docker build")
    assert cog.cosine(left, left) > 0.99
    assert cog.cosine(left, cog.embed("teljesen más szöveg")) < 0.99


def test_parse_vector_accepts_multiple_encodings():
    assert cog.parse_vector([1, 2]) == [1.0, 2.0]
    assert cog.parse_vector("[1, 2]") == [1.0, 2.0]
    assert cog.parse_vector("1,2") == [1.0, 2.0]
    assert cog.parse_vector(None) == []


def test_valid_dsl_requires_all_sections():
    good = "SKILL x\nWHEN a\nREQUIRE b\nSTEP c\nVERIFY d\nRECOVER e"
    assert cog.valid_dsl(good)
    assert not cog.valid_dsl("SKILL x\nWHEN a")


def test_lifecycle_rejects_illegal_transition():
    lifecycle = cog.Lifecycle(lambda a, b: asyncio.sleep(0))
    with pytest.raises(AgentFault):
        asyncio.get_event_loop().run_until_complete(lifecycle.to("osValidate"))


@pytest.mark.asyncio
async def test_deliberate_parses_planner_output():
    async def think(system: str, user: str) -> str:
        return json.dumps(
            {
                "strategy": "először mérj",
                "subgoals": [{"id": "s1", "text": "mérés", "done": False}],
                "next_steps": ["futtasd a tesztet"],
                "vector": [0.9] * 16,
                "goal": {
                    "state_pointers": [{"pointer": "/facts/0", "equals": "ok"}],
                    "required_files": ["/tmp/out.txt"],
                    "file_contents": [{"path": "/tmp/out.txt", "contains": "hello"}],
                },
            }
        )

    sigma = cog.empty_sigma("mérd meg")
    result = await cog.deliberate(think, sigma, "nyomvonal", "nincs")
    assert result.strategy == "először mérj"
    assert result.subgoals[0].id == "s1"
    assert result.next_steps == ["futtasd a tesztet"]
    assert result.goal.required_files == ["/tmp/out.txt"]
    assert result.goal.state_pointers[0]["pointer"] == "/facts/0"


@pytest.mark.asyncio
async def test_deliberate_rejects_invalid_payload():
    async def think(system: str, user: str) -> str:
        return "nem json"

    with pytest.raises(AgentFault):
        await cog.deliberate(think, cog.empty_sigma("x"), "", "")


@pytest.mark.asyncio
async def test_reflect_validates_component():
    async def think(system: str, user: str) -> str:
        return json.dumps(
            {
                "failure_point": "hiba",
                "component": "not_a_component",
                "root_cause": "ok",
                "pivot_action": "javítás",
            }
        )

    with pytest.raises(AgentFault):
        await cog.reflect(think, "cél", "trace", "action")


@pytest.mark.asyncio
async def test_verify_reports_missing_evidence():
    sigma = cog.empty_sigma("cél")
    result = await cog.verify(sigma, None)
    assert not result["ok"]
    assert any("bizonyíték" in failure for failure in result["failures"])


@pytest.mark.asyncio
async def test_verify_accepts_satisfied_goal():
    sigma = cog.empty_sigma("cél")
    sigma.facts.append("igazolt tény")
    result = await cog.verify(sigma, None)
    assert result["ok"]
    assert result["failures"] == []


@pytest.mark.asyncio
async def test_reason_tree_builds_recursive_structure():
    async def think(system: str, user: str) -> str:
        if "Bontsd" in system:
            return json.dumps(["alkérdés 1"])
        return "válasz"

    tree = await cog.reason_tree(think, "fő kérdés", 1, 2)
    assert tree.question == "fő kérdés"
    assert tree.answer
    assert len(tree.children) == 1
    assert tree.children[0].question == "alkérdés 1"


@pytest.mark.asyncio
async def test_reason_tree_rejects_oversized_request():
    async def think(system: str, user: str) -> str:
        return ""

    with pytest.raises(AgentFault):
        await cog.reason_tree(think, "kérdés", 4, 6)


def test_reason_node_serialization():
    node = cog.ReasonNode("q", "a", [cog.ReasonNode("q2", "a2", [])])
    payload = cog.reason_node_to_dict(node)
    assert payload["question"] == "q"
    assert payload["children"][0]["answer"] == "a2"
