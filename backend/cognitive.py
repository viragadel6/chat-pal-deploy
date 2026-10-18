from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from .db import Supabase, eq, gt, lt, utc_now
from .errors import AppError

Think = Callable[[str, str], Awaitable[str]]
Probe = Callable[[str], Awaitable[bool]]
Phase = Literal["osPerceive", "osDeliberate", "osAct", "osValidate", "osReflect", "osConsolidate"]
Decision = Literal["act", "probe", "replan"]

TRANSITIONS: dict[Phase, set[Phase]] = {
    "osPerceive": {"osDeliberate", "osConsolidate"},
    "osDeliberate": {"osAct", "osConsolidate"},
    "osAct": {"osAct", "osValidate", "osReflect", "osPerceive", "osDeliberate", "osConsolidate"},
    "osValidate": {"osReflect", "osConsolidate"},
    "osReflect": {"osDeliberate", "osConsolidate"},
    "osConsolidate": set(),
}
VECTOR_DIMS = (
    "confidence", "progress", "risk", "uncertainty", "complexity", "novelty", "urgency", "tool_need",
    "info_need", "verification_need", "reversibility", "resource_cost", "alignment", "coherence", "momentum", "blocker_pressure",
)
GATE_WEIGHTS = (1.6, 0.8, -1.4, -1.2, -0.4, -0.3, 0.3, 0.2, -0.9, -0.3, 0.7, -0.4, 1.0, 0.9, 0.5, -1.3)
COMPONENTS = {"planner", "executor", "tool", "environment", "verifier", "model"}
DB_LOCK = asyncio.Lock()


@dataclass(slots=True)
class Subgoal:
    id: str
    text: str
    done: bool = False


@dataclass(slots=True)
class GoalSpec:
    state_pointers: list[dict[str, Any]] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    file_contents: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class Sigma:
    objective: str
    strategy: str = ""
    subgoals: list[Subgoal] = field(default_factory=list)
    closed_subgoals: int = 0
    facts: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=lambda: [0.5] * 16)
    gate: float = 0.0
    decision: Decision = "act"
    goal: GoalSpec = field(default_factory=GoalSpec)

    def data(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "strategy": self.strategy,
            "subgoals": [{"id": item.id, "text": item.text, "done": item.done} for item in self.subgoals],
            "closed_subgoals": self.closed_subgoals,
            "facts": self.facts,
            "blockers": self.blockers,
            "next_steps": self.next_steps,
            "vector": self.vector,
            "gate": self.gate,
            "decision": self.decision,
            "goal": {
                "state_pointers": self.goal.state_pointers,
                "required_files": self.goal.required_files,
                "file_contents": self.goal.file_contents,
            },
        }


@dataclass(slots=True)
class Reflection:
    failure_point: str
    component: str
    root_cause: str
    pivot_action: str
    patch: str
    failed_action: str
    corrective_action: str


@dataclass(slots=True)
class Skill:
    id: str
    name: str
    dsl: str
    terms: str
    vector: list[float]
    successes: int
    failures: int


class Lifecycle:
    def __init__(self, on_change: Callable[[Phase, Phase], Awaitable[None]]) -> None:
        self.phase: Phase = "osPerceive"
        self.trail: list[Phase] = [self.phase]
        self.on_change = on_change

    async def to(self, next_phase: Phase) -> None:
        if next_phase not in TRANSITIONS[self.phase]:
            raise AppError("LIFECYCLE_FAILURE", f"Tiltott fázisátmenet: {self.phase} -> {next_phase}", 500)
        current = self.phase
        await self.on_change(current, next_phase)
        self.phase = next_phase
        self.trail.append(next_phase)


def clip(value: str, maximum: int) -> str:
    return value[:maximum]


def rec(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def gate(vector: list[float]) -> tuple[float, Decision]:
    z = 0.1 + sum(weight * (vector[index] if index < len(vector) and math.isfinite(vector[index]) else 0.5) for index, weight in enumerate(GATE_WEIGHTS))
    value = 1 / (1 + math.exp(-max(-500, min(500, z))))
    return value, "act" if value >= 0.5 else "probe" if value >= 0.3 else "replan"


def empty_sigma(objective: str) -> Sigma:
    value, decision = gate([0.5] * 16)
    return Sigma(objective=clip(objective, 4000), gate=value, decision=decision)


def maintain(sigma: Sigma) -> Sigma:
    closed_now = sum(1 for item in sigma.subgoals if item.done)
    open_subgoals = [Subgoal(clip(item.id, 40), clip(item.text, 400), False) for item in sigma.subgoals if not item.done][:12]
    unique_facts = list(dict.fromkeys(clip(value, 400) for value in sigma.facts if value.strip()))[-20:]
    unique_blockers = list(dict.fromkeys(clip(value, 400) for value in sigma.blockers if value.strip()))[-8:]
    sigma.strategy = clip(sigma.strategy, 2000)
    sigma.subgoals = open_subgoals
    sigma.closed_subgoals = max(0, sigma.closed_subgoals) + closed_now
    sigma.facts = unique_facts
    sigma.blockers = unique_blockers
    sigma.next_steps = [clip(value, 400) for value in sigma.next_steps if value.strip()][:8]
    sigma.vector = [min(1.0, max(0.0, float(value))) if isinstance(value, (int, float)) and math.isfinite(value) else 0.5 for value in sigma.vector[:16]]
    sigma.vector.extend([0.5] * (16 - len(sigma.vector)))
    sigma.goal.state_pointers = sigma.goal.state_pointers[:10]
    sigma.goal.required_files = [clip(value, 4096) for value in sigma.goal.required_files if value][:20]
    sigma.goal.file_contents = [{"path": clip(item["path"], 4096), "contains": clip(item["contains"], 8000)} for item in sigma.goal.file_contents[:20] if isinstance(item.get("path"), str) and isinstance(item.get("contains"), str)]
    sigma.gate, sigma.decision = gate(sigma.vector)
    return sigma


def extract_json(text: str) -> Any:
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    for candidate in [*fenced, text]:
        for start, character in enumerate(candidate):
            if character not in "[{":
                continue
            stack = ["}" if character == "{" else "]"]
            quoted = False
            escaped = False
            for end in range(start + 1, len(candidate)):
                current = candidate[end]
                if quoted:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == '"':
                        quoted = False
                    continue
                if current == '"':
                    quoted = True
                elif current == "{":
                    stack.append("}")
                elif current == "[":
                    stack.append("]")
                elif current in "}]":
                    if not stack or current != stack[-1]:
                        break
                    stack.pop()
                    if not stack:
                        try:
                            return json.loads(candidate[start : end + 1])
                        except json.JSONDecodeError:
                            break
    return None


def string_list(value: Any, maximum: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:maximum]


async def deliberate(think: Think, sigma: Sigma, trace: str, guidance: str) -> Sigma:
    system = "\n".join((
        "Te a System 2 stratégiai tervező vagy. Kizárólag egyetlen JSON objektumot adj vissza, magyarázat nélkül.",
        "Séma: {\"strategy\": string, \"subgoals\": [{\"id\": string, \"text\": string, \"done\": boolean}], \"next_steps\": [string], \"vector\": [16 szám 0 és 1 között], \"goal\": {\"state_pointers\": [{\"pointer\": JSON-pointer a munkamemóriába, \"equals\": érték}], \"required_files\": [abszolút útvonal], \"file_contents\": [{\"path\": string, \"contains\": string}]}}",
        f"A vector dimenziói sorrendben: {', '.join(VECTOR_DIMS)}.",
        "A next_steps legfeljebb 5 konkrét, azonnal végrehajtható mikrolépés legyen.",
        "Egyszerű beszélgetésnél a subgoals és a goal minden listája legyen üres.",
    ))
    raw = await think(system, f"MUNKAMEMÓRIA:\n{json.dumps(sigma.data(), ensure_ascii=False)}\n\nTANULT ÚTMUTATÁS:\n{guidance or 'nincs'}\n\nFRISS NYOMVONAL:\n{clip(trace, 8000)}")
    parsed = rec(extract_json(raw))
    if parsed is None:
        raise AppError("DELIBERATION_FAILURE", "A stratégiai tervező érvénytelen JSON választ adott.", 502)
    next_sigma = Sigma(
        objective=sigma.objective,
        strategy=sigma.strategy,
        subgoals=[Subgoal(item.id, item.text, item.done) for item in sigma.subgoals],
        closed_subgoals=sigma.closed_subgoals,
        facts=[*sigma.facts],
        blockers=[*sigma.blockers],
        next_steps=[*sigma.next_steps],
        vector=[*sigma.vector],
        gate=sigma.gate,
        decision=sigma.decision,
        goal=GoalSpec([*sigma.goal.state_pointers], [*sigma.goal.required_files], [*sigma.goal.file_contents]),
    )
    strategy = parsed.get("strategy")
    if isinstance(strategy, str):
        next_sigma.strategy = strategy
    previous_by_id = {item.id: item for item in sigma.subgoals}
    previous_by_text = {item.text: item for item in sigma.subgoals}
    candidates: list[Subgoal] = []
    identifiers: set[str] = set()
    for index, item in enumerate(parsed.get("subgoals", [])):
        item_record = rec(item)
        if item_record is None or not isinstance(item_record.get("text"), str) or not item_record["text"].strip():
            continue
        text = item_record["text"].strip()
        requested_id = item_record.get("id") if isinstance(item_record.get("id"), str) else ""
        old = previous_by_id.get(requested_id) or previous_by_text.get(text)
        identifier = old.id if old else requested_id.strip() or f"sg_{int(time.time() * 1000):x}_{index}"
        if identifier in identifiers:
            continue
        identifiers.add(identifier)
        candidates.append(Subgoal(identifier, text, old.done if old else False))
    next_sigma.subgoals = candidates
    next_sigma.next_steps = string_list(parsed.get("next_steps"), 5)
    vector = parsed.get("vector")
    if isinstance(vector, list) and len(vector) == 16:
        next_sigma.vector = [float(item) if isinstance(item, (int, float)) and math.isfinite(item) else 0.5 for item in vector]
    goal_record = rec(parsed.get("goal"))
    if goal_record is not None:
        pointers = []
        for item in goal_record.get("state_pointers", []):
            item_record = rec(item)
            if item_record and isinstance(item_record.get("pointer"), str) and "equals" in item_record:
                pointers.append({"pointer": item_record["pointer"], "equals": item_record["equals"]})
        contents = []
        for item in goal_record.get("file_contents", []):
            item_record = rec(item)
            if item_record and isinstance(item_record.get("path"), str) and isinstance(item_record.get("contains"), str):
                contents.append({"path": item_record["path"], "contains": item_record["contains"]})
        next_sigma.goal = GoalSpec(pointers[:10], string_list(goal_record.get("required_files"), 20), contents[:20])
    return maintain(next_sigma)


def apply_state_update(sigma: Sigma, values: dict[str, Any]) -> Sigma:
    next_sigma = Sigma(
        objective=sigma.objective,
        strategy=sigma.strategy,
        subgoals=[Subgoal(item.id, item.text, item.done) for item in sigma.subgoals],
        closed_subgoals=sigma.closed_subgoals,
        facts=[*sigma.facts],
        blockers=[*sigma.blockers],
        next_steps=[*sigma.next_steps],
        vector=[*sigma.vector],
        gate=sigma.gate,
        decision=sigma.decision,
        goal=GoalSpec([*sigma.goal.state_pointers], [*sigma.goal.required_files], [*sigma.goal.file_contents]),
    )
    if isinstance(values.get("fact"), str) and values["fact"].strip():
        next_sigma.facts.append(values["fact"].strip())
    if isinstance(values.get("blocker"), str) and values["blocker"].strip():
        next_sigma.blockers.append(values["blocker"].strip())
    if isinstance(values.get("resolve_blocker"), str) and values["resolve_blocker"].strip():
        needle = values["resolve_blocker"].strip().casefold()
        next_sigma.blockers = [item for item in next_sigma.blockers if item.strip().casefold() != needle]
    if isinstance(values.get("complete_subgoal"), str) and values["complete_subgoal"].strip():
        needle = values["complete_subgoal"].strip()
        for item in next_sigma.subgoals:
            if item.id == needle or item.text == needle:
                item.done = True
    return maintain(next_sigma)


def json_pointer(root: Any, location: str) -> Any:
    if location == "":
        return root
    if not location.startswith("/"):
        return None
    current = root
    for raw in location[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            return None
    return current


async def verify(sigma: Sigma, probe: Probe | None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    root = sigma.data()
    for pointer in sigma.goal.state_pointers:
        expected = pointer.get("equals")
        actual = json_pointer(root, str(pointer.get("pointer", "")))
        if actual != expected:
            failures.append(f"Állapot nem egyezik: {pointer.get('pointer', '')}")
    if (sigma.goal.required_files or sigma.goal.file_contents) and probe is None:
        failures.append("Fájlellenőrzés szükséges, de nincs elérhető VM.")
    if probe:
        for path in sigma.goal.required_files:
            if not await probe(f"test -f {shell_quote(path)}"):
                failures.append(f"Hiányzó fájl: {path}")
        for requirement in sigma.goal.file_contents:
            path = requirement["path"]
            needle = requirement["contains"]
            program = 'import pathlib,sys;sys.exit(0 if sys.argv[2] in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8") else 1)'
            command = f"python3 -c {shell_quote(program)} {shell_quote(path)} {shell_quote(needle)}"
            if not await probe(command):
                failures.append(f"Fájltartalom nem egyezik: {path}")
    if sigma.blockers:
        failures.append(f"Nyitott blokkolók: {'; '.join(sigma.blockers)}")
    if sigma.subgoals:
        failures.append(f"Lezáratlan részcélok: {', '.join(item.id for item in sigma.subgoals)}")
    if not (sigma.goal.state_pointers or sigma.goal.required_files or sigma.goal.file_contents or sigma.closed_subgoals or sigma.facts):
        failures.append("Nincs ellenőrizhető bizonyíték a feladat teljesítésére.")
    return not failures, failures


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


async def reflect(think: Think, objective: str, trace: str, failed_action: str) -> Reflection:
    system = "Hibaelemző vagy. A nyers nyomvonal alapján végezz gyökérok-elemzést. Kizárólag JSON-t adj vissza. Séma: {\"failure_point\": string, \"component\": planner|executor|tool|environment|verifier|model, \"root_cause\": string, \"pivot_action\": string, \"patch\": string, \"corrective_action\": string}"
    parsed = rec(extract_json(await think(system, f"CÉL:\n{objective}\n\nSIKERTELEN MŰVELET:\n{clip(failed_action, 2000)}\n\nNYOMVONAL:\n{clip(trace, 10000)}")))
    if parsed is None:
        raise AppError("REFLECTION_FAILURE", "A hibaelemző érvénytelen JSON választ adott.", 502)
    def required(name: str) -> str:
        value = parsed.get(name)
        if not isinstance(value, str) or not value.strip():
            raise AppError("REFLECTION_FAILURE", f"A hibaelemzésből hiányzik: {name}", 502)
        return clip(value.strip(), 2000)
    component = required("component").casefold()
    if component not in COMPONENTS:
        raise AppError("REFLECTION_FAILURE", "A hibaelemzés érvénytelen komponenst adott.", 502)
    pivot = required("pivot_action")
    patch = clip(str(parsed.get("patch") or ""), 2000)
    corrective = clip(str(parsed.get("corrective_action") or ""), 2000) or patch or pivot
    return Reflection(required("failure_point"), component, required("root_cause"), pivot, patch, clip(failed_action, 2000), corrective)


def tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def fnv(text: str) -> int:
    result = 0x811C9DC5
    for character in text:
        result ^= ord(character)
        result = (result * 0x01000193) & 0xFFFFFFFF
    return result


def embed(text: str) -> list[float]:
    vector = [0.0] * 256
    words = tokens(text)
    for index, word in enumerate(words):
        vector[fnv(word) % 256] += 1
        if index + 1 < len(words):
            vector[fnv(f"{word} {words[index + 1]}") % 256] += 0.5
        for offset in range(max(0, len(word) - 2)):
            vector[fnv(f"#{word[offset:offset + 3]}") % 256] += 0.25
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.strip("[]{}() ").split(",") if part.strip()]
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            number = float(item)
            result.append(number if math.isfinite(number) else 0.0)
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def cosine(a: Any, b: Any) -> float:
    left = parse_vector(a)
    right = parse_vector(b)
    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(x * y for x, y in zip(left, right))
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return numerator / denominator if denominator else 0.0


async def retrieve(db: Supabase, objective: str, context: str) -> tuple[list[Skill], str]:
    skills_rows, playbooks = await asyncio.gather(
        db.admin_select("agent_skills", "id,name,dsl,terms,vector,successes,failures", [eq("status", "active")], limit=500),
        db.admin_select("agent_playbooks", "content", order="created_at.desc", limit=1),
    )
    query = f"{objective}\n{context}"
    query_vector = embed(query)
    query_tokens = set(tokens(query))
    scored: list[tuple[float, Skill]] = []
    for row in skills_rows:
        vector = parse_vector(row.get("vector"))
        terms = tokens(f"{row.get('name', '')} {row.get('terms', '')}")
        lexical = sum(1 for item in terms if item in query_tokens) / math.sqrt(max(1, len(terms) * len(query_tokens)))
        successes = int(row.get("successes", 0))
        failures = int(row.get("failures", 0))
        rate = (successes + 1) / (successes + failures + 2)
        skill = Skill(str(row["id"]), str(row["name"]), str(row["dsl"]), str(row.get("terms", "")), vector, successes, failures)
        score = 0.5 * cosine(query_vector, vector) + 0.3 * min(1.0, lexical) + 0.2 * rate
        if score > 0.25:
            scored.append((score, skill))
    scored.sort(key=lambda value: value[0], reverse=True)
    return [item for _, item in scored[:3]], str(playbooks[0].get("content", "")) if playbooks else ""


async def policy_guidance(db: Supabase) -> str:
    positive, negative = await asyncio.gather(
        db.admin_select("agent_policy_weights", "ngram,weight", [gt("weight", 0.01)], order="weight.desc", limit=8),
        db.admin_select("agent_policy_weights", "ngram,weight", [lt("weight", -0.01)], order="weight.asc", limit=8),
    )
    lines: list[str] = []
    if positive:
        lines.append("FAVOR: " + ", ".join(f'"{row["ngram"]}"' for row in positive))
    if negative:
        lines.append("AVOID: " + ", ".join(f'"{row["ngram"]}"' for row in negative))
    return "\n".join(lines)


async def take_operator_messages(db: Supabase, task_id: str) -> list[str]:
    rows = await db.admin_update("agent_operator_messages", {"consumed": True}, [eq("task_id", task_id), eq("consumed", False)], "id,content,created_at")
    rows.sort(key=lambda item: str(item.get("created_at", "")))
    return [str(item["content"]) for item in rows]


async def post_operator_message(db: Supabase, task_id: str, content: str) -> None:
    await db.admin_insert("agent_operator_messages", {"task_id": task_id, "content": clip(content, 8000), "consumed": False}, "minimal")


def ngrams(text: str) -> list[str]:
    values = tokens(text)
    return [f"{values[index]} {values[index + 1]}" for index in range(len(values) - 1)]


async def apply_policy_delta(db: Supabase, ngram: str, delta: float) -> None:
    for _ in range(8):
        current = await db.admin_single("agent_policy_weights", "ngram,weight,observations", [eq("ngram", ngram)])
        if current is None:
            try:
                await db.admin_insert("agent_policy_weights", {"ngram": ngram, "weight": delta, "observations": 1, "updated_at": utc_now()})
                return
            except AppError:
                continue
        observations = int(current["observations"])
        updated = await db.admin_update("agent_policy_weights", {"weight": float(current["weight"]) + delta, "observations": observations + 1, "updated_at": utc_now()}, [eq("ngram", ngram), eq("observations", observations)], "ngram")
        if updated:
            return
    raise AppError("LEARNING_CONFLICT", f"A politikasúly frissítése sikertelen: {ngram}", 503)


async def distill(db: Supabase, pairs: list[tuple[str, str, float]]) -> None:
    deltas: dict[str, float] = {}
    for failed, corrective, advantage in pairs:
        if advantage <= 0:
            continue
        before = ngrams(failed)
        after = ngrams(corrective)
        if not before or not after:
            continue
        counts_before = {item: before.count(item) / len(before) for item in set(before)}
        counts_after = {item: after.count(item) / len(after) for item in set(after)}
        for item in set(counts_before) | set(counts_after):
            previous = counts_before.get(item, 0.0) + 0.001
            next_value = counts_after.get(item, 0.0) + 0.001
            delta = 0.5 * min(1.0, advantage) * (previous + next_value) * math.log(next_value / previous)
            if abs(delta) > 0.0001:
                deltas[item] = deltas.get(item, 0.0) + delta
    async with DB_LOCK:
        for item, delta in sorted(deltas.items(), key=lambda value: abs(value[1]), reverse=True)[:400]:
            await apply_policy_delta(db, item, delta)


async def record_skill_outcome(db: Supabase, skill_id: str, verified: bool) -> None:
    for _ in range(8):
        current = await db.admin_single("agent_skills", "successes,failures", [eq("id", skill_id)])
        if current is None:
            return
        successes = int(current["successes"])
        failures = int(current["failures"])
        values = {"successes": successes + 1} if verified else {"failures": failures + 1}
        updated = await db.admin_update("agent_skills", values, [eq("id", skill_id), eq("successes", successes), eq("failures", failures)], "id")
        if updated:
            return
    raise AppError("LEARNING_CONFLICT", f"A képesség frissítése sikertelen: {skill_id}", 503)


def valid_dsl(value: str) -> bool:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return all(any(line.startswith(f"{token} ") or line.startswith(f"{token}:") for line in lines) for token in ("SKILL", "WHEN", "REQUIRE", "STEP", "VERIFY", "RECOVER"))


def solves(dsl: str, task: dict[str, Any]) -> bool:
    skill_tokens = set(tokens(dsl))
    needed = list(dict.fromkeys(item for item in tokens(f"{task.get('pivot_action', '')} {task.get('root_cause', '')}") if len(item) >= 4))
    if not needed or sum(item in skill_tokens for item in needed) / len(needed) < 0.5:
        return False
    return cosine(embed(f"{task.get('objective', '')}\n{task.get('root_cause', '')}"), embed(dsl)) >= 0.2


async def regression_gate(db: Supabase, candidate: str, reflection_ids: set[str], task_ids: set[str], component: str) -> tuple[bool, float, float]:
    active, reflections = await asyncio.gather(
        db.admin_select("agent_skills", "dsl", [eq("status", "active")], limit=100),
        db.admin_select("agent_reflections", "id,task_id,objective,component,root_cause,pivot_action", [eq("component", component)], order="created_at.desc", limit=200),
    )
    held_out: list[dict[str, Any]] = []
    objectives: set[str] = set()
    for item in reflections:
        if str(item.get("id")) in reflection_ids or str(item.get("task_id")) in task_ids or str(item.get("objective", "")) in objectives:
            continue
        objectives.add(str(item.get("objective", "")))
        held_out.append(item)
        if len(held_out) == 20:
            break
    if not held_out:
        return False, 0.0, 0.0
    baseline_solved = 0
    candidate_solved = 0
    known = [str(item.get("dsl", "")) for item in active]
    for task in held_out:
        baseline = any(solves(dsl, task) for dsl in known)
        candidate_hit = baseline or solves(candidate, task)
        baseline_solved += int(baseline)
        candidate_solved += int(candidate_hit)
    baseline_rate = baseline_solved / len(held_out)
    candidate_rate = candidate_solved / len(held_out)
    return candidate_rate > baseline_rate, baseline_rate, candidate_rate


async def synthesize_skills(db: Supabase, think: Think, log: Callable[[str], Awaitable[None]]) -> None:
    reflections, existing = await asyncio.gather(
        db.admin_select("agent_reflections", "id,task_id,component,root_cause,pivot_action,patch,objective", order="created_at.desc", limit=50),
        db.admin_select("agent_skills", "vector", limit=1000),
    )
    vectors = [parse_vector(item.get("vector")) for item in existing]
    items = [{**item, "embedding": embed(f"{item.get('component', '')} {item.get('root_cause', '')}")} for item in reflections]
    used: set[int] = set()
    for index, seed in enumerate(items):
        if index in used:
            continue
        cluster_indices = [index]
        for other_index in range(index + 1, len(items)):
            other = items[other_index]
            if other_index not in used and other.get("component") == seed.get("component") and cosine(seed["embedding"], other["embedding"]) >= 0.55:
                cluster_indices.append(other_index)
        if len(cluster_indices) < 2:
            continue
        used.update(cluster_indices)
        group = [items[item] for item in cluster_indices]
        centroid = embed(" ".join(f"{item.get('component', '')} {item.get('root_cause', '')}" for item in group))
        if any(cosine(vector, centroid) >= 0.8 for vector in vectors):
            continue
        system = "\n".join((
            "Meta-ágens vagy: a visszatérő hibamintából újrahasznosítható képességet szintetizálsz SKILL DSL nyelven. Csak a DSL-t add vissza.",
            "Kötelező formátum, minden záradék külön sorban:",
            "SKILL <rövid_snake_case_név>",
            "WHEN <alkalmazási feltétel>",
            "REQUIRE <előfeltétel>",
            "STEP <lépés> (több STEP sor is lehet)",
            "VERIFY <ellenőrizhető sikerfeltétel>",
            "RECOVER <helyreállítási lépés hiba esetén>",
        ))
        prompt = "\n---\n".join(f"KOMPONENS: {item.get('component', '')}\nGYÖKÉROK: {item.get('root_cause', '')}\nKORREKCIÓ: {item.get('pivot_action', '')}\nJAVÍTÁS: {item.get('patch', '')}" for item in group)
        dsl = re.sub(r"```[a-z]*\n?|```", "", await think(system, prompt), flags=re.IGNORECASE).strip()
        if not valid_dsl(dsl):
            await log("A szintetizált képesség érvénytelen DSL-t adott, elvetve.")
            continue
        match = re.search(r"^[ \t]*SKILL[: \t]+(\S[^\n]*)$", dsl, re.MULTILINE)
        name = clip(match.group(1).strip() if match else f"skill_{int(time.time() * 1000):x}", 80)
        accepted, baseline, candidate = await regression_gate(db, dsl, {str(item["id"]) for item in group}, {str(item["task_id"]) for item in group}, str(seed["component"]))
        await db.admin_insert("agent_skills", {"name": name, "dsl": dsl, "terms": " ".join(tokens(dsl)[:200]), "vector": centroid, "successes": 0, "failures": 0, "status": "active" if accepted else "rejected", "baseline_rate": baseline, "candidate_rate": candidate}, "minimal")
        vectors.append(centroid)
        await log(f"Új képesség „{name}”: {'aktiválva' if accepted else 'elutasítva'} (alap {baseline:.2f}, jelölt {candidate:.2f}).")


async def consolidate(db: Supabase, think: Think, input_data: dict[str, Any], log: Callable[[str], Awaitable[None]]) -> None:
    task_id = str(input_data["task_id"])
    objective = clip(str(input_data["objective"]), 4000)
    reflections: list[Reflection] = input_data.get("reflections", [])
    await db.admin_upsert("agent_tasks", {"id": task_id, "objective": objective, "outcome": str(input_data["outcome"]), "steps": int(input_data["steps"]), "verified": bool(input_data["verified"])}, "id", "minimal")
    if reflections:
        values = [{"task_id": task_id, "objective": objective, "failure_point": item.failure_point, "component": item.component, "root_cause": item.root_cause, "pivot_action": item.pivot_action, "patch": item.patch, "failed_action": item.failed_action, "corrective_action": item.corrective_action} for item in reflections]
        await db.admin_insert("agent_reflections", values, "minimal")
        await distill(db, [(item.failed_action, item.corrective_action, 1.0 if input_data["verified"] else 0.25) for item in reflections if item.corrective_action])
    for skill_id in set(input_data.get("used_skills", [])):
        await record_skill_outcome(db, str(skill_id), bool(input_data["verified"]))
    async with DB_LOCK:
        if reflections:
            await synthesize_skills(db, think, log)
        total, latest = await asyncio.gather(db.admin_count("agent_tasks"), db.admin_select("agent_playbooks", "task_count,content", order="created_at.desc", limit=1))
        last_count = int(latest[0]["task_count"]) if latest else 0
        if total - last_count < 5:
            return
        reflections_rows, skills_rows = await asyncio.gather(
            db.admin_select("agent_reflections", "component,root_cause,pivot_action", order="created_at.desc", limit=40),
            db.admin_select("agent_skills", "name,dsl", [eq("status", "active")], limit=20),
        )
        prompt = "\n".join(f"- [{row['component']}] {row['root_cause']} => {row['pivot_action']}" for row in reflections_rows)
        known = "\n\n".join(str(row["dsl"]) for row in skills_rows)
        previous = str(latest[0]["content"]) if latest else "nincs"
        content = (await think("Tapasztalati tudáskonszolidátor vagy. Írj tömör, magyar működési kézikönyvet két szakasszal: Bevált eljárások és Elkerülendő anti-patternek. Legfeljebb 40 pont.", f"ELŐZŐ KÉZIKÖNYV:\n{previous}\n\nREFLEXIÓK:\n{prompt}\n\nAKTÍV KÉPESSÉGEK:\n{known}")).strip()
        if not content:
            return
        inserted = await db.admin_insert("agent_playbooks", {"content": clip(content, 20000), "task_count": total})
        own_id = inserted[0].get("id") if inserted else None
        if own_id is not None:
            first = await db.admin_select("agent_playbooks", "id", [eq("task_count", total)], order="created_at.asc,id.asc", limit=1)
            oldest = first[0].get("id") if first else None
            if oldest is not None and str(oldest) != str(own_id):
                await db.admin_delete("agent_playbooks", [eq("id", str(own_id))])
                return
        await log("A működési kézikönyv frissült.")


@dataclass(slots=True)
class ReasonNode:
    question: str
    answer: str
    children: list["ReasonNode"]

    def data(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "children": [child.data() for child in self.children]}


async def reason_tree(think: Think, question: str, depth: int, branches: int) -> ReasonNode:
    if depth < 1 or depth > 3 or branches < 1 or branches > 3:
        raise AppError("INVALID_ARGUMENT", "A következtetési fa mélysége és elágazásszáma érvénytelen.", 400)
    async def build(current: str, remaining: int) -> ReasonNode:
        if remaining == 0:
            answer = await think("Izolált következtető vagy. Tömören, tényszerűen válaszolj a kérdésre; ne módosíts állapotot.", current)
            return ReasonNode(current, clip(answer, 3000), [])
        decomposition = await think(f"Bontsd a kérdést legfeljebb {branches} független, kisebb alkérdésre. Csak JSON tömböt adj stringekkel. Atomi kérdésnél üres tömböt adj.", current)
        questions = string_list(extract_json(decomposition), branches)
        children = await asyncio.gather(*(build(item, remaining - 1) for item in questions))
        answer = await think("Összegző vagy. Az alkérdések válaszaiból adj tömör, következetes választ az eredeti kérdésre.", f"EREDETI KÉRDÉS:\n{current}\n\nALVÁLASZOK:\n" + "\n".join(f"- {item.question}: {item.answer}" for item in children)) if children else await think("Izolált következtető vagy. Tömören, tényszerűen válaszolj a kérdésre.", current)
        return ReasonNode(current, clip(answer, 3000), list(children))
    return await build(question, depth)
