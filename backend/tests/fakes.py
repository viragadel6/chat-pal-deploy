from __future__ import annotations

import re
from typing import Any


class FakeResponse:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        count: int | None = None,
        data: Any = None,
    ):
        self.rows = rows if rows is not None else []
        self.count = count
        self.data = data


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == "eq":
        return left == right
    if operator == "neq":
        return left != right
    if operator == "gt":
        return left is not None and right is not None and left > right
    if operator == "gte":
        return left is not None and right is not None and left >= right
    if operator == "lt":
        return left is not None and right is not None and left < right
    if operator == "lte":
        return left is not None and right is not None and left <= right
    if operator == "like":
        return isinstance(left, str) and re.match(
            re.escape(str(right)).replace("%", ".*").replace("_", "."), left
        ) is not None
    if operator == "ilike":
        return isinstance(left, str) and re.match(
            re.escape(str(right)).replace("%", ".*").replace("_", "."), left.lower()
        ) is not None
    if operator == "is":
        if right is None:
            return left is None
        return left == right
    if operator == "in":
        return left in list(right or [])
    return False


def _sort_key(row: dict[str, Any], column: str) -> tuple[int, str, float]:
    value = row.get(column)
    if value is None:
        return (1, "", 0.0)
    if isinstance(value, bool):
        return (0, "", float(value))
    if isinstance(value, (int, float)):
        return (0, "", float(value))
    return (0, str(value), 0.0)


class FakeQuery:
    def __init__(self, table: FakeTable):
        self._table = table
        self._method = "GET"
        self._payload: Any = None
        self._filters: list[tuple[str, str, Any]] = []
        self._order: list[tuple[str, bool]] = []
        self._limit: int | None = None
        self._offset: int | None = None
        self._columns: str | None = None
        self._on_conflict: str | None = None
        self._prefer: str | None = None

    def select(self, *columns: str, count: str | None = None, head: bool = False) -> "FakeQuery":
        self._method = "GET"
        self._columns = ",".join(columns) if columns else "*"
        return self

    def insert(self, payload) -> "FakeQuery":
        self._method = "POST"
        self._payload = payload
        self._prefer = "return=representation"
        return self

    def upsert(self, payload, on_conflict: str = "", ignore_duplicates: bool = False) -> "FakeQuery":
        self._method = "POST"
        self._payload = payload
        self._on_conflict = on_conflict
        self._prefer = "return=representation"
        return self

    def update(self, payload) -> "FakeQuery":
        self._method = "PATCH"
        self._payload = payload
        self._prefer = "return=representation"
        return self

    def delete(self) -> "FakeQuery":
        self._method = "DELETE"
        self._prefer = "return=representation"
        return self

    def eq(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("eq", column, value))
        return self

    def neq(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("neq", column, value))
        return self

    def gt(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("gt", column, value))
        return self

    def gte(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("gte", column, value))
        return self

    def lt(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("lt", column, value))
        return self

    def lte(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("lte", column, value))
        return self

    def like(self, column: str, pattern: str) -> "FakeQuery":
        self._filters.append(("like", column, pattern))
        return self

    def ilike(self, column: str, pattern: str) -> "FakeQuery":
        self._filters.append(("ilike", column, pattern))
        return self

    def is_(self, column: str, value: Any) -> "FakeQuery":
        self._filters.append(("is", column, value))
        return self

    def in_(self, column: str, values) -> "FakeQuery":
        self._filters.append(("in", column, list(values)))
        return self

    def order(self, column: str, desc: bool = False, nullsfirst: bool = False) -> "FakeQuery":
        self._order.append((column, desc))
        return self

    def limit(self, count: int) -> "FakeQuery":
        self._limit = count
        return self

    def offset(self, count: int) -> "FakeQuery":
        self._offset = count
        return self

    def range(self, start: int, end: int) -> "FakeQuery":
        self._offset = start
        self._limit = max(0, end - start)
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        for operator, column, value in self._filters:
            if not _compare(row.get(column), operator, value):
                return False
        return True

    def _sorted(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = list(rows)
        for column, desc in reversed(self._order):
            ordered.sort(key=lambda row: _sort_key(row, column), reverse=desc)
        return ordered

    def _page(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._offset is not None:
            rows = rows[self._offset :]
        if self._limit is not None:
            rows = rows[: self._limit]
        return rows

    async def execute(self) -> FakeResponse:
        return self._table._execute(self)

    async def count_exact(self) -> int:
        return self._table._count(self)


class FakeTable:
    def __init__(self, database: "FakeDatabase", name: str):
        self._database = database
        self._name = name

    def _execute(self, query: FakeQuery) -> FakeResponse:
        handler = self._database.handlers.get((self._name, query._method))
        if handler is not None:
            return handler(query)
        if self._database.stateful:
            return self._database.store.apply(self._name, query)
        return FakeResponse([])

    def _count(self, query: FakeQuery) -> int:
        handler = self._database.handlers.get((self._name, "COUNT"))
        if handler is not None:
            return int(handler(query))
        if self._database.stateful:
            return len(self._database.store.matching(self._name, query))
        return 0

    def select(self, *columns: str, count: str | None = None, head: bool = False) -> FakeQuery:
        return FakeQuery(self).select(*columns, count=count, head=head)

    def insert(self, payload) -> FakeQuery:
        return FakeQuery(self).insert(payload)

    def upsert(self, payload, on_conflict: str = "", ignore_duplicates: bool = False) -> FakeQuery:
        return FakeQuery(self).upsert(payload, on_conflict=on_conflict, ignore_duplicates=ignore_duplicates)

    def update(self, payload) -> FakeQuery:
        return FakeQuery(self).update(payload)

    def delete(self) -> FakeQuery:
        return FakeQuery(self).delete()


class FakeRpc:
    def __init__(self, database: "FakeDatabase", function: str, payload: dict[str, Any] | None):
        self._database = database
        self._function = function
        self._payload = dict(payload or {})

    async def execute(self) -> FakeResponse:
        handler = self._database.rpc_handlers.get(self._function)
        if handler is None:
            return FakeResponse([])
        return handler(self._payload)


class MemoryStore:
    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.sequence: dict[str, int] = {}

    def table(self, name: str) -> list[dict[str, Any]]:
        return self.rows.setdefault(name, [])

    def next_id(self, name: str) -> int:
        value = self.sequence.get(name, 0) + 1
        self.sequence[name] = value
        return value

    def matching(self, name: str, query: FakeQuery) -> list[dict[str, Any]]:
        rows = [row for row in self.table(name) if query._matches(row)]
        return query._sorted(rows)

    def apply(self, name: str, query: FakeQuery) -> FakeResponse:
        if query._method == "POST":
            return self._insert(name, query)
        if query._method == "PATCH":
            return self._update(name, query)
        if query._method == "DELETE":
            return self._delete(name, query)
        rows = query._page(self.matching(name, query))
        return FakeResponse([dict(row) for row in rows], count=len(self.matching(name, query)))

    def _insert(self, name: str, query: FakeQuery) -> FakeResponse:
        payload = query._payload
        items = payload if isinstance(payload, list) else [payload]
        inserted: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if "id" not in row:
                row["id"] = self.next_id(name)
            if query._on_conflict:
                existing = next(
                    (
                        candidate
                        for candidate in self.table(name)
                        if candidate.get("id") == row.get("id")
                    ),
                    None,
                )
                if existing is not None:
                    existing.update(row)
                    inserted.append(existing)
                    continue
            self.table(name).append(row)
            inserted.append(row)
        return FakeResponse([dict(row) for row in inserted])

    def _update(self, name: str, query: FakeQuery) -> FakeResponse:
        updated: list[dict[str, Any]] = []
        for row in self.table(name):
            if not query._matches(row):
                continue
            if isinstance(query._payload, dict):
                row.update(query._payload)
            updated.append(row)
        return FakeResponse([dict(row) for row in updated])

    def _delete(self, name: str, query: FakeQuery) -> FakeResponse:
        removed: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for row in self.table(name):
            if query._matches(row):
                removed.append(row)
            else:
                remaining.append(row)
        self.rows[name] = remaining
        return FakeResponse([dict(row) for row in removed])


class FakeDatabase:
    def __init__(self, stateful: bool = False) -> None:
        self.handlers: dict[tuple[str, str], Any] = {}
        self.rpc_handlers: dict[str, Any] = {}
        self.calls: list[tuple[str, str, Any]] = []
        self.stateful = stateful
        self.store = MemoryStore()

    def table(self, name: str) -> FakeTable:
        return FakeTable(self, name)

    def rpc(self, function: str, payload: dict[str, Any] | None = None) -> FakeRpc:
        return FakeRpc(self, function, payload)

    def register(self, table: str, method: str, handler) -> None:
        self.handlers[(table, method)] = handler

    def register_rpc(self, function: str, handler) -> None:
        self.rpc_handlers[function] = handler
