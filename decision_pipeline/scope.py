"""
Scope resolution — the runtime's job, never the pipeline's.

Before node 0 the runtime walks `session -> device -> user -> associated ->
tenant -> global` for the caller and turns whatever rows come back into a
`ScopeContext`. Two things come out of it and nothing else:

    ctx.features(node_id)  all scopes CONCATENATE (features are content-free by
                           the form x scope rule, so nothing is dropped for being wide)
    ctx.override(key)      the SINGLE narrowest row, applied at read

A pipeline's code never reads a user id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .types import (
    DIRECT_FORM_MAX_SCOPE,
    FeatureRow,
    OverrideRow,
    ScopeContext,
    rank_index,
    scope_at_most,
    scope_index,
)


@dataclass(kw_only=True)
class ScopeContextInput:
    ids: Optional[Mapping[str, str]] = None
    grants: Optional[Sequence[str]] = None
    features: Optional[Sequence[Any]] = None
    overrides: Optional[Sequence[Any]] = None


@dataclass(kw_only=True)
class ScopeRejection:
    """Rows the form x scope rule refuses, kept so the runtime can record the drop."""

    id: str
    form: str  # "feature" | "override"
    scope: str
    reason: str = "form_scope_rule"


class BuiltScopeContextImpl:
    """The concrete `types.ScopeContext`: read-only ids and grants, scoped lookups, consumption tracking."""

    def __init__(
        self,
        ids: Mapping[str, str],
        grants: Sequence[str],
        by_node: dict[str, list[FeatureRow]],
        overrides: list[OverrideRow],
    ):
        self._ids = MappingProxyType(dict(ids))
        self._grants = tuple(grants)
        self._by_node = by_node
        self._overrides = overrides
        self._consumed: dict[str, None] = {}

    @property
    def ids(self) -> Mapping[str, str]:
        return self._ids

    @property
    def grants(self) -> tuple[str, ...]:
        return self._grants

    def features(self, node_id: str) -> list[FeatureRow]:
        rows = self._by_node.get(node_id, [])
        for row in rows:
            self._consumed[row.id] = None
        return list(rows)

    def override(self, key: str) -> Optional[OverrideRow]:
        # Already sorted narrowest-first, so the first match IS the narrowest.
        for row in self._overrides:
            if row.key == key:
                self._consumed[row.id] = None
                return row
        return None

    def consumed(self) -> list[str]:
        return list(self._consumed)


@dataclass(kw_only=True)
class BuiltScopeContext:
    ctx: ScopeContext
    rejected: list[ScopeRejection] = field(default_factory=list)


_SNAKE = {"nodeId": "node_id", "createdAt": "created_at"}


def _row(row: Any, cls: type) -> Any:
    if isinstance(row, cls):
        return row
    if isinstance(row, Mapping):
        return cls(**{_SNAKE.get(k, k): v for k, v in row.items()})  # type: ignore[arg-type]
    return row


def _sort_key(row: Any) -> tuple[int, int, str]:
    """Narrowest scope first; within a scope the stronger rank first; then oldest first."""
    return (scope_index(row.scope), -rank_index(row.rank), row.created_at or "")


def _allowed_direct_form(scope: str) -> bool:
    """
    The form x scope rule, enforced at read as well as at manifest load.
    A manifest can only declare edges; rows outlive manifests, so a row that
    arrives from a scope the rule forbids is dropped here and recorded — never
    quietly honoured.
    """
    return scope_at_most(scope, DIRECT_FORM_MAX_SCOPE)


def build_scope_context(input: Any = None) -> BuiltScopeContext:
    if input is None:
        input = ScopeContextInput()
    elif isinstance(input, Mapping):
        input = ScopeContextInput(
            ids=input.get("ids"),
            grants=input.get("grants"),
            features=input.get("features"),
            overrides=input.get("overrides"),
        )

    rejected: list[ScopeRejection] = []

    features: list[FeatureRow] = []
    for row in (_row(r, FeatureRow) for r in (input.features or [])):
        if not _allowed_direct_form(row.scope):
            rejected.append(ScopeRejection(id=row.id, form="feature", scope=row.scope, reason="form_scope_rule"))
            continue
        features.append(row)
    features.sort(key=_sort_key)

    overrides: list[OverrideRow] = []
    for row in (_row(r, OverrideRow) for r in (input.overrides or [])):
        if not _allowed_direct_form(row.scope):
            rejected.append(ScopeRejection(id=row.id, form="override", scope=row.scope, reason="form_scope_rule"))
            continue
        overrides.append(row)
    overrides.sort(key=_sort_key)

    by_node: dict[str, list[FeatureRow]] = {}
    for row in features:
        by_node.setdefault(row.node_id, []).append(row)

    ctx = BuiltScopeContextImpl(input.ids or {}, input.grants or [], by_node, overrides)
    return BuiltScopeContext(ctx=ctx, rejected=rejected)


def empty_scope_context() -> ScopeContext:
    """An empty context — what an unauthenticated or unscoped run gets."""
    return build_scope_context().ctx
