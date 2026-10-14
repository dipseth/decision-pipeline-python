"""
Applying a `ManifestPatch` — how an arm, or an offline replay, becomes a
runnable manifest.

Only the declarative half moves. There is no way to patch a `run:` body,
and that is the point: a variant that needs new code is a new NODE VERSION,
not a variant.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ._compat import js_number
from .types import Edge, EdgeGuard, has_thresholds, tool_spec_from_dict

if TYPE_CHECKING:  # pragma: no cover
    from .experiments import ManifestPatch


@dataclass(kw_only=True)
class PatchedManifest:
    nodes: dict[str, Any]
    edges: list[Edge]
    #: Human-readable arm diff — the same list the run record and a review see.
    applied: list[str]


def _get(obj: Any, name: str, *aliases: str) -> Any:
    """Attribute (or mapping key) of a patch or a guard; None when absent."""
    for n in (name, *aliases):
        v = obj.get(n) if isinstance(obj, Mapping) else getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _show(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return js_number(value)
    return str(value)


def apply_manifest_patch(
    nodes: Mapping[str, Any],
    edges: list[Edge],
    patch: Optional["ManifestPatch"] = None,
) -> PatchedManifest:
    applied: list[str] = []
    fields = ("thresholds", "prompts", "routes", "tools", "questions", "guards")
    if patch is None or all(_get(patch, f) is None for f in fields):
        return PatchedManifest(nodes=dict(nodes), edges=[dataclasses.replace(e) for e in edges], applied=applied)

    nxt: dict[str, Any] = dict(nodes)

    def clone(node_id: str) -> Any:
        n = nxt.get(node_id)
        if n is None:
            return None
        copy = dataclasses.replace(n)
        nxt[node_id] = copy
        return copy

    for node_id, values in (_get(patch, "thresholds") or {}).items():
        n = clone(node_id)
        if n is None or not has_thresholds(n):
            continue
        merged = dict(n.thresholds)
        for name, value in values.items():
            merged[name] = value
            applied.append(f"{node_id}.thresholds.{name}={_show(value)}")
        n.thresholds = merged

    for node_id, prompt in (_get(patch, "prompts") or {}).items():
        n = clone(node_id)
        if n is None or n.kind != "generate":
            continue
        n.prompt = prompt
        applied.append(f"{node_id}.prompt={prompt}")

    for node_id, route in (_get(patch, "routes") or {}).items():
        n = clone(node_id)
        if n is None or n.kind != "generate":
            continue
        n.route = route
        applied.append(f"{node_id}.route={route}")

    for node_id, tools in (_get(patch, "tools") or {}).items():
        n = clone(node_id)
        if n is None or n.kind != "generate":
            continue
        n.tools = tool_spec_from_dict(tools)
        applied.append(f"{node_id}.tools")

    for node_id, questions in (_get(patch, "questions") or {}).items():
        n = clone(node_id)
        if n is None or n.kind != "decide":
            continue
        n.questions = questions
        applied.append(f"{node_id}.questions={questions}")

    next_edges = [dataclasses.replace(e) for e in edges]
    for guard in _get(patch, "guards") or []:
        g_from = _get(guard, "from_", "from")
        g_to = _get(guard, "to")
        edge = next((e for e in next_edges if e.from_ == g_from and e.to == g_to), None)
        if edge is None:
            continue
        when = _get(guard, "when")
        if when is None:
            edge.when = None
            applied.append(f"{g_from}->{g_to}.when=none")
        else:
            gate, branch = _get(when, "gate"), _get(when, "branch")
            edge.when = EdgeGuard(gate=gate, branch=branch)
            applied.append(f"{g_from}->{g_to}.when={gate}:{branch}")

    return PatchedManifest(nodes=nxt, edges=next_edges, applied=applied)
