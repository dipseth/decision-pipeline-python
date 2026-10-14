"""
Tool resolution for a `generate` node: `static u selected`, bounded by
`allow`, minus every tool whose MCP scope the run was not granted.

The registry itself is NOT here — the host owns it. This module adds selection
and accounting on top, because both belong to the pipeline layer.

Rule, same as branching: a `decide` node never attaches a tool. It emits one
Noul per candidate; a `code` gate turns those into a subset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from .types import ToolSpec, tool_spec_from_dict


class ToolRegistry(Protocol):
    """The host's view of the global registry. Only what selection needs; both methods are optional."""

    def has(self, name: str) -> bool:
        """Does this name resolve to a real tool? Unknown names are dropped, loudly."""
        ...

    def scope_of(self, name: str) -> Optional[str]:
        """
        The MCP scope a tool needs (`recipes:read` / `list:sync` / `cook:assist`),
        or None when it needs none. A run can never offer a tool wider than its grant.
        """
        ...


@dataclass(kw_only=True)
class ResolveToolsInput:
    spec: Optional[ToolSpec] = None
    #: What `static: "prompt-config"` resolves to — the Langfuse prompt's `config.tools`.
    prompt_config_tools: Optional[Sequence[str]] = None
    #: The preceding decide node's distributions, for `selectable`.
    distributions: Optional[Mapping[str, Sequence[float]]] = None
    #: Resolved value of `selectable.threshold`.
    select_threshold: Optional[float] = None
    grants: Optional[Sequence[str]] = None
    registry: Optional[ToolRegistry] = None


@dataclass(kw_only=True)
class ResolvedTools:
    offered: list[str] = field(default_factory=list)
    #: Selected candidate -> the Noul that selected it. Lands on the run record.
    selected: dict[str, float] = field(default_factory=dict)
    dropped_for_scope: list[str] = field(default_factory=list)
    dropped_for_allow: list[str] = field(default_factory=list)
    dropped_unknown: list[str] = field(default_factory=list)


DEFAULT_TOOL_SELECT_THRESHOLD = 0.5


def tool_noul(distributions: Optional[Mapping[str, Sequence[float]]], candidate: str) -> Optional[float]:
    """
    Where a candidate tool's Noul lives in the decide node's distributions.
    `tool:<name>` is the convention; a bare `<name>` key is accepted so a decide
    node that already asks about a tool by name needs no rewrite.
    """
    if not distributions:
        return None
    dist = distributions.get(f"tool:{candidate}")
    if dist is None:
        dist = distributions.get(candidate)
    if not dist:
        return None
    # A Noul is one absolute probability. When a Choice is used instead, the
    # FIRST entry is the "yes" option by convention, matching how the gate on a
    # two-option Choice already reads.
    p = dist[0]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
        return None
    return p


def resolve_tools(input: Optional[ResolveToolsInput] = None, **kwargs: Any) -> ResolvedTools:
    if input is None:
        input = ResolveToolsInput(**kwargs)
    elif kwargs:
        raise TypeError("resolve_tools takes an input or keyword arguments, not both")

    spec = tool_spec_from_dict(input.spec)
    registry = input.registry
    out = ResolvedTools()
    if spec is None:
        return out

    if spec.static == "prompt-config":
        base = list(input.prompt_config_tools or [])
    else:
        base = list(spec.static or [])

    threshold = DEFAULT_TOOL_SELECT_THRESHOLD if input.select_threshold is None else input.select_threshold
    candidates = spec.selectable.candidates if spec.selectable else []
    for candidate in candidates:
        p = tool_noul(input.distributions, candidate)
        if p is not None and p >= threshold:
            out.selected[candidate] = p

    allow = set(spec.allow) if spec.allow is not None else None
    grants = set(input.grants or [])
    seen: set[str] = set()

    has = getattr(registry, "has", None)
    scope_of = getattr(registry, "scope_of", None)

    for name in [*base, *out.selected]:
        if name in seen:
            continue
        seen.add(name)

        # `allow` is validated at manifest load for names the manifest itself
        # lists; a prompt-config set resolves at RUN time, so it is checked here
        # too rather than trusted.
        if allow is not None and name not in allow:
            out.dropped_for_allow.append(name)
            out.selected.pop(name, None)
            continue
        if has is not None and not has(name):
            out.dropped_unknown.append(name)
            out.selected.pop(name, None)
            continue
        needed = scope_of(name) if scope_of is not None else None
        if needed and needed not in grants:
            out.dropped_for_scope.append(name)
            out.selected.pop(name, None)
            continue
        out.offered.append(name)

    return out
