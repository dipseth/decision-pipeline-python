"""
The DAG: topological order, and the skip rule.

A node whose EVERY inbound guard is false is skipped, and the trace shows it
skipped WITH THE REASON — never missing. That is the difference between a
trace you can read six weeks later and one you cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Collection, Optional, Sequence

from .types import Edge


@dataclass(kw_only=True)
class TopoResult:
    order: list[str]
    #: Node ids that take part in a cycle. Empty when the graph is a DAG.
    cycle: list[str] = field(default_factory=list)


def topological_order(node_ids: Sequence[str], edges: Sequence[Edge]) -> TopoResult:
    """
    Kahn's algorithm, tie-broken by DECLARATION order so a run's node order is
    stable across processes — the run record is compared across runs.
    """
    indegree: dict[str, int] = {}
    out: dict[str, list[str]] = {}
    rank: dict[str, int] = {}

    for i, node_id in enumerate(node_ids):
        indegree[node_id] = 0
        out[node_id] = []
        rank[node_id] = i

    for edge in edges:
        if edge.from_ not in indegree or edge.to not in indegree:
            continue
        out[edge.from_].append(edge.to)
        indegree[edge.to] = indegree.get(edge.to, 0) + 1

    ready = [n for n in node_ids if indegree.get(n, 0) == 0]
    order: list[str] = []

    while ready:
        ready.sort(key=lambda n: rank.get(n, 0))
        node_id = ready.pop(0)
        order.append(node_id)
        for nxt in out.get(node_id, []):
            remaining = indegree.get(nxt, 0) - 1
            indegree[nxt] = remaining
            if remaining == 0:
                ready.append(nxt)

    placed = set(order)
    cycle = [n for n in node_ids if n not in placed]
    return TopoResult(order=order, cycle=cycle)


def inbound_edges(edges: Sequence[Edge], node_id: str) -> list[Edge]:
    return [e for e in edges if e.to == node_id]


def outbound_edges(edges: Sequence[Edge], node_id: str) -> list[Edge]:
    return [e for e in edges if e.from_ == node_id]


#: The branch a gate took, or None when the gate has not run (or was skipped).
BranchLookup = Callable[[str], Optional[str]]


@dataclass(kw_only=True)
class InboundDecision:
    #: Edges whose source ran AND whose guard (if any) matched.
    open: list[Edge]
    #: Why the node is being skipped, when `open` is empty. None when it runs.
    skip_reason: Optional[str] = None


def resolve_inbound(
    edges: Sequence[Edge],
    node_id: str,
    ran: Collection[str],
    branch_of: BranchLookup,
) -> InboundDecision:
    """
    A root node (no inbound edges) always runs. Otherwise at least one inbound
    edge has to be open: its source produced an output, and either it carries no
    guard or the named gate took the named branch.
    """
    inbound = inbound_edges(edges, node_id)
    if not inbound:
        return InboundDecision(open=[], skip_reason=None)

    open_: list[Edge] = []
    closed_because: list[str] = []

    for edge in inbound:
        if edge.from_ not in ran:
            closed_because.append(f"{edge.from_} skipped")
            continue
        if edge.when is None:
            open_.append(edge)
            continue
        taken = branch_of(edge.when.gate)
        if taken == edge.when.branch:
            open_.append(edge)
        else:
            shown = "no branch" if taken is None else f'"{taken}"'
            closed_because.append(f'{edge.when.gate} took {shown}, needed "{edge.when.branch}"')

    if open_:
        return InboundDecision(open=open_, skip_reason=None)
    return InboundDecision(open=open_, skip_reason="; ".join(closed_because) or "no open inbound edge")


def sink_nodes(node_ids: Sequence[str], edges: Sequence[Edge]) -> list[str]:
    """Nodes with no outbound flow edges — the graph's sinks."""
    has_out = {e.from_ for e in edges}
    return [n for n in node_ids if n not in has_out]
