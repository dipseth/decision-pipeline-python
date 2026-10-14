"""
The run record — the training artifact. One schema for every pipeline, so
that three pipelines emitting it makes a shared model a training-script
change rather than an architecture change.

`distributions` and `input_hash` are MANDATORY on every decide node: they
are what lets a threshold be re-tuned by offline replay with zero model
calls, and what makes the cheapest kind of experiment free.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .probes import ProbeRecord


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@dataclass(kw_only=True)
class RunRecordNode:
    id: str
    kind: str
    version: str
    input_hash: str
    ms: float
    cost_usd: float
    #: `decide`/`generate` nodes that RAN: whether the port reported a cost at all.
    #: "did not report" and "reported zero" are different facts — `cost_usd: 0` alone
    #: collapses them. Absent on every other node (#325).
    cost_reported: Optional[bool] = None
    #: Present ONLY when the node's output came from `cache`: no port was called.
    cached: Optional[bool] = None
    #: Present ONLY when the node did not run; the value is the reason.
    skipped: Optional[str] = None
    #: Gate nodes: the branch taken.
    branch: Optional[str] = None
    #: Gate nodes: WHY that branch, when the body said so.
    branch_reason: Optional[str] = None
    route: Optional[str] = None
    #: decide nodes: the raw Jev output. The decide log.
    distributions: Optional[dict[str, list[float]]] = None
    #: decide nodes: the option keys behind each distribution, in the same order as its numbers.
    #: A bare number list is only replayable if the option ORDER is recoverable, and
    #: prompt versions move (#318).
    distribution_options: Optional[dict[str, list[str]]] = None
    #: decide nodes with `probes`: the TEXT of every probe asked, what was dropped and why,
    #: and which went unanswered.
    probes: Optional["ProbeRecord"] = None
    #: gate nodes: the values actually used, after env / prompt-config override.
    thresholds: Optional[dict[str, float]] = None
    #: generate nodes: `{"offered", "selected", "called", "dropped_for_scope"}`.
    tools: Optional[dict[str, Any]] = None
    #: decide nodes that batch (cuisine sends <=12 recipes per request).
    batch_trace_id: Optional[str] = None
    #: Present when the node failed and its on_failure absorbed it.
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON like the TS record: optional fields that are None are omitted."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if f.name == "probes" and hasattr(v, "to_dict"):
                v = v.to_dict()
            else:
                v = copy.deepcopy(v)
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecordNode":
        names = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in d.items() if k in names}
        probes = data.get("probes")
        if isinstance(probes, Mapping):
            from .probes import ProbeRecord

            data["probes"] = ProbeRecord.from_dict(probes)
        return cls(**data)


@dataclass(kw_only=True)
class RunRecord:
    pipeline: str
    #: Composite hash: node versions + prompt versions + thresholds + knowledge + ARMS.
    version: str
    #: This ATTEMPT's id — the root trace id. A retry or a resume gets a new one.
    run_id: str
    #: The DURABLE identity of the work, stable across every attempt at it. Falls back to
    #: `run_id` when the host has no stable key.
    run_key: str
    #: The Langfuse grouping key.
    group: str
    tenant: Optional[str]
    user: Optional[str]
    trigger: str
    input_hash: str
    cost_usd: float
    ms: float
    nodes: list[RunRecordNode]
    #: Flattened and pipeline-agnostic: gate margins, entropies, item counts, lang, factor...
    features: dict[str, float]
    fact: str
    fact_route: str
    fact_confidence: Optional[float]
    #: Provenance both ways: which feature/override rows this run READ.
    consumed_feedback_ids: list[str]

    # --- experiments. Unbackfillable, so they land in the first migration step.
    #: experiment id -> arm. A MAP, so cross-pipeline interaction stays detectable.
    arms: dict[str, str]
    #: An arm rides down into sub-pipelines; this is the edge that attributes an outer label to an inner arm.
    parent_run_id: Optional[str]
    #: The scope an arm was assigned at, and the hash of the id it was assigned from.
    assignment_unit: Optional[str]
    assignment_hash: Optional[str]

    created_at: str

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "nodes":
                out["nodes"] = [n.to_dict() if isinstance(n, RunRecordNode) else copy.deepcopy(n) for n in v]
            else:
                out[f.name] = copy.deepcopy(v)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunRecord":
        names = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in d.items() if k in names}
        data["nodes"] = [n if isinstance(n, RunRecordNode) else RunRecordNode.from_dict(n) for n in data.get("nodes", [])]
        return cls(**data)


@dataclass(kw_only=True)
class RunLabel:
    #: The attempt the signal was collected against — where the score lives.
    run_id: str
    #: The durable run identity, when the host had one. THIS is what a training join should key on.
    run_key: Optional[str] = None
    kind: Literal["score", "override_disagreement", "verify", "implicit"]
    value: float
    #: The CATEGORICAL label behind `value`, when there is one.
    label: Optional[str] = None
    #: The Langfuse score name, so a label row joins back to the score.
    score_name: Optional[str] = None
    #: The feedback edge that produced this label.
    edge_id: Optional[str] = None
    scope: str
    rank: str
    source_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(dataclasses.asdict(self))


# ---------------------------------------------------------------------------
# Gate metrics — named extractors, not ad-hoc queries
# ---------------------------------------------------------------------------

#: `eval.gate_metric` names one of these. The point of a registry rather than a
#: query string: the offline eval and an online split then compute the SAME
#: number, so a disagreement is the change under test and not the metric.
#:
#: Returns None when the run cannot contribute to the metric (no label yet,
#: gate skipped) — a None is dropped from the aggregate, never counted as 0.
GateMetric = Callable[[RunRecord, Sequence[RunLabel]], Optional[float]]

_METRICS: dict[str, GateMetric] = {}


def define_gate_metric(name: str, fn: GateMetric) -> str:
    existing = _METRICS.get(name)
    if existing is not None and existing is not fn:
        raise ValueError(f'Gate metric "{name}" is already defined with a different body.')
    _METRICS[name] = fn
    return name


def get_gate_metric(name: str) -> Optional[GateMetric]:
    return _METRICS.get(name)


def has_gate_metric(name: str) -> bool:
    return name in _METRICS


def gate_metric_names() -> list[str]:
    return sorted(_METRICS.keys())


def clear_gate_metrics() -> None:
    """Test seam — the registry is process-global, so a suite has to be able to reset it."""
    _METRICS.clear()


def _row_field(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def aggregate_gate_metric(name: str, rows: Sequence[Any]) -> dict[str, Any]:
    """`rows` are `{"record": RunRecord, "labels": [RunLabel]}` dicts (or objects with those attributes)."""
    fn = _METRICS.get(name)
    if fn is None:
        raise ValueError(f'Unknown gate metric "{name}".')
    values: list[float] = []
    for row in rows:
        v = fn(_row_field(row, "record"), _row_field(row, "labels"))
        if v is not None and math.isfinite(v):
            values.append(v)
    if not values:
        return {"name": name, "n": 0, "mean": None}
    return {"name": name, "n": len(values), "mean": sum(values) / len(values)}


# ---------------------------------------------------------------------------
# Two metrics every pipeline can use, so `eval.gate_metric` is never a stub
# ---------------------------------------------------------------------------


def _score_positive(_record: RunRecord, labels: Sequence[RunLabel]) -> Optional[float]:
    scores = [lb for lb in labels if lb.kind == "score"]
    if not scores:
        return None
    # Strongest rank wins; admin over human over llm, as everywhere else.
    return 1 if any(lb.value > 0 for lb in scores) else 0


def _no_override_disagreement(_record: RunRecord, labels: Sequence[RunLabel]) -> Optional[float]:
    seen = [lb for lb in labels if lb.kind == "override_disagreement"]
    if not seen:
        return None
    return 0 if any(lb.value != 0 for lb in seen) else 1


#: Share of runs whose explicit score label was positive.
GATE_METRIC_SCORE_POSITIVE = define_gate_metric("score_positive", _score_positive)

#: Share of runs a human did NOT have to correct.
GATE_METRIC_NO_DISAGREEMENT = define_gate_metric("no_override_disagreement", _no_override_disagreement)
