"""
`define_pipeline` — the manifest, and the contract check that makes it one.

The seven clauses of the definition are not prose here; six of the seven are
checked, and a manifest that fails any of them will not load. Clause 7 (one
run = one trace) is the runtime's.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from ._compat import as_schema, is_schema
from .bindings import binding_problems
from .errors import PipelineContractError
from .experiments import ExperimentSpec, ManifestPatch, experiment_spec_from_dict, widest_direct_feedback_scope
from .graph import topological_order
from .record import gate_metric_names, has_gate_metric
from .review import REVIEW_QUEUE_BUDGET, has_lane, lane_names
from .schema import declaration_problems, plain
from .types import (
    DIRECT_FORM_MAX_SCOPE,
    CodeNode,
    DecideNode,
    Edge,
    FeedbackEdge,
    GateNode,
    GenerateNode,
    ReadNode,
    ScopeContext,
    StoreNode,
    edge_from_dict,
    feedback_edge_from_dict,
    has_thresholds,
    is_gate_node,
    node_from_dict,
    scope_at_most,
    scope_index,
)


@dataclass(kw_only=True)
class EvalSpec:
    """Mandatory: a gate without a frozen eval cannot be tuned."""

    dataset: str
    gate_metric: str


@dataclass(kw_only=True)
class Overridable:
    scope: str
    #: The authoritative read-time key. Canonical, never a point id.
    key: Callable[[Any], str]
    #: The same key when it is derivable from the INPUT.
    key_from_input: Optional[Callable[[Any], str]] = None


@dataclass(kw_only=True)
class Precedence:
    over: list[str]


@dataclass(kw_only=True)
class FeatureExtractionArgs:
    input: Any
    #: Every node's output, by node id. A skipped node is absent.
    outputs: Mapping[str, Any]
    #: Gate node id -> branch taken.
    branches: Mapping[str, str]
    ctx: ScopeContext


@dataclass(kw_only=True)
class PipelineSpec:
    id: str
    #: The persisted field name. Metadata is `<fact>_*`.
    fact: str
    #: A pydantic model class, a TypeAdapter, or anything with `parse()`.
    input: Any
    output: Any
    trigger: list[str]
    #: Grouping key for the tracer, e.g. `cook-<recipeId>`. Named `group`, not `session`.
    group: Callable[[Any], str]
    #: Node id -> node (a dataclass from `types`, or a dict literal).
    nodes: dict[str, Any]
    edges: list[Any]
    feedback: list[Any]
    #: Which node's output IS the fact. A list when branch-dependent: the FIRST one that ran wins.
    result: Union[str, list[str]]
    overridable: Optional[Overridable] = None
    eval: Any
    experiments: Optional[list[Any]] = None
    #: Rank this pipeline writes at. Default `llm`; a code-only pipeline is `heuristic`.
    decided_by: Optional[str] = None
    precedence: Optional[Precedence] = None
    #: Domain features for the run record, merged over the pipeline-agnostic ones.
    features: Optional[Callable[[FeatureExtractionArgs], dict[str, float]]] = None
    #: Folded into the composite version hash.
    knowledge_version: Optional[str] = None


@dataclass(kw_only=True)
class DecisionPipeline(PipelineSpec):
    result: list[str]  # type: ignore[assignment]
    decided_by: str  # type: ignore[assignment]
    eval: EvalSpec
    #: Topological node order, tie-broken by declaration order. Computed once.
    order: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """The declarative half as plain data (the TS `PipelineJSON` shape). JSON is an EXPORT, never the source."""
        return {
            "id": self.id,
            "fact": self.fact,
            "trigger": list(self.trigger),
            "result": list(self.result),
            "decidedBy": self.decided_by,
            "overridable": {"scope": self.overridable.scope} if self.overridable else None,
            "eval": {"dataset": self.eval.dataset, "gateMetric": self.eval.gate_metric},
            "precedence": {"over": list(self.precedence.over)} if self.precedence else None,
            "nodes": [_node_to_json(nid, n) for nid, n in self.nodes.items()],
            "edges": [_edge_to_json(e) for e in self.edges],
            "feedback": [_feedback_to_json(f) for f in self.feedback],
            "experiments": [_experiment_to_json(e) for e in (self.experiments or [])],
            "order": list(self.order),
        }


# ---------------------------------------------------------------------------
# Normalisation: dict literals → dataclasses
# ---------------------------------------------------------------------------


def _eval_spec(value: Any) -> Any:
    if value is None or isinstance(value, EvalSpec):
        return value
    if isinstance(value, Mapping):
        return EvalSpec(dataset=value.get("dataset"), gate_metric=value.get("gate_metric", value.get("gateMetric")))  # type: ignore[arg-type]
    return value


def _overridable(value: Any) -> Any:
    if value is None or isinstance(value, Overridable):
        return value
    if isinstance(value, Mapping):
        return Overridable(scope=value.get("scope"), key=value.get("key"), key_from_input=value.get("key_from_input", value.get("keyFromInput")))  # type: ignore[arg-type]
    return value


def _precedence(value: Any) -> Any:
    if value is None or isinstance(value, Precedence):
        return value
    if isinstance(value, Mapping):
        return Precedence(over=list(value.get("over", [])))
    return value


def normalize_spec(spec: Union[PipelineSpec, Mapping[str, Any]]) -> tuple[PipelineSpec, list[str]]:
    """Accept dataclasses or dict literals everywhere; return a spec of dataclasses plus any shape problems met on the way."""
    problems: list[str] = []
    if isinstance(spec, Mapping):
        spec = PipelineSpec(**{_CAMEL_SPEC.get(k, k): v for k, v in spec.items()})
    nodes: dict[str, Any] = {}
    for nid, n in (spec.nodes or {}).items():
        try:
            nodes[nid] = node_from_dict(n)
        except (TypeError, KeyError) as err:
            problems.append(f'node "{nid}": {err}')
            nodes[nid] = n
    edges = []
    for e in spec.edges or []:
        try:
            edges.append(edge_from_dict(e))
        except (TypeError, KeyError) as err:
            problems.append(f"edge: {err}")
    feedback = []
    for f in spec.feedback or []:
        try:
            feedback.append(feedback_edge_from_dict(f))
        except (TypeError, KeyError) as err:
            problems.append(f"feedback: {err}")
    experiments = None
    if spec.experiments is not None:
        experiments = []
        for x in spec.experiments:
            try:
                experiments.append(experiment_spec_from_dict(x))
            except (TypeError, KeyError) as err:
                problems.append(f"experiment: {err}")
    normalized = dataclasses.replace(
        spec,
        nodes=nodes,
        edges=edges,
        feedback=feedback,
        experiments=experiments,
        eval=_eval_spec(spec.eval),
        overridable=_overridable(spec.overridable),
        precedence=_precedence(spec.precedence),
    )
    return normalized, problems


_CAMEL_SPEC = {"decidedBy": "decided_by", "knowledgeVersion": "knowledge_version"}

# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def ancestors_of(edges: Sequence[Edge], node_id: str) -> set[str]:
    """Every node id reachable backwards from `node_id` along flow edges."""
    seen: set[str] = set()
    stack = [node_id]
    while stack:
        current = stack.pop()
        for edge in edges:
            if edge.to != current or edge.from_ in seen:
                continue
            seen.add(edge.from_)
            stack.append(edge.from_)
    return seen


def _as_list(value: Union[str, Sequence[str]]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _declaration(spec: PipelineSpec) -> dict[str, Any]:
    """What pydantic sees. Dataclass nodes become dicts of their fields; bodies ride along and are ignored."""
    nodes = {}
    for nid, n in spec.nodes.items():
        d = plain(n) if dataclasses.is_dataclass(n) else dict(n) if isinstance(n, Mapping) else n
        if isinstance(d, dict):
            for key in ("load", "run", "interpret", "fallback"):
                d.pop(key, None)
            if n.__class__ is GateNode or (isinstance(d, dict) and d.get("role") == "gate"):
                d["role"] = "gate"
        nodes[nid] = d
    return {
        "id": spec.id,
        "fact": spec.fact,
        "trigger": spec.trigger,
        "nodes": nodes,
        "edges": [plain(e) for e in spec.edges],
        "feedback": [{**plain(f), "key": None} for f in spec.feedback],
        "result": spec.result,
        "overridable": {"scope": spec.overridable.scope} if spec.overridable else None,
        "eval": plain(spec.eval),
        "experiments": [plain(x) for x in spec.experiments] if spec.experiments is not None else None,
        "decided_by": spec.decided_by,
        "precedence": plain(spec.precedence),
        "knowledge_version": spec.knowledge_version,
    }


def validate_pipeline(spec: Union[PipelineSpec, Mapping[str, Any]]) -> list[str]:
    """Returns every contract violation, rather than the first — a manifest author should see the whole list."""
    spec, problems = normalize_spec(spec)

    # --- the declarative half, through pydantic
    problems.extend(declaration_problems(_declaration(spec)))

    node_ids = list(spec.nodes.keys())
    node = spec.nodes.get

    if len(node_ids) == 0:
        problems.append("no nodes declared")
    if not callable(spec.group):
        problems.append("`group(input)` is required")
    if not is_schema(spec.input):
        problems.append("`input` must be a schema (a pydantic model, a TypeAdapter, or an object with parse())")
    if not is_schema(spec.output):
        problems.append("`output` must be a schema (a pydantic model, a TypeAdapter, or an object with parse())")

    # --- edges reference real nodes, and the graph is a DAG
    for edge in spec.edges:
        if not node(edge.from_):
            problems.append(f'edge {edge.from_} -> {edge.to}: unknown node "{edge.from_}"')
        if not node(edge.to):
            problems.append(f'edge {edge.from_} -> {edge.to}: unknown node "{edge.to}"')
    topo = topological_order(node_ids, spec.edges)
    if topo.cycle:
        problems.append(f"flow edges form a cycle through: {', '.join(topo.cycle)}")

    # --- clause 2: at least one gate, and only `code` branches
    gates = [nid for nid in node_ids if is_gate_node(node(nid))]
    if not gates:
        problems.append("no gate node: a DAG without a code gate is a chain, not a decision pipeline")

    for edge in spec.edges:
        if not edge.when:
            continue
        gate = node(edge.when.gate)
        if not gate:
            problems.append(f'edge {edge.from_} -> {edge.to}: guard names unknown node "{edge.when.gate}"')
            continue
        if not is_gate_node(gate):
            problems.append(
                f'edge {edge.from_} -> {edge.to}: guard names "{edge.when.gate}", which is a {gate.kind} node — only a code gate may branch'
            )
            continue
        if edge.when.branch not in gate.branches:
            problems.append(
                f'edge {edge.from_} -> {edge.to}: gate "{edge.when.gate}" has no branch "{edge.when.branch}" (has: {", ".join(gate.branches)})'
            )

    # --- clause 3: every branch has somewhere to go
    for gate_id in gates:
        gate = node(gate_id)
        assert gate is not None  # `gates` only holds ids of existing gate nodes
        for branch in gate.branches:
            taken = any(e.when and e.when.gate == gate_id and e.when.branch == branch for e in spec.edges)
            if not taken:
                problems.append(
                    f'gate "{gate_id}" declares branch "{branch}" with no outbound edge — every branch needs a deterministic destination'
                )

    # --- clause 4: at least one feedback edge, landing on a real node
    if len(spec.feedback) == 0:
        problems.append("no feedback edge: a DAG with no human signal connected to it is a chain, not a decision pipeline")
    feedback_ids: set[str] = set()
    for fe in spec.feedback:
        if fe.id in feedback_ids:
            problems.append(f'duplicate feedback edge id "{fe.id}"')
        feedback_ids.add(fe.id)

        if not node(fe.to):
            problems.append(f'feedback "{fe.id}": lands on unknown node "{fe.to}"')
        # The form x scope rule.
        if fe.form in ("override", "feature") and not scope_at_most(fe.scope, DIRECT_FORM_MAX_SCOPE):
            problems.append(
                f'feedback "{fe.id}": form "{fe.form}" is not allowed at scope "{fe.scope}" — at "{DIRECT_FORM_MAX_SCOPE}" and wider, only queue / derived / score'
            )
        # A structural change is always reviewable and revertable, never silent drift.
        if fe.form == "derived" and not fe.promote_via:
            problems.append(f'feedback "{fe.id}": form "derived" changes the pipeline\'s questions, so it needs a `promoteVia` review queue')
        if fe.form == "queue" and not fe.promote_via:
            problems.append(f'feedback "{fe.id}": form "queue" needs a `promoteVia` lane')
        # A `score` edge that never names its score has no landing spot.
        if fe.form == "score" and not fe.score:
            problems.append(
                f'feedback "{fe.id}": form "score" must declare a `score` config — otherwise the signal has no Langfuse score to land on'
            )
        if fe.score:
            cats = fe.score.categories or []
            if fe.score.data_type == "CATEGORICAL" and not cats:
                problems.append(f'feedback "{fe.id}": score "{fe.score.name}" is CATEGORICAL but declares no categories')
            if fe.score.data_type != "CATEGORICAL" and cats:
                problems.append(f'feedback "{fe.id}": score "{fe.score.name}" declares categories but is {fe.score.data_type}')
        # A lane, NOT a queue: annotation queues are capped per project.
        if fe.promote_via and not has_lane(fe.promote_via):
            problems.append(
                f'feedback "{fe.id}": lane "{fe.promote_via}" is not registered — call define_review_lane(lane=, queue=) (registered: {", ".join(lane_names()) or "none"}; queues are capped at {REVIEW_QUEUE_BUDGET} per Langfuse project, lanes are free)'
            )

    # --- clause 6: separability needs the pipeline's own answer persisted
    stores = [nid for nid in node_ids if getattr(node(nid), "kind", None) == "store"]
    if spec.overridable and not stores:
        problems.append(
            "`overridable` is declared but no store node persists the pipeline's own answer — an override that mutates is not separable"
        )

    # --- the fact has a producer
    for rid in _as_list(spec.result):
        if not node(rid):
            problems.append(f'result: unknown node "{rid}"')
        elif getattr(node(rid), "kind", None) == "store":
            problems.append(f'result: "{rid}" is a store node — result must name the node that PRODUCES the fact, not the one that writes it')

    # --- eval binding: a named extractor, not an ad-hoc query
    gm = getattr(spec.eval, "gate_metric", None)
    if gm and not has_gate_metric(gm):
        problems.append(
            f'eval.gateMetric "{gm}" is not a registered gate metric (known: {", ".join(gate_metric_names()) or "none"}) — define it with define_gate_metric() so the offline eval and an online split compute the same number'
        )

    # --- per-node checks
    for nid in node_ids:
        n0 = node(nid)
        if not dataclasses.is_dataclass(n0):
            continue
        n: Any = n0

        if n.inputs:
            direct_inbound = {e.from_ for e in spec.edges if e.to == nid}
            problems.extend(binding_problems(nid, n.inputs, direct_inbound))

        if n.on_failure == "fallback" and not callable(n.fallback):
            problems.append(f'node "{nid}": onFailure "fallback" needs a `fallback` body')
        if n.kind == "store" and n.cache and n.cache != "none":
            problems.append(f'node "{nid}": a store node may not be cached — a cached write is a missing write')
        if n.on_failure == "revert":
            if not any(e.to == nid for e in spec.edges):
                problems.append(f'node "{nid}": onFailure "revert" needs an inbound edge to revert to')

        if isinstance(n, DecideNode):
            if not n.questions and not n.probes:
                problems.append(f'node "{nid}": a decide node needs `questions`, `probes`, or both')
            if n.probes:
                writer = node(n.probes.from_)
                if not writer:
                    problems.append(f'node "{nid}": probes.from names unknown node "{n.probes.from_}"')
                elif writer.kind != "generate":
                    problems.append(
                        f'node "{nid}": probes.from must be a generate node — probes are written by a model — but "{n.probes.from_}" is {writer.kind}'
                    )
                elif not any(e.from_ == n.probes.from_ and e.to == nid for e in spec.edges):
                    problems.append(f'node "{nid}": probes.from "{n.probes.from_}" needs a direct edge {n.probes.from_} -> {nid}')
                if n.cache and n.cache != "none":
                    problems.append(f'node "{nid}": a decide node with probes may not be cached — its probe answers would not be restored')

        if isinstance(n, GenerateNode) and n.tools:
            tools = n.tools
            allow = set(tools.allow) if tools.allow is not None else None
            if allow is not None and isinstance(tools.static, list):
                for name in tools.static:
                    if name not in allow:
                        problems.append(f'node "{nid}": static tool "{name}" is outside `allow`')
            selectable = tools.selectable
            if selectable:
                if allow is not None:
                    for name in selectable.candidates:
                        if name not in allow:
                            problems.append(f'node "{nid}": selectable candidate "{name}" is outside `allow`')
                upstream = ancestors_of(spec.edges, nid)
                source = node(selectable.from_)
                if not source:
                    problems.append(f'node "{nid}": tools.selectable.from names unknown node "{selectable.from_}"')
                elif source.kind != "decide":
                    problems.append(f'node "{nid}": tools.selectable.from must be a decide node, but "{selectable.from_}" is {source.kind}')
                elif selectable.from_ not in upstream:
                    problems.append(
                        f'node "{nid}": tools.selectable.from "{selectable.from_}" is not upstream — selection has to be decided before the writer runs'
                    )
                gate = node(selectable.gate)
                if not gate:
                    problems.append(f'node "{nid}": tools.selectable.gate names unknown node "{selectable.gate}"')
                elif not is_gate_node(gate):
                    problems.append(f'node "{nid}": tools.selectable.gate must be a code gate, but "{selectable.gate}" is {gate.kind}')
                elif selectable.gate not in upstream:
                    problems.append(f'node "{nid}": tools.selectable.gate "{selectable.gate}" is not upstream')

    # --- experiments
    widest = widest_direct_feedback_scope(spec.feedback)
    experiment_ids: set[str] = set()
    for exp in spec.experiments or []:
        if not isinstance(exp, ExperimentSpec):
            continue
        if exp.id in experiment_ids:
            problems.append(f'duplicate experiment id "{exp.id}"')
        experiment_ids.add(exp.id)

        if exp.champion not in exp.arms:
            problems.append(f'experiment "{exp.id}": champion "{exp.champion}" is not one of its arms')
        if len(exp.arms) < 2:
            problems.append(f'experiment "{exp.id}": needs at least two arms')
        if widest is not None and scope_index(exp.unit) < scope_index(widest):
            problems.append(
                f'experiment "{exp.id}": assignment unit "{exp.unit}" is narrower than "{widest}", the widest scope a feedback edge writes to — arms would contaminate each other'
            )
        for arm_name, patch in exp.arms.items():
            problems.extend(f'experiment "{exp.id}" arm "{arm_name}": {p}' for p in validate_patch(spec, patch))

    return problems


def validate_patch(spec: Union[PipelineSpec, Mapping[str, Any]], patch: ManifestPatch) -> list[str]:
    """A patch may only touch the declarative half, and only fields the node kind has."""
    if isinstance(spec, Mapping) or not all(dataclasses.is_dataclass(n) for n in spec.nodes.values()):
        spec, _ = normalize_spec(spec)
    problems: list[str] = []

    def expect(node_id: str, field_name: str, predicate: Callable[[Any], bool], want: str) -> None:
        n = spec.nodes.get(node_id)
        if n is None:
            problems.append(f'{field_name} patches unknown node "{node_id}"')
        elif not predicate(n):
            problems.append(f'{field_name} patches "{node_id}", which is not {want}')

    for node_id, values in (patch.thresholds or {}).items():
        expect(node_id, "thresholds", has_thresholds, "a code node that declares thresholds")
        n = spec.nodes.get(node_id)
        if n is not None and has_thresholds(n):
            for name in values.keys():
                if name not in n.thresholds:
                    problems.append(f'thresholds patch "{node_id}.{name}" is not a threshold the node declares')
    for node_id in (patch.prompts or {}).keys():
        expect(node_id, "prompts", lambda n: n.kind == "generate", "a generate node")
    for node_id in (patch.routes or {}).keys():
        expect(node_id, "routes", lambda n: n.kind == "generate", "a generate node")
    for node_id in (patch.tools or {}).keys():
        expect(node_id, "tools", lambda n: n.kind == "generate", "a generate node")
    for node_id in (patch.questions or {}).keys():
        expect(node_id, "questions", lambda n: n.kind == "decide", "a decide node")
    for guard in patch.guards or []:
        exists = any(e.from_ == guard.from_ and e.to == guard.to for e in spec.edges)
        if not exists:
            problems.append(f"guards patch {guard.from_} -> {guard.to}: no such edge — a variant patches guards, it does not add edges")
    return problems


# ---------------------------------------------------------------------------
# define_pipeline
# ---------------------------------------------------------------------------


def _threshold_json(t: Any) -> Any:
    if isinstance(t, Mapping):
        d = dict(t)
        if "prompt_config" in d:
            d["promptConfig"] = d.pop("prompt_config")
        return d
    return t


def _tools_json(t: Any) -> Any:
    if t is None:
        return None
    out: dict[str, Any] = {}
    if t.static is not None:
        out["static"] = list(t.static) if isinstance(t.static, list) else t.static
    if t.selectable is not None:
        sel: dict[str, Any] = {"candidates": list(t.selectable.candidates), "from": t.selectable.from_, "gate": t.selectable.gate}
        if t.selectable.threshold is not None:
            sel["threshold"] = _threshold_json(t.selectable.threshold)
        out["selectable"] = sel
    if t.allow is not None:
        out["allow"] = list(t.allow)
    return out


def _node_to_json(nid: str, n: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": nid,
        "kind": n.kind,
        "version": n.version,
        "onFailure": n.on_failure or "fail",
        "cache": n.cache or "none",
    }
    if n.timeout_ms:
        base["timeoutMs"] = n.timeout_ms
    if n.describe:
        base["describe"] = n.describe
    if n.inputs:
        base["inputs"] = dict(n.inputs)
    if isinstance(n, ReadNode):
        if n.provides_features:
            base["providesFeatures"] = True
    elif isinstance(n, DecideNode):
        if n.questions:
            base["questions"] = n.questions
        if n.probes:
            probes: dict[str, Any] = {"from": n.probes.from_, "max": n.probes.max}
            if n.probes.paths is not None:
                probes["paths"] = list(n.probes.paths)
            if n.probes.max_chars is not None:
                probes["maxChars"] = n.probes.max_chars
            base["probes"] = probes
        if n.state:
            base["state"] = list(n.state)
        if n.accepts_features:
            base["acceptsFeatures"] = True
        if n.batched:
            base["batched"] = True
    elif isinstance(n, GenerateNode):
        base["prompt"] = n.prompt
        base["route"] = n.route
        if n.submit:
            base["submit"] = n.submit
        if n.tools:
            base["tools"] = _tools_json(n.tools)
    elif isinstance(n, (CodeNode, GateNode)):
        base["role"] = n.role
        if n.thresholds is not None:
            base["thresholds"] = {k: _threshold_json(v) for k, v in n.thresholds.items()}
        if isinstance(n, GateNode):
            base["branches"] = list(n.branches)
    elif isinstance(n, StoreNode):
        base["target"] = n.target
        base["scope"] = n.scope
        if n.ttl:
            base["ttl"] = n.ttl
    return base


def _edge_to_json(e: Edge) -> dict[str, Any]:
    out: dict[str, Any] = {"from": e.from_, "to": e.to}
    if e.when:
        out["when"] = {"gate": e.when.gate, "branch": e.when.branch}
    return out


def _score_to_json(s: Any) -> Any:
    if s is None:
        return None
    out: dict[str, Any] = {"name": s.name, "dataType": s.data_type}
    if s.categories is not None:
        out["categories"] = [{"label": c.label, "value": c.value} for c in s.categories]
    if s.description is not None:
        out["description"] = s.description
    return out


def _feedback_to_json(f: FeedbackEdge) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": f.id,
        "from": f.from_,
        "to": f.to,
        "source": f.source,
        "scope": f.scope,
        "form": f.form,
        "latency": f.latency,
    }
    if f.score is not None:
        out["score"] = _score_to_json(f.score)
    if f.promote_via is not None:
        out["promoteVia"] = f.promote_via
    out["key"] = callable(f.key)
    return out


def _experiment_to_json(e: ExperimentSpec) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": e.id,
        "unit": e.unit,
        "mode": e.mode,
        "champion": e.champion,
        "arms": {name: p.to_dict() for name, p in e.arms.items()},
    }
    if e.weights is not None:
        out["weights"] = dict(e.weights)
    if e.enabled is not None:
        out["enabled"] = e.enabled
    return out


def define_pipeline(spec: Union[PipelineSpec, Mapping[str, Any], None] = None, /, **kwargs: Any) -> DecisionPipeline:
    """Check the contract and return the manifest, or raise `PipelineContractError` listing EVERY violation."""
    if spec is None:
        spec = PipelineSpec(**{_CAMEL_SPEC.get(k, k): v for k, v in kwargs.items()})
    elif kwargs:
        raise TypeError("pass either a PipelineSpec or keyword arguments, not both")
    problems = validate_pipeline(spec)
    if problems:
        pid = spec.id if isinstance(spec, PipelineSpec) else str(spec.get("id"))
        raise PipelineContractError(pid, problems)

    normalized, _ = normalize_spec(spec)
    result = _as_list(normalized.result)
    decided_by = normalized.decided_by or "llm"
    order = topological_order(list(normalized.nodes.keys()), normalized.edges).order

    fields = {f.name: getattr(normalized, f.name) for f in dataclasses.fields(PipelineSpec)}
    fields.update(
        input=as_schema(normalized.input),
        output=as_schema(normalized.output),
        result=result,
        decided_by=decided_by,
        eval=normalized.eval,
        order=tuple(order),
    )
    return DecisionPipeline(**fields)
