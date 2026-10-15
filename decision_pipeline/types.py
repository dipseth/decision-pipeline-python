"""
The vocabulary of a decision pipeline.

Nothing in this file imports anything from the package: the type layer is the
part that has to survive being lifted out.

TS `from` is `from_` here and TS `in` is `in_`; everything else is the TS name
in snake_case. Nodes are dataclasses; `define_pipeline` also accepts plain
dicts for nodes, edges and feedback edges (see `node_from_dict`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional, Protocol, Sequence, TypedDict, Union

# ---------------------------------------------------------------------------
# Scope and rank
# ---------------------------------------------------------------------------

#: Who a signal binds to. NARROWEST FIRST — the order is load-bearing.
SCOPES: tuple[str, ...] = ("session", "device", "user", "associated", "tenant", "cohort", "global")

Scope = Literal["session", "device", "user", "associated", "tenant", "cohort", "global"]


def scope_index(scope: str) -> int:
    """Index of a scope in the narrowest-first chain; -1 for an unknown string."""
    return SCOPES.index(scope) if scope in SCOPES else -1


def scope_at_most(a: str, b: str) -> bool:
    """True when `a` is narrower than (or the same as) `b`."""
    return scope_index(a) <= scope_index(b)


#: The widest scope at which `override` and `feature` forms are allowed.
DIRECT_FORM_MAX_SCOPE: str = "associated"

#: A property of the WRITER, never of the feedback edge. Never downgraded. WEAKEST FIRST.
RANKS: tuple[str, ...] = ("heuristic", "llm", "human", "admin")
Rank = Literal["heuristic", "llm", "human", "admin"]


def rank_index(rank: str) -> int:
    return RANKS.index(rank) if rank in RANKS else -1


def max_rank(a: str, b: str) -> str:
    """The stronger of two ranks. A write never lowers the rank of a fact."""
    return a if rank_index(a) >= rank_index(b) else b


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

NodeKind = Literal["read", "decide", "generate", "code", "store"]
CodeRole = Literal["transform", "validate", "derive", "gate"]
Trigger = Literal["ingest", "cron", "on-demand"]
OnFailure = Literal["fallback", "skip", "revert", "fail"]
CacheMode = Literal["none", "per-run", "per-key-forever"]

#: A threshold is a literal, or a literal with a named runtime override:
#: `0.7`, `{"env": "NAME", "default": 0.7}`, `{"promptConfig": "key", "default": 0.7}`.
#: Snake-case keys (`prompt_config`) are accepted too.
Threshold = Union[float, int, Mapping[str, Any]]


@dataclass(kw_only=True)
class ToolSelectable:
    candidates: list[str]
    #: decide node id
    from_: str
    #: code gate node id
    gate: str
    #: Noul/Choice value a candidate must clear to be offered.
    threshold: Optional[Threshold] = None


@dataclass(kw_only=True)
class ToolSpec:
    """What a `generate` node may be offered."""

    #: Always offered. `"prompt-config"` means the prompt's config decides.
    static: Union[list[str], Literal["prompt-config"], None] = None
    selectable: Optional[ToolSelectable] = None
    #: Hard ceiling, validated when the manifest loads.
    allow: Optional[list[str]] = None


# --- probes: Jev questions an LLM writes for one run -------------------------

#: A question a `generate` node wrote for Jev: `{"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ..., "why"?: ...}`.
#: Kept as a plain dict so it is JSON as-is (it is what the decide port sends).
ProbeQuestion = dict


@dataclass(kw_only=True)
class ProbeSpec:
    """What a decide node declares to ask probes."""

    #: The `generate` node that writes the probes. Must be a DIRECT inbound edge.
    from_: str
    #: Probes asked per run, 1..MAX_PROBES. Extra ones are dropped, in order.
    max: int
    #: State paths a probe may address (backticked in its instructions).
    paths: Optional[list[str]] = None
    #: Per-probe text ceiling (instructions + criteria). Default 800.
    max_chars: Optional[int] = None


# --- box: where something is in an image (Jev 1.14; see ./box) ---------------


class Box(TypedDict):
    """Four edges as fractions of the image, origin top-left; each 0..1, `left < right`, `top < bottom`."""

    top: float
    bottom: float
    left: float
    right: float


class BoxAnswer(TypedDict):
    """One Box question's answer. There is no "not found": pair it with a Noul."""

    box: Box
    #: 0..1. Higher means the model is more sure. No recommended threshold yet: tune it.
    confidence: float


@dataclass(kw_only=True)
class ProbeAnswer:
    """A probe beside its answer — what downstream non-gate nodes see."""

    key: str
    question: ProbeQuestion
    distribution: list[float]
    options: list[str]


@dataclass(kw_only=True)
class BranchTaken:
    """A gate's verdict, as later nodes and the record see it."""

    branch: str
    reason: Optional[str] = None
    confidence: Optional[float] = None


class ScopeContext(Protocol):
    """
    Built by the runtime BEFORE node 0, from whatever ids the request carries.
    The pipeline's own code never sees a user id.
    """

    @property
    def ids(self) -> Mapping[str, str]: ...

    @property
    def grants(self) -> Sequence[str]: ...

    def features(self, node_id: str) -> list["FeatureRow"]: ...

    def override(self, key: str) -> Optional["OverrideRow"]: ...

    def consumed(self) -> list[str]: ...


@dataclass(kw_only=True)
class NodeRunArgs:
    """What every node body receives. Deliberately uniform — it is what makes a runtime possible."""

    #: The pipeline's validated input.
    input: Any
    #: Outputs of inbound nodes, by node id. A skipped inbound node is ABSENT.
    from_: dict[str, Any]
    #: The node's declared `inputs`, resolved (bindings). Empty when the node declares none.
    in_: Mapping[str, Any]
    #: The output of the node's primary (first declared) inbound edge, or the input at a root.
    primary: Any
    #: Scoped features and overrides. The pipeline never reads a user id.
    ctx: ScopeContext
    #: Resolved threshold VALUES for this node — gate bodies read these, never env.
    thresholds: dict[str, float]
    node: "NodeRef"
    #: Gate id -> the verdict it reached, for gates that have already run. A node never ROUTES on this.
    branches: Mapping[str, BranchTaken]
    #: Decide node id -> its answered probes. ALWAYS EMPTY for a gate.
    probes: Mapping[str, Sequence[ProbeAnswer]]
    #: Decide node id -> its checked Box answers (Jev 1.14), for decide nodes that already ran and
    #: asked one. Unlike probes, a gate sees these: a Box is a fixed question, so its confidence is tunable.
    boxes: Mapping[str, Mapping[str, BoxAnswer]] = field(default_factory=dict)
    #: Cancellation, when the host supplied one (an `asyncio.Event` or anything with `is_set()`).
    signal: Any = None


@dataclass(frozen=True)
class NodeRef:
    id: str
    version: str


NodeBody = Callable[[NodeRunArgs], Union[Any, Awaitable[Any]]]
#: A gate body returns a branch name, a `BranchTaken`, or `{"branch": ..., "reason"?: ..., "confidence"?: ...}`.
GateBody = Callable[[NodeRunArgs], Any]


@dataclass(kw_only=True)
class _NodeCommon:
    version: str
    #: Named inputs: binding name -> ref (`"evidence.row"`, `"$input.factor"`, `"maps | retry_maps"`).
    inputs: Optional[dict[str, str]] = None
    on_failure: Optional[OnFailure] = None
    #: The node's deadline in ms. Past it the node fails with `NodeTimeoutError` (so `on_failure` decides)
    #: and `args.signal` aborts. A body that ignores the signal keeps running unheard: its late result is discarded.
    timeout_ms: Optional[int] = None
    cache: Optional[CacheMode] = None
    #: Used only when `on_failure == "fallback"`.
    fallback: Optional[NodeBody] = None
    #: Free-form notes that ride onto the span and the JSON export.
    describe: Optional[str] = None


@dataclass(kw_only=True)
class ReadNode(_NodeCommon):
    kind: str = field(default="read", init=False)
    load: NodeBody
    #: Hand this node's scoped features to a downstream decide node.
    provides_features: Optional[bool] = None


@dataclass(kw_only=True)
class DecideNode(_NodeCommon):
    kind: str = field(default="decide", init=False)
    #: Jev question prompt name — wording lives in the host, not here. Optional only when `probes` is declared.
    questions: Optional[str] = None
    probes: Optional[ProbeSpec] = None
    #: Named state blocks the host resolves and hands to the decide port.
    state: Optional[list[str]] = None
    accepts_features: Optional[bool] = None
    #: Turns the port's raw distributions into the value downstream nodes consume. `(distributions, args) -> value`,
    #: or `(distributions, args, boxes) -> value` to receive the node's checked Box answers (Jev 1.14) as in TS;
    #: a two-argument `interpret` reads them as `args.boxes[args.node.id]`.
    interpret: Optional[Callable[..., Any]] = None
    batched: Optional[bool] = None


@dataclass(kw_only=True)
class GenerateNode(_NodeCommon):
    kind: str = field(default="generate", init=False)
    prompt: str
    route: str
    submit: Optional[str] = None
    tools: Optional[ToolSpec] = None


@dataclass(kw_only=True)
class CodeNode(_NodeCommon):
    kind: str = field(default="code", init=False)
    role: Literal["transform", "validate", "derive"]
    run: NodeBody
    #: A transform that APPLIES a decision without branching on it owns numbers exactly as a gate does.
    thresholds: Optional[dict[str, Threshold]] = None


@dataclass(kw_only=True)
class GateNode(_NodeCommon):
    kind: str = field(default="code", init=False)
    role: str = field(default="gate", init=False)
    run: GateBody
    thresholds: dict[str, Threshold]
    #: Every branch needs an outbound edge — that is the deterministic-fallback rule, checked.
    branches: list[str]


@dataclass(kw_only=True)
class StoreNode(_NodeCommon):
    kind: str = field(default="store", init=False)
    target: str
    scope: str
    ttl: Optional[str] = None


PipelineNode = Union[ReadNode, DecideNode, GenerateNode, CodeNode, GateNode, StoreNode]


def is_gate_node(node: Any) -> bool:
    return isinstance(node, GateNode)


def has_thresholds(node: Any) -> bool:
    """Any code node that declares thresholds — a gate always does, a transform may."""
    return isinstance(node, (GateNode, CodeNode)) and node.thresholds is not None


# ---------------------------------------------------------------------------
# Cancellation — the shape of `signal`
# ---------------------------------------------------------------------------


class AbortSignal:
    """
    What `args.signal` / `request.signal` is: `aborted` becomes true when the
    run is cancelled or the node's deadline passes. Mirrors the DOM shape the
    TS package uses. `AbortSignal.any(...)` combines signals; an `asyncio.Event`
    (or anything with `is_set()`) is accepted as a parent too.
    """

    def __init__(self, parents: Sequence[Any] = ()):
        self._aborted = False
        self._parents = tuple(p for p in parents if p is not None)

    @staticmethod
    def _is_aborted(s: Any) -> bool:
        if isinstance(s, AbortSignal):
            return s.aborted
        if hasattr(s, "aborted"):
            return bool(s.aborted)
        if hasattr(s, "is_set"):
            return bool(s.is_set())
        return False

    @property
    def aborted(self) -> bool:
        return self._aborted or any(self._is_aborted(p) for p in self._parents)

    def is_set(self) -> bool:
        return self.aborted

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise AbortedError()

    @staticmethod
    def any(*signals: Any) -> "AbortSignal":
        return AbortSignal(parents=signals)


class AbortedError(Exception):
    def __init__(self) -> None:
        super().__init__("aborted")


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self) -> None:
        self.signal._aborted = True


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeGuard:
    gate: str
    branch: str


@dataclass(kw_only=True)
class Edge:
    from_: str
    to: str
    #: A node whose EVERY inbound guard is false is skipped, with the reason recorded.
    when: Optional[EdgeGuard] = None


FeedbackSource = Literal["explicit", "implicit"]
FeedbackForm = Literal["override", "feature", "queue", "derived", "score"]
FeedbackLatency = Literal["immediate", "next-run", "deferred"]


@dataclass(kw_only=True)
class ScoreCategory:
    label: str
    value: float


@dataclass(kw_only=True)
class ScoreConfigSpec:
    """A Langfuse score config. Configs are NOT capped per project (queues are)."""

    name: str
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"]
    #: CATEGORICAL only.
    categories: Optional[list[ScoreCategory]] = None
    description: Optional[str] = None


@dataclass(kw_only=True)
class FeedbackEdge:
    id: str
    #: A user-facing surface id or a review queue — NOT a node.
    from_: str
    #: An EARLIER node id. Required even when the effect lands months later.
    to: str
    source: FeedbackSource
    scope: str
    form: FeedbackForm
    latency: FeedbackLatency
    #: REQUIRED for `form: "score"`; optional elsewhere.
    score: Optional[ScoreConfigSpec] = None
    #: Review **lane** (see review.py), not a queue name.
    promote_via: Optional[str] = None
    #: What an override/feature binds to — canonical, never a point id.
    key: Optional[Callable[[Any], str]] = None


# ---------------------------------------------------------------------------
# Scope context rows
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class FeatureRow:
    """A `feature`-form row addressed to a node, as the runtime hands it to a decide port."""

    id: str
    node_id: str
    scope: str
    rank: str
    value: Any
    arm: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(kw_only=True)
class OverrideRow:
    """An `override`-form row. NARROWEST WINS; applied at read, never inside the run."""

    id: str
    key: str
    scope: str
    rank: str
    value: Any
    arm: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(kw_only=True)
class ChangeRow:
    """The shared why-panel row. Why panels derive from a code-built change list, never from a model."""

    kind: str
    subject: str
    before: Any
    after: Any
    reason: str
    node_id: str


# ---------------------------------------------------------------------------
# Dicts → dataclasses, so a manifest can be written as literals too
# ---------------------------------------------------------------------------

_CAMEL = {
    "onFailure": "on_failure",
    "timeoutMs": "timeout_ms",
    "providesFeatures": "provides_features",
    "acceptsFeatures": "accepts_features",
    "maxChars": "max_chars",
    "promoteVia": "promote_via",
    "dataType": "data_type",
    "promptConfig": "prompt_config",
    "from": "from_",
    "in": "in_",
}


def _snake(d: Mapping[str, Any]) -> dict[str, Any]:
    return {_CAMEL.get(k, k): v for k, v in d.items()}


def tool_spec_from_dict(d: Any) -> Optional[ToolSpec]:
    if d is None or isinstance(d, ToolSpec):
        return d
    d = _snake(d)
    sel = d.get("selectable")
    if sel is not None and not isinstance(sel, ToolSelectable):
        sel = ToolSelectable(**_snake(sel))
    return ToolSpec(static=d.get("static"), selectable=sel, allow=d.get("allow"))


def probe_spec_from_dict(d: Any) -> Optional[ProbeSpec]:
    if d is None or isinstance(d, ProbeSpec):
        return d
    return ProbeSpec(**_snake(d))


def score_config_from_dict(d: Any) -> Optional[ScoreConfigSpec]:
    if d is None or isinstance(d, ScoreConfigSpec):
        return d
    d = _snake(d)
    cats = d.get("categories")
    if cats is not None:
        cats = [c if isinstance(c, ScoreCategory) else ScoreCategory(**c) for c in cats]
    return ScoreConfigSpec(name=d["name"], data_type=d["data_type"], categories=cats, description=d.get("description"))


def edge_from_dict(d: Any) -> Edge:
    if isinstance(d, Edge):
        return d
    d = _snake(d)
    when = d.get("when")
    if when is not None and not isinstance(when, EdgeGuard):
        when = EdgeGuard(gate=when["gate"], branch=when["branch"])
    return Edge(from_=d["from_"], to=d["to"], when=when)


def feedback_edge_from_dict(d: Any) -> FeedbackEdge:
    if isinstance(d, FeedbackEdge):
        return d
    d = _snake(d)
    return FeedbackEdge(
        id=d["id"],
        from_=d["from_"],
        to=d["to"],
        source=d["source"],
        scope=d["scope"],
        form=d["form"],
        latency=d["latency"],
        score=score_config_from_dict(d.get("score")),
        promote_via=d.get("promote_via"),
        key=d.get("key"),
    )


def node_from_dict(d: Any) -> Any:
    """
    A node literal (`{"kind": "decide", "questions": "...", "version": "1"}`,
    camelCase or snake_case keys) → the dataclass. A dataclass passes through.
    Unknown or malformed shapes raise `TypeError`; `validate_pipeline` reports
    them as problems instead.
    """
    if not isinstance(d, Mapping):
        return d
    d = _snake(d)
    kind = d.get("kind")
    common = {k: d.get(k) for k in ("version", "inputs", "on_failure", "timeout_ms", "cache", "fallback", "describe")}
    if kind == "read":
        return ReadNode(load=d["load"], provides_features=d.get("provides_features"), **common)
    if kind == "decide":
        return DecideNode(
            questions=d.get("questions"),
            probes=probe_spec_from_dict(d.get("probes")),
            state=d.get("state"),
            accepts_features=d.get("accepts_features"),
            interpret=d.get("interpret"),
            batched=d.get("batched"),
            **common,
        )
    if kind == "generate":
        return GenerateNode(prompt=d["prompt"], route=d["route"], submit=d.get("submit"), tools=tool_spec_from_dict(d.get("tools")), **common)
    if kind == "code":
        if d.get("role") == "gate":
            return GateNode(run=d["run"], thresholds=d.get("thresholds") or {}, branches=d.get("branches") or [], **common)
        return CodeNode(role=d["role"], run=d["run"], thresholds=d.get("thresholds"), **common)
    if kind == "store":
        return StoreNode(target=d["target"], scope=d["scope"], ttl=d.get("ttl"), **common)
    raise TypeError(f"node kind {kind!r} is not one of read, decide, generate, code, store")
