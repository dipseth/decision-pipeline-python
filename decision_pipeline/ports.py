"""
Ports — every side effect the runtime needs, as callables and small protocols
the HOST implements. Nothing in this package imports Langfuse, a database, a
model SDK or a tool registry; that is what keeps the core liftable and what
makes the whole thing testable with fakes (see testing.py).

Every port may be sync or async: the runtime awaits what it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Optional, Protocol, Sequence, Union

from .types import BranchTaken, FeatureRow, OverrideRow, ProbeAnswer, ProbeQuestion, ScopeContext, ScoreConfigSpec

if TYPE_CHECKING:  # pragma: no cover
    from .experiments import ManifestPatch
    from .observers import RunObserver
    from .record import RunRecord
    from .thresholds import ThresholdSource
    from .tools import ResolvedTools, ToolRegistry

MaybeAwaitable = Union[Any, Awaitable[Any]]

# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class SpanHandle(Protocol):
    @property
    def trace_id(self) -> Optional[str]:
        """`trace_id` of the root span is the RUN ID. None when the host is running untraced."""
        ...

    def set_metadata(self, metadata: Mapping[str, Any]) -> None: ...

    def set_output(self, output: Any) -> None: ...

    def set_tags(self, tags: Sequence[str]) -> None:
        """Optional: add trace tags AFTER the root opened. A tracer that cannot amend tags may omit it."""
        ...


@dataclass(kw_only=True)
class RootSpanOpts:
    #: Always `pipeline:<id>`.
    name: str
    input: Any
    #: The manifest's `group(input)` — the tracer's session id.
    group: str
    tags: list[str]
    metadata: dict[str, Any]
    #: Who the run is attributed to, from the scope context's `user` id.
    user_id: Optional[str] = None


@dataclass(kw_only=True)
class NodeSpanOpts:
    name: str
    input: Any
    metadata: dict[str, Any]


class TracePort(Protocol):
    """
    Callback-shaped on purpose: OTel needs an active-span callback for children
    to nest. `fn` is an async callable; `root`/`span` must await it and return its value.
    """

    async def root(self, opts: RootSpanOpts, fn: Callable[[SpanHandle], Awaitable[Any]]) -> Any: ...

    async def span(self, opts: NodeSpanOpts, fn: Callable[[SpanHandle], Awaitable[Any]]) -> Any: ...


# ---------------------------------------------------------------------------
# Decide (Jev)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DecideRequest:
    node_id: str
    node_version: str
    #: Jev question prompt name. None on a probe-only node.
    questions: Optional[str]
    #: Probes to ask in the SAME request, keyed by slot (`probe_0`, …) — already validated, capped and de-duplicated.
    probes: dict[str, ProbeQuestion]
    #: Named state blocks the host resolves.
    state: list[str]
    #: Scoped `feature` rows addressed to this node, when `accepts_features`.
    features: list[FeatureRow]
    input: Any
    from_: dict[str, Any]
    #: The node's resolved `inputs`; empty when it declares none.
    in_: Mapping[str, Any]
    batched: bool
    signal: Any = None


@dataclass(kw_only=True)
class DecideResult:
    #: MANDATORY. Offline replay is impossible without it.
    distributions: dict[str, list[float]]
    #: Option keys per distribution, in the same order as its numbers.
    distribution_options: Optional[dict[str, list[str]]] = None
    #: What downstream nodes consume, when the node has no `interpret`.
    value: Any = None
    cost_usd: Optional[float] = None
    #: A batched call gets its own trace; each run records the id and a pro-rata cost.
    batch_trace_id: Optional[str] = None


DecidePort = Callable[[DecideRequest], MaybeAwaitable]

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GenerateRequest:
    node_id: str
    node_version: str
    prompt: str
    route: str
    submit: Optional[str]
    tools: "ResolvedTools"
    input: Any
    from_: dict[str, Any]
    in_: Mapping[str, Any]
    primary: Any
    ctx: ScopeContext
    #: Gate verdicts already reached — for metadata, never for routing.
    branches: Mapping[str, BranchTaken]
    #: Answered probes from decide nodes that already ran — evidence for the writer.
    probes: Mapping[str, Sequence[ProbeAnswer]]
    signal: Any = None


@dataclass(kw_only=True)
class GenerateResult:
    value: Any
    cost_usd: Optional[float] = None
    tools_called: Optional[list[str]] = None
    #: Feeds the composite version hash.
    prompt_version: Optional[str] = None
    #: Root-trace tags this node earned, e.g. `prompt-override`.
    tags: Optional[Sequence[str]] = None


GeneratePort = Callable[[GenerateRequest], MaybeAwaitable]

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class StoreRequest:
    node_id: str
    target: str
    scope: str
    ttl: Optional[str]
    #: The persisted field name. Metadata is `<fact>_*`.
    fact: str
    value: Any
    route: str
    confidence: Optional[float]
    version: str
    run_id: Optional[str]
    decided_by: str
    #: FALSE for a shadow challenger.
    writes_fact: bool
    ctx: ScopeContext


StorePort = Callable[[StoreRequest], MaybeAwaitable]

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ScopeResolveRequest:
    pipeline: str
    #: Node ids that accept features, so a resolver can query narrowly.
    feature_nodes: list[str]
    #: The manifest's override key for this run's output, when overridable.
    override_key: Optional[str]
    input: Any


#: What a scope resolver returns: a `ScopeContextInput` (scope.py) or a dict with the same keys
#: (`ids`, `grants`, `features`, `overrides`).
ScopeResolver = Callable[[ScopeResolveRequest], MaybeAwaitable]

# ---------------------------------------------------------------------------
# Prompts, cache, records
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PromptResolution:
    version: Optional[str] = None
    #: What `tools.static: "prompt-config"` resolves to.
    tools: Optional[Sequence[str]] = None


#: Resolved once per run, before node 0, so the version hash can include it. `(ref) -> PromptResolution`.
PromptResolver = Callable[[str], MaybeAwaitable]


class CachePort(Protocol):
    def get(self, key: str) -> MaybeAwaitable:
        """Key is `<pipeline>:<node_id>:<node_version>:<input_hash>`. Return None on a miss."""
        ...

    def set(self, key: str, value: Any) -> MaybeAwaitable: ...


#: Sugar for the one observer every host wants (`record_sink_observer`). Best-effort.
RunRecordSink = Callable[["RunRecord"], MaybeAwaitable]

# ---------------------------------------------------------------------------
# Human signal — scores, review queues, datasets
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ScoreRequest:
    #: The run's trace id. Scores attach HERE, which is why the run id is the trace id.
    run_id: str
    name: str
    value: float
    #: CATEGORICAL only.
    label: Optional[str] = None
    data_type: str
    comment: Optional[str] = None
    #: The config to create-if-missing, so a name never lands without its categories.
    config: Optional[ScoreConfigSpec] = None
    user_id: Optional[str] = None


#: Best-effort by contract: the tracer being down must never block a feedback row.
ScorePort = Callable[[ScoreRequest], MaybeAwaitable]


@dataclass(kw_only=True)
class QueueRequest:
    run_id: str
    #: The PHYSICAL queue, already resolved from the lane.
    queue: str
    #: The lane the manifest named.
    lane: str
    pipeline: str
    #: The feedback edge that sent it.
    edge_id: str
    form: str
    score_config: Optional[ScoreConfigSpec] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class QueueResult:
    queue_id: str
    item_id: str
    url: Optional[str] = None


QueuePort = Callable[[QueueRequest], MaybeAwaitable]


@dataclass(kw_only=True)
class DatasetItem:
    id: str
    input: Any
    expected_output: Any = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class DatasetRunRequest:
    dataset: str
    #: One run per evaluation. For an experiment this is the ARM.
    run: str
    run_metadata: Optional[dict[str, Any]] = None
    items: list[DatasetItem] = field(default_factory=list)
    #: The eval's root trace; run items and the gate metric attach here.
    trace_id: Optional[str] = None
    #: Aggregate metrics: `(name, value)` pairs.
    scores: Optional[Sequence[tuple[str, float]]] = None


DatasetPort = Callable[[DatasetRunRequest], MaybeAwaitable]

# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PipelinePorts:
    tracer: Optional[TracePort] = None
    decide: Optional[DecidePort] = None
    generate: Optional[GeneratePort] = None
    store: Optional[StorePort] = None
    scope: Optional[ScopeResolver] = None
    scores: Optional[ScorePort] = None
    queue: Optional[QueuePort] = None
    dataset: Optional[DatasetPort] = None
    prompts: Optional[PromptResolver] = None
    thresholds: Optional["ThresholdSource"] = None
    tool_registry: Optional["ToolRegistry"] = None
    cache: Optional[CachePort] = None
    records: Optional[RunRecordSink] = None
    #: Watchers. Notified in order, after `records`, and BEST-EFFORT.
    observers: Optional[Sequence["RunObserver"]] = None
    #: Returns milliseconds, like `Date.now()`.
    clock: Optional[Callable[[], float]] = None
    #: Bumped into the composite version hash.
    knowledge_version: Optional[str] = None


@dataclass(kw_only=True)
class RunOptions:
    trigger: Optional[str] = None
    #: A stable identity for the WORK, surviving retries. Omitted, the run's own trace id stands in.
    run_key: Optional[str] = None
    parent_run_id: Optional[str] = None
    #: Force an arm — how a shadow challenger and an offline replay are run.
    force_arm: Optional[dict[str, str]] = None
    #: Run as a shadow challenger: recorded and tagged `shadow`, but `writes_fact` is false whatever the arms say.
    shadow: bool = False
    #: Extra patches applied after the arm patch. Replay uses this.
    patch: Optional["ManifestPatch"] = None
    signal: Any = None
    #: Merged into the root span metadata. Never into the version hash.
    metadata: Optional[dict[str, Any]] = None
    #: Extra root-trace TAGS, appended to the ones the runtime derives.
    tags: Optional[Sequence[str]] = None
