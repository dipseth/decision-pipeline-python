"""
In-memory ports.

These are the point of the port layer: a pipeline's whole contract —
routing, skipping, tool selection, persistence metadata, the run record —
is testable with no tracer, no database, no model call and no network. They
ship from the package (not from a test file) so the APP's tests can use the
same fakes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Union

from ._compat import maybe_await
from .observers import NodeEndEvent, NodeStartEvent, RunEndEvent, RunFailEvent, RunObserver, RunStartEvent
from .ports import (
    DatasetRunRequest,
    DecideRequest,
    DecideResult,
    GenerateRequest,
    GenerateResult,
    NodeSpanOpts,
    PipelinePorts,
    PromptResolution,
    QueueRequest,
    QueueResult,
    RootSpanOpts,
    ScoreRequest,
    StoreRequest,
)
from .record import RunRecord
from .scope import ScopeContextInput

# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RecordedSpan:
    name: str
    #: 0 for the root.
    depth: int
    input: Any
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Root spans only.
    tags: list[str] = field(default_factory=list)
    #: Root spans only. Who the run is attributed to.
    user_id: Optional[str] = None


class _SpanHandle:
    def __init__(self, trace_id: str, recorded: RecordedSpan):
        self.trace_id = trace_id
        self._recorded = recorded

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._recorded.metadata.update(metadata)

    def set_tags(self, tags: Sequence[str]) -> None:
        for t in tags:
            if t not in self._recorded.tags:
                self._recorded.tags.append(t)

    def set_output(self, output: Any) -> None:
        self._recorded.output = output


class RecordingTracer:
    def __init__(self, trace_id: str = "trace-0001"):
        self.trace_id = trace_id
        self.spans: list[RecordedSpan] = []
        #: Roots opened. More than one means the "one run = one trace" rule is broken.
        self.roots: list[RecordedSpan] = []
        self._depth = -1

    def span_(self, name: str) -> Optional[RecordedSpan]:
        return next((s for s in self.spans if s.name == name), None)

    async def _open(self, name: str, input: Any, metadata: Mapping[str, Any], tags: Sequence[str], user_id: Optional[str], is_root: bool, fn: Any) -> Any:
        self._depth += 1
        recorded = RecordedSpan(name=name, depth=self._depth, input=input, metadata=dict(metadata), tags=list(tags), user_id=user_id)
        self.spans.append(recorded)
        if is_root:
            self.roots.append(recorded)
        try:
            return await fn(_SpanHandle(self.trace_id, recorded))
        finally:
            self._depth -= 1

    async def root(self, opts: RootSpanOpts, fn: Any) -> Any:
        return await self._open(opts.name, opts.input, opts.metadata, opts.tags, opts.user_id, True, fn)

    async def span(self, opts: NodeSpanOpts, fn: Any) -> Any:
        return await self._open(opts.name, opts.input, opts.metadata, [], None, False, fn)


def recording_tracer(trace_id: str = "trace-0001") -> RecordingTracer:
    return RecordingTracer(trace_id)


# ---------------------------------------------------------------------------
# Decide / generate
# ---------------------------------------------------------------------------

#: node id -> a result (`DecideResult` or a dict with `distributions`), or a callable of the request returning one.
DecideScript = dict[str, Any]
GenerateScript = dict[str, Any]


@dataclass
class ScriptedDecide:
    port: Callable[[DecideRequest], Awaitable[Any]]
    calls: list[DecideRequest]


def scripted_decide(script: Optional[DecideScript] = None) -> ScriptedDecide:
    script = script or {}
    calls: list[DecideRequest] = []

    async def port(request: DecideRequest) -> Any:
        calls.append(request)
        if request.node_id not in script:
            raise Exception(f'scripted_decide: no script for node "{request.node_id}" (have: {", ".join(script.keys()) or "none"})')
        entry = script[request.node_id]
        return await maybe_await(entry(request)) if callable(entry) else entry

    return ScriptedDecide(port=port, calls=calls)


@dataclass
class ScriptedGenerate:
    port: Callable[[GenerateRequest], Awaitable[Any]]
    calls: list[GenerateRequest]


def scripted_generate(script: Optional[GenerateScript] = None) -> ScriptedGenerate:
    script = script or {}
    calls: list[GenerateRequest] = []

    async def port(request: GenerateRequest) -> Any:
        calls.append(request)
        if request.node_id not in script:
            raise Exception(f'scripted_generate: no script for node "{request.node_id}" (have: {", ".join(script.keys()) or "none"})')
        entry = script[request.node_id]
        return await maybe_await(entry(request)) if callable(entry) else entry

    return ScriptedGenerate(port=port, calls=calls)


# ---------------------------------------------------------------------------
# Store / cache / records / scope / prompts / registry
# ---------------------------------------------------------------------------


@dataclass
class CollectingStore:
    port: Callable[[StoreRequest], Any]
    writes: list[StoreRequest]


def collecting_store() -> CollectingStore:
    writes: list[StoreRequest] = []

    def port(request: StoreRequest) -> None:
        writes.append(request)

    return CollectingStore(port=port, writes=writes)


class MemoryCache:
    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.entries.get(key)

    def set(self, key: str, value: Any) -> None:
        self.entries[key] = value


def memory_cache() -> MemoryCache:
    return MemoryCache()


@dataclass
class CollectedRecords:
    sink: Callable[[RunRecord], Any]
    records: list[RunRecord]

    def last(self) -> Optional[RunRecord]:
        return self.records[-1] if self.records else None


def collecting_records() -> CollectedRecords:
    records: list[RunRecord] = []

    def sink(record: RunRecord) -> None:
        records.append(record)

    return CollectedRecords(sink=sink, records=records)


@dataclass
class ObservedEvent:
    """One entry per hook call, in the order the runtime made them."""

    hook: str
    event: Union[RunStartEvent, NodeStartEvent, NodeEndEvent, RunEndEvent, RunFailEvent]


class _CollectingObserver:
    def __init__(self, name: str, events: list[ObservedEvent]):
        self.name = name
        self._events = events

    def on_run_start(self, event: RunStartEvent) -> None:
        self._events.append(ObservedEvent("on_run_start", event))

    def on_node_start(self, event: NodeStartEvent) -> None:
        self._events.append(ObservedEvent("on_node_start", event))

    def on_node_end(self, event: NodeEndEvent) -> None:
        self._events.append(ObservedEvent("on_node_end", event))

    def on_run_end(self, event: RunEndEvent) -> None:
        self._events.append(ObservedEvent("on_run_end", event))

    def on_run_fail(self, event: RunFailEvent) -> None:
        self._events.append(ObservedEvent("on_run_fail", event))


@dataclass
class CollectingObserver:
    observer: Any
    events: list[ObservedEvent]

    def stream(self) -> list[str]:
        """The hook names in order — what an "it fired in this sequence" assertion reads."""
        return [e.hook for e in self.events]

    def node(self, node_id: str) -> list[ObservedEvent]:
        """`on_node_start`/`on_node_end` narrowed to one node."""
        return [e for e in self.events if e.hook in ("on_node_start", "on_node_end") and getattr(e.event, "node_id", None) == node_id]


def collecting_observer(name: str = "collecting") -> CollectingObserver:
    """A watcher that records what it was told. Asserting on THIS is the point of the observer seam."""
    events: list[ObservedEvent] = []
    return CollectingObserver(observer=_CollectingObserver(name, events), events=events)


class _ThrowingObserver:
    def __init__(self, name: str):
        self.name = name

    def _boom(self, event: Any) -> None:
        raise Exception(f"{self.name} exploded")

    on_run_start = _boom
    on_node_start = _boom
    on_node_end = _boom
    on_run_end = _boom
    on_run_fail = _boom


def throwing_observer(name: str = "throwing") -> Any:
    """A watcher that fails on every hook — the "best-effort" contract's test double."""
    return _ThrowingObserver(name)


@dataclass
class CollectingScores:
    port: Callable[[ScoreRequest], Any]
    scores: list[ScoreRequest]


def collecting_scores() -> CollectingScores:
    scores: list[ScoreRequest] = []

    def port(request: ScoreRequest) -> None:
        scores.append(request)

    return CollectingScores(port=port, scores=scores)


@dataclass
class CollectingQueue:
    port: Callable[[QueueRequest], Any]
    items: list[QueueRequest]


def collecting_queue() -> CollectingQueue:
    items: list[QueueRequest] = []

    def port(request: QueueRequest) -> QueueResult:
        items.append(request)
        return QueueResult(queue_id=f"q-{request.queue}", item_id=f"item-{len(items)}")

    return CollectingQueue(port=port, items=items)


@dataclass
class CollectingDataset:
    port: Callable[[DatasetRunRequest], Any]
    runs: list[DatasetRunRequest]


def collecting_dataset() -> CollectingDataset:
    runs: list[DatasetRunRequest] = []

    def port(request: DatasetRunRequest) -> None:
        runs.append(request)

    return CollectingDataset(port=port, runs=runs)


class MemoryTrialStore:
    """The reference `TrialStore`: everything in dicts, copied in and out so a caller cannot edit the record."""

    def __init__(self) -> None:
        self.locks: dict[str, Any] = {}
        self.rows_by_trial: dict[str, list[Any]] = {}
        self.judgements_by_trial: dict[str, list[Any]] = {}

    async def get_lock(self, trial: str) -> Any:
        return copy.deepcopy(self.locks[trial]) if trial in self.locks else None

    async def put_lock(self, lock: Any) -> None:
        self.locks[lock.trial] = copy.deepcopy(lock)

    async def append_rows(self, trial: str, rows: Sequence[Any]) -> None:
        self.rows_by_trial[trial] = [*self.rows_by_trial.get(trial, []), *copy.deepcopy(list(rows))]

    async def rows(self, trial: str) -> list[Any]:
        return copy.deepcopy(self.rows_by_trial.get(trial, []))

    async def append_judgement(self, trial: str, judgement: Any) -> None:
        self.judgements_by_trial[trial] = [*self.judgements_by_trial.get(trial, []), copy.deepcopy(judgement)]

    async def judgements(self, trial: str) -> list[Any]:
        return copy.deepcopy(self.judgements_by_trial.get(trial, []))


def memory_trial_store() -> MemoryTrialStore:
    return MemoryTrialStore()


def static_scope(resolved: Optional[Union[ScopeContextInput, Mapping[str, Any]]] = None) -> Callable[[Any], Any]:
    """A scope resolver that always returns the same rows."""
    return lambda _request: resolved


def static_prompts(prompts: Optional[Mapping[str, Any]] = None) -> Callable[[str], PromptResolution]:
    """Prompt versions keyed by ref; unknown refs resolve to None, as an unconfigured host would."""
    prompts = prompts or {}

    def resolve(ref: str) -> PromptResolution:
        hit = prompts.get(ref)
        if hit is None:
            return PromptResolution(version=None)
        if isinstance(hit, PromptResolution):
            return hit
        return PromptResolution(version=hit.get("version"), tools=hit.get("tools"))

    return resolve


class StaticRegistry:
    """A registry where every listed tool exists and needs the MCP scope given."""

    def __init__(self, tools: Mapping[str, Optional[str]]):
        self.tools = dict(tools)

    def has(self, name: str) -> bool:
        return name in self.tools

    def scope_of(self, name: str) -> Optional[str]:
        return self.tools.get(name)


def static_registry(tools: Mapping[str, Optional[str]]) -> StaticRegistry:
    return StaticRegistry(tools)


def stepping_clock(start_ms: float = 1_700_000_000_000, step_ms: float = 1) -> Callable[[], float]:
    """A clock that advances a fixed number of ms per read, so `ms` fields are deterministic."""
    now = start_ms

    def clock() -> float:
        nonlocal now
        value = now
        now += step_ms
        return value

    return clock


class _StaticThresholds:
    def __init__(self, env: Mapping[str, str], prompt_config: Mapping[str, float]):
        self._env = dict(env)
        self._prompt_config = dict(prompt_config)

    def env(self, name: str) -> Optional[str]:
        return self._env.get(name)

    def prompt_config(self, key: str) -> Optional[float]:
        return self._prompt_config.get(key)


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TestPorts(PipelinePorts):
    tracer: RecordingTracer
    #: The event stream `ports.observers[0]` saw.
    observed: list[ObservedEvent]
    store_writes: list[StoreRequest]
    recorded: list[RunRecord]
    decide_calls: list[DecideRequest]
    generate_calls: list[GenerateRequest]
    scores_written: list[ScoreRequest]
    queue_items: list[QueueRequest]
    dataset_runs: list[DatasetRunRequest]


def test_ports(
    *,
    decide: Optional[DecideScript] = None,
    generate: Optional[GenerateScript] = None,
    scope: Optional[Union[ScopeContextInput, Mapping[str, Any]]] = None,
    prompts: Optional[Mapping[str, Any]] = None,
    registry: Optional[Mapping[str, Optional[str]]] = None,
    env: Optional[Mapping[str, str]] = None,
    prompt_config: Optional[Mapping[str, float]] = None,
    knowledge_version: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> TestPorts:
    """Everything a pipeline needs to run, with nothing real behind it."""
    tracer = recording_tracer(trace_id or "trace-0001")
    d = scripted_decide(decide)
    g = scripted_generate(generate)
    store = collecting_store()
    records = collecting_records()
    scores = collecting_scores()
    queue = collecting_queue()
    dataset = collecting_dataset()
    watcher = collecting_observer()

    return TestPorts(
        tracer=tracer,
        observers=[watcher.observer],
        observed=watcher.events,
        decide=d.port,
        generate=g.port,
        store=store.port,
        scores=scores.port,
        queue=queue.port,
        dataset=dataset.port,
        scope=static_scope(scope),
        prompts=static_prompts(prompts),
        tool_registry=static_registry(registry) if registry is not None else None,
        thresholds=_StaticThresholds(env or {}, prompt_config or {}),  # type: ignore[arg-type]
        cache=memory_cache(),
        records=records.sink,
        clock=stepping_clock(),
        knowledge_version=knowledge_version,
        store_writes=store.writes,
        recorded=records.records,
        decide_calls=d.calls,
        generate_calls=g.calls,
        scores_written=scores.scores,
        queue_items=queue.items,
        dataset_runs=dataset.runs,
    )


# pytest would otherwise try to collect `test_ports` as a test when this module is imported into a test file.
test_ports.__test__ = False  # type: ignore[attr-defined]
TestPorts.__test__ = False  # type: ignore[attr-defined]
