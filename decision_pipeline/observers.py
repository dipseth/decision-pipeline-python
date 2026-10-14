"""
Run observers — the watchers (#325).

The line this draws, and the whole reason this file is not `ports.py`:

  A **port** returns a value the run consumes.
  An **observer** consumes events and returns nothing the run reads.

By that rule `tracer` is NOT an observer, however much it looks like one: it
produces the run's identity (the root span's trace id IS the run id, which every
feedback row and every training join keys on), so removing it would change
what the run persists. `cache` is not one either — it returns a resumed value
that skips a node. Both stay ports. What IS an observer is anything that only
watches: the run record sink, a cost meter, a debug recorder, a gate-student
sampler.

Two consequences follow from "returns nothing the run reads", and both are
contract, not implementation detail:

  1. An observer that throws MUST NOT fail the run. The run already happened;
     a watcher's failure is a lost measurement, not a lost answer.
  2. An observer must not be load-bearing. If removing one changes an output,
     a persisted fact or an id, it was a port wearing an observer's clothes.

Shape borrowed from Burr's lifecycle hooks (Apache-2.0) — the idea that a
small fixed set of hook points lets anything watch a run without the core
knowing it exists. The shape only; no code was copied, so no attribution is
owed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional, Protocol, Sequence, Union

from ._compat import maybe_await
from .record import RunRecord

_log = logging.getLogger("decision_pipeline")

HookName = Literal["on_run_start", "on_node_start", "on_node_end", "on_run_end", "on_run_fail"]


@dataclass(kw_only=True)
class RunStartEvent:
    pipeline: str
    #: The root trace id. None when the host runs untraced.
    run_id: Optional[str]
    #: The durable identity — `options.run_key`, or this attempt's id. Never None.
    run_key: str
    #: The composite version hash: nodes + prompts + thresholds + knowledge + arms.
    version: str
    input: Any
    group: str
    arms: dict[str, str]
    trigger: str
    #: FALSE for a shadow challenger — a watcher must not bill a run that ships nothing.
    writes_fact: bool


@dataclass(kw_only=True)
class NodeStartEvent:
    pipeline: str
    run_id: Optional[str]
    node_id: str
    kind: str
    node_version: str
    #: `short_hash({primary, from})`. Empty for a node skipped before it ran.
    input_hash: str
    #: True when the node's output came from `cache`, so nothing was spent.
    cache_hit: bool


@dataclass(kw_only=True)
class NodeEndEvent(NodeStartEvent):
    ms: float
    cost_usd: float
    #: A gate's verdict.
    branch: Optional[str] = None
    #: A generate node's route.
    route: Optional[str] = None
    #: The node's output, as the runtime recorded it. None when it did not run
    #: (TS `undefined`; a node whose output is `None` reports the same).
    output: Any = None
    #: The failure message, whatever `on_failure` then did about it.
    error: Optional[str] = None
    #: Why it did not run — an unopened edge, or `error: <message>` under `on_failure: "skip"`.
    skipped: Optional[str] = None


@dataclass(kw_only=True)
class RunEndEvent:
    pipeline: str
    run_id: Optional[str]
    run_key: str
    version: str
    record: RunRecord
    output: Any
    #: The branches taken, joined — what lands as `<fact>_route`.
    route: str
    confidence: Optional[float]
    decided_by: str
    writes_fact: bool


@dataclass(kw_only=True)
class RunFailEvent:
    """
    A run that threw. There is no `RunRecord` — the fact was never produced —
    so this carries what the run had accumulated when it died: the nodes it
    considered (the one that killed it last, with its `error`) and what they cost.
    """

    pipeline: str
    run_id: Optional[str]
    run_key: str
    version: str
    group: str
    trigger: str
    arms: dict[str, str]
    writes_fact: bool
    error: str
    #: The node whose failure ended the run; None when it died outside a node.
    failed_node: Optional[str]
    nodes: list[Any]
    cost_usd: float
    ms: float


class RunObserver(Protocol):
    """
    Every node the runtime CONSIDERED gets exactly one `on_node_start` and exactly
    one `on_node_end`, skips included — a watcher that pairs them can rely on that
    rather than reconstructing which halves are missing.

    A run that got as far as `on_run_start` ends in exactly one of `on_run_end`
    (it produced its fact; here is the record) or `on_run_fail` (it threw; here
    is what it had spent). `on_run_fail` fires after the root span has closed and
    before the caller sees the exception. A failed run never reaches the record
    sink: a training table of runs that answered must not grow rows that didn't.

    Any object works: every attribute below is optional, and each hook may be
    sync or async. `name` is for logs, and what a warning names when this observer throws.
    """

    name: Optional[str]

    def on_run_start(self, event: RunStartEvent) -> Union[None, Awaitable[None]]: ...

    def on_node_start(self, event: NodeStartEvent) -> Union[None, Awaitable[None]]: ...

    def on_node_end(self, event: NodeEndEvent) -> Union[None, Awaitable[None]]: ...

    def on_run_end(self, event: RunEndEvent) -> Union[None, Awaitable[None]]: ...


async def notify_observers(
    observers: Optional[Sequence[Any]],
    hook_name: HookName,
    event: Union[RunStartEvent, NodeStartEvent, NodeEndEvent, RunEndEvent, RunFailEvent],
) -> None:
    """
    Call one hook on every observer, in registration order, swallowing failures.

    Sequential rather than gathered on purpose: a debug recorder's event
    stream is only readable if it arrives in order, and a watcher doing real
    work per event is a watcher doing it wrong.
    """
    if not observers:
        return
    for observer in observers:
        fn = getattr(observer, hook_name, None)
        if not callable(fn):
            continue
        try:
            await maybe_await(fn(event))
        except Exception as err:
            # The run already happened. A watcher failing is a lost measurement.
            _log.warning(
                "[pipeline/observers] %s.%s threw (ignored): %s",
                getattr(observer, "name", None) or "anonymous",
                hook_name,
                str(err),
            )


class _RecordSinkObserver:
    name = "records"

    def __init__(self, sink: Callable[[RunRecord], Any]):
        self._sink = sink

    def on_run_end(self, event: RunEndEvent) -> Any:
        return self._sink(event.record)


def record_sink_observer(sink: Callable[[RunRecord], Union[Awaitable[None], None]]) -> RunObserver:
    """
    `ports.records` as an observer, so there is ONE notification path rather than
    a sink beside a list. Note the contract change this carries: a record sink
    that throws no longer fails the run. It used to, which meant a run whose
    answer was already computed could still fail on the way to the training
    table — an outcome nothing wanted.
    """
    return _RecordSinkObserver(sink)  # type: ignore[return-value]
