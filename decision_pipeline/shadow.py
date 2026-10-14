"""
Shadow runs — a challenger PIPELINE beside the champion, on the same input.

An experiment arm (./experiments) is a patch, so it cannot add a node. A
proposal from ./propose can, so it runs here instead: the champion runs and
ships exactly as it would alone, then the challenger runs with `shadow=True`
and its store captured, never handed to the host. One ships, both are
recorded, and `compare_runs` says where they parted.

Model answers are SHARED by default. The champion's decide and generate
calls are memoized on what the model would be shown, and a challenger
request that would show the same thing gets the same answer. Two things
follow. The comparison measures the PIPELINE, not two samples of Jev (a
route near a gate flips between batches, #321 probes). And a challenger
that changes one branch costs one branch, not a second run. The fresh
calls, and their cost, are counted.

A challenger that throws never fails the champion: its error is returned.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

from ._compat import error_message, maybe_await, to_plain
from .define import DecisionPipeline
from .hash import stable_stringify
from .ports import DecideRequest, GenerateRequest, PipelinePorts, RunOptions, StoreRequest
from .run import RunResult, run_pipeline

__all__ = [
    "ShadowCalls",
    "ShadowResult",
    "RunSide",
    "RunComparison",
    "compare_runs",
    "run_shadow",
]


@dataclass(kw_only=True)
class ShadowCalls:
    #: Challenger model calls served from the champion's answers.
    shared: int = 0
    #: Challenger model calls that went to the port.
    fresh: int = 0
    #: What the fresh calls cost — the shadow's real price.
    fresh_cost_usd: float = 0


@dataclass(kw_only=True)
class RunSide:
    route: str
    confidence: Optional[float]
    cost_usd: float
    ms: float
    #: Nodes that ran (not skipped), in run order.
    ran: list[str]


@dataclass(kw_only=True)
class RunComparison:
    same_output: bool
    same_route: bool
    champion: RunSide
    challenger: RunSide
    only_champion: list[str]
    only_challenger: list[str]
    #: Gates both ran that chose differently: `{"id", "champion", "challenger"}`.
    branch_changed: list[dict[str, str]]
    #: challenger − champion, as each run's record states it (shared calls included).
    cost_delta_usd: float


@dataclass(kw_only=True)
class ShadowResult:
    champion: RunResult
    challenger: Optional[RunResult]
    challenger_error: Optional[str]
    #: What the challenger would have written. The host store never sees it.
    challenger_stores: list[StoreRequest] = field(default_factory=list)
    calls: ShadowCalls = field(default_factory=ShadowCalls)
    comparison: Optional[RunComparison] = None


def _side_of(r: RunResult) -> RunSide:
    return RunSide(
        route=r.route,
        confidence=r.confidence,
        cost_usd=r.record.cost_usd,
        ms=r.record.ms,
        ran=[n.id for n in r.record.nodes if n.skipped is None],
    )


def compare_runs(champion: RunResult, challenger: RunResult) -> RunComparison:
    a = _side_of(champion)
    b = _side_of(challenger)

    def branch_of(r: RunResult) -> dict[str, str]:
        return {n.id: n.branch for n in r.record.nodes if n.branch is not None}

    a_branch = branch_of(champion)
    b_branch = branch_of(challenger)
    branch_changed: list[dict[str, str]] = []
    for nid, branch in a_branch.items():
        other = b_branch.get(nid)
        if other is not None and other != branch:
            branch_changed.append({"id": nid, "champion": branch, "challenger": other})
    return RunComparison(
        same_output=stable_stringify(champion.output) == stable_stringify(challenger.output),
        same_route=a.route == b.route,
        champion=a,
        challenger=b,
        only_champion=[i for i in a.ran if i not in b.ran],
        only_challenger=[i for i in b.ran if i not in a.ran],
        branch_changed=branch_changed,
        cost_delta_usd=b.cost_usd - a.cost_usd,
    )


# What the model would be shown. Node ids, versions and gate verdicts are
# left out, and `from` is keyed by VALUE: a challenger that renames a node or
# inserts a gate (which passes its input through) upstream of a model call
# still asks the same question, and must get the same answer.
#
# The request's mappings (`in_`, `probes`) are read-only views; `to_plain`
# turns them, and the dataclasses inside, into the plain data TS would stringify.


def _key(value: Mapping[str, Any]) -> str:
    return stable_stringify(to_plain(value))


def _from_values(from_: Mapping[str, Any]) -> list[str]:
    return sorted({stable_stringify(to_plain(v)) for v in from_.values()})


def _decide_key(r: DecideRequest) -> str:
    return _key(
        {
            "d": r.questions,
            "probes": r.probes,
            "state": r.state,
            "features": r.features,
            "input": r.input,
            "from": _from_values(r.from_),
            "in": r.in_,
            "batched": r.batched,
        }
    )


def _generate_key(r: GenerateRequest) -> str:
    return _key(
        {
            "g": r.prompt,
            "route": r.route,
            "submit": r.submit,
            "tools": r.tools,
            "input": r.input,
            "from": _from_values(r.from_),
            "in": r.in_,
            "primary": r.primary,
            "probes": r.probes,
        }
    )


class _Failed:
    """The memo's marker for a champion call that raised."""

    __slots__ = ()


_FAILED = _Failed()


def _cost_of(value: Any) -> float:
    """`value.costUsd ?? 0` over a `DecideResult`/`GenerateResult` or the dict a port may return."""
    if isinstance(value, Mapping):
        c = value.get("cost_usd", value.get("costUsd"))
    else:
        c = getattr(value, "cost_usd", None)
    return 0 if c is None else c


async def run_shadow(
    *,
    champion: DecisionPipeline,
    challenger: DecisionPipeline,
    input: Any,
    ports: PipelinePorts,
    options: Optional[RunOptions] = None,
    label: str,
    share: bool = True,
) -> ShadowResult:
    """
    Run `champion` as it would run alone, then `challenger` as a shadow on the
    same input. `label` names the challenger (a proposal id) and is tagged
    `challenger:<label>` on its trace. `share` (default True) reuses the
    champion's model answers wherever the challenger would ask the same thing.
    """
    options = options or RunOptions()
    memo: dict[str, Any] = {}
    calls = ShadowCalls()

    def memoize(port: Callable[[Any], Any], key_of: Callable[[Any], str]) -> Callable[[Any], Awaitable[Any]]:
        # TS memoizes the promise; here the resolved value (or a failure marker) is stored under the key.
        async def call(r: Any) -> Any:
            key = key_of(r)
            try:
                value = await maybe_await(port(r))
            except BaseException:
                memo[key] = _FAILED
                raise
            memo[key] = value
            return value

        return call

    champion_ports = ports
    if share:
        changes: dict[str, Any] = {}
        if ports.decide:
            changes["decide"] = memoize(ports.decide, _decide_key)
        if ports.generate:
            changes["generate"] = memoize(ports.generate, _generate_key)
        champion_ports = dataclasses.replace(ports, **changes)

    champion_run = await run_pipeline(champion, input, champion_ports, options)

    async def replay(key: str, call: Callable[[], Any]) -> Any:
        hit = memo.get(key) if share else None
        # A champion call that failed is a miss, not an answer: the challenger
        # must not inherit an error its own call might not hit.
        if hit is not None and hit is not _FAILED:
            calls.shared += 1
            return hit
        value = await maybe_await(call())
        calls.fresh += 1
        calls.fresh_cost_usd += _cost_of(value)
        return value

    challenger_stores: list[StoreRequest] = []

    def capture(r: StoreRequest) -> None:
        challenger_stores.append(r)

    challenger_changes: dict[str, Any] = {
        # Belt and braces with `shadow=True`: the host's store is never reached.
        "store": capture,
    }
    if ports.decide:
        decide = ports.decide
        challenger_changes["decide"] = lambda r: replay(_decide_key(r), lambda: decide(r))
    if ports.generate:
        generate = ports.generate
        challenger_changes["generate"] = lambda r: replay(_generate_key(r), lambda: generate(r))
    challenger_ports = dataclasses.replace(ports, **challenger_changes)

    challenger_run: Optional[RunResult] = None
    challenger_error: Optional[str] = None
    try:
        challenger_run = await run_pipeline(
            challenger,
            input,
            challenger_ports,
            dataclasses.replace(
                options,
                shadow=True,
                # Same work, but not the same record: a join on run_key must not find two facts.
                run_key=f"{champion_run.run_key}:shadow:{label}",
                tags=[*(options.tags or []), f"challenger:{label}"],
                metadata={**(options.metadata or {}), "shadow_of": champion_run.run_id, "challenger": label},
            ),
        )
    except Exception as err:  # noqa: BLE001 - a challenger never fails the champion
        challenger_error = error_message(err)

    return ShadowResult(
        champion=champion_run,
        challenger=challenger_run,
        challenger_error=challenger_error,
        challenger_stores=challenger_stores,
        calls=calls,
        comparison=compare_runs(champion_run, challenger_run) if challenger_run is not None else None,
    )
