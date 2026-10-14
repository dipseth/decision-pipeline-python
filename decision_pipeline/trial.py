"""
Trials — a locked hypothesis tested on the pipelines it names.

`hypothesis.py` judges rows; this is where the rows come from. A trial pins
three things before any data exists: the claim (its hash), the champion and
the challenger (their fingerprints). Then it runs the challenger as a shadow
on the inputs the claim's split allows, asks the host for truth, keeps every
row in a store, and judges.

  register_trial   lock once; a changed claim under the same id is refused
  collect_trial    shadow runs → rows, appended as they finish (resumable)
  judge_trial      evaluate_hypothesis over the stored rows, recorded as a READ

Every judgement is kept, and reads of the holdout are counted: the first is
the confirmatory one, a later read is a peek and says so.

The store is a port, like everything else here: the core owns no disk and
no database. `memory_trial_store` (./testing) is the reference.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol, Sequence

from ._compat import maybe_await
from .hash import short_hash
from .hypothesis import (
    CompiledHypothesis,
    HypothesisLock,
    HypothesisVerdict,
    evaluate_hypothesis,
    lock_hypothesis,
    observations_from_shadow,
    split_of,
)
from .metrics import ObservationRow
from .ports import PipelinePorts, RunOptions

if TYPE_CHECKING:  # pragma: no cover
    from .define import DecisionPipeline


def _edge_json(e: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"from": e.from_, "to": e.to}
    if e.when is not None:
        out["when"] = {"gate": e.when.gate, "branch": e.when.branch}
    return out


def pipeline_fingerprint(p: "DecisionPipeline") -> str:
    """
    What a pipeline IS, for pinning: its declarative graph, plus the live node
    versions and edges (a pipeline copied with `dataclasses.replace(base, nodes=...)`
    keeps the base's declared graph). Code bodies are not in it: a compiled spec
    node's version carries its body hash, a Python node's declared version is the author's promise.
    """
    return short_hash(
        {
            "json": p.to_json(),
            "nodes": {nid: n.version for nid, n in p.nodes.items()},
            "edges": [_edge_json(e) for e in p.edges],
        }
    )


@dataclass(kw_only=True)
class TrialArm:
    #: The arm name the hypothesis's estimand uses.
    name: str
    fingerprint: str


@dataclass(kw_only=True)
class TrialLock:
    trial: str
    hypothesis: HypothesisLock
    champion: TrialArm
    challenger: TrialArm
    #: Over the claim's hash and both fingerprints.
    hash: str
    registered_at: str


@dataclass(kw_only=True)
class TrialJudgement:
    verdict: HypothesisVerdict
    #: 1 = the first read of this split. Only the first read of the holdout is confirmatory.
    read: int
    #: "holdout" | "dev" | "all"
    split: str
    rows: int


class TrialStore(Protocol):
    """Where locks, rows and judgements live. Appends only. Methods may be sync or async."""

    async def get_lock(self, trial: str) -> Optional[TrialLock]: ...

    async def put_lock(self, lock: TrialLock) -> None: ...

    async def append_rows(self, trial: str, rows: Sequence[ObservationRow]) -> None: ...

    async def rows(self, trial: str) -> list[ObservationRow]: ...

    async def append_judgement(self, trial: str, judgement: TrialJudgement) -> None: ...

    async def judgements(self, trial: str) -> list[TrialJudgement]: ...


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, Mapping) else getattr(obj, key)


def lock_trial(h: CompiledHypothesis, arms: Any, at: Optional[datetime] = None) -> TrialLock:
    """`arms`: `{"champion": {"name", "pipeline"}, "challenger": {"name", "pipeline"}}` (dicts or objects)."""
    hypothesis = lock_hypothesis(h, at)
    ch, cl = _get(arms, "champion"), _get(arms, "challenger")
    champion = TrialArm(name=_get(ch, "name"), fingerprint=pipeline_fingerprint(_get(ch, "pipeline")))
    challenger = TrialArm(name=_get(cl, "name"), fingerprint=pipeline_fingerprint(_get(cl, "pipeline")))
    return TrialLock(
        trial=h.spec.id,
        hypothesis=hypothesis,
        champion=champion,
        challenger=challenger,
        hash=short_hash({"hypothesis": h.hash, "champion": champion, "challenger": challenger}),
        registered_at=hypothesis.registered_at,
    )


async def register_trial(store: Any, lock: TrialLock) -> TrialLock:
    """
    Stores the lock, or returns the one already stored when it is the same
    trial. A different claim or pipeline under a registered id raises: a
    preregistration that can be edited is not one — give the new version a new id.
    """
    existing = await maybe_await(store.get_lock(lock.trial))
    if existing:
        if existing.hash != lock.hash:
            raise Exception(
                f'trial "{lock.trial}" is already locked ({existing.hash}, {existing.registered_at}) with a different claim or pipeline ({lock.hash}) — register the change under a new id'
            )
        return existing
    await maybe_await(store.put_lock(lock))
    return lock


@dataclass(kw_only=True)
class TruthArgs:
    input: Any
    id: str
    champion: Any
    challenger: Any


@dataclass(kw_only=True)
class MetaArgs(TruthArgs):
    """What `meta` sees: the truth args, plus the shadow result, the row's arm and its output."""

    shadow: Any
    arm: str
    output: Any


@dataclass(kw_only=True)
class TrialPair:
    """What `on_pair` receives as each pair lands."""

    id: str
    shadow: Any
    rows: list[ObservationRow]
    truth_error: Optional[str]


@dataclass(kw_only=True)
class TrialSkipped:
    other_split: int = 0
    already_collected: int = 0


@dataclass(kw_only=True)
class TrialCalls:
    shared: float = 0
    fresh: float = 0
    fresh_cost_usd: float = 0


@dataclass(kw_only=True)
class CollectTrialResult:
    collected: int = 0
    skipped: TrialSkipped = field(default_factory=TrialSkipped)
    challenger_errors: int = 0
    truth_errors: int = 0
    calls: TrialCalls = field(default_factory=TrialCalls)
    cost_usd: dict[str, float] = field(default_factory=dict)


async def _require_lock(h: CompiledHypothesis, store: Any) -> TrialLock:
    lock = await maybe_await(store.get_lock(h.spec.id))
    if not lock:
        raise Exception(f'trial "{h.spec.id}" is not registered — lock it before collecting')
    if lock.hypothesis.hash != h.hash:
        raise Exception(f'trial "{h.spec.id}": the hypothesis changed since it was locked ({lock.hypothesis.hash} → {h.hash})')
    return lock


async def collect_trial(
    *,
    hypothesis: CompiledHypothesis,
    store: Any,
    champion: "DecisionPipeline",
    challenger: "DecisionPipeline",
    inputs: Sequence[Any],
    id: Callable[[Any], str],
    unit: Optional[Callable[[Any], str]] = None,
    ports: PipelinePorts,
    options: Optional[RunOptions] = None,
    truth: Optional[Callable[[TruthArgs], Any]] = None,
    meta: Optional[Callable[[MetaArgs], Mapping[str, Any]]] = None,
    concurrency: int = 1,
    on_pair: Optional[Callable[[TrialPair], Any]] = None,
) -> CollectTrialResult:
    """
    Runs the shadows the lock allows, and appends each pair's rows as it lands.

    `id`: the ITEM id — a row's `id`, and how a resumed collection knows what it
    already has. `unit`: the independent unit (a recipe, a user), default the id;
    it decides the split BEFORE anything runs and becomes `row.unit`, so a
    hypothesis that names `population.unit` must name one this agrees with.
    `truth`: ground truth for one pair, or None when there is none (sync or async);
    it sees both outputs so a judge can be blind to arms. `meta`: extra `meta`
    per row. `on_pair`: called as each pair lands.
    """
    from .shadow import run_shadow  # the shadow runner imports the runtime; keep this module light

    lock = await _require_lock(hypothesis, store)
    for side, p in (("champion", champion), ("challenger", challenger)):
        now = pipeline_fingerprint(p)
        locked = getattr(lock, side).fingerprint
        if now != locked:
            raise Exception(f'trial "{lock.trial}": the {side} is not the pipeline that was locked ({locked} → {now})')

    split = hypothesis.spec.population.split
    unit_of = unit if unit is not None else id
    have = {r.id for r in await maybe_await(store.rows(lock.trial))}
    out = CollectTrialResult(cost_usd={lock.champion.name: 0, lock.challenger.name: 0})

    queue: list[Any] = []
    for item in inputs:
        if split is not None and split_of(unit_of(item), split) != split.use:
            out.skipped.other_split += 1
        elif id(item) in have:
            out.skipped.already_collected += 1
        else:
            queue.append(item)

    base_options = options if options is not None else RunOptions()
    run_options = dataclasses.replace(base_options, tags=[*(base_options.tags or []), f"trial:{lock.trial}"])

    async def one(item: Any) -> None:
        item_id = id(item)
        shadow = await run_shadow(
            champion=champion,
            challenger=challenger,
            input=item,
            ports=ports,
            options=run_options,
            label=lock.challenger.name,
        )
        out.calls.shared += shadow.calls.shared
        out.calls.fresh += shadow.calls.fresh
        out.calls.fresh_cost_usd += shadow.calls.fresh_cost_usd
        if shadow.challenger_error:
            out.challenger_errors += 1

        args = TruthArgs(
            input=item,
            id=item_id,
            champion=shadow.champion.output,
            challenger=shadow.challenger.output if shadow.challenger is not None else None,
        )
        truth_value: Any = None
        truth_error: Optional[str] = None
        if truth is not None:
            try:
                truth_value = await maybe_await(truth(args))
            except Exception as err:  # noqa: BLE001 - counted, the pair is kept without truth
                truth_error = str(err)
                out.truth_errors += 1
        rows = observations_from_shadow(
            shadow,
            id=item_id,
            unit=unit_of(item),
            truth=truth_value,
            champion=lock.champion.name,
            challenger=lock.challenger.name,
        )
        if meta is not None:
            rows = [
                dataclasses.replace(
                    r,
                    meta={
                        **(r.meta or {}),
                        **meta(
                            MetaArgs(
                                input=args.input,
                                id=args.id,
                                champion=args.champion,
                                challenger=args.challenger,
                                shadow=shadow,
                                arm=r.arm,  # type: ignore[arg-type]
                                output=r.prediction,
                            )
                        ),
                    },
                )
                for r in rows
            ]
        for r in rows:
            cost = (r.meta or {}).get("cost_usd")
            out.cost_usd[r.arm] = out.cost_usd.get(r.arm, 0) + (0 if cost is None else float(cost))  # type: ignore[index,arg-type]
        await maybe_await(store.append_rows(lock.trial, rows))
        out.collected += 1
        if on_pair is not None:
            on_pair(TrialPair(id=item_id, shadow=shadow, rows=rows, truth_error=truth_error))

    width = max(1, concurrency if concurrency is not None else 1)
    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while next_index < len(queue):
            item = queue[next_index]
            next_index += 1
            await one(item)

    await asyncio.gather(*(worker() for _ in range(min(width, len(queue)))))
    return out


async def judge_trial(hypothesis: CompiledHypothesis, store: Any, now: Optional[datetime] = None) -> TrialJudgement:
    """
    Judges the stored rows and records the read. A judgement of a split that was
    already read says so: the first holdout read is the confirmatory verdict.
    """
    lock = await _require_lock(hypothesis, store)
    rows = await maybe_await(store.rows(lock.trial))
    verdict = await evaluate_hypothesis(hypothesis, rows, lock=lock.hypothesis, now=now)
    split = hypothesis.spec.population.split.use if hypothesis.spec.population.split is not None else "all"
    read = len([j for j in await maybe_await(store.judgements(lock.trial)) if j.split == split]) + 1
    if read > 1 and split == "holdout":
        verdict.warnings.append(f"holdout read {read}: only read 1 is confirmatory — this one is a re-read")
    judgement = TrialJudgement(verdict=verdict, read=read, split=split, rows=len(rows))
    await maybe_await(store.append_judgement(lock.trial, judgement))
    return judgement
