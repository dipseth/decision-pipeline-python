"""
Annotation queues and score configs — the scarce half of the Langfuse model.

A Langfuse **annotation queue is capped per project** (3 on the Core plan,
both projects). Five bespoke queue names are already declared across
recipes-core, and three of those modules carry a `catch` that logs "no
annotation queue (plan cap or API error)" and proceeds WITHOUT one — which
is the worst outcome available: the contract promises the signal a landing
spot and it silently has none.

So a feedback edge's `promote_via` names a **lane**, not a queue. Lanes are
unlimited; they resolve to one of a small number of registered physical
queues, and carry their own **score config** (configs are NOT capped, so
per-lane categories cost nothing). Exceeding the physical budget is a
manifest-load error here, not a runtime warning in production.

Nothing in this file talks to Langfuse. It is the registry the manifest
validates against; `QueuePort` is what actually writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .types import ScoreConfigSpec, score_config_from_dict

#: Langfuse Core plan, per project. Raising this is a billing decision, not a code one.
REVIEW_QUEUE_BUDGET = 3


@dataclass(kw_only=True)
class ReviewQueueSpec:
    #: The Langfuse queue name. One of at most REVIEW_QUEUE_BUDGET per project.
    name: str
    description: str


@dataclass(kw_only=True)
class ReviewLaneSpec:
    #: What a manifest's `promote_via` names.
    lane: str
    #: The physical queue it lands in. Must be registered.
    queue: str
    #: The score config a reviewer grades the item with (a dict is converted).
    score_config: Optional[ScoreConfigSpec] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if self.score_config is not None and not isinstance(self.score_config, ScoreConfigSpec):
            self.score_config = score_config_from_dict(self.score_config)


_QUEUES: dict[str, ReviewQueueSpec] = {}
_LANES: dict[str, ReviewLaneSpec] = {}


class ReviewBudgetError(Exception):
    def __init__(self, name: str, budget: int, existing: list[str]):
        super().__init__(
            f'Cannot register annotation queue "{name}": the budget of {budget} per Langfuse project '
            f'is already spent on [{", ".join(existing)}]. Add a LANE on an existing queue instead '
            "— lanes are free, queues are not."
        )
        self.name = "ReviewBudgetError"


def define_review_queue(
    spec: Optional[ReviewQueueSpec | Mapping[str, Any]] = None, *, name: Optional[str] = None, description: Optional[str] = None
) -> str:
    """Register a queue (spec dataclass, mapping, or `name=`/`description=` keywords). Returns its name."""
    if spec is None:
        spec = ReviewQueueSpec(name=name, description=description)  # type: ignore[arg-type]
    elif isinstance(spec, Mapping):
        spec = ReviewQueueSpec(**spec)
    if spec.name in _QUEUES:
        return spec.name
    if len(_QUEUES) >= REVIEW_QUEUE_BUDGET:
        raise ReviewBudgetError(spec.name, REVIEW_QUEUE_BUDGET, list(_QUEUES.keys()))
    _QUEUES[spec.name] = spec
    return spec.name


def define_review_lane(
    spec: Optional[ReviewLaneSpec | Mapping[str, Any]] = None,
    *,
    lane: Optional[str] = None,
    queue: Optional[str] = None,
    score_config: Any = None,
    description: Optional[str] = None,
) -> str:
    """Register a lane (spec dataclass, mapping, or keywords). Returns its name."""
    if spec is None:
        spec = ReviewLaneSpec(lane=lane, queue=queue, score_config=score_config, description=description)  # type: ignore[arg-type]
    elif isinstance(spec, Mapping):
        spec = ReviewLaneSpec(**{("score_config" if k == "scoreConfig" else k): v for k, v in spec.items()})
    if spec.queue not in _QUEUES:
        raise ValueError(
            f'Lane "{spec.lane}" points at unregistered queue "{spec.queue}" '
            f'(registered: {", ".join(_QUEUES.keys()) or "none"}).'
        )
    existing = _LANES.get(spec.lane)
    if existing is not None and existing.queue != spec.queue:
        raise ValueError(
            f'Lane "{spec.lane}" is already registered on queue "{existing.queue}"; '
            f'refusing to move it to "{spec.queue}".'
        )
    _LANES[spec.lane] = spec
    return spec.lane


def resolve_lane(lane: str) -> Optional[ReviewLaneSpec]:
    return _LANES.get(lane)


def has_lane(lane: str) -> bool:
    return lane in _LANES


def lane_names() -> list[str]:
    return sorted(_LANES.keys())


def review_queues() -> list[ReviewQueueSpec]:
    return list(_QUEUES.values())


def clear_review_registry() -> None:
    """Test seam — both registries are process-global."""
    _QUEUES.clear()
    _LANES.clear()


@dataclass(kw_only=True)
class ReviewRegistrySnapshot:
    queues: list[ReviewQueueSpec]
    lanes: list[ReviewLaneSpec]


def snapshot_review_registry() -> ReviewRegistrySnapshot:
    """
    Save/restore rather than clear: the package registers the two shared queues
    at import, so a test that clears would break every later test in the file.
    """
    return ReviewRegistrySnapshot(queues=list(_QUEUES.values()), lanes=list(_LANES.values()))


def restore_review_registry(snapshot: ReviewRegistrySnapshot) -> None:
    _QUEUES.clear()
    _LANES.clear()
    for q in snapshot.queues:
        _QUEUES[q.name] = q
    for lane in snapshot.lanes:
        _LANES[lane.lane] = lane


# ---------------------------------------------------------------------------
# The two queues every pipeline shares
# ---------------------------------------------------------------------------

#: Every `queue`- and `derived`-form promotion across every pipeline. The item
#: is the run's TRACE, and which pipeline and which feedback edge sent it ride
#: in the item's metadata — that is what a lane buys over a queue per pipeline.
PIPELINE_REVIEW_QUEUE = define_review_queue(
    name="pipeline-review",
    description=(
        "Decision-pipeline promotions (#321). One item = one pipeline run's trace. The item's metadata names "
        "the pipeline, the feedback edge and the lane; grade it with the lane's score config. A `derived` lane "
        "is proposing a CHANGE TO THE PIPELINE (a new Jev question, a taxonomy slug, a threshold) — promoting "
        "it edits structure, so read the proposal, not just the run."
    ),
)

#: Runs pulled for a hand read rather than promoted by a user — the gate
#: student's flagging surface (#276-style) and the #293 read.
PIPELINE_HAND_READ_QUEUE = define_review_queue(
    name="pipeline-hand-read",
    description=(
        "Decision-pipeline hand reads (#321). Sampled or gate-student-flagged runs, not user promotions. One "
        "item = one run's trace: read the node spans top to bottom, then grade whether the BRANCH the gate took "
        "was right — the pipeline's own answer is persisted separately from any human override, so a corrected "
        "run is still scorable."
    ),
)

# The third queue is deliberately left for the host to spend.
