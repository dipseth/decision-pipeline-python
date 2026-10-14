"""
Submitting a human signal — the producer of `RunLabel`.

One entry point, `submit_feedback`, so every surface — a 👍/👎, a rating, a
review promotion — goes through the same rules:

  - a `score` edge writes a score on the RUN'S TRACE and a label row;
  - a `queue` / `derived` edge puts the run's trace in the lane's physical
    queue, and writes a score too when the edge declares one;
  - an `override` / `feature` edge returns the ROW for the host to persist
    (the core never owns a database), and an override that disagrees with
    the pipeline's own answer emits the disagreement label.

Every port call is best-effort: the tracer being down must never lose the durable row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from ._compat import error_message, iso_from_ms, maybe_await, now_ms
from .define import DecisionPipeline
from .ports import PipelinePorts, QueueRequest, QueueResult, ScoreRequest
from .record import RunLabel
from .review import resolve_lane
from .types import DIRECT_FORM_MAX_SCOPE, FeatureRow, FeedbackEdge, OverrideRow, ScoreConfigSpec, scope_at_most


@dataclass(kw_only=True)
class FeedbackSubmission:
    #: The run's trace id — what the score and the queue item attach to.
    run_id: str
    #: The run's DURABLE key (`RunRecord.run_key`), when the caller has it.
    run_key: Optional[str] = None
    #: Which declared feedback edge fired.
    edge_id: str
    #: Numeric value. Optional when `label` names a category of the edge's config.
    value: Optional[float] = None
    #: CATEGORICAL label; resolves to a value through the edge's score config.
    label: Optional[str] = None
    comment: Optional[str] = None
    #: A property of the WRITER. Default `human`.
    rank: Optional[str] = None
    #: The surface, row or person this came from.
    source_id: str
    user_id: Optional[str] = None
    #: `override` / `feature` forms: what to bind, and to what key.
    key: Optional[str] = None
    #: `override` / `feature` forms: the value to persist. `MISSING`-free: None is a value.
    payload: Any = None
    #: `override` form: the pipeline's OWN answer, so a disagreement can be labelled. Leave unset for none.
    model_value: Any = None
    #: Target node for a `feature` row. Defaults to the edge's `to`.
    node_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    arm: Optional[str] = None
    #: Set when `model_value` was given (None is a legitimate model answer).
    has_model_value: bool = False


@dataclass(kw_only=True)
class FeedbackResult:
    edge: FeedbackEdge
    #: The label row to persist and later join to the run record.
    label: Optional[RunLabel]
    #: The durable row the HOST persists. The core never owns a database.
    row: Union[FeatureRow, OverrideRow, None]
    queued: Optional[QueueResult]
    scored: bool
    #: Anything refused or degraded — never thrown, so a row is never lost.
    warnings: list[str]


class UnknownFeedbackEdgeError(Exception):
    def __init__(self, pipeline: str, edge_id: str, known: list[str]):
        super().__init__(f'Pipeline "{pipeline}" has no feedback edge "{edge_id}" (has: {", ".join(known) or "none"}).')


def category_value(config: Optional[ScoreConfigSpec], label: Optional[str]) -> Optional[float]:
    """A category's number, or None when the label is not one of them."""
    if not config or not config.categories or label is None:
        return None
    hit = next((c for c in config.categories if c.label == label), None)
    return hit.value if hit else None


def _submission(submission: Union[FeedbackSubmission, dict[str, Any], None], kwargs: dict[str, Any]) -> FeedbackSubmission:
    if isinstance(submission, FeedbackSubmission):
        return submission
    d = {**(submission or {}), **kwargs}
    if "model_value" in d:
        d["has_model_value"] = True
    return FeedbackSubmission(**d)


async def submit_feedback(
    pipeline: DecisionPipeline,
    submission: Union[FeedbackSubmission, dict[str, Any], None] = None,
    ports: Optional[PipelinePorts] = None,
    **kwargs: Any,
) -> FeedbackResult:
    ports = ports or PipelinePorts()
    s = _submission(submission, kwargs)
    edge = next((f for f in pipeline.feedback if f.id == s.edge_id), None)
    if not edge:
        raise UnknownFeedbackEdgeError(pipeline.id, s.edge_id, [f.id for f in pipeline.feedback])

    warnings: list[str] = []
    rank = s.rank or "human"
    created_at = iso_from_ms(ports.clock() if ports.clock else now_ms())

    # --- resolve the number, from a category label when that is what came in
    from_label = category_value(edge.score, s.label)
    if s.label is not None and from_label is None and edge.score and edge.score.categories:
        warnings.append(f'label "{s.label}" is not a category of score config "{edge.score.name}"')
    value = s.value if s.value is not None else from_label if from_label is not None else 0

    result = FeedbackResult(edge=edge, label=None, row=None, queued=None, scored=False, warnings=warnings)

    async def write_score(kind: Literal["score", "override_disagreement", "verify", "implicit"]) -> None:
        """The score. Any form may carry one; `score` form must."""
        if not edge.score:
            return
        if not ports.scores:
            warnings.append("no `scores` port supplied — the Langfuse score was not written")
        else:
            try:
                await maybe_await(
                    ports.scores(
                        ScoreRequest(
                            run_id=s.run_id,
                            name=edge.score.name,
                            value=value,
                            label=s.label,
                            data_type=edge.score.data_type,
                            comment=s.comment,
                            config=edge.score,
                            user_id=s.user_id,
                        )
                    )
                )
                result.scored = True
            except Exception as err:  # noqa: BLE001 — best-effort by contract
                warnings.append(f"score write failed: {error_message(err)}")
        result.label = RunLabel(
            run_id=s.run_id,
            run_key=s.run_key,
            kind=kind,
            value=value,
            label=s.label,
            score_name=edge.score.name,
            edge_id=edge.id,
            scope=edge.scope,
            rank=rank,
            source_id=s.source_id,
            created_at=created_at,
        )

    if edge.form == "score":
        if not edge.score:
            # define_pipeline refuses this, so it means a hand-built manifest.
            warnings.append(f'edge "{edge.id}" is form "score" but declares no score config')
        else:
            await write_score("score")

    elif edge.form in ("queue", "derived"):
        lane = resolve_lane(edge.promote_via) if edge.promote_via else None
        if not lane:
            warnings.append(f'lane "{edge.promote_via or "(none)"}" is not registered — nothing was enqueued')
        elif not ports.queue:
            warnings.append("no `queue` port supplied — nothing was enqueued")
        else:
            try:
                queued = await maybe_await(
                    ports.queue(
                        QueueRequest(
                            run_id=s.run_id,
                            queue=lane.queue,
                            lane=lane.lane,
                            pipeline=pipeline.id,
                            edge_id=edge.id,
                            form=edge.form,
                            score_config=lane.score_config or edge.score,
                            metadata={
                                "fact": pipeline.fact,
                                "node": edge.to,
                                "source": edge.source,
                                "scope": edge.scope,
                                "arm": s.arm,
                                **(s.metadata or {}),
                            },
                        )
                    )
                )
                result.queued = queued if queued is not None else None
            except Exception as err:  # noqa: BLE001
                warnings.append(f"enqueue failed: {error_message(err)}")
        # A score is not state: an edge that also carries one writes it too.
        await write_score("implicit")

    elif edge.form in ("override", "feature"):
        # The form x scope rule, enforced on WRITE as well as on read.
        if not scope_at_most(edge.scope, DIRECT_FORM_MAX_SCOPE):
            warnings.append(f'form "{edge.form}" is not allowed at scope "{edge.scope}" — no row was produced')
        else:
            row_id = f"{pipeline.id}:{edge.id}:{s.source_id}"
            if edge.form == "override":
                if s.key is None:
                    warnings.append("an override row needs a `key` — no row was produced")
                else:
                    result.row = OverrideRow(id=row_id, key=s.key, scope=edge.scope, rank=rank, value=s.payload, arm=s.arm, created_at=created_at)
                    # The override IS the disagreement — but only when the human actually differed.
                    if s.has_model_value:
                        disagreed = s.payload != s.model_value
                        result.label = RunLabel(
                            run_id=s.run_id,
                            run_key=s.run_key,
                            kind="override_disagreement",
                            value=1 if disagreed else 0,
                            edge_id=edge.id,
                            scope=edge.scope,
                            rank=rank,
                            source_id=s.source_id,
                            created_at=created_at,
                        )
                    if edge.score:
                        await write_score("implicit" if edge.source == "implicit" else "score")
            else:
                result.row = FeatureRow(id=row_id, node_id=s.node_id or edge.to, scope=edge.scope, rank=rank, value=s.payload, arm=s.arm, created_at=created_at)
                if edge.score:
                    await write_score("implicit" if edge.source == "implicit" else "score")

    return result


async def enqueue_for_hand_read(
    pipeline: DecisionPipeline,
    *,
    run_id: str,
    reason: str,
    lane: str,
    metadata: Optional[dict[str, Any]] = None,
    ports: Optional[PipelinePorts] = None,
) -> Optional[QueueResult]:
    """Pull a run into the hand-read queue. Not a user promotion: sampling, or a gate student flagging a run."""
    ports = ports or PipelinePorts()
    resolved = resolve_lane(lane)
    if not resolved or not ports.queue:
        return None
    queued = await maybe_await(
        ports.queue(
            QueueRequest(
                run_id=run_id,
                queue=resolved.queue,
                lane=resolved.lane,
                pipeline=pipeline.id,
                edge_id="hand-read",
                form="queue",
                score_config=resolved.score_config,
                metadata={"fact": pipeline.fact, "reason": reason, **(metadata or {})},
            )
        )
    )
    return queued if queued is not None else None
