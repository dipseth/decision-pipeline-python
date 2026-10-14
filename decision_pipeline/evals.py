"""
Binding `eval` to the tracer's own experiment model — a dataset and a dataset
run — so an arm comparison is a chart in the UI rather than two scratch
folders. The run NAME is the arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ._compat import error_message, maybe_await
from .define import DecisionPipeline
from .ports import DatasetItem, DatasetRunRequest, PipelinePorts
from .record import RunLabel, RunRecord, aggregate_gate_metric


def eval_item_id(dataset: str, id: str) -> str:
    """Deterministic, so re-running upserts instead of duplicating."""
    return f"{dataset}/{id}"[:255]


def eval_run_name(*, pipeline: str, version: str, experiment: Optional[str] = None, arm: Optional[str] = None) -> str:
    """`<pipeline>@<version>` for a plain eval, `<pipeline>:<experiment>=<arm>@<version>` for an arm."""
    head = f"{pipeline}:{experiment}={arm}" if experiment and arm else pipeline
    return f"{head}@{version}"


@dataclass(kw_only=True)
class EvalRow:
    id: str
    input: Any
    record: RunRecord
    labels: Sequence[RunLabel] = ()
    expected_output: Any = None


@dataclass(kw_only=True)
class PipelineEvalRun:
    rows: list[EvalRow]
    #: The harness's own root trace. Run items and the aggregate scores attach here.
    trace_id: Optional[str] = None
    experiment: Optional[str] = None
    arm: Optional[str] = None
    #: Extra aggregates beyond `eval.gate_metric`, as `(name, value)` pairs.
    extra_scores: Optional[Sequence[tuple[str, float]]] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class PipelineEvalResult:
    dataset: str
    run: str
    items: int
    #: The manifest's `eval.gate_metric`, computed the ONE way: `{"name", "n", "mean"}`.
    gate_metric: dict[str, Any]
    submitted: bool
    warnings: list[str] = field(default_factory=list)


def _row(r: Any) -> EvalRow:
    if isinstance(r, EvalRow):
        return r
    return EvalRow(id=r["id"], input=r.get("input"), record=r["record"], labels=r.get("labels", ()), expected_output=r.get("expected_output", r.get("expectedOutput")))


async def record_pipeline_eval_run(
    pipeline: DecisionPipeline,
    run: PipelineEvalRun,
    ports: Optional[PipelinePorts] = None,
) -> PipelineEvalResult:
    """Computes `eval.gate_metric` through the registered extractor and registers the run through the dataset port."""
    ports = ports or PipelinePorts()
    rows = [_row(r) for r in run.rows]
    warnings: list[str] = []
    version = rows[0].record.version if rows else "unversioned"
    name = eval_run_name(pipeline=pipeline.id, version=version, experiment=run.experiment, arm=run.arm)

    versions = []
    for r in rows:
        if r.record.version not in versions:
            versions.append(r.record.version)
    if len(versions) > 1:
        warnings.append(f"rows span {len(versions)} composite versions ({', '.join(versions)}) — the run name uses the first")

    gate_metric = aggregate_gate_metric(pipeline.eval.gate_metric, [{"record": r.record, "labels": r.labels} for r in rows])

    scores: list[tuple[str, float]] = []
    if gate_metric["mean"] is not None:
        scores.append((gate_metric["name"], gate_metric["mean"]))
    for s in run.extra_scores or []:
        scores.append(tuple(s))  # type: ignore[arg-type]

    request = DatasetRunRequest(
        dataset=pipeline.eval.dataset,
        run=name,
        run_metadata={
            "pipeline": pipeline.id,
            "fact": pipeline.fact,
            "version": version,
            "experiment": run.experiment,
            "arm": run.arm,
            "gate_metric": gate_metric["name"],
            "n": gate_metric["n"],
            **(run.metadata or {}),
        },
        items=[
            DatasetItem(
                id=r.id,
                input=r.input,
                expected_output=r.expected_output,
                metadata={"run_id": r.record.run_id, "route": r.record.fact_route, "confidence": r.record.fact_confidence, "arms": r.record.arms},
            )
            for r in rows
        ],
        trace_id=run.trace_id,
        scores=scores,
    )

    submitted = False
    if not ports.dataset:
        warnings.append("no `dataset` port supplied — the run was computed but not registered")
    else:
        try:
            await maybe_await(ports.dataset(request))
            submitted = True
        except Exception as err:  # noqa: BLE001
            warnings.append(f"dataset run failed: {error_message(err)}")

    return PipelineEvalResult(dataset=pipeline.eval.dataset, run=name, items=len(rows), gate_metric=gate_metric, submitted=submitted, warnings=warnings)
