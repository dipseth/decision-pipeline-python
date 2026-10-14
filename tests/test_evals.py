"""Port of src/eval.test.ts."""

from __future__ import annotations

import dataclasses

from decision_pipeline import PipelinePorts, RunLabel, RunRecord
from decision_pipeline import testing as t
from decision_pipeline.evals import EvalRow, PipelineEvalRun, eval_item_id, eval_run_name, record_pipeline_eval_run
from decision_pipeline.fixtures import build_scale_pipeline

scale = build_scale_pipeline()


def record(**over) -> RunRecord:
    base = dict(
        pipeline="scale", version="v-aaa", run_id="r1", run_key="r1", group="cook-1",
        tenant=None, user=None, trigger="on-demand", input_hash="h",
        cost_usd=0, ms=1, nodes=[], features={}, fact="scaled_recipe",
        fact_route="direct", fact_confidence=0.9, consumed_feedback_ids=[],
        arms={}, parent_run_id=None, assignment_unit=None, assignment_hash=None,
        created_at="2026-09-22T00:00:00.000Z",
    )
    base.update(over)
    return RunRecord(**base)


def label(value: float) -> RunLabel:
    return RunLabel(
        run_id="r1", kind="score", value=value, scope="user", rank="human",
        source_id="s", created_at="2026-09-22T00:00:00.000Z",
    )


# --- naming -------------------------------------------------------------------


def test_makes_item_ids_deterministic_so_a_re_run_upserts():
    assert eval_item_id("scale-eval", "recipe-7") == "scale-eval/recipe-7"
    assert len(eval_item_id("d", "x" * 400)) == 255


def test_puts_the_arm_in_the_run_name_so_langfuses_run_comparison_is_the_experiment_readout():
    assert eval_run_name(pipeline="scale", version="v1") == "scale@v1"
    assert eval_run_name(pipeline="scale", version="v1", experiment="gate-tune", arm="tight") == "scale:gate-tune=tight@v1"


# --- record_pipeline_eval_run -------------------------------------------------

ROWS = [
    EvalRow(id="a", input={"id": "a"}, record=record(run_id="r1"), labels=[label(1)]),
    EvalRow(id="b", input={"id": "b"}, record=record(run_id="r2"), labels=[label(-1)]),
    EvalRow(id="c", input={"id": "c"}, record=record(run_id="r3"), labels=[]),
]


async def test_computes_eval_gate_metric_through_the_registered_extractor_not_an_ad_hoc_query():
    ports = t.test_ports()
    result = await record_pipeline_eval_run(scale, PipelineEvalRun(rows=ROWS, trace_id="t-root"), ports)

    # One label positive, one negative, one unlabelled and therefore dropped.
    assert result.gate_metric == {"name": "score_positive", "n": 2, "mean": 0.5}
    assert result.submitted is True


async def test_registers_the_dataset_run_with_the_metric_as_a_score_on_the_harness_trace():
    ports = t.test_ports()
    await record_pipeline_eval_run(
        scale,
        PipelineEvalRun(rows=ROWS, trace_id="t-root", experiment="gate-tune", arm="tight", extra_scores=[("cost_usd", 0.004)]),
        ports,
    )

    run = ports.dataset_runs[0]
    assert run.dataset == "scale-eval"
    assert run.run == "scale:gate-tune=tight@v-aaa"
    assert run.trace_id == "t-root"
    assert len(run.items) == 3
    assert [tuple(s) for s in run.scores] == [("score_positive", 0.5), ("cost_usd", 0.004)]
    assert {k: run.run_metadata[k] for k in ("pipeline", "arm", "n")} == {"pipeline": "scale", "arm": "tight", "n": 2}


async def test_warns_when_rows_span_composite_versions_two_such_runs_are_not_comparable():
    mixed = [ROWS[0], dataclasses.replace(ROWS[1], record=record(version="v-bbb"))]
    result = await record_pipeline_eval_run(scale, PipelineEvalRun(rows=mixed, trace_id="t"), t.test_ports())
    assert "span 2 composite versions" in ",".join(result.warnings)


async def test_still_computes_the_metric_with_no_dataset_port_and_says_it_did_not_register():
    result = await record_pipeline_eval_run(scale, PipelineEvalRun(rows=ROWS, trace_id="t"), PipelinePorts())
    assert result.gate_metric["mean"] == 0.5
    assert result.submitted is False
    assert "no `dataset` port" in ",".join(result.warnings)
