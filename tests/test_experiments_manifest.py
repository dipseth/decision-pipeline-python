# Needs define.py and fixtures.py (the core). Ported from the `definePipeline` /
# `buildScalePipeline` tests in src/experiments.test.ts.
import pytest
from pydantic import BaseModel

from decision_pipeline.define import EvalSpec, PipelineSpec, define_pipeline, validate_pipeline  # noqa: F401
from decision_pipeline.errors import PipelineContractError  # noqa: F401
from decision_pipeline.experiments import ExperimentSpec, GuardPatch, ManifestPatch
from decision_pipeline.fixtures import build_scale_pipeline
from decision_pipeline.types import CodeNode, Edge, EdgeGuard, FeedbackEdge, GateNode


def split(**over) -> ExperimentSpec:
    base = dict(
        id="gate-0.7-vs-0.6",
        unit="user",
        mode="split",
        champion="control",
        enabled=True,
        arms={
            "control": ManifestPatch(),
            "b": ManifestPatch(thresholds={"gate": {"category": 0.6}}),
        },
    )
    base.update(over)
    return ExperimentSpec(**base)


class In(BaseModel):
    id: str


class Out(BaseModel):
    ok: bool


def spec_with(unit: str) -> PipelineSpec:
    return PipelineSpec(
        id="demo",
        fact="f",
        input=In,
        output=Out,
        trigger=["on-demand"],
        group=lambda _i: "g",
        result="yes",
        nodes={
            "gate": GateNode(run=lambda _a: "yes", branches=["yes"], thresholds={"category": 0.7}, version="1"),
            "yes": CodeNode(role="derive", run=lambda _a: {"ok": True}, version="1"),
        },
        edges=[Edge(from_="gate", to="yes", when=EdgeGuard(gate="gate", branch="yes"))],
        feedback=[
            FeedbackEdge(
                id="drag",
                from_="surface:x",
                to="gate",
                source="implicit",
                scope="user",
                form="override",
                latency="immediate",
            )
        ],
        eval=EvalSpec(dataset="d", gate_metric="score_positive"),
        experiments=[split(unit=unit)],
    )


# --- assignment unit validation ---


def test_refuses_an_assignment_unit_narrower_than_the_widest_feedback_scope():
    # Otherwise a row written under arm B could be read by an arm-A run.
    assert "contaminate" in "\n".join(validate_pipeline(spec_with("device")))


def test_accepts_the_widest_feedback_scope_itself_or_wider():
    assert validate_pipeline(spec_with("user")) == []
    assert validate_pipeline(spec_with("tenant")) == []


# --- a variant patches only the declarative half ---


def test_accepts_a_threshold_patch_on_a_gate_that_declares_it():
    build_scale_pipeline(experiments=[split()])


def test_refuses_a_threshold_the_gate_never_declared():
    with pytest.raises(PipelineContractError, match="not a threshold the node declares"):
        build_scale_pipeline(
            experiments=[split(arms={"control": ManifestPatch(), "b": ManifestPatch(thresholds={"gate": {"nope": 0.1}})})]
        )


def test_refuses_patching_a_prompt_onto_a_node_that_is_not_a_generate_node():
    with pytest.raises(PipelineContractError, match="not a generate node"):
        build_scale_pipeline(
            experiments=[split(arms={"control": ManifestPatch(), "b": ManifestPatch(prompts={"table": "p"})})]
        )


def test_refuses_a_guard_patch_on_an_edge_that_does_not_exist_a_variant_does_not_add_edges():
    with pytest.raises(PipelineContractError, match="no such edge"):
        build_scale_pipeline(
            experiments=[
                split(
                    arms={
                        "control": ManifestPatch(),
                        "b": ManifestPatch(guards=[GuardPatch(from_="table", to="writer", when=None)]),
                    }
                )
            ]
        )


def test_refuses_an_experiment_with_one_arm_or_a_champion_that_is_not_an_arm():
    with pytest.raises(PipelineContractError, match="at least two arms"):
        build_scale_pipeline(experiments=[split(arms={"control": ManifestPatch()})])
    with pytest.raises(PipelineContractError, match="is not one of its arms"):
        build_scale_pipeline(experiments=[split(champion="ghost")])
