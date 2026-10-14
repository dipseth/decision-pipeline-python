"""
Port of src/transform-thresholds.test.ts — thresholds on a transform (#321).

`apply` turns Jev's Nouls into cuts at thresholds and never branches. These
tests pin the rule that a transform's thresholds behave exactly like a gate's.
"""

from __future__ import annotations

from pydantic import BaseModel

from decision_pipeline import define_pipeline, run_pipeline, validate_patch
from decision_pipeline import testing as t
from decision_pipeline.experiments import ManifestPatch
from decision_pipeline.types import CodeNode, Edge, EdgeGuard, GateNode


class In(BaseModel):
    p: float


class Out(BaseModel):
    kept: bool


def build():
    return define_pipeline(
        id="transform-thresholds",
        fact="kept",
        input=In,
        output=Out,
        trigger=["on-demand"],
        group=lambda _i: "g",
        result=["apply"],
        nodes={
            "apply": CodeNode(
                role="transform",
                thresholds={"keep": {"env": "TT_KEEP_P", "default": 0.5}},
                run=lambda args: {"kept": args.input.p >= args.thresholds["keep"]},
                version="1",
            ),
            "gate": GateNode(thresholds={}, branches=["done"], run=lambda _a: "done", version="1"),
            "sink": CodeNode(role="derive", run=lambda args: args.primary, version="1"),
        },
        edges=[
            Edge(from_="apply", to="gate"),
            Edge(from_="gate", to="sink", when=EdgeGuard(gate="gate", branch="done")),
        ],
        feedback=[
            {
                "id": "vote",
                "from": "surface:test",
                "to": "apply",
                "source": "explicit",
                "scope": "user",
                "form": "score",
                "latency": "deferred",
                "score": {"name": "tt_vote", "dataType": "NUMERIC"},
            }
        ],
        eval={"dataset": "tt", "gateMetric": "score_positive"},
    )


def apply_node(run):
    return next(n for n in run.record.nodes if n.id == "apply")


async def test_hands_the_resolved_number_to_the_body_and_records_it_on_the_node():
    run = await run_pipeline(build(), {"p": 0.6}, t.test_ports())
    assert run.output == Out(kept=True)
    assert apply_node(run).thresholds == {"keep": 0.5}


async def test_reads_the_env_override_the_same_way_a_gate_does():
    run = await run_pipeline(build(), {"p": 0.6}, t.test_ports(env={"TT_KEEP_P": "0.7"}))
    assert run.output == Out(kept=False)
    assert apply_node(run).thresholds == {"keep": 0.7}


async def test_is_patchable_by_a_replay_and_the_patch_moves_the_version():
    base = await run_pipeline(build(), {"p": 0.6}, t.test_ports())
    tuned = await run_pipeline(build(), {"p": 0.6}, t.test_ports(), patch=ManifestPatch(thresholds={"apply": {"keep": 0.9}}))
    assert tuned.output == Out(kept=False)
    assert tuned.patch_applied == ["apply.thresholds.keep=0.9"]
    assert tuned.version != base.version


def test_refuses_a_patch_naming_a_threshold_the_transform_never_declared():
    problems = validate_patch(build(), ManifestPatch(thresholds={"apply": {"nope": 1}}))
    assert '"apply.nope" is not a threshold the node declares' in "\n".join(problems)


def test_still_refuses_a_thresholds_patch_on_a_transform_that_declares_none():
    problems = validate_patch(build(), ManifestPatch(thresholds={"sink": {"x": 1}}))
    assert "not a code node that declares thresholds" in "\n".join(problems)


def test_exports_the_thresholds_in_to_json():
    apply = next(n for n in build().to_json()["nodes"] if n["id"] == "apply")
    assert apply["thresholds"] == {"keep": {"env": "TT_KEEP_P", "default": 0.5}}
