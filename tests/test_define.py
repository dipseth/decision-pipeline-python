"""Port of src/define.test.ts."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from decision_pipeline import PipelineContractError, define_pipeline, validate_pipeline
from decision_pipeline.fixtures import SCALE_LINE_LANE, build_scale_pipeline


def noop(_args):
    return {"ok": True}


class DemoIn(BaseModel):
    id: str


class DemoOut(BaseModel):
    ok: bool


def base() -> dict:
    """The smallest manifest that satisfies the contract. Each test breaks ONE clause."""
    return {
        "id": "demo",
        "fact": "demo_fact",
        "input": DemoIn,
        "output": DemoOut,
        "trigger": ["on-demand"],
        "group": lambda i: f"demo-{i.id}",
        "result": "yes",
        "nodes": {
            "gate": {
                "kind": "code",
                "role": "gate",
                "run": lambda _a: "yes",
                "branches": ["yes", "no"],
                "thresholds": {"t": 0.5},
                "version": "1",
            },
            "yes": {"kind": "code", "role": "derive", "run": noop, "version": "1"},
            "no": {"kind": "code", "role": "derive", "run": noop, "version": "1"},
        },
        "edges": [
            {"from": "gate", "to": "yes", "when": {"gate": "gate", "branch": "yes"}},
            {"from": "gate", "to": "no", "when": {"gate": "gate", "branch": "no"}},
        ],
        "feedback": [
            {
                "id": "rating",
                "from": "surface:demo",
                "to": "gate",
                "source": "explicit",
                "scope": "user",
                "form": "score",
                "latency": "deferred",
                "score": {"name": "demo_score", "dataType": "NUMERIC"},
            }
        ],
        "eval": {"dataset": "demo-eval", "gateMetric": "score_positive"},
    }


def problems_for(mutate) -> list[str]:
    spec = base()
    mutate(spec)
    return validate_pipeline(spec)


def joined(problems: list[str]) -> str:
    return "\n".join(problems)


# --- a manifest that satisfies the contract -----------------------------------


def test_loads_and_the_docs_worked_example_loads_too():
    assert validate_pipeline(base()) == []
    build_scale_pipeline()


def test_exposes_a_topological_order_and_a_plain_data_export():
    scale = build_scale_pipeline()
    assert scale.order.index("classify") < scale.order.index("gate")

    data = scale.to_json()
    assert data["id"] == "scale"
    assert data["fact"] == "scaled_recipe"
    assert "writer" in [n["id"] for n in data["nodes"]]
    # YAML/JSON is an EXPORT: no function survives into it (json.dumps would raise on one).
    assert "function" not in json.dumps(data)
    assert all(isinstance(f["key"], bool) for f in data["feedback"])


def test_defaults_decided_by_to_llm_and_normalises_result_to_a_list():
    p = define_pipeline(base())
    assert p.decided_by == "llm"
    assert p.result == ["yes"]


def test_throws_a_pipeline_contract_error_listing_every_problem_not_just_the_first():
    spec = base()
    spec["feedback"] = []
    spec["nodes"] = {"only": {"kind": "code", "role": "derive", "run": noop, "version": "1"}}
    spec["edges"] = []
    spec["result"] = "only"
    with pytest.raises(PipelineContractError) as info:
        define_pipeline(spec)
    problems = info.value.problems
    assert len(problems) > 1
    assert "no gate node" in joined(problems)
    assert "no feedback edge" in joined(problems)


# --- clause 2 — at least one code gate ----------------------------------------


def test_rejects_a_chain():
    def mutate(spec):
        spec["nodes"] = {"yes": {"kind": "code", "role": "derive", "run": noop, "version": "1"}}
        spec["edges"] = []

    assert "not a decision pipeline" in joined(problems_for(mutate))


def test_refuses_a_guard_on_a_node_that_is_not_a_gate():
    def mutate(spec):
        spec["nodes"]["reader"] = {"kind": "read", "load": noop, "version": "1"}
        spec["edges"].append({"from": "reader", "to": "yes", "when": {"gate": "reader", "branch": "yes"}})

    assert "only a code gate may branch" in joined(problems_for(mutate))


def test_refuses_a_branch_the_gate_never_declared():
    def mutate(spec):
        spec["edges"][0] = {"from": "gate", "to": "yes", "when": {"gate": "gate", "branch": "maybe"}}

    assert 'has no branch "maybe"' in joined(problems_for(mutate))


# --- clause 3 — every branch has somewhere to go ------------------------------


def test_rejects_a_declared_branch_with_no_outbound_edge():
    def mutate(spec):
        spec["edges"] = [{"from": "gate", "to": "yes", "when": {"gate": "gate", "branch": "yes"}}]

    assert 'branch "no" with no outbound edge' in joined(problems_for(mutate))


# --- clause 4 — a feedback edge, landing on a real node -----------------------


def test_rejects_a_pipeline_with_none():
    assert "no feedback edge" in joined(problems_for(lambda s: s.update(feedback=[])))


def test_rejects_an_edge_landing_nowhere():
    def mutate(s):
        s["feedback"][0]["to"] = "ghost"

    assert 'lands on unknown node "ghost"' in joined(problems_for(mutate))


def test_rejects_duplicate_feedback_ids():
    def mutate(s):
        s["feedback"].append({**s["feedback"][0], "to": "gate"})

    assert "duplicate feedback edge id" in joined(problems_for(mutate))


# --- the form x scope rule ----------------------------------------------------


@pytest.mark.parametrize("scope", ["tenant", "cohort", "global"])
def test_refuses_an_override_at(scope):
    def mutate(s):
        s["feedback"][0] = {**s["feedback"][0], "form": "override", "scope": scope}

    assert "only queue / derived / score" in joined(problems_for(mutate))


@pytest.mark.parametrize("scope", ["session", "device", "user", "associated"])
def test_allows_a_feature_at(scope):
    def mutate(s):
        s["feedback"][0] = {**s["feedback"][0], "form": "feature", "scope": scope}

    assert problems_for(mutate) == []


def test_requires_a_review_queue_for_derived_a_structural_change_is_never_silent_drift():
    def mutate(s):
        s["feedback"][0] = {**s["feedback"][0], "form": "derived", "scope": "tenant"}

    assert "promoteVia" in joined(problems_for(mutate))


def test_requires_a_queue_name_for_queue():
    def mutate(s):
        s["feedback"][0] = {**s["feedback"][0], "form": "queue"}

    assert "promoteVia" in joined(problems_for(mutate))


# --- clause 6 — separability --------------------------------------------------


def test_refuses_overridable_with_nothing_persisting_the_pipelines_own_answer():
    def mutate(s):
        s["overridable"] = {"scope": "device", "key": lambda _o: "k"}

    assert "not separable" in joined(problems_for(mutate))


def test_accepts_it_once_a_store_node_exists():
    def mutate(s):
        s["overridable"] = {"scope": "device", "key": lambda _o: "k"}
        s["nodes"]["persist"] = {"kind": "store", "target": "qdrant", "scope": "tenant", "version": "1"}
        s["edges"].append({"from": "yes", "to": "persist"})

    assert problems_for(mutate) == []


# --- structure ----------------------------------------------------------------


def test_rejects_a_cycle():
    assert "cycle" in joined(problems_for(lambda s: s["edges"].append({"from": "yes", "to": "gate"})))


def test_rejects_an_edge_to_a_node_that_does_not_exist():
    assert 'unknown node "ghost"' in joined(problems_for(lambda s: s["edges"].append({"from": "yes", "to": "ghost"})))


def test_refuses_a_result_that_names_the_store_node_rather_than_the_producer():
    def mutate(s):
        s["nodes"]["persist"] = {"kind": "store", "target": "qdrant", "scope": "tenant", "version": "1"}
        s["edges"].append({"from": "yes", "to": "persist"})
        s["result"] = "persist"

    assert "not the one that writes it" in joined(problems_for(mutate))


def test_refuses_on_failure_fallback_with_no_fallback_body_and_a_cached_store_node():
    def mutate(s):
        s["nodes"]["yes"] = {"kind": "code", "role": "derive", "run": noop, "version": "1", "onFailure": "fallback"}
        s["nodes"]["persist"] = {"kind": "store", "target": "qdrant", "scope": "tenant", "version": "1", "cache": "per-key-forever"}
        s["edges"].append({"from": "yes", "to": "persist"})

    problems = joined(problems_for(mutate))
    assert "needs a `fallback` body" in problems
    assert "a cached write is a missing write" in problems


# --- Langfuse backing ---------------------------------------------------------


def test_refuses_a_score_edge_that_names_no_langfuse_score():
    def mutate(s):
        del s["feedback"][0]["score"]

    assert "no Langfuse score to land on" in joined(problems_for(mutate))


def test_refuses_a_categorical_config_with_no_categories_and_categories_on_a_numeric_one():
    def categorical(s):
        s["feedback"][0]["score"] = {"name": "c", "dataType": "CATEGORICAL"}

    assert "declares no categories" in joined(problems_for(categorical))

    def numeric(s):
        s["feedback"][0]["score"] = {"name": "c", "dataType": "NUMERIC", "categories": [{"label": "ok", "value": 1}]}

    assert "declares categories but is NUMERIC" in joined(problems_for(numeric))


def test_refuses_an_unregistered_review_lane_queues_are_capped_so_a_queue_per_pipeline_cannot_work():
    def mutate(s):
        s["feedback"].append(
            {
                "id": "promote", "from": "surface:x", "to": "gate", "source": "explicit",
                "scope": "tenant", "form": "queue", "latency": "deferred",
                "promoteVia": "a-queue-nobody-registered",
            }
        )

    problems = joined(problems_for(mutate))
    assert "is not registered" in problems
    # TS names `defineReviewLane`; the port names the Python function.
    assert "define_review_lane" in problems


def test_accepts_a_registered_lane():
    def mutate(s):
        s["feedback"].append(
            {
                "id": "promote", "from": "surface:x", "to": "gate", "source": "explicit",
                "scope": "tenant", "form": "queue", "latency": "deferred",
                "promoteVia": SCALE_LINE_LANE,
            }
        )

    assert problems_for(mutate) == []


# --- eval binding -------------------------------------------------------------


def test_refuses_a_gate_metric_that_is_not_a_registered_extractor():
    def mutate(s):
        s["eval"] = {"dataset": "d", "gateMetric": "whatever_i_felt_like"}

    assert "not a registered gate metric" in joined(problems_for(mutate))


def test_refuses_a_manifest_with_no_eval_at_all():
    assert len(problems_for(lambda s: s.update(eval=None))) > 0


# --- tools --------------------------------------------------------------------


def with_writer(spec):
    spec["nodes"]["classify"] = {"kind": "decide", "questions": "jev-demo", "version": "1"}
    spec["nodes"]["writer"] = {
        "kind": "generate",
        "prompt": "p",
        "route": "r",
        "version": "1",
        "tools": {
            "static": ["a"],
            "selectable": {"candidates": ["b"], "from": "classify", "gate": "gate"},
            "allow": ["a", "b"],
        },
    }
    spec["edges"].insert(0, {"from": "classify", "to": "gate"})
    spec["edges"].append({"from": "yes", "to": "writer"})


def test_accepts_a_well_formed_selectable_spec():
    assert problems_for(with_writer) == []


def test_refuses_a_static_tool_outside_the_allow_ceiling():
    def mutate(s):
        with_writer(s)
        s["nodes"]["writer"]["tools"]["static"] = ["z"]

    assert 'static tool "z" is outside `allow`' in joined(problems_for(mutate))


def test_refuses_selection_from_a_node_that_is_not_a_decide_node():
    def mutate(s):
        with_writer(s)
        s["nodes"]["writer"]["tools"]["selectable"]["from"] = "yes"

    assert "must be a decide node" in joined(problems_for(mutate))


def test_refuses_selection_from_a_decide_node_that_is_not_upstream():
    def mutate(s):
        with_writer(s)
        s["nodes"]["late"] = {"kind": "decide", "questions": "jev-late", "version": "1"}
        s["edges"].append({"from": "writer", "to": "late"})
        s["nodes"]["writer"]["tools"]["selectable"]["from"] = "late"

    assert "is not upstream" in joined(problems_for(mutate))
