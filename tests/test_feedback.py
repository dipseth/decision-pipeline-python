"""Port of src/feedback.test.ts."""

from __future__ import annotations

import dataclasses

import pytest

from decision_pipeline import (
    PIPELINE_HAND_READ_QUEUE,
    PipelinePorts,
    QueueResult,
    ScoreCategory,
    ScoreConfigSpec,
    UnknownFeedbackEdgeError,
    category_value,
    define_review_lane,
    enqueue_for_hand_read,
    feedback_edge_from_dict,
    submit_feedback,
)
from decision_pipeline import testing as t
from decision_pipeline.fixtures import SCALE_LINE_LANE, build_scale_pipeline

scale = build_scale_pipeline()


def with_edges(edges):
    return dataclasses.replace(scale, feedback=[feedback_edge_from_dict(e) for e in edges])


def warnings_of(result) -> str:
    return ",".join(result.warnings)


# --- a `score` edge -----------------------------------------------------------


async def test_writes_the_langfuse_score_on_the_runs_trace_and_returns_a_label_row():
    ports = t.test_ports()
    result = await submit_feedback(
        scale,
        {"run_id": "trace-abc", "edge_id": "rating", "value": 0.5, "comment": "okay · ×2 · direct", "source_id": "u-1"},
        ports,
    )

    assert len(ports.scores_written) == 1
    score = ports.scores_written[0]
    assert score.run_id == "trace-abc"
    assert score.name == "scale_feedback"
    assert score.value == 0.5
    assert score.data_type == "NUMERIC"
    assert score.comment == "okay · ×2 · direct"
    assert result.scored is True
    label = result.label
    assert label.run_id == "trace-abc"
    assert label.kind == "score"
    assert label.value == 0.5
    assert label.score_name == "scale_feedback"
    assert label.edge_id == "rating"
    assert label.scope == "user"
    assert label.rank == "human"


async def test_keeps_a_categorical_label_beside_the_number():
    edge = {
        "id": "read", "from": "surface:hand-read", "to": "gate", "source": "explicit",
        "scope": "user", "form": "score", "latency": "deferred",
        "score": {
            "name": "pipeline_read", "dataType": "CATEGORICAL",
            "categories": [{"label": "p5", "value": 0}, {"label": "meh", "value": 1}, {"label": "ok", "value": 2}],
        },
    }
    ports = t.test_ports()
    result = await submit_feedback(with_edges([edge]), {"run_id": "t1", "edge_id": "read", "label": "meh", "source_id": "reviewer-1"}, ports)
    assert result.label.value == 1
    assert result.label.label == "meh"
    assert ports.scores_written[0].label == "meh"


async def test_warns_on_a_label_the_config_does_not_have_rather_than_inventing_a_value():
    edge = {
        "id": "read", "from": "s", "to": "gate", "source": "explicit", "scope": "user",
        "form": "score", "latency": "deferred",
        "score": {"name": "c", "dataType": "CATEGORICAL", "categories": [{"label": "ok", "value": 1}]},
    }
    result = await submit_feedback(with_edges([edge]), {"run_id": "t1", "edge_id": "read", "label": "nonsense", "source_id": "s"}, t.test_ports())
    assert "not a category" in warnings_of(result)


async def test_still_returns_the_durable_label_when_langfuse_is_down():
    def down(_request):
        raise Exception("langfuse 503")

    ports = dataclasses.replace(t.test_ports(), scores=down)
    result = await submit_feedback(scale, {"run_id": "t1", "edge_id": "rating", "value": 1, "source_id": "u"}, ports)
    assert result.scored is False
    assert "langfuse 503" in warnings_of(result)
    assert result.label.value == 1


# --- a `queue` edge -----------------------------------------------------------


async def test_enqueues_the_runs_trace_into_the_lanes_physical_queue():
    ports = t.test_ports()
    result = await submit_feedback(
        scale,
        {"run_id": "trace-xyz", "edge_id": "flagged-lines", "source_id": "u-1", "metadata": {"line": 3}},
        ports,
    )

    assert len(ports.queue_items) == 1
    item = ports.queue_items[0]
    # One shared queue; which pipeline and edge sent it rides on the item.
    assert item.queue == "pipeline-review"
    assert item.lane == SCALE_LINE_LANE
    assert item.pipeline == "scale"
    assert item.edge_id == "flagged-lines"
    assert item.run_id == "trace-xyz"
    assert {k: item.metadata[k] for k in ("fact", "node", "line")} == {"fact": "scaled_recipe", "node": "classify", "line": 3}
    # The lane's own score config is what a reviewer grades with.
    assert item.score_config.name == "scale_line_verdict"
    assert result.queued == QueueResult(queue_id="q-pipeline-review", item_id="item-1")


async def test_warns_instead_of_silently_enqueueing_nowhere_when_the_lane_is_unknown():
    edge = {
        "id": "q", "from": "s", "to": "gate", "source": "explicit", "scope": "tenant",
        "form": "queue", "latency": "deferred", "promoteVia": "never-registered",
    }
    result = await submit_feedback(with_edges([edge]), {"run_id": "t1", "edge_id": "q", "source_id": "s"}, t.test_ports())
    assert result.queued is None
    assert "not registered" in warnings_of(result)


# --- an `override` edge -------------------------------------------------------

OVERRIDE_EDGE = {
    "id": "drag", "from": "surface:list", "to": "classify", "source": "implicit",
    "scope": "device", "form": "override", "latency": "immediate",
}


async def test_returns_the_row_for_the_host_to_persist_the_core_owns_no_database():
    result = await submit_feedback(
        with_edges([OVERRIDE_EDGE]),
        {"run_id": "t1", "edge_id": "drag", "key": "cilantro", "payload": "produce", "source_id": "dev-9", "arm": "b"},
        t.test_ports(),
    )
    row = result.row
    assert (row.key, row.scope, row.rank, row.value, row.arm) == ("cilantro", "device", "human", "produce", "b")


async def test_labels_the_disagreement_only_when_the_human_actually_differed():
    disagreed = await submit_feedback(
        with_edges([OVERRIDE_EDGE]),
        {"run_id": "t1", "edge_id": "drag", "key": "c", "payload": "produce", "model_value": "spices", "source_id": "d"},
        t.test_ports(),
    )
    assert (disagreed.label.kind, disagreed.label.value) == ("override_disagreement", 1)

    agreed = await submit_feedback(
        with_edges([OVERRIDE_EDGE]),
        {"run_id": "t1", "edge_id": "drag", "key": "c", "payload": "produce", "model_value": "produce", "source_id": "d"},
        t.test_ports(),
    )
    assert agreed.label.value == 0


async def test_refuses_a_row_at_a_scope_the_form_x_scope_rule_forbids():
    wide = {**OVERRIDE_EDGE, "scope": "tenant"}
    result = await submit_feedback(
        with_edges([wide]),
        {"run_id": "t1", "edge_id": "drag", "key": "c", "payload": "produce", "source_id": "d"},
        t.test_ports(),
    )
    assert result.row is None
    assert "not allowed at scope" in warnings_of(result)


async def test_needs_a_key():
    result = await submit_feedback(
        with_edges([OVERRIDE_EDGE]), {"run_id": "t1", "edge_id": "drag", "payload": "x", "source_id": "d"}, t.test_ports()
    )
    assert result.row is None
    assert "needs a `key`" in warnings_of(result)


# --- a `feature` edge ---------------------------------------------------------


async def test_returns_a_row_addressed_to_the_node_the_edge_lands_on():
    edge = {
        "id": "habit", "from": "surface:list", "to": "classify", "source": "implicit",
        "scope": "user", "form": "feature", "latency": "next-run",
    }
    result = await submit_feedback(
        with_edges([edge]),
        {"run_id": "t1", "edge_id": "habit", "payload": "always halves salt", "source_id": "u-1"},
        t.test_ports(),
    )
    row = result.row
    assert (row.node_id, row.scope, row.value) == ("classify", "user", "always halves salt")


# --- misc ---------------------------------------------------------------------


async def test_throws_on_an_edge_the_manifest_never_declared():
    with pytest.raises(UnknownFeedbackEdgeError):
        await submit_feedback(scale, {"run_id": "t1", "edge_id": "ghost", "source_id": "s"}, t.test_ports())


async def test_warns_when_no_scores_port_is_wired_rather_than_pretending_it_landed():
    result = await submit_feedback(scale, {"run_id": "t1", "edge_id": "rating", "value": 1, "source_id": "u"}, PipelinePorts())
    assert result.scored is False
    assert "no `scores` port" in warnings_of(result)


def test_category_value_resolves_a_label_through_its_config():
    config = ScoreConfigSpec(name="c", data_type="CATEGORICAL", categories=[ScoreCategory(label="ok", value=2)])
    assert category_value(config, "ok") == 2
    assert category_value(config, "nope") is None
    assert category_value(None, "ok") is None


# --- enqueue_for_hand_read ----------------------------------------------------


async def test_uses_the_second_shared_queue_and_is_not_a_user_promotion():
    lane = define_review_lane(lane="scale-hand-read", queue=PIPELINE_HAND_READ_QUEUE)
    ports = t.test_ports()
    queued = await enqueue_for_hand_read(scale, run_id="t1", reason="gate margin 0.02", lane=lane, ports=ports)
    assert queued.queue_id == "q-pipeline-hand-read"
    assert ports.queue_items[0].edge_id == "hand-read"
    assert ports.queue_items[0].metadata["reason"] == "gate margin 0.02"


async def test_is_a_no_op_without_a_queue_port_or_a_registered_lane():
    assert await enqueue_for_hand_read(scale, run_id="t1", reason="r", lane="ghost", ports=t.test_ports()) is None
    assert await enqueue_for_hand_read(scale, run_id="t1", reason="r", lane="scale-hand-read", ports=PipelinePorts()) is None
