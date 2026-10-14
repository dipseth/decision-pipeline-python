import pytest

from decision_pipeline.record import (
    RunLabel,
    RunRecord,
    RunRecordNode,
    aggregate_gate_metric,
    define_gate_metric,
    gate_metric_names,
    get_gate_metric,
)


def record(**over) -> RunRecord:
    base = dict(
        pipeline="scale",
        version="v1",
        run_id="r1",
        run_key="r1",
        group="cook-1",
        tenant=None,
        user=None,
        trigger="on-demand",
        input_hash="h",
        cost_usd=0,
        ms=1,
        nodes=[],
        features={},
        fact="scaled_recipe",
        fact_route="direct",
        fact_confidence=0.9,
        consumed_feedback_ids=[],
        arms={},
        parent_run_id=None,
        assignment_unit=None,
        assignment_hash=None,
        created_at="2026-09-22T00:00:00.000Z",
    )
    base.update(over)
    return RunRecord(**base)


def label(**over) -> RunLabel:
    base = dict(
        run_id="r1",
        kind="score",
        value=1,
        scope="user",
        rank="human",
        source_id="s1",
        created_at="2026-09-22T00:00:00.000Z",
    )
    base.update(over)
    return RunLabel(**base)


EXTRA = "test_only_metric"


def test_ships_two_metrics_so_eval_gate_metric_is_never_a_stub():
    assert "score_positive" in gate_metric_names()
    assert "no_override_disagreement" in gate_metric_names()


def test_refuses_to_redefine_a_name_with_a_different_body():
    def fn(_r, _l):
        return 1

    define_gate_metric(EXTRA, fn)
    define_gate_metric(EXTRA, fn)  # same body: fine
    with pytest.raises(Exception, match="already defined"):
        define_gate_metric(EXTRA, lambda _r, _l: 0)


def test_returns_null_when_a_run_cannot_contribute_never_a_silent_zero():
    metric = get_gate_metric("score_positive")
    assert metric(record(), []) is None
    assert metric(record(), [label(value=1)]) == 1
    assert metric(record(), [label(value=-1)]) == 0


def test_drops_nulls_from_the_aggregate_rather_than_counting_them_as_zero():
    result = aggregate_gate_metric(
        "score_positive",
        [
            {"record": record(), "labels": [label(value=1)]},
            {"record": record(), "labels": [label(value=-1)]},
            {"record": record(), "labels": []},
        ],
    )
    assert result == {"name": "score_positive", "n": 2, "mean": 0.5}


def test_reports_n_0_and_a_null_mean_when_nothing_is_labelled_yet():
    assert aggregate_gate_metric("score_positive", [{"record": record(), "labels": []}]) == {
        "name": "score_positive",
        "n": 0,
        "mean": None,
    }


def test_throws_on_an_unknown_metric_instead_of_quietly_returning_nothing():
    with pytest.raises(Exception, match="Unknown gate metric"):
        aggregate_gate_metric("nope", [])


def test_scores_the_disagreement_set():
    metric = get_gate_metric("no_override_disagreement")
    assert metric(record(), [label(kind="override_disagreement", value=1)]) == 0
    assert metric(record(), [label(kind="override_disagreement", value=0)]) == 1


def test_to_dict_omits_none_optional_node_fields():
    node = RunRecordNode(id="n", kind="code", version="1", input_hash="h", ms=1, cost_usd=0, branch="yes")
    assert node.to_dict() == {"id": "n", "kind": "code", "version": "1", "input_hash": "h", "ms": 1, "cost_usd": 0, "branch": "yes"}
    d = record(nodes=[node]).to_dict()
    assert d["nodes"] == [node.to_dict()]
    assert d["tenant"] is None  # record-level nullable fields stay, like TS
    assert RunRecord.from_dict(d) == record(nodes=[node])
