import pytest

from decision_pipeline.review import (
    PIPELINE_HAND_READ_QUEUE,
    PIPELINE_REVIEW_QUEUE,
    REVIEW_QUEUE_BUDGET,
    ReviewBudgetError,
    ReviewRegistrySnapshot,
    define_review_lane,
    define_review_queue,
    has_lane,
    lane_names,
    resolve_lane,
    restore_review_registry,
    review_queues,
    snapshot_review_registry,
)

@pytest.fixture(autouse=True)
def _restore():
    # vitest gives each test file fresh module state; pytest shares one process, so
    # lanes other test modules registered (fixtures.py's scale lane) must not leak in.
    saved = snapshot_review_registry()
    shared = [q for q in saved.queues if q.name in (PIPELINE_REVIEW_QUEUE, PIPELINE_HAND_READ_QUEUE)]
    restore_review_registry(ReviewRegistrySnapshot(queues=shared, lanes=[]))
    yield
    restore_review_registry(saved)


# --- the queue budget ---


def test_ships_two_shared_queues_and_leaves_one_for_the_host():
    assert [q.name for q in review_queues()] == [PIPELINE_REVIEW_QUEUE, PIPELINE_HAND_READ_QUEUE]
    assert REVIEW_QUEUE_BUDGET - len(review_queues()) == 1


def test_allows_the_host_to_spend_the_third():
    define_review_queue(name="third", description="the host's")


def test_refuses_a_fourth_naming_what_the_budget_is_spent_on():
    define_review_queue(name="third", description="the host's")
    # Today this fails at RUNTIME with a console.warn and nothing enqueued, in
    # three separate recipes-core modules. A load error is the point.
    with pytest.raises(ReviewBudgetError) as exc:
        define_review_queue(name="fourth", description="one too many")
    assert "pipeline-review" in str(exc.value)
    assert "Add a LANE" in str(exc.value)


def test_registering_the_same_queue_twice_is_a_no_op_not_a_budget_spend():
    define_review_queue(name=PIPELINE_REVIEW_QUEUE, description="again")
    assert len(review_queues()) == 2


# --- lanes ---


def test_lanes_are_unlimited_many_lanes_ride_on_one_queue():
    for i in range(20):
        define_review_lane(lane=f"lane-{i}", queue=PIPELINE_REVIEW_QUEUE)
    assert len(lane_names()) == 20
    assert len(review_queues()) == 2


def test_lanes_carry_their_own_score_config_which_langfuse_does_not_cap():
    define_review_lane(
        lane="aisle-drag",
        queue=PIPELINE_REVIEW_QUEUE,
        score_config={
            "name": "aisle_verdict",
            "dataType": "CATEGORICAL",
            "categories": [{"label": "wrong", "value": 0}, {"label": "right", "value": 1}],
        },
    )
    assert resolve_lane("aisle-drag").score_config.name == "aisle_verdict"
    assert resolve_lane("aisle-drag").queue == PIPELINE_REVIEW_QUEUE


def test_lanes_refuse_an_unregistered_queue():
    with pytest.raises(Exception, match="unregistered queue"):
        define_review_lane(lane="x", queue="no-such-queue")


def test_lanes_refuse_to_be_moved_to_a_different_queue_once_registered():
    define_review_lane(lane="x", queue=PIPELINE_REVIEW_QUEUE)
    with pytest.raises(Exception, match="refusing to move it"):
        define_review_lane(lane="x", queue=PIPELINE_HAND_READ_QUEUE)
    define_review_lane(lane="x", queue=PIPELINE_REVIEW_QUEUE)


def test_lanes_report_whether_a_lane_exists_which_is_what_the_manifest_validates():
    assert has_lane("nope") is False
    define_review_lane(lane="nope", queue=PIPELINE_REVIEW_QUEUE)
    assert has_lane("nope") is True
