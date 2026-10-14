import math

from decision_pipeline.features import (
    DistributionFeatureInput,
    distribution_features,
    normalized_entropy,
    top_margin,
)


def test_is_0_for_a_certainty_and_1_for_a_uniform_distribution():
    assert normalized_entropy([1]) == 0
    assert math.isclose(normalized_entropy([0.5, 0.5]), 1, abs_tol=1e-10)
    assert math.isclose(normalized_entropy([0.25, 0.25, 0.25, 0.25]), 1, abs_tol=1e-10)


def test_sits_between_the_two_for_a_lopsided_one():
    h = normalized_entropy([0.9, 0.1])
    assert h > 0
    assert h < 1


def test_is_the_gap_to_the_runner_up_how_nearly_the_gate_went_the_other_way():
    assert math.isclose(top_margin([0.9, 0.1]), 0.8, abs_tol=1e-10)
    assert math.isclose(top_margin([0.51, 0.49]), 0.02, abs_tol=1e-10)
    assert math.isclose(top_margin([0.7]), 0.7, abs_tol=1e-10)
    assert top_margin([]) is None


def test_flattens_to_node_question_stat_keys():
    f = distribution_features(
        [DistributionFeatureInput(node_id="classify", distributions={"line:0": [0.92, 0.08]})]
    )
    assert sorted(f.keys()) == [
        "classify.line:0.entropy",
        "classify.line:0.margin",
        "classify.line:0.top",
    ]
    assert math.isclose(f["classify.line:0.top"], 0.92, abs_tol=1e-10)
