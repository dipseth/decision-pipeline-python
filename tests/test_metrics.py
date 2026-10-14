import json
import math

import pytest

from decision_pipeline._compat import as_schema
from decision_pipeline.metrics import BUILTIN_METRICS, Observation, ObservationRow, metric_catalog

row = ObservationRow(id="r", prediction=None)


def obs(prediction, truth=None):
    return Observation(prediction=prediction, truth=truth, row=row)


def score(ref, prediction, truth=None, args=None):
    m = BUILTIN_METRICS[ref]
    if m.scope != "row":
        raise ValueError(f"{ref} is not a row metric")
    return m.score(obs(prediction, truth), as_schema(m.args).parse(args or {}))


def compute(ref, pairs, args=None):
    m = BUILTIN_METRICS[ref]
    if m.scope != "sample":
        raise ValueError(f"{ref} is not a sample metric")
    return m.compute([obs(p, t) for p, t in pairs], as_schema(m.args).parse(args or {}))


def close(a, b, digits=2):
    return abs(a - b) < 10 ** (-digits) / 2


# describe("classification")
def test_exact_match_trims_optionally_ignores_case_compares_objects_structurally():
    assert score("exact_match@1", " french ", "french") == 1
    assert score("exact_match@1", "French", "french") == 0
    assert score("exact_match@1", "French", "french", {"caseInsensitive": True}) == 1
    assert score("exact_match@1", {"a": 1, "b": 2}, {"b": 2, "a": 1}) == 1
    assert score("exact_match@1", "x", None) is None


def test_top_k_reads_a_ranked_list_or_a_probability_map():
    assert score("top_k@1", ["a", "b", "c"], "b", {"k": 2}) == 1
    assert score("top_k@1", {"a": 0.1, "b": 0.2, "c": 0.7}, "a", {"k": 2}) == 0
    assert score("top_k@1", {"a": 0.1, "b": 0.2, "c": 0.7}, "b", {"k": 2}) == 1


# describe("probabilistic")
def test_log_loss_and_brier_binary_and_multiclass():
    assert close(score("log_loss@1", 0.8, True), -math.log(0.8), 10)
    assert close(score("log_loss@1", 0.8, 0), -math.log(0.2), 10)
    assert close(score("log_loss@1", {"a": 0.5, "b": 0.5}, "c"), -math.log(1e-15), 5)
    assert close(score("brier@1", 0.8, 1), 0.04, 10)
    # (0.7−1)² + (0.3−0)²
    assert close(score("brier@1", {"a": 0.7, "b": 0.3}, "a"), 0.18, 10)
    # truth absent from the map: 1 for the missing label + 0.7² + 0.3²
    assert close(score("brier@1", {"a": 0.7, "b": 0.3}, "c"), 1.58, 10)
    assert score("prob_of_truth@1", [{"label": "a", "p": 0.6}], "a") == 0.6


def test_crps_gaussian_reduces_to_absolute_error_as_sd_0_and_rewards_honest_spread():
    assert score("crps_gaussian@1", {"mean": 2, "sd": 0}, 5) == 3
    tight = score("crps_gaussian@1", {"mean": 0, "sd": 0.1}, 3)
    honest = score("crps_gaussian@1", {"mean": 0, "sd": 2}, 3)
    assert honest < tight


def test_auc_perfect_separation_1_pure_ties_0_5_one_class_undefined():
    assert compute("auc@1", [(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]) == 1
    assert compute("auc@1", [(0.5, 1), (0.5, 0)]) == 0.5
    assert compute("auc@1", [(0.5, 1), (0.7, 1)]) is None


def test_ece_is_0_for_a_calibrated_bin_and_gap_for_a_miscalibrated_one():
    calibrated = [(0.75, 1), (0.75, 1), (0.75, 1), (0.75, 0)]
    assert close(compute("ece@1", calibrated), 0, 10)
    overconfident = [(0.95, 1), (0.95, 0)]
    assert close(compute("ece@1", overconfident), 0.45, 10)


# describe("ranking")
def test_reciprocal_rank_hit_precision_recall_at_k_average_precision():
    assert close(score("reciprocal_rank@1", ["x", "y", "z"], "z"), 1 / 3, 10)
    assert score("reciprocal_rank@1", ["x", "y", "z"], "z", {"k": 2}) == 0
    assert score("hit_at_k@1", ["x", "y"], ["q", "y"], {"k": 2}) == 1
    assert score("precision_at_k@1", ["x", "y", "z"], ["x", "z"], {"k": 2}) == 0.5
    assert score("recall_at_k@1", ["x", "y", "z"], ["x", "z"], {"k": 2}) == 0.5
    # relevant at 1 and 3: (1/1 + 2/3) / 2
    assert close(score("average_precision@1", ["x", "y", "z"], ["x", "z"]), (1 + 2 / 3) / 2, 10)


def test_ndcg_is_1_for_the_ideal_order_and_uses_graded_gains():
    assert close(score("ndcg_at_k@1", ["a", "b"], {"a": 3, "b": 1}, {"k": 2}), 1, 10)
    assert score("ndcg_at_k@1", ["b", "a"], {"a": 3, "b": 1}, {"k": 2}) < 1
    assert score("ndcg_at_k@1", ["c"], {"a": 0}, {"k": 2}) is None


def test_kendall_tau_identical_1_reversed_minus_1_only_shared_items_count():
    assert score("kendall_tau@1", ["a", "b", "c"], ["a", "b", "c"]) == 1
    assert score("kendall_tau@1", ["c", "b", "a"], ["a", "b", "c"]) == -1
    assert score("kendall_tau@1", ["a", "x", "b"], ["a", "b"]) == 1


# describe("sets")
def test_jaccard_set_f1_set_exact():
    assert close(score("jaccard@1", ["a", "b"], ["b", "c"]), 1 / 3, 10)
    assert score("set_f1@1", ["a", "b"], ["b", "c"]) == 0.5
    assert score("set_exact@1", ["b", "a"], ["a", "b"]) == 1
    assert score("jaccard@1", [], []) == 1


# describe("regression and intervals")
def test_errors():
    assert score("abs_error@1", 3, 5) == 2
    assert score("squared_error@1", 3, 5) == 4
    assert score("signed_error@1", 3, 5) == -2
    assert score("abs_pct_error@1", 3, 0) is None
    assert close(score("log_ratio_error@1", 2, 1), math.log(2), 10)
    assert close(score("log_ratio_error@1", 1, 2), math.log(2), 10)
    assert score("within_tolerance@1", 10.4, 10, {"rel": 0.05}) == 1
    assert score("within_tolerance@1", 10.6, 10, {"rel": 0.05}) == 0
    with pytest.raises(Exception):
        as_schema(BUILTIN_METRICS["within_tolerance@1"].args).parse({})


def test_interval_coverage_width_score_pinball():
    assert score("interval_covers@1", {"lo": 1, "hi": 3}, 2) == 1
    assert score("interval_covers@1", [1, 3], 4) == 0
    assert score("interval_width@1", {"low": 1, "high": 4}, None, {"lo": "low", "hi": "high"}) == 3
    # width 2 + (2/0.1)·(4 − 3)
    assert close(score("interval_score@1", [1, 3], 4, {"alpha": 0.1}), 22, 10)
    assert close(score("pinball@1", 5, 7, {"q": 0.9}), 1.8, 10)
    assert close(score("pinball@1", 5, 3, {"q": 0.9}), 0.2, 10)


def test_sample_correlations():
    mono = [(1, 1), (2, 4), (3, 9), (4, 16)]
    assert close(compute("spearman@1", mono), 1, 10)
    assert compute("pearson@1", mono) < 1
    assert compute("r2@1", [(1, 1), (2, 2), (3, 3)]) == 1


# describe("agreement and generic")
def test_cohen_kappa_and_macro_f1():
    assert compute("cohen_kappa@1", [("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]) == 1
    assert compute("cohen_kappa@1", [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]) == 0
    assert compute("macro_f1@1", [("a", "a"), ("b", "b")]) == 1


def test_value_truthy_equals_read_the_prediction_alone():
    assert score("value@1", 0.0031) == 0.0031
    assert score("value@1", "n/a") is None
    assert score("truthy@1", "yes") == 1
    assert score("equals@1", "tail", None, {"value": "tail"}) == 1
    assert score("equals@1", "leaf", None, {"in": ["tail", "probe"]}) == 0


# describe("catalog")
def test_lists_every_metric_with_its_args_as_json_schema():
    catalog = metric_catalog(BUILTIN_METRICS)
    assert len(catalog) == len(BUILTIN_METRICS)
    assert len(BUILTIN_METRICS) == 36
    ndcg = next(c for c in catalog if c["ref"] == "ndcg_at_k@1")
    assert ndcg["task"] == "ranking"
    assert '"k"' in json.dumps(ndcg["args"])
