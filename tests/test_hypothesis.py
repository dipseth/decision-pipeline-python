import dataclasses
import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from decision_pipeline.hypothesis import (
    compile_hypothesis,
    evaluate_hypothesis,
    lock_hypothesis,
    partition_rows,
    plan_hypothesis,
)
from decision_pipeline.jsonata import jsonata_engine
from decision_pipeline.metrics import BUILTIN_METRICS, ObservationRow, define_metric, metric_registry


def compile(spec, **opts):
    r = compile_hypothesis(spec, **opts)
    if not r.ok:
        raise ValueError("\n".join(r.problems))
    return r.hypothesis


def close(a, b, digits=2):
    return abs(a - b) < 10 ** (-digits) / 2


agreement = {
    "id": "probe-gate-keeps-cuisine",
    "claim": "The probe-gate challenger returns the champion's primary cuisine on at least 90% of recipes.",
    "population": {"describe": "held-out recipes"},
    "metric": {"ref": "exact_match@1", "prediction": "prediction.slug", "truth": "truth"},
    "estimand": {"kind": "level", "arm": "challenger"},
    "test": {"kind": "greater", "than": 0.9},
}


def cuisine_rows(n, wrong=0):
    return [
        ObservationRow(
            id=f"r{i}",
            arm="challenger",
            prediction={"slug": "french" if i < wrong else "italian"},
            truth="italian",
            recorded_at="2026-09-24T00:00:00Z",
        )
        for i in range(n)
    ]


# describe("compileHypothesis")
def test_returns_every_problem_at_once():
    r = compile_hypothesis(
        {
            **agreement,
            "metric": {"ref": "no_such@1"},
            "guardrails": [
                {"name": "cost", "metric": {"ref": "within_tolerance@1", "args": {}}, "test": {"kind": "less", "than": 0}},
                {"name": "cost", "metric": {"ref": "value@1"}, "test": {"kind": "less", "than": 0}},
            ],
        }
    )
    assert r.ok is False
    assert any("unknown metric no_such@1" in p for p in r.problems)
    assert any(p.startswith("guardrails.0.metric.args") for p in r.problems)
    assert any('duplicate "cost"' in p for p in r.problems)


def test_refuses_shape_mismatches_the_data_could_never_fix():
    def bad(patch):
        r = compile_hypothesis({**agreement, **patch})
        return [] if r.ok else r.problems

    assert re.search(r"outside \[0, 1\]", bad({"test": {"kind": "greater", "than": 1.2}})[0])
    assert re.search(r"wilson needs a binary metric", bad({"metric": {"ref": "abs_error@1"}, "interval": "wilson"})[0])
    assert re.search(r"only a bootstrap", bad({"metric": {"ref": "auc@1"}, "interval": "normal"})[0])
    assert re.search(r"exactly one of ref, expr", bad({"metric": {"ref": "exact_match@1", "expr": "1"}})[0])
    assert re.search(r"needs an ExpressionEngine", bad({"metric": {"expr": "prediction = truth"}})[0])
    assert re.search(r"must differ", bad({"estimand": {"kind": "difference", "treatment": "a", "control": "a"}})[0])


def test_warns_when_the_claim_says_the_treatment_is_worse_on_a_lower_is_better_metric():
    r = compile_hypothesis(
        {
            **agreement,
            "metric": {"ref": "abs_error@1"},
            "estimand": {"kind": "difference", "treatment": "b", "control": "a"},
            "test": {"kind": "greater", "than": 0},
        }
    )
    assert r.ok is True
    assert re.search(r"treatment is WORSE", r.warnings[0])


def test_hashes_the_meaning_key_order_is_irrelevant_the_claim_is_not():
    a = compile(agreement)

    def reverse_keys(v):
        if isinstance(v, list):
            return [reverse_keys(x) for x in v]
        if isinstance(v, dict):
            return {k: reverse_keys(x) for k, x in reversed(list(v.items()))}
        return v

    b = compile(reverse_keys(agreement))
    assert b.hash == a.hash
    # An explicit default is the same claim.
    assert compile({**agreement, "alpha": 0.05}).hash == a.hash
    assert compile({**agreement, "test": {"kind": "greater", "than": 0.85}}).hash != a.hash
    assert compile({**agreement, "metric": {**agreement["metric"], "args": {"caseInsensitive": True}}}).hash != a.hash


def test_hash_matches_the_typescript_package():
    # `compileHypothesis(agreement).hypothesis.hash` in src/hypothesis.ts.
    assert compile({**agreement, "claim": "The probe-gate challenger returns the champion primary cuisine on at least 90% of recipes."}).hash == "357a3ba7c7b12a64"


# describe("evaluateHypothesis — the decision rule")
async def test_40_40_supports_gt_0_9_with_a_wilson_interval_instead_of_a_zero_width_one():
    v = await evaluate_hypothesis(compile(agreement), cuisine_rows(40))
    assert v.status == "supported"
    assert v.primary.interval.method == "wilson"
    assert close(v.primary.interval.confidence, 0.9, 10)  # one-sided α=0.05 → 90% two-sided
    assert v.primary.interval.hi == 1
    assert v.primary.interval.lo > 0.9
    assert v.preregistered is False
    assert re.search(r"EXPLORATORY", " ".join(v.warnings))


async def test_10_10_is_inconclusive_for_gt_0_9_and_says_how_much_data_would_settle_it():
    v = await evaluate_hypothesis(compile({**agreement, "plan": {"minEffect": 0.05}}), cuisine_rows(10))
    assert v.status == "inconclusive"
    assert v.primary.power.underpowered is True
    assert v.primary.power.required_n > 10
    assert re.search(r"units would detect", v.reason)


async def test_refutes_when_the_whole_interval_is_on_the_wrong_side():
    v = await evaluate_hypothesis(compile(agreement), cuisine_rows(200, 60))
    assert v.status == "refuted"
    assert close(v.primary.estimate, 0.7, 10)


async def test_counts_unscorable_rows_instead_of_guessing():
    rows = [*cuisine_rows(30), ObservationRow(id="x", arm="challenger", prediction={}, truth=None)]
    v = await evaluate_hypothesis(compile(agreement), rows)
    assert v.primary.n.unscored == 1
    assert v.primary.n.scored["challenger"] == 30


# describe("preregistration and the holdout")
async def test_a_lock_whose_hash_moved_makes_the_verdict_invalid():
    h = compile(agreement)
    lock = lock_hypothesis(h, datetime(2026, 9, 23, tzinfo=timezone.utc))
    moved = compile({**agreement, "test": {"kind": "greater", "than": 0.8}})
    v = await evaluate_hypothesis(moved, cuisine_rows(40), lock=lock)
    assert v.status == "invalid"


async def test_rows_recorded_before_the_lock_are_dropped_undated_rows_are_flagged():
    h = compile(agreement)
    lock = lock_hypothesis(h, datetime(2026, 9, 23, 12, tzinfo=timezone.utc))
    rows = [
        *cuisine_rows(40),
        *[dataclasses.replace(r, id=f"old{i}", recorded_at="2026-09-22T00:00:00Z") for i, r in enumerate(cuisine_rows(5, 5))],
        dataclasses.replace(cuisine_rows(1)[0], id="undated", recorded_at=None),
    ]
    v = await evaluate_hypothesis(h, rows, lock=lock)
    assert v.preregistered is True
    assert v.excluded.before_lock == 5
    assert v.undated == 1
    assert v.primary.n.scored["challenger"] == 41


async def test_splits_by_unit_deterministically_and_judges_only_the_declared_side():
    spec = {**agreement, "population": {"describe": "x", "unit": "meta.user", "split": {"salt": "s1", "holdout": 0.3, "use": "holdout"}}}
    rows = [dataclasses.replace(r, meta={"user": f"u{i % 80}"}) for i, r in enumerate(cuisine_rows(400))]
    parts = partition_rows(compile(spec).spec, rows)
    dev, holdout = parts.dev, parts.holdout
    assert len(dev) + len(holdout) == 400
    # Whole users on one side.
    holdout_users = {r.meta["user"] for r in holdout}
    assert not any(r.meta["user"] in holdout_users for r in dev)
    assert len(holdout_users) / 80 > 0.15
    assert len(holdout_users) / 80 < 0.45
    v = await evaluate_hypothesis(compile(spec), rows)
    assert v.excluded.outside_split == len(dev)
    assert v.primary.n.units == len(holdout_users)


async def test_filters_the_population():
    spec = {**agreement, "population": {"describe": "tail only", "where": [{"path": "meta.route", "equals": "tail"}]}}
    rows = [dataclasses.replace(r, meta={"route": "tail" if i % 3 == 0 else "leaf"}) for i, r in enumerate(cuisine_rows(30))]
    v = await evaluate_hypothesis(compile(spec), rows)
    assert v.excluded.filtered == 20


# describe("prediction types beyond classification")
def paired_rows(n, f):
    out = []
    for i in range(n):
        d = f(i)
        out.append(ObservationRow(id=f"i{i}", arm="champion", prediction=d["a"], truth=d["truth"]))
        out.append(ObservationRow(id=f"i{i}", arm="challenger", prediction=d["b"], truth=d["truth"]))
    return out


async def test_regression_a_paired_mae_reduction_bootstrap():
    h = compile(
        {
            "id": "scale-mae",
            "claim": "The challenger's scaled quantities are closer to the cook's by at least 0.1 on average.",
            "population": {"describe": "scaled recipes"},
            "metric": {"ref": "abs_error@1"},
            "estimand": {"kind": "difference", "treatment": "challenger", "control": "champion"},
            "test": {"kind": "less", "than": -0.1},
        }
    )
    rows = paired_rows(60, lambda i: {"truth": i, "a": i + 1 + (i % 5) * 0.1, "b": i + 0.5 + (i % 5) * 0.1})
    v = await evaluate_hypothesis(h, rows)
    assert v.primary.interval.method == "bootstrap"
    assert close(v.primary.estimate, -0.5, 10)
    assert v.status == "supported"


async def test_ranking_equivalent_ndcg_within_a_margin_a_cheaper_ranker_is_as_good():
    h = compile(
        {
            "id": "graph-ndcg",
            "claim": "The cheap ranker's NDCG@3 is within 0.05 of the current ranker's.",
            "population": {"describe": "search queries"},
            "metric": {"ref": "ndcg_at_k@1", "args": {"k": 3}},
            "estimand": {"kind": "difference", "treatment": "challenger", "control": "champion"},
            "test": {"kind": "equivalent", "within": 0.05},
        }
    )
    rows = paired_rows(
        80,
        lambda i: {"truth": {"a": 3, "b": 2, "c": 1}, "a": ["a", "b", "c"], "b": ["b", "a", "c"] if i % 20 == 0 else ["a", "b", "c"]},
    )
    v = await evaluate_hypothesis(h, rows)
    assert v.status == "supported"
    assert v.primary.estimate < 0


async def test_probabilistic_auc_above_chance_is_a_sample_metric_with_a_bootstrap_interval():
    h = compile(
        {
            "id": "confidence-auc",
            "claim": "The gate's confidence separates right from wrong answers better than chance.",
            "population": {"describe": "cuisine runs with a hand label"},
            "metric": {"ref": "auc@1", "prediction": "meta.confidence", "truth": "truth.correct"},
            "estimand": {"kind": "level"},
            "test": {"kind": "greater", "than": 0.5},
        }
    )
    rows = [
        ObservationRow(
            id=f"r{i}",
            prediction=None,
            truth={"correct": i % 2 == 0},
            meta={"confidence": 0.6 + (i % 7) * 0.05 if i % 2 == 0 else 0.4 + (i % 9) * 0.05},
        )
        for i in range(80)
    ]
    v = await evaluate_hypothesis(h, rows)
    assert v.primary.interval.method == "bootstrap"
    assert v.status == "supported"


async def test_paired_binary_adds_mcnemar_s_exact_test():
    h = compile(
        {
            **agreement,
            "estimand": {"kind": "difference", "treatment": "challenger", "control": "champion"},
            "test": {"kind": "different", "from": 0},
        }
    )
    rows = paired_rows(50, lambda i: {"truth": "it", "a": {"slug": "fr" if i < 10 else "it"}, "b": {"slug": "it"}})
    v = await evaluate_hypothesis(h, rows)
    assert v.primary.extra["mcnemar_only_treatment"] == 10
    assert v.primary.extra["mcnemar_only_control"] == 0
    assert close(v.primary.extra["mcnemar_p"], 2 / 1024, 10)
    assert v.status == "supported"


async def test_an_expression_covers_the_metric_nobody_registered():
    h = compile(
        {
            "id": "servings-parity",
            "claim": "The challenger keeps the servings count within one of the truth on at least 80% of recipes.",
            "population": {"describe": "scaled recipes"},
            "metric": {"expr": "$abs(prediction - truth) <= 1", "binary": True, "prediction": "prediction.servings"},
            "estimand": {"kind": "level"},
            "test": {"kind": "greater", "than": 0.8},
        },
        engine=jsonata_engine(),
    )
    rows = [ObservationRow(id=f"r{i}", prediction={"servings": 9 if i % 20 == 0 else 4}, truth=4) for i in range(100)]
    v = await evaluate_hypothesis(h, rows)
    assert re.match(r"^expr~", v.primary.metric)
    assert v.primary.interval.method == "wilson"
    assert close(v.primary.estimate, 0.95, 10)
    assert v.status == "supported"


async def test_an_expression_that_reads_truth_leaves_rows_with_no_truth_unscored_not_wrong():
    h = compile(
        {
            "id": "judged-right",
            "claim": "At least half the answers are judged right by the reviewer.",
            "population": {"describe": "judged answers"},
            "metric": {"expr": "prediction in truth.right", "binary": True, "direction": "higher"},
            "estimand": {"kind": "level"},
            "test": {"kind": "greater", "than": 0.5},
        },
        engine=jsonata_engine(),
    )
    rows = [
        ObservationRow(id=f"r{i}", prediction="thai", **({"truth": {"right": ["thai"]}} if i < 30 else {}))
        for i in range(40)
    ]
    v = await evaluate_hypothesis(h, rows)
    assert v.primary.n.unscored == 10
    assert v.primary.estimate == 1


async def test_a_host_metric_registers_like_a_built_in():
    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    length_ratio = define_metric(
        id="steps_ratio",
        version="1",
        describe="predicted steps / true steps",
        task="regression",
        direction="target",
        binary=False,
        needs_truth=True,
        scope="row",
        args=NoArgs,
        score=lambda o, _a: len(o.prediction) / len(o.truth),
    )
    h = compile(
        {
            "id": "steps",
            "claim": "Break-it-down keeps step count within 10% of the source's on average.",
            "population": {"describe": "recipes"},
            "metric": {"ref": "steps_ratio@1"},
            "estimand": {"kind": "level"},
            "test": {"kind": "equivalent", "around": 1, "within": 0.1},
        },
        metrics={**BUILTIN_METRICS, **metric_registry(length_ratio)},
    )
    rows = [ObservationRow(id=f"r{i}", prediction=[0] * (10 + (i % 3) - 1), truth=[0] * 10) for i in range(40)]
    assert (await evaluate_hypothesis(h, rows)).status == "supported"


# describe("guardrails and segments")
guard_rows = [
    ObservationRow(
        id=f"r{i}",
        arm="challenger",
        prediction={"slug": "italian"},
        truth="italian",
        meta={"cost_usd": 0.004 + (i % 3) * 0.001, "lang": "en" if i % 2 else "de"},
    )
    for i in range(60)
]


async def test_a_supported_primary_with_a_failed_guardrail_is_refuted_and_says_which():
    h = compile(
        {
            **agreement,
            "guardrails": [
                {
                    "name": "cost",
                    "metric": {"ref": "value@1", "prediction": "meta.cost_usd"},
                    "estimand": {"kind": "level"},
                    "test": {"kind": "less", "than": 0.002},
                }
            ],
        }
    )
    v = await evaluate_hypothesis(h, guard_rows)
    assert v.primary.status == "supported"
    assert v.guardrails[0].status == "refuted"
    assert v.status == "refuted"
    assert re.search(r'guardrail "cost" failed', v.reason)


async def test_segments_are_reported_per_value_and_never_change_the_verdict():
    h = compile({**agreement, "segments": [{"name": "lang", "by": "meta.lang"}]})
    v = await evaluate_hypothesis(h, guard_rows)
    assert [s.value for s in v.segments] == ["de", "en"]
    assert all(s.result.n.scored["challenger"] == 30 for s in v.segments)
    assert v.status == "supported"


async def test_a_peek_before_plan_n_is_marked_not_final():
    v = await evaluate_hypothesis(compile({**agreement, "plan": {"n": 100}}), guard_rows)
    assert v.final is False


# describe("planHypothesis")
def test_sizes_a_binary_claim_from_its_baseline():
    h = compile({**agreement, "test": {"kind": "greater", "than": 0.85}, "plan": {"minEffect": 0.05, "baseline": 0.9}})
    p = plan_hypothesis(h)
    assert p.design == "one-sample"
    # ((1.645 + 0.8416) · 0.3 / 0.05)² ≈ 222.6
    assert p.required_n == 223


def test_says_what_it_is_missing_instead_of_inventing_a_number():
    assert plan_hypothesis(compile(agreement)).required_n is None
