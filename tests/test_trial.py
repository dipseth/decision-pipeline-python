# Needs the core: define / run / shadow / testing / fixtures. Written against that API
# before shadow.py existed; the importorskip below keeps the suite green on a tree without it.
import dataclasses
import re
from datetime import datetime, timezone

import pytest

pytest.importorskip("decision_pipeline.shadow", reason="trial tests wait for the core (shadow.py)")

from decision_pipeline.fixtures import build_scale_pipeline, confident_distributions  # noqa: E402
from decision_pipeline.hypothesis import compile_hypothesis, split_of  # noqa: E402
from decision_pipeline.ports import DecideResult, GenerateResult  # noqa: E402
from decision_pipeline.testing import memory_trial_store, test_ports  # noqa: E402
from decision_pipeline.trial import collect_trial, judge_trial, lock_trial, register_trial  # noqa: E402


def compile(spec):
    r = compile_hypothesis(spec)
    if not r.ok:
        raise ValueError("\n".join(r.problems))
    return r.hypothesis


def ports():
    """Jev is confident, so the champion goes direct; the stricter challenger pays for the writer."""
    return test_ports(
        decide={"classify": DecideResult(distributions=confident_distributions(2), cost_usd=0.0002)},
        generate={
            "writer": GenerateResult(
                value={"scaledIngredients": ["2 cup flour", "2 tsp salt"], "scalingNotes": [], "factor": 2, "changes": []},
                cost_usd=0.004,
            )
        },
        prompts={"jev-scale-classify": {"version": "12"}, "recipe-scale": {"version": "7", "tools": []}},
    )


def stricter():
    base = build_scale_pipeline()
    gate = base.nodes["gate"]
    return dataclasses.replace(
        base,
        nodes={**base.nodes, "gate": dataclasses.replace(gate, thresholds={"category": 0.99, "tools": 0.5}, version="2")},
    )


inputs = [
    {"recipeId": f"r-{i}", "factor": 2, "notes": "", "substitutionCount": 0, "ingredients": ["1 cup flour", f"{i + 1} tsp salt"]}
    for i in range(24)
]


def recipe_id(i):
    return i["recipeId"] if isinstance(i, dict) else i.recipe_id


def costlier(split=None):
    return {
        "id": "strict-gate-costs-more",
        "claim": "The stricter gate costs more per scale than the champion.",
        "population": {"describe": "test recipes", **({"split": split} if split else {})},
        "metric": {"ref": "value@1", "prediction": "meta.cost_usd"},
        "estimand": {"kind": "difference", "treatment": "strict", "control": "champion"},
        "test": {"kind": "greater", "than": 0},
    }


#: test_ports' clock starts in 2023, so a lock must predate it for the rows to count.
LOCKED_AT = datetime.fromtimestamp(1_600_000_000, tz=timezone.utc)


def arms():
    return {"champion": {"name": "champion", "pipeline": build_scale_pipeline()}, "challenger": {"name": "strict", "pipeline": stricter()}}


def base(h, store, p=None):
    return dict(
        hypothesis=h,
        store=store,
        champion=build_scale_pipeline(),
        challenger=stricter(),
        inputs=inputs,
        id=recipe_id,
        ports=p if p is not None else ports(),
    )


# describe("trials")
async def test_locks_collects_shadow_pairs_and_judges_them_as_a_preregistered_verdict():
    h = compile(costlier())
    store = memory_trial_store()
    await register_trial(store, lock_trial(h, arms(), LOCKED_AT))

    async def truth(a):
        return f"truth-{a.id}"

    out = await collect_trial(
        **base(h, store),
        truth=truth,
        meta=lambda a: {"title": recipe_id(a.input), "arm_again": a.arm},
    )
    assert out.collected == 24
    assert out.cost_usd["strict"] - out.cost_usd["champion"] == pytest.approx(24 * 0.004, abs=0.005)
    rows = await store.rows(h.spec.id)
    assert len(rows) == 48
    assert rows[0].id == "r-0"
    assert rows[0].unit == "r-0"
    assert rows[0].truth == "truth-r-0"
    assert rows[0].meta["title"] == "r-0"
    assert rows[0].meta["arm_again"] == "champion"
    assert rows[1].arm == "strict"
    assert rows[1].meta["arm_again"] == "strict"

    j = await judge_trial(h, store)
    assert j.verdict.status == "supported", j.verdict.reason
    assert j.verdict.preregistered is True
    assert j.read == 1


async def test_resumes_a_second_collection_skips_what_the_store_already_has():
    h = compile(costlier())
    store = memory_trial_store()
    await register_trial(store, lock_trial(h, arms()))
    await collect_trial(**{**base(h, store), "inputs": inputs[:5]})
    p = ports()
    again = await collect_trial(**base(h, store, p))
    assert again.skipped.already_collected == 5
    assert again.collected == 19
    assert len(p.decide_calls) == 19


async def test_runs_only_the_split_the_claim_is_judged_on_and_counts_holdout_reads():
    split = {"salt": "t", "holdout": 0.5, "use": "holdout"}
    h = compile(costlier(split))
    store = memory_trial_store()
    await register_trial(store, lock_trial(h, arms(), LOCKED_AT))
    p = ports()
    out = await collect_trial(**base(h, store, p))
    holdout = len([i for i in inputs if split_of(recipe_id(i), split) == "holdout"])
    assert out.collected == holdout
    assert out.skipped.other_split == 24 - holdout
    assert len(p.decide_calls) == holdout

    assert not re.search(r"re-read", ",".join((await judge_trial(h, store)).verdict.warnings))
    second = await judge_trial(h, store)
    assert second.read == 2
    assert re.search(r"holdout read 2", ",".join(second.verdict.warnings))


async def test_refuses_a_changed_claim_under_a_locked_id_an_unregistered_trial_and_a_swapped_pipeline():
    h = compile(costlier())
    store = memory_trial_store()
    with pytest.raises(Exception, match=r"not registered"):
        await collect_trial(**base(h, store))

    lock = await register_trial(store, lock_trial(h, arms()))
    # Same trial again is idempotent and keeps the first registration time.
    again = await register_trial(store, lock_trial(h, arms(), datetime.fromtimestamp(1_650_000_000, tz=timezone.utc)))
    assert again.registered_at == lock.registered_at

    edited = compile({**costlier(), "test": {"kind": "greater", "than": 0.001}})
    with pytest.raises(Exception, match=r"already locked"):
        await register_trial(store, lock_trial(edited, arms()))
    with pytest.raises(Exception, match=r"changed since it was locked"):
        await collect_trial(**base(edited, store))

    with pytest.raises(Exception, match=r"challenger is not the pipeline that was locked"):
        await collect_trial(**{**base(h, store), "challenger": build_scale_pipeline()})
