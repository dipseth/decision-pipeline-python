"""
The verdict engine, tested against worlds where the answer is KNOWN.

A verdict is only worth something if (a) with no real effect it claims one
about α of the time, and (b) with the planned effect it finds it about
`plan.power` of the time. These simulate both, plus the failure that
clustering causes when the unit is declared wrong. The bands are wide
enough for the simulation counts; a real bug (a 1 − α interval where 1 − 2α
belongs, a bootstrap that ignores clusters) lands far outside them.

The generator is the TS mulberry32, bit for bit, so each simulation sees the
same data the TS suite sees.
"""

import math

from decision_pipeline.hypothesis import compile_hypothesis, evaluate_hypothesis, plan_hypothesis
from decision_pipeline.metrics import ObservationRow
from decision_pipeline.stats import seeded_random


def compile(spec):
    r = compile_hypothesis(spec)
    if not r.ok:
        raise ValueError("\n".join(r.problems))
    return r.hypothesis


def gaussian(random):
    def draw():
        u = max(random(), 1e-12)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * random())

    return draw


async def rate(sims, one):
    hits = 0
    for s in range(sims):
        if await one(s):
            hits += 1
    return hits / sims


# describe("calibration")
async def test_a_a_binary_unpaired_claims_a_difference_about_alpha_of_the_time_newcombe():
    h = compile(
        {
            "id": "aa-binary",
            "claim": "Two identical arms differ in their success rate.",
            "population": {"describe": "simulated"},
            "metric": {"ref": "truthy@1"},
            "estimand": {"kind": "difference", "treatment": "b", "control": "a", "paired": False},
            "test": {"kind": "different", "from": 0},
        }
    )
    random = seeded_random("aa-binary")

    async def one(s):
        rows = []
        for i in range(120):
            rows.append(ObservationRow(id=f"a{s}-{i}", arm="a", prediction=random() < 0.7))
            rows.append(ObservationRow(id=f"b{s}-{i}", arm="b", prediction=random() < 0.7))
        return (await evaluate_hypothesis(h, rows)).status == "supported"

    fp = await rate(400, one)
    assert fp > 0.015
    assert fp < 0.09


async def test_one_sided_coverage_a_true_rate_exactly_at_the_bound_is_supported_about_alpha_of_the_time_wilson():
    h = compile(
        {
            "id": "at-bound",
            "claim": "The rate is above 0.9.",
            "population": {"describe": "simulated"},
            "metric": {"ref": "truthy@1"},
            "estimand": {"kind": "level"},
            "test": {"kind": "greater", "than": 0.9},
        }
    )
    random = seeded_random("at-bound")

    async def one(s):
        rows = [ObservationRow(id=f"{s}-{i}", prediction=random() < 0.9) for i in range(150)]
        return (await evaluate_hypothesis(h, rows)).status == "supported"

    fp = await rate(600, one)
    assert fp > 0.015
    assert fp < 0.09


async def test_a_planted_paired_effect_is_found_at_about_the_planned_power_bootstrap():
    spec = {
        "id": "planted",
        "claim": "The challenger scores higher than the champion.",
        "population": {"describe": "simulated"},
        "metric": {"ref": "value@1"},
        "estimand": {"kind": "difference", "treatment": "challenger", "control": "champion"},
        "test": {"kind": "greater", "than": 0},
        "bootstrap": {"resamples": 400},
        "plan": {"minEffect": 0.3, "sd": 0.5, "power": 0.8},
    }
    n = plan_hypothesis(compile(spec)).required_n
    assert n == 18
    h = compile({**spec, "plan": {**spec["plan"], "n": n}})
    random = seeded_random("planted")
    normal = gaussian(random)

    async def one(s):
        rows = []
        for i in range(n):
            item = normal()
            rows.append(ObservationRow(id=f"{s}-{i}", arm="champion", prediction=item))
            rows.append(ObservationRow(id=f"{s}-{i}", arm="challenger", prediction=item + 0.3 + 0.5 * normal()))
        return (await evaluate_hypothesis(h, rows)).status == "supported"

    power = await rate(200, one)
    # Percentile bootstrap runs a little liberal at n=18; the band allows it.
    assert power > 0.65
    assert power < 0.95


async def test_shadow_like_pairs_true_difference_at_the_non_inferiority_margin_newcombe_paired_holds_alpha_where_the_bootstrap_does_not():
    # A shadow that shares its champion's answers agrees on most pairs, so
    # few are discordant. Here 1% favour the challenger and 6% the champion:
    # the true difference is exactly the −0.05 margin, so "supported" is a
    # false positive. The percentile bootstrap resamples almost no discordant
    # pairs at small n and its interval shrinks toward the point estimate.
    def spec(interval):
        return {
            "id": f"margin-{interval}",
            "claim": "The challenger is at most 5 points worse.",
            "population": {"describe": "simulated"},
            "metric": {"ref": "truthy@1"},
            "estimand": {"kind": "difference", "treatment": "challenger", "control": "champion"},
            "test": {"kind": "greater", "than": -0.05},
            "interval": interval,
            "bootstrap": {"resamples": 500},
        }

    async def sim(interval):
        h = compile(spec(interval))
        random = seeded_random("margin")

        async def one(s):
            rows = []
            for i in range(30):
                u = random()
                t, c = (True, False) if u < 0.01 else (False, True) if u < 0.07 else (True, True) if u < 0.87 else (False, False)
                rows.append(ObservationRow(id=f"{s}-{i}", arm="challenger", prediction=t))
                rows.append(ObservationRow(id=f"{s}-{i}", arm="champion", prediction=c))
            return (await evaluate_hypothesis(h, rows)).status == "supported"

        return await rate(300, one)

    newcombe = await sim("auto")
    bootstrap = await sim("bootstrap")
    assert newcombe < 0.09
    assert bootstrap > 0.1


async def test_zero_discordant_pairs_at_small_n_is_not_evidence_12_identical_pairs_stay_inconclusive():
    h = compile(
        {
            "id": "zero-discordant",
            "claim": "The challenger is at most 5 points worse.",
            "population": {"describe": "simulated"},
            "metric": {"ref": "truthy@1"},
            "estimand": {"kind": "difference", "treatment": "challenger", "control": "champion"},
            "test": {"kind": "greater", "than": -0.05},
        }
    )
    rows = []
    for i in range(12):
        rows.append(ObservationRow(id=f"{i}", arm="challenger", prediction=i < 10))
        rows.append(ObservationRow(id=f"{i}", arm="champion", prediction=i < 10))
    v = await evaluate_hypothesis(h, rows)
    assert v.primary.interval.method == "newcombe_paired"
    assert v.status == "inconclusive"
    assert v.primary.interval.lo < -0.1


async def test_clustered_a_a_declaring_the_right_unit_keeps_false_positives_near_alpha_ignoring_it_does_not():
    base = {
        "id": "clustered",
        "claim": "The two arms differ in their mean.",
        "population": {"describe": "simulated users, 10 rows each"},
        "metric": {"ref": "value@1"},
        "estimand": {"kind": "difference", "treatment": "b", "control": "a", "paired": False},
        "test": {"kind": "different", "from": 0},
        "interval": "normal",
    }
    naive = compile(base)
    clustered = compile({**base, "population": {**base["population"], "unit": "meta.user"}})
    random = seeded_random("clustered")
    normal = gaussian(random)
    naive_fp = 0
    clustered_fp = 0
    sims = 300
    for s in range(sims):
        rows = []
        for arm in ("a", "b"):
            for u in range(15):
                user_effect = normal()  # strong per-user effect, no arm effect
                for k in range(10):
                    rows.append(
                        ObservationRow(
                            id=f"{s}-{arm}-{u}-{k}", arm=arm, prediction=user_effect + 0.3 * normal(), meta={"user": f"{arm}{u}"}
                        )
                    )
        if (await evaluate_hypothesis(naive, rows)).status == "supported":
            naive_fp += 1
        if (await evaluate_hypothesis(clustered, rows)).status == "supported":
            clustered_fp += 1
    assert naive_fp / sims > 0.3
    assert clustered_fp / sims < 0.12
