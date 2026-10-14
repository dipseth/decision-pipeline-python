import math

import pytest

from decision_pipeline.stats import (
    binomial_cdf,
    cluster_bootstrap,
    cluster_robust_se,
    mcnemar_exact,
    newcombe_interval,
    normal_cdf,
    normal_quantile,
    percentile_interval,
    required_sample_size,
    seeded_random,
    wilson_interval,
)


EXPECTED_X = [0.8552622240968049, 0.3278664539102465, 0.27123957639560103, 0.049846729496493936]


def close(a, b, digits):
    # vitest toBeCloseTo(b, digits): |a - b| < 10^-digits / 2
    return abs(a - b) < 10 ** (-digits) / 2


# describe("normal distribution")
def test_quantile_and_cdf_invert_each_other_at_the_usual_points():
    assert close(normal_quantile(0.975), 1.959964, 5)
    assert close(normal_quantile(0.05), -1.644854, 5)
    assert close(normal_quantile(0.001), -3.090232, 5)
    assert close(normal_cdf(1.959964), 0.975, 6)
    assert close(normal_cdf(0), 0.5, 7)


def test_normal_quantile_rejects_p_outside_0_1():
    with pytest.raises(ValueError, match=r"normalQuantile: p must be in \(0, 1\), got 1"):
        normal_quantile(1)


# describe("proportion intervals")
def test_wilson_does_not_collapse_at_n_n_40_40_is_about_0_912_1():
    iv = wilson_interval(40, 40, 0.95)
    assert close(iv.lo, 0.9124, 3)
    assert iv.hi == 1


def test_newcombe_matches_his_published_example_56_70_vs_48_80():
    iv = newcombe_interval(56, 70, 48, 80, 0.95)
    assert close(iv.lo, 0.0524, 3)
    assert close(iv.hi, 0.3339, 3)


# describe("exact tests")
def test_binomial_cdf_and_mcnemar():
    assert close(binomial_cdf(0, 5, 0.5), 1 / 32, 10)
    assert close(mcnemar_exact(0, 5), 2 / 32, 10)
    assert mcnemar_exact(3, 3) == 1
    assert mcnemar_exact(0, 0) == 1


# describe("cluster-robust se")
def test_equals_sd_sqrt_n_with_the_g_g_1_factor_when_every_row_is_its_own_cluster():
    xs = [1, 2, 3, 4, 5, 6]
    se = cluster_robust_se(xs, [str(x) for x in xs])
    sd = math.sqrt(sum((x - 3.5) ** 2 for x in xs) / 5)
    assert close(se, sd / math.sqrt(6), 10)


def test_grows_when_rows_are_duplicated_inside_clusters_copies_are_not_new_evidence():
    xs = [0, 1, 0, 1, 1, 0, 1, 1]
    independent = cluster_robust_se(xs, [str(i) for i in range(len(xs))])
    dup = xs + xs
    clustered = cluster_robust_se(dup, [str(i % len(xs)) for i in range(len(dup))])
    naive = cluster_robust_se(dup, [str(i) for i in range(len(dup))])
    assert naive < independent
    assert clustered > naive


# describe("bootstrap")
def test_is_reproducible_from_its_seed_and_differs_across_seeds():
    data = [[1], [2], [3], [4], [5]]

    def run(seed):
        return cluster_bootstrap(
            [data],
            lambda sample: sum(sample[0]) / len(sample[0]),
            resamples=300,
            random=seeded_random(seed),
        )

    assert run("a").draws == run("a").draws
    assert run("a").draws != run("b").draws
    iv = percentile_interval(run("a").draws, 0.9)
    assert iv.lo < 3
    assert iv.hi > 3


def test_seeded_random_matches_the_ts_mulberry32_sequence():
    # First draws of TS `seededRandom("x")` (node, src/stats.ts).
    r = seeded_random("x")
    assert [r() for _ in range(4)] == EXPECTED_X


# describe("sample size")
def test_one_sample_paired_and_two_sample_designs():
    # ((1.96 + 0.8416) · 1 / 0.5)² = 31.4
    assert required_sample_size(sd=1, effect=0.5, alpha_per_side=0.025, power=0.8, design="one-sample") == 32
    assert required_sample_size(sd=1, effect=0.5, alpha_per_side=0.025, power=0.8, design="paired") == 32
    assert required_sample_size(sd=1, effect=0.5, alpha_per_side=0.025, power=0.8, design="two-sample") == 63
