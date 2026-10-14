"""
The statistics a hypothesis verdict needs — intervals, paired tests, power —
written out rather than imported, so the package keeps pydantic as its only
runtime dependency.

Everything that draws randomness takes a SEEDED generator: a verdict is
replayable, and a bootstrap that moves on every re-run is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence, TypeVar

from ._compat import js_number
from .hash import sha256_hex

R = TypeVar("R")

#: Bumped whenever an estimator changes. Recorded on every verdict.
STATS_VERSION = "3"

_M32 = 0xFFFFFFFF

# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------


def _imul(a: int, b: int) -> int:
    """JS `Math.imul`: the low 32 bits of the product, as an unsigned int here."""
    return ((a & _M32) * (b & _M32)) & _M32


def seeded_random(seed: str) -> Callable[[], float]:
    """mulberry32, seeded from a string via sha256. Uniform on [0, 1). Bit-identical to the TS sequence."""
    state = [int(sha256_hex(seed)[:8], 16) & _M32]

    def nxt() -> float:
        a = (state[0] + 0x6D2B79F5) & _M32
        state[0] = a
        t = a
        t = _imul(t ^ (t >> 15), t | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & _M32
        t &= _M32
        return ((t ^ (t >> 14)) & _M32) / 4294967296

    return nxt


# ---------------------------------------------------------------------------
# The normal distribution
# ---------------------------------------------------------------------------


def normal_cdf(x: float) -> float:
    """Φ(x). Abramowitz & Stegun 7.1.26 on erf — absolute error < 1.5e-7."""
    z = abs(x) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1 - poly * math.exp(-z * z)
    return (1 + erf) / 2 if x >= 0 else (1 - erf) / 2


_A = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239)
_B = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
_C = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
_D = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)


def _fmt(p: float) -> str:
    if isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p):
        return js_number(p)
    if p != p:
        return "NaN"
    return "Infinity" if p > 0 else "-Infinity"


def normal_quantile(p: float) -> float:
    """Φ⁻¹(p). Acklam's rational approximation — relative error < 1.2e-9."""
    if not (p > 0 and p < 1):
        raise ValueError(f"normalQuantile: p must be in (0, 1), got {_fmt(p)}")
    a0, a1, a2, a3, a4, a5 = _A
    b0, b1, b2, b3, b4 = _B
    c0, c1, c2, c3, c4, c5 = _C
    d0, d1, d2, d3 = _D
    lo = 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1)
    if p > 1 - lo:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1)
    q = p - 0.5
    r = q * q
    return ((((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q) / (
        ((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1
    )


def z_for(confidence: float) -> float:
    """The two-sided critical value for a `confidence` interval (0.95 → 1.96)."""
    return normal_quantile(1 - (1 - confidence) / 2)


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------


def mean(xs: Sequence[float]) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s / len(xs) if len(xs) else math.nan


def variance(xs: Sequence[float]) -> float:
    """Sample variance (n − 1). 0 for fewer than two values."""
    if len(xs) < 2:
        return 0
    m = mean(xs)
    s = 0.0
    for x in xs:
        s += (x - m) ** 2
    return s / (len(xs) - 1)


def quantile_sorted(sorted_: Sequence[float], q: float) -> float:
    """Type-7 (linear interpolation) quantile of an ascending-sorted array."""
    if len(sorted_) == 0:
        return math.nan
    h = (len(sorted_) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    a = sorted_[lo]
    b = sorted_[hi]
    return a + (h - lo) * (b - a)


def cluster_robust_se(values: Sequence[float], clusters: Sequence[str]) -> float:
    """
    Standard error of the mean of `values`, where rows sharing a cluster are NOT
    independent (one user's twenty recipes are not twenty draws). The CR1
    cluster-robust variance of a ratio estimator; with one row per cluster it is
    the ordinary `sd / √n` up to the n/(n−1) factor.
    """
    n = len(values)
    if n < 2:
        return math.nan
    m = mean(values)
    totals: dict[str, float] = {}
    for i in range(n):
        key = clusters[i]
        totals[key] = totals.get(key, 0) + (values[i] - m)
    g = len(totals)
    if g < 2:
        return math.nan
    s = 0.0
    for t in totals.values():
        s += t * t
    return math.sqrt((g / (g - 1)) * s) / n


# ---------------------------------------------------------------------------
# Intervals for proportions
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Interval:
    lo: float
    hi: float


def wilson_interval(successes: float, n: float, confidence: float) -> Interval:
    """
    Wilson score interval. Unlike `p̂ ± z·se` it stays inside [0, 1] and does not
    collapse to zero width at 0/n or n/n — 40/40 is [0.91, 1], not [1, 1].
    """
    if n <= 0:
        return Interval(lo=0, hi=1)
    z = z_for(confidence)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom
    return Interval(lo=max(0, center - half), hi=min(1, center + half))


def newcombe_interval(x1: float, n1: float, x2: float, n2: float, confidence: float) -> Interval:
    """Newcombe's hybrid score interval for p1 − p2 (independent samples; his method 10)."""
    p1 = x1 / n1
    p2 = x2 / n2
    a = wilson_interval(x1, n1, confidence)
    b = wilson_interval(x2, n2, confidence)
    d = p1 - p2
    return Interval(
        lo=d - math.sqrt((p1 - a.lo) ** 2 + (b.hi - p2) ** 2),
        hi=d + math.sqrt((a.hi - p1) ** 2 + (p2 - b.lo) ** 2),
    )


def newcombe_paired_interval(a: float, b: float, c: float, d: float, confidence: float) -> Interval:
    """
    Newcombe's hybrid score interval for p1 − p2 on PAIRED binary data (his
    1998 method 10). Counts: a both 1, b only the first, c only the second, d
    both 0. The percentile bootstrap collapses to [0, 0] when no pair is
    discordant, which is the usual state of a shadow sharing its champion's
    answers; this does not — 0 discordant of 12 is still a wide interval.
    """
    n = a + b + c + d
    if n <= 0:
        return Interval(lo=-1, hi=1)
    p1 = (a + b) / n
    p2 = (a + c) / n
    w1 = wilson_interval(a + b, n, confidence)
    w2 = wilson_interval(a + c, n, confidence)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denom) if denom > 0 else 0
    dl1 = p1 - w1.lo
    du1 = w1.hi - p1
    dl2 = p2 - w2.lo
    du2 = w2.hi - p2
    theta = p1 - p2
    return Interval(
        lo=max(-1, theta - math.sqrt(max(0, dl1**2 - 2 * phi * dl1 * du2 + du2**2))),
        hi=min(1, theta + math.sqrt(max(0, du1**2 - 2 * phi * du1 * dl2 + dl2**2))),
    )


# ---------------------------------------------------------------------------
# Exact tests
# ---------------------------------------------------------------------------

_log_factorial_cache: list[float] = [0.0]


def _log_factorial(n: int) -> float:
    cache = _log_factorial_cache
    for i in range(len(cache), n + 1):
        cache.append(cache[i - 1] + math.log(i))
    return cache[n]


def _log(x: float) -> float:
    """`Math.log`: -Infinity at 0 rather than a ValueError."""
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0
    if k >= n:
        return 1
    s = 0.0
    lp = _log(p)
    lq = _log(1 - p)
    for i in range(int(k) + 1):
        # 0 · log(0) is NaN, exactly as in JS.
        s += math.exp(_log_factorial(n) - _log_factorial(i) - _log_factorial(n - i) + i * lp + (n - i) * lq)
    return min(1, s) if s == s else math.nan


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact McNemar test for paired binary outcomes. `b` = pairs where only the
    treatment succeeded, `c` = only the control. Two-sided p. Ties (both or
    neither) carry no information about the difference and do not enter.
    """
    n = b + c
    if n == 0:
        return 1
    return min(1, 2 * binomial_cdf(min(b, c), n, 0.5))


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BootstrapResult:
    #: Draws where the statistic was defined, ascending.
    draws: list[float] = field(default_factory=list)
    #: Draws where it was not (e.g. an AUC over a resample with one class).
    undefined_draws: int = 0


def cluster_bootstrap(
    groups: Sequence[Sequence[Sequence[R]]],
    stat: Callable[[list[list[R]]], Optional[float]],
    resamples: int,
    random: Callable[[], float],
) -> BootstrapResult:
    """
    Cluster bootstrap, stratified by group. `groups[g][c]` is the rows of
    cluster `c` in group `g`; each resample redraws clusters with replacement
    INSIDE each group (so an unpaired two-arm comparison keeps both arm sizes)
    and hands `stat` each group's rows flattened.
    """
    draws: list[float] = []
    undefined_draws = 0
    for _ in range(resamples):
        sample: list[list[R]] = []
        for clusters in groups:
            rows: list[R] = []
            k = len(clusters)
            for _i in range(k):
                pick = clusters[math.floor(random() * k)]
                rows.extend(pick)
            sample.append(rows)
        v = stat(sample)
        if v is None or isinstance(v, bool) or not math.isfinite(v):
            undefined_draws += 1
        else:
            draws.append(v)
    draws.sort()
    return BootstrapResult(draws=draws, undefined_draws=undefined_draws)


def percentile_interval(sorted_draws: Sequence[float], confidence: float) -> Interval:
    tail = (1 - confidence) / 2
    return Interval(lo=quantile_sorted(sorted_draws, tail), hi=quantile_sorted(sorted_draws, 1 - tail))


# ---------------------------------------------------------------------------
# p-values and power
# ---------------------------------------------------------------------------


def p_upper(z: float) -> float:
    """One-sided upper-tail p for a z statistic."""
    return 1 - normal_cdf(z)


def detectable_effect(se: float, alpha_per_side: float, power: float) -> float:
    """
    The smallest true effect a design with standard error `se` detects with
    probability `power`, testing at `alpha` (already per-side).
    """
    return (normal_quantile(1 - alpha_per_side) + normal_quantile(power)) * se


SampleDesign = Literal["one-sample", "two-sample", "paired"]


def required_sample_size(*, sd: float, effect: float, alpha_per_side: float, power: float, design: str) -> float:
    """
    Units needed (PER ARM for `two-sample`) to detect `effect` given the per-unit
    standard deviation `sd` — of the metric for one-/two-sample, of the per-pair
    DIFFERENCE for `paired`. Normal approximation.
    """
    z = normal_quantile(1 - alpha_per_side) + normal_quantile(power)
    if effect == 0:
        return math.inf if z * sd != 0 else math.nan  # JS division by zero
    base = ((z * sd) / abs(effect)) ** 2
    v = 2 * base if design == "two-sample" else base
    return math.ceil(v) if math.isfinite(v) else v
