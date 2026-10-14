"""
Metrics — what a hypothesis MEASURES, one registered function per idea,
referenced `id@version` exactly like a spec primitive.

The trick that makes one verdict engine serve every kind of prediction is
that almost any prediction task reduces to a per-row number:

  classification   exact_match · top_k            (0/1 per row)
  probabilistic    log_loss · brier · prob_of_truth · crps_gaussian
  ranking          reciprocal_rank · ndcg_at_k · hit_at_k · average_precision · kendall_tau
  sets             jaccard · set_f1 · set_exact
  regression       abs_error · squared_error · log_ratio_error · within_tolerance · signed_error
  intervals        interval_covers · interval_width · interval_score · pinball
  generic          value · truthy · equals        (cost, latency, a route, a human score)

A handful of ideas are NOT a mean of per-row numbers — AUC, calibration
error, kappa, correlation, macro-F1. Those are `sample` metrics: computed
over the whole sample, and always given a bootstrap interval.

Anything not listed is either a host-registered metric (`define_metric`) or a
`expr` metric in the hypothesis itself — which is how an LLM author covers
the problem nobody thought of without anyone writing Python.

TS `undefined` and `null` are both `None` here. Every built-in already treats
an absent truth and a null truth alike; for the PREDICTION, a null value is
"absent" too (TS compared an explicit `null` prediction as the key `"null"`).

A metric's args are a pydantic model (zod `.strict()` → `extra="forbid"`);
the parsed model instance is what `score` / `compute` receive. Arg names stay
camelCase where TS has them (`caseInsensitive`): they are what a hypothesis
JSON names.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from ._compat import MISSING, as_schema
from .hash import stable_stringify
from .stats import mean, normal_cdf

# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ObservationRow:
    """
    One observation: what was predicted, and (when known) what was true. The
    verdict engine knows nothing about pipelines — `observation_from_run` builds
    these from a run result, but any source works.
    """

    #: The ITEM. Rows of two arms with the same id are a pair.
    id: str
    #: Which arm produced the prediction. None = a single-arm sample.
    arm: Optional[str] = None
    #: The independent unit this row belongs to (a user, a recipe) — rows sharing
    #: one are resampled together. Defaults to `id`. A hypothesis may instead name
    #: a path (`population.unit`).
    unit: Optional[str] = None
    prediction: Any
    #: Ground truth, a human label, a reference answer. None = not yet known.
    truth: Any = None
    #: ISO time the prediction was made — what the preregistration check reads.
    #: Read by paths as `recordedAt` (the wire name).
    recorded_at: Optional[str] = None
    #: Anything else a filter, segment, or `value` metric may read (cost, route, lang).
    meta: Optional[dict[str, Any]] = None


@dataclass(kw_only=True)
class Observation:
    """What a metric function sees: the picked prediction and truth, plus the row."""

    prediction: Any
    truth: Any
    row: ObservationRow


_DIGITS = re.compile(r"^[0-9]+$")


def _pick(value: Any, path: Optional[str]) -> Any:
    """`pick_path` that returns MISSING (TS `undefined`) for an absent step. Internal."""
    if not path:
        return value
    cur = value
    for part in path.split("."):
        if cur is None or cur is MISSING:
            return MISSING
        if isinstance(cur, (list, tuple)):
            if _DIGITS.match(part):
                i = int(part)
                cur = cur[i] if i < len(cur) else MISSING
            else:
                return MISSING
        elif isinstance(cur, Mapping):
            cur = cur.get(part, MISSING)
        elif isinstance(cur, (str, bytes, int, float, bool)):
            return MISSING
        elif part.startswith("_"):
            return MISSING
        else:
            cur = getattr(cur, part, MISSING)
    return cur


def pick_path(value: Any, path: Optional[str]) -> Any:
    """`a.b.0.c` into nested dicts / lists (or attributes). An empty path is the value itself. Absent → None."""
    v = _pick(value, path)
    return None if v is MISSING else v


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

MetricTask = Literal["classification", "probabilistic", "ranking", "set", "regression", "interval", "agreement", "generic"]

#: Which way is better — so an author can tell whether to claim `greater` or `less`.
MetricDirection = Literal["higher", "lower", "target", "either"]


@dataclass(kw_only=True)
class RowMetric:
    #: Dotted, lower_snake: `ndcg_at_k`. Referenced as `id@version`.
    id: str
    version: str
    #: One or two sentences, for the catalog a model reads.
    describe: str
    task: str
    direction: str
    #: Row scores are always 0/1 — unlocks Wilson / Newcombe / McNemar.
    binary: bool
    #: False when the metric reads only the prediction (cost, width, a route).
    needs_truth: bool
    #: A pydantic model class, a TypeAdapter, or anything `as_schema` accepts.
    args: Any
    #: None = this row cannot be scored (missing or malformed); it is counted, not guessed.
    score: Callable[[Observation, Any], Optional[float]]
    scope: str = dataclasses.field(default="row", init=False)


@dataclass(kw_only=True)
class SampleMetric:
    id: str
    version: str
    describe: str
    task: str
    direction: str
    binary: bool
    needs_truth: bool
    args: Any
    compute: Callable[[Sequence[Observation], Any], Optional[float]]
    scope: str = dataclasses.field(default="sample", init=False)


Metric = Union[RowMetric, SampleMetric]

MetricRegistry = Mapping[str, Metric]


def define_metric(m: Optional[Metric] = None, /, **kw: Any) -> Metric:
    """
    `define_metric(RowMetric(...))` returns it as-is. Keyword form:
    `define_metric(id=..., scope="row", score=...)` / `scope="sample", compute=...`.
    """
    if m is not None:
        return m
    scope = kw.pop("scope", "row")
    if scope == "sample":
        return SampleMetric(**kw)
    return RowMetric(**kw)


def metric_ref(m: Any) -> str:
    return f"{m.id}@{m.version}"


def metric_registry(*metrics: Metric) -> dict[str, Metric]:
    """Keyed `id@version`. Later metrics win, so a host can override a built-in's version."""
    out: dict[str, Metric] = {}
    for m in metrics:
        out[metric_ref(m)] = m
    return out


def metric_catalog(registry: MetricRegistry) -> list[dict[str, Any]]:
    """The catalog a model authors against: every metric with its args as JSON Schema."""
    return [
        {
            "ref": ref,
            "describe": m.describe,
            "task": m.task,
            "scope": m.scope,
            "direction": m.direction,
            "binary": m.binary,
            "needsTruth": m.needs_truth,
            "args": as_schema(m.args).json_schema(),
        }
        for ref, m in registry.items()
    ]


# ---------------------------------------------------------------------------
# Coercions — lenient in, None out
# ---------------------------------------------------------------------------


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def key_of(v: Any) -> str:
    return v if isinstance(v, str) else stable_stringify(v)


_JS_DECIMAL = re.compile(r"^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_JS_RADIX = re.compile(r"^0([xXoObB])([0-9a-fA-F]+)$")


def _js_number_from_string(s: str) -> Optional[float]:
    """`Number(s)` for a non-blank string; None when it is NaN or not finite."""
    t = s.strip()
    if _JS_DECIMAL.match(t):
        n = float(t)
        return n if math.isfinite(n) else None
    m = _JS_RADIX.match(t)
    if m:
        base = {"x": 16, "o": 8, "b": 2}[m.group(1).lower()]
        try:
            return int(m.group(2), base)
        except ValueError:
            return None
    return None


def as_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1 if v else 0
    if _is_num(v):
        return v if math.isfinite(v) else None
    if isinstance(v, str) and v.strip() != "":
        return _js_number_from_string(v)
    return None


def as_binary(v: Any) -> Optional[int]:
    if v is True or (_is_num(v) and v == 1):
        return 1
    if v is False or (_is_num(v) and v == 0):
        return 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y"):
            return 1
        if s in ("0", "false", "no", "n"):
            return 0
    return None


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))


def as_keys(v: Any) -> Optional[list[str]]:
    """A list of item keys, in order. A bare scalar is a list of one."""
    if v is None or v is MISSING:
        return None
    if isinstance(v, (list, tuple)):
        return [key_of(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return [key_of(x) for x in v]
    if not _is_scalar(v):
        return None
    return [key_of(v)]


def as_prob_map(v: Any) -> Optional[dict[str, float]]:
    """`{label: p}` — also accepts `[{label, p}]`."""
    if isinstance(v, (list, tuple)):
        m: dict[str, float] = {}
        for e in v:
            if not isinstance(e, Mapping):
                return None
            label = e.get("label", MISSING)
            n = as_number(e.get("p"))
            if label is MISSING or n is None:
                return None
            m[key_of(label)] = n
        return m
    if isinstance(v, Mapping):
        m = {}
        for k, p in v.items():
            n = as_number(p)
            if n is None:
                return None
            m[str(k)] = n
        return m
    return None


def as_ranking(v: Any) -> Optional[list[str]]:
    """A ranked list from either a list or a probability map (sorted by p, then label)."""
    if isinstance(v, (list, tuple)):
        return [key_of(x) for x in v]
    m = as_prob_map(v)
    if m is None:
        return None
    return [k for k, _ in sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))]


def as_gains(v: Any) -> Optional[dict[str, float]]:
    """Relevance: a single item, a list of items (gain 1), or `{item: gain}`."""
    if v is None or v is MISSING:
        return None
    if isinstance(v, Mapping):
        return as_prob_map(v)
    keys = as_keys(v)
    return {k: 1 for k in keys} if keys is not None else None


def prob_of_truth(prediction: Any, truth: Any) -> Optional[float]:
    """P(truth) under the prediction: a number is P(true) of a binary outcome; a map is per label."""
    p = as_number(prediction)
    if p is not None and not isinstance(prediction, bool):
        y = as_binary(truth)
        return None if y is None else (p if y == 1 else 1 - p)
    m = as_prob_map(prediction)
    if m is None or truth is None:
        return None
    got = m.get(key_of(truth))
    return 0 if got is None else got


def _pair_numbers(os_: Sequence[Observation]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for o in os_:
        p = as_number(o.prediction)
        t = as_number(o.truth)
        if p is not None and t is not None:
            out.append((p, t))
    return out


def _average_ranks(xs: Sequence[float]) -> list[float]:
    idx = sorted(((x, i) for i, x in enumerate(xs)), key=lambda e: e[0])
    ranks: list[float] = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1][0] == idx[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[idx[k][1]] = r
        i = j + 1
    return ranks


def _pearson(pairs: Sequence[tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 3:
        return None
    mx = mean([p[0] for p in pairs])
    my = mean([p[1] for p in pairs])
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for x, y in pairs:
        sxy += (x - mx) * (y - my)
        sxx += (x - mx) ** 2
        syy += (y - my) ** 2
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def _pow2(g: float) -> float:
    try:
        return 2.0 ** float(g)
    except OverflowError:
        return math.inf


# ---------------------------------------------------------------------------
# Args models (zod `.strict()` → extra="forbid"; strict types, no coercion)
# ---------------------------------------------------------------------------


def _integral(v: Any) -> Any:
    """zod `.int()` accepts 2.0 (it IS 2 in JS); pydantic strict int would not."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


JsInt = Annotated[int, BeforeValidator(_integral)]

_STRICT = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class NoArgs(BaseModel):
    model_config = _STRICT


class KArg(BaseModel):
    model_config = _STRICT
    k: JsInt = Field(ge=1)


class OptionalKArg(BaseModel):
    model_config = _STRICT
    k: Optional[JsInt] = Field(default=None, ge=1)


class ExactMatchArgs(BaseModel):
    model_config = _STRICT
    caseInsensitive: bool = False


class LogLossArgs(BaseModel):
    model_config = _STRICT
    eps: float = Field(default=1e-15, gt=0, le=0.1)


class WithinToleranceArgs(BaseModel):
    model_config = _STRICT
    abs: Optional[float] = Field(default=None, ge=0)
    rel: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _one(self) -> "WithinToleranceArgs":
        if self.abs is None and self.rel is None:
            raise PydanticCustomError("custom", "give abs, rel, or both")
        return self


class BoundKeys(BaseModel):
    model_config = _STRICT
    lo: str = "lo"
    hi: str = "hi"


class IntervalScoreArgs(BoundKeys):
    alpha: float = Field(gt=0, lt=1)


class PinballArgs(BaseModel):
    model_config = _STRICT
    q: float = Field(gt=0, lt=1)


class EqualsArgs(BaseModel):
    model_config = _STRICT
    value: Any = None
    in_: list[Any] = Field(default=None, alias="in")  # type: ignore[assignment,arg-type]

    @model_validator(mode="after")
    def _exactly_one(self) -> "EqualsArgs":
        # Given vs absent, as zod's `!== undefined`: an explicit null counts as given.
        if ("value" in self.model_fields_set) == ("in_" in self.model_fields_set):
            raise PydanticCustomError("custom", "give exactly one of value, in")
        return self


class EceArgs(BaseModel):
    model_config = _STRICT
    bins: JsInt = Field(default=10, ge=2, le=100)


# ---------------------------------------------------------------------------
# Built-ins
# ---------------------------------------------------------------------------


def _exact_match_fn(o: Observation, a: ExactMatchArgs) -> Optional[float]:
    if o.prediction is None or o.truth is None:
        return None

    def norm(v: Any) -> str:
        if not isinstance(v, str):
            return key_of(v)
        s = v.strip()
        return s.lower() if a.caseInsensitive else s

    return 1 if norm(o.prediction) == norm(o.truth) else 0


_exact_match = RowMetric(
    id="exact_match",
    version="1",
    describe="1 when the prediction equals the truth (strings trimmed; optionally case-insensitive), else 0. Accuracy, as a mean.",
    task="classification",
    direction="higher",
    binary=True,
    needs_truth=True,
    args=ExactMatchArgs,
    score=_exact_match_fn,
)


def _top_k_fn(o: Observation, a: KArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    if ranked is None or o.truth is None:
        return None
    return 1 if key_of(o.truth) in ranked[: a.k] else 0


_top_k = RowMetric(
    id="top_k",
    version="1",
    describe="1 when the truth is among the first k of a ranked list or the k most probable labels of a probability map.",
    task="classification",
    direction="higher",
    binary=True,
    needs_truth=True,
    args=KArg,
    score=_top_k_fn,
)


def _log_loss_fn(o: Observation, a: LogLossArgs) -> Optional[float]:
    p = prob_of_truth(o.prediction, o.truth)
    return None if p is None else -math.log(min(1 - a.eps, max(a.eps, p)))


_log_loss = RowMetric(
    id="log_loss",
    version="1",
    describe="−ln P(truth). Prediction is P(true) for a binary truth, or a {label: p} map. Clamped at eps. Lower is better.",
    task="probabilistic",
    direction="lower",
    binary=False,
    needs_truth=True,
    args=LogLossArgs,
    score=_log_loss_fn,
)


def _brier_fn(o: Observation, _a: Any) -> Optional[float]:
    p = as_number(o.prediction)
    if p is not None and not isinstance(o.prediction, bool):
        y = as_binary(o.truth)
        return None if y is None else (p - y) ** 2
    m = as_prob_map(o.prediction)
    if m is None or o.truth is None:
        return None
    t = key_of(o.truth)
    s: float = 0 if t in m else 1
    for k, q in m.items():
        s += (q - (1 if k == t else 0)) ** 2
    return s


_brier = RowMetric(
    id="brier",
    version="1",
    describe="Squared probability error: (p − y)² for P(true), or Σ over labels of (p_label − 1[label = truth])² for a map. Lower is better.",
    task="probabilistic",
    direction="lower",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    score=_brier_fn,
)

_prob_truth = RowMetric(
    id="prob_of_truth",
    version="1",
    describe="The probability the prediction assigned to what turned out true. Higher is better.",
    task="probabilistic",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    score=lambda o, _a: prob_of_truth(o.prediction, o.truth),
)


def _crps_gaussian_fn(o: Observation, _a: Any) -> Optional[float]:
    mu = as_number(pick_path(o.prediction, "mean"))
    sd = as_number(pick_path(o.prediction, "sd"))
    t = as_number(o.truth)
    if mu is None or sd is None or t is None or sd < 0:
        return None
    if sd == 0:
        return abs(t - mu)
    z = (t - mu) / sd
    pdf = math.exp(-(z * z) / 2) / math.sqrt(2 * math.pi)
    return sd * (z * (2 * normal_cdf(z) - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


_crps_gaussian = RowMetric(
    id="crps_gaussian",
    version="1",
    describe="Continuous ranked probability score of a Gaussian forecast {mean, sd} against a numeric truth. Lower is better; equals abs error when sd → 0.",
    task="probabilistic",
    direction="lower",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    score=_crps_gaussian_fn,
)


def _gain(gains: Mapping[str, float], item: str) -> float:
    g = gains.get(item)
    return 0 if g is None else g


def _reciprocal_rank_fn(o: Observation, a: OptionalKArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    gains = as_gains(o.truth)
    if ranked is None or gains is None:
        return None
    top = ranked if a.k is None else ranked[: a.k]
    for i, item in enumerate(top):
        if _gain(gains, item) > 0:
            return 1 / (i + 1)
    return 0


_reciprocal_rank = RowMetric(
    id="reciprocal_rank",
    version="1",
    describe="1 / (position of the first relevant item) in a ranked prediction, 0 if none in the first k. Truth: an item, a list, or {item: gain}. MRR, as a mean.",
    task="ranking",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=OptionalKArg,
    score=_reciprocal_rank_fn,
)


def _hit_at_k_fn(o: Observation, a: KArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    gains = as_gains(o.truth)
    if ranked is None or gains is None:
        return None
    return 1 if any(_gain(gains, item) > 0 for item in ranked[: a.k]) else 0


_hit_at_k = RowMetric(
    id="hit_at_k",
    version="1",
    describe="1 when any relevant item appears in the first k of the ranking.",
    task="ranking",
    direction="higher",
    binary=True,
    needs_truth=True,
    args=KArg,
    score=_hit_at_k_fn,
)


def _precision_at_k_fn(o: Observation, a: KArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    gains = as_gains(o.truth)
    if ranked is None or gains is None:
        return None
    return len([item for item in ranked[: a.k] if _gain(gains, item) > 0]) / a.k


_precision_at_k = RowMetric(
    id="precision_at_k",
    version="1",
    describe="Relevant items in the first k, divided by k.",
    task="ranking",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=KArg,
    score=_precision_at_k_fn,
)


def _recall_at_k_fn(o: Observation, a: KArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    gains = as_gains(o.truth)
    if ranked is None or gains is None:
        return None
    relevant = len([g for g in gains.values() if g > 0])
    if relevant == 0:
        return None
    return len([item for item in ranked[: a.k] if _gain(gains, item) > 0]) / relevant


_recall_at_k = RowMetric(
    id="recall_at_k",
    version="1",
    describe="Relevant items in the first k, divided by all relevant items. Unscored when nothing is relevant.",
    task="ranking",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=KArg,
    score=_recall_at_k_fn,
)


def _average_precision_fn(o: Observation, a: OptionalKArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    gains = as_gains(o.truth)
    if ranked is None or gains is None:
        return None
    relevant = len([g for g in gains.values() if g > 0])
    if relevant == 0:
        return None
    top = ranked if a.k is None else ranked[: a.k]
    hits = 0
    s = 0.0
    for i, item in enumerate(top):
        if _gain(gains, item) > 0:
            hits += 1
            s += hits / (i + 1)
    return s / min(relevant, len(top) or relevant)


_average_precision = RowMetric(
    id="average_precision",
    version="1",
    describe="Mean of precision@i over the positions i of relevant items (within the first k if given). MAP, as a mean.",
    task="ranking",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=OptionalKArg,
    score=_average_precision_fn,
)


def _dcg(gs: Sequence[float]) -> float:
    s = 0.0
    for i, g in enumerate(gs):
        s += (_pow2(g) - 1) / math.log2(i + 2)
    return s


def _ndcg_at_k_fn(o: Observation, a: KArg) -> Optional[float]:
    ranked = as_ranking(o.prediction)
    gains = as_gains(o.truth)
    if ranked is None or gains is None:
        return None
    ideal = _dcg(sorted([g for g in gains.values() if g > 0], reverse=True)[: a.k])
    if ideal == 0:
        return None
    return _dcg([_gain(gains, item) for item in ranked[: a.k]]) / ideal


_ndcg_at_k = RowMetric(
    id="ndcg_at_k",
    version="1",
    describe="Normalised discounted cumulative gain over the first k. Truth: relevant items (gain 1) or {item: gain} for graded relevance.",
    task="ranking",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=KArg,
    score=_ndcg_at_k_fn,
)


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _kendall_tau_fn(o: Observation, _a: Any) -> Optional[float]:
    p = as_ranking(o.prediction)
    t = as_keys(o.truth)
    if p is None or t is None:
        return None
    pos = {k: i for i, k in enumerate(t)}
    common = [k for k in p if k in pos]
    n = len(common)
    if n < 2:
        return None
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += _sign(pos[common[j]] - pos[common[i]])
    return s / ((n * (n - 1)) / 2)


_kendall_tau = RowMetric(
    id="kendall_tau",
    version="1",
    describe="Kendall's tau-a between the predicted and the true ORDER of the items both lists contain (−1 reversed … 1 identical).",
    task="ranking",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    score=_kendall_tau_fn,
)


def _set_ops(prediction: Any, truth: Any) -> Optional[tuple[int, int, int]]:
    p = as_keys(prediction)
    t = as_keys(truth)
    if p is None or t is None:
        return None
    ps = set(p)
    ts = set(t)
    return len(ps), len(ts), len(ps & ts)


def _jaccard_fn(o: Observation, _a: Any) -> Optional[float]:
    s = _set_ops(o.prediction, o.truth)
    if s is None:
        return None
    p, t, inter = s
    union = p + t - inter
    return 1 if union == 0 else inter / union


_jaccard = RowMetric(
    id="jaccard",
    version="1",
    describe="|predicted ∩ true| / |predicted ∪ true| for set-valued predictions (tags, ingredients). Two empty sets score 1.",
    task="set",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    score=_jaccard_fn,
)


def _set_f1_fn(o: Observation, _a: Any) -> Optional[float]:
    s = _set_ops(o.prediction, o.truth)
    if s is None:
        return None
    p, t, inter = s
    return 1 if p + t == 0 else (2 * inter) / (p + t)


_set_f1 = RowMetric(
    id="set_f1",
    version="1",
    describe="Per-row F1 between predicted and true sets: 2|P∩T| / (|P| + |T|). Two empty sets score 1.",
    task="set",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    score=_set_f1_fn,
)


def _set_exact_fn(o: Observation, _a: Any) -> Optional[float]:
    s = _set_ops(o.prediction, o.truth)
    if s is None:
        return None
    p, t, inter = s
    return 1 if inter == p and inter == t else 0


_set_exact = RowMetric(
    id="set_exact",
    version="1",
    describe="1 when the predicted and true sets are identical (order ignored).",
    task="set",
    direction="higher",
    binary=True,
    needs_truth=True,
    args=NoArgs,
    score=_set_exact_fn,
)


def _numeric(fn: Callable[[float, float], Optional[float]]) -> Callable[[Observation, Any], Optional[float]]:
    def score(o: Observation, _a: Any = None) -> Optional[float]:
        p = as_number(o.prediction)
        t = as_number(o.truth)
        return None if p is None or t is None else fn(p, t)

    return score


def _regression(id: str, describe: str, direction: str, fn: Callable[[float, float], Optional[float]]) -> RowMetric:
    return RowMetric(
        id=id,
        version="1",
        describe=describe,
        task="regression",
        direction=direction,
        binary=False,
        needs_truth=True,
        args=NoArgs,
        score=_numeric(fn),
    )


abs_error = _regression("abs_error", "|prediction − truth|. MAE, as a mean.", "lower", lambda p, t: abs(p - t))
squared_error = _regression("squared_error", "(prediction − truth)². MSE, as a mean.", "lower", lambda p, t: (p - t) ** 2)
signed_error = _regression(
    "signed_error",
    "prediction − truth. Its mean is the BIAS; claim it is equivalent to 0.",
    "target",
    lambda p, t: p - t,
)
abs_pct_error = _regression(
    "abs_pct_error",
    "|prediction − truth| / |truth|. MAPE, as a mean. Unscored when truth is 0.",
    "lower",
    lambda p, t: None if t == 0 else abs(p - t) / abs(t),
)
log_ratio_error = _regression(
    "log_ratio_error",
    "|ln(prediction / truth)| — symmetric multiplicative error, for positive quantities (scale factors, amounts).",
    "lower",
    lambda p, t: abs(math.log(p / t)) if p > 0 and t > 0 else None,
)


def _within_tolerance_fn(o: Observation, a: WithinToleranceArgs) -> Optional[float]:
    p = as_number(o.prediction)
    t = as_number(o.truth)
    if p is None or t is None:
        return None
    return 1 if abs(p - t) <= max(a.abs if a.abs is not None else 0, (a.rel if a.rel is not None else 0) * abs(t)) else 0


_within_tolerance = RowMetric(
    id="within_tolerance",
    version="1",
    describe="1 when |prediction − truth| ≤ max(abs, rel·|truth|). Give abs, rel, or both.",
    task="regression",
    direction="higher",
    binary=True,
    needs_truth=True,
    args=WithinToleranceArgs,
    score=_within_tolerance_fn,
)


def _bounds(prediction: Any, lo: str, hi: str) -> Optional[tuple[float, float]]:
    if isinstance(prediction, (list, tuple)):
        a = as_number(prediction[0] if len(prediction) > 0 else None)
        b = as_number(prediction[1] if len(prediction) > 1 else None)
    else:
        a = as_number(pick_path(prediction, lo))
        b = as_number(pick_path(prediction, hi))
    return None if a is None or b is None or a > b else (a, b)


def _interval_covers_fn(o: Observation, a: BoundKeys) -> Optional[float]:
    b = _bounds(o.prediction, a.lo, a.hi)
    t = as_number(o.truth)
    return None if b is None or t is None else (1 if b[0] <= t <= b[1] else 0)


_interval_covers = RowMetric(
    id="interval_covers",
    version="1",
    describe="1 when the truth lies inside a predicted interval ({lo, hi} or [lo, hi]). Its mean is empirical COVERAGE — claim it is equivalent to the nominal level.",
    task="interval",
    direction="target",
    binary=True,
    needs_truth=True,
    args=BoundKeys,
    score=_interval_covers_fn,
)


def _interval_width_fn(o: Observation, a: BoundKeys) -> Optional[float]:
    b = _bounds(o.prediction, a.lo, a.hi)
    return b[1] - b[0] if b is not None else None


_interval_width = RowMetric(
    id="interval_width",
    version="1",
    describe="hi − lo of a predicted interval. Sharpness; needs no truth.",
    task="interval",
    direction="lower",
    binary=False,
    needs_truth=False,
    args=BoundKeys,
    score=_interval_width_fn,
)


def _interval_score_fn(o: Observation, a: IntervalScoreArgs) -> Optional[float]:
    b = _bounds(o.prediction, a.lo, a.hi)
    t = as_number(o.truth)
    if b is None or t is None:
        return None
    lo, hi = b
    return hi - lo + (2 / a.alpha) * max(0, lo - t) + (2 / a.alpha) * max(0, t - hi)


_interval_score = RowMetric(
    id="interval_score",
    version="1",
    describe="Gneiting–Raftery interval score for a central (1 − alpha) interval: width plus 2/alpha × the miss distance. Rewards narrow AND honest intervals. Lower is better.",
    task="interval",
    direction="lower",
    binary=False,
    needs_truth=True,
    args=IntervalScoreArgs,
    score=_interval_score_fn,
)

_pinball = RowMetric(
    id="pinball",
    version="1",
    describe="Quantile (pinball) loss of a predicted q-quantile against the truth. Lower is better.",
    task="interval",
    direction="lower",
    binary=False,
    needs_truth=True,
    args=PinballArgs,
    score=lambda o, a: _numeric(lambda p, t: max(a.q * (t - p), (a.q - 1) * (t - p)))(o, a),
)

_value = RowMetric(
    id="value",
    version="1",
    describe="The prediction itself as a number (booleans → 0/1). For cost, latency, confidence, a human score — anything already numeric. Point `prediction` at it.",
    task="generic",
    direction="either",
    binary=False,
    needs_truth=False,
    args=NoArgs,
    score=lambda o, _a: as_number(o.prediction),
)

_truthy = RowMetric(
    id="truthy",
    version="1",
    describe='1 when the prediction is true / 1 / "yes", 0 when false / 0 / "no". A rate of some yes/no property.',
    task="generic",
    direction="either",
    binary=True,
    needs_truth=False,
    args=NoArgs,
    score=lambda o, _a: as_binary(o.prediction),
)


def _equals_fn(o: Observation, a: EqualsArgs) -> Optional[float]:
    if o.prediction is None:
        return None
    k = key_of(o.prediction)
    allowed = a.in_ if a.in_ is not None else [a.value]
    return 1 if any(key_of(v) == k for v in allowed) else 0


_equals = RowMetric(
    id="equals",
    version="1",
    describe="1 when the prediction equals `value` (or is one of `in`). The rate of a particular answer — a route, a label.",
    task="generic",
    direction="either",
    binary=True,
    needs_truth=False,
    args=EqualsArgs,
    score=_equals_fn,
)

# --- sample-scope --------------------------------------------------------


def _labels(os_: Sequence[Observation]) -> list[tuple[str, str]]:
    return [(key_of(o.prediction), key_of(o.truth)) for o in os_ if o.prediction is not None and o.truth is not None]


def _cohen_kappa_fn(os_: Sequence[Observation], _a: Any) -> Optional[float]:
    pairs = _labels(os_)
    n = len(pairs)
    if n == 0:
        return None
    pc: dict[str, int] = {}
    tc: dict[str, int] = {}
    agree = 0
    for p, t in pairs:
        pc[p] = pc.get(p, 0) + 1
        tc[t] = tc.get(t, 0) + 1
        if p == t:
            agree += 1
    pe = 0.0
    for k, c in pc.items():
        pe += (c / n) * (tc.get(k, 0) / n)
    return None if pe == 1 else (agree / n - pe) / (1 - pe)


_cohen_kappa = SampleMetric(
    id="cohen_kappa",
    version="1",
    describe="Cohen's kappa between predicted and true categories — agreement beyond chance. Use it to check an LLM judge against human labels.",
    task="agreement",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    compute=_cohen_kappa_fn,
)


def _macro_f1_fn(os_: Sequence[Observation], _a: Any) -> Optional[float]:
    pairs = _labels(os_)
    if len(pairs) == 0:
        return None
    all_labels = dict.fromkeys(x for p, t in pairs for x in (p, t))
    f1s: list[float] = []
    for lab in all_labels:
        tp = fp = fn = 0
        for p, t in pairs:
            if p == lab and t == lab:
                tp += 1
            elif p == lab:
                fp += 1
            elif t == lab:
                fn += 1
        f1s.append((2 * tp) / (2 * tp + fp + fn))
    return mean(f1s)


_macro_f1 = SampleMetric(
    id="macro_f1",
    version="1",
    describe="Unweighted mean over labels of per-label F1 — classification quality that a dominant class cannot carry.",
    task="classification",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    compute=_macro_f1_fn,
)


def _auc_fn(os_: Sequence[Observation], _a: Any) -> Optional[float]:
    scored: list[tuple[float, int]] = []
    for o in os_:
        s = as_number(o.prediction)
        y = as_binary(o.truth)
        if s is not None and y is not None:
            scored.append((s, y))
    pos = len([x for x in scored if x[1] == 1])
    neg = len(scored) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _average_ranks([x[0] for x in scored])
    rank_sum = 0.0
    for i, x in enumerate(scored):
        if x[1] == 1:
            rank_sum += ranks[i]
    return (rank_sum - (pos * (pos + 1)) / 2) / (pos * neg)


_auc = SampleMetric(
    id="auc",
    version="1",
    describe="ROC AUC: probability a random positive (truth true) gets a higher prediction score than a random negative. Undefined with one class.",
    task="probabilistic",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    compute=_auc_fn,
)


def _ece_fn(os_: Sequence[Observation], a: EceArgs) -> Optional[float]:
    bins = a.bins
    acc = [{"n": 0, "p": 0.0, "y": 0} for _ in range(bins)]
    n = 0
    for o in os_:
        p = as_number(o.prediction)
        y = as_binary(o.truth)
        if p is None or y is None or p < 0 or p > 1:
            continue
        b = acc[min(bins - 1, math.floor(p * bins))]
        b["n"] += 1
        b["p"] += p
        b["y"] += y
        n += 1
    if n == 0:
        return None
    s = 0.0
    for b in acc:
        if b["n"] != 0:
            s = s + (b["n"] / n) * abs(b["y"] / b["n"] - b["p"] / b["n"])
    return s


_ece = SampleMetric(
    id="ece",
    version="1",
    describe="Expected calibration error: bins predictions by stated probability and averages |observed rate − stated probability|. Prediction: a confidence in [0,1]; truth: whether it was right. Lower is better.",
    task="probabilistic",
    direction="lower",
    binary=False,
    needs_truth=True,
    args=EceArgs,
    compute=_ece_fn,
)

_pearson_metric = SampleMetric(
    id="pearson",
    version="1",
    describe="Pearson correlation between numeric predictions and truths across rows.",
    task="regression",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    compute=lambda os_, _a: _pearson(_pair_numbers(os_)),
)


def _spearman_fn(os_: Sequence[Observation], _a: Any) -> Optional[float]:
    pairs = _pair_numbers(os_)
    rx = _average_ranks([p[0] for p in pairs])
    ry = _average_ranks([p[1] for p in pairs])
    return _pearson(list(zip(rx, ry)))


_spearman = SampleMetric(
    id="spearman",
    version="1",
    describe="Spearman rank correlation between numeric predictions and truths across rows — monotone agreement, robust to scale.",
    task="regression",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    compute=_spearman_fn,
)


def _r2_fn(os_: Sequence[Observation], _a: Any) -> Optional[float]:
    pairs = _pair_numbers(os_)
    if len(pairs) < 2:
        return None
    my = mean([p[1] for p in pairs])
    res = 0.0
    tot = 0.0
    for p, t in pairs:
        res += (t - p) ** 2
        tot += (t - my) ** 2
    return None if tot == 0 else 1 - res / tot


_r2 = SampleMetric(
    id="r2",
    version="1",
    describe="Coefficient of determination 1 − SS_res / SS_tot of predictions against truths.",
    task="regression",
    direction="higher",
    binary=False,
    needs_truth=True,
    args=NoArgs,
    compute=_r2_fn,
)

BUILTIN_METRICS: dict[str, Metric] = metric_registry(
    _exact_match,
    _top_k,
    _log_loss,
    _brier,
    _prob_truth,
    _crps_gaussian,
    _reciprocal_rank,
    _hit_at_k,
    _precision_at_k,
    _recall_at_k,
    _average_precision,
    _ndcg_at_k,
    _kendall_tau,
    _jaccard,
    _set_f1,
    _set_exact,
    abs_error,
    squared_error,
    signed_error,
    abs_pct_error,
    log_ratio_error,
    _within_tolerance,
    _interval_covers,
    _interval_width,
    _interval_score,
    _pinball,
    _value,
    _truthy,
    _equals,
    _cohen_kappa,
    _macro_f1,
    _auc,
    _ece,
    _pearson_metric,
    _spearman,
    _r2,
)
