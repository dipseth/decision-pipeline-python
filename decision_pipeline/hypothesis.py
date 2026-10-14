"""
Hypotheses — a claim, declared as data, that a sample can SUPPORT, REFUTE,
or leave INCONCLUSIVE, with the interval that decided it.

Everything the pipeline had before this measured things; nothing turned a
measurement into a verdict. A hypothesis is the missing object:

  population   which rows count (filters, the independent unit, a sealed holdout)
  metric       what is measured per row — a registered metric (./metrics) or an expression
  estimand     the quantity: one arm's LEVEL, or the DIFFERENCE between two (paired or not)
  test         the claim about it: greater · less · equivalent · different
  plan         minimum effect worth detecting, power, planned n
  guardrails   claims that must ALSO hold (cost, latency, failure rate)
  segments     preregistered breakdowns, reported but never used to rescue a verdict

It is JSON on purpose, like a pipeline spec: an LLM authors it, the compiler
returns every problem at once, and `lock_hypothesis` hashes it BEFORE data is
collected. `evaluate_hypothesis` refuses a spec whose hash moved and drops
rows recorded before the lock — changing the claim after seeing the data is
the failure this exists to prevent.

Decision rule — one rule for every claim shape. Build a confidence interval
for the estimand; the claim is SUPPORTED when the whole interval satisfies
it, REFUTED when the whole interval contradicts it, INCONCLUSIVE otherwise.
One-sided claims (greater, less) and equivalence use a 1 − 2α interval,
which is exactly a one-sided α test (TOST for equivalence); `different`
uses 1 − α. A `different` claim can never be refuted — absence of a
difference is an `equivalent` claim, which has to be made on purpose.

The spec is wire format: its keys stay camelCase (`minEffect`), `in` and
`from` are read as `in_` / `from_` on the parsed models.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Awaitable, Callable, Literal, Mapping, Optional, Sequence, Union

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from ._compat import MISSING, SchemaError, as_schema, iso_from_ms, js_number, maybe_await, ms_from_iso, now_ms, to_fixed, to_precision
from .expressions import CompiledExpression, ExpressionEngine
from .hash import hash_to_unit, short_hash, stable_stringify
from .metrics import BUILTIN_METRICS, Observation, ObservationRow, _pick, pick_path
from .stats import (
    STATS_VERSION,
    cluster_bootstrap,
    cluster_robust_se,
    detectable_effect,
    mcnemar_exact,
    mean,
    newcombe_interval,
    newcombe_paired_interval,
    p_upper,
    percentile_interval,
    required_sample_size,
    seeded_random,
    wilson_interval,
    z_for,
)

#: Bumped when the spec's MEANING changes. Part of every hash.
HYPOTHESIS_FORMAT = "1"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


def _pattern(pattern: str, message: str) -> Any:
    rx = re.compile(pattern)

    def check(v: str) -> str:
        if not rx.fullmatch(v):
            raise PydanticCustomError("custom", message)
        return v

    return Annotated[str, AfterValidator(check), WithJsonSchema({"type": "string", "pattern": f"^{pattern}$"})]


def _integral(v: Any) -> Any:
    """zod `.int()` accepts 2.0 (it IS 2 in JS); pydantic strict int would not."""
    return int(v) if isinstance(v, float) and v.is_integer() else v


JsInt = Annotated[int, BeforeValidator(_integral)]
NonEmpty = Annotated[str, Field(min_length=1)]

#: A dotted path into the ROW: `prediction.primary.cuisine_slug`, `truth.slug`, `meta.cost_usd`, `meta.user`, `arm`, `id`.
PathStr = _pattern(r"[A-Za-z0-9_$-]+(\.[A-Za-z0-9_$-]+)*", "a dotted path into the row, e.g. prediction.primary.cuisine_slug")
RefStr = _pattern(r"[a-z0-9_.]+@[\w.-]+", "id@version")
SlugStr = _pattern(r"[a-z0-9][a-z0-9_.-]*", "lower-case slug")

Direction = Literal["higher", "lower", "target", "either"]


class MetricSpec(BaseModel):
    model_config = _STRICT
    #: A registered metric, `id@version` — see `metric_catalog()`.
    ref: RefStr = None  # type: ignore[valid-type]
    args: dict[str, Any] = Field(default_factory=dict)
    #: A per-row expression instead of a registered metric. Sees `prediction`, `truth`
    #: (already picked) and `row`; returns a number, a boolean, or null (unscorable).
    expr: NonEmpty = None  # type: ignore[assignment]
    #: `expr` only: its scores are always 0/1 (unlocks Wilson / Newcombe / McNemar).
    binary: bool = None  # type: ignore[assignment]
    #: `expr` only: which way is better.
    direction: Direction = None  # type: ignore[assignment]
    #: Where the metric's prediction comes from. Default `prediction`.
    prediction: PathStr = "prediction"  # type: ignore[valid-type]
    #: Where its truth comes from. Default `truth`.
    truth: PathStr = "truth"  # type: ignore[valid-type]

    @model_validator(mode="after")
    def _refine(self) -> "MetricSpec":
        if (self.ref is None) == (self.expr is None):
            raise PydanticCustomError("custom", "give exactly one of ref, expr")
        if self.ref is not None and (self.binary is not None or self.direction is not None):
            raise PydanticCustomError("custom", "binary / direction come from the registered metric; set them only on an expr")
        return self


class LevelEstimand(BaseModel):
    model_config = _STRICT
    kind: Literal["level"]
    #: Only rows of this arm. Absent: every row.
    arm: NonEmpty = None  # type: ignore[assignment]


class DifferenceEstimand(BaseModel):
    model_config = _STRICT
    kind: Literal["difference"]
    #: Estimand = treatment − control.
    treatment: NonEmpty
    control: NonEmpty
    #: Rows of both arms sharing an `id` are one PAIR (a shadow run is paired by
    #: construction). Pairing removes item-to-item variation, often most of the noise.
    paired: bool = True

    @model_validator(mode="after")
    def _differ(self) -> "DifferenceEstimand":
        if self.treatment == self.control:
            raise PydanticCustomError("custom", "treatment and control must differ")
        return self


Estimand = Annotated[Union[LevelEstimand, DifferenceEstimand], Field(discriminator="kind")]


class GreaterTest(BaseModel):
    model_config = _STRICT
    kind: Literal["greater"]
    than: float


class LessTest(BaseModel):
    model_config = _STRICT
    kind: Literal["less"]
    than: float


class EquivalentTest(BaseModel):
    model_config = _STRICT
    kind: Literal["equivalent"]
    around: float = 0
    #: The margin inside which a difference does not matter.
    within: float = Field(gt=0)


class DifferentTest(BaseModel):
    model_config = _STRICT
    kind: Literal["different"]
    from_: float = Field(default=0, alias="from")


ClaimTest = Annotated[Union[GreaterTest, LessTest, EquivalentTest, DifferentTest], Field(discriminator="kind")]


class PopulationFilter(BaseModel):
    model_config = _STRICT
    path: PathStr  # type: ignore[valid-type]
    equals: Any = None
    in_: list[Any] = Field(default=None, alias="in")  # type: ignore[assignment,arg-type]
    exists: bool = None  # type: ignore[assignment]

    @model_validator(mode="after")
    def _one(self) -> "PopulationFilter":
        given = [k for k in ("equals", "in_", "exists") if k in self.model_fields_set]
        if len(given) != 1:
            raise PydanticCustomError("custom", "give exactly one of equals, in, exists")
        return self


class PopulationSplit(BaseModel):
    model_config = _STRICT
    salt: NonEmpty
    holdout: float = Field(gt=0, lt=1)
    use: Literal["holdout", "dev"]


class Population(BaseModel):
    model_config = _STRICT
    #: In words: what the sample is drawn from. A claim is only ever about this.
    describe: NonEmpty
    #: The INDEPENDENT unit (`meta.user`, `meta.recipe`). Rows sharing one are
    #: resampled together and land in the same split. Default: `row.unit`, else `row.id`.
    unit: PathStr = None  # type: ignore[valid-type]
    #: All must hold.
    where: list[PopulationFilter] = Field(default_factory=list)
    #: A deterministic split by unit. The author iterates on `dev`; the claim is
    #: judged on `holdout`, which it never saw.
    split: PopulationSplit = None  # type: ignore[assignment]


class Plan(BaseModel):
    model_config = _STRICT
    #: The smallest effect worth detecting — the distance from the bound that matters.
    minEffect: float = Field(default=None, gt=0)  # type: ignore[assignment]
    power: float = Field(default=0.8, gt=0.5, lt=1)
    #: Per-unit SD of the metric (of the per-pair difference when paired), for planning before data.
    sd: float = Field(default=None, gt=0)  # type: ignore[assignment]
    #: Expected level of a binary metric (the control's, for a difference), for planning before data.
    baseline: float = Field(default=None, ge=0, le=1)  # type: ignore[assignment]
    #: Planned units (per arm when unpaired). A verdict before it is reached is marked `final: false`.
    n: JsInt = Field(default=None, gt=0)  # type: ignore[assignment]


class Guardrail(BaseModel):
    model_config = _STRICT
    name: NonEmpty
    metric: MetricSpec
    #: Default: the primary estimand.
    estimand: Estimand = None  # type: ignore[assignment]
    test: ClaimTest
    alpha: float = Field(default=None, gt=0, lt=0.5)  # type: ignore[assignment]


class Bootstrap(BaseModel):
    model_config = _STRICT
    resamples: JsInt = Field(default=2000, ge=200, le=100_000)
    #: Default: the hypothesis id.
    seed: str = None  # type: ignore[assignment]


class Segment(BaseModel):
    model_config = _STRICT
    name: NonEmpty
    by: PathStr  # type: ignore[valid-type]


IntervalMethod = Literal["auto", "wilson", "newcombe", "newcombe_paired", "normal", "bootstrap"]


class HypothesisSpec(BaseModel):
    model_config = _STRICT
    id: SlugStr  # type: ignore[valid-type]
    #: The assertion in one plain sentence.
    claim: str = Field(min_length=10)
    rationale: str = None  # type: ignore[assignment]
    population: Population
    metric: MetricSpec
    estimand: Estimand
    test: ClaimTest
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    #: `auto`: Wilson for a binary level, Newcombe for an unpaired binary
    #: difference, Newcombe's paired interval for a paired binary difference,
    #: the cluster bootstrap for everything else (and for any clustered sample).
    interval: IntervalMethod = "auto"
    bootstrap: Bootstrap = Field(default_factory=Bootstrap)
    plan: Plan = Field(default_factory=Plan)
    guardrails: list[Guardrail] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)


hypothesis_spec_schema: TypeAdapter[HypothesisSpec] = TypeAdapter(HypothesisSpec)


def hypothesis_json_schema() -> dict[str, Any]:
    """What a model authors against."""
    return hypothesis_spec_schema.json_schema(mode="validation")


_TAGS = {
    "estimand": {"level", "difference"},
    "test": {"greater", "less", "equivalent", "different"},
}


def _problems(err: ValidationError) -> list[str]:
    """`path: message` like zod, without pydantic's discriminator tag in the path."""
    out: list[str] = []
    for issue in err.errors():
        loc = list(issue.get("loc", ()))
        kept: list[str] = []
        for i, part in enumerate(loc):
            if i > 0 and isinstance(loc[i - 1], str) and part in _TAGS.get(str(loc[i - 1]), ()):
                continue
            kept.append(str(part))
        out.append(f"{'.'.join(kept) if kept else '(root)'}: {issue.get('msg', 'invalid')}")
    return out


def _dump(v: Any) -> Any:
    """
    A parsed model as the JSON zod would have produced: defaults applied, wire
    (alias) keys, and an absent optional DROPPED rather than written as null.
    """
    if isinstance(v, BaseModel):
        out: dict[str, Any] = {}
        fields_set = v.model_fields_set
        for name, info in type(v).model_fields.items():
            x = getattr(v, name)
            if x is None and name not in fields_set:
                continue
            out[info.alias or name] = _dump(x)
        return out
    if isinstance(v, (list, tuple)):
        return [_dump(x) for x in v]
    if isinstance(v, dict):
        return {k: _dump(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CompiledMetric:
    #: `exact_match@1`, or `expr~<hash>`.
    label: str
    scope: str
    binary: bool
    needs_truth: bool
    direction: str
    observe: Callable[[ObservationRow], Observation]
    #: Row scope: one score per row, None = unscorable.
    score_rows: Callable[[Sequence[ObservationRow]], Awaitable[list[Optional[float]]]]
    #: Sample scope: the metric over a set of observations.
    compute: Callable[[Sequence[Observation]], Optional[float]]


@dataclass(kw_only=True)
class CompiledClaim:
    name: str
    metric: CompiledMetric
    estimand: Any
    test: Any
    alpha: float
    interval: str


@dataclass(kw_only=True)
class CompiledHypothesis:
    spec: HypothesisSpec
    #: Over the spec as parsed (defaults applied) and every metric's parsed args.
    hash: str
    primary: CompiledClaim
    guardrails: list[CompiledClaim]


@dataclass(kw_only=True)
class CompileHypothesisResult:
    ok: bool
    hypothesis: Optional[CompiledHypothesis] = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _row_view(row: ObservationRow) -> dict[str, Any]:
    """The view every path reads."""
    return {
        "id": row.id,
        "arm": row.arm,
        "unit": row.unit,
        "prediction": row.prediction,
        "truth": row.truth,
        "recordedAt": row.recorded_at,
        "meta": row.meta if row.meta is not None else {},
    }


def _read_path(row: ObservationRow, path: str) -> Any:
    return pick_path(_row_view(row), path)


def _key_of(v: Any) -> str:
    if v is MISSING:
        return '"__undefined__"'
    return v if isinstance(v, str) else stable_stringify(v)


EXPR_ROOTS = {"prediction", "truth", "row"}
_TRUTH_WORD = re.compile(r"\btruth\b", re.ASCII)


def _sanitize(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return v
    return None


def _compile_metric(
    spec: MetricSpec,
    where: str,
    metrics: Mapping[str, Any],
    engine: Optional[ExpressionEngine],
    problems: list[str],
) -> Optional[tuple[CompiledMetric, Any]]:
    def observe(row: ObservationRow) -> Observation:
        return Observation(prediction=_read_path(row, spec.prediction), truth=_read_path(row, spec.truth), row=row)

    if spec.ref is not None:
        m = metrics.get(spec.ref)
        if m is None:
            problems.append(f"{where}.ref: unknown metric {spec.ref} (known: {', '.join(sorted(metrics.keys()))})")
            return None
        try:
            args = as_schema(m.args).parse(spec.args)
        except SchemaError as err:
            for p in err.problems:
                problems.append(f"{where}.args.{p}")
            return None
        except Exception as err:  # noqa: BLE001 - a host schema with its own parse()
            problems.append(f"{where}.args.(root): {err}")
            return None

        async def score_rows(rows: Sequence[ObservationRow]) -> list[Optional[float]]:
            if m.scope == "row":
                return [_sanitize(m.score(observe(r), args)) for r in rows]
            return [None for _ in rows]

        def compute(os_: Sequence[Observation]) -> Optional[float]:
            return _sanitize(m.compute(os_, args)) if m.scope == "sample" else None

        metric = CompiledMetric(
            label=spec.ref,
            scope=m.scope,
            binary=m.binary,
            needs_truth=m.needs_truth,
            direction=m.direction,
            observe=observe,
            score_rows=score_rows,
            compute=compute,
        )
        return metric, {"ref": spec.ref, "args": _dump(args), "prediction": spec.prediction, "truth": spec.truth}

    # expr
    source: str = spec.expr  # type: ignore[assignment]
    if engine is None:
        problems.append(f"{where}.expr: an expression metric needs an ExpressionEngine (pass options.engine)")
        return None
    try:
        compiled: CompiledExpression = engine.compile(source)
    except Exception as err:  # noqa: BLE001
        problems.append(f"{where}.expr: {err}")
        return None
    roots_fn = getattr(compiled, "roots", None)
    roots = (roots_fn() if callable(roots_fn) else None) or []
    bad = [r for r in roots if r not in EXPR_ROOTS]
    if bad:
        problems.append(f"{where}.expr reads {', '.join(bad)} — an expression sees only prediction, truth, row")
        return None
    # An expression that reads truth cannot score a row whose truth is unknown:
    # JSONata would happily evaluate `x in undefined` to false and count a miss.
    needs_truth = "truth" in roots or bool(_TRUTH_WORD.search(source))

    async def score_expr(rows: Sequence[ObservationRow]) -> list[Optional[float]]:
        out: list[Optional[float]] = []
        for r in rows:
            o = observe(r)
            if needs_truth and o.truth is None:
                out.append(None)
                continue
            try:
                v = await maybe_await(compiled.evaluate({"prediction": o.prediction, "truth": o.truth, "row": _row_view(r)}, {}))
                out.append(_sanitize((1 if v else 0) if isinstance(v, bool) else v))
            except Exception:  # noqa: BLE001 - an expression that fails on a row leaves it unscored
                out.append(None)
        return out

    metric = CompiledMetric(
        label=f"expr~{short_hash(source)[:8]}",
        scope="row",
        binary=spec.binary if spec.binary is not None else False,
        needs_truth=needs_truth,
        direction=spec.direction if spec.direction is not None else "either",
        observe=observe,
        score_rows=score_expr,
        compute=lambda _os: None,
    )
    return metric, {
        "expr": source,
        "binary": spec.binary if spec.binary is not None else False,
        "prediction": spec.prediction,
        "truth": spec.truth,
    }


def _check_claim(claim: CompiledClaim, where: str, problems: list[str], warnings: list[str]) -> None:
    metric, estimand, test, interval = claim.metric, claim.estimand, claim.test, claim.interval
    if metric.scope == "sample" and interval not in ("auto", "bootstrap"):
        problems.append(f"{where}: {metric.label} is a sample metric — only a bootstrap interval applies")
    if interval == "wilson" and not (estimand.kind == "level" and metric.binary):
        problems.append(f"{where}: wilson needs a binary metric and a level estimand")
    if interval == "newcombe" and not (estimand.kind == "difference" and not estimand.paired and metric.binary):
        problems.append(f"{where}: newcombe needs a binary metric and an UNPAIRED difference")
    if interval == "newcombe_paired" and not (
        estimand.kind == "difference" and estimand.paired and metric.binary and metric.scope == "row"
    ):
        problems.append(f"{where}: newcombe_paired needs a binary row metric and a PAIRED difference")
    if metric.binary:
        lo, hi = (0, 1) if estimand.kind == "level" else (-1, 1)
        if test.kind in ("greater", "less"):
            bounds = [test.than]
        elif test.kind == "different":
            bounds = [test.from_]
        else:
            bounds = [test.around]
        for b in bounds:
            if b < lo or b > hi:
                problems.append(
                    f"{where}.test: bound {js_number(b)} is outside [{lo}, {hi}] for a rate — the claim is decided before any data"
                )
    worse = (metric.direction == "lower" and test.kind == "greater") or (metric.direction == "higher" and test.kind == "less")
    if worse and estimand.kind == "difference":
        warnings.append(
            f'{where}: {metric.label} is {metric.direction}-is-better, so a "{test.kind}" difference claims the treatment is WORSE — intended?'
        )


def compile_hypothesis(
    input: Any,
    metrics: Optional[Mapping[str, Any]] = None,
    engine: Optional[ExpressionEngine] = None,
) -> CompileHypothesisResult:
    """
    Parse and check a hypothesis. Never raises: every problem comes back at once
    so an author can repair them in one round. Warnings do not block.

    `metrics` defaults to `BUILTIN_METRICS` (pass `{**BUILTIN_METRICS, **yours}` to
    extend). `engine` is required only for `expr` metrics (`jsonata_engine()`).
    """
    warnings: list[str] = []
    try:
        spec = hypothesis_spec_schema.validate_python(input)
    except ValidationError as err:
        return CompileHypothesisResult(ok=False, problems=_problems(err), warnings=warnings)
    registry = metrics if metrics is not None else BUILTIN_METRICS
    problems: list[str] = []

    primary_metric = _compile_metric(spec.metric, "metric", registry, engine, problems)
    guardrail_metrics = [
        _compile_metric(g.metric, f"guardrails.{i}.metric", registry, engine, problems) for i, g in enumerate(spec.guardrails)
    ]

    names: set[str] = set()
    for i, g in enumerate(spec.guardrails):
        if g.name in names:
            problems.append(f'guardrails.{i}.name: duplicate "{g.name}"')
        names.add(g.name)
    seg_names: set[str] = set()
    for i, s in enumerate(spec.segments):
        if s.name in seg_names:
            problems.append(f'segments.{i}.name: duplicate "{s.name}"')
        seg_names.add(s.name)

    if primary_metric is None or any(g is None for g in guardrail_metrics):
        return CompileHypothesisResult(ok=False, problems=problems, warnings=warnings)

    primary = CompiledClaim(
        name="primary",
        metric=primary_metric[0],
        estimand=spec.estimand,
        test=spec.test,
        alpha=spec.alpha,
        interval=spec.interval,
    )
    _check_claim(primary, "primary", problems, warnings)
    guardrails: list[CompiledClaim] = []
    for i, g in enumerate(spec.guardrails):
        claim = CompiledClaim(
            name=g.name,
            metric=guardrail_metrics[i][0],  # type: ignore[index]
            estimand=g.estimand if g.estimand is not None else spec.estimand,
            test=g.test,
            alpha=g.alpha if g.alpha is not None else spec.alpha,
            interval="auto",
        )
        _check_claim(claim, f"guardrails.{i}", problems, warnings)
        guardrails.append(claim)

    if spec.plan.baseline is not None and not primary.metric.binary:
        warnings.append("plan.baseline only informs planning for a binary metric; give plan.sd instead")
    if problems:
        return CompileHypothesisResult(ok=False, problems=problems, warnings=warnings)

    dumped = _dump(spec)
    dumped.pop("metric", None)
    dumped["guardrails"] = [{k: v for k, v in g.items() if k != "metric"} for g in dumped["guardrails"]]
    h = short_hash(
        {
            "format": HYPOTHESIS_FORMAT,
            "spec": dumped,
            "metric": primary_metric[1],
            "guardrails": [g[1] for g in guardrail_metrics],  # type: ignore[index]
        }
    )
    return CompileHypothesisResult(
        ok=True,
        hypothesis=CompiledHypothesis(spec=spec, hash=h, primary=primary, guardrails=guardrails),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Preregistration
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HypothesisLock:
    """
    Freezes a hypothesis before its data exists. Persist it where the claim's
    readers will look (the host's job — a Langfuse dataset's metadata, a row).
    """

    hypothesis: str
    hash: str
    registered_at: str
    spec: HypothesisSpec


def _ms_of(at: Optional[datetime]) -> float:
    if at is None:
        return now_ms()
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.timestamp() * 1000


def lock_hypothesis(h: CompiledHypothesis, at: Optional[datetime] = None) -> HypothesisLock:
    return HypothesisLock(hypothesis=h.spec.id, hash=h.hash, registered_at=iso_from_ms(_ms_of(at)), spec=h.spec)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def _unit_of(row: ObservationRow, path: Optional[str]) -> str:
    if path:
        v = _read_path(row, path)
        return row.id if v is None else _key_of(v)
    return row.unit if row.unit is not None else row.id


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, Mapping) else getattr(obj, key)


def split_of(unit: str, split: Any) -> str:
    """Deterministic, by unit — a unit's rows always land on the same side. `split` has `salt` and `holdout`."""
    return "holdout" if hash_to_unit(f"{_get(split, 'salt')}:{unit}") < _get(split, "holdout") else "dev"


@dataclass(kw_only=True)
class RowPartition:
    dev: list[ObservationRow]
    holdout: list[ObservationRow]


def partition_rows(spec: Any, rows: Sequence[ObservationRow]) -> RowPartition:
    """What an author may iterate on (`dev`) vs. what judges it (`holdout`)."""
    split = spec.population.split
    if split is None:
        return RowPartition(dev=list(rows), holdout=[])
    out = RowPartition(dev=[], holdout=[])
    for r in rows:
        (out.holdout if split_of(_unit_of(r, spec.population.unit), split) == "holdout" else out.dev).append(r)
    return out


def _passes(row: ObservationRow, f: PopulationFilter) -> bool:
    v = _pick(_row_view(row), f.path)
    given = f.model_fields_set
    if "exists" in given:
        return (v is not MISSING and v is not None) == f.exists
    if "in_" in given:
        return any(_key_of(x) == _key_of(v) for x in f.in_)
    return _key_of(f.equals) == _key_of(v)


# ---------------------------------------------------------------------------
# Evaluate one claim
# ---------------------------------------------------------------------------

ClaimStatus = Literal["supported", "refuted", "inconclusive"]


@dataclass(kw_only=True)
class ClaimInterval:
    lo: float
    hi: float
    confidence: float
    method: str


@dataclass(kw_only=True)
class ClaimCounts:
    #: Rows with a score, per arm.
    scored: dict[str, int]
    unscored: int
    #: Independent units in the analysis (pairs' units when paired).
    units: int
    #: Rows of either arm with no partner (paired only).
    unpaired: int


@dataclass(kw_only=True)
class ClaimPower:
    se: Optional[float]
    #: The smallest true effect this sample detects with `plan.power`.
    detectable_effect: Optional[float]
    #: Units needed (per arm when unpaired) for `plan.minEffect` — scaled from THIS sample's se.
    required_n: Optional[int]
    underpowered: bool


@dataclass(kw_only=True)
class ClaimResult:
    name: str
    metric: str
    status: str
    #: One plain sentence: which interval, against what, why this status.
    reason: str
    estimate: Optional[float]
    interval: Optional[ClaimInterval]
    #: Normal approximation from the interval's implied standard error. The interval decides; this is for readers.
    p_value: Optional[float]
    n: ClaimCounts
    power: ClaimPower
    #: Method-specific extras: McNemar's discordant counts and exact p, undefined bootstrap draws.
    extra: dict[str, float]
    warnings: list[str]


class _Scored:
    __slots__ = ("row", "unit", "score", "obs")

    def __init__(self, row: ObservationRow, unit: str, score: Optional[float], obs: Observation):
        self.row = row
        self.unit = unit
        self.score = score
        self.obs = obs


class _Pair:
    __slots__ = ("t", "c", "unit")

    def __init__(self, t: _Scored, c: _Scored, unit: str):
        self.t = t
        self.c = c
        self.unit = unit


def _fmt(x: float) -> str:
    return to_fixed(x, 1) if abs(x) >= 100 else to_fixed(x, 3) if abs(x) >= 1 else to_precision(x, 3)


def _describe_test(t: Any) -> str:
    if t.kind == "greater":
        return f"> {_fmt(t.than)}"
    if t.kind == "less":
        return f"< {_fmt(t.than)}"
    if t.kind == "equivalent":
        return f"within ±{_fmt(t.within)} of {_fmt(t.around)}"
    return f"≠ {_fmt(t.from_)}"


def _decide(t: Any, lo: float, hi: float) -> str:
    if t.kind == "greater":
        return "supported" if lo > t.than else "refuted" if hi < t.than else "inconclusive"
    if t.kind == "less":
        return "supported" if hi < t.than else "refuted" if lo > t.than else "inconclusive"
    if t.kind == "equivalent":
        a = t.around - t.within
        b = t.around + t.within
        return "supported" if lo > a and hi < b else "refuted" if hi < a or lo > b else "inconclusive"
    return "supported" if lo > t.from_ or hi < t.from_ else "inconclusive"


def _p_value_for(t: Any, est: float, se: float) -> Optional[float]:
    if not (se > 0):
        return None
    if t.kind == "greater":
        return p_upper((est - t.than) / se)
    if t.kind == "less":
        return p_upper((t.than - est) / se)
    if t.kind == "different":
        return min(1, 2 * p_upper(abs(est - t.from_) / se))
    return max(p_upper((est - (t.around - t.within)) / se), p_upper((t.around + t.within - est) / se))


def _group_by_unit(items: Sequence[Any], unit: Callable[[Any], str]) -> list[list[Any]]:
    m: dict[str, list[Any]] = {}
    for it in items:
        k = unit(it)
        lst = m.get(k)
        if lst is None:
            m[k] = [it]
        else:
            lst.append(it)
    return list(m.values())


def _distinct(xs: Sequence[str]) -> int:
    return len(set(xs))


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


@dataclass(kw_only=True)
class _EvalContext:
    unit_path: Optional[str]
    resamples: int
    seed: str
    power: float
    min_effect: Optional[float]


def _js_round(x: float) -> float:
    """`Math.round`: halves go up."""
    return math.floor(x + 0.5)


async def _evaluate_claim(claim: CompiledClaim, rows: Sequence[ObservationRow], ctx: _EvalContext) -> ClaimResult:
    metric, estimand, test, alpha = claim.metric, claim.estimand, claim.test, claim.alpha
    warnings: list[str] = []
    confidence = 1 - alpha if test.kind == "different" else 1 - 2 * alpha
    alpha_per_side = alpha / 2 if test.kind == "different" else alpha

    scores = await metric.score_rows(rows)
    all_: list[_Scored] = []
    for i, row in enumerate(rows):
        obs = metric.observe(row)
        score = scores[i] if i < len(scores) else None
        if metric.scope == "sample":
            missing = obs.prediction is None or (metric.needs_truth and obs.truth is None)
            score = None if missing else 0  # a placeholder: "this row takes part"
        all_.append(_Scored(row, _unit_of(row, ctx.unit_path), score, obs))
    usable = [s for s in all_ if s.score is not None]
    unscored = len(all_) - len(usable)
    if unscored > 0:
        warnings.append(f"{unscored} row(s) could not be scored by {metric.label} and were left out")

    if metric.scope == "row":

        def stat(items: Sequence[_Scored]) -> Optional[float]:
            if len(items) == 0:
                return None
            return mean([s.score for s in items])  # type: ignore[misc]

    else:

        def stat(items: Sequence[_Scored]) -> Optional[float]:
            if len(items) == 0:
                return None
            return metric.compute([s.obs for s in items])

    extra: dict[str, float] = {}

    def empty(reason: str, scored: dict[str, int], units: int, unpaired: int = 0) -> ClaimResult:
        return ClaimResult(
            name=claim.name,
            metric=metric.label,
            status="inconclusive",
            reason=reason,
            estimate=None,
            interval=None,
            p_value=None,
            n=ClaimCounts(scored=scored, unscored=unscored, units=units, unpaired=unpaired),
            power=ClaimPower(se=None, detectable_effect=None, required_n=None, underpowered=True),
            extra=extra,
            warnings=warnings,
        )

    # --- shape the sample ------------------------------------------------------
    unpaired = 0
    items: list[_Scored] = []
    t_items: list[_Scored] = []
    c_items: list[_Scored] = []
    pairs: list[_Pair] = []
    if estimand.kind == "level":
        shape = "level"
        items = usable if estimand.arm is None else [s for s in usable if s.row.arm == estimand.arm]
        if estimand.arm is None:
            arms = _distinct([s.row.arm if s.row.arm is not None else "" for s in usable])
            if arms > 1:
                warnings.append(f"a level estimand with no arm pooled {arms} arms")
        scored = {estimand.arm if estimand.arm is not None else "all": len(items)}
        units = _distinct([s.unit for s in items])
    elif not estimand.paired:
        shape = "unpaired"
        t_items = [s for s in usable if s.row.arm == estimand.treatment]
        c_items = [s for s in usable if s.row.arm == estimand.control]
        scored = {estimand.treatment: len(t_items), estimand.control: len(c_items)}
        units = min(_distinct([s.unit for s in t_items]), _distinct([s.unit for s in c_items]))
    else:
        shape = "paired"

        def by_id(arm: str) -> dict[str, _Scored]:
            m: dict[str, _Scored] = {}
            dup = 0
            for s in usable:
                if s.row.arm != arm:
                    continue
                if s.row.id in m:
                    dup += 1
                else:
                    m[s.row.id] = s
            if dup:
                warnings.append(f"{dup} duplicate id(s) in arm {arm}; the first of each was paired")
            return m

        t_map = by_id(estimand.treatment)
        c_map = by_id(estimand.control)
        for id_, ts in t_map.items():
            cs = c_map.get(id_)
            if cs is not None:
                pairs.append(_Pair(ts, cs, ts.unit))
        unpaired = len(t_map) + len(c_map) - 2 * len(pairs)
        if unpaired:
            warnings.append(f"{unpaired} row(s) had no partner in the other arm and were left out")
        scored = {estimand.treatment: len(pairs), estimand.control: len(pairs)}
        units = _distinct([p.unit for p in pairs])

    if units < 2:
        return empty(f"not enough data: {units} independent unit(s)", scored, units, unpaired)

    def paired_stat(ps: Sequence[_Pair]) -> Optional[float]:
        if metric.scope == "row":
            return mean([p.t.score - p.c.score for p in ps])  # type: ignore[operator]
        return _diff(stat([p.t for p in ps]), stat([p.c for p in ps]))

    # --- the point estimate -------------------------------------------------------
    if shape == "level":
        estimate = stat(items)
    elif shape == "unpaired":
        estimate = _diff(stat(t_items), stat(c_items))
    else:
        estimate = paired_stat(pairs)
    if estimate is None:
        return empty(f"{metric.label} is undefined on this sample", scored, units, unpaired)

    # --- the method -------------------------------------------------------------
    row_count = len(items) if shape == "level" else len(t_items) + len(c_items) if shape == "unpaired" else len(pairs)
    if shape == "unpaired":
        clustered = _distinct([s.unit for s in t_items]) < len(t_items) or _distinct([s.unit for s in c_items]) < len(c_items)
    else:
        clustered = units < row_count
    if claim.interval != "auto":
        method = claim.interval
    elif metric.scope == "sample" or not metric.binary or clustered:
        method = "bootstrap"
    else:
        method = "wilson" if shape == "level" else "newcombe" if shape == "unpaired" else "newcombe_paired"
    if method in ("wilson", "newcombe", "newcombe_paired") and clustered:
        warnings.append(f"{method} assumes independent rows, but {row_count} rows share {units} units — the interval is too narrow")

    z = z_for(confidence)

    if method == "wilson" and shape == "level":
        x = 0.0
        for s in items:
            x += s.score  # type: ignore[operator]
        iv = wilson_interval(x, len(items), confidence)
        lo, hi = iv.lo, iv.hi
    elif method == "newcombe" and shape == "unpaired":
        x1 = 0.0
        for s in t_items:
            x1 += s.score  # type: ignore[operator]
        x2 = 0.0
        for s in c_items:
            x2 += s.score  # type: ignore[operator]
        iv = newcombe_interval(x1, len(t_items), x2, len(c_items), confidence)
        lo, hi = iv.lo, iv.hi
    elif method == "newcombe_paired" and shape == "paired":
        cell = {"a": 0, "b": 0, "c": 0, "d": 0}
        for p in pairs:
            cell[("a" if p.c.score == 1 else "b") if p.t.score == 1 else ("c" if p.c.score == 1 else "d")] += 1
        iv = newcombe_paired_interval(cell["a"], cell["b"], cell["c"], cell["d"], confidence)
        lo, hi = iv.lo, iv.hi
    elif method == "normal":
        if shape == "level":
            se_n = cluster_robust_se([s.score for s in items], [s.unit for s in items])  # type: ignore[misc]
        elif shape == "unpaired":
            a = cluster_robust_se([s.score for s in t_items], [s.unit for s in t_items])  # type: ignore[misc]
            b = cluster_robust_se([s.score for s in c_items], [s.unit for s in c_items])  # type: ignore[misc]
            se_n = math.sqrt(a * a + b * b)
        else:
            se_n = cluster_robust_se([p.t.score - p.c.score for p in pairs], [p.unit for p in pairs])  # type: ignore[operator]
        if not math.isfinite(se_n):
            return empty("not enough independent units for a standard error", scored, units, unpaired)
        lo = estimate - z * se_n
        hi = estimate + z * se_n
        if units < 30:
            warnings.append(f"normal interval on {units} units — the bootstrap or a t-interval would be safer")
    else:
        method = "bootstrap"
        random = seeded_random(f"{ctx.seed}:{claim.name}")
        if shape == "level":
            boot = cluster_bootstrap([_group_by_unit(items, lambda s: s.unit)], lambda g: stat(g[0]), ctx.resamples, random)
        elif shape == "unpaired":
            boot = cluster_bootstrap(
                [_group_by_unit(t_items, lambda s: s.unit), _group_by_unit(c_items, lambda s: s.unit)],
                lambda g: _diff(stat(g[0]), stat(g[1])),
                ctx.resamples,
                random,
            )
        else:
            boot = cluster_bootstrap([_group_by_unit(pairs, lambda p: p.unit)], lambda g: paired_stat(g[0]), ctx.resamples, random)
        if len(boot.draws) < ctx.resamples / 2:
            return empty(f"{metric.label} was undefined on most bootstrap resamples", scored, units, unpaired)
        if boot.undefined_draws > 0:
            extra["bootstrap_undefined"] = boot.undefined_draws
            if boot.undefined_draws > ctx.resamples * 0.05:
                warnings.append(
                    f"{boot.undefined_draws}/{ctx.resamples} resamples left {metric.label} undefined — the interval is conditional on the rest"
                )
        iv = percentile_interval(boot.draws, confidence)
        lo, hi = iv.lo, iv.hi
        if hi - lo == 0:
            warnings.append(
                "the bootstrap interval has zero width — the sample shows no variation, which understates the uncertainty (collect more, or use wilson for a rate)"
            )
        if units < 20:
            warnings.append(f"bootstrap on {units} units — percentile intervals run narrow below ~20")

    if shape == "paired" and metric.binary and metric.scope == "row":
        b_ = 0
        c_ = 0
        for p in pairs:
            if p.t.score == 1 and p.c.score == 0:
                b_ += 1
            elif p.t.score == 0 and p.c.score == 1:
                c_ += 1
        extra["mcnemar_only_treatment"] = b_
        extra["mcnemar_only_control"] = c_
        extra["mcnemar_p"] = mcnemar_exact(b_, c_)

    # --- verdict + power ----------------------------------------------------------
    status = _decide(test, lo, hi)
    se = (hi - lo) / (2 * z)
    detectable = detectable_effect(se, alpha_per_side, ctx.power) if se > 0 else None
    min_effect = ctx.min_effect if ctx.min_effect is not None else (test.within if test.kind == "equivalent" else None)
    n_now = units
    required_n = math.ceil(n_now * (detectable / min_effect) ** 2) if min_effect is not None and detectable is not None else None
    underpowered = status == "inconclusive" and (True if required_n is None else n_now < required_n)

    pct = f"{js_number(_js_round(confidence * 1000) / 10)}%"
    where = f"{pct} {method} interval [{_fmt(lo)}, {_fmt(hi)}] (n={n_now} units)"
    if status == "supported":
        reason = f"{where} lies entirely {_describe_test(test)}"
    elif status == "refuted":
        reason = f"{where} lies entirely outside {_describe_test(test)}"
    else:
        reason = f"{where} straddles {_describe_test(test)}"
        if required_n is not None and n_now < required_n:
            reason += f" — about {required_n} units would detect an effect of {_fmt(min_effect)}"  # type: ignore[arg-type]
        elif test.kind == "different":
            reason += " — to claim NO difference, test equivalence"

    return ClaimResult(
        name=claim.name,
        metric=metric.label,
        status=status,
        reason=reason,
        estimate=estimate,
        interval=ClaimInterval(lo=lo, hi=hi, confidence=confidence, method=method),
        p_value=_p_value_for(test, estimate, se),
        n=ClaimCounts(scored=scored, unscored=unscored, units=units, unpaired=unpaired),
        power=ClaimPower(se=se, detectable_effect=detectable, required_n=required_n, underpowered=underpowered),
        extra=extra,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Evaluate a hypothesis
# ---------------------------------------------------------------------------

HypothesisStatus = Literal["supported", "refuted", "inconclusive", "invalid"]


@dataclass(kw_only=True)
class SegmentResult:
    name: str
    value: str
    result: ClaimResult


@dataclass(kw_only=True)
class Excluded:
    before_lock: int = 0
    filtered: int = 0
    outside_split: int = 0


@dataclass(kw_only=True)
class HypothesisVerdict:
    hypothesis: str
    hash: str
    claim: str
    status: str
    reason: str
    #: Locked before the data it was judged on. False = exploratory, whatever it says.
    preregistered: bool
    #: False while `plan.n` has not been reached: this read is a peek.
    final: bool = True
    primary: Optional[ClaimResult] = None
    guardrails: list[ClaimResult] = field(default_factory=list)
    #: Preregistered breakdowns. Descriptive: never multiplicity-corrected, never a rescue.
    segments: list[SegmentResult] = field(default_factory=list)
    excluded: Excluded = field(default_factory=Excluded)
    #: Rows with no `recorded_at` — kept, but the lock cannot vouch for them.
    undated: int = 0
    warnings: list[str] = field(default_factory=list)
    stats_version: str = STATS_VERSION
    evaluated_at: str = ""


async def evaluate_hypothesis(
    h: CompiledHypothesis,
    rows: Sequence[ObservationRow],
    lock: Optional[HypothesisLock] = None,
    now: Optional[datetime] = None,
) -> HypothesisVerdict:
    """`lock`: the lock taken before collection. Without it the verdict is exploratory."""
    spec = h.spec
    evaluated_at = iso_from_ms(_ms_of(now))
    warnings: list[str] = []

    def verdict(**partial: Any) -> HypothesisVerdict:
        base: dict[str, Any] = {
            "hypothesis": spec.id,
            "hash": h.hash,
            "claim": spec.claim,
            "preregistered": lock is not None,
            "warnings": warnings,
            "evaluated_at": evaluated_at,
        }
        base.update(partial)
        return HypothesisVerdict(**base)

    if lock is not None and lock.hash != h.hash:
        return verdict(
            status="invalid",
            reason=f"the hypothesis changed after it was locked (locked {lock.hash}, now {h.hash}) — lock the new version and collect fresh data",
            preregistered=False,
        )
    if lock is None:
        warnings.append("no lock: this verdict is EXPLORATORY — lock the hypothesis before collecting data to make it confirmatory")

    # --- population ---------------------------------------------------------------
    excluded = Excluded()
    undated = 0
    locked_at = ms_from_iso(lock.registered_at) if lock is not None else None
    population: list[ObservationRow] = []
    split = spec.population.split
    for row in rows:
        if locked_at is not None:
            if not row.recorded_at:
                undated += 1
            elif ms_from_iso(row.recorded_at) < locked_at:
                excluded.before_lock += 1
                continue
        if not all(_passes(row, f) for f in spec.population.where):
            excluded.filtered += 1
            continue
        if split is not None and split_of(_unit_of(row, spec.population.unit), split) != split.use:
            excluded.outside_split += 1
            continue
        population.append(row)
    if undated:
        warnings.append(f"{undated} row(s) carry no recordedAt — the lock cannot vouch that they came after it")
    if excluded.before_lock:
        warnings.append(f"{excluded.before_lock} row(s) predate the lock and were left out")
    if split is not None and split.use == "dev":
        warnings.append("judged on the DEV split — fine for iterating, never for the confirmatory verdict")

    ctx = _EvalContext(
        unit_path=spec.population.unit,
        resamples=spec.bootstrap.resamples,
        seed=spec.bootstrap.seed if spec.bootstrap.seed is not None else spec.id,
        power=spec.plan.power,
        min_effect=spec.plan.minEffect,
    )

    primary = await _evaluate_claim(h.primary, population, ctx)
    guardrails: list[ClaimResult] = []
    for g in h.guardrails:
        guardrails.append(await _evaluate_claim(g, population, dataclasses.replace(ctx, min_effect=None)))

    segments: list[SegmentResult] = []
    for seg in spec.segments:
        groups: dict[str, list[ObservationRow]] = {}
        for r in population:
            v = _read_path(r, seg.by)
            k = "(none)" if v is None else _key_of(v)
            groups.setdefault(k, []).append(r)
        for value, members in sorted(groups.items(), key=lambda kv: kv[0]):
            result = await _evaluate_claim(dataclasses.replace(h.primary, name=f"segment:{seg.name}={value}"), members, ctx)
            segments.append(SegmentResult(name=seg.name, value=value, result=result))

    final = spec.plan.n is None or primary.n.units >= spec.plan.n
    if not final:
        warnings.append(f"{primary.n.units} of {spec.plan.n} planned units — a verdict before the plan is reached is a peek")

    status = primary.status
    reason = primary.reason
    if primary.status == "supported":
        failed = next((g for g in guardrails if g.status == "refuted"), None)
        open_ = next((g for g in guardrails if g.status == "inconclusive"), None)
        if failed is not None:
            status = "refuted"
            reason = f'primary held ({primary.reason}), but guardrail "{failed.name}" failed: {failed.reason}'
        elif open_ is not None:
            status = "inconclusive"
            reason = f'primary held ({primary.reason}), but guardrail "{open_.name}" is unproven: {open_.reason}'

    return verdict(
        status=status,
        reason=reason,
        final=final,
        primary=primary,
        guardrails=guardrails,
        segments=segments,
        excluded=excluded,
        undated=undated,
    )


# ---------------------------------------------------------------------------
# Planning — before any data
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HypothesisPlan:
    #: Units (per arm for an unpaired difference) the plan needs, or None when it cannot say.
    required_n: Optional[int]
    design: str
    #: What the number assumed, so a reader can dispute it.
    assumptions: list[str]


def plan_hypothesis(h: CompiledHypothesis) -> HypothesisPlan:
    """
    How much data the claim needs, from `plan` alone. Needs `plan.minEffect`
    plus `plan.sd` — or, for a binary metric, `plan.baseline`.
    """
    spec, primary = h.spec, h.primary
    design = "one-sample" if primary.estimand.kind == "level" else "paired" if primary.estimand.paired else "two-sample"
    assumptions: list[str] = []
    effect = spec.plan.minEffect if spec.plan.minEffect is not None else (spec.test.within if spec.test.kind == "equivalent" else None)
    if effect is None:
        return HypothesisPlan(required_n=None, design=design, assumptions=["no plan.minEffect"])
    sd = spec.plan.sd
    if sd is None and primary.metric.binary and spec.plan.baseline is not None:
        p = spec.plan.baseline
        sd = math.sqrt(2 * p * (1 - p)) if design == "paired" else math.sqrt(p * (1 - p))
        assumptions.append(
            f"sd of the paired difference ≈ √(2p(1−p)) at baseline {js_number(p)} — the worst case, uncorrelated arms"
            if design == "paired"
            else f"sd ≈ √(p(1−p)) at baseline {js_number(p)}"
        )
    if sd is None:
        return HypothesisPlan(required_n=None, design=design, assumptions=["no plan.sd (or plan.baseline for a binary metric)"])
    alpha_per_side = spec.alpha / 2 if spec.test.kind == "different" else spec.alpha
    assumptions.append(
        f"normal approximation; α={js_number(spec.alpha)} ({js_number(alpha_per_side)} per side), power {js_number(spec.plan.power)}, effect {js_number(effect)}, sd {js_number(sd)}"
    )
    n = required_sample_size(sd=sd, effect=effect, alpha_per_side=alpha_per_side, power=spec.plan.power, design=design)
    return HypothesisPlan(required_n=n, design=design, assumptions=assumptions)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# From pipeline runs
# ---------------------------------------------------------------------------


def observation_from_run(
    run: Any,
    arm: Optional[str] = None,
    truth: Any = None,
    id: Optional[str] = None,
    unit: Optional[str] = None,
) -> ObservationRow:
    """One run (anything with `.output` and `.record`) as an observation. `meta` carries what guardrails and segments usually read."""
    r = run.record

    def rec(name: str) -> Any:
        return getattr(r, name, None)

    return ObservationRow(
        id=id if id is not None else r.input_hash,
        arm=arm,
        unit=unit,
        prediction=run.output,
        truth=truth,
        recorded_at=rec("created_at"),
        meta={
            "route": rec("fact_route"),
            "confidence": rec("fact_confidence"),
            "cost_usd": rec("cost_usd"),
            "ms": rec("ms"),
            "version": rec("version"),
            "run_key": rec("run_key"),
            "user": rec("user"),
            "tenant": rec("tenant"),
            "group": rec("group"),
            "arms": rec("arms"),
            "features": rec("features"),
        },
    )


def observations_from_shadow(
    shadow: Any,
    truth: Any = None,
    id: Optional[str] = None,
    unit: Optional[str] = None,
    champion: Optional[str] = None,
    challenger: Optional[str] = None,
) -> list[ObservationRow]:
    """A shadow run as a PAIR: one row per side, sharing the champion's id."""
    pair_id = id if id is not None else shadow.champion.record.input_hash
    rows = [
        observation_from_run(
            shadow.champion, arm=champion if champion is not None else "champion", truth=truth, id=pair_id, unit=unit
        )
    ]
    if shadow.challenger is not None:
        rows.append(
            observation_from_run(
                shadow.challenger, arm=challenger if challenger is not None else "challenger", truth=truth, id=pair_id, unit=unit
            )
        )
    return rows
