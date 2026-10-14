"""
Feature discovery (#321): an author proposes questions, a question-answering
port answers them for every row, a small model learns from the answers, and
the model's worst-predicted rows go back to the author for the next round.

After TypeSafe's "Autoresearch feature discovery" cookbook, with three
departures:
  - the learner is LINEAR, so it is pure Python, inspectable, and writable into a
    spec as one expression (.learned): L2 logistic regression for a binary
    target (e.g. "was the cheap branch's answer right?"), ridge regression for a
    numeric one (the cookbook's critic score). A gate needs a calibrated
    probability more than it needs the last point of accuracy, and at a few
    hundred negatives a boosted model memorises;
  - folds are GROUPED (a row's `group`, e.g. the recipe a transformed copy
    came from), so a copy never sits in the fold that judges its parent;
  - every round is scored against `base` columns the host already has for
    free (today's gate inputs). A discovered question has to beat what the
    gate can already read, not a mean-only baseline.

Accept rules follow the cookbook. An add goes in unless its column is flat.
A revise or drop is tried first by refitting, which costs no port calls, and is
kept only if the CV loss drops (log loss, or RMSE for a numeric target). Only dev
rows may reach this function. The host scores its held-out rows once, outside it.

Arithmetic mirrors the TS source operation for operation (plain left-to-right
sums, not Python's compensated `sum()`), so a fit here and a fit there agree to
the last bit wherever the platform's `exp`/`log` agree with V8's.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping, NamedTuple, Optional, Sequence, Union

from ._compat import js_number, maybe_await, to_fixed
from .hash import hash_to_unit

QuestionKind = Literal["intensity", "presence"]
#: binary: labels are 0/1, logistic regression, log loss. numeric: any number, ridge regression, RMSE.
DiscoverTarget = Literal["binary", "numeric"]


# ---------------------------------------------------------------------------
# JS arithmetic helpers (internal)
# ---------------------------------------------------------------------------


def _sum(values: Iterable[float]) -> float:
    """`xs.reduce((a, v) => a + v, 0)`: plain left-to-right, unlike 3.12's compensated `sum()`."""
    a: float = 0
    for v in values:
        a += v
    return a


def _or(v: float, d: float) -> float:
    """JS `v || d` for a number: `0` and `NaN` take the default."""
    return d if v == 0 or v != v else v


def _truthy(v: float) -> bool:
    return not (v == 0 or v != v)


def _div(a: float, b: float) -> float:
    """JS division: `x / 0` is ±Infinity, `0 / 0` is NaN."""
    if b == 0:
        if a == 0 or a != a:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)
    return a / b


def _exp(x: float) -> float:
    """`Math.exp`: overflows to Infinity instead of raising."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def _js_max(*xs: float) -> float:
    """`Math.max`: NaN wins."""
    for x in xs:
        if x != x:
            return math.nan
    return max(xs)


def _js_min(*xs: float) -> float:
    """`Math.min`: NaN wins."""
    for x in xs:
        if x != x:
            return math.nan
    return min(xs)


def _js_round(x: float) -> float:
    """`Math.round`: half rounds toward +Infinity."""
    if x != x or x in (math.inf, -math.inf):
        return x
    r = math.floor(x)
    return r + 1 if x - r >= 0.5 else r


# JS `String.prototype.trim` whitespace: WhiteSpace + LineTerminator.
_JS_WS = "\t\n\v\f\r                  　﻿"


def _js_trim(s: str) -> str:
    return s.strip(_JS_WS)


_ARRAY_INDEX = re.compile(r"(0|[1-9][0-9]*)")


def _js_key_order(keys: Iterable[str]) -> list[str]:
    """`Object.keys` order: array-index keys ascending first, then the rest in insertion order."""
    keys = list(keys)
    idx = [k for k in keys if _ARRAY_INDEX.fullmatch(k) and int(k) < 2**32 - 1]
    rest = [k for k in keys if k not in set(idx)]
    return sorted(idx, key=int) + rest


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DiscoveredQuestion:
    name: str
    kind: QuestionKind
    question: str
    #: Round the current wording came from; the answer cache is keyed on it.
    round: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "question": self.question, "round": self.round}

    @classmethod
    def from_dict(cls, d: Any) -> "DiscoveredQuestion":
        if isinstance(d, DiscoveredQuestion):
            return d
        return cls(name=d["name"], kind=d["kind"], question=d["question"], round=d["round"])


@dataclass(kw_only=True)
class DiscoverAction:
    op: Literal["add", "revise", "drop"]
    #: revise/drop: the existing question's name.
    target: str = ""
    name: str = ""
    kind: QuestionKind = "presence"
    question: str = ""


def discover_action_from_dict(d: Any) -> DiscoverAction:
    """An author port may return `DiscoverAction`s or plain dicts (e.g. parsed tool-call JSON)."""
    if isinstance(d, DiscoverAction):
        return d
    return DiscoverAction(
        op=d.get("op", "add"),
        target=d.get("target") or "",
        name=d.get("name") or "",
        kind=d.get("kind") or "presence",
        question=d.get("question") or "",
    )


@dataclass(kw_only=True)
class DiscoverRow:
    id: str
    #: Rows sharing a group always share a fold.
    group: str
    #: 0 or 1 for a binary target; any finite number for a numeric one.
    label: float
    #: Columns the host already has (named by `base_names`).
    base: list[float]
    #: What the author reads for this row.
    text: str


@dataclass(kw_only=True)
class DiscoverAuthorRequest:
    round: int
    accepted: list[DiscoveredQuestion]
    #: Rendered rows: round 1 spans the labels; later rounds show the worst half and the best half.
    examples: str
    #: The scoreboard: CV history, per-question importance and spread. Empty on round 1.
    feedback: str
    max_actions: int


@dataclass(kw_only=True)
class DiscoverPorts:
    """
    `author(req) -> list[DiscoverAction | dict]`.
    `answer(rows, questions) -> {question name: one probability vector per row, in row order}`:
    intensity = 5 level probabilities, presence = [p(true)]. Hosts should cache by
    (row, question text). Either may be sync or async.
    """

    author: Callable[[DiscoverAuthorRequest], Union[Sequence[Any], Awaitable[Sequence[Any]]]]
    answer: Callable[[list[DiscoverRow], list[DiscoveredQuestion]], Union[Mapping[str, list[list[float]]], Awaitable[Mapping[str, list[list[float]]]]]]


@dataclass(kw_only=True)
class JournalEntry:
    round: int
    what: Literal["add", "flat", "revise", "reject", "drop", "keep", "stale"]
    name: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"round": self.round, "what": self.what, "name": self.name, "note": self.note}

    @classmethod
    def from_dict(cls, d: Any) -> "JournalEntry":
        if isinstance(d, JournalEntry):
            return d
        return cls(round=d["round"], what=d["what"], name=d["name"], note=d["note"])


@dataclass(kw_only=True)
class CvScore:
    #: What the loop minimises: log loss (binary) or RMSE (numeric).
    loss: float
    columns: int
    #: Binary target only.
    log_loss: Optional[float] = None
    auc: Optional[float] = None
    #: Numeric target only.
    rmse: Optional[float] = None
    spearman: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """The TS JSON shape (`logLoss`), key order as TS builds it."""
        out: dict[str, Any] = {"loss": self.loss}
        if self.log_loss is not None:
            out["logLoss"] = self.log_loss
        if self.auc is not None:
            out["auc"] = self.auc
        if self.rmse is not None:
            out["rmse"] = self.rmse
        if self.spearman is not None:
            out["spearman"] = self.spearman
        out["columns"] = self.columns
        return out

    @classmethod
    def from_dict(cls, d: Any) -> "CvScore":
        if isinstance(d, CvScore):
            return d
        return cls(
            loss=d["loss"],
            columns=d["columns"],
            log_loss=d.get("logLoss", d.get("log_loss")),
            auc=d.get("auc"),
            rmse=d.get("rmse"),
            spearman=d.get("spearman"),
        )


@dataclass(kw_only=True)
class DiscoverResult:
    accepted: list[DiscoveredQuestion]
    #: Every answered question id (`name@round`) → per-row probability vectors.
    answers: dict[str, list[list[float]]]
    #: CV score of `base` alone: the bar.
    base_cv: CvScore
    history: list[CvScore]
    snapshots: list[list[DiscoveredQuestion]]
    journal: list[JournalEntry]

    def to_dict(self) -> dict[str, Any]:
        """The TS `DiscoverResult` JSON shape, so a run saved here loads there and back."""
        return {
            "accepted": [q.to_dict() for q in self.accepted],
            "answers": self.answers,
            "baseCv": self.base_cv.to_dict(),
            "history": [h.to_dict() for h in self.history],
            "snapshots": [[q.to_dict() for q in s] for s in self.snapshots],
            "journal": [j.to_dict() for j in self.journal],
        }

    @classmethod
    def from_dict(cls, d: Any) -> "DiscoverResult":
        if isinstance(d, DiscoverResult):
            return d
        return cls(
            accepted=[DiscoveredQuestion.from_dict(q) for q in d["accepted"]],
            answers={k: v for k, v in d["answers"].items()},
            base_cv=CvScore.from_dict(d.get("baseCv", d.get("base_cv"))),
            history=[CvScore.from_dict(h) for h in d["history"]],
            snapshots=[[DiscoveredQuestion.from_dict(q) for q in s] for s in d["snapshots"]],
            journal=[JournalEntry.from_dict(j) for j in d["journal"]],
        )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

INTENSITY_LEVELS: tuple[str, ...] = (
    "Not present at all",
    "Barely present, mentioned once in passing",
    "Present at a moderate level",
    "Present strongly, the text dwells on it",
    "Dominant, the text is largely about this",
)


def question_id(q: Any) -> str:
    """`name@round`. Takes a `DiscoveredQuestion` or a dict with `name` and `round`."""
    if isinstance(q, Mapping):
        return f"{q['name']}@{js_number(q['round'])}"
    return f"{q.name}@{js_number(q.round)}"


def encode_answer(kind: str, p: Sequence[float]) -> list[float]:
    """intensity → [mean level, sd]; presence → [p]."""
    if kind == "presence":
        first = p[0] if len(p) > 0 else None
        return [0 if first is None else first]
    total = _or(_sum(p), 1)
    mean: float = 0
    sq: float = 0
    for i, v in enumerate(p):
        mean += (i * v) / total
        sq += (i * i * v) / total
    return [mean, math.sqrt(_js_max(0, sq - mean * mean))]


class DesignMatrix(NamedTuple):
    X: list[list[float]]
    names: list[str]
    owner: list[str]


def design_matrix(
    rows: Sequence[DiscoverRow],
    questions: Sequence[DiscoveredQuestion],
    answers: Mapping[str, list[list[float]]],
) -> DesignMatrix:
    names: list[str] = []
    owner: list[str] = []
    X = [list(r.base) for r in rows]
    for q in questions:
        cols = answers.get(question_id(q))
        if cols is None:
            raise ValueError(f"no answers for {question_id(q)}")
        width = len(encode_answer(q.kind, cols[0] if len(cols) > 0 and cols[0] is not None else []))
        for j in range(width):
            names.append(q.name if j == 0 else f"{q.name}_sd")
            owner.append(q.name)
        for i in range(len(rows)):
            cell = cols[i] if i < len(cols) and cols[i] is not None else []
            X[i].extend(encode_answer(q.kind, cell))
    return DesignMatrix(X, names, owner)


# ---------------------------------------------------------------------------
# Learner: L2 logistic regression, Newton steps on standardised columns
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LogisticModel:
    w: list[float]
    b: float
    mean: list[float]
    sd: list[float]


def _sigmoid(z: float) -> float:
    return 1 / (1 + _exp(-z)) if z >= 0 else _exp(z) / (1 + _exp(z))


def _solve(A: list[list[float]], g: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; a zero pivot is treated as 1e-12, as in TS."""
    n = len(g)
    M = [list(row) + [g[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(M[r][c]) > abs(M[p][c]):
                p = r
        M[c], M[p] = M[p], M[c]
        piv = _or(M[c][c], 1e-12)
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / piv
            if _truthy(f):
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [row[n] / _or(row[i], 1e-12) for i, row in enumerate(M)]


def _column_stats(X: Sequence[Sequence[float]], d: int, n: int) -> tuple[list[float], list[float]]:
    mean = [_sum(r[j] for r in X) / max(1, n) for j in range(d)]
    sd = []
    for j in range(d):
        acc: float = 0
        for r in X:
            dv = r[j] - mean[j]
            acc += dv * dv
        sd.append(_or(math.sqrt(acc / max(1, n)), 1))
    return mean, sd


def fit_logistic(X: Sequence[Sequence[float]], y: Sequence[float], lambda_: Optional[float] = 1.0, iters: int = 25) -> LogisticModel:
    lam = 1.0 if lambda_ is None else lambda_
    d = len(X[0]) if len(X) > 0 else 0
    n = len(X)
    mean, sd = _column_stats(X, d, n)
    Z = [[(v - mean[j]) / sd[j] for j, v in enumerate(r)] for r in X]
    # Parameter vector: [b, w...]; the intercept is not penalised.
    beta: list[float] = [0] * (d + 1)
    base = _sum(y) / max(1, n)
    beta[0] = math.log(_js_max(1e-6, base) / _js_max(1e-6, 1 - base))
    for _ in range(iters):
        g: list[float] = [0] * (d + 1)
        H: list[list[float]] = [[0] * (d + 1) for _ in range(d + 1)]
        for i in range(n):
            z = Z[i]
            s: float = 0
            for j, v in enumerate(z):
                s += v * beta[j + 1]
            p = _sigmoid(beta[0] + s)
            r = y[i] - p
            wgt = _js_max(p * (1 - p), 1e-9)
            x = [1, *z]
            for a in range(d + 1):
                g[a] += r * x[a]
                Ha = H[a]
                for b in range(a, d + 1):
                    Ha[b] += wgt * x[a] * x[b]
        for a in range(1, d + 1):
            g[a] -= lam * beta[a]
            H[a][a] += lam
        for a in range(d + 1):
            for b in range(a):
                H[a][b] = H[b][a]
        step = _solve(H, g)
        beta = [v + step[i] for i, v in enumerate(beta)]
        if _sum(abs(v) for v in step) < 1e-6:
            break
    return LogisticModel(b=beta[0], w=beta[1:], mean=mean, sd=sd)


def predict_logistic(m: LogisticModel, x: Sequence[float]) -> float:
    a: float = 0
    for j, v in enumerate(x):
        a += ((v - m.mean[j]) / m.sd[j]) * m.w[j]
    return _sigmoid(m.b + a)


def fit_ridge(X: Sequence[Sequence[float]], y: Sequence[float], lambda_: Optional[float] = 1.0) -> LogisticModel:
    """Ridge regression on standardised columns, closed form; the intercept is not penalised. Same shape as a logistic fit."""
    lam = 1.0 if lambda_ is None else lambda_
    d = len(X[0]) if len(X) > 0 else 0
    n = len(X)
    mean, sd = _column_stats(X, d, n)
    y_mean = _sum(y) / max(1, n)
    # Centred columns and target, so the intercept is the target mean and drops out.
    A: list[list[float]] = [[lam if j == k else 0 for k in range(d)] for j in range(d)]
    g: list[float] = [0] * d
    for i in range(n):
        z = [(v - mean[j]) / sd[j] for j, v in enumerate(X[i])]
        r = y[i] - y_mean
        for j in range(d):
            g[j] += z[j] * r
            Aj = A[j]
            for k in range(j, d):
                Aj[k] += z[j] * z[k]
    for j in range(d):
        for k in range(j):
            A[j][k] = A[k][j]
    return LogisticModel(b=y_mean, w=_solve(A, g) if d else [], mean=mean, sd=sd)


def predict_linear(m: LogisticModel, x: Sequence[float]) -> float:
    a: float = 0
    for j, v in enumerate(x):
        a += ((v - m.mean[j]) / m.sd[j]) * m.w[j]
    return m.b + a


class Learner(NamedTuple):
    fit: Callable[..., LogisticModel]
    predict: Callable[[LogisticModel, Sequence[float]], float]


def _fit_logistic_default_iters(X: Sequence[Sequence[float]], y: Sequence[float], lambda_: Optional[float] = None) -> LogisticModel:
    return fit_logistic(X, y, lambda_)


def learner_for(target: str) -> Learner:
    """The learner a target implies."""
    if target == "numeric":
        return Learner(fit_ridge, predict_linear)
    return Learner(_fit_logistic_default_iters, predict_logistic)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def log_loss(y: Sequence[float], p: Sequence[float]) -> float:
    a: float = 0
    for i, v in enumerate(y):
        q = _js_min(1 - 1e-6, _js_max(1e-6, p[i]))
        a = a - (v * math.log(q) + (1 - v) * math.log(1 - q))
    return a / max(1, len(y))


def _average_ranks(v: Sequence[float], offset: float) -> list[float]:
    idx = sorted(((x, i) for i, x in enumerate(v)), key=lambda t: t[0])
    out: list[float] = [0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1][0] == idx[i][0]:
            j += 1
        for k in range(i, j + 1):
            out[idx[k][1]] = (i + j) / 2 + offset
        i = j + 1
    return out


def auc(y: Sequence[float], p: Sequence[float]) -> float:
    """Rank AUC with ties averaged."""
    ranks = _average_ranks(p, 1)
    pos = len([v for v in y if v == 1])
    neg = len(y) - pos
    if not pos or not neg:
        return 0.5
    sum_pos: float = 0
    for i, r in enumerate(ranks):
        sum_pos += r if y[i] == 1 else 0
    return (sum_pos - (pos * (pos + 1)) / 2) / (pos * neg)


def rmse(y: Sequence[float], p: Sequence[float]) -> float:
    a: float = 0
    for i, v in enumerate(y):
        dv = v - p[i]
        a += dv * dv
    return math.sqrt(a / max(1, len(y)))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, ties averaged: does the model order the rows the way the labels do?"""
    ra, rb = _average_ranks(a, 0), _average_ranks(b, 0)
    n = len(ra)
    if n < 2:
        return 0
    ma, mb = _sum(ra) / n, _sum(rb) / n
    num: float = 0
    da: float = 0
    db: float = 0
    for i in range(n):
        num += (ra[i] - ma) * (rb[i] - mb)
        da += (ra[i] - ma) ** 2
        db += (rb[i] - mb) ** 2
    return num / math.sqrt(da * db) if _truthy(da) and _truthy(db) else 0


def describe_cv(s: CvScore) -> str:
    """One line for a log or the author's scoreboard."""
    if s.rmse is not None:
        return f"RMSE {to_fixed(s.rmse, 3)}, Spearman {to_fixed(0 if s.spearman is None else s.spearman, 3)}"
    return f"log loss {to_fixed(s.loss if s.log_loss is None else s.log_loss, 4)}, AUC {to_fixed(0.5 if s.auc is None else s.auc, 3)}"


def group_folds(groups: Sequence[str], k: int, seed: int) -> list[int]:
    """Fold index per row: whole groups dealt to folds by salted hash."""
    return [min(k - 1, math.floor(hash_to_unit(f"fold{js_number(seed)}:{g}") * k)) for g in groups]


class CvResult(NamedTuple):
    oof: list[float]
    score: CvScore


def cross_validate(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    groups: Sequence[str],
    *,
    folds: Optional[int] = None,
    repeats: Optional[int] = None,
    lambda_: Optional[float] = None,
    target: Optional[str] = None,
) -> CvResult:
    k = 5 if folds is None else folds
    reps = 2 if repeats is None else repeats
    numeric = target == "numeric"
    fit, predict = learner_for(target or "binary")
    oof: list[float] = [0] * len(y)
    loss: float = 0
    for rep in range(reps):
        fold = group_folds(groups, k, rep)
        pred: list[float] = [0] * len(y)
        for f in range(k):
            tr = [i for i in range(len(y)) if fold[i] != f]
            te = [i for i in range(len(y)) if fold[i] == f]
            if not te or not tr:
                continue
            m = fit([X[i] for i in tr], [y[i] for i in tr], lambda_)
            for i in te:
                pred[i] = predict(m, X[i])
        loss += rmse(y, pred) if numeric else log_loss(y, pred)
        for i, v in enumerate(pred):
            oof[i] += _div(v, reps)
    columns = len(X[0]) if len(X) > 0 else 0
    mean = _div(loss, reps)
    if numeric:
        return CvResult(oof, CvScore(loss=mean, rmse=mean, spearman=spearman(y, oof), columns=columns))
    return CvResult(oof, CvScore(loss=mean, log_loss=mean, auc=auc(y, oof), columns=columns))


def importance_by_owner(m: LogisticModel, owners: Sequence[str]) -> dict[str, float]:
    """Share of |standardised coefficient| per owner (a question owns its mean + sd columns)."""
    total = _or(_sum(abs(v) for v in m.w), 1)
    out: dict[str, float] = {}
    for j, o in enumerate(owners):
        w = abs(m.w[j]) if j < len(m.w) else math.nan
        out[o] = out.get(o, 0) + w / total
    return out


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "feature"
    c, n = base, 2
    while c in taken:
        c = f"{base}_{n}"
        n += 1
    return c


@dataclass
class _Example:
    row: DiscoverRow
    p: Optional[float]


def _pick_examples(rows: Sequence[DiscoverRow], oof: Optional[list[float]], n: int, target: str) -> list[_Example]:
    if oof is None and target == "numeric":
        # Round 1: evenly spaced across the label range, ties broken by hash so it is stable.
        sorted_rows = sorted(rows, key=lambda r: (r.label, hash_to_unit(r.id)))
        m = min(n, len(sorted_rows))
        at = [int(_js_round((k * (len(sorted_rows) - 1)) / max(1, m - 1))) for k in range(m)]
        return [_Example(sorted_rows[i], None) for i in dict.fromkeys(at)]
    if oof is None:
        # Round 1: half of each label, spread by hash so it is stable.
        by_hash = sorted(rows, key=lambda r: hash_to_unit(r.id))
        pos = [r for r in by_hash if r.label == 1][: math.ceil(n / 2)]
        neg = [r for r in by_hash if r.label == 0][: n - len(pos)]
        return [_Example(row, None) for row in [*neg, *pos]]
    order: list[dict[str, Any]] = sorted(
        ({"row": r, "p": oof[i], "err": abs(r.label - oof[i])} for i, r in enumerate(rows)),
        key=lambda o: o["err"],
        reverse=True,  # stable, like the TS comparator `b.err - a.err`
    )
    # Python slices clamp negative bounds exactly like `Array.prototype.slice`.
    return [_Example(o["row"], o["p"]) for o in [*order[: math.floor(n / 2)], *order[len(order) - math.ceil(n / 2) :]]]


def _render_examples(ex: list[_Example], target: str) -> str:
    def label(v: float) -> str:
        return js_number(_js_round(v * 100) / 100) if target == "numeric" else js_number(v)

    def pred(v: float) -> str:
        return to_fixed(v, 1 if target == "numeric" else 2)

    if ex and ex[0].p is None:
        return "\n".join(["Example rows with their label:", *[f"- label {label(e.row.label)}:\n{e.row.text}" for e in ex]])
    return "\n".join(
        [
            "Rows, worst-predicted first. The first half is where the current questions miss by the most, the second half where they are already right, so what separates the halves is what the questions have not captured.",
            *[f"- label {label(e.row.label)}, predicted {pred(e.p)}:\n{e.row.text}" for e in ex],  # type: ignore[arg-type]
        ]
    )


@dataclass(kw_only=True)
class _FreshQuestion(DiscoveredQuestion):
    """A question answered this round; `replaces` names the question a revise would replace ("" for an add)."""

    replaces: str


def _strip(q: _FreshQuestion) -> DiscoveredQuestion:
    return DiscoveredQuestion(name=q.name, kind=q.kind, question=q.question, round=q.round)


def _ports(ports: Any) -> tuple[Callable[..., Any], Callable[..., Any]]:
    if isinstance(ports, Mapping):
        return ports["author"], ports["answer"]
    return ports.author, ports.answer


def _spread(col: list[float]) -> float:
    m = _div(_sum(col), len(col))
    acc: float = 0
    for v in col:
        dv = v - m
        acc += dv * dv
    return math.sqrt(_div(acc, len(col)))


async def discover_features(
    *,
    rows: Sequence[DiscoverRow],
    base_names: Sequence[str],
    ports: Any,
    target: Optional[str] = "binary",
    rounds: Optional[int] = 4,
    examples: Optional[int] = 40,
    max_actions: Optional[int] = 12,
    min_spread: Optional[float] = 0.05,
    folds: Optional[int] = 5,
    repeats: Optional[int] = 2,
    lambda_: Optional[float] = 1.0,
    patience: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> DiscoverResult:
    """
    Run the loop. `ports` is a `DiscoverPorts` (or a dict with `author` and `answer`).
    `min_spread`: a column whose std over rows is below this cannot separate anything.
    `patience`: stop early after this many rounds with no CV improvement.
    """
    author, answer = _ports(ports)
    rounds = 4 if rounds is None else rounds
    min_spread = 0.05 if min_spread is None else min_spread
    target = "binary" if target is None else target
    n_examples = 40 if examples is None else examples
    n_actions = 12 if max_actions is None else max_actions
    cv_opts = dict(
        folds=5 if folds is None else folds,
        repeats=2 if repeats is None else repeats,
        lambda_=1.0 if lambda_ is None else lambda_,
        target=target,
    )
    emit: Callable[[str], None] = log if log is not None else (lambda _line: None)
    y = [r.label for r in rows]
    if target == "binary" and any(v != 0 and v != 1 for v in y):
        raise ValueError('a binary target needs 0/1 labels; pass target: "numeric" for a score')
    if any(not math.isfinite(v) for v in y):
        raise ValueError("every label must be a finite number")
    groups = [r.group for r in rows]

    answers: dict[str, list[list[float]]] = {}
    accepted: list[DiscoveredQuestion] = []
    history: list[CvScore] = []
    snapshots: list[list[DiscoveredQuestion]] = []
    journal: list[JournalEntry] = []

    def cv(qs: Sequence[DiscoveredQuestion]) -> CvResult:
        return cross_validate(design_matrix(rows, qs, answers).X, y, groups, **cv_opts)  # type: ignore[arg-type]

    base_run = cv([])
    base_cv = base_run.score
    emit(f"base ({len(base_names)} columns): CV {describe_cv(base_cv)}")
    oof: Optional[list[float]] = None
    feedback = ""
    stale = 0

    for round_ in range(1, rounds + 1):
        req = DiscoverAuthorRequest(
            round=round_,
            accepted=accepted,
            examples=_render_examples(_pick_examples(rows, oof, n_examples, target), target),
            feedback=feedback,
            max_actions=n_actions,
        )
        raw = await maybe_await(author(req))
        actions = [discover_action_from_dict(a) for a in list(raw)[: max(0, n_actions)]]
        live = {q.name: q for q in accepted}
        drops = [a.target for a in actions if a.op == "drop" and a.target in live]
        replacing = {a.target for a in actions if a.op == "revise" and a.target in live}
        taken = {n for n in live if n not in replacing}
        fresh: list[_FreshQuestion] = []
        for a in actions:
            if a.op == "drop" or (a.op == "revise" and a.target not in live) or not _js_trim(a.question):
                continue
            name = _slug(a.name, taken)
            taken.add(name)
            fresh.append(
                _FreshQuestion(
                    name=name,
                    kind="intensity" if a.kind == "intensity" else "presence",
                    question=_js_trim(a.question),
                    round=round_,
                    replaces=a.target if a.op == "revise" else "",
                )
            )
        if fresh:
            got = await maybe_await(answer(rows, fresh))
            for q in fresh:
                cols = got.get(q.name)
                if cols is None or len(cols) != len(rows):
                    raise ValueError(f"answer port returned {0 if cols is None else len(cols)} rows for {q.name}, expected {len(rows)}")
                answers[question_id(q)] = cols
        for q in [f for f in fresh if not f.replaces]:
            spread = _spread([encode_answer(q.kind, p)[0] for p in answers[question_id(q)]])
            if spread < min_spread:
                journal.append(JournalEntry(round=round_, what="flat", name=q.name, note=f"spread {to_fixed(spread, 3)}"))
                continue
            accepted.append(_strip(q))
            journal.append(JournalEntry(round=round_, what="add", name=q.name, note=""))
        current = cv(accepted).score
        for q in [f for f in fresh if f.replaces]:
            at = next((i for i, a in enumerate(accepted) if a.name == q.replaces), -1)
            if at < 0:
                journal.append(JournalEntry(round=round_, what="stale", name=q.name, note=f"{q.replaces} is gone"))
                continue
            trial = list(accepted)
            trial[at] = _strip(q)
            s = cv(trial).score
            if s.loss < current.loss:
                accepted = trial
                journal.append(JournalEntry(round=round_, what="revise", name=q.name, note=f"was {q.replaces}, {to_fixed(current.loss, 4)} → {to_fixed(s.loss, 4)}"))
                current = s
            else:
                journal.append(JournalEntry(round=round_, what="reject", name=q.name, note=f"would cost {to_fixed(s.loss - current.loss, 4)}"))
        for name in drops:
            trial = [a for a in accepted if a.name != name]
            if len(trial) == len(accepted):
                continue
            s = cv(trial).score
            if s.loss < current.loss:
                accepted = trial
                journal.append(JournalEntry(round=round_, what="drop", name=name, note=f"{to_fixed(current.loss, 4)} → {to_fixed(s.loss, 4)}"))
                current = s
            else:
                journal.append(JournalEntry(round=round_, what="keep", name=name, note=f"dropping would cost {to_fixed(s.loss - current.loss, 4)}"))

        run = cv(accepted)
        oof = run.oof
        best = base_cv.loss
        for h in history:
            best = _js_min(best, h.loss)
        stale = 0 if run.score.loss < best - 1e-4 else stale + 1
        history.append(run.score)
        snapshots.append(list(accepted))
        X, _names, owner = design_matrix(rows, accepted, answers)
        imp = importance_by_owner(learner_for(target).fit(X, y, cv_opts["lambda_"]), [*base_names, *owner])

        def spread_of(name: str) -> str:
            q = next((a for a in accepted if a.name == name), None)
            if q is None:
                return ""
            return f", spread {to_fixed(_spread([encode_answer(q.kind, p)[0] for p in answers[question_id(q)]]), 2)}"

        accepted_names = {a.name for a in accepted}
        ranked = sorted(((k, imp[k]) for k in _js_key_order(imp.keys())), key=lambda kv: kv[1], reverse=True)
        feedback = "\n".join(
            [
                f"Cross-validated {'RMSE' if target == 'numeric' else 'log loss'} (lower is better). Base columns alone: {describe_cv(base_cv)}.",
                *[f"  round {i + 1}: {describe_cv(h)} ({js_number(h.columns)} columns)" for i, h in enumerate(history)],
                "",
                "Importance share of every column the model reads, base columns included, and the spread of each question's answer across rows. A question with a low share or a low spread is not earning its place; revise or drop it.",
                *[
                    f"  {k}: {to_fixed(100 * v, 1)}%{' (base)' if k in base_names and k not in accepted_names else spread_of(k)}"
                    for k, v in ranked
                ],
            ]
        )
        emit(f"round {round_}: {len(fresh)} answered, {len(accepted)} kept, CV {describe_cv(run.score)}")
        for j in journal:
            if j.round == round_:
                emit(f"  {j.what.ljust(7)}{j.name.ljust(40)}{j.note}")
        if patience and stale >= patience:
            emit(f"stopping: no improvement for {stale} rounds")
            break
    return DiscoverResult(accepted=accepted, answers=answers, base_cv=base_cv, history=history, snapshots=snapshots, journal=journal)
