"""
Learned gates — what feature discovery hands a pipeline (#321).

`discover_features` (.discover) ends with a question set and a linear model
over their answers. This file writes both into a spec with no new node kind:

  decide   asks the discovered questions (the host publishes the wording
           under the node's `questions` name, as it does for any decide node)
  derive   a `code` node whose `expr` IS the fitted model: the weights are
           literals, so a refit is a new expression, which the spec compiler
           already turns into a new node version
  gate     optional: cuts the score into branches on a named threshold, so
           the cut stays tunable like any other threshold

Discovery is not a node. A node runs once per input; discovery runs over
many recorded runs and their labels, and its output has to clear a shadow
and a locked trial before it ships. So it is a spec operation, beside
`propose` and `shadow`: `discover_for_spec` runs the loop and returns a
patched spec, and `attach_learned_gate` is the pure half that does the patch.

The decide port's contract for these questions is the one the discovery
answer port already had: presence → [P(true)], intensity → one probability
per `INTENSITY_LEVELS` level. The weights only mean in production what they
meant in the loop if both ports return the same shape AND ask the same
wording: build the answer port's questions with `learned_question_set` and
pass the same `presence` criteria to the attach.

The spec is the JSON pipeline spec as a plain dict, in the TS JSON shape
(camelCase keys, nodes keyed by id, edges `{"from", "to", "when"?}`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from ._compat import js_number, js_round_precision
from .discover import (
    INTENSITY_LEVELS,
    DiscoveredQuestion,
    DiscoverResult,
    DiscoverRow,
    design_matrix,
    discover_features,
    encode_answer,
    learner_for,
    question_id,
)
from .hash import short_hash
from .types import ProbeQuestion

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BaseColumn:
    """A base column the host already has, by name."""

    base: str

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base}


@dataclass(kw_only=True)
class QuestionColumn:
    """An encoding of one question's answer."""

    question: str
    kind: Literal["intensity", "presence"]
    encode: Literal["p", "mean", "sd"]

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "kind": self.kind, "encode": self.encode}


#: One column the model reads: a base column by name, or an encoding of one question's answer.
LearnedColumn = Union[BaseColumn, QuestionColumn]


def learned_column_from_dict(d: Any) -> LearnedColumn:
    if isinstance(d, (BaseColumn, QuestionColumn)):
        return d
    if "base" in d:
        return BaseColumn(base=d["base"])
    return QuestionColumn(question=d["question"], kind=d["kind"], encode=d["encode"])


@dataclass(kw_only=True)
class LearnedModel:
    target: Literal["binary", "numeric"]
    columns: list[LearnedColumn]
    #: Standardised-space weights, as the learner fitted them.
    w: list[float]
    b: float
    mean: list[float]
    sd: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "columns": [c.to_dict() for c in self.columns],
            "w": list(self.w),
            "b": self.b,
            "mean": list(self.mean),
            "sd": list(self.sd),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "LearnedModel":
        if isinstance(d, LearnedModel):
            return d
        return cls(
            target=d["target"],
            columns=[learned_column_from_dict(c) for c in d["columns"]],
            w=list(d["w"]),
            b=d["b"],
            mean=list(d["mean"]),
            sd=list(d["sd"]),
        )


def fit_learned_model(
    *,
    rows: Sequence[DiscoverRow],
    base_names: Sequence[str],
    questions: Sequence[DiscoveredQuestion],
    answers: Mapping[str, list[list[float]]],
    target: Optional[str] = "binary",
    lambda_: Optional[float] = 1.0,
) -> LearnedModel:
    """Fit the final model on every row (CV already chose the questions)."""
    target = "binary" if target is None else target
    X = design_matrix(rows, questions, answers).X
    m = learner_for(target).fit(X, [r.label for r in rows], 1.0 if lambda_ is None else lambda_)
    columns: list[LearnedColumn] = [BaseColumn(base=b) for b in base_names]
    for q in questions:
        if q.kind == "presence":
            columns.append(QuestionColumn(question=q.name, kind=q.kind, encode="p"))
        else:
            columns.append(QuestionColumn(question=q.name, kind=q.kind, encode="mean"))
            columns.append(QuestionColumn(question=q.name, kind=q.kind, encode="sd"))
    if len(columns) != len(m.w):
        raise ValueError(f"model has {len(m.w)} weights for {len(columns)} columns")
    return LearnedModel(target=target, columns=columns, w=m.w, b=m.b, mean=m.mean, sd=m.sd)  # type: ignore[arg-type]


def _column_value(c: LearnedColumn, answers: Mapping[str, Sequence[float]], base: Mapping[str, float], fill: float) -> float:
    if isinstance(c, BaseColumn):
        v = base.get(c.base)
        return fill if v is None else v
    d = answers.get(c.question)
    if d is None or len(d) == 0:
        return fill
    enc = encode_answer(c.kind, d)
    if c.encode == "sd":
        return enc[1] if len(enc) > 1 and enc[1] is not None else fill
    return enc[0] if len(enc) > 0 and enc[0] is not None else fill


def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def score_learned(model: LearnedModel, answers: Mapping[str, Sequence[float]], base: Optional[Mapping[str, float]] = None) -> float:
    """
    The reference implementation of `learned_expression`: the score for one run's
    answers (decide distributions keyed by question name) and base columns. A
    missing answer or base column takes its training mean, so it moves nothing.
    """
    model = LearnedModel.from_dict(model)
    base = {} if base is None else base
    a: float = 0
    for j, col in enumerate(model.columns):
        a += ((_column_value(col, answers, base, model.mean[j]) - model.mean[j]) / model.sd[j]) * model.w[j]
    z = model.b + a
    return 1 / (1 + _exp(-z)) if model.target == "binary" else z


def learned_importance(model: LearnedModel) -> dict[str, float]:
    """Share of |standardised weight| per question or base column."""
    model = LearnedModel.from_dict(model)
    total: float = 0
    for v in model.w:
        total += abs(v)
    if total == 0 or total != total:
        total = 1
    out: dict[str, float] = {}
    for j, col in enumerate(model.columns):
        k = col.base if isinstance(col, BaseColumn) else col.question
        out[k] = out.get(k, 0) + abs(model.w[j]) / total
    return out


# ---------------------------------------------------------------------------
# The model as a JSONata expression
# ---------------------------------------------------------------------------


def _num(v: float) -> str:
    """Ten significant digits: exact enough, and short enough to read in a diff."""
    s = js_number(js_round_precision(v, 10))
    return f"({s})" if v < 0 else s


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NOT_WORD = re.compile(r"[^A-Za-z0-9_]")


def _var(name: str) -> str:
    # `/[^A-Za-z0-9_]/g` in JS replaces UTF-16 code units: an astral character becomes two underscores.
    return "$" + _NOT_WORD.sub(lambda m: "__" if ord(m.group(0)) > 0xFFFF else "_", name)


def _json_string(s: str) -> str:
    """`JSON.stringify(s)` for a string."""
    return json.dumps(s, ensure_ascii=False)


def learned_expression(
    model: LearnedModel,
    roots: Optional[Mapping[str, Optional[str]]] = None,
    *,
    answers: Optional[str] = None,
    base: Optional[str] = None,
) -> str:
    """
    The model as one JSONata expression, weights folded into raw-space
    coefficients so it reads as `intercept + c₁·x₁ + …`. `answers` and `base`
    name the roots it reads (the derive node's `inputs`); a missing answer or
    base column falls back to its training mean, like `score_learned`.
    Roots go as keywords (`answers="answers", base="base"`) or as a dict like TS.
    """
    model = LearnedModel.from_dict(model)
    if roots is not None:
        answers = roots.get("answers") if answers is None else answers
        base = roots.get("base") if base is None else base
    if answers is None or not _IDENT.fullmatch(answers) or (base is not None and not _IDENT.fullmatch(base)):
        raise ValueError("roots must be plain identifiers")
    if any(isinstance(col, BaseColumn) for col in model.columns) and not base:
        raise ValueError("the model reads base columns: name the `base` root")
    coef = [w / model.sd[j] for j, w in enumerate(model.w)]
    acc: float = 0
    for j, col in enumerate(coef):
        acc += col * model.mean[j]
    intercept = model.b - acc
    lets: list[str] = []
    terms: list[str] = []
    seen: set[str] = set()
    for j, column in enumerate(model.columns):
        fill = _num(model.mean[j])
        if isinstance(column, BaseColumn):
            x = _var(f"b_{column.base}")
            lets.append(f'{x} := $lookup({base}, {_json_string(column.base)}); {x} := $type({x}) = "number" ? {x} : {fill};')
            terms.append(f"{_num(coef[j])} * {x}")
            continue
        d = _var(f"d_{column.question}")
        if column.question not in seen:
            seen.add(column.question)
            # $lookup, not a path: a one-element array read by path collapses to its element.
            lets.append(f"{d} := $lookup({answers}, {_json_string(column.question)});")
            if column.kind == "intensity":
                m = _var(f"m_{column.question}")
                lets.append(f"{m} := $exists({d}) ? $sum($map({d}, function($p, $i) {{ $p * $i }})) / $sum({d}) : null;")
                lets.append(
                    f"{_var(f's_{column.question}')} := $exists({d}) ? $sqrt($max([0, $sum($map({d}, function($p, $i) {{ $p * $i * $i }})) / $sum({d}) - {m} * {m}])) : null;"
                )
        if column.encode == "p":
            x = f"($exists({d}) ? {d}[0] : {fill})"
        else:
            mv = _var(f"{'m' if column.encode == 'mean' else 's'}_{column.question}")
            x = f"({mv} != null ? {mv} : {fill})"
        terms.append(f"{_num(coef[j])} * {x}")
    z = " + ".join([_num(intercept), *terms])
    score = "1 / (1 + $power(2.718281828459045, -($z)))" if model.target == "binary" else "$z"
    return "\n".join(["(", *[f"  {l}" for l in lets], f"  $z := {z};", f'  {{ "score": {score} }}', ")"])


# ---------------------------------------------------------------------------
# Question wording
# ---------------------------------------------------------------------------

DEFAULT_PRESENCE_CRITERIA: dict[str, str] = {
    "true": "The state states this or clearly implies it",
    "false": "The state gives no indication of this",
}


def learned_question_set(
    questions: Sequence[DiscoveredQuestion],
    presence: Optional[Mapping[str, str]] = None,
) -> dict[str, ProbeQuestion]:
    """The question set the host publishes under the decide node's `questions` name."""
    criteria = DEFAULT_PRESENCE_CRITERIA if presence is None else presence
    return {
        q.name: (
            {"type": "score", "instructions": q.question, "criteria": list(INTENSITY_LEVELS)}
            if q.kind == "intensity"
            else {"type": "noul", "instructions": q.question, "criteria": criteria}
        )
        for q in questions
    }


# ---------------------------------------------------------------------------
# Attaching it to a spec
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LearnedGate:
    """A gate on the score: `score >= cut` takes `above`. `routes` maps each branch to the nodes it feeds."""

    #: A threshold: a number, or `{"env": ..., "default": ...}` / `{"promptConfig": ..., "default": ...}`.
    cut: Any
    above: str
    below: str
    routes: dict[str, list[str]]


@dataclass(kw_only=True)
class LearnedGateAttach:
    #: Prefix for the fragment's node ids: `<prefix>_ask`, `<prefix>_score`, `<prefix>_gate`.
    prefix: str
    #: The node the decide node hangs off (its only inbound edge).
    after: str
    #: The decide node's `questions` name — what the host resolves to `question_set`.
    questions_name: str
    #: Named state blocks for the decide node, when the host resolves them.
    state: Optional[list[str]] = None
    #: The presence criteria the answer port asked with. The published wording
    #: must be the wording the model was fitted on, or the weights read answers
    #: to a different question. Default `DEFAULT_PRESENCE_CRITERIA`.
    presence: Optional[dict[str, str]] = None
    #: Where the base columns come from: a ref on a DIRECT inbound node (`features.base`). Its node gets an edge into the score node.
    base: Optional[str] = None
    #: No gate: the score node's output feeds these nodes.
    feeds: Optional[list[str]] = None
    gate: Optional[LearnedGate] = None

    @classmethod
    def from_dict(cls, d: Any) -> "LearnedGateAttach":
        """camelCase (`questionsName`) or snake_case keys."""
        if isinstance(d, LearnedGateAttach):
            return d
        g = d.get("gate")
        if g is not None and not isinstance(g, LearnedGate):
            g = LearnedGate(cut=g["cut"], above=g["above"], below=g["below"], routes=dict(g["routes"]))
        return cls(
            prefix=d["prefix"],
            after=d["after"],
            questions_name=d["questionsName"] if "questionsName" in d else d["questions_name"],
            state=d.get("state"),
            presence=d.get("presence"),
            base=d.get("base"),
            feeds=d.get("feeds"),
            gate=g,
        )


@dataclass(kw_only=True)
class AttachResult:
    #: The patched spec, a new dict in the TS JSON shape.
    spec: dict[str, Any]
    #: `{"ask": ..., "score": ..., "gate": ... | None}`.
    ids: dict[str, Optional[str]]
    #: Human-readable diff, for the review and the commit message.
    applied: list[str]


def _spec_dict(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        return dict(spec)
    dump = getattr(spec, "model_dump", None)
    if callable(dump):
        return dump(by_alias=True, exclude_none=True)  # type: ignore[no-any-return]
    raise TypeError("spec must be a JSON pipeline spec dict")


def attach_learned_gate(
    spec: Mapping[str, Any],
    model: LearnedModel,
    questions: Sequence[DiscoveredQuestion],
    at: Union[LearnedGateAttach, Mapping[str, Any]],
) -> AttachResult:
    """
    Add (or replace) a learned gate in a spec. Pure: no model calls. Re-running
    it with a refitted model replaces the fragment's nodes and edges in place,
    so the version history of `<prefix>_score` IS the model's history.
    """
    spec = _spec_dict(spec)
    model = LearnedModel.from_dict(model)
    at = LearnedGateAttach.from_dict(at)
    ids: dict[str, Optional[str]] = {
        "ask": f"{at.prefix}_ask",
        "score": f"{at.prefix}_score",
        "gate": f"{at.prefix}_gate" if at.gate else None,
    }
    ask, score_id = f"{at.prefix}_ask", f"{at.prefix}_score"
    mine = {ask, score_id, f"{at.prefix}_gate"}
    applied: list[str] = []
    spec_nodes: Mapping[str, Any] = spec["nodes"]
    replacing = [nid for nid in spec_nodes if nid in mine]
    if replacing:
        applied.append(f"replaced {', '.join(replacing)}")
    if spec_nodes.get(at.after) is None:
        raise ValueError(f'attach.after: no node "{at.after}"')
    base_node = at.base.split(".")[0] if at.base is not None else None
    if base_node and spec_nodes.get(base_node) is None:
        raise ValueError(f'attach.base: no node "{base_node}"')
    uses_base = any(isinstance(c, BaseColumn) for c in model.columns)
    if uses_base and not at.base:
        raise ValueError("the model reads base columns: pass attach.base")

    nodes: dict[str, Any] = {nid: n for nid, n in spec_nodes.items() if nid not in mine}
    edges: list[dict[str, Any]] = [e for e in spec["edges"] if e["from"] not in mine and e["to"] not in mine]

    question_set = learned_question_set(questions, at.presence)
    nodes[ask] = {
        "kind": "decide",
        "version": f"q{short_hash(question_set)}",
        "describe": f"{len(questions)} discovered questions ({at.questions_name})",
        "questions": at.questions_name,
        **({"state": at.state} if at.state is not None else {}),
    }
    edges.append({"from": at.after, "to": ask})
    applied.append(f'{ask}: decide, {len(questions)} questions from "{at.questions_name}"')

    nodes[score_id] = {
        "kind": "code",
        "role": "derive",
        "version": "1",
        "describe": f"learned {'logistic' if model.target == 'binary' else 'ridge'} model over {len(model.columns)} columns",
        "inputs": {"answers": ask, **({"base": at.base} if uses_base else {})},
        "expr": learned_expression(model, answers="answers", base="base" if uses_base else None),
    }
    edges.append({"from": ask, "to": score_id})
    if uses_base and base_node and base_node != ask:
        edges.append({"from": base_node, "to": score_id})
    applied.append(f"{score_id}: derive, {model.target} score over {len(model.columns)} columns")

    if at.gate:
        g = at.gate
        gate_id = f"{at.prefix}_gate"
        nodes[gate_id] = {
            "kind": "code",
            "role": "gate",
            "version": "1",
            "inputs": {"score": f"{score_id}.score"},
            "thresholds": {"cut": g.cut},
            "branches": [g.above, g.below],
            "rules": [
                {"when": "score >= $t.cut", "branch": g.above, "reason": "'score ' & $string(score) & ' >= ' & $string($t.cut)"},
                {"branch": g.below, "reason": "'score ' & $string(score) & ' < ' & $string($t.cut)"},
            ],
        }
        edges.append({"from": score_id, "to": gate_id})
        for branch, targets in g.routes.items():
            for to in targets:
                edges.append({"from": gate_id, "to": to, "when": {"gate": gate_id, "branch": branch}})
        applied.append(f"{gate_id}: gate, score >= cut → {g.above}, else {g.below}")
    else:
        for to in at.feeds or []:
            edges.append({"from": score_id, "to": to})
    return AttachResult(spec={**spec, "nodes": nodes, "edges": edges}, ids=ids, applied=applied)


# ---------------------------------------------------------------------------
# The whole step
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DiscoverForSpecCheck:
    """Compile the patched spec and report its problems."""

    registry: Any
    engine: Any


@dataclass(kw_only=True)
class DiscoverForSpecResult(AttachResult):
    discovery: DiscoverResult
    model: LearnedModel
    #: Publish this under `attach.questions_name` before the spec runs.
    question_set: dict[str, ProbeQuestion]
    #: Empty when the patched spec compiles (or when no `check` was given).
    problems: list[str] = field(default_factory=list)


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, Mapping) else getattr(obj, key)


async def discover_for_spec(
    *,
    spec: Mapping[str, Any],
    attach: Union[LearnedGateAttach, Mapping[str, Any]],
    rows: Sequence[DiscoverRow],
    base_names: Sequence[str],
    discovery: Union[DiscoverResult, Mapping[str, Any], None] = None,
    check: Union[DiscoverForSpecCheck, Mapping[str, Any], None] = None,
    target: Optional[str] = "binary",
    lambda_: Optional[float] = 1.0,
    **discover_kwargs: Any,
) -> DiscoverForSpecResult:
    """
    Discovery as a spec operation: run the loop on dev rows, fit the final
    model, and return the spec with the learned gate attached. What it returns
    is a CHALLENGER — shadow it beside the champion and lock a trial before it
    ships. Held-out rows never reach this function; score them once, outside.

    `discovery` reuses a finished loop instead of running one (its answers must
    cover `rows`); a dict in the TS JSON shape is accepted. `discover_kwargs`
    go to `discover_features` (`ports`, `rounds`, `examples`, ...).
    """
    at = LearnedGateAttach.from_dict(attach)
    if discovery is None:
        found = await discover_features(rows=rows, base_names=base_names, target=target, lambda_=lambda_, **discover_kwargs)
    else:
        found = DiscoverResult.from_dict(discovery)
    answers = {question_id(q): found.answers.get(question_id(q)) for q in found.accepted}
    model = fit_learned_model(
        rows=rows,
        base_names=base_names,
        questions=found.accepted,
        answers=answers,  # type: ignore[arg-type]
        target=target,
        lambda_=lambda_,
    )
    attached = attach_learned_gate(spec, model, found.accepted, at)
    problems: list[str] = []
    if check is not None:
        from .spec import compile_pipeline_spec

        compiled = compile_pipeline_spec(attached.spec, _get(check, "registry"), _get(check, "engine"))
        problems = list(_get(compiled, "problems"))
    return DiscoverForSpecResult(
        spec=attached.spec,
        ids=attached.ids,
        applied=attached.applied,
        discovery=found,
        model=model,
        question_set=learned_question_set(found.accepted, at.presence),
        problems=problems,
    )
