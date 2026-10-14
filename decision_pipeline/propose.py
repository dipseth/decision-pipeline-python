"""
The proposer — a model writes a pipeline spec (./spec) toward a goal, and
the compiler's problem list drives its repair loop.

The model is a port (`SpecAuthor`), so this file stays host-agnostic: the
host decides which model, how it is traced and what it costs. One round is:
prompt -> reply -> parse JSON -> compile -> vocabulary and base checks. A
round with no problems ends the loop; otherwise the next prompt carries the
previous reply and EVERY problem, so a model can fix them all at once.

Beyond what `compile_pipeline_spec` checks, a proposal against a `base` must

  keep its contract   same `fact`, `input` and `output` — it will run beside
                      the base as an arm, so it must answer the same question
                      in the same shape.
  stay in vocabulary  prompts, routes, Jev question sets and store targets
                      name things the HOST's ports dispatch on. A spec can
                      compile against a route no port handles and fail only
                      mid-run, so an unknown name is a problem here. The
                      default vocabulary is the base's own.
  change something    a proposal identical to the base is a wasted arm.

What it never does is RUN the proposal. A compiled spec is a candidate; the
shadow arm and the replay harness judge it.

Specs are plain dicts in the TS JSON shape (camelCase keys).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

from ._compat import error_message, maybe_await, to_plain
from .define import DecisionPipeline
from .expressions import ExpressionEngine
from .hash import stable_stringify
from .spec import SpecRegistry, compile_pipeline_spec, describe_registry, pipeline_spec_json_schema

__all__ = [
    "AuthorRequest",
    "SpecAuthor",
    "SpecVocabulary",
    "ProposalRound",
    "SpecDiff",
    "Proposal",
    "spec_vocabulary",
    "diff_specs",
    "extract_spec_json",
    "propose_pipeline_spec",
    "SPEC_LANGUAGE",
]


@dataclass(kw_only=True)
class AuthorRequest:
    #: The spec language, the catalog and the JSON Schema. Identical every round — cacheable.
    system: str
    #: Round 1: the goal and the base. Later rounds: the previous reply and its problems.
    prompt: str
    #: 1-based.
    round: int
    signal: Any = None


#: The model. Returns its reply — text with a JSON object in it, or the object itself. Sync or async.
SpecAuthor = Callable[[AuthorRequest], Union[Any, Awaitable[Any]]]


@dataclass(kw_only=True)
class SpecVocabulary:
    """
    Names a spec may use that the host's ports dispatch on. A field left `None`
    is unconstrained (the TS `Partial<SpecVocabulary>`); `spec_vocabulary`
    always fills all four. Wherever a vocabulary is accepted, a dict with the
    same keys is too.
    """

    prompts: Optional[list[str]] = None
    routes: Optional[list[str]] = None
    questions: Optional[list[str]] = None
    targets: Optional[list[str]] = None


_VOCAB_KEYS = [f.name for f in fields(SpecVocabulary)]


@dataclass(kw_only=True)
class ProposalRound:
    round: int
    #: The parsed spec, or None when the reply held no JSON object.
    spec: Any
    problems: list[str]


@dataclass(kw_only=True)
class SpecDiff:
    #: `{"added", "removed", "changed"}` node ids.
    nodes: dict[str, list[str]]
    #: `{"added", "removed"}` edge keys (`from -> to [gate=branch]`).
    edges: dict[str, list[str]]
    #: Top-level keys other than nodes and edges whose value changed.
    fields: list[str]


@dataclass(kw_only=True)
class Proposal:
    #: Set only when the last round had no problems.
    spec: Optional[dict[str, Any]]
    pipeline: Optional[DecisionPipeline]
    rounds: list[ProposalRound] = field(default_factory=list)
    #: The last round's problems — empty on success.
    problems: list[str] = field(default_factory=list)
    #: Against `base`, when there is one and the proposal compiled.
    diff: Optional[SpecDiff] = None


# ---------------------------------------------------------------------------
# Vocabulary and diff
# ---------------------------------------------------------------------------


def spec_vocabulary(spec: Mapping[str, Any]) -> SpecVocabulary:
    """The host-dispatched names a spec uses."""
    v = SpecVocabulary(prompts=[], routes=[], questions=[], targets=[])

    def add(lst: list[str], name: Any) -> None:
        if name is not None and name not in lst:
            lst.append(name)

    for n in spec["nodes"].values():
        kind = n.get("kind")
        if kind == "generate":
            add(v.prompts, n.get("prompt"))  # type: ignore[arg-type]
            add(v.routes, n.get("route"))  # type: ignore[arg-type]
        elif kind == "decide":
            add(v.questions, n.get("questions"))  # type: ignore[arg-type]
        elif kind == "store":
            add(v.targets, n.get("target"))  # type: ignore[arg-type]
    return v


def _edge_key(e: Mapping[str, Any]) -> str:
    when = e.get("when")
    return f"{e['from']} -> {e['to']}" + (f" [{when['gate']}={when['branch']}]" if when else "")


def diff_specs(base: Mapping[str, Any], next: Mapping[str, Any]) -> SpecDiff:  # noqa: A002 - the TS name
    base_nodes: Mapping[str, Any] = base["nodes"]
    next_nodes: Mapping[str, Any] = next["nodes"]
    base_edges = list(dict.fromkeys(_edge_key(e) for e in base["edges"]))
    next_edges = list(dict.fromkeys(_edge_key(e) for e in next["edges"]))
    keys = list(dict.fromkeys([*base.keys(), *next.keys()]))
    return SpecDiff(
        nodes={
            "added": [i for i in next_nodes if i not in base_nodes],
            "removed": [i for i in base_nodes if i not in next_nodes],
            "changed": [i for i in next_nodes if i in base_nodes and stable_stringify(base_nodes[i]) != stable_stringify(next_nodes[i])],
        },
        edges={
            "added": [k for k in next_edges if k not in base_edges],
            "removed": [k for k in base_edges if k not in next_edges],
        },
        fields=[k for k in keys if k not in ("nodes", "edges") and stable_stringify(base.get(k)) != stable_stringify(next.get(k))],
    )


def _is_empty_diff(d: SpecDiff) -> bool:
    return (
        len(d.nodes["added"]) + len(d.nodes["removed"]) + len(d.nodes["changed"]) + len(d.edges["added"]) + len(d.edges["removed"]) + len(d.fields)
        == 0
    )


# ---------------------------------------------------------------------------
# Checks beyond the compiler
# ---------------------------------------------------------------------------


def _vocabulary_problems(spec: Mapping[str, Any], allowed: Mapping[str, Optional[list[str]]]) -> list[str]:
    problems: list[str] = []

    def check(where: str, what: str, name: Any, lst: Optional[list[str]]) -> None:
        if name is None or lst is None or name in lst:
            return
        problems.append(f'{where}: unknown {what} "{name}" — the host handles only {", ".join(f"{chr(34)}{x}{chr(34)}" for x in lst) or "none"}')

    for nid, n in spec["nodes"].items():
        where = f"nodes.{nid}"
        kind = n.get("kind")
        if kind == "generate":
            check(where, "prompt", n.get("prompt"), allowed.get("prompts"))
            check(where, "route", n.get("route"), allowed.get("routes"))
        elif kind == "decide":
            check(where, "question set", n.get("questions"), allowed.get("questions"))
        elif kind == "store":
            check(where, "store target", n.get("target"), allowed.get("targets"))
    return problems


def _contract_problems(spec: Mapping[str, Any], base: Mapping[str, Any]) -> list[str]:
    return [
        f'{k}: must stay "{base.get(k)}" (was changed to "{spec.get(k)}") — a proposal runs beside the base and must decide the same fact in the same shape'
        for k in ("fact", "input", "output")
        if spec.get(k) != base.get(k)
    ]


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n([\s\S]*?)```")


def _reject_constant(name: str) -> Any:
    # JSON.parse has no NaN / Infinity; json.loads would accept them.
    raise ValueError(f"Unexpected token {name}")


def extract_spec_json(reply: Any) -> dict[str, Any]:
    """
    The JSON object in a reply: a fenced ```json block, else the outermost braces.

    Returns `{"value": <the object>}` or `{"problem": <text>}` — a dict with exactly
    one of the two keys, the TS union as data.
    """
    if reply is not None and not isinstance(reply, (str, bytes, bool, int, float)):
        return {"value": reply}
    if not isinstance(reply, str):
        return {"problem": "the reply was neither text nor an object"}
    fenced = [m.group(1) for m in _FENCE_RE.finditer(reply)]
    # `reply.slice(indexOf("{"), lastIndexOf("}") + 1)`: Python slicing agrees with JS slice on -1 and on an empty range.
    candidates = fenced if fenced else [reply[reply.find("{") : reply.rfind("}") + 1]]
    last_error = "no JSON object found"
    # The last fenced block wins: a model that shows a fragment first ends with the whole spec.
    for c in reversed(candidates):
        if not c.strip():
            continue
        try:
            value = json.loads(c, parse_constant=_reject_constant)
            if isinstance(value, dict):
                return {"value": value}
            last_error = "the JSON is not an object"
        except ValueError as err:
            last_error = f"the JSON does not parse: {error_message(err)}"
    return {"problem": f"reply: {last_error}. Reply with the whole spec as ONE ```json block."}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SPEC_LANGUAGE = """You write decision-pipeline specs: JSON describing a DAG of nodes that decides one fact.

Node kinds:
- read: loads data by calling a registered primitive (`call`: "id@version").
- decide: asks Jev (a probabilistic evaluator) a named question set (`questions`), or answers `probes` written upstream.
- generate: an LLM call; `prompt` and `route` name what the host runs.
- code: one of
    call   a registered primitive with `args` (each arg a JSONata expression), or with its `inputs` as args;
    rules  a gate (role "gate"): ordered { when, branch, reason?, confidence? }, first match wins, the LAST rule has no `when`; every branch needs a rule;
    expr   a pure JSONata expression (mapping, validation);
    assert [{ that, message }] — every `that` must be truthy;
    inputs alone: the node's output is its bindings.
- store: persists the result (`target`, `scope`).

Data flow:
- `inputs`: { name: "node.path | default" } — a ref names a node with an edge INTO this node. A default after `|` is JSON.
  "$input.x" reads the pipeline input.
- An expression's root is the node's `inputs` (when declared), else an object keyed by the ids of nodes with an edge into it.
- Expression bindings: $input, $t (the node's thresholds), $primary, $branches, $probes (never inside a gate), plus the registry's $functions.
- A gate passes its input through: downstream of gate "route", read what reached it as `route.<name>`.
- Edges: { from, to } or, out of a gate, { from, to, when: { gate, branch } }.
- `result` lists the pipeline's routes: the branches that end in its answer.
- `thresholds`: { name: number } or { name: { env, default } }.
- Every node has a `version` string. Keep a node's version when you keep its meaning; bump it when you change what it does.

Only use primitives, functions and schemas from the catalog. Reuse the base's prompts, routes, question sets and store targets; you cannot invent new ones.

Reply with a one-paragraph rationale, then the WHOLE spec as ONE ```json block."""


def _js_numbers(value: Any) -> Any:
    """Integral floats as ints, so `json.dumps` prints `4` where `JSON.stringify` does, not `4.0`."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value == int(value) and abs(value) < 2**53:
        return int(value)
    if isinstance(value, dict):
        return {k: _js_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_js_numbers(v) for v in value]
    return value


def _json_default(o: Any) -> Any:
    plain = to_plain(o)
    return str(o) if plain is o else plain


def _pretty(value: Any) -> str:
    """`JSON.stringify(x, null, 2)`."""
    return json.dumps(_js_numbers(to_plain(value)), indent=2, ensure_ascii=False, default=_json_default)


def _compact(value: Any) -> str:
    """`JSON.stringify(x)`."""
    return json.dumps(_js_numbers(to_plain(value)), separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _system_prompt(registry: SpecRegistry) -> str:
    return "\n\n".join(
        [
            SPEC_LANGUAGE,
            "## Catalog\n```json\n" + _pretty(describe_registry(registry)) + "\n```",
            "## Spec JSON Schema\n```json\n" + _compact(pipeline_spec_json_schema()) + "\n```",
        ]
    )


def _first_prompt(goal: str, base: Optional[Mapping[str, Any]], vocabulary: Mapping[str, Optional[list[str]]]) -> str:
    parts = [f"## Goal\n{goal}"]
    if base:
        parts.append("## Base spec (vary this)\n```json\n" + _pretty(base) + "\n```")
    names = [(k, v) for k, v in vocabulary.items() if v is not None]
    if names:
        parts.append("## Host names you may use\n" + "\n".join(f"- {k}: {', '.join(v) or '(none)'}" for k, v in names))
    return "\n\n".join(parts)


def _repair_prompt(goal: str, previous: Any, problems: list[str]) -> str:
    return "\n\n".join(
        [
            f"## Goal\n{goal}",
            "## Your previous spec\n```json\n" + ("(none — the reply held no JSON object)" if previous is None else _pretty(previous)) + "\n```",
            "## Problems (fix ALL of them)\n" + "\n".join(f"- {p}" for p in problems),
        ]
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _vocabulary_dict(v: Any) -> dict[str, Optional[list[str]]]:
    """
    A vocabulary as the keys it SETS, like a TS object literal: a dataclass's
    `None` fields are left out (so they do not erase the base's names), a dict's
    keys are kept as given (an explicit `None` is TS's explicit `undefined`).
    """
    if v is None:
        return {}
    if isinstance(v, SpecVocabulary):
        return {k: list(getattr(v, k)) for k in _VOCAB_KEYS if getattr(v, k) is not None}
    return {k: (None if x is None else list(x)) for k, x in dict(v).items()}


def _throw_if_aborted(signal: Any) -> None:
    """`signal?.throwIfAborted()` for whatever the host passes: a `throw_if_aborted()` method, or an `is_set()` / `aborted` flag."""
    if signal is None:
        return
    check = getattr(signal, "throw_if_aborted", None)
    if callable(check):
        check()
        return
    is_set = getattr(signal, "is_set", None)
    if (callable(is_set) and is_set()) or getattr(signal, "aborted", False) is True:
        raise Exception("This operation was aborted")


async def propose_pipeline_spec(
    *,
    goal: str,
    registry: SpecRegistry,
    engine: ExpressionEngine,
    author: SpecAuthor,
    base: Optional[Mapping[str, Any]] = None,
    vocabulary: Optional[Union[SpecVocabulary, Mapping[str, Optional[list[str]]]]] = None,
    max_rounds: Optional[int] = 3,
    signal: Any = None,
) -> Proposal:
    """
    `base`: the spec being varied — without one the model writes from scratch and
    only the compiler checks it. `vocabulary`: default the base's own names
    (nothing, without a base: every name allowed). `max_rounds`: default 3, one
    write and two repairs.
    """
    rounds_max = max(1, 3 if max_rounds is None else max_rounds)
    allowed: dict[str, Optional[list[str]]] = {**(_vocabulary_dict(spec_vocabulary(base)) if base else {}), **_vocabulary_dict(vocabulary)}
    system = _system_prompt(registry)
    rounds: list[ProposalRound] = []

    prompt = _first_prompt(goal, base, allowed)
    for round_ in range(1, rounds_max + 1):
        _throw_if_aborted(signal)
        reply = await maybe_await(author(AuthorRequest(system=system, prompt=prompt, round=round_, signal=signal)))
        extracted = extract_spec_json(reply)
        if "problem" in extracted:
            rounds.append(ProposalRound(round=round_, spec=None, problems=[extracted["problem"]]))
            prompt = _repair_prompt(goal, None, [extracted["problem"]])
            continue
        value = extracted["value"]
        compiled = compile_pipeline_spec(value, registry, engine)
        problems = list(compiled.problems)
        # Only a spec that parsed has fields to hold against the base and the vocabulary.
        if compiled.pipeline:
            spec = value
            problems = [*_vocabulary_problems(spec, allowed), *(_contract_problems(spec, base) if base else [])]
            diff = diff_specs(base, spec) if base else None
            if diff is not None and _is_empty_diff(diff):
                problems.append("the proposal is identical to the base — change something toward the goal")
            if not problems:
                rounds.append(ProposalRound(round=round_, spec=spec, problems=problems))
                return Proposal(spec=spec, pipeline=compiled.pipeline, rounds=rounds, problems=problems, diff=diff)
        rounds.append(ProposalRound(round=round_, spec=value, problems=problems))
        prompt = _repair_prompt(goal, value, problems)
    return Proposal(spec=None, pipeline=None, rounds=rounds, problems=rounds[-1].problems, diff=None)
