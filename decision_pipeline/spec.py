"""
Pipelines as data — a JSON spec that compiles into the same manifest a Python
author writes with `define_pipeline`, so every contract check still runs.

A spec's code node is one of

  call    a registered PRIMITIVE (`cuisine.jevLeafResult@1`) — the domain
          function with a schema on its args.
  rules   a gate as an ordered list of `{ when, branch, reason }`, first
          match wins, the last rule a catch-all.
  expr    a pure expression, for mapping and validation.
  inputs  alone: a node whose output is its bindings.

Expressions go through an injected `ExpressionEngine` (JSONata in
`decision_pipeline.jsonata`). An expression sees the node's resolved `inputs`
as its root (or `from`, by node id, when it declares none) and `$input`, `$t`
(thresholds), `$primary`, `$branches`, `$probes` (never for a gate) and the
registry's functions as bindings.

A compiled node's version is the declared one plus a hash of its body
(`1~3fa9…`): editing an expression IS a new node version.

The JSON shape is the TS one, camelCase keys included.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, ValidationError

from ._compat import SchemaError, as_schema, maybe_await, pydantic_problems, to_plain
from .define import DecisionPipeline, EvalSpec, Overridable, PipelineSpec, Precedence, define_pipeline
from .errors import PipelineContractError
from .expressions import CompiledExpression, ExpressionEngine
from .hash import short_hash
from .schema import (
    CacheLiteral,
    CodeRoleLiteral,
    EdgeSchema,
    EvalSchema,
    ExperimentSchema,
    FeedbackEdgeSchema,
    OnFailureLiteral,
    ProbeSpecSchema,
    RankLiteral,
    ScopeLiteral,
    ThresholdSchema,
    ToolSpecSchema,
    TriggerLiteral,
)
from .types import (
    CodeNode,
    DecideNode,
    Edge,
    EdgeGuard,
    FeedbackEdge,
    GateNode,
    GenerateNode,
    NodeRunArgs,
    ProbeSpec,
    ReadNode,
    ScoreCategory,
    ScoreConfigSpec,
    StoreNode,
    ToolSelectable,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PrimitiveContext:
    input: Any
    thresholds: Mapping[str, float]
    node: Any
    signal: Any = None


@dataclass(kw_only=True)
class Primitive:
    #: Dotted, domain first: `cuisine.jevLeafResult`. Referenced as `id@version`.
    id: str
    version: str
    #: What it does, for the catalog a model reads.
    describe: str
    #: Validated on every call: a pydantic model class, a TypeAdapter, or anything with `parse()`.
    args: Any
    #: `(args, ctx) -> result`, sync or async.
    run: Callable[[Any, PrimitiveContext], Any]


RegisteredPrimitive = Primitive


@dataclass(kw_only=True)
class ExpressionFunction:
    describe: str
    #: Called from inside expressions with whatever the expression passes.
    fn: Callable[..., Any]


FeatureFn = Callable[[Any], dict[str, float]]


@dataclass(kw_only=True)
class SpecRegistry:
    #: Keyed `id@version`.
    primitives: dict[str, Primitive] = field(default_factory=dict)
    #: Exposed to expressions as `$<name>`.
    functions: dict[str, ExpressionFunction] = field(default_factory=dict)
    #: Input / output schemas a spec names.
    schemas: dict[str, Any] = field(default_factory=dict)
    features: dict[str, FeatureFn] = field(default_factory=dict)


def primitive_ref(p: Any) -> str:
    return f"{p.id}@{p.version}"


#: Names an expression already has — a registry function may not shadow them.
RESERVED_BINDINGS = ["input", "t", "primary", "branches", "probes"]

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def define_primitive(p: Optional[Primitive] = None, /, **kwargs: Any) -> Primitive:
    return p if p is not None else Primitive(**kwargs)


def _function(value: Any) -> ExpressionFunction:
    if isinstance(value, ExpressionFunction):
        return value
    if isinstance(value, Mapping):
        return ExpressionFunction(describe=value.get("describe", ""), fn=value["fn"])
    return ExpressionFunction(describe="", fn=value)


def define_registry(
    *,
    primitives: Optional[list[Primitive]] = None,
    functions: Optional[Mapping[str, Any]] = None,
    schemas: Optional[Mapping[str, Any]] = None,
    features: Optional[Mapping[str, FeatureFn]] = None,
) -> SpecRegistry:
    prims: dict[str, Primitive] = {}
    for p in primitives or []:
        ref = primitive_ref(p)
        if ref in prims:
            raise ValueError(f'registry: primitive "{ref}" registered twice')
        prims[ref] = p
    for name in (functions or {}).keys():
        if not _IDENT_RE.match(name):
            raise ValueError(f'registry: function name "{name}" is not an identifier')
        if name in RESERVED_BINDINGS:
            raise ValueError(f'registry: function "${name}" would shadow a reserved binding')
    return SpecRegistry(
        primitives=prims,
        functions={k: _function(v) for k, v in (functions or {}).items()},
        schemas=dict(schemas or {}),
        features=dict(features or {}),
    )


def merge_registries(*registries: SpecRegistry) -> SpecRegistry:
    """Later registries win nothing — a clash is an error, like a double registration."""
    out = SpecRegistry()
    for r in registries:
        for k in ("primitives", "functions", "schemas", "features"):
            target = getattr(out, k)
            for name, v in getattr(r, k).items():
                if name in target:
                    raise ValueError(f'registry: {k} "{name}" defined twice')
                target[name] = v
    return out


# ---------------------------------------------------------------------------
# The spec's shape
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


Expr = Field(min_length=1)
PRIMITIVE_REF_RE = r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+$"
PrimitiveRef = Field(pattern=PRIMITIVE_REF_RE)


class FallbackValue(_Strict):
    value: Any = None


class FallbackExpr(_Strict):
    expr: str = Expr


class FallbackCall(_Strict):
    call: str = PrimitiveRef
    args: Optional[dict[str, str]] = None


def _fallback_tag(v: Any) -> Optional[str]:
    if isinstance(v, Mapping):
        if "value" in v:
            return "value"
        if "expr" in v:
            return "expr"
        if "call" in v:
            return "call"
    return None


FallbackSpec = Annotated[
    Union[Annotated[FallbackValue, Tag("value")], Annotated[FallbackExpr, Tag("expr")], Annotated[FallbackCall, Tag("call")]],
    Discriminator(_fallback_tag),
]


class _Common(_Strict):
    version: str = Field(min_length=1)
    describe: Optional[str] = None
    on_failure: Optional[OnFailureLiteral] = Field(default=None, alias="onFailure")
    timeout_ms: Optional[int] = Field(default=None, alias="timeoutMs", gt=0)
    cache: Optional[CacheLiteral] = None
    inputs: Optional[dict[str, str]] = None
    fallback: Optional[FallbackSpec] = None


class ReadSpec(_Common):
    kind: Literal["read"]
    call: str = PrimitiveRef
    args: Optional[dict[str, str]] = None
    provides_features: Optional[bool] = Field(default=None, alias="providesFeatures")


class DecideSpec(_Common):
    kind: Literal["decide"]
    questions: Optional[str] = Field(default=None, min_length=1)
    probes: Optional[ProbeSpecSchema] = None
    state: Optional[list[str]] = None
    accepts_features: Optional[bool] = Field(default=None, alias="acceptsFeatures")
    batched: Optional[bool] = None


class GenerateSpec(_Common):
    kind: Literal["generate"]
    prompt: str = Field(min_length=1)
    route: str = Field(min_length=1)
    submit: Optional[str] = Field(default=None, min_length=1)
    tools: Optional[ToolSpecSchema] = None


class AssertSpec(_Strict):
    that: str = Expr
    message: str = Field(min_length=1)


class CodeSpec(_Common):
    kind: Literal["code"]
    role: Literal["transform", "validate", "derive"]
    thresholds: Optional[dict[str, ThresholdSchema]] = None
    call: Optional[str] = Field(default=None, pattern=PRIMITIVE_REF_RE)
    args: Optional[dict[str, str]] = None
    expr: Optional[str] = Field(default=None, min_length=1)
    #: Every `that` must be truthy or the node fails with `message`.
    assert_: Optional[list[AssertSpec]] = Field(default=None, alias="assert", min_length=1)


class GateRuleSpec(_Strict):
    #: Absent = always matches. Only the LAST rule may omit it.
    when: Optional[str] = Field(default=None, min_length=1)
    branch: str = Field(min_length=1)
    reason: Optional[str] = Field(default=None, min_length=1)
    confidence: Optional[str] = Field(default=None, min_length=1)


class GateSpec(_Common):
    kind: Literal["code"]
    role: Literal["gate"]
    thresholds: dict[str, ThresholdSchema]
    branches: list[str] = Field(min_length=1)
    rules: Optional[list[GateRuleSpec]] = Field(default=None, min_length=1)
    call: Optional[str] = Field(default=None, pattern=PRIMITIVE_REF_RE)
    args: Optional[dict[str, str]] = None


class StoreSpec(_Common):
    kind: Literal["store"]
    target: str = Field(min_length=1)
    scope: ScopeLiteral
    ttl: Optional[str] = Field(default=None, min_length=1)


def _node_tag(v: Any) -> Optional[str]:
    if isinstance(v, BaseModel):
        return "gate" if isinstance(v, GateSpec) else getattr(v, "kind", None)
    if isinstance(v, Mapping):
        kind = v.get("kind")
        if kind == "code":
            return "gate" if v.get("role") == "gate" else "code"
        return kind if isinstance(kind, str) else None
    return None


NodeSpec = Annotated[
    Union[
        Annotated[ReadSpec, Tag("read")],
        Annotated[DecideSpec, Tag("decide")],
        Annotated[GenerateSpec, Tag("generate")],
        Annotated[GateSpec, Tag("gate")],
        Annotated[CodeSpec, Tag("code")],
        Annotated[StoreSpec, Tag("store")],
    ],
    Discriminator(_node_tag),
]
node_spec_schema: TypeAdapter[Any] = TypeAdapter(NodeSpec)
NodeSpecJSON = dict


class FeedbackSpec(FeedbackEdgeSchema):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    key: None = None


class OverridableSpec(_Strict):
    scope: ScopeLiteral
    key: str = Field(min_length=1)
    key_from_input: Optional[str] = Field(default=None, alias="keyFromInput", min_length=1)


class PrecedenceSpec(_Strict):
    over: list[str]


class PipelineSpecModel(_Strict):
    id: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    #: Registry schema names.
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    trigger: list[TriggerLiteral] = Field(min_length=1)
    #: Template over the input — the grouping key.
    group: str = Field(min_length=1)
    result: Union[str, list[str]]
    decided_by: Optional[RankLiteral] = Field(default=None, alias="decidedBy")
    eval: EvalSchema
    precedence: Optional[PrecedenceSpec] = None
    knowledge_version: Optional[str] = Field(default=None, alias="knowledgeVersion", min_length=1)
    overridable: Optional[OverridableSpec] = None
    nodes: dict[str, NodeSpec]
    edges: list[EdgeSchema]
    feedback: list[FeedbackSpec]
    experiments: Optional[list[ExperimentSchema]] = None
    #: Registry feature-extractor name.
    features: Optional[str] = Field(default=None, min_length=1)


pipeline_spec_schema: TypeAdapter[Any] = TypeAdapter(PipelineSpecModel)
PipelineSpecJSON = dict


def pipeline_spec_json_schema() -> dict[str, Any]:
    """The spec's JSON Schema — what a model is handed for structured output."""
    return pipeline_spec_schema.json_schema(mode="validation", by_alias=True)


def describe_registry(registry: SpecRegistry) -> dict[str, Any]:
    """What a model needs to know to write a spec against this registry."""
    return {
        "primitives": [{"ref": ref, "describe": p.describe, "args": as_schema(p.args).json_schema()} for ref, p in registry.primitives.items()],
        "functions": [{"name": f"${name}", "describe": f.describe} for name, f in registry.functions.items()],
        "schemas": list(registry.schemas.keys()),
        "features": list(registry.features.keys()),
    }


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}")


def _get(v: Any, seg: str) -> Any:
    if v is None:
        return None
    if isinstance(v, Mapping):
        return v.get(seg)
    return getattr(v, seg, None)


def compile_template(src: str) -> Callable[[Any], str]:
    """`{path}` placeholders over an object: `cuisine-{pointId}`."""

    def render(value: Any) -> str:
        def sub(m: "re.Match[str]") -> str:
            v = value
            for seg in m.group(1).split("."):
                v = _get(v, seg)
            return "" if v is None else str(v)

        return PLACEHOLDER_RE.sub(sub, src)

    return render


_BODY_KEYS = ["inputs", "call", "args", "expr", "rules", "assert", "fallback"]


def _body_of(n: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Everything that changes what a node DOES, for its version hash."""
    body = {k: n[k] for k in _BODY_KEYS if n.get(k) is not None}
    return body or None


@dataclass
class CompileResult:
    pipeline: Optional[DecisionPipeline]
    #: Empty when `pipeline` is set. Every problem at once, so a model can fix them in one round.
    problems: list[str]


@dataclass
class _Readable:
    names: set[str]
    say: str


def _threshold_plain(t: Any) -> Any:
    if isinstance(t, BaseModel):
        return t.model_dump(by_alias=True, exclude_none=True)
    return t


def _tool_spec(t: Optional[ToolSpecSchema]) -> Optional[ToolSpec]:
    if t is None:
        return None
    sel = None
    if t.selectable is not None:
        sel = ToolSelectable(candidates=list(t.selectable.candidates), from_=t.selectable.from_, gate=t.selectable.gate, threshold=_threshold_plain(t.selectable.threshold))
    return ToolSpec(static=t.static, selectable=sel, allow=t.allow)


def _probe_spec(p: Optional[ProbeSpecSchema]) -> Optional[ProbeSpec]:
    if p is None:
        return None
    return ProbeSpec(from_=p.from_, max=p.max, paths=p.paths, max_chars=p.max_chars)


def _feedback(f: FeedbackSpec) -> FeedbackEdge:
    score = None
    if f.score is not None:
        score = ScoreConfigSpec(
            name=f.score.name,
            data_type=f.score.data_type,
            categories=[ScoreCategory(label=c.label, value=c.value) for c in f.score.categories] if f.score.categories is not None else None,
            description=f.score.description,
        )
    return FeedbackEdge(id=f.id, from_=f.from_, to=f.to, source=f.source, scope=f.scope, form=f.form, latency=f.latency, score=score, promote_via=f.promote_via)


def _edge(e: EdgeSchema) -> Edge:
    return Edge(from_=e.from_, to=e.to, when=EdgeGuard(gate=e.when.gate, branch=e.when.branch) if e.when else None)


def compile_pipeline_spec(json: Any, registry: SpecRegistry, engine: ExpressionEngine) -> CompileResult:
    """
    JSON -> manifest. Never throws: a spec a model wrote comes back with every
    problem listed (shape, registry refs, expression syntax, then the full
    `define_pipeline` contract), which is the repair loop's input.
    """
    try:
        spec: PipelineSpecModel = pipeline_spec_schema.validate_python(json)
    except ValidationError as err:
        return CompileResult(pipeline=None, problems=pydantic_problems(err))
    problems: list[str] = []
    raw_nodes: Mapping[str, Any] = json.get("nodes", {}) if isinstance(json, Mapping) else {}

    def readable_for(nid: str, inputs: Optional[Mapping[str, str]]) -> _Readable:
        """What an expression on a node may read off its root: its inputs, or the nodes with an edge into it."""
        if inputs is not None:
            return _Readable(set(inputs.keys()), f"one of its inputs ({', '.join(inputs.keys())})")
        inbound = [e.from_ for e in spec.edges if e.to == nid]
        gates = [f for f in inbound if isinstance(spec.nodes.get(f), GateSpec)]
        hint = f' — a gate passes its input through, so what reached "{gates[0]}" is read as `{gates[0]}`' if gates else ""
        return _Readable(
            set(inbound),
            f"a node with an edge into it ({', '.join(inbound)}){hint}" if inbound else "anything: it has no inbound edge and no inputs (use $input)",
        )

    def compile_expr(where: str, source: str, readable: Optional[_Readable] = None) -> Optional[CompiledExpression]:
        try:
            c = engine.compile(source)
        except Exception as err:  # noqa: BLE001 — a syntax error is a problem, not a crash
            problems.append(f"{where}: expression does not parse: {err}")
            return None
        roots = c.roots() if readable is not None and hasattr(c, "roots") else None
        if readable is not None and roots:
            for name in roots:
                if name not in readable.names:
                    problems.append(f"{where}: reads `{name}`, which is not {readable.say}")
        return c

    def primitive(where: str, ref: str) -> Optional[Primitive]:
        p = registry.primitives.get(ref)
        if not p:
            problems.append(f'{where}: unknown primitive "{ref}" (registered: {", ".join(registry.primitives.keys()) or "none"})')
            return None
        return p

    def schema(where: str, name: str) -> Any:
        s = registry.schemas.get(name)
        if s is None:
            problems.append(f'{where}: unknown schema "{name}" (registered: {", ".join(registry.schemas.keys()) or "none"})')
        return s

    fns = {k: f.fn for k, f in registry.functions.items()}

    def bindings_for(args: NodeRunArgs, gate: bool) -> dict[str, Any]:
        return {
            **fns,
            "input": to_plain(args.input),
            "t": dict(args.thresholds),
            "primary": to_plain(args.primary),
            "branches": to_plain(args.branches),
            "probes": {} if gate else to_plain(args.probes),
        }

    def root_for(args: NodeRunArgs, declares_inputs: bool) -> Any:
        return to_plain(args.in_) if declares_inputs else to_plain(args.from_)

    def ctx_for(args: NodeRunArgs) -> PrimitiveContext:
        return PrimitiveContext(input=args.input, thresholds=args.thresholds, node=args.node, signal=args.signal)

    def call_body(where: str, ref: str, arg_exprs: Optional[Mapping[str, str]], readable: _Readable, declares_inputs: bool, gate: bool = False) -> Any:
        """A body that calls a primitive with `args` expressions, or with the node's bindings."""
        p = primitive(where, ref)
        compiled = [(k, compile_expr(f"{where}.args.{k}", src, readable)) for k, src in (arg_exprs or {}).items()]
        if not p or any(c is None for _, c in compiled):
            return None
        arg_schema = as_schema(p.args)

        async def run(args: NodeRunArgs) -> Any:
            if arg_exprs is not None:
                root = root_for(args, declares_inputs)
                b = bindings_for(args, gate)
                raw: Any = {}
                for k, c in compiled:
                    raw[k] = await maybe_await(c.evaluate(root, b))  # type: ignore[union-attr]
            else:
                raw = dict(args.in_)
            try:
                checked = arg_schema.parse(raw)
            except SchemaError as err:
                raise Exception(f"{where}: args for {ref} do not match its schema — {'; '.join(err.problems)}") from err
            return await maybe_await(p.run(checked, ctx_for(args)))

        return run

    def expr_body(where: str, source: str, readable: _Readable, declares_inputs: bool) -> Any:
        c = compile_expr(where, source, readable)
        if c is None:
            return None

        async def run(args: NodeRunArgs) -> Any:
            return await maybe_await(c.evaluate(root_for(args, declares_inputs), bindings_for(args, False)))

        return run

    def fallback_body(where: str, f: Any, readable: _Readable, declares_inputs: bool) -> Any:
        if isinstance(f, FallbackValue):
            value = f.value
            return lambda _args: copy.deepcopy(value)
        if isinstance(f, FallbackExpr):
            return expr_body(f"{where}.fallback", f.expr, readable, declares_inputs)
        return call_body(f"{where}.fallback", f.call, f.args, readable, declares_inputs)

    nodes: dict[str, Any] = {}
    for nid, n in spec.nodes.items():
        where = f"nodes.{nid}"
        declares_inputs = n.inputs is not None
        readable = readable_for(nid, n.inputs)
        body = _body_of(raw_nodes.get(nid, {}) if isinstance(raw_nodes.get(nid), Mapping) else {})
        version = f"{n.version}~{short_hash(body)[:8]}" if body else n.version
        base: dict[str, Any] = {"version": version, "describe": n.describe, "on_failure": n.on_failure, "timeout_ms": n.timeout_ms, "cache": n.cache, "inputs": n.inputs}
        fallback = fallback_body(where, n.fallback, readable, declares_inputs) if n.fallback is not None else None
        if n.on_failure == "fallback" and n.fallback is None:
            problems.append(f'{where}: onFailure "fallback" needs a `fallback`')
        if fallback is not None:
            base["fallback"] = fallback

        if isinstance(n, ReadSpec):
            load = call_body(where, n.call, n.args, readable, declares_inputs)
            if load:
                nodes[nid] = ReadNode(load=load, provides_features=n.provides_features, **base)
        elif isinstance(n, DecideSpec):
            nodes[nid] = DecideNode(questions=n.questions, probes=_probe_spec(n.probes), state=n.state, accepts_features=n.accepts_features, batched=n.batched, **base)
        elif isinstance(n, GenerateSpec):
            nodes[nid] = GenerateNode(prompt=n.prompt, route=n.route, submit=n.submit, tools=_tool_spec(n.tools), **base)
        elif isinstance(n, StoreSpec):
            nodes[nid] = StoreNode(target=n.target, scope=n.scope, ttl=n.ttl, **base)
        elif isinstance(n, GateSpec):
            if bool(n.rules) == bool(n.call):
                problems.append(f"{where}: a gate needs exactly one of `rules` or `call`")
                continue
            run: Any = None
            if n.call:
                run = call_body(where, n.call, n.args, readable, declares_inputs, True)
            if n.rules:
                rules = n.rules
                for i, r in enumerate(rules):
                    if r.branch not in n.branches:
                        problems.append(f'{where}.rules[{i}]: branch "{r.branch}" is not one of {", ".join(n.branches)}')
                    if r.when is None and i != len(rules) - 1:
                        problems.append(f"{where}.rules[{i}]: only the last rule may omit `when` — nothing after it is reachable")
                if rules[-1].when is not None:
                    problems.append(f"{where}: the last rule must have no `when` — a gate needs a branch for every input")
                for b in n.branches:
                    if not any(r.branch == b for r in rules):
                        problems.append(f'{where}: no rule produces branch "{b}"')
                compiled_rules = [
                    {
                        "branch": r.branch,
                        "when": None if r.when is None else compile_expr(f"{where}.rules[{i}].when", r.when, readable),
                        "reason": None if r.reason is None else compile_expr(f"{where}.rules[{i}].reason", r.reason, readable),
                        "confidence": None if r.confidence is None else compile_expr(f"{where}.rules[{i}].confidence", r.confidence, readable),
                    }
                    for i, r in enumerate(rules)
                ]

                def make_rules_body(compiled_rules: list[dict[str, Any]], declares_inputs: bool, nid: str) -> Any:
                    async def run_rules(args: NodeRunArgs) -> Any:
                        root = root_for(args, declares_inputs)
                        b = bindings_for(args, True)
                        for r in compiled_rules:
                            if r["when"] is not None and not await maybe_await(r["when"].evaluate(root, b)):
                                continue
                            reason = await maybe_await(r["reason"].evaluate(root, b)) if r["reason"] is not None else None
                            confidence = await maybe_await(r["confidence"].evaluate(root, b)) if r["confidence"] is not None else None
                            out: dict[str, Any] = {"branch": r["branch"]}
                            if reason is not None:
                                out["reason"] = str(reason)
                            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                                out["confidence"] = confidence
                            return out
                        # Unreachable once the catch-all rule is checked; kept for a spec compiled with problems ignored.
                        raise Exception(f"{nid}: no gate rule matched")

                    return run_rules

                run = make_rules_body(compiled_rules, declares_inputs, nid)
            if run:
                nodes[nid] = GateNode(run=run, thresholds={k: _threshold_plain(v) for k, v in n.thresholds.items()}, branches=list(n.branches), **base)
        elif isinstance(n, CodeSpec):
            bodies = sum(1 for x in (n.call, n.expr) if x is not None)
            if bodies > 1:
                problems.append(f"{where}: a code node takes `call` or `expr`, not both")
                continue
            if n.args is not None and n.call is None:
                problems.append(f"{where}: `args` only goes with `call`")
            if bodies == 0 and n.assert_ is None and not declares_inputs:
                problems.append(f"{where}: a code node needs `call`, `expr`, `assert` or `inputs`")
                continue
            main: Any
            if n.call:
                main = call_body(where, n.call, n.args, readable, declares_inputs)
            elif n.expr:
                main = expr_body(where, n.expr, readable, declares_inputs)
            elif n.assert_:
                main = lambda args: args.primary  # noqa: E731
            else:
                main = lambda args: dict(args.in_)  # noqa: E731
            asserts = [(a.message, compile_expr(f"{where}.assert[{i}]", a.that, readable)) for i, a in enumerate(n.assert_ or [])]
            if main is None:
                continue

            def make_code_body(main: Any, asserts: list[tuple[str, Any]], declares_inputs: bool, nid: str) -> Any:
                if not asserts:
                    return main

                async def run_code(args: NodeRunArgs) -> Any:
                    root = root_for(args, declares_inputs)
                    b = bindings_for(args, False)
                    for message, that in asserts:
                        if that is not None and not await maybe_await(that.evaluate(root, b)):
                            raise Exception(f"{nid}: {message}")
                    return await maybe_await(main(args))

                return run_code

            nodes[nid] = CodeNode(
                role=n.role,
                run=make_code_body(main, asserts, declares_inputs, nid),
                thresholds={k: _threshold_plain(v) for k, v in n.thresholds.items()} if n.thresholds is not None else None,
                **base,
            )

    input_schema = schema("input", spec.input)
    output_schema = schema("output", spec.output)
    features: Optional[FeatureFn] = None
    if spec.features:
        features = registry.features.get(spec.features)
        if not features:
            problems.append(f'features: unknown extractor "{spec.features}" (registered: {", ".join(registry.features.keys()) or "none"})')
    if problems:
        return CompileResult(pipeline=None, problems=problems)

    group = compile_template(spec.group)
    from .experiments import experiment_spec_from_dict

    manifest = PipelineSpec(
        id=spec.id,
        fact=spec.fact,
        input=input_schema,
        output=output_schema,
        trigger=list(spec.trigger),
        group=group,
        nodes=nodes,
        edges=[_edge(e) for e in spec.edges],
        feedback=[_feedback(f) for f in spec.feedback],
        result=spec.result,
        eval=EvalSpec(dataset=spec.eval.dataset, gate_metric=spec.eval.gate_metric),
        decided_by=spec.decided_by,
        precedence=Precedence(over=list(spec.precedence.over)) if spec.precedence else None,
        knowledge_version=spec.knowledge_version,
        experiments=[experiment_spec_from_dict(e.model_dump(by_alias=True, exclude_none=True)) for e in spec.experiments] if spec.experiments is not None else None,
        features=features,
        overridable=Overridable(
            scope=spec.overridable.scope,
            key=compile_template(spec.overridable.key),
            key_from_input=compile_template(spec.overridable.key_from_input) if spec.overridable.key_from_input else None,
        )
        if spec.overridable
        else None,
    )
    try:
        return CompileResult(pipeline=define_pipeline(manifest), problems=[])
    except PipelineContractError as err:
        return CompileResult(pipeline=None, problems=err.problems)


def pipeline_from_spec(json: Any, registry: SpecRegistry, engine: ExpressionEngine) -> DecisionPipeline:
    """`compile_pipeline_spec`, raising a `PipelineContractError` with every problem."""
    result = compile_pipeline_spec(json, registry, engine)
    if not result.pipeline:
        pid = json["id"] if isinstance(json, Mapping) and isinstance(json.get("id"), str) else "(spec)"
        raise PipelineContractError(pid, result.problems)
    return result.pipeline
