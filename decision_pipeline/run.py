"""
`run_pipeline` — the runtime.

What it takes off every pipeline author's hands: topological order and the
skip rule, ONE root trace with one child span per node, the shared decide
runner, threshold resolution, the gate's branch bookkeeping, tool
resolution, on_failure, caching, the persistence contract, and the run
record. What it deliberately does NOT do: anything a node body does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union

from ._compat import MISSING, error_message, iso_from_ms, maybe_await, now_ms
from .bindings import resolve_inputs
from .define import DecisionPipeline, FeatureExtractionArgs
from .errors import MissingPortError, NodeFailedError, NodeSkipped, NodeTimeoutError, UnknownBranchError
from .experiments import ArmAssignment, ManifestPatch, assign_arm, merge_patches
from .box import BOX_CONFIDENCE_OPTION, box_features, check_box_answer
from .features import distribution_features
from .graph import resolve_inbound
from .hash import short_hash
from .observers import notify_observers, record_sink_observer
from .patch import apply_manifest_patch
from .ports import (
    DecideRequest,
    DecideResult,
    GenerateRequest,
    GenerateResult,
    NodeSpanOpts,
    PipelinePorts,
    PromptResolution,
    RootSpanOpts,
    RunOptions,
    ScopeResolveRequest,
    StoreRequest,
)
from .probes import ProbeRecord, ProbeDrop, collect_probes, probe_answers, probe_features, split_probe_distributions
from .record import RunRecord, RunRecordNode
from .scope import ScopeRejection, build_scope_context
from .thresholds import resolve_thresholds, threshold_values
from .tools import DEFAULT_TOOL_SELECT_THRESHOLD, ResolvedTools, resolve_tools
from .types import (
    AbortController,
    AbortSignal,
    BranchTaken,
    CodeNode,
    DecideNode,
    GateNode,
    GenerateNode,
    NodeRef,
    NodeRunArgs,
    OverrideRow,
    ProbeAnswer,
    ReadNode,
    ScopeContext,
    StoreNode,
    has_thresholds,
    is_gate_node,
)
from .version import VersionInputs, composite_version, node_cache_key


@dataclass(kw_only=True)
class RunResult:
    output: Any
    #: The root trace id. None when the host runs untraced.
    run_id: Optional[str]
    #: The durable identity — `options.run_key`, or this attempt's id.
    run_key: str
    version: str
    #: The branches taken, joined — what lands as `<fact>_route`.
    route: str
    confidence: Optional[float]
    decided_by: str
    arms: dict[str, str]
    record: RunRecord
    #: Node id -> why it did not run. Never silently missing.
    skipped: dict[str, str]
    #: Feedback rows the form x scope rule refused at read.
    scope_rejections: list[ScopeRejection]
    #: The arm/replay diff actually applied to the manifest.
    patch_applied: list[str]


#: What a gate sees in `args.probes`: nothing, ever.
NO_PROBES: Mapping[str, Sequence[ProbeAnswer]] = MappingProxyType({})


class _NullSpan:
    trace_id = None

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        pass

    def set_output(self, output: Any) -> None:
        pass


class _NullTracer:
    """A tracer that does nothing, so a host without Langfuse still runs."""

    async def root(self, opts: RootSpanOpts, fn: Any) -> Any:
        return await fn(_NullSpan())

    async def span(self, opts: NodeSpanOpts, fn: Any) -> Any:
        return await fn(_NullSpan())


NULL_TRACER = _NullTracer()


def normalize_gate(result: Any) -> BranchTaken:
    if isinstance(result, BranchTaken):
        return result
    if isinstance(result, str):
        return BranchTaken(branch=result)
    if isinstance(result, Mapping) and "branch" in result:
        reason = result.get("reason")
        confidence = result.get("confidence")
        return BranchTaken(
            branch=str(result["branch"]),
            reason=reason if isinstance(reason, str) else None,
            confidence=confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
        )
    if hasattr(result, "branch"):
        reason = getattr(result, "reason", None)
        confidence = getattr(result, "confidence", None)
        return BranchTaken(
            branch=str(result.branch),
            reason=reason if isinstance(reason, str) else None,
            confidence=confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
        )
    return BranchTaken(branch=str(result))


def _takes_boxes(fn: Any) -> bool:
    """Whether an `interpret` accepts the TS signature's third argument (`boxes`)."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3


def _as_decide_result(value: Any) -> DecideResult:
    if isinstance(value, DecideResult):
        return value
    if isinstance(value, Mapping):
        d = {_CAMEL.get(k, k): v for k, v in value.items()}
        return DecideResult(**{k: v for k, v in d.items() if k in _DECIDE_FIELDS})
    raise TypeError("a decide port must return a DecideResult (or a dict with `distributions`)")


def _as_generate_result(value: Any) -> GenerateResult:
    if isinstance(value, GenerateResult):
        return value
    if isinstance(value, Mapping) and "value" in value:
        d = {_CAMEL.get(k, k): v for k, v in value.items()}
        return GenerateResult(**{k: v for k, v in d.items() if k in _GENERATE_FIELDS})
    raise TypeError("a generate port must return a GenerateResult (or a dict with `value`)")


def _as_prompt_resolution(value: Any) -> PromptResolution:
    if isinstance(value, PromptResolution):
        return value
    if value is None:
        return PromptResolution(version=None)
    if isinstance(value, Mapping):
        return PromptResolution(version=value.get("version"), tools=value.get("tools"))
    return PromptResolution(version=getattr(value, "version", None), tools=getattr(value, "tools", None))


_CAMEL = {"distributionOptions": "distribution_options", "costUsd": "cost_usd", "batchTraceId": "batch_trace_id", "toolsCalled": "tools_called", "promptVersion": "prompt_version"}
_DECIDE_FIELDS = {f.name for f in dataclasses.fields(DecideResult)}
_GENERATE_FIELDS = {f.name for f in dataclasses.fields(GenerateResult)}


def apply_override(pipeline: DecisionPipeline, output: Any, ctx: ScopeContext) -> tuple[Any, Optional[OverrideRow]]:
    """
    The override, applied AT READ. The pipeline's own answer is untouched beside
    it — that separability is what keeps the gate scorable on corrected runs.
    Returns `(value, override_row_or_None)`.
    """
    if not pipeline.overridable:
        return output, None
    key = pipeline.overridable.key(output)
    row = ctx.override(key)
    return (row.value if row else output), row


async def run_pipeline(
    pipeline: DecisionPipeline,
    raw_input: Any,
    ports: Optional[PipelinePorts] = None,
    options: Optional[RunOptions] = None,
    **option_kwargs: Any,
) -> RunResult:
    ports = ports or PipelinePorts()
    if options is None:
        options = RunOptions(**option_kwargs)
    elif option_kwargs:
        options = dataclasses.replace(options, **option_kwargs)

    clock = ports.clock or now_ms
    tracer = ports.tracer or NULL_TRACER
    # `records` is just the first watcher, kept in its old position so a host
    # that wires both still sees the record land before anything else reads it.
    observers = [*([record_sink_observer(ports.records)] if ports.records else []), *(ports.observers or [])]
    started_at = clock()

    input = pipeline.input.parse(raw_input)
    trigger = options.trigger or (pipeline.trigger[0] if pipeline.trigger else "on-demand")
    group = pipeline.group(input)

    # --- 1. Scope, before node 0. The pipeline never reads a user id.
    feature_nodes = [nid for nid, n in pipeline.nodes.items() if isinstance(n, DecideNode) and n.accepts_features is True]
    override_key = pipeline.overridable.key_from_input(input) if pipeline.overridable and pipeline.overridable.key_from_input else None

    resolved = (
        await maybe_await(ports.scope(ScopeResolveRequest(pipeline=pipeline.id, feature_nodes=feature_nodes, override_key=override_key, input=input)))
        if ports.scope
        else None
    )
    built = build_scope_context(resolved)
    ctx, rejected = built.ctx, built.rejected

    # --- 2. Arms, in the same place and for the same reason as scope.
    arms: dict[str, str] = {}
    assignments: list[ArmAssignment] = []
    arm_patches: list[ManifestPatch] = []
    for experiment in pipeline.experiments or []:
        forced = (options.force_arm or {}).get(experiment.id)
        if forced:
            assignment = ArmAssignment(
                experiment=experiment.id,
                arm=forced,
                unit=experiment.unit,
                hash=None,
                ships=experiment.mode != "shadow" or forced == experiment.champion,
                reason="assigned",
            )
        else:
            assignment = assign_arm(experiment, ctx.ids.get(experiment.unit))
        assignments.append(assignment)
        arms[experiment.id] = assignment.arm
        patch = experiment.arms.get(assignment.arm)
        if patch:
            arm_patches.append(patch)
    # A shadow challenger must NOT write the fact, or it overwrites production.
    writes_fact = options.shadow is not True and all(a.ships for a in assignments)
    assignment_unit = assignments[0].unit if assignments else None
    assignment_hash = assignments[0].hash if assignments else None

    patch = merge_patches([*arm_patches, *([options.patch] if options.patch else [])])
    patched = apply_manifest_patch(pipeline.nodes, pipeline.edges, patch)
    nodes, edges, patch_applied = patched.nodes, patched.edges, patched.applied

    # --- 3. Resolve everything the version hash needs, once, before the run.
    prompt_versions: dict[str, Optional[str]] = {}
    prompt_tools: dict[str, Sequence[str]] = {}
    for nid, n in nodes.items():
        ref = n.prompt if isinstance(n, GenerateNode) else n.questions if isinstance(n, DecideNode) else None
        if not ref:
            continue
        resolution = _as_prompt_resolution(await maybe_await(ports.prompts(ref))) if ports.prompts else PromptResolution(version=None)
        prompt_versions[nid] = resolution.version
        if resolution.tools:
            prompt_tools[nid] = resolution.tools

    gate_thresholds = {}
    for nid, n in nodes.items():
        if not has_thresholds(n):
            continue
        gate_thresholds[nid] = await resolve_thresholds(n.thresholds, ports.thresholds)
    threshold_numbers: dict[str, dict[str, float]] = {nid: threshold_values(r) for nid, r in gate_thresholds.items()}

    node_versions = {nid: n.version for nid, n in nodes.items()}

    version = composite_version(
        VersionInputs(
        pipeline=pipeline.id,
        node_versions=node_versions,
        prompt_versions=prompt_versions,
        thresholds=threshold_numbers,
        knowledge_version=pipeline.knowledge_version if pipeline.knowledge_version is not None else ports.knowledge_version,
        arms=arms,
        )
    )

    input_hash = short_hash(input)

    #: Set once the run has started: what a watcher is told if it then throws.
    report_failure: Any = None

    # --- 4. One root observation. Every node is a child span.
    async def body(root_span: Any) -> RunResult:
        nonlocal report_failure
        run_id = root_span.trace_id
        # Resolved HERE rather than beside the record, because a watcher that
        # fires at run start still has to name the same run the record will.
        attempt_id = run_id if run_id is not None else f"local-{short_hash({'pipeline': pipeline.id, 'inputHash': input_hash, 'startedAt': started_at})}"
        # No host key means this attempt IS the identity. Never null.
        run_key = options.run_key if options.run_key is not None else attempt_id

        from .observers import NodeEndEvent, NodeStartEvent, RunEndEvent, RunFailEvent, RunStartEvent

        await notify_observers(
            observers,
            "on_run_start",
            RunStartEvent(pipeline=pipeline.id, run_id=run_id, run_key=run_key, version=version, input=input, group=group, arms=arms, trigger=trigger, writes_fact=writes_fact),
        )

        outputs: dict[str, Any] = {}
        ran: set[str] = set()
        branches: dict[str, str] = {}
        #: The same verdicts, with the reason, for nodes that must record it.
        verdicts: dict[str, BranchTaken] = {}
        skipped: dict[str, str] = {}
        record_nodes: list[RunRecordNode] = []
        decide_distributions: list[Any] = []
        per_run_cache: dict[str, Any] = {}
        #: Decide node id -> its answered probes. Never handed to a gate.
        probes_by_node: dict[str, Sequence[ProbeAnswer]] = {}
        #: Decide node id -> its checked Box answers. Gates see these.
        boxes_by_node: dict[str, dict[str, Any]] = {}
        decide_boxes: list[Any] = []
        probe_feature_values: dict[str, float] = {}
        cost_usd = 0.0
        last_gate_confidence: Optional[float] = None
        #: Tags earned during the run, applied to the root once at the end.
        earned_tags: list[str] = []

        def earn(tag: str) -> None:
            if tag not in earned_tags:
                earned_tags.append(tag)

        async def _report_failure(err: BaseException) -> None:
            await notify_observers(
                observers,
                "on_run_fail",
                RunFailEvent(
                    pipeline=pipeline.id,
                    run_id=run_id,
                    run_key=run_key,
                    version=version,
                    group=group,
                    trigger=trigger,
                    arms=arms,
                    writes_fact=writes_fact,
                    error=error_message(err),
                    failed_node=err.node_id if isinstance(err, NodeFailedError) else None,
                    nodes=record_nodes,
                    cost_usd=cost_usd,
                    ms=clock() - started_at,
                ),
            )

        report_failure = _report_failure

        async def notify_node(hook: str, entry: RunRecordNode, cache_hit: bool = False, output: Any = MISSING) -> None:
            """Every node the runtime considered gets a start and an end, skips included."""
            base = dict(pipeline=pipeline.id, run_id=run_id, node_id=entry.id, kind=entry.kind, node_version=entry.version, input_hash=entry.input_hash, cache_hit=cache_hit)
            if hook == "on_node_start":
                await notify_observers(observers, "on_node_start", NodeStartEvent(**base))
                return
            await notify_observers(
                observers,
                "on_node_end",
                NodeEndEvent(
                    **base,
                    ms=entry.ms,
                    cost_usd=entry.cost_usd,
                    branch=entry.branch,
                    route=entry.route,
                    output=None if output is MISSING else output,
                    error=entry.error,
                    skipped=entry.skipped,
                ),
            )

        async def note_skip(nid: str, n: Any, reason: str) -> None:
            skipped[nid] = reason
            entry = RunRecordNode(id=nid, kind=n.kind, version=n.version, input_hash="", ms=0, cost_usd=0, skipped=reason)
            record_nodes.append(entry)
            await notify_node("on_node_start", entry)
            await notify_node("on_node_end", entry)

        for nid in pipeline.order:
            n = nodes.get(nid)
            if n is None:
                continue

            inbound = resolve_inbound(edges, nid, ran, lambda gate_id: branches.get(gate_id))
            if inbound.skip_reason is not None:
                await note_skip(nid, n, inbound.skip_reason)
                continue

            from_: dict[str, Any] = {edge.from_: outputs.get(edge.from_) for edge in inbound.open}
            primary = outputs.get(inbound.open[0].from_) if inbound.open else input

            bound = MappingProxyType(resolve_inputs(n.inputs, from_, input))

            node_input_hash = short_hash({"primary": primary, "from": from_})
            thresholds = threshold_numbers.get(nid, {})
            # A node with a deadline gets its own signal: the run's, or the
            # deadline, whichever comes first.
            deadline = AbortController() if n.timeout_ms else None
            signal = AbortSignal.any(options.signal, deadline.signal) if deadline else options.signal
            state: dict[str, Any] = {"timed_out": False, "skip_reason": None}

            async def in_time(pending: Any, *, _nid: str = nid, _n: Any = n, _state: dict[str, Any] = state) -> Any:
                """
                Await a body or a port, then refuse the result if the deadline has
                passed — the run has moved on, and a late answer must not be
                written into it (nor its cost into a record already handed over).
                """
                value = await maybe_await(pending)
                if _state["timed_out"]:
                    raise NodeTimeoutError(_nid, _n.timeout_ms or 0)
                return value

            args = NodeRunArgs(
                input=input,
                from_=from_,
                in_=bound,
                primary=primary,
                ctx=ctx,
                thresholds=thresholds,
                node=NodeRef(id=nid, version=n.version),
                branches=verdicts,
                probes=NO_PROBES if is_gate_node(n) else probes_by_node,
                boxes=boxes_by_node,
                signal=signal,
            )

            cache_mode = n.cache or "none"
            cache_key = node_cache_key(pipeline.id, nid, n.version, node_input_hash)
            cached: Any = None
            cache_hit = False
            if cache_mode == "per-run" and cache_key in per_run_cache:
                cached = per_run_cache[cache_key]
                cache_hit = True
            elif cache_mode == "per-key-forever" and ports.cache:
                hit = await maybe_await(ports.cache.get(cache_key))
                if hit is not None:
                    cached = hit
                    cache_hit = True

            node_start = clock()
            entry = RunRecordNode(id=nid, kind=n.kind, version=n.version, input_hash=node_input_hash, ms=0, cost_usd=0)

            await notify_node("on_node_start", entry, cache_hit=cache_hit)

            def finish(value: Any, span: Any, *, _nid: str = nid, _n: Any = n, _entry: RunRecordNode = entry, _start: float = node_start, _hash: str = node_input_hash, _hit: bool = cache_hit) -> None:
                _entry.ms = clock() - _start
                outputs[_nid] = value
                ran.add(_nid)
                meta: dict[str, Any] = {"step_id": _nid, "kind": _n.kind, "node_version": _n.version, "input_hash": _hash, "cost_usd": _entry.cost_usd}
                if _entry.branch:
                    meta["branch"] = _entry.branch
                if _entry.route:
                    meta["route"] = _entry.route
                if _hit:
                    meta["cache"] = "hit"
                span.set_metadata(meta)
                span.set_output(value)

            async def run_node(span: Any, *, _nid: str = nid, _n: Any = n, _entry: RunRecordNode = entry, _args: NodeRunArgs = args, _from: dict[str, Any] = from_, _primary: Any = primary, _bound: Any = bound, _thresholds: dict[str, float] = thresholds, _cached: Any = cached, _hit: bool = cache_hit) -> None:
                nonlocal cost_usd, last_gate_confidence
                if _hit:
                    _entry.cached = True
                    finish(_cached, span)
                    return

                if isinstance(_n, ReadNode):
                    finish(await in_time(_n.load(_args)), span)
                    return

                if isinstance(_n, DecideNode):
                    if not ports.decide:
                        raise MissingPortError("decide", _nid, "decide")
                    # Probes first: validated before any port call, recorded
                    # even when there are none to ask.
                    probe_record: Optional[ProbeRecord] = None
                    if _n.probes:
                        if _n.probes.from_ in _from:
                            pset = collect_probes(_from[_n.probes.from_], _n.probes)
                            asked, dropped = pset.asked, pset.dropped
                        else:
                            asked, dropped = {}, [ProbeDrop(index=-1, reason=f"{_n.probes.from_} did not run")]
                        probe_record = ProbeRecord(from_=_n.probes.from_, asked=asked, dropped=dropped, unanswered=[])
                        _entry.probes = probe_record
                        span.set_metadata({"probes_asked": len(asked), "probes_dropped": len(dropped)})
                        if not _n.questions and len(asked) == 0:
                            raise Exception(
                                f"no probes to ask ({len(dropped)} dropped: {'; '.join(d.reason for d in dropped) or 'none written'})"
                            )
                    result: Any = _as_decide_result(
                        await in_time(
                            ports.decide(
                                DecideRequest(
                                    node_id=_nid,
                                    node_version=_n.version,
                                    questions=_n.questions,
                                    probes=probe_record.asked if probe_record else {},
                                    state=list(_n.state or []),
                                    features=ctx.features(_nid) if _n.accepts_features else [],
                                    input=input,
                                    from_=_from,
                                    in_=_bound,
                                    batched=_n.batched is True,
                                    signal=_args.signal,
                                )
                            )
                        )
                    )
                    # Boxes are estimates like any distribution: checked here
                    # (a bad one fails the node, so on_failure decides), recorded,
                    # and their confidence mirrored into `distributions` so a gate
                    # thresholds a Box exactly as it does a Noul.
                    boxes: dict[str, Any] = {}
                    distributions = result.distributions
                    distribution_options = result.distribution_options
                    if result.boxes:
                        bad = [(k, why) for k, a in result.boxes.items() if (why := check_box_answer(a)) is not None]
                        if bad:
                            raise Exception("bad box answer: " + "; ".join(f"{k} ({why})" for k, why in bad))
                        distributions = dict(distributions)
                        opts = dict(distribution_options or {})
                        for k, a in result.boxes.items():
                            boxes[k] = {"box": dict(a["box"]), "confidence": a["confidence"]}
                            if k not in distributions:
                                distributions[k] = [a["confidence"]]
                                opts[k] = [BOX_CONFIDENCE_OPTION]
                        distribution_options = opts
                        _entry.boxes = boxes
                    # The record keeps EVERY distribution (replay needs the
                    # probes too); gates, `interpret` and the per-question
                    # features see only the static questions.
                    _entry.distributions = distributions
                    if distribution_options:
                        _entry.distribution_options = distribution_options
                    if result.batch_trace_id:
                        _entry.batch_trace_id = result.batch_trace_id
                    _entry.cost_reported = result.cost_usd is not None
                    _entry.cost_usd = result.cost_usd or 0
                    cost_usd += _entry.cost_usd
                    fixed = split_probe_distributions(distributions).fixed
                    decide_distributions.append({"node_id": _nid, "distributions": fixed})
                    fixed_boxes = {k: v for k, v in boxes.items() if k in fixed}
                    if fixed_boxes:
                        decide_boxes.append({"node_id": _nid, "boxes": fixed_boxes})
                        boxes_by_node[_nid] = fixed_boxes
                    if probe_record:
                        pa = probe_answers(probe_record.asked, distributions, distribution_options)
                        probe_record.unanswered = pa.unanswered
                        probes_by_node[_nid] = pa.answers
                        probe_feature_values.update(probe_features(_nid, probe_record, pa.answers))
                    # The value keeps its shape whether or not a Box was asked (its
                    # confidence is in `fixed`); the edges reach later nodes on
                    # `args.boxes`, and `interpret` here.
                    if _n.interpret:
                        if _takes_boxes(_n.interpret):
                            value = await maybe_await(_n.interpret(fixed, _args, MappingProxyType(fixed_boxes)))
                        else:
                            value = await maybe_await(_n.interpret(fixed, _args))
                    else:
                        value = result.value if result.value is not None else fixed
                    finish(value, span)
                    return

                if isinstance(_n, GenerateNode):
                    if not ports.generate:
                        raise MissingPortError("generate", _nid, "generate")
                    tools = _resolve_tools_for_node(
                        _n,
                        prompt_config_tools=prompt_tools.get(_nid),
                        distributions=_recorded_distributions(record_nodes, _n.tools.selectable.from_) if _n.tools and _n.tools.selectable else None,
                        select_threshold=_tool_select_threshold(_n, threshold_numbers),
                        grants=ctx.grants,
                        registry=ports.tool_registry,
                    )
                    result = _as_generate_result(
                        await in_time(
                            ports.generate(
                                GenerateRequest(
                                    node_id=_nid,
                                    node_version=_n.version,
                                    prompt=_n.prompt,
                                    route=_n.route,
                                    submit=_n.submit,
                                    tools=tools,
                                    input=input,
                                    from_=_from,
                                    in_=_bound,
                                    primary=_primary,
                                    ctx=ctx,
                                    branches=verdicts,
                                    probes=probes_by_node,
                                    signal=_args.signal,
                                )
                            )
                        )
                    )
                    _entry.route = _n.route
                    _entry.tools = {
                        "offered": tools.offered,
                        "selected": tools.selected,
                        "called": list(result.tools_called or []),
                        "dropped_for_scope": tools.dropped_for_scope,
                    }
                    _entry.cost_reported = result.cost_usd is not None
                    _entry.cost_usd = result.cost_usd or 0
                    cost_usd += _entry.cost_usd
                    for tag in result.tags or []:
                        earn(tag)
                    finish(result.value, span)
                    return

                if isinstance(_n, GateNode):
                    gate = normalize_gate(await in_time(_n.run(_args)))
                    if gate.branch not in _n.branches:
                        raise UnknownBranchError(_nid, gate.branch, _n.branches)
                    branches[_nid] = gate.branch
                    verdicts[_nid] = BranchTaken(branch=gate.branch, reason=gate.reason, confidence=gate.confidence)
                    _entry.branch = gate.branch
                    if gate.reason is not None:
                        _entry.branch_reason = gate.reason
                    _entry.thresholds = _thresholds
                    if gate.confidence is not None:
                        last_gate_confidence = gate.confidence
                    meta: dict[str, Any] = {"gate": _nid, "branch": gate.branch}
                    if gate.reason is not None:
                        meta["branch_reason"] = gate.reason
                    meta["thresholds"] = _thresholds
                    span.set_metadata(meta)
                    # A gate routes; it does not transform. Downstream sees what came in.
                    finish(_primary, span)
                    return

                if isinstance(_n, CodeNode):
                    # A transform that applies thresholds records them like a gate does.
                    if _n.thresholds is not None:
                        _entry.thresholds = _thresholds
                        span.set_metadata({"thresholds": _thresholds})
                    finish(await in_time(_n.run(_args)), span)
                    return

                if isinstance(_n, StoreNode):
                    if not ports.store:
                        raise MissingPortError("store", _nid, "store")
                    await in_time(
                        ports.store(
                            StoreRequest(
                                node_id=_nid,
                                target=_n.target,
                                scope=_n.scope,
                                ttl=_n.ttl,
                                fact=pipeline.fact,
                                value=_primary,
                                route=_route_of(branches, pipeline.order),
                                confidence=last_gate_confidence,
                                version=version,
                                run_id=run_id,
                                decided_by=pipeline.decided_by,
                                writes_fact=writes_fact,
                                ctx=ctx,
                            )
                        )
                    )
                    finish(_primary, span)
                    return

                raise TypeError(f'node "{_nid}": unknown node kind {getattr(_n, "kind", None)!r}')

            # A body that raises NodeSkipped chose not to run: caught INSIDE the
            # span, so the tracer closes it normally instead of marking an error.
            async def skippable(span: Any, *, _nid: str = nid, _n: Any = n, _state: dict[str, Any] = state, _run: Any = run_node) -> None:
                try:
                    await _run(span)
                except NodeSkipped as err:
                    _state["skip_reason"] = err.reason
                    span.set_metadata({"step_id": _nid, "kind": _n.kind, "node_version": _n.version, "skipped": err.reason})
                    span.set_output({"skipped": err.reason})

            try:
                pending = tracer.span(
                    NodeSpanOpts(name=f"{nid}", input={"primary": primary, "from": list(from_.keys())}, metadata={"step_id": nid, "kind": n.kind, "node_version": n.version}),
                    skippable,
                )
                if deadline and n.timeout_ms:
                    task = asyncio.ensure_future(pending)
                    done, _ = await asyncio.wait({task}, timeout=n.timeout_ms / 1000)
                    if task not in done:
                        state["timed_out"] = True
                        deadline.abort()
                        # Whatever the abandoned body does next is nobody's to hear.
                        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
                        raise NodeTimeoutError(nid, n.timeout_ms)
                    task.result()
                else:
                    await pending

                if state["skip_reason"] is not None:
                    entry.ms = clock() - node_start
                    entry.skipped = state["skip_reason"]
                    record_nodes.append(entry)
                    skipped[nid] = state["skip_reason"]
                    await notify_node("on_node_end", entry, cache_hit=cache_hit)
                    continue

                if not cache_hit and cache_mode != "none":
                    value = outputs.get(nid)
                    if cache_mode == "per-run":
                        per_run_cache[cache_key] = value
                    elif ports.cache:
                        await maybe_await(ports.cache.set(cache_key, value))
            except Exception as err:  # noqa: BLE001 — on_failure decides
                message = error_message(err)
                entry.error = message
                entry.ms = clock() - node_start

                # A missing port is the HOST being misconfigured, not the node failing.
                if isinstance(err, MissingPortError):
                    await notify_node("on_node_end", entry, cache_hit=cache_hit)
                    raise
                on_failure = n.on_failure or "fail"

                if on_failure == "fail":
                    record_nodes.append(entry)
                    await notify_node("on_node_end", entry, cache_hit=cache_hit)
                    raise err if isinstance(err, NodeFailedError) else NodeFailedError(nid, err) from err
                if on_failure == "skip":
                    entry.skipped = f"error: {message}"
                    record_nodes.append(entry)
                    skipped[nid] = entry.skipped
                    await notify_node("on_node_end", entry, cache_hit=cache_hit)
                    continue
                if on_failure == "revert":
                    # Discard this node's work; the inbound payload passes through.
                    outputs[nid] = primary
                    ran.add(nid)
                    entry.branch = "reverted"
                    record_nodes.append(entry)
                    await notify_node("on_node_end", entry, cache_hit=cache_hit, output=primary)
                    continue
                # fallback. The node's own signal may be the deadline that just
                # fired; the fallback answers to the run's.
                value = await maybe_await(n.fallback(dataclasses.replace(args, signal=options.signal))) if n.fallback else primary
                outputs[nid] = value
                ran.add(nid)
                record_nodes.append(entry)
                await notify_node("on_node_end", entry, cache_hit=cache_hit, output=value)
                continue

            record_nodes.append(entry)
            await notify_node("on_node_end", entry, cache_hit=cache_hit, output=outputs.get(nid))

        # --- 5. The fact: the first `result` node that actually ran.
        producer = next((rid for rid in pipeline.result if rid in ran), None)
        if producer is None:
            raise Exception(f'Pipeline "{pipeline.id}" produced no fact: none of [{", ".join(pipeline.result)}] ran.')
        output = pipeline.output.parse(outputs.get(producer))
        route = _route_of(branches, pipeline.order)

        features: dict[str, float] = {
            **distribution_features(decide_distributions),
            **box_features(decide_boxes),
            **probe_feature_values,
            **(pipeline.features(FeatureExtractionArgs(input=input, outputs=outputs, branches=branches, ctx=ctx)) if pipeline.features else {}),
            "nodes_ran": len(ran),
            "nodes_skipped": len(skipped),
            "cost_usd": cost_usd,
        }
        record = RunRecord(
            pipeline=pipeline.id,
            version=version,
            run_id=attempt_id,
            run_key=run_key,
            group=group,
            tenant=ctx.ids.get("tenant"),
            user=ctx.ids.get("user"),
            trigger=trigger,
            input_hash=input_hash,
            cost_usd=cost_usd,
            ms=clock() - started_at,
            nodes=record_nodes,
            features=features,
            fact=pipeline.fact,
            fact_route=route,
            fact_confidence=last_gate_confidence,
            consumed_feedback_ids=ctx.consumed(),
            arms=arms,
            parent_run_id=options.parent_run_id,
            assignment_unit=assignment_unit,
            assignment_hash=assignment_hash,
            created_at=iso_from_ms(started_at),
        )

        # A route is `direct` or `writer/quick`; each SEGMENT is its own tag.
        for segment in route.split("/"):
            if segment and segment != "no_gate":
                earn(segment)
        earn(f"route:{route}")
        set_tags = getattr(root_span, "set_tags", None)
        if earned_tags and callable(set_tags):
            set_tags(list(earned_tags))

        root_span.set_metadata(
            {
                f"{pipeline.fact}_route": route,
                f"{pipeline.fact}_version": version,
                f"{pipeline.fact}_run_id": record.run_id,
                "run_key": record.run_key,
                "decided_by": pipeline.decided_by,
                "cost_usd": cost_usd,
                "skipped": list(skipped.keys()),
            }
        )
        root_span.set_output(output)

        # `records` is the first entry in `observers`, so this one call is both
        # the record sink and every watcher behind it.
        await notify_observers(
            observers,
            "on_run_end",
            RunEndEvent(pipeline=pipeline.id, run_id=run_id, run_key=run_key, version=version, record=record, output=output, route=route, confidence=last_gate_confidence, decided_by=pipeline.decided_by, writes_fact=writes_fact),
        )

        return RunResult(
            output=output,
            run_id=run_id,
            run_key=record.run_key,
            version=version,
            route=route,
            confidence=last_gate_confidence,
            decided_by=pipeline.decided_by,
            arms=arms,
            record=record,
            skipped=skipped,
            scope_rejections=rejected,
            patch_applied=patch_applied,
        )

    settled = tracer.root(
        RootSpanOpts(
            name=f"pipeline:{pipeline.id}",
            input=input,
            group=group,
            # Arms as TAGS, not only metadata: a tracer filters and compares on tags.
            tags=[
                f"pipeline:{pipeline.id}",
                f"fact:{pipeline.fact}",
                *(f"arm:{eid}={arm}" for eid, arm in arms.items()),
                *([] if writes_fact else ["shadow"]),
                *(options.tags or []),
            ],
            user_id=ctx.ids.get("user"),
            metadata={
                "pipeline": pipeline.id,
                "fact": pipeline.fact,
                "pipeline_version": version,
                "trigger": trigger,
                "arms": arms,
                "parent_run_id": options.parent_run_id,
                "assignment_unit": assignment_unit,
                "writes_fact": writes_fact,
                **(options.metadata or {}),
            },
        ),
        body,
    )
    try:
        return await settled
    except Exception as err:
        if report_failure is not None:
            await report_failure(err)
        raise


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _route_of(branches: Mapping[str, str], order: Sequence[str]) -> str:
    """Branches taken, in topological order — what lands as `<fact>_route`."""
    taken = [branches[nid] for nid in order if nid in branches]
    return "/".join(taken) if taken else "no_gate"


def _recorded_distributions(record_nodes: Sequence[RunRecordNode], decide_node_id: str) -> Optional[dict[str, list[float]]]:
    """The distributions a preceding decide node recorded, for tool selection."""
    for r in record_nodes:
        if r.id == decide_node_id and not r.skipped:
            return r.distributions
    return None


def _tool_select_threshold(node: GenerateNode, threshold_numbers: Mapping[str, Mapping[str, float]]) -> float:
    """`selectable.threshold` wins; otherwise the named gate's `tools` threshold; otherwise the default."""
    selectable = node.tools.selectable if node.tools else None
    if not selectable:
        return DEFAULT_TOOL_SELECT_THRESHOLD
    declared = selectable.threshold
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        return float(declared)
    if isinstance(declared, Mapping) and "default" in declared:
        return float(declared["default"])
    return threshold_numbers.get(selectable.gate, {}).get("tools", DEFAULT_TOOL_SELECT_THRESHOLD)


def _resolve_tools_for_node(node: GenerateNode, *, prompt_config_tools: Any, distributions: Any, select_threshold: float, grants: Sequence[str], registry: Any) -> ResolvedTools:
    return resolve_tools(
        spec=node.tools,
        prompt_config_tools=prompt_config_tools,
        distributions=distributions,
        select_threshold=select_threshold,
        grants=grants,
        registry=registry,
    )
