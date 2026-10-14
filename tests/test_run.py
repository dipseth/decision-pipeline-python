"""Port of src/run.test.ts."""

from __future__ import annotations

import dataclasses
import re

import pytest

import asyncio

from decision_pipeline import MissingPortError, NodeFailedError, NodeSkipped, NodeTimeoutError, PipelinePorts, apply_override, build_scope_context, run_pipeline
from decision_pipeline import testing as t
from decision_pipeline.define import Overridable
from decision_pipeline.experiments import ExperimentSpec, ManifestPatch
from decision_pipeline.fixtures import build_scale_pipeline, confident_distributions, unsure_distributions

INPUT = {
    "recipeId": "r-1",
    "factor": 2,
    "notes": "",
    "substitutionCount": 0,
    "ingredients": ["1 cup flour", "1 tsp salt"],
}

WRITER_OUTPUT = {
    "scaledIngredients": ["2 cup flour", "2 tsp salt"],
    "scalingNotes": ["the writer's prose"],
    "factor": 2,
    "changes": [],
}

DIRECT_PORTS_INPUT = dict(
    decide={"classify": {"distributions": confident_distributions(2), "costUsd": 0.0002}},
    prompts={"jev-scale-classify": {"version": "12"}},
    scope={"ids": {"user": "u-1", "tenant": "t-1"}, "grants": ["recipes:read"]},
)


def direct_ports() -> t.TestPorts:
    return t.test_ports(**DIRECT_PORTS_INPUT)


def writer_ports(**over) -> t.TestPorts:
    kwargs = dict(
        decide={
            "classify": {
                "distributions": {
                    **unsure_distributions(2),
                    "tool:lookup_pan_conversion": [0.81],
                    "tool:lookup_cooking_adjustment": [0.12],
                },
                "costUsd": 0.0002,
            }
        },
        generate={"writer": {"value": WRITER_OUTPUT, "costUsd": 0.004, "toolsCalled": ["lookup_pan_conversion"]}},
        prompts={
            "jev-scale-classify": {"version": "12"},
            "recipe-scale": {"version": "7", "tools": ["lookup_scaling_rule"]},
        },
        registry={
            "lookup_pan_conversion": "recipes:read",
            "lookup_cooking_adjustment": "cook:assist",
            "lookup_scaling_rule": None,
        },
        scope={"ids": {"user": "u-1", "tenant": "t-1"}, "grants": ["recipes:read"]},
    )
    kwargs.update(over)
    return t.test_ports(**kwargs)


def node(result, node_id):
    return next((n for n in result.record.nodes if n.id == node_id), None)


# --- the direct branch --------------------------------------------------------


async def test_routes_past_the_writer_and_never_calls_it():
    scale = build_scale_pipeline()
    ports = direct_ports()
    result = await run_pipeline(scale, INPUT, ports)

    assert result.route == "direct"
    assert len(ports.generate_calls) == 0
    assert result.output.scaled_ingredients == ["2x 1 cup flour", "2x 1 tsp salt"]
    assert len(result.output.changes) == 2


async def test_records_the_skipped_nodes_with_a_reason_never_missing():
    result = await run_pipeline(build_scale_pipeline(), INPUT, direct_ports())

    assert sorted(result.skipped.keys()) == ["guard", "writer"]
    assert 'gate took "direct", needed "writer"' in result.skipped["writer"]
    assert "writer skipped" in result.skipped["guard"]

    ids = [n.id for n in result.record.nodes]
    assert "writer" in ids
    assert node(result, "writer").skipped


# --- the writer branch --------------------------------------------------------


async def test_runs_writer_guard_changes_when_the_gate_is_not_confident():
    ports = writer_ports()
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert result.route == "writer"
    assert len(ports.generate_calls) == 1
    assert list(result.skipped.keys()) == ["direct"]
    assert 'needed "direct"' in result.skipped["direct"]
    assert result.output.scaling_notes == ["the writer's prose"]


async def test_offers_static_union_jev_selected_tools_and_drops_what_the_grant_does_not_cover():
    ports = writer_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    offered = ports.generate_calls[0].tools
    # static comes from the prompt config; lookup_pan_conversion was selected at 0.81.
    assert offered.offered == ["lookup_scaling_rule", "lookup_pan_conversion"]
    assert offered.selected == {"lookup_pan_conversion": 0.81}
    # lookup_cooking_adjustment lost on its Noul (0.12), not on scope.
    assert offered.dropped_for_scope == []


async def test_drops_a_selected_tool_the_runs_mcp_grant_does_not_cover_and_records_it():
    ports = writer_ports(
        decide={
            "classify": {
                "distributions": {
                    **unsure_distributions(2),
                    "tool:lookup_pan_conversion": [0.81],
                    "tool:lookup_cooking_adjustment": [0.93],
                }
            }
        }
    )
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert ports.generate_calls[0].tools.dropped_for_scope == ["lookup_cooking_adjustment"]
    writer_node = node(result, "writer")
    assert writer_node.tools["dropped_for_scope"] == ["lookup_cooking_adjustment"]
    assert writer_node.tools["called"] == ["lookup_pan_conversion"]


# --- one run = one trace ------------------------------------------------------


async def test_opens_exactly_one_root_and_one_child_span_per_node_that_ran():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert len(ports.tracer.roots) == 1
    assert ports.tracer.roots[0].name == "pipeline:scale"

    children = [s.name for s in ports.tracer.spans if s.depth > 0]
    assert children == ["classify", "table", "gate", "direct", "changes", "persist"]


async def test_puts_the_branch_the_thresholds_and_the_node_version_on_the_spans():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    gate = ports.tracer.span_("gate")
    assert gate.metadata["branch"] == "direct"
    assert gate.metadata["thresholds"] == {"category": 0.7, "tools": 0.5}
    assert ports.tracer.span_("classify").metadata["node_version"] == "3"


async def test_tags_the_trace_with_the_pipeline_the_fact_and_the_route_it_took():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)
    # Each route SEGMENT is its own tag as well as the whole route.
    assert ports.tracer.roots[0].tags == ["pipeline:scale", "fact:scaled_recipe", "direct", "route:direct"]


async def test_takes_host_tags_from_the_caller_for_what_the_manifest_cannot_know():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports, tags=["prod", "scale"])
    tags = ports.tracer.roots[0].tags
    for tag in ["prod", "scale", "direct"]:
        assert tag in tags


async def test_keeps_one_durable_run_key_across_attempts_with_a_new_trace_id_each_time():
    first = await run_pipeline(
        build_scale_pipeline(), INPUT, t.test_ports(**DIRECT_PORTS_INPUT, trace_id="trace-attempt-1"), run_key="scale:abc123"
    )
    second = await run_pipeline(
        build_scale_pipeline(), INPUT, t.test_ports(**DIRECT_PORTS_INPUT, trace_id="trace-attempt-2"), run_key="scale:abc123"
    )

    assert first.run_id == "trace-attempt-1"
    assert second.run_id == "trace-attempt-2"
    assert first.run_key == "scale:abc123"
    assert second.run_key == "scale:abc123"
    assert first.record.run_key == second.record.run_key
    assert first.record.run_id != second.record.run_id


async def test_falls_back_to_the_attempt_id_when_the_host_has_no_stable_key():
    run = await run_pipeline(build_scale_pipeline(), INPUT, direct_ports())
    assert run.run_key == run.record.run_id
    assert run.record.run_key == run.record.run_id


async def test_puts_the_run_key_on_the_trace_so_a_hand_read_can_find_the_retries():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports, run_key="scale:abc123")
    assert ports.tracer.roots[0].metadata["run_key"] == "scale:abc123"


async def test_attributes_the_trace_to_the_user_the_scope_context_resolved():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)
    assert ports.tracer.roots[0].user_id == "u-1"


async def test_tags_the_arms_so_an_arm_is_filterable_and_comparable_in_the_langfuse_ui():
    armed = ExperimentSpec(
        id="gate-tune",
        unit="user",
        mode="shadow",
        champion="control",
        enabled=True,
        arms={"control": ManifestPatch(), "tight": ManifestPatch(thresholds={"gate": {"category": 0.99}})},
    )
    scale = build_scale_pipeline(experiments=[armed])
    ports = writer_ports(decide={"classify": {"distributions": confident_distributions(2)}})

    await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "tight"})
    tags = ports.tracer.roots[0].tags
    assert "arm:gate-tune=tight" in tags
    # A shadow challenger is tagged as one.
    assert "shadow" in tags

    await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "control"})
    assert "shadow" not in ports.tracer.roots[1].tags


async def test_stamps_the_persistence_contract_onto_the_root():
    ports = direct_ports()
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)
    root = ports.tracer.roots[0]

    assert root.metadata["scaled_recipe_route"] == "direct"
    assert root.metadata["scaled_recipe_version"] == result.version
    assert root.metadata["scaled_recipe_run_id"] == result.record.run_id
    assert root.metadata["decided_by"] == "llm"


# --- the persistence contract -------------------------------------------------


async def test_hands_the_store_node_the_fact_its_route_version_run_id_and_rank():
    ports = direct_ports()
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert len(ports.store_writes) == 1
    write = ports.store_writes[0]
    assert write.fact == "scaled_recipe"
    assert write.route == "direct"
    assert write.version == result.version
    assert write.run_id == result.run_id
    assert write.decided_by == "llm"
    assert write.confidence == pytest.approx(0.92, abs=1e-10)
    assert write.scope == "tenant"
    assert write.writes_fact is True


# --- the run record -----------------------------------------------------------


async def test_carries_distributions_and_an_input_hash_on_every_decide_node():
    result = await run_pipeline(build_scale_pipeline(), INPUT, direct_ports())
    classify = node(result, "classify")

    assert classify.distributions == confident_distributions(2)
    assert re.fullmatch(r"[0-9a-f]{16}", classify.input_hash)


async def test_carries_the_thresholds_actually_used_on_the_gate_node():
    result = await run_pipeline(build_scale_pipeline(), INPUT, direct_ports())
    assert node(result, "gate").thresholds == {"category": 0.7, "tools": 0.5}


async def test_totals_the_cost_and_fills_the_identity_fields():
    ports = direct_ports()
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports, parent_run_id="outer-run-9")
    record = ports.recorded[0]

    assert record.pipeline == "scale"
    assert record.group == "cook-r-1"
    assert record.tenant == "t-1"
    assert record.user == "u-1"
    assert record.trigger == "on-demand"
    assert record.cost_usd == pytest.approx(0.0002, abs=1e-10)
    assert record.fact_route == "direct"
    # An arm has to be able to ride down into a sub-pipeline.
    assert record.parent_run_id == "outer-run-9"
    assert result.record == record


async def test_extracts_pipeline_agnostic_features_from_the_distributions():
    result = await run_pipeline(build_scale_pipeline(), INPUT, direct_ports())
    assert result.record.features["classify.line:0.margin"] == pytest.approx(0.84, abs=1e-10)
    assert result.record.features["nodes_ran"] == 6
    assert result.record.features["nodes_skipped"] == 2


async def test_names_the_feedback_rows_the_run_actually_read():
    ports = t.test_ports(
        decide={"classify": {"distributions": confident_distributions(2)}},
        scope={
            "ids": {"user": "u-1"},
            "features": [
                {"id": "f-1", "nodeId": "classify", "scope": "user", "rank": "human", "value": "no salt"},
                {"id": "f-2", "nodeId": "table", "scope": "user", "rank": "human", "value": "unread"},
            ],
        },
    )
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert [f.id for f in ports.decide_calls[0].features] == ["f-1"]
    assert result.record.consumed_feedback_ids == ["f-1"]


# --- thresholds move the branch and the version -------------------------------


async def test_an_env_override_flips_direct_to_writer():
    ports = writer_ports(
        decide={"classify": {"distributions": confident_distributions(2)}},
        env={"RECIPES_SCALE_CATEGORY_GATE": "0.95"},
    )
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)
    assert result.route == "writer"


async def test_and_changes_the_composite_version_so_two_configurations_never_share_one():
    plain = await run_pipeline(build_scale_pipeline(), INPUT, direct_ports())
    tuned = await run_pipeline(
        build_scale_pipeline(),
        INPUT,
        t.test_ports(
            decide={"classify": {"distributions": confident_distributions(2)}},
            env={"RECIPES_SCALE_CATEGORY_GATE": "0.6"},
        ),
    )
    assert tuned.version != plain.version


async def test_a_prompt_version_bump_changes_it_too():
    a = await run_pipeline(
        build_scale_pipeline(),
        INPUT,
        t.test_ports(
            decide={"classify": {"distributions": confident_distributions(2)}},
            prompts={"jev-scale-classify": {"version": "12"}},
        ),
    )
    b = await run_pipeline(
        build_scale_pipeline(),
        INPUT,
        t.test_ports(
            decide={"classify": {"distributions": confident_distributions(2)}},
            prompts={"jev-scale-classify": {"version": "13"}},
        ),
    )
    assert a.version != b.version


# --- on_failure ---------------------------------------------------------------


async def test_revert_discards_the_nodes_work_and_passes_the_inbound_payload_through():
    # The guard throws when the writer drops a line; the run survives on the
    # writer's own output rather than aborting.
    ports = writer_ports(generate={"writer": {"value": {**WRITER_OUTPUT, "scaledIngredients": ["only one"]}}})
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert result.output.scaled_ingredients == ["only one"]
    assert "dropped a line" in node(result, "guard").error


async def test_fail_is_the_default_and_aborts_the_run():
    def timed_out(_request):
        raise Exception("Jev timed out")

    ports = t.test_ports(decide={"classify": timed_out})
    with pytest.raises(NodeFailedError):
        await run_pipeline(build_scale_pipeline(), INPUT, ports)


async def test_a_node_skipped_is_a_skip_not_a_failure_even_under_fail_and_its_span_closes_cleanly():
    def switched_off(_request):
        raise NodeSkipped("classifier switched off")

    ports = t.test_ports(decide={"classify": switched_off})
    threw: list[str] = []
    span = ports.tracer.span

    async def watching_span(opts, fn):
        async def inner(h):
            try:
                return await fn(h)
            except Exception:
                threw.append(opts.name)
                raise

        return await span(opts, inner)

    ports.tracer.span = watching_span  # type: ignore[method-assign]

    try:
        result = await run_pipeline(build_scale_pipeline(), INPUT, ports)
    except Exception as err:  # noqa: BLE001
        result = err

    assert not isinstance(result, NodeFailedError)
    assert "classify" not in threw
    assert ports.tracer.span_("classify").metadata["skipped"] == "classifier switched off"


async def test_names_the_missing_port_rather_than_failing_obscurely():
    scale = build_scale_pipeline()
    with pytest.raises(MissingPortError):
        await run_pipeline(scale, INPUT, PipelinePorts(scope=lambda _r: {}))


# --- a node deadline ---------------------------------------------------------


def with_node(node_id, **over):
    scale = build_scale_pipeline()
    return dataclasses.replace(scale, nodes={**scale.nodes, node_id: dataclasses.replace(scale.nodes[node_id], **over)})


async def test_fails_the_node_past_timeout_ms_and_aborts_the_signal_its_port_was_given():
    seen = {}

    async def never(request):
        seen["signal"] = request.signal
        await asyncio.Event().wait()

    ports = writer_ports(generate={"writer": never})

    with pytest.raises(NodeFailedError) as info:
        await run_pipeline(with_node("writer", timeout_ms=20), INPUT, ports)

    assert isinstance(info.value.cause, NodeTimeoutError)
    assert seen["signal"].aborted is True


async def test_hands_the_timeout_to_on_failure_and_never_lets_the_late_answer_into_the_run():
    # `guard` reverts on failure: past its deadline the writer's output stands.
    late = {**WRITER_OUTPUT, "scalingNotes": ["the guard, too late"]}

    async def slow_guard(_args):
        await asyncio.sleep(0.06)
        return late

    ports = writer_ports()
    result = await run_pipeline(with_node("guard", timeout_ms=20, run=slow_guard), INPUT, ports)
    guard = node(result, "guard")
    assert "20 ms deadline" in guard.error
    assert result.output.scaling_notes == WRITER_OUTPUT["scalingNotes"]

    await asyncio.sleep(0.08)
    assert guard.branch == "reverted"
    assert result.output.scaling_notes == WRITER_OUTPUT["scalingNotes"]


async def test_leaves_a_node_that_answers_in_time_alone():
    ports = writer_ports()
    result = await run_pipeline(with_node("writer", timeout_ms=5_000), INPUT, ports)
    assert node(result, "writer").error is None
    assert result.output.scaling_notes == WRITER_OUTPUT["scalingNotes"]


# --- caching ------------------------------------------------------------------


def _with_classify(pipeline, **changes):
    return dataclasses.replace(
        pipeline, nodes={**pipeline.nodes, "classify": dataclasses.replace(pipeline.nodes["classify"], **changes)}
    )


async def test_keys_per_node_version_input_hash_and_skips_the_port_on_a_hit():
    scale = build_scale_pipeline()
    cached = _with_classify(scale, cache="per-key-forever")
    ports = direct_ports()

    await run_pipeline(cached, INPUT, ports)
    assert len(ports.decide_calls) == 1

    await run_pipeline(cached, INPUT, dataclasses.replace(ports, decide=ports.decide))
    # Same node version, same input hash: the decide port is not called again.
    assert len(ports.decide_calls) == 1

    bumped = _with_classify(cached, version="4")
    await run_pipeline(bumped, INPUT, ports)
    assert len(ports.decide_calls) == 2


async def test_marks_a_cache_hit_on_the_record_so_0_has_a_reason():
    scale = build_scale_pipeline()
    cached = _with_classify(scale, cache="per-key-forever")
    ports = direct_ports()

    first = await run_pipeline(cached, INPUT, ports)
    assert node(first, "classify").cached is None

    second = await run_pipeline(cached, INPUT, dataclasses.replace(ports, decide=ports.decide))
    assert node(second, "classify").cached is True


# --- cost provenance ----------------------------------------------------------


async def test_separates_a_port_that_priced_at_zero_from_one_that_reported_nothing():
    scale = build_scale_pipeline()

    priced = t.test_ports(
        decide={"classify": {"distributions": confident_distributions(2), "costUsd": 0}},
        prompts={"jev-scale-classify": {"version": "12"}},
    )
    with_zero = await run_pipeline(scale, INPUT, priced)
    zero_node = node(with_zero, "classify")
    assert zero_node.cost_usd == 0
    assert zero_node.cost_reported is True

    silent = t.test_ports(
        decide={"classify": {"distributions": confident_distributions(2)}},
        prompts={"jev-scale-classify": {"version": "12"}},
    )
    with_none = await run_pipeline(scale, INPUT, silent)
    silent_node = node(with_none, "classify")
    assert silent_node.cost_usd == 0
    assert silent_node.cost_reported is False


# --- experiments --------------------------------------------------------------


def armed(mode: str = "split") -> ExperimentSpec:
    return ExperimentSpec(
        id="gate-tune",
        unit="user",
        mode=mode,
        champion="control",
        enabled=True,
        arms={"control": ManifestPatch(), "tight": ManifestPatch(thresholds={"gate": {"category": 0.99}})},
    )


async def test_an_arm_patch_changes_the_branch_and_the_arm_is_on_the_record():
    scale = build_scale_pipeline(experiments=[armed()])
    ports = writer_ports(decide={"classify": {"distributions": confident_distributions(2)}})

    control = await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "control"})
    assert control.route == "direct"

    tight = await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "tight"})
    assert tight.route == "writer"
    assert tight.arms == {"gate-tune": "tight"}
    assert tight.record.arms == {"gate-tune": "tight"}
    assert tight.patch_applied == ["gate.thresholds.category=0.99"]


async def test_folds_the_arm_into_the_version_so_two_arms_never_collide_under_one():
    scale = build_scale_pipeline(experiments=[armed()])
    ports = writer_ports(decide={"classify": {"distributions": confident_distributions(2)}})

    a = await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "control"})
    b = await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "tight"})
    assert a.version != b.version


async def test_records_the_assignment_unit_and_hash_so_an_assignment_is_auditable():
    scale = build_scale_pipeline(experiments=[armed()])
    # No `force_arm`: this exercises the real assignment path.
    result = await run_pipeline(scale, INPUT, writer_ports())
    assert result.record.assignment_unit == "user"
    assert result.record.assignment_hash == "gate-tune:user:u-1"


async def test_a_shadow_challenger_must_not_write_the_fact():
    scale = build_scale_pipeline(experiments=[armed("shadow")])
    ports = writer_ports(decide={"classify": {"distributions": confident_distributions(2)}})

    await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "tight"})
    assert ports.store_writes[0].writes_fact is False

    await run_pipeline(scale, INPUT, ports, force_arm={"gate-tune": "control"})
    assert ports.store_writes[1].writes_fact is True


# --- overrides layer, never mutate --------------------------------------------


async def test_applies_the_narrowest_override_at_read_leaving_the_runs_own_answer_untouched():
    scale = build_scale_pipeline()
    result = await run_pipeline(scale, INPUT, direct_ports())

    overridable = dataclasses.replace(scale, overridable=Overridable(scope="device", key=lambda _o: "scaled:r-1"))
    ctx = build_scope_context(
        {
            "overrides": [
                {"id": "o-1", "key": "scaled:r-1", "scope": "user", "rank": "human", "value": "the human's"},
                {"id": "o-2", "key": "scaled:r-1", "scope": "device", "rank": "human", "value": "the device's"},
            ]
        }
    ).ctx

    value, override = apply_override(overridable, result.output, ctx)
    # Narrowest wins: the device row, not the user row.
    assert value == "the device's"
    assert override.id == "o-2"
    # What the pipeline decided is still exactly what it decided.
    assert result.output.scaled_ingredients == ["2x 1 cup flour", "2x 1 tsp salt"]


# --- output validation --------------------------------------------------------


async def test_rejects_a_shape_the_manifests_output_schema_does_not_describe():
    ports = writer_ports(generate={"writer": {"value": {"nonsense": True}}})
    with pytest.raises(Exception):
        await run_pipeline(build_scale_pipeline(), INPUT, ports)


async def test_rejects_an_input_the_manifests_input_schema_does_not_describe():
    with pytest.raises(Exception):
        await run_pipeline(build_scale_pipeline(), {"recipeId": 7}, direct_ports())
