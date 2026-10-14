"""Port of src/spec.test.ts."""

from __future__ import annotations

import re
from typing import Any

import pytest

from decision_pipeline import testing as t
from decision_pipeline._compat import to_plain
from decision_pipeline.errors import PipelineContractError  # noqa: F401 - pipeline_from_spec raises it
from decision_pipeline.fixtures import demo_registry as registry
from decision_pipeline.fixtures import demo_spec as base
from decision_pipeline.jsonata import jsonata_engine
from decision_pipeline.run import run_pipeline
from decision_pipeline.spec import (
    compile_pipeline_spec,
    define_registry,
    describe_registry,
    merge_registries,
    pipeline_from_spec,
    pipeline_spec_json_schema,
)

engine = jsonata_engine()

INPUT: dict[str, Any] = {"id": "r1", "title": "pad thai", "factor": 4}


def matches(obj: Any, expected: dict[str, Any]) -> bool:
    """vitest `toMatchObject` over dicts / dataclasses."""
    for k, v in expected.items():
        got = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
        if isinstance(v, dict):
            if not matches(got, v):
                return False
        elif got != v:
            return False
    return True


# --- compilePipelineSpec — a spec runs like a TS manifest ---


async def test_routes_by_rules_calls_primitives_packs_bindings_and_records_the_gates_reason() -> None:
    pipeline = pipeline_from_spec(base(), registry, engine)
    ports = t.test_ports()
    run = await run_pipeline(pipeline, INPUT, ports)
    assert to_plain(run.output) == {"text": "PAD THAI", "n": 2}
    assert run.route == "loud"
    gate = next(n for n in run.record.nodes if n.id == "gate")
    assert matches(gate, {"branch": "loud", "branch_reason": "p 0.8", "thresholds": {"p": 0.5}})
    assert ports.tracer.roots[0].name == "pipeline:spec-demo"
    assert run.record.group == "demo-r1"


async def test_takes_the_catch_all_branch_when_no_earlier_rule_matches() -> None:
    spec = base()
    spec["nodes"]["gate"]["thresholds"] = {"p": 0.9}
    run = await run_pipeline(pipeline_from_spec(spec, registry, engine), INPUT, t.test_ports())
    assert run.route == "plain"
    assert to_plain(run.output) == {"text": "pad thai", "n": 0}


def test_a_nodes_version_carries_a_hash_of_its_body_so_an_edited_expression_is_a_new_version() -> None:
    a = pipeline_from_spec(base(), registry, engine)
    edited = base()
    edited["nodes"]["plain"]["expr"] = '{ "text": gate.title, "n": 1 }'
    b = pipeline_from_spec(edited, registry, engine)
    assert re.match(r"^1~[0-9a-f]{8}$", a.nodes["plain"].version)
    assert a.nodes["plain"].version != b.nodes["plain"].version
    assert a.nodes["gate"].version == b.nodes["gate"].version
    assert a.nodes["persist"].version == "1"


async def test_args_that_miss_the_primitives_schema_fail_the_node_and_on_failure_decides() -> None:
    spec = base()
    loud = spec["nodes"]["loud"]
    loud["args"]["n"] = "'four'"
    with pytest.raises(Exception, match=r"args for demo.shout@1 do not match its schema — n:"):
        await run_pipeline(pipeline_from_spec(spec, registry, engine), INPUT, t.test_ports())
    loud["onFailure"] = "fallback"
    loud["fallback"] = {"value": {"text": "fallback", "n": -1}}
    run = await run_pipeline(pipeline_from_spec(spec, registry, engine), INPUT, t.test_ports())
    assert to_plain(run.output) == {"text": "fallback", "n": -1}


async def test_an_assert_that_fails_names_its_message() -> None:
    spec = base()
    spec["nodes"]["plain"]["expr"] = '{ "text": "", "n": 0 }'
    spec["nodes"]["gate"]["thresholds"] = {"p": 0.9}
    with pytest.raises(Exception, match=r"check: empty text"):
        await run_pipeline(pipeline_from_spec(spec, registry, engine), INPUT, t.test_ports())


# --- compilePipelineSpec — every problem at once, for a repair loop ---


def test_lists_registry_expression_rule_and_contract_problems_together() -> None:
    spec = base()
    spec["nodes"]["loud"]["call"] = "demo.whisper@1"
    spec["nodes"]["plain"]["expr"] = "{ unclosed"
    spec["nodes"]["gate"]["rules"] = [{"branch": "loud"}, {"when": "true", "branch": "quiet"}]
    result = compile_pipeline_spec(spec, registry, engine)
    assert result.pipeline is None
    text = "\n".join(result.problems)
    assert re.search(r'nodes.loud: unknown primitive "demo.whisper@1"', text)
    assert re.search(r"nodes.plain: expression does not parse", text)
    assert re.search(r'rules\[1\]: branch "quiet" is not one of loud, plain', text)
    assert re.search(r"rules\[0\]: only the last rule may omit `when`", text)
    assert re.search(r"the last rule must have no `when`", text)
    assert re.search(r'no rule produces branch "plain"', text)


def test_refuses_an_expression_that_reads_something_not_flowing_into_the_node_and_says_why() -> None:
    spec = base()
    spec["nodes"]["plain"]["expr"] = '{ "text": bundle.title, "n": $count(items[p > 0].slug) }'
    text = "\n".join(compile_pipeline_spec(spec, registry, engine).problems)
    assert re.search(r"nodes.plain: reads `bundle`, which is not a node with an edge into it \(gate\) — a gate passes its input through", text)
    assert re.search(r"reads `items`", text)
    # Fields after the first step and names inside the filter are not roots.
    assert not re.search(r"reads `(title|p|slug)`", text)


def test_runs_the_define_pipeline_contract_once_the_spec_itself_is_sound() -> None:
    spec = base()
    spec["feedback"] = []
    spec["nodes"]["bundle"] = {**spec["nodes"]["bundle"], "inputs": {"p": "persist.p"}}
    problems = compile_pipeline_spec(spec, registry, engine).problems
    assert re.search(r"no feedback edge", "\n".join(problems))
    assert re.search(r'input "p" reads "persist", which has no edge persist -> bundle', "\n".join(problems))


def test_rejects_unknown_fields_instead_of_ignoring_them() -> None:
    spec = base()
    spec["nodes"]["plain"]["run"] = "() => 1"
    # TS asserts only that the problem names the node; pydantic's path is `nodes.plain.code.run`
    # (the discriminator tag sits in the path), so the stable part is the `nodes.plain` prefix.
    assert re.search(r"nodes\.plain\..*run", "\n".join(compile_pipeline_spec(spec, registry, engine).problems))


# --- registry and catalog ---


def test_describes_itself_for_a_model_and_the_spec_has_a_json_schema() -> None:
    d = describe_registry(registry)
    assert [p["ref"] for p in d["primitives"]] == ["demo.load@1", "demo.shout@1"]
    assert matches(d["primitives"][1]["args"], {"type": "object", "properties": {"title": {"type": "string"}}})
    assert d["functions"] == [{"name": "$half", "describe": "x / 2"}]
    assert matches(pipeline_spec_json_schema(), {"type": "object"})


def test_refuses_a_double_registration_and_a_shadowed_binding() -> None:
    with pytest.raises(ValueError, match="defined twice"):
        merge_registries(registry, registry)
    with pytest.raises(ValueError, match="shadow"):
        define_registry(functions={"t": {"describe": "", "fn": lambda: 1}})


# --- jsonataEngine ---


async def test_returns_plain_arrays_and_wraps_jsonata_errors_as_errors() -> None:
    from decision_pipeline._compat import maybe_await

    out = await maybe_await(engine.compile("[1,2,3].($ * 2)").evaluate({}, {}))
    assert out == [2, 4, 6]
    # TS checks `Object.keys(out)` is ["0","1","2"] — no sequence marker. Here: a plain list, not JSONata's subclass.
    assert type(out) is list
    with pytest.raises(Exception) as caught:
        await maybe_await(engine.compile("$nope()").evaluate({}, {}))
    assert isinstance(caught.value, Exception)
    with pytest.raises(Exception):
        engine.compile("(")


async def test_timeboxes_runaway_recursion_by_depth_and_by_time() -> None:
    from decision_pipeline._compat import maybe_await

    # Not a tail call — JSONata optimises those into a loop that never deepens.
    deep = jsonata_engine(max_depth=50).compile("( $f := function($n) { 1 + $f($n + 1) }; $f(0) )")
    with pytest.raises(Exception, match=r"exceeded depth 50"):
        await maybe_await(deep.evaluate({}, {}))
    loop = jsonata_engine(timeout_ms=50).compile("( $f := function($n) { $f($n + 1) }; $f(0) )")
    with pytest.raises(Exception, match=r"exceeded 50 ms"):
        await maybe_await(loop.evaluate({}, {}))
