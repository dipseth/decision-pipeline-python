"""
Port of src/observers.test.ts — the observer contract (#325): what a watcher
is promised, and what it is NOT allowed to do to the run it is watching.
"""

from __future__ import annotations

import dataclasses

import pytest

from decision_pipeline import MissingPortError, NodeFailedError, run_pipeline
from decision_pipeline import testing as t
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


def direct_ports() -> t.TestPorts:
    return t.test_ports(
        decide={"classify": {"distributions": confident_distributions(2), "costUsd": 0.0002}},
        prompts={"jev-scale-classify": {"version": "12"}},
        scope={"ids": {"user": "u-1", "tenant": "t-1"}},
    )


def writer_ports() -> t.TestPorts:
    return t.test_ports(
        decide={"classify": {"distributions": unsure_distributions(2), "costUsd": 0.0002}},
        generate={"writer": {"value": WRITER_OUTPUT, "costUsd": 0.004}},
        prompts={"jev-scale-classify": {"version": "12"}, "recipe-scale": {"version": "7"}},
        scope={"ids": {"user": "u-1", "tenant": "t-1"}},
    )


class Observer:
    """A plain watcher: any object with some of the hooks works."""

    def __init__(self, name, **hooks):
        self.name = name
        for hook, fn in hooks.items():
            setattr(self, hook, fn)


# --- the event stream ---------------------------------------------------------


async def test_brackets_the_run_one_start_one_end_and_node_events_in_between():
    ports = direct_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    stream = [e.hook for e in ports.observed]
    assert stream[0] == "on_run_start"
    assert stream[-1] == "on_run_end"
    assert stream.count("on_run_start") == 1
    assert stream.count("on_run_end") == 1


async def test_pairs_every_node_the_runtime_considered_skips_included():
    ports = direct_ports()
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert sorted(result.skipped.keys()) == ["guard", "writer"]

    for nid in build_scale_pipeline().nodes.keys():
        starts = [e for e in ports.observed if e.hook == "on_node_start" and e.event.node_id == nid]
        ends = [e for e in ports.observed if e.hook == "on_node_end" and e.event.node_id == nid]
        assert [nid, len(starts), len(ends)] == [nid, 1, 1]

    writer_end = next(e for e in ports.observed if e.hook == "on_node_end" and e.event.node_id == "writer")
    assert 'gate took "direct"' in writer_end.event.skipped


async def test_names_the_same_run_the_record_will_before_a_single_node_has_run():
    ports = t.test_ports(
        decide={"classify": {"distributions": confident_distributions(2)}},
        prompts={"jev-scale-classify": {"version": "12"}},
        trace_id="trace-abc",
    )
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports, run_key="job-99")

    start = ports.observed[0]
    assert start.hook == "on_run_start"
    assert start.event.run_id == "trace-abc"
    assert start.event.run_key == "job-99"
    assert start.event.run_key == result.record.run_key
    assert start.event.version == result.version


async def test_carries_each_nodes_cost_and_the_gates_branch():
    ports = writer_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    ends = [e for e in ports.observed if e.hook == "on_node_end"]
    cost = sum(e.event.cost_usd for e in ends)
    assert cost == pytest.approx(0.0042, abs=5e-7)

    gate = next(e for e in ends if e.event.node_id == "gate")
    assert gate.event.branch == "writer"


async def test_hands_on_run_end_the_record_the_output_and_the_route():
    ports = direct_ports()
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    end = ports.observed[-1]
    assert end.hook == "on_run_end"
    assert end.event.route == "direct"
    assert end.event.record is result.record
    assert end.event.output == result.output
    assert end.event.writes_fact is True


# --- best-effort, by contract -------------------------------------------------


async def test_a_watcher_that_throws_on_every_hook_does_not_fail_the_run():
    ports = dataclasses.replace(direct_ports(), observers=[t.throwing_observer()])
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert result.route == "direct"
    assert len(result.output.scaled_ingredients) == 2


async def test_one_thrower_does_not_rob_the_watchers_behind_it():
    watcher = t.collecting_observer("after")
    ports = dataclasses.replace(direct_ports(), observers=[t.throwing_observer(), watcher.observer])
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert "on_run_start" in watcher.stream()
    assert "on_run_end" in watcher.stream()


async def test_a_record_sink_that_throws_no_longer_fails_a_run_that_already_answered():
    def sink(_record):
        raise Exception("training table is down")

    ports = dataclasses.replace(direct_ports(), records=sink)
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)
    assert result.route == "direct"


# --- `records` is the first watcher, not a separate path ----------------------


async def test_still_receives_the_record_before_the_observers_behind_it():
    seen: list[str] = []
    ports = dataclasses.replace(
        direct_ports(),
        records=lambda _r: seen.append("records"),
        observers=[Observer("after", on_run_end=lambda _e: seen.append("observer"))],
    )
    result = await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert seen == ["records", "observer"]
    assert result.record.run_id


async def test_notifies_in_registration_order():
    order: list[str] = []

    def mark(name):
        return Observer(name, on_run_end=lambda _e: order.append(name))

    ports = dataclasses.replace(direct_ports(), observers=[mark("a"), mark("b"), mark("c")])
    await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert order == ["a", "b", "c"]


# --- a failed run -------------------------------------------------------------


async def test_ends_the_node_that_killed_it_then_reports_the_failure_as_before():
    ports = writer_ports()
    scale = build_scale_pipeline()

    def explode(_args):
        raise Exception("changes blew up")

    # `changes` has no on_failure, so it defaults to "fail" — the run throws.
    exploding = dataclasses.replace(scale, nodes={**scale.nodes, "changes": dataclasses.replace(scale.nodes["changes"], run=explode)})

    with pytest.raises(NodeFailedError):
        await run_pipeline(exploding, INPUT, ports)

    end = next(e for e in ports.observed if e.hook == "on_node_end" and e.event.node_id == "changes")
    assert "changes blew up" in end.event.error
    # No record, so no `on_run_end` — the run ends in `on_run_fail` instead, last.
    assert not any(e.hook == "on_run_end" for e in ports.observed)
    fail = ports.observed[-1]
    assert fail.hook == "on_run_fail", "expected on_run_fail last"
    assert fail.event.failed_node == "changes"
    assert "changes blew up" in fail.event.error
    # What the run had done when it died: every node up to the one that killed it.
    assert fail.event.nodes[-1].id == "changes"
    assert len(fail.event.nodes) > 1
    assert fail.event.cost_usd == sum(n.cost_usd for n in fail.event.nodes)


async def test_a_failed_run_never_reaches_the_record_sink():
    ports = writer_ports()
    scale = build_scale_pipeline()

    def explode(_args):
        raise Exception("changes blew up")

    exploding = dataclasses.replace(scale, nodes={**scale.nodes, "changes": dataclasses.replace(scale.nodes["changes"], run=explode)})
    records = []
    ports.records = lambda r: records.append(r)

    with pytest.raises(NodeFailedError):
        await run_pipeline(exploding, INPUT, ports)
    assert records == []


async def test_a_successful_run_never_hears_on_run_fail():
    ports = writer_ports()
    await run_pipeline(build_scale_pipeline(), INPUT, ports)
    assert not any(e.hook == "on_run_fail" for e in ports.observed)


async def test_pairs_the_node_even_when_a_missing_port_kills_the_run():
    ports = t.test_ports(prompts={"jev-scale-classify": {"version": "12"}})
    ports.decide = None

    with pytest.raises(MissingPortError):
        await run_pipeline(build_scale_pipeline(), INPUT, ports)

    assert [e.hook for e in ports.observed] == ["on_run_start", "on_node_start", "on_node_end", "on_run_fail"]
    end = ports.observed[2]
    assert end.event.node_id == "classify"
    assert "decide" in end.event.error
