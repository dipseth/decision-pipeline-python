"""Port of src/shadow.test.ts."""

from __future__ import annotations

import dataclasses
from typing import Any, NoReturn

from decision_pipeline import testing as t
from decision_pipeline.define import DecisionPipeline
from decision_pipeline.fixtures import build_scale_pipeline, confident_distributions
from decision_pipeline.ports import RunOptions
from decision_pipeline.run import run_pipeline
from decision_pipeline.shadow import ShadowCalls, run_shadow

INPUT: dict[str, Any] = {
    "recipeId": "r-1",
    "factor": 2,
    "notes": "",
    "substitutionCount": 0,
    "ingredients": ["1 cup flour", "1 tsp salt"],
}

WRITER_OUTPUT: dict[str, Any] = {
    "scaledIngredients": ["2 cup flour", "2 tsp salt"],
    "scalingNotes": ["the writer's prose"],
    "factor": 2,
    "changes": [],
}


def close(a: float, b: float, digits: int = 2) -> bool:
    """vitest `toBeCloseTo`."""
    return abs(a - b) < 10 ** (-digits) / 2


def ports() -> t.TestPorts:
    """Jev is confident (0.95), so the champion goes direct."""
    return t.test_ports(
        decide={"classify": {"distributions": confident_distributions(2), "costUsd": 0.0002}},
        generate={"writer": {"value": WRITER_OUTPUT, "costUsd": 0.004}},
        prompts={"jev-scale-classify": {"version": "12"}, "recipe-scale": {"version": "7", "tools": []}},
        scope={"ids": {"user": "u-1", "tenant": "t-1"}, "grants": ["recipes:read"]},
    )


def stricter() -> DecisionPipeline:
    """The same pipeline with a gate 0.95 cannot clear — a stand-in for a proposed spec."""
    base = build_scale_pipeline()
    gate = dataclasses.replace(base.nodes["gate"], thresholds={"category": 0.99, "tools": 0.5}, version="2")
    return dataclasses.replace(base, nodes={**base.nodes, "gate": gate})


# --- runShadow ---


async def test_ships_the_champion_keeps_the_challengers_store_from_the_host_and_says_where_they_parted() -> None:
    p = ports()
    out = await run_shadow(champion=build_scale_pipeline(), challenger=stricter(), input=INPUT, ports=p, label="strict", options=RunOptions(run_key="job-1"))

    assert out.champion.route == "direct"
    assert out.challenger is not None and out.challenger.route == "writer"
    # Only the champion's fact reached the host store.
    assert len(p.store_writes) == 1
    assert p.store_writes[0].writes_fact is True
    assert len(out.challenger_stores) == 1
    assert out.challenger_stores[0].writes_fact is False

    c = out.comparison
    assert c is not None
    assert c.same_output is False
    assert c.same_route is False
    assert c.only_champion == ["direct"]
    assert c.only_challenger == ["writer", "guard"]
    assert c.branch_changed == [{"id": "gate", "champion": "direct", "challenger": "writer"}]
    assert close(c.cost_delta_usd, 0.004)


async def test_shares_the_champions_model_answers_so_the_shadow_pays_only_for_what_differs() -> None:
    p = ports()
    out = await run_shadow(champion=build_scale_pipeline(), challenger=stricter(), input=INPUT, ports=p, label="strict")

    assert len(p.decide_calls) == 1
    assert len(p.generate_calls) == 1
    assert out.calls == ShadowCalls(shared=1, fresh=1, fresh_cost_usd=0.004)
    # The challenger's record still states what its path costs.
    assert out.challenger is not None
    assert close(out.challenger.record.cost_usd, 0.0042)


async def test_asks_again_when_sharing_is_off() -> None:
    p = ports()
    out = await run_shadow(champion=build_scale_pipeline(), challenger=stricter(), input=INPUT, ports=p, label="strict", share=False)
    assert len(p.decide_calls) == 2
    assert out.calls.fresh == 2


async def test_tags_the_challenger_as_a_shadow_and_gives_it_its_own_run_key() -> None:
    p = ports()
    out = await run_shadow(
        champion=build_scale_pipeline(), challenger=stricter(), input=INPUT, ports=p, label="strict", options=RunOptions(run_key="job-1", tags=["t"])
    )

    champ, chall = p.tracer.roots
    assert "shadow" not in champ.tags
    assert {"shadow", "challenger:strict", "t"} <= set(chall.tags)
    assert chall.metadata["writes_fact"] is False
    assert chall.metadata["challenger"] == "strict"
    assert chall.metadata["shadow_of"] == out.champion.run_id
    assert out.champion.run_key == "job-1"
    assert out.challenger is not None and out.challenger.run_key == "job-1:shadow:strict"


async def test_never_fails_the_champion_when_the_challenger_throws() -> None:
    base = build_scale_pipeline()

    def boom(_args: Any) -> NoReturn:
        raise Exception("boom")

    failing = dataclasses.replace(base, nodes={**base.nodes, "direct": dataclasses.replace(base.nodes["direct"], run=boom)})
    out = await run_shadow(champion=build_scale_pipeline(), challenger=failing, input=INPUT, ports=ports(), label="broken")
    assert out.champion.route == "direct"
    assert out.challenger is None
    assert out.challenger_error is not None and "boom" in out.challenger_error
    assert out.comparison is None


# --- RunOptions.shadow ---


async def test_clears_writes_fact_whatever_the_arms_say() -> None:
    p = ports()
    await run_pipeline(build_scale_pipeline(), INPUT, p, RunOptions(shadow=True))
    assert p.store_writes[0].writes_fact is False
    assert "shadow" in p.tracer.roots[0].tags
