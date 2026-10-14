import pytest

from decision_pipeline.thresholds import (
    ResolvedThreshold,
    ThresholdSource,
    resolve_threshold,
    resolve_thresholds,
    threshold_values,
)


async def test_passes_a_literal_straight_through():
    assert await resolve_threshold(0.7) == ResolvedThreshold(value=0.7, source="literal")


async def test_reads_an_env_override_and_records_where_it_came_from():
    r = await resolve_threshold(
        {"env": "RECIPES_SCALE_CATEGORY_GATE", "default": 0.7},
        ThresholdSource(env=lambda _n: "0.6"),
    )
    assert r == ResolvedThreshold(value=0.6, source="env", ref="RECIPES_SCALE_CATEGORY_GATE")


@pytest.mark.parametrize("raw", [None, "", "   ", "not-a-number"])
async def test_falls_back_to_the_default_when_env_is_unset_empty_or_not_a_number(raw):
    r = await resolve_threshold({"env": "X", "default": 0.7}, ThresholdSource(env=lambda _n: raw))
    assert r == ResolvedThreshold(value=0.7, source="default", ref="X")


async def test_reads_a_langfuse_prompt_config_value():
    async def pc(_key):
        return 0.45

    r = await resolve_threshold(
        {"promptConfig": "direct_gate", "default": 0.7},
        ThresholdSource(prompt_config=pc),
    )
    assert r == ResolvedThreshold(value=0.45, source="prompt-config", ref="direct_gate")


async def test_accepts_prompt_config_as_the_snake_case_key_too():
    r = await resolve_threshold(
        {"prompt_config": "direct_gate", "default": 0.7},
        ThresholdSource(prompt_config=lambda _k: 0.45),
    )
    assert r == ResolvedThreshold(value=0.45, source="prompt-config", ref="direct_gate")


async def test_falls_back_when_the_prompt_config_has_no_such_key():
    r = await resolve_threshold(
        {"promptConfig": "missing", "default": 0.7},
        ThresholdSource(prompt_config=lambda _k: None),
    )
    assert r.source == "default"
    assert r.value == 0.7


async def test_resolves_a_whole_set_and_flattens_it_to_numbers_for_a_gate_body():
    resolved = await resolve_thresholds(
        {"category": {"env": "GATE", "default": 0.7}, "tools": 0.5},
        ThresholdSource(env=lambda _n: "0.8"),
    )
    assert threshold_values(resolved) == {"category": 0.8, "tools": 0.5}
