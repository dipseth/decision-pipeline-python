from typing import Optional

from decision_pipeline.tools import resolve_tools, tool_noul
from decision_pipeline.types import ToolSelectable, ToolSpec


class _StaticRegistry:
    def __init__(self, tools: dict[str, Optional[str]]):
        self._tools = tools

    def has(self, name: str) -> bool:
        return name in self._tools

    def scope_of(self, name: str) -> Optional[str]:
        return self._tools.get(name)


def static_registry(tools: dict[str, Optional[str]]) -> _StaticRegistry:
    return _StaticRegistry(tools)


registry = static_registry(
    {
        "lookup_pan_conversion": "recipes:read",
        "lookup_cooking_adjustment": "cook:assist",
        "lookup_scaling_rule": None,
        "sync_list": "list:sync",
    }
)


def test_reads_the_tool_name_key_and_a_bare_name_as_a_fallback():
    assert tool_noul({"tool:a": [0.9]}, "a") == 0.9
    assert tool_noul({"a": [0.4]}, "a") == 0.4
    assert tool_noul({}, "a") is None
    assert tool_noul(None, "a") is None


def test_offers_nothing_when_the_node_declares_no_tools():
    assert resolve_tools(spec=None).offered == []


def test_resolves_prompt_config_to_what_the_langfuse_prompt_says_langfuse_wins():
    r = resolve_tools(
        spec=ToolSpec(static="prompt-config"),
        prompt_config_tools=["lookup_scaling_rule"],
        registry=registry,
        grants=[],
    )
    assert r.offered == ["lookup_scaling_rule"]


def test_offers_static_union_selected_and_records_the_noul_that_selected_each():
    r = resolve_tools(
        spec=ToolSpec(
            static=["lookup_scaling_rule"],
            selectable=ToolSelectable(
                candidates=["lookup_pan_conversion", "lookup_cooking_adjustment"],
                from_="classify",
                gate="gate",
            ),
        ),
        distributions={
            "tool:lookup_pan_conversion": [0.81],
            "tool:lookup_cooking_adjustment": [0.12],
        },
        select_threshold=0.5,
        registry=registry,
        grants=["recipes:read", "cook:assist"],
    )
    assert r.offered == ["lookup_scaling_rule", "lookup_pan_conversion"]
    assert r.selected == {"lookup_pan_conversion": 0.81}


def test_selects_nothing_when_the_decide_node_never_ran():
    r = resolve_tools(
        spec=ToolSpec(
            selectable=ToolSelectable(candidates=["lookup_pan_conversion"], from_="classify", gate="gate"),
        ),
        distributions=None,
        registry=registry,
        grants=["recipes:read"],
    )
    assert r.offered == []
    assert r.selected == {}


def test_drops_a_tool_whose_mcp_scope_the_run_was_not_granted_and_records_the_drop():
    r = resolve_tools(
        spec=ToolSpec(static=["lookup_pan_conversion", "sync_list", "lookup_scaling_rule"]),
        registry=registry,
        grants=["recipes:read"],
    )
    # A pipeline running on a user's behalf can never offer a tool wider than
    # that user's grant. `lookup_scaling_rule` needs no scope at all.
    assert r.offered == ["lookup_pan_conversion", "lookup_scaling_rule"]
    assert r.dropped_for_scope == ["sync_list"]


def test_enforces_the_allow_ceiling_against_a_prompt_config_set_resolved_at_run_time():
    r = resolve_tools(
        spec=ToolSpec(static="prompt-config", allow=["lookup_scaling_rule"]),
        prompt_config_tools=["lookup_scaling_rule", "sync_list"],
        registry=registry,
        grants=["list:sync"],
    )
    assert r.offered == ["lookup_scaling_rule"]
    assert r.dropped_for_allow == ["sync_list"]


def test_drops_a_name_the_registry_does_not_know():
    r = resolve_tools(spec=ToolSpec(static=["no_such_tool"]), registry=registry, grants=[])
    assert r.offered == []
    assert r.dropped_unknown == ["no_such_tool"]


def test_never_offers_the_same_tool_twice():
    r = resolve_tools(
        spec=ToolSpec(
            static=["lookup_pan_conversion"],
            selectable=ToolSelectable(candidates=["lookup_pan_conversion"], from_="c", gate="g"),
        ),
        distributions={"tool:lookup_pan_conversion": [0.99]},
        registry=registry,
        grants=["recipes:read"],
    )
    assert r.offered == ["lookup_pan_conversion"]
