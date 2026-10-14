import pytest

from decision_pipeline.scope import build_scope_context, empty_scope_context
from decision_pipeline.types import FeatureRow, OverrideRow


def feature(**over) -> FeatureRow:
    base = dict(node_id="classify", rank="human", value="cilantro is produce")
    base.update(over)
    return FeatureRow(**base)


def override(**over) -> OverrideRow:
    base = dict(key="cilantro", rank="human", value="produce")
    base.update(over)
    return OverrideRow(**base)


def test_concatenates_every_scope_in_the_chain_narrowest_first():
    ctx = build_scope_context(
        {
            "features": [
                feature(id="f-user", scope="user"),
                feature(id="f-device", scope="device"),
                feature(id="f-assoc", scope="associated"),
            ]
        }
    ).ctx
    assert [f.id for f in ctx.features("classify")] == ["f-device", "f-user", "f-assoc"]


def test_only_hands_a_node_the_rows_addressed_to_it():
    ctx = build_scope_context(
        {
            "features": [
                feature(id="a", scope="user", node_id="classify"),
                feature(id="b", scope="user", node_id="table"),
            ]
        }
    ).ctx
    assert [f.id for f in ctx.features("classify")] == ["a"]
    assert ctx.features("nobody") == []


def test_orders_a_stronger_rank_first_inside_one_scope():
    ctx = build_scope_context(
        {
            "features": [
                feature(id="llm", scope="user", rank="llm"),
                feature(id="admin", scope="user", rank="admin"),
            ]
        }
    ).ctx
    assert [f.id for f in ctx.features("classify")] == ["admin", "llm"]


def test_orders_oldest_first_inside_one_scope_and_rank():
    ctx = build_scope_context(
        {
            "features": [
                feature(id="new", scope="user", created_at="2024-02-01T00:00:00.000Z"),
                feature(id="none", scope="user"),
                feature(id="old", scope="user", created_at="2024-01-01T00:00:00.000Z"),
            ]
        }
    ).ctx
    assert [f.id for f in ctx.features("classify")] == ["none", "old", "new"]


def test_accepts_plain_dict_rows_with_camel_case_keys():
    ctx = build_scope_context(
        {
            "features": [
                {"id": "f", "nodeId": "classify", "scope": "user", "rank": "human", "value": 1, "createdAt": "x"}
            ],
            "overrides": [{"id": "o", "key": "k", "scope": "user", "rank": "human", "value": 2}],
        }
    ).ctx
    assert ctx.features("classify")[0].created_at == "x"
    assert ctx.override("k").id == "o"


def test_drops_a_feature_row_from_tenant_or_wider_and_records_the_drop():
    built = build_scope_context(
        {
            "features": [
                feature(id="ok", scope="associated"),
                feature(id="too-wide", scope="tenant"),
                feature(id="way-too-wide", scope="global"),
            ]
        }
    )
    ctx, rejected = built.ctx, built.rejected
    assert [f.id for f in ctx.features("classify")] == ["ok"]
    assert sorted(r.id for r in rejected) == ["too-wide", "way-too-wide"]
    assert rejected[0].reason == "form_scope_rule"


def test_drops_a_wide_override_row_too_a_manifest_can_be_edited_rows_outlive_it():
    built = build_scope_context({"overrides": [override(id="cohort-wide", scope="cohort")]})
    assert built.ctx.override("cilantro") is None
    assert len(built.rejected) == 1


def test_returns_the_single_narrowest_row():
    ctx = build_scope_context(
        {
            "overrides": [
                override(id="user", scope="user", value="produce"),
                override(id="device", scope="device", value="spices"),
                override(id="assoc", scope="associated", value="dairy"),
            ]
        }
    ).ctx
    assert ctx.override("cilantro").id == "device"


def test_prefers_the_stronger_rank_when_two_rows_share_a_scope():
    ctx = build_scope_context(
        {
            "overrides": [
                override(id="human", scope="user", rank="human"),
                override(id="admin", scope="user", rank="admin"),
            ]
        }
    ).ctx
    assert ctx.override("cilantro").id == "admin"


def test_returns_null_for_an_unknown_key():
    ctx = build_scope_context({"overrides": [override(id="a", scope="user")]}).ctx
    assert ctx.override("basil") is None


def test_records_every_row_the_run_actually_read_and_nothing_it_did_not():
    ctx = build_scope_context(
        {
            "features": [
                feature(id="read-me", scope="user", node_id="classify"),
                feature(id="never-read", scope="user", node_id="table"),
            ],
            "overrides": [override(id="o1", scope="user")],
        }
    ).ctx
    ctx.features("classify")
    ctx.override("cilantro")
    assert sorted(ctx.consumed()) == ["o1", "read-me"]


def test_exposes_them_read_only():
    ctx = build_scope_context({"ids": {"user": "u1", "tenant": "t1"}, "grants": ["recipes:read"]}).ctx
    assert ctx.ids["user"] == "u1"
    assert list(ctx.grants) == ["recipes:read"]
    with pytest.raises(TypeError):
        ctx.ids["user"] = "u2"  # type: ignore[index]


def test_empty_scope_context_has_nothing():
    ctx = empty_scope_context()
    assert dict(ctx.ids) == {}
    assert ctx.features("x") == []
    assert ctx.override("k") is None
    assert ctx.consumed() == []
