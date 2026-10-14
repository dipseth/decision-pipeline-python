from types import SimpleNamespace

from decision_pipeline.experiments import (
    ExperimentSpec,
    GuardPatch,
    ManifestPatch,
    assign_arm,
    experiment_spec_from_dict,
    is_empty_patch,
    manifest_patch_from_dict,
    merge_patches,
    shadow_challenger,
    widest_direct_feedback_scope,
)
from decision_pipeline.types import ToolSelectable, ToolSpec


def split(**over) -> ExperimentSpec:
    base = dict(
        id="gate-0.7-vs-0.6",
        unit="user",
        mode="split",
        champion="control",
        enabled=True,
        arms={
            "control": ManifestPatch(),
            "b": ManifestPatch(thresholds={"gate": {"category": 0.6}}),
        },
    )
    base.update(over)
    return ExperimentSpec(**base)


# --- assign_arm ---


def test_is_the_champion_until_the_experiment_is_enabled_declared_is_not_running():
    assert assign_arm(split(enabled=False), "u1").arm == "control"
    assert assign_arm(split(enabled=False), "u1").reason == "disabled"


def test_is_the_champion_for_the_offline_modes_they_never_touch_a_live_run():
    for mode in ("replay", "replay-redecide"):
        assert assign_arm(split(mode=mode), "u1").arm == "control"


def test_is_the_champion_when_the_unit_has_no_id():
    a = assign_arm(split(), None)
    assert a.arm == "control"
    assert a.reason == "no-unit-id"


def test_is_deterministic_and_recomputable_from_the_recorded_hash():
    first = assign_arm(split(), "user-42")
    second = assign_arm(split(), "user-42")
    assert first == second
    assert first.hash == "gate-0.7-vs-0.6:user:user-42"


def test_splits_roughly_evenly_across_two_arms():
    b = sum(1 for i in range(1000) if assign_arm(split(), f"u{i}").arm == "b")
    assert 400 < b < 600


def test_honours_weights():
    spec = split(weights={"control": 9, "b": 1})
    b = sum(1 for i in range(2000) if assign_arm(spec, f"u{i}").arm == "b")
    assert 120 < b < 280


def test_ships_the_champion_in_shadow_mode_the_challenger_runs_separately():
    live = assign_arm(split(mode="shadow"), "u1")
    assert live.arm == "control"
    assert live.ships is True

    challenger = shadow_challenger(split(mode="shadow"), "b", "u1")
    assert challenger.arm == "b"
    # Without this the first shadow test overwrites production facts.
    assert challenger.ships is False


def test_independent_experiments_stay_orthogonal():
    a = split(id="exp-a")
    b = split(id="exp-b")
    agree = sum(1 for i in range(500) if assign_arm(a, f"u{i}").arm == assign_arm(b, f"u{i}").arm)
    assert 200 < agree < 300


# --- widest_direct_feedback_scope ---


def test_looks_only_at_override_and_feature_forms():
    assert (
        widest_direct_feedback_scope(
            [
                {"form": "override", "scope": "device"},
                SimpleNamespace(form="feature", scope="user"),
                {"form": "queue", "scope": "global"},
            ]
        )
        == "user"
    )
    assert widest_direct_feedback_scope([{"form": "score", "scope": "global"}]) is None


# --- merge_patches ---


def test_merges_per_key_and_concatenates_guards():
    merged = merge_patches(
        [
            ManifestPatch(thresholds={"gate": {"category": 0.6}}, guards=[GuardPatch(from_="a", to="b", when=None)]),
            ManifestPatch(thresholds={"gate": {"tools": 0.9}}, prompts={"writer": "p2"}),
        ]
    )
    assert merged.thresholds == {"gate": {"category": 0.6, "tools": 0.9}}
    assert merged.prompts == {"writer": "p2"}
    assert len(merged.guards) == 1


def test_lets_a_later_patch_win_on_the_same_key():
    merged = merge_patches(
        [
            ManifestPatch(thresholds={"gate": {"category": 0.6}}),
            ManifestPatch(thresholds={"gate": {"category": 0.4}}),
        ]
    )
    assert merged.thresholds["gate"]["category"] == 0.4


# --- the wire shape ---


def test_patch_round_trips_the_ts_json_shape():
    wire = {
        "thresholds": {"gate": {"category": 0.6}},
        "guards": [
            {"from": "a", "to": "b", "when": None},
            {"from": "c", "to": "d", "when": {"gate": "g", "branch": "yes"}},
        ],
        "tools": {"writer": {"static": ["x"], "selectable": {"candidates": ["y"], "from": "dec", "gate": "g", "threshold": 0.5}}},
    }
    patch = manifest_patch_from_dict(wire)
    assert patch.guards[0] == GuardPatch(from_="a", to="b", when=None)
    assert patch.guards[1].when.branch == "yes"
    assert isinstance(patch.tools["writer"], ToolSpec)
    assert patch.tools["writer"].selectable == ToolSelectable(candidates=["y"], from_="dec", gate="g", threshold=0.5)
    assert patch.to_dict() == wire


def test_is_empty_patch():
    assert is_empty_patch(ManifestPatch())
    assert is_empty_patch(manifest_patch_from_dict({}))
    assert not is_empty_patch(ManifestPatch(prompts={"w": "p"}))


def test_experiment_spec_from_dict():
    spec = experiment_spec_from_dict(
        {
            "id": "e",
            "unit": "user",
            "mode": "split",
            "champion": "control",
            "arms": {"control": {}, "b": {"routes": {"w": "r"}}},
            "enabled": True,
        }
    )
    assert spec.arms["b"].routes == {"w": "r"}
    assert spec.enabled is True
