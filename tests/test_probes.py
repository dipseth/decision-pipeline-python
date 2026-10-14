import pytest

from decision_pipeline.probes import (
    MAX_PROBES,
    ProbeRecord,
    collect_probes,
    is_probe_key,
    probe_answers,
    probe_features,
    probe_key,
    probe_question_schema,
    split_probe_distributions,
)
from decision_pipeline.types import ProbeSpec


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


# --- collect_probes ---


def test_accepts_the_three_shapes_and_assigns_slot_keys_in_order():
    s = collect_probes(
        [
            noul("Does `recipe.ingredients` include a chili paste?"),
            {"type": "choice", "instructions": "Which technique leads `recipe.steps`?", "criteria": {"wok": "stir-fry", "braise": "braise"}},
            {"type": "score", "instructions": "How regional is `recipe.notes`?", "criteria": ["generic", "national", "regional"]},
        ],
        ProbeSpec(from_="w", max=3, paths=["recipe"]),
    )
    assert list(s.asked.keys()) == ["probe_0", "probe_1", "probe_2"]
    assert s.dropped == []
    # a parsed probe is a plain dict with exactly the keys it had
    assert s.asked["probe_0"] == noul("Does `recipe.ingredients` include a chili paste?")


def test_drops_invalid_off_state_duplicate_too_long_and_over_max_probes_with_a_reason_each():
    s = collect_probes(
        {
            "probes": [
                noul("Is `recipe.title` Peruvian?"),
                {"type": "choice", "instructions": "Pick `recipe.steps`", "criteria": {"only": "one option"}},
                noul("Is the weather nice?"),
                noul("is  `recipe.title`  PERUVIAN?"),
                noul(f"Is `recipe.notes` {'x' * 900}?"),
                noul("Is `recipe.steps[0]` a marinade?"),
                noul("Is `recipe.ingredients` vegetarian?"),
            ]
        },
        ProbeSpec(from_="w", max=2, paths=["recipe"]),
    )
    assert [q["instructions"] for q in s.asked.values()] == [
        "Is `recipe.title` Peruvian?",
        "Is `recipe.steps[0]` a marinade?",
    ]
    assert [[d.index, d.reason.split(":")[0]] for d in s.dropped] == [
        [1, "invalid"],
        [2, "addresses no state path (want one of"],
        [3, "duplicate"],
        [4, "too long"],
        [6, "over max (2)"],
    ]


def test_never_throws_on_garbage_and_never_exceeds_max_probes_whatever_the_spec_says():
    assert collect_probes("nope", ProbeSpec(from_="w", max=3)).dropped[0].index == -1
    many = [noul(f"question {i}?") for i in range(20)]
    assert len(collect_probes(many, ProbeSpec(from_="w", max=99)).asked) == MAX_PROBES


# --- probe answers and features ---

ASKED = {
    "probe_0": {"type": "noul", "instructions": "a?"},
    "probe_1": {"type": "choice", "instructions": "b?", "criteria": {"x": "x", "y": "y"}},
    "probe_2": {"type": "noul", "instructions": "c?"},
}


def test_pairs_answers_with_default_option_keys_and_lists_the_unanswered():
    answers, unanswered = probe_answers(ASKED, {"probe_0": [0.9], "probe_1": [0.6, 0.4]}, None)
    assert [[a.key, a.options] for a in answers] == [
        ["probe_0", ["true"]],
        ["probe_1", ["x", "y"]],
    ]
    assert unanswered == ["probe_2"]


def test_emits_aggregates_only_no_per_slot_columns():
    answers, _ = probe_answers(ASKED, {"probe_0": [0.9], "probe_1": [0.6, 0.4]}, None)
    f = probe_features("d", ProbeRecord(from_="w", asked=ASKED, dropped=[], unanswered=["probe_2"]), answers)
    assert not any("probe_" in k for k in f)
    assert f["d.probes.asked"] == 3
    assert f["d.probes.answered"] == 2
    assert f["d.probes.min_margin"] == pytest.approx(0.2)


def test_splits_probe_distributions_from_the_static_questions():
    fixed, probes = split_probe_distributions({"dish": [0.9], "probe_0": [0.2]})
    assert fixed == {"dish": [0.9]}
    assert probes == {"probe_0": [0.2]}


# --- extras for the Python port: the schema and key helpers ---


def test_probe_keys():
    assert probe_key(3) == "probe_3"
    assert is_probe_key("probe_3") and not is_probe_key("dish")


def test_schema_trims_ignores_extras_and_keeps_optional_keys_absent():
    q = probe_question_schema.validate_python({"type": "noul", "instructions": "  hi?  ", "junk": 1})
    assert q.model_dump(exclude_unset=True) == {"type": "noul", "instructions": "hi?"}


def _reasons(item) -> str:
    return collect_probes([item], ProbeSpec(from_="w", max=1)).dropped[0].reason


def test_schema_messages_match_ts():
    assert "option keys are snake_case" in _reasons({"type": "choice", "instructions": "q", "criteria": {"Bad": "x", "ok": "y"}})
    assert "a choice needs at least two options" in _reasons({"type": "choice", "instructions": "q", "criteria": {"a": "x"}})
    too_many = {f"k{i}": "v" for i in range(13)}
    assert "a choice allows at most 12 options" in _reasons({"type": "choice", "instructions": "q", "criteria": too_many})
    assert "a score needs at least two levels" in _reasons({"type": "score", "instructions": "q", "criteria": ["only"]})
    assert _reasons({"type": "score", "instructions": "q", "criteria": [str(i) for i in range(13)]}).startswith("invalid: criteria")
    assert _reasons({"type": "noul", "instructions": "   "}).startswith("invalid: instructions")
    assert _reasons({"type": "noul", "instructions": "q", "criteria": {"true": "a"}}).startswith("invalid: criteria.false")
    assert _reasons("nope").startswith("invalid: ")
    assert _reasons({"type": "weird", "instructions": "q"}).startswith("invalid: ")


def test_noul_with_criteria_and_why_round_trips():
    q = {"type": "noul", "instructions": "q?", "criteria": {"true": "yes", "false": "no"}, "why": "because"}
    assert collect_probes([q], ProbeSpec(from_="w", max=1)).asked["probe_0"] == q


def test_one_number_margin_is_distance_from_a_coin_flip():
    asked = {"probe_0": ASKED["probe_0"]}
    answers, _ = probe_answers(asked, {"probe_0": [0.9]}, None)
    f = probe_features("d", ProbeRecord(from_="w", asked=asked, dropped=[], unanswered=[]), answers)
    assert f["d.probes.min_margin"] == pytest.approx(0.8)
    assert "d.probes.max_entropy" not in f


def test_record_to_dict_uses_the_ts_keys():
    r = ProbeRecord(from_="w", asked={"probe_0": ASKED["probe_0"]}, dropped=[], unanswered=[])
    assert r.to_dict() == {"from": "w", "asked": {"probe_0": ASKED["probe_0"]}, "dropped": [], "unanswered": []}
    assert ProbeRecord.from_dict(r.to_dict()) == r
