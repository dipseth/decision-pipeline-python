# Ported from src/jev.test.ts.
import pytest

from decision_pipeline.jev import (
    JEV_IMAGES_HEADER,
    JEV_MODEL,
    check_jev_request,
    jev_answers_to_decide_result,
    jev_headers,
    jev_needs_images,
    jev_supports_images,
)

MUG_REQUEST = {
    "model": JEV_MODEL,
    "state": {"photo": "6e1d0b7c-kitchen"},
    "assets": {"6e1d0b7c-kitchen": {"type": "image", "part": "kitchen-photo"}},
    "questions": {
        "mug_visible": {"type": "noul", "instructions": "Is there a coffee mug in `photo`?"},
        "mug": {"type": "box", "instructions": "the coffee mug in `photo`"},
    },
}


def test_knows_which_models_read_images():
    assert JEV_MODEL == "jev-1.14.0"
    assert jev_supports_images("jev-1.14.0")
    assert jev_supports_images("jev-1.15.2")
    assert jev_supports_images("jev-preview")
    assert not jev_supports_images("jev-1.13.0")


def test_sends_the_images_header_only_when_an_image_or_a_box_is_in_the_request():
    assert jev_needs_images(MUG_REQUEST)
    assert jev_headers("k", MUG_REQUEST) == {"authorization": "Bearer k", JEV_IMAGES_HEADER: "true"}
    text_only = {"questions": {"q": {"type": "noul", "instructions": "Is `note` dry?"}}}
    assert jev_headers("k", text_only) == {"authorization": "Bearer k"}


def test_refuses_a_box_on_a_model_that_cannot_answer_one_and_a_box_with_criteria():
    assert check_jev_request(MUG_REQUEST) == []
    problems = check_jev_request(
        {**MUG_REQUEST, "model": "jev-1.13.0", "questions": {"mug": {"type": "box", "instructions": "the mug in `photo`", "criteria": {"a": "b"}}}}
    )
    assert len(problems) == 2
    assert "takes no criteria" in problems[0]
    assert "does not answer box questions" in problems[1]


def test_maps_noul_choice_and_box_answers_onto_the_decide_ports_result():
    result = jev_answers_to_decide_result(
        {
            "mug_visible": {"type": "noul", "noul": 0.97},
            "room": {"type": "choice", "probabilities": {"kitchen": 0.8, "office": 0.2}},
            "mug": {"type": "box", "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}, "confidence": 0.92},
        },
        usage={"input_tokens": 290, "output_tokens": 21},
        cost_usd=lambda u: u.get("input_tokens", 0) * 1e-6,
    )
    assert result.distributions == {"mug_visible": [0.97], "room": [0.8, 0.2], "mug": [0.92]}
    assert result.distribution_options == {"mug_visible": ["true"], "room": ["kitchen", "office"], "mug": ["confidence"]}
    assert result.boxes["mug"]["box"]["left"] == 0.44
    assert result.cost_usd == pytest.approx(0.00029)
