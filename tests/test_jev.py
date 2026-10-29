# Ported from src/jev.test.ts.
import pytest

from decision_pipeline.jev import (
    JEV_IMAGES_HEADER,
    JEV_LONG_CONTEXT_HEADER,
    JEV_MODEL,
    check_jev_request,
    jev_answers_to_decide_result,
    jev_headers,
    jev_needs_images,
    jev_supports_box,
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


def test_knows_which_models_answer_a_box():
    assert JEV_MODEL == "jev-1.14.0"
    assert jev_supports_box("jev-1.14.0")
    assert jev_supports_box("jev-1.15.2")
    assert jev_supports_box("jev-preview")
    assert jev_supports_box("jev-latest")
    assert not jev_supports_box("jev-1.13.0")


def test_sends_the_images_header_only_when_an_image_or_a_box_is_in_the_request():
    assert jev_needs_images(MUG_REQUEST)
    assert jev_headers("k", MUG_REQUEST) == {"authorization": "Bearer k", JEV_IMAGES_HEADER: "true"}
    text_only = {"questions": {"q": {"type": "noul", "instructions": "Is `note` dry?"}}}
    assert jev_headers("k", text_only) == {"authorization": "Bearer k"}
    assert jev_headers("k", text_only, long_context=True) == {"authorization": "Bearer k", JEV_LONG_CONTEXT_HEADER: "true"}


def test_lets_images_through_on_1_13_but_refuses_a_box_there_and_a_box_with_criteria():
    assert check_jev_request(MUG_REQUEST) == []
    images_on_113 = {**MUG_REQUEST, "model": "jev-1.13.0", "questions": {"mug_visible": MUG_REQUEST["questions"]["mug_visible"]}}
    assert check_jev_request(images_on_113) == []
    assert jev_needs_images(images_on_113)
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


# ---- typesafe-sdk objects (Python only: the SDK client is the port's, these helpers just read its types)


def test_reads_sdk_question_objects_and_sdk_images():
    sdk = pytest.importorskip("typesafe_sdk")
    from decision_pipeline.jev import jev_extra_headers

    text_only = {"q": sdk.Noul(instructions="Is `note` dry?")}
    assert jev_extra_headers(text_only, {"note": "wet"}) == {}
    assert jev_extra_headers(text_only, long_context=True) == {JEV_LONG_CONTEXT_HEADER: "true"}
    assert jev_extra_headers(text_only, images=True) == {JEV_IMAGES_HEADER: "true"}

    with_box = {"seen": sdk.Noul(instructions="Is there a mug?"), "mug": sdk.Box(instructions="the coffee mug")}
    assert jev_extra_headers(with_box) == {JEV_IMAGES_HEADER: "true"}
    assert check_jev_request({"model": "jev-1.13.0", "questions": with_box}) == [
        'model "jev-1.13.0" does not answer box questions; send "jev-1.14.0" or "jev-preview"'
    ]
    assert check_jev_request({"model": JEV_MODEL, "questions": with_box}) == []

    photo = sdk.Image(b"\xff\xd8\xff\xe0fake-jpeg")
    assert jev_extra_headers(text_only, {"claim": "chipped rim", "photo": photo}) == {JEV_IMAGES_HEADER: "true"}
    assert jev_extra_headers({"q": sdk.Noul(instructions=["Is this damaged?", photo])}) == {JEV_IMAGES_HEADER: "true"}
    assert jev_needs_images({"questions": text_only, "assets": {"<photo>": photo}})
    assert jev_extra_headers(text_only, "<photo> arrived chipped", assets={"<photo>": photo}) == {JEV_IMAGES_HEADER: "true"}
    # A box whose instructions are content, not a string, is fine; a missing one is flagged.
    assert check_jev_request({"model": JEV_MODEL, "questions": {"mug": sdk.Box(instructions=["the mug in", photo])}}) == []
    assert "needs instructions" in check_jev_request({"model": JEV_MODEL, "questions": {"mug": sdk.Box()}})[0]


def test_maps_an_sdk_system_one_response_onto_the_decide_ports_result():
    sdk = pytest.importorskip("typesafe_sdk")
    from decision_pipeline.jev import jev_response_to_decide_result

    import json

    # The SDK decodes the wire JSON in strict mode, where score keys arrive as "0", "1", ...
    response = sdk.SystemOneResponse.model_validate_json(json.dumps(
        {
            "model": "jev-1.14.0",
            "usage": {"input_tokens": 290, "output_tokens": None},
            "answers": {
                "mug_visible": {"type": "noul", "noul": 0.97},
                "room": {"type": "choice", "choice": "kitchen", "confidence": 0.8, "probabilities": {"kitchen": 0.8, "office": 0.2}},
                "urgency": {"type": "score", "score": 1.7, "confidence": 0.8, "legend": {"0": "later", "1": "soon", "2": "now"},
                            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}},
                "mug": {"type": "box", "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}, "confidence": 0.92},
            },
        }
    ))
    seen: list = []
    result = jev_response_to_decide_result(response, cost_usd=lambda u: seen.append(dict(u)) or u.get("input_tokens", 0) * 1e-6)
    assert result.distributions == {"mug_visible": [0.97], "room": [0.8, 0.2], "urgency": [0.1, 0.1, 0.8], "mug": [0.92]}
    assert result.distribution_options == {
        "mug_visible": ["true"], "room": ["kitchen", "office"], "urgency": ["0", "1", "2"], "mug": ["confidence"],
    }
    assert result.boxes == {"mug": {"box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}, "confidence": 0.92}}
    assert seen == [{"input_tokens": 290}]  # a count the API left out is dropped, not passed on as None
    assert result.cost_usd == pytest.approx(0.00029)
    # The raw response dict reads the same.
    raw = jev_response_to_decide_result(response.model_dump(mode="json"))
    assert raw.distributions == result.distributions
