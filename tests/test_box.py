# Ported from src/box.test.ts.
import pytest
from pydantic import BaseModel

from decision_pipeline import testing as dp_testing  # not imported by name: pytest would collect `test_ports`
from decision_pipeline.box import (
    box_area,
    box_from_tile,
    box_iou,
    box_to_pixels,
    check_box_answer,
    edge_check_questions,
    expand_box_edges,
    pad_box,
    tile_boxes,
)
from decision_pipeline.define import EvalSpec, PipelineSpec, define_pipeline
from decision_pipeline.errors import NodeFailedError
from decision_pipeline.run import run_pipeline
from decision_pipeline.types import (
    CodeNode,
    DecideNode,
    Edge,
    EdgeGuard,
    FeedbackEdge,
    GateNode,
    ScoreConfigSpec,
    StoreNode,
)

MUG = {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}


# --- box geometry ---


def test_accepts_the_docs_mug_and_names_what_is_wrong_with_a_bad_box():
    assert check_box_answer({"box": MUG, "confidence": 0.92}) is None
    assert "left 0.9 is not less than right" in check_box_answer({"box": {**MUG, "left": 0.9}, "confidence": 0.9})
    assert "bottom 1.2 is not in 0..1" in check_box_answer({"box": {**MUG, "bottom": 1.2}, "confidence": 0.9})
    assert "confidence" in check_box_answer({"box": MUG, "confidence": 2})
    assert check_box_answer({"confidence": 0.5}) == "no box"


def test_converts_to_pixels_in_pil_crop_order():
    assert box_to_pixels(MUG, 960, 654) == {"left": 422, "top": 164, "right": 806, "bottom": 536}


def test_pads_outward_and_stays_inside_the_image():
    padded = pad_box({"top": 0.02, "bottom": 0.99, "left": 0.3, "right": 0.6}, 0.05)
    assert padded["top"] == 0 and padded["bottom"] == 1
    assert padded["left"] == pytest.approx(0.25) and padded["right"] == pytest.approx(0.65)


def test_scores_overlap_as_iou():
    assert box_iou(MUG, MUG) == pytest.approx(1)
    assert box_iou(MUG, {"top": 0, "bottom": 0.1, "left": 0, "right": 0.1}) == 0
    half = {**MUG, "right": (MUG["left"] + MUG["right"]) / 2}
    assert box_iou(MUG, half) == pytest.approx(0.5)


def test_tiles_the_image_and_maps_a_box_on_a_tile_back_to_the_full_image():
    tiles = tile_boxes(2, 2)
    assert list(tiles) == ["r0c0", "r0c1", "r1c0", "r1c1"]
    full = box_from_tile(tiles["r1c1"], {"top": 0, "bottom": 0.5, "left": 0.5, "right": 1})
    assert full == {"top": 0.5, "bottom": 0.75, "left": 0.75, "right": 1}
    assert box_area(full) == pytest.approx(0.0625)
    with pytest.raises(ValueError, match="positive integers"):
        tile_boxes(0, 2)


def test_writes_one_noul_per_edge_and_moves_out_the_edges_the_target_continues_past():
    qs = edge_check_questions("the mug", "crop")
    assert list(qs) == ["past_top", "past_bottom", "past_left", "past_right"]
    assert qs["past_left"]["instructions"] == "Does the mug in `crop` continue past the left edge of the image?"

    out = expand_box_edges(MUG, {"left": 0.8, "right": 0.3, "top": 0.7, "bottom": 0.99}, 0.6, 0.05)
    assert out["moved"] == ["top", "bottom", "left"]
    assert out["box"]["left"] == pytest.approx(0.39)
    assert out["box"]["right"] == MUG["right"]
    # An edge already at the image's border cannot move, and is not reported as moved.
    assert expand_box_edges({**MUG, "top": 0}, {"top": 0.9}, 0.5, 0.1)["moved"] == []


# --- a Box inside a pipeline: the Noul says whether it is there, the Box says
# where, the gate decides on both confidences, code crops. ---


class In(BaseModel):
    id: str
    width: int
    height: int


class Out(BaseModel):
    crop: dict | None


def spec(seen: dict | None = None, **overrides) -> PipelineSpec:
    seen = {} if seen is None else seen

    def interpret(d, _args, boxes):
        seen["boxes"] = boxes
        return {"visible": (d.get("mug_visible") or [0])[0], "mug": boxes.get("mug")}

    def gate_run(args):
        visible, mug = args.primary["visible"], args.primary["mug"]
        t = args.thresholds
        if visible < t["visible"] or not mug:
            return {"branch": "none", "reason": f"mug_visible {visible}"}
        if mug["confidence"] >= t["box"]:
            return {"branch": "crop", "confidence": mug["confidence"]}
        return {"branch": "pad_crop", "reason": f"box confidence {mug['confidence']}", "confidence": mug["confidence"]}

    nodes = {
        "locate": DecideNode(questions="jev-mug-locate", state=["photo"], interpret=interpret, version="1"),
        "gate": GateNode(run=gate_run, branches=["crop", "pad_crop", "none"], thresholds={"visible": 0.5, "box": 0.6}, version="1"),
        # A gate passes its primary input through: this is `locate`'s value.
        "crop": CodeNode(
            role="derive",
            run=lambda a: {"crop": box_to_pixels(a.primary["mug"]["box"], a.input.width, a.input.height)},
            version="1",
        ),
        "pad_crop": CodeNode(
            role="derive",
            thresholds={"pad": 0.05},
            run=lambda a: {"crop": box_to_pixels(pad_box(a.primary["mug"]["box"], a.thresholds["pad"]), a.input.width, a.input.height)},
            version="1",
        ),
        "none": CodeNode(role="derive", run=lambda _a: {"crop": None}, version="1"),
        "persist": StoreNode(target="memory", scope="user", version="1"),
    }
    nodes.update(overrides)
    return PipelineSpec(
        id="mug-finder",
        fact="mug_crop",
        input=In,
        output=Out,
        trigger=["on-demand"],
        group=lambda i: f"photo-{i.id}",
        result=["crop", "pad_crop", "none"],
        nodes=nodes,
        edges=[
            Edge(from_="locate", to="gate"),
            Edge(from_="gate", to="crop", when=EdgeGuard(gate="gate", branch="crop")),
            Edge(from_="gate", to="pad_crop", when=EdgeGuard(gate="gate", branch="pad_crop")),
            Edge(from_="gate", to="none", when=EdgeGuard(gate="gate", branch="none")),
            Edge(from_="crop", to="persist"),
            Edge(from_="pad_crop", to="persist"),
            Edge(from_="none", to="persist"),
        ],
        feedback=[
            FeedbackEdge(
                id="redraw",
                from_="surface:crop-editor",
                to="gate",
                source="explicit",
                scope="user",
                form="score",
                latency="deferred",
                score=ScoreConfigSpec(name="crop_iou", data_type="NUMERIC"),
            )
        ],
        eval=EvalSpec(dataset="mug-eval", gate_metric="score_positive"),
    )


PHOTO = In(id="p1", width=960, height=654)


def located(visible: float, box: dict, confidence: float) -> dict:
    return {"distributions": {"mug_visible": [visible]}, "boxes": {"mug": {"box": box, "confidence": confidence}}}


async def test_records_the_box_mirrors_its_confidence_for_the_gate_and_adds_geometry_features():
    seen: dict = {}
    ports = dp_testing.test_ports(decide={"locate": located(0.97, MUG, 0.92)})
    result = await run_pipeline(define_pipeline(spec(seen)), PHOTO, ports)

    assert result.route == "crop"
    assert result.output.crop == {"left": 422, "top": 164, "right": 806, "bottom": 536}
    node = next(n for n in result.record.nodes if n.id == "locate")
    assert node.boxes == {"mug": {"box": MUG, "confidence": 0.92}}
    assert node.distributions["mug"] == [0.92]
    assert node.distribution_options["mug"] == ["confidence"]
    assert seen["boxes"]["mug"]["confidence"] == 0.92
    features = result.record.features
    assert features["locate.mug.top"] == 0.92
    assert features["locate.mug.box_area"] == pytest.approx(0.4 * 0.57)
    assert features["locate.mug.box_cx"] == pytest.approx(0.64)


async def test_pads_a_low_confidence_box_and_ignores_the_box_when_the_noul_says_it_is_absent():
    padded = await run_pipeline(define_pipeline(spec()), PHOTO, dp_testing.test_ports(decide={"locate": located(0.9, MUG, 0.4)}))
    assert padded.route == "pad_crop"
    assert padded.output.crop == {"left": 374, "top": 131, "right": 854, "bottom": 569}

    none = await run_pipeline(define_pipeline(spec()), PHOTO, dp_testing.test_ports(decide={"locate": located(0.1, MUG, 0.95)}))
    assert none.route == "none"
    assert next(n for n in none.record.nodes if n.id == "gate").branch_reason == "mug_visible 0.1"


async def test_fails_the_node_on_a_box_whose_edges_cross():
    ports = dp_testing.test_ports(decide={"locate": located(0.9, {**MUG, "top": 0.9}, 0.9)})
    with pytest.raises(NodeFailedError, match=r"bad box answer: mug \(top 0.9 is not less than bottom"):
        await run_pipeline(define_pipeline(spec()), PHOTO, ports)


async def test_keeps_a_decide_nodes_value_shape_and_hands_the_edges_on_as_args_boxes():
    seen: dict = {}

    def gate_run(args):
        seen["primary"], seen["boxes"] = args.primary, args.boxes
        d, t = args.primary, args.thresholds
        return "crop" if d["mug_visible"][0] >= t["visible"] and d["mug"][0] >= t["box"] else "none"

    pipeline = define_pipeline(
        spec(
            locate=DecideNode(questions="jev-mug-locate", version="1"),
            gate=GateNode(run=gate_run, branches=["crop", "pad_crop", "none"], thresholds={"visible": 0.5, "box": 0.6}, version="1"),
            crop=CodeNode(
                role="derive",
                run=lambda a: {"crop": box_to_pixels(a.boxes["locate"]["mug"]["box"], a.input.width, a.input.height)},
                version="1",
            ),
        )
    )
    result = await run_pipeline(pipeline, PHOTO, dp_testing.test_ports(decide={"locate": located(0.97, MUG, 0.92)}))
    assert seen["primary"] == {"mug_visible": [0.97], "mug": [0.92]}
    assert dict(seen["boxes"]) == {"locate": {"mug": {"box": MUG, "confidence": 0.92}}}
    assert result.output.crop == {"left": 422, "top": 164, "right": 806, "bottom": 536}


async def test_a_two_argument_interpret_still_works_and_reads_its_boxes_from_args():
    seen: dict = {}

    def interpret(d, args):
        seen["mine"] = args.boxes.get(args.node.id)
        return {"visible": d["mug_visible"][0], "mug": args.boxes[args.node.id]["mug"]}

    pipeline = define_pipeline(spec(locate=DecideNode(questions="jev-mug-locate", interpret=interpret, version="1")))
    result = await run_pipeline(pipeline, PHOTO, dp_testing.test_ports(decide={"locate": located(0.97, MUG, 0.92)}))
    assert result.route == "crop"
    assert seen["mine"]["mug"]["confidence"] == 0.92
