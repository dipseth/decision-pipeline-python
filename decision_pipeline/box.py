"""
Box — where something is in an image (Jev 1.14, `type: "box"`).

A Box answer is four edges as fractions of the image (`left`/`right` of the
width, `top`/`bottom` of the height, origin top-left) plus a confidence. It
is an ESTIMATE like any decide answer: the decide port returns it under
`boxes`, the runtime checks it, records it on the node, mirrors its
confidence into `distributions` as `[confidence]` (option `"confidence"`) so
a gate thresholds it exactly like a Noul, and adds flat geometry features.
What to DO with a box (crop, pad, ask again on a tile) stays in code.

Two facts from the API shape every pipeline that asks one:
 - There is no "not found". A Box about something absent still returns four
   edges, so pair it with a Noul ("is there a mug in `photo`?") in the same
   request and let a gate ignore the box when the Noul is low.
 - One box per question. Several targets are several Box questions.

Boxes are plain dicts (`Box`, `BoxAnswer` in ./types), so they are JSON as-is
and match the TS record byte for byte.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from .types import Box, BoxAnswer

__all__ = [
    "BOX_CONFIDENCE_OPTION",
    "BOX_EDGES",
    "BoxQuestion",
    "check_box_answer",
    "box_width",
    "box_height",
    "box_area",
    "box_center",
    "pad_box",
    "box_to_pixels",
    "box_iou",
    "tile_boxes",
    "box_from_tile",
    "edge_check_questions",
    "expand_box_edges",
    "box_features",
]

#: A Box question as Jev takes it: `{"type": "box", "instructions": "the coffee mug in `photo`"}`. No criteria.
BoxQuestion = dict

#: The option key a Box's mirrored distribution carries.
BOX_CONFIDENCE_OPTION = "confidence"

#: The four edges, in the order edge checks are asked and reported.
BOX_EDGES: tuple[str, ...] = ("top", "bottom", "left", "right")


def _unit(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1


def _fmt(v: Any) -> str:
    # Match JS `String(n)`: 0.9, 1, undefined.
    if v is None:
        return "undefined"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def check_box_answer(a: Any) -> Optional[str]:
    """None when the answer is a usable box, else why not."""
    if not isinstance(a, Mapping):
        return "not an object"
    confidence = a.get("confidence")
    if not _unit(confidence):
        return f"confidence {_fmt(confidence)} is not in 0..1"
    box = a.get("box")
    if not isinstance(box, Mapping):
        return "no box"
    for edge in BOX_EDGES:
        if not _unit(box.get(edge)):
            return f"{edge} {_fmt(box.get(edge))} is not in 0..1"
    if not box["left"] < box["right"]:
        return f"left {_fmt(box['left'])} is not less than right {_fmt(box['right'])}"
    if not box["top"] < box["bottom"]:
        return f"top {_fmt(box['top'])} is not less than bottom {_fmt(box['bottom'])}"
    return None


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def box_width(b: Box) -> float:
    return b["right"] - b["left"]


def box_height(b: Box) -> float:
    return b["bottom"] - b["top"]


def box_area(b: Box) -> float:
    """Fraction of the image the box covers."""
    return box_width(b) * box_height(b)


def box_center(b: Box) -> dict[str, float]:
    return {"x": (b["left"] + b["right"]) / 2, "y": (b["top"] + b["bottom"]) / 2}


def pad_box(b: Box, pad: float) -> Box:
    """
    Move every edge outward by `pad` (a fraction of the image), kept inside it.
    The docs' answer to a low-confidence box you must not cut off when cropping.
    """
    return {
        "left": _clamp01(b["left"] - pad),
        "top": _clamp01(b["top"] - pad),
        "right": _clamp01(b["right"] + pad),
        "bottom": _clamp01(b["bottom"] + pad),
    }


def _round(v: float) -> int:
    # JS Math.round (half up), not Python's banker's rounding: the TS port gets the same pixels.
    return math.floor(v + 0.5)


def box_to_pixels(b: Box, width: int, height: int) -> dict[str, int]:
    """
    Pixel rectangle for an image of `width` x `height`, rounded. PIL's crop
    takes `(r["left"], r["top"], r["right"], r["bottom"])`.
    """
    return {
        "left": _round(b["left"] * width),
        "top": _round(b["top"] * height),
        "right": _round(b["right"] * width),
        "bottom": _round(b["bottom"] * height),
    }


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union, 0..1. The score a corrected box earns against the pipeline's."""
    w = min(a["right"], b["right"]) - max(a["left"], b["left"])
    h = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def tile_boxes(rows: int, cols: int) -> dict[str, Box]:
    """
    Split the image into a `rows` x `cols` grid, named `r<row>c<col>`. Find a
    small target in a large image by asking a Choice over the tiles first, then
    a Box on the chosen tile's crop, then `box_from_tile` back to the full image.
    """
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        raise ValueError(f"tileBoxes: rows and cols must be positive integers, got {rows} x {cols}")
    return {
        f"r{r}c{c}": {"top": r / rows, "bottom": (r + 1) / rows, "left": c / cols, "right": (c + 1) / cols}
        for r in range(rows)
        for c in range(cols)
    }


def box_from_tile(tile: Box, inner: Box) -> Box:
    """A box answered on a crop (`tile`, in full-image fractions) mapped back to the full image."""
    return {
        "left": tile["left"] + inner["left"] * box_width(tile),
        "right": tile["left"] + inner["right"] * box_width(tile),
        "top": tile["top"] + inner["top"] * box_height(tile),
        "bottom": tile["top"] + inner["bottom"] * box_height(tile),
    }


def edge_check_questions(target: str, crop_path: str, prefix: str = "past") -> dict[str, dict[str, str]]:
    """
    One Noul per edge — "does the target in `crop` continue past the left edge
    of the image?" — the check that caught 30 of 36 cut-off crops in TypeSafe's
    test, against 26 for one "is all of it inside?" question. Keys are
    `<prefix>_<edge>`; ask them on the CROP, not the full image.
    """
    return {
        f"{prefix}_{edge}": {
            "type": "noul",
            "instructions": f"Does {target} in `{crop_path}` continue past the {edge} edge of the image?",
        }
        for edge in BOX_EDGES
    }


def expand_box_edges(
    b: Box,
    past_edge: Mapping[str, float],
    threshold: float,
    step: float,
) -> dict[str, Any]:
    """
    Move out each edge whose edge-check Noul is at or above `threshold`, by
    `step` (a fraction of the image), kept inside it. A code decision on model
    estimates: the threshold is data, so declare it on the node that calls this.
    Returns `{"box", "moved"}`; nothing moved means the box holds.
    """
    box: dict[str, float] = {edge: b[edge] for edge in ("top", "bottom", "left", "right")}  # type: ignore[literal-required]
    moved: list[str] = []
    for edge in BOX_EDGES:
        p = past_edge.get(edge)
        if not isinstance(p, (int, float)) or p < threshold:
            continue
        before = box[edge]
        box[edge] = _clamp01(before - step if edge in ("top", "left") else before + step)
        if box[edge] != before:
            moved.append(edge)
    return {"box": box, "moved": moved}


def box_features(inputs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """
    `<node_id>.<question>.box_area`, `.box_cx`, `.box_cy` for every box a decide
    node answered. Confidence needs no column of its own: it is mirrored into
    `distributions`, so `.top` already carries it.
    """
    out: dict[str, float] = {}
    for item in inputs:
        node_id = item["node_id"]
        answers: Mapping[str, BoxAnswer] = item["boxes"]
        for question, a in answers.items():
            center = box_center(a["box"])
            out[f"{node_id}.{question}.box_area"] = box_area(a["box"])
            out[f"{node_id}.{question}.box_cx"] = center["x"]
            out[f"{node_id}.{question}.box_cy"] = center["y"]
    return out
