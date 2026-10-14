"""
A decision pipeline drawn from its own JSON (#321).

One implementation for every surface: the docs export, the API and the app
all call `render_pipeline_svg(manifest.to_json())`. Nothing here imports
anything beyond the standard library — the layout is a small layered DAG,
top-down, with orthogonal edges; the output is an SVG string, byte for byte
the one the TypeScript package draws for the same input.

What the picture says, and why:
  - a node's colour is its KIND: read · decide (a model estimates) ·
    generate (a model writes) · code · gate (code decides) · store;
  - a gate is the only hexagon, and the branch that reaches a node is a
    pill on its doorstep, so "where does this fork and on what" is never in
    a tooltip;
  - a dashed outline is a node whose failure the pipeline survives
    (`onFailure` fallback / skip / revert);
  - edges are consolidated: every edge out of a node leaves by one trunk,
    every edge into a node arrives by one drop, and an edge that skips
    layers runs down a rail at the left with a tap into each layer it
    feeds — a node feeding six others is one line, not six;
  - feedback edges are the loop: a human surface (or a review queue) below
    the run, joined to the run's end by a faint dotted line and back UP to
    an EARLIER node by a dashed rail at the right — the contract's rule that
    a run never waits on a person, drawn.

Colours are CSS custom properties with fallbacks (`--dp-*`), text and plain
edges use `currentColor`, so the same string is theme-aware inline in a page
and self-contained as a file (`standalone=True` embeds a light/dark style).

The input is the `PipelineJSON` dict (`to_json()`): camelCase keys, `nodes` as
a list of dicts, `edges` with `from`/`to`/`when`, `feedback` with
`from`/`to`/`form`/`promoteVia`, optional `order`.

Numbers are printed like JavaScript's `String(n)` and every coordinate is
computed with the same operations in the same order as the TS source, so the
two packages agree to the last digit. String lengths are counted in UTF-16
code units, as JS counts them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ._compat import js_number

__all__ = [
    "DiagramLayout",
    "GapLanes",
    "LaidOutNode",
    "LaidOutSurface",
    "Layout",
    "LayoutNode",
    "feedback_label",
    "layout_pipeline",
    "node_kind_label",
    "node_name",
    "node_subtitle",
    "node_visual",
    "render_pipeline_svg",
]

_UNDEFINED: Any = object()  # TS `undefined`, internal only

# ---------------------------------------------------------------------------
# JavaScript semantics the drawing depends on
# ---------------------------------------------------------------------------


def _len16(s: str) -> int:
    """`s.length`: UTF-16 code units."""
    return len(s.encode("utf-16-le", "surrogatepass")) // 2


def _truthy(v: Any) -> bool:
    """JS truthiness: `[]` and `{}` are true, `0`, `""`, `NaN`, null and undefined are false."""
    if v is None or v is _UNDEFINED or v is False:
        return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v == v and v != 0
    if isinstance(v, str):
        return v != ""
    return True


def _num(v: float) -> str:
    """`String(n)`, including the non-finite values a degenerate layout can produce."""
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN" if v != v else ("Infinity" if v > 0 else "-Infinity")
    return js_number(v)


def _js_keys(d: Mapping[str, Any]) -> list[str]:
    """`Object.keys(d)`: integer-like keys first, ascending, then the rest in insertion order."""
    keys = [str(k) for k in d.keys()]
    ints = [k for k in keys if re.fullmatch(r"0|[1-9][0-9]*", k) and int(k) < 2**32 - 1]
    return sorted(ints, key=int) + [k for k in keys if k not in ints]


def _js_str(v: Any) -> str:
    """`String(v)` for JSON-like data."""
    if v is _UNDEFINED:
        return "undefined"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _num(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return ",".join("" if x is None or x is _UNDEFINED else _js_str(x) for x in v)
    if isinstance(v, Mapping):
        return "[object Object]"
    return str(v)


def _get(d: Any, key: str) -> Any:
    """`d[key]`, `undefined` when absent."""
    if isinstance(d, Mapping):
        return d.get(key, _UNDEFINED)
    return _UNDEFINED


def _opt(d: Any, key: str) -> Any:
    """`d?.[key]` collapsed to None for null and undefined alike (TS `??` semantics)."""
    v = _get(d, key)
    return None if v is _UNDEFINED else v


def _hypot(x: float, y: float) -> float:
    """`Math.hypot` exactly as V8 computes it (scaled, Kahan-summed)."""
    a, b = abs(x), abs(y)
    m = max(a, b)
    if m == 0:
        return 0.0
    total = 0.0
    comp = 0.0
    for v in (a, b):
        n = v / m
        summand = n * n - comp
        prelim = total + summand
        comp = (prelim - total) - summand
        total = prelim
    return math.sqrt(total) * m


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


# ICU root collation order of ASCII punctuation and symbols (what `localeCompare` uses).
_ICU_PUNCT = "_-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$"


def _locale_key(s: str) -> tuple[Any, ...]:
    """An approximation of `a.localeCompare(b)` (ICU root): punctuation < digits < letters, case second, lower first."""
    primary = []
    tertiary = []
    for c in s:
        if c in _ICU_PUNCT:
            primary.append((1, _ICU_PUNCT.index(c), ""))
        elif c.isspace():
            primary.append((0, ord(c), ""))
        elif c.isdigit():
            primary.append((2, int(c) if c.isascii() else ord(c), ""))
        elif c.isalpha():
            primary.append((3, 0, c.lower()))
        else:
            primary.append((4, ord(c), ""))
        tertiary.append(1 if c.isupper() else 0)
    return (tuple(primary), tuple(tertiary), s)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LaidOutNode:
    id: str
    node: dict[str, Any]
    layer: int
    x: float
    y: float
    w: float
    h: float
    label: str
    #: Second line: the kind, or the code role.
    kind_label: str
    #: Third line: the one name that identifies the node (prompt, questions, target…).
    name: str
    visual: str


@dataclass(kw_only=True)
class LaidOutSurface:
    id: str
    #: `surface` / `queue` / `hand-read` / `lane` … — the prefix of `from`.
    channel: str
    name: str
    x: float
    y: float
    w: float
    h: float
    edges: list[dict[str, Any]]


@dataclass(kw_only=True)
class GapLanes:
    """The horizontal lanes in the gap below layer `g`, one per trunk or rail that routes through it."""

    top: float
    bottom: float
    #: lane key (`src:<id>` or `dst:<id>`) → y of its lane
    lane: dict[str, float] = field(default_factory=dict)


@dataclass(kw_only=True)
class DiagramLayout:
    nodes: list[LaidOutNode]
    layers: list[list[str]]
    gaps: list[GapLanes]
    #: Rail key → x of its rail at the left, for edges that skip a layer. A node
    #: fanning out to several far nodes owns one rail (`src:<id>`); several nodes
    #: feeding one far node share one (`dst:<id>`).
    rails: dict[str, float]
    surfaces: list[LaidOutSurface]
    #: Feedback edge id → x of its rail at the right.
    feedback_rails: dict[str, float]
    edges: list[dict[str, Any]]
    width: float
    height: float
    #: Node ids with no outbound flow edge — where the run ends.
    sinks: list[str]
    #: The rail key of every edge that skips a layer (edge index → key).
    rail_of: dict[int, str]
    #: Left and right x of the node columns (rails sit outside).
    graph_left: float
    graph_right: float


# Short names, as the porting brief calls them.
Layout = DiagramLayout
LayoutNode = LaidOutNode


def node_visual(n: Mapping[str, Any]) -> str:
    """The six visual classes a node can have. Gate is a role, drawn as its own thing."""
    kind = _get(n, "kind")
    if kind == "code":
        return "gate" if _get(n, "role") == "gate" else "code"
    if kind in ("read", "decide", "generate", "store"):
        return kind
    return "code"


NODE_H = 62
NODE_MIN_W = 96
NODE_MAX_W = 168
COL_GAP = 28
GAP_BASE = 30
LANE_STEP = 9
RAIL_STEP = 11
PAD = 20
MONO_CH = 7.2  # 12px ui-monospace
MONO_SM_CH = 6.0  # 10px ui-monospace
SANS_CH = 5.4  # 10px system-ui
SURFACE_H = 30
SURFACE_GAP = 52


def _cut(s: str, max_len: int) -> str:
    if _len16(s) <= max_len:
        return s
    keep = max(1, max_len - 1)
    units = s.encode("utf-16-le", "surrogatepass")[: 2 * keep]
    return units.decode("utf-16-le", "replace") + "…"


def node_name(n: Mapping[str, Any]) -> str:
    """The one name that identifies a node — what its third line shows."""
    v = node_visual(n)
    if v == "read":
        return "features" if _truthy(_get(n, "providesFeatures")) else ""
    if v == "decide":
        q = _opt(n, "questions")
        if q is not None:
            return _js_str(q)
        probes = _get(n, "probes")
        return f"probes ← {_js_str(_get(probes, 'from'))}" if _truthy(probes) else ""
    if v == "generate":
        p = _opt(n, "prompt")
        return "" if p is None else _js_str(p)
    if v == "gate":
        t = _get(n, "thresholds")
        return ", ".join(_js_keys(t)) if _truthy(t) and isinstance(t, Mapping) else ""
    if v == "store":
        return " · ".join(_js_str(x) for x in (_get(n, "target"), _get(n, "scope")) if _truthy(x))
    t = _get(n, "thresholds")
    keys = _js_keys(t) if _truthy(t) and isinstance(t, Mapping) else []
    return f"{len(keys)} threshold{'' if len(keys) == 1 else 's'}" if keys else ""


def node_kind_label(n: Mapping[str, Any]) -> str:
    """The node's kind as its second line reads it."""
    v = node_visual(n)
    if v == "code":
        role = _opt(n, "role")
        return "code" if role is None else _js_str(role)
    if v == "decide" and _truthy(_get(n, "batched")):
        return "decide · batched"
    if v == "decide" and _truthy(_get(n, "questions")) and _truthy(_get(n, "probes")):
        return "decide + probes"
    return v


def node_subtitle(n: Mapping[str, Any]) -> str:
    """Kind and name on one line — for chips, tooltips and tests."""
    name = node_name(n)
    return f"{node_kind_label(n)} · {name}" if name else node_kind_label(n)


def _topo_order(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> list[str]:
    indeg: dict[str, int] = {n["id"]: 0 for n in nodes}
    for e in edges:
        indeg[e["to"]] = indeg.get(e["to"], 0) + 1
    queue = [n["id"] for n in nodes if indeg.get(n["id"], 0) == 0]
    out: list[str] = []
    while queue:
        nid = queue.pop(0)
        out.append(nid)
        for e in edges:
            if e["from"] != nid:
                continue
            d = indeg.get(e["to"], 1) - 1
            indeg[e["to"]] = d
            if d == 0:
                queue.append(e["to"])
    return out if len(out) == len(nodes) else [n["id"] for n in nodes]


def _lane_of(f: Mapping[str, Any]) -> Any:
    pv = _get(f, "promoteVia")
    if isinstance(pv, str):
        return pv
    return _get(pv, "lane") if isinstance(pv, Mapping) else _UNDEFINED


def feedback_label(f: Mapping[str, Any]) -> str:
    """The words on a feedback edge: its form, scope and latency, and whether a review lane promotes it."""
    lane = _lane_of(f)
    text = f"{_js_str(_get(f, 'form'))} · {_js_str(_get(f, 'scope'))} · {_js_str(_get(f, 'latency'))}"
    return _cut(text + (" · lane" if _truthy(lane) else ""), 34)


def _parse_from(src: str) -> tuple[str, str]:
    i = src.find(":")
    return ("surface", src) if i < 0 else (src[:i], src[i + 1 :])


def _js_min(values: Sequence[float], *extra: float) -> float:
    vals = list(extra) + list(values)
    return min(vals) if vals else math.inf


def _js_max(values: Sequence[float], *extra: float) -> float:
    vals = list(extra) + list(values)
    return max(vals) if vals else -math.inf


def layout_pipeline(
    input: Mapping[str, Any],
    *,
    feedback: bool = True,
    max_label_chars: Optional[int] = None,
    **_ignored: Any,
) -> DiagramLayout:
    """Lay out a pipeline: layers by longest path, barycenter ordering, lanes, rails and feedback surfaces."""
    max_label = 22 if max_label_chars is None else max_label_chars
    nodes: list[dict[str, Any]] = [
        {**raw, "id": _js_str(_get(raw, "id")), "kind": _js_str(_get(raw, "kind"))} for raw in input["nodes"]
    ]
    known = {n["id"] for n in nodes}
    edges: list[dict[str, Any]] = [e for e in input["edges"] if e["from"] in known and e["to"] in known]
    in_order = input.get("order")
    order: list[str] = list(in_order) if in_order is not None and len(in_order) == len(nodes) else _topo_order(nodes, edges)

    # --- layers: longest path from a root
    layer_of: dict[str, int] = {}
    for nid in order:
        lv = 0
        for e in edges:
            if e["to"] == nid:
                lv = max(lv, layer_of.get(e["from"], 0) + 1)
        layer_of[nid] = lv
    layer_count = max([0, *layer_of.values()]) + 1
    layers: list[list[str]] = [[] for _ in range(layer_count)]
    for nid in order:
        layers[layer_of.get(nid, 0)].append(nid)

    # --- order inside a layer: barycenter sweeps, down then up
    pos: dict[str, int] = {}

    def reindex() -> None:
        for ids in layers:
            for i, nid in enumerate(ids):
                pos[nid] = i

    reindex()

    def bary(nid: str, direction: str) -> float:
        if direction == "in":
            ns = [pos.get(e["from"], 0) for e in edges if e["to"] == nid]
        else:
            ns = [pos.get(e["to"], 0) for e in edges if e["from"] == nid]
        if not ns:
            return pos.get(nid, 0)
        total: float = 0
        for v in ns:
            total += v
        return total / len(ns)

    for _sweep in range(4):
        for lv in range(1, len(layers)):
            keyed = [(bary(nid, "in"), i, nid) for i, nid in enumerate(layers[lv])]
            keyed.sort(key=lambda k: (k[0], k[1]))
            layers[lv] = [k[2] for k in keyed]
            reindex()
        for lv in range(len(layers) - 2, -1, -1):
            keyed = [(bary(nid, "out"), i, nid) for i, nid in enumerate(layers[lv])]
            keyed.sort(key=lambda k: (k[0], k[1]))
            layers[lv] = [k[2] for k in keyed]
            reindex()

    # --- sizes
    laid: dict[str, LaidOutNode] = {}
    for n in nodes:
        label = _cut(n["id"], max_label)
        kind_label = _cut(node_kind_label(n), max_label + 4)
        name = _cut(node_name(n), max_label + 4)
        visual = node_visual(n)
        text_w = max(_len16(label) * MONO_CH, _len16(name) * MONO_SM_CH, _len16(kind_label) * SANS_CH)
        w = max(NODE_MIN_W, min(NODE_MAX_W, math.ceil(text_w + 24 + (22 if visual == "gate" else 0))))
        laid[n["id"]] = LaidOutNode(
            id=n["id"], node=n, layer=layer_of.get(n["id"], 0), x=0, y=0, w=w, h=NODE_H,
            label=label, kind_label=kind_label, name=name, visual=visual,
        )

    # --- x: stack each layer, centre on the widest, then pull toward neighbours' median
    layer_w: list[float] = []
    for ids in layers:
        s: float = 0
        for nid in ids:
            s = s + (laid[nid].w if nid in laid else 0)
        layer_w.append(s + max(0, len(ids) - 1) * COL_GAP)
    widest = max([0, *layer_w])
    for lv, ids in enumerate(layers):
        x: float = (widest - layer_w[lv]) / 2
        for nid in ids:
            node = laid[nid]
            node.x = x
            x += node.w + COL_GAP

    def settle(ids: list[str], direction: str) -> None:
        want: list[float] = []
        for nid in ids:
            if direction == "in":
                ns = [laid[e["from"]] for e in edges if e["to"] == nid and e["from"] in laid]
            else:
                ns = [laid[e["to"]] for e in edges if e["from"] == nid and e["to"] in laid]
            cur = laid[nid]
            if not ns:
                want.append(cur.x)
                continue
            acc: float = 0
            for m in ns:
                acc = acc + m.x + m.w / 2
            want.append(acc / len(ns) - cur.w / 2)
        floor = -math.inf
        for i, nid in enumerate(ids):
            node = laid[nid]
            node.x = max(want[i], floor)
            floor = node.x + node.w + COL_GAP

    for lv in range(1, len(layers)):
        settle(layers[lv], "in")
    for lv in range(len(layers) - 2, -1, -1):
        settle(layers[lv], "out")
    for lv in range(1, len(layers)):
        settle(layers[lv], "in")

    # --- rails at the left, for edges that skip a layer. Consolidated: a node
    # fanning out owns a rail; nodes fanning in to one far node share one.
    def span(e: Mapping[str, Any]) -> int:
        return (laid[e["to"]].layer if e["to"] in laid else 0) - (laid[e["from"]].layer if e["from"] in laid else 0)

    long_out: dict[str, int] = {}
    long_in: dict[str, int] = {}
    for e in edges:
        if span(e) <= 1:
            continue
        long_out[e["from"]] = long_out.get(e["from"], 0) + 1
        long_in[e["to"]] = long_in.get(e["to"], 0) + 1
    rail_of: dict[int, str] = {}
    for i, e in enumerate(edges):
        if span(e) <= 1:
            continue
        if long_out.get(e["from"], 0) >= 2:
            key = f"src:{e['from']}"
        elif long_in.get(e["to"], 0) >= 2:
            key = f"dst:{e['to']}"
        else:
            key = f"src:{e['from']}"
        rail_of[i] = key
    rail_top: dict[str, float] = {}
    for i, e in enumerate(edges):
        top_key = rail_of.get(i)
        if top_key is None:
            continue
        rail_top[top_key] = min(rail_top.get(top_key, math.inf), laid[e["from"]].layer if e["from"] in laid else 0)
    # The rail that starts highest is outermost: its taps then cross no later rail.
    rail_keys = sorted(rail_top.keys(), key=lambda k: (rail_top[k], _locale_key(k)))
    rails: dict[str, float] = {}
    for i, key in enumerate(rail_keys):
        rails[key] = -(RAIL_STEP * (len(rail_keys) - i) + 10)

    # --- gaps between layers: a lane per trunk or rail that routes through the gap
    def lane_key_below(e: Mapping[str, Any], i: int) -> str:
        return rail_of.get(i, f"src:{e['from']}")

    gaps: list[GapLanes] = []
    lane_sources: list[list[str]] = [[] for _ in layers]
    for g in range(len(layers) - 1):
        direct: list[str] = []
        via_rail: list[str] = []
        for i, e in enumerate(edges):
            key = lane_key_below(e, i)
            a = laid[e["from"]]
            b = laid[e["to"]]
            if a.layer == g and key not in direct:
                direct.append(key)
            if i in rail_of and b.layer == g + 1 and a.layer != g and key not in via_rail:
                via_rail.append(key)

        def x_of(key: str) -> float:
            n = laid.get(key[4:])
            return n.x if n is not None else 0

        direct.sort(key=x_of)
        lane_sources[g] = [*direct, *(k for k in via_rail if k not in direct)]

    # --- y: layers stacked, each gap as tall as its lanes need
    min_x = _js_min([n.x for n in laid.values()], 0)
    rail_left = _js_min(list(rails.values()), 0)
    shift_x = PAD - rail_left - min_x
    y: float = PAD
    for lv, ids in enumerate(layers):
        for nid in ids:
            node = laid[nid]
            node.y = y
            node.x += shift_x
        y += NODE_H
        if lv < len(layers) - 1:
            sources = lane_sources[lv]
            lanes_h = max(1, len(sources)) * LANE_STEP
            top = y
            bottom = y + GAP_BASE + lanes_h
            lane: dict[str, float] = {}
            for i, sid in enumerate(sources):
                lane[sid] = top + (GAP_BASE - LANE_STEP) / 2 + LANE_STEP * (i + 1)
            gaps.append(GapLanes(top=top, bottom=bottom, lane=lane))
            y = bottom
    for key in list(rails):
        rails[key] = rails[key] + shift_x + min_x
    graph_left = _js_min([n.x for n in laid.values()])
    graph_right = _js_max([n.x + n.w for n in laid.values()])
    max_y = y

    # --- feedback: surfaces below, rails at the right
    sinks = [n["id"] for n in nodes if not any(e["from"] == n["id"] for e in edges)]
    surfaces: list[LaidOutSurface] = []
    feedback_rails: dict[str, float] = {}
    fb: list[dict[str, Any]] = [] if feedback is False else [f for f in (input.get("feedback") or []) if f["to"] in known]
    if fb:
        groups: dict[str, list[dict[str, Any]]] = {}
        for f in fb:
            groups.setdefault(f["from"], []).append(f)
        row_y = max_y + SURFACE_GAP
        sx = graph_left
        for src, fs in groups.items():
            channel, name = _parse_from(src)
            label = _cut(name, max_label + 6)
            w = math.ceil(_len16(channel) * 6.4 + _len16(label) * MONO_CH + 52)
            surfaces.append(LaidOutSurface(id=src, channel=channel, name=label, x=sx, y=row_y, w=w, h=SURFACE_H, edges=fs))
            sx += w + 24
        # Rails in order of how high they reach: the highest tap is the outermost, so nothing crosses it.
        by_reach = sorted(fb, key=lambda f: laid[f["to"]].y if f["to"] in laid else 0)
        for i, f in enumerate(by_reach):
            feedback_rails[f["id"]] = graph_right + 22 + RAIL_STEP * (len(by_reach) - i)
        max_y = row_y + SURFACE_H + 10 + 7 * len(surfaces)

    label_room: float = 10 + max(_len16(feedback_label(f)) for f in fb) * SANS_CH + 10 if fb else 0
    width = (
        _js_max(
            [*(s.x + s.w for s in surfaces), *(x + label_room for x in feedback_rails.values())],
            graph_right,
        )
        + PAD
    )
    return DiagramLayout(
        nodes=list(laid.values()),
        layers=layers,
        gaps=gaps,
        rails=rails,
        surfaces=surfaces,
        feedback_rails=feedback_rails,
        edges=edges,
        width=width,
        height=max_y + PAD,
        sinks=sinks,
        rail_of=rail_of,
        graph_left=graph_left,
        graph_right=graph_right,
    )


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# The README's palette: blue estimates, amber decides, violet writes, green saves.
_COLOR: dict[str, tuple[str, str]] = {
    "edge": ("--dp-edge", "#a8a29e"),
    "read": ("--dp-read", "#0e7490"),
    "decide": ("--dp-decide", "#2563eb"),
    "generate": ("--dp-generate", "#7c3aed"),
    "code": ("--dp-code", "#64748b"),
    "gate": ("--dp-gate", "#d97706"),
    "store": ("--dp-store", "#16a34a"),
    "feedback": ("--dp-feedback", "#e11d48"),
}


def _col(k: str) -> str:
    v, hexv = _COLOR[k]
    return f"var({v}, {hexv})"


_LEGEND: list[tuple[str, str]] = [
    ("read", "read"),
    ("decide", "decide — a model estimates"),
    ("generate", "generate — a model writes"),
    ("code", "code"),
    ("gate", "gate — code decides"),
    ("store", "store"),
    ("survive", "survives its own failure"),
    ("feedback", "human feedback, back to an earlier node"),
]


def _node_title(n: Mapping[str, Any]) -> str:
    parts = [f"{_js_str(_get(n, 'id'))} — {node_subtitle(n)}"]
    if _truthy(_get(n, "describe")):
        parts.append(_js_str(n["describe"]))
    if _truthy(_get(n, "route")):
        parts.append(f"route: {_js_str(n['route'])}")
    if _truthy(_get(n, "branches")):
        b = n["branches"]
        parts.append(f"branches: {' | '.join('' if x is None else _js_str(x) for x in b) if isinstance(b, (list, tuple)) else _js_str(b)}")
    t = _get(n, "thresholds")
    if _truthy(t) and isinstance(t, Mapping) and len(t) > 0:
        items = []
        for k in _js_keys(t):
            v = t[k] if k in t else t[int(k)]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                items.append(f"{k}={_num(v)}")
            elif isinstance(v, Mapping) and "default" in v:
                items.append(f"{k}={_js_str(v['default'])}")
            else:
                items.append(k)
        parts.append(f"thresholds: {', '.join(items)}")
    on_failure = _get(n, "onFailure")
    if _truthy(on_failure) and on_failure != "fail":
        parts.append(f"on failure: {_js_str(on_failure)}")
    if _truthy(_get(n, "version")):
        parts.append(f"v{_js_str(n['version'])}")
    return "\n".join(parts)


def _feedback_title(f: Mapping[str, Any]) -> str:
    lane = _lane_of(f)
    score = _get(f, "score")
    g = lambda k: _js_str(_get(f, k))  # noqa: E731
    parts = [
        f"{g('id')}: {g('from')} → {g('to')}",
        f"{g('source')} · {g('form')} · {g('scope')} · {g('latency')}",
        f"score: {_js_str(_get(score, 'name'))} ({_js_str(_get(score, 'dataType'))})" if _truthy(score) else "",
        f"review lane: {_js_str(lane)}" if _truthy(lane) else "",
    ]
    return "\n".join(p for p in parts if p)


def _ortho_path(pts: list[tuple[float, float]], r: float = 6) -> str:
    """An orthogonal polyline with rounded corners. Collinear points are dropped."""
    p: list[tuple[float, float]] = []
    for q in pts:
        last = p[-1] if p else None
        if last is not None and abs(last[0] - q[0]) < 0.01 and abs(last[1] - q[1]) < 0.01:
            continue
        prev = p[-2] if len(p) >= 2 else None
        if last is not None and prev is not None and (
            (prev[0] == last[0] and last[0] == q[0]) or (prev[1] == last[1] and last[1] == q[1])
        ):
            p.pop()
        p.append(q)
    if not p:
        return ""
    first = p[0]
    d = f"M{_num(first[0])},{_num(first[1])}"
    for i in range(1, len(p) - 1):
        ax, ay = p[i - 1]
        bx, by = p[i]
        cx, cy = p[i + 1]
        rr = min(r, _hypot(bx - ax, by - ay) / 2, _hypot(cx - bx, cy - by) / 2)
        in_x = _sign(bx - ax) * rr
        in_y = _sign(by - ay) * rr
        out_x = _sign(cx - bx) * rr
        out_y = _sign(cy - by) * rr
        d += f" L{_num(bx - in_x)},{_num(by - in_y)} Q{_num(bx)},{_num(by)} {_num(bx + out_x)},{_num(by + out_y)}"
    last = p[-1]
    d += f" L{_num(last[0])},{_num(last[1])}"
    return d


def _hexagon(x: float, y: float, w: float, h: float) -> str:
    k = min(12, h / 4)
    return (
        f"M{_num(x + k)},{_num(y)} H{_num(x + w - k)} L{_num(x + w)},{_num(y + h / 2)} "
        f"L{_num(x + w - k)},{_num(y + h)} H{_num(x + k)} L{_num(x)},{_num(y + h / 2)} Z"
    )


def _cylinder(x: float, y: float, w: float, h: float, c: str, dash: str) -> str:
    ry = 5
    return (
        f'<path d="M{_num(x)},{_num(y + ry)} A{_num(w / 2)},{ry} 0 0 1 {_num(x + w)},{_num(y + ry)} V{_num(y + h - ry)} '
        f'A{_num(w / 2)},{ry} 0 0 1 {_num(x)},{_num(y + h - ry)} Z" fill="{c}" fill-opacity="0.12" stroke="{c}" stroke-width="1.4"{dash}/>'
        f'<path d="M{_num(x)},{_num(y + ry)} A{_num(w / 2)},{ry} 0 0 0 {_num(x + w)},{_num(y + ry)}" fill="none" stroke="{c}" stroke-width="1.2"/>'
    )


def _person(cx: float, cy: float) -> str:
    """A small outline person — the one glyph, for a human surface."""
    return (
        f'<circle cx="{_num(cx)}" cy="{_num(cy - 3.5)}" r="2.6" fill="none" stroke="currentColor" stroke-width="1.3"/>'
        f'<path d="M{_num(cx - 5)},{_num(cy + 5.5)} a5,4.5 0 0 1 10,0" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
    )


def _uid_part(s: str) -> str:
    """`s.replace(/[^a-zA-Z0-9_-]/g, "_")`, one `_` per UTF-16 code unit like JS."""
    return "".join(c if re.fullmatch(r"[a-zA-Z0-9_-]", c) else "_" * _len16(c) for c in s)


def render_pipeline_svg(
    input: Mapping[str, Any],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    header: bool = True,
    legend: bool = True,
    feedback: bool = True,
    standalone: bool = False,
    max_label_chars: Optional[int] = None,
) -> str:
    """
    The pipeline as one SVG string.

    `title` defaults to the pipeline id; `subtitle` to fact · trigger · results
    (parts separated by `"   ·   "`). `standalone=True` embeds a light/dark
    `<style>` and a background so the file stands on its own; off, the SVG
    inherits `color` and the `--dp-*` custom properties from the page it is
    inlined in. `max_label_chars` (default 22) is the longest text on a node
    before it is cut with an ellipsis.
    """
    layout = layout_pipeline(input, feedback=feedback, max_label_chars=max_label_chars)
    by_id = {n.id: n for n in layout.nodes}
    pid = _js_str(input["id"])
    width = max(layout.width, 380)
    if header:
        if subtitle is not None:
            raw_parts = subtitle.split("   ·   ")
        else:
            trig = input.get("trigger")
            res = input.get("result")
            raw_parts = [
                f"fact: {_js_str(input['fact'])}" if _truthy(input.get("fact")) else "",
                f"trigger: {', '.join(_js_str(t) for t in trig)}" if trig else "",
                f"ends in: {' | '.join(_js_str(r) for r in res)}" if res else "",
                f"decided by: {_js_str(input['decidedBy'])}" if _truthy(input.get("decidedBy")) else "",
            ]
        subtitle_parts = [p for p in raw_parts if p]
    else:
        subtitle_parts = []
    subtitle_lines: list[str] = []
    for part in subtitle_parts:
        last = subtitle_lines[-1] if subtitle_lines else None
        joined = part if last is None else f"{last}   ·   {part}"
        if last is not None and _len16(joined) * SANS_CH * 1.05 <= width - 2 * PAD:
            subtitle_lines[-1] = joined
        else:
            subtitle_lines.append(part)
    header_h = 32 + len(subtitle_lines) * 14 if header else 0

    # legend rows, wrapped to the width
    legend_rows: list[list[tuple[str, str, float]]] = []
    if legend:
        row: list[tuple[str, str, float]] = []
        x: float = PAD
        for k, text in _LEGEND:
            w = 23 + _len16(text) * SANS_CH + 16
            if row and x + w > width - PAD:
                legend_rows.append(row)
                row = []
                x = PAD
            row.append((k, text, x))
            x += w
        if row:
            legend_rows.append(row)
    legend_h = 14 + len(legend_rows) * 18 if legend else 0
    height = layout.height + header_h + legend_h
    uid = f"dp-{_uid_part(pid)}"
    out: list[str] = []
    style_attr = ' style="color:#1c1917;background:#ffffff"' if standalone else ""

    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_num(width)} {_num(height)}" width="{_num(width)}" height="{_num(height)}" '
        f'class="dp-diagram" data-pipeline="{_esc(pid)}" role="img" aria-labelledby="{uid}-title" '
        f'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif"{style_attr}>'
    )
    shown_title = pid if title is None else title
    out.append(f'<title id="{uid}-title">{_esc(shown_title)} — decision pipeline</title>')
    if standalone:
        out.append(
            "<style>svg.dp-diagram{--dp-bg:#ffffff}@media (prefers-color-scheme: dark){svg.dp-diagram{color:#e7e5e4;background:#1c1917;--dp-bg:#1c1917}}</style>"
        )
    out.append("<defs>")
    for k in ("flow", "branch", "feedback"):
        stroke = _col("edge") if k == "flow" else _col("gate") if k == "branch" else _col("feedback")
        out.append(
            f'<marker id="{uid}-arrow-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M1,1 L9,5 L1,9" fill="none" stroke="{stroke}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></marker>'
        )
    out.append("</defs>")

    # --- header
    if header:
        out.append('<g class="dp-header">')
        out.append(f'<text x="{PAD}" y="22" font-size="15" font-weight="600" fill="currentColor">{_esc(shown_title)}</text>')
        for i, line in enumerate(subtitle_lines):
            out.append(f'<text x="{PAD}" y="{38 + i * 14}" font-size="10.5" fill="currentColor" opacity="0.62">{_esc(line)}</text>')
        out.append("</g>")

    out.append(f'<g transform="translate(0,{_num(header_h)})">')

    # --- flow edges: trunk → lane → (rail →) drop. Shared segments overlap into one line.
    def cx(n: LaidOutNode) -> float:
        return n.x + n.w / 2

    def gap_lane(g: int, key: str) -> Optional[float]:
        if 0 <= g < len(layout.gaps):
            return layout.gaps[g].lane.get(key)
        return None

    branch_into: dict[str, list[tuple[str, str]]] = {}
    out.append('<g class="dp-edges">')
    for i, e in enumerate(layout.edges):
        a = by_id.get(e["from"])
        b = by_id.get(e["to"])
        if a is None or b is None:
            continue
        branch = _opt(_get(e, "when"), "branch") if isinstance(_get(e, "when"), Mapping) else None
        has_branch = _truthy(branch)
        if has_branch:
            branch_into.setdefault(e["to"], []).append((_js_str(branch), e["from"]))
        rail_key = layout.rail_of.get(i)
        lane_key = rail_key if rail_key is not None else f"src:{e['from']}"
        pts: list[tuple[float, float]] = [(cx(a), a.y + a.h)]
        out_lane = gap_lane(a.layer, lane_key)
        if out_lane is None:
            out_lane = a.y + a.h + GAP_BASE / 2
        if rail_key is not None:
            rail = layout.rails.get(rail_key)
            if rail is None:
                rail = layout.graph_left - 16
            in_lane = gap_lane(b.layer - 1, rail_key)
            if in_lane is None:
                in_lane = b.y - GAP_BASE / 2
            pts += [(cx(a), out_lane), (rail, out_lane), (rail, in_lane), (cx(b), in_lane)]
        else:
            pts += [(cx(a), out_lane), (cx(b), out_lane)]
        pts.append((cx(b), b.y))
        stroke = _col("gate") if has_branch else _col("edge")
        tip = f"{e['from']} → {e['to']}" + (f" when {e['from']} = {_js_str(branch)}" if has_branch else "")
        branch_attr = f' data-branch="{_esc(_js_str(branch))}"' if has_branch else ""
        out.append(
            f'<path d="{_ortho_path(pts)}" fill="none" stroke="{stroke}" stroke-width="{1.7 if has_branch else 1.4}" stroke-linecap="round" '
            f'marker-end="url(#{uid}-arrow-{"branch" if has_branch else "flow"})" '
            f'class="dp-edge{" dp-edge-branch" if has_branch else ""}{" dp-edge-long" if rail_key is not None else ""}" '
            f'data-edge="{_esc(e["from"])}→{_esc(e["to"])}"{branch_attr}>'
            f"<title>{_esc(tip)}</title></path>"
        )
    out.append("</g>")

    # --- feedback: the loop
    labels: list[str] = []
    if layout.surfaces:
        out.append('<g class="dp-feedback">')
        row0 = layout.surfaces[0]
        # faint dotted: the run's end (its deepest sinks) down to the surface row — the output a person sees
        first_x = _js_min([s.x for s in layout.surfaces])
        last_x = _js_max([s.x + s.w for s in layout.surfaces])
        deepest = _js_max([by_id[sid].layer if sid in by_id else 0 for sid in layout.sinks])
        for sink_id in layout.sinks:
            sink = by_id.get(sink_id)
            if sink is None or sink.layer != deepest:
                continue
            x0 = cx(sink)
            drop_to = row0.y - 10
            toward = first_x if x0 < first_x else last_x if x0 > last_x else x0
            out.append(
                f'<path d="{_ortho_path([(x0, sink.y + sink.h), (x0, drop_to), (toward, drop_to), (toward, row0.y)])}" fill="none" '
                f'stroke="{_col("edge")}" stroke-width="1" stroke-dasharray="1.5 4" stroke-linecap="round" class="dp-edge-shown"/>'
            )
        outermost = _js_max(list(layout.feedback_rails.values()))
        per_target: dict[str, int] = {}
        for si, s in enumerate(layout.surfaces):
            for f in s.edges:
                t = by_id.get(f["to"])
                rail = layout.feedback_rails.get(f["id"])
                if t is None or rail is None:
                    continue
                slot = per_target.get(f["to"], 0)
                per_target[f["to"]] = slot + 1
                ty = t.y + t.h / 2 + slot * 9
                under = row0.y + row0.h + 12 + si * 7
                d = _ortho_path(
                    [(s.x + s.w / 2, s.y + s.h), (s.x + s.w / 2, under), (rail, under), (rail, ty), (t.x + t.w, ty)], 8
                )
                fid = _esc(_js_str(f["id"]))
                ftitle = _esc(_feedback_title(f))
                out.append(
                    f'<path d="{d}" fill="none" stroke="{_col("feedback")}" stroke-width="1.5" stroke-dasharray="5 4" stroke-linecap="round" '
                    f'marker-end="url(#{uid}-arrow-feedback)" class="dp-edge dp-edge-feedback" data-feedback="{fid}"><title>{ftitle}</title></path>'
                )
                # the label sits right of every rail, level with its tap — no line crosses it
                text = feedback_label(f)
                w = _len16(text) * SANS_CH + 10
                lx = outermost + 8
                ly = ty + slot * 9
                labels.append(
                    f'<g class="dp-feedback-label" data-feedback="{fid}"><rect x="{_num(lx)}" y="{_num(ly - 7.5)}" width="{_num(w)}" height="15" rx="4" '
                    f'fill="{_col("feedback")}" fill-opacity="0.1" stroke="{_col("feedback")}" stroke-opacity="0.5" stroke-width="0.8"/>'
                    f'<text x="{_num(lx + 5)}" y="{_num(ly + 3.5)}" font-size="9.5" fill="{_col("feedback")}">{_esc(text)}</text><title>{ftitle}</title></g>'
                )
        for s in layout.surfaces:
            chan_w = _len16(s.channel) * 6.4 + 6
            edge_titles = "\n\n".join(_feedback_title(f) for f in s.edges)
            out.append(
                f'<g class="dp-surface" data-surface="{_esc(s.id)}"><rect x="{_num(s.x)}" y="{_num(s.y)}" width="{_num(s.w)}" height="{_num(s.h)}" rx="15" '
                f'fill="{_col("feedback")}" fill-opacity="0.1" stroke="{_col("feedback")}" stroke-width="1.4"/>{_person(s.x + 16, s.y + s.h / 2)}'
                f'<text x="{_num(s.x + 28)}" y="{_num(s.y + s.h / 2 + 3.5)}" font-size="8.5" letter-spacing="0.06em" fill="{_col("feedback")}">{_esc(s.channel.upper())}</text>'
                f'<text x="{_num(s.x + 32 + chan_w)}" y="{_num(s.y + s.h / 2 + 4)}" font-size="11.5" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" fill="currentColor">{_esc(s.name)}</text>'
                f"<title>{_esc(s.channel + ': ' + s.name)}\n{_esc(edge_titles)}</title></g>"
            )
        out.append("</g>")

    # --- nodes
    out.append('<g class="dp-nodes">')
    for n in layout.nodes:
        c = _col(n.visual)
        on_failure = _get(n.node, "onFailure")
        survivable = _truthy(on_failure) and on_failure != "fail"
        dash = ' stroke-dasharray="4 3"' if survivable else ""
        if n.visual == "gate":
            shape = f'<path d="{_hexagon(n.x, n.y, n.w, n.h)}" fill="{c}" fill-opacity="0.13" stroke="{c}" stroke-width="1.6"{dash}/>'
        elif n.visual == "store":
            shape = _cylinder(n.x, n.y, n.w, n.h, c, dash)
        else:
            shape = (
                f'<rect x="{_num(n.x)}" y="{_num(n.y)}" width="{_num(n.w)}" height="{_num(n.h)}" rx="8" fill="{c}" fill-opacity="0.12" '
                f'stroke="{c}" stroke-width="1.4"{dash}/>'
            )
        bar = (
            ""
            if n.visual in ("gate", "store")
            else f'<rect x="{_num(n.x + 8)}" y="{_num(n.y + 8)}" width="{_num(n.w - 16)}" height="3" rx="1.5" fill="{c}"/>'
        )
        tx = n.x + n.w / 2
        base_y = n.y + 6 if n.visual == "store" else n.y
        role = _get(n.node, "role")
        role_attr = f' data-role="{_esc(_js_str(role))}"' if _truthy(role) else ""
        mono = 'font-family="ui-monospace, SFMono-Regular, Menlo, monospace"'
        out.append(
            f'<g class="dp-node dp-{n.visual}" data-node="{_esc(n.id)}" data-kind="{_esc(_js_str(n.node["kind"]))}"'
            f"{role_attr}>"
            f"<title>{_esc(_node_title(n.node))}</title>"
            + shape
            + bar
            + f'<text x="{_num(tx)}" y="{_num(base_y + 27)}" font-size="12" font-weight="600" text-anchor="middle" {mono} fill="currentColor">{_esc(n.label)}</text>'
            + f'<text x="{_num(tx)}" y="{_num(base_y + 40)}" font-size="9.5" text-anchor="middle" fill="{c}" letter-spacing="0.02em">{_esc(n.kind_label)}</text>'
            + (
                f'<text x="{_num(tx)}" y="{_num(base_y + 52)}" font-size="9.5" text-anchor="middle" {mono} fill="currentColor" opacity="0.7">{_esc(n.name)}</text>'
                if n.name
                else ""
            )
            + "</g>"
        )
        # the branch that reaches this node, on its doorstep
        reached = branch_into.get(n.id, [])
        for i, (br, gate) in enumerate(reached):
            text = _cut(br, 16)
            w = _len16(text) * SANS_CH + 12
            px = tx + (i - (len(reached) - 1) / 2) * (w + 4)
            labels.append(
                f'<g class="dp-branch" data-branch="{_esc(br)}" data-gate="{_esc(gate)}"><rect x="{_num(px - w / 2)}" y="{_num(n.y - 19)}" width="{_num(w)}" height="16" rx="8" fill="{_col("gate")}"/>'
                f'<text x="{_num(px)}" y="{_num(n.y - 7.5)}" font-size="10" font-weight="600" text-anchor="middle" fill="#ffffff">{_esc(text)}</text>'
                f"<title>{_esc(f'{gate} = {br}')}</title></g>"
            )
    out.append("</g>")
    if labels:
        out.append(f'<g class="dp-labels">{"".join(labels)}</g>')
    out.append("</g>")

    # --- legend
    if legend:
        out.append('<g class="dp-legend" font-size="9.5">')
        for r, row in enumerate(legend_rows):
            ly = height - legend_h + 14 + r * 18
            for k, text, x in row:
                if k == "survive":
                    out.append(
                        f'<rect x="{_num(x)}" y="{_num(ly - 6)}" width="18" height="12" rx="3" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.7"/>'
                    )
                elif k == "feedback":
                    out.append(
                        f'<path d="M{_num(x)},{_num(ly)} h18" stroke="{_col("feedback")}" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#{uid}-arrow-feedback)"/>'
                    )
                elif k == "gate":
                    out.append(
                        f'<path d="{_hexagon(x, ly - 6, 18, 12)}" fill="{_col("gate")}" fill-opacity="0.13" stroke="{_col("gate")}" stroke-width="1.2"/>'
                    )
                else:
                    c = _col(k)
                    out.append(
                        f'<rect x="{_num(x)}" y="{_num(ly - 6)}" width="18" height="12" rx="{6 if k == "store" else 3}" fill="{c}" fill-opacity="0.12" stroke="{c}" stroke-width="1.2"/>'
                    )
                out.append(f'<text x="{_num(x + 23)}" y="{_num(ly + 3.5)}" fill="currentColor" opacity="0.8">{_esc(text)}</text>')
        out.append("</g>")

    out.append("</svg>")
    return "\n".join(out)
