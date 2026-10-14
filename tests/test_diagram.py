import re

from decision_pipeline.diagram import layout_pipeline, node_subtitle, render_pipeline_svg

# The shape of Scale (#317): decide → transform → gate ⑂2 → gate ⑂2 → writers → guard → derive → store, two feedback edges.
scale = {
    "id": "scale",
    "fact": "scaled_recipe",
    "trigger": ["on-demand"],
    "result": ["changes"],
    "decidedBy": "llm",
    "nodes": [
        {"id": "classify", "kind": "decide", "questions": "jev-scale-classify", "onFailure": "fallback", "version": "3"},
        {"id": "table", "kind": "code", "role": "transform", "version": "2"},
        {"id": "gate", "kind": "code", "role": "gate", "branches": ["direct", "writer"], "thresholds": {"direct_line": {"env": "X", "default": 0.8}}, "version": "1"},
        {"id": "direct", "kind": "code", "role": "derive", "version": "1"},
        {"id": "agent", "kind": "code", "role": "gate", "branches": ["quick", "full"], "thresholds": {}, "version": "1"},
        {"id": "writer_quick", "kind": "generate", "prompt": "recipe-scale-quick", "route": "scale-writer-quick", "version": "7"},
        {"id": "writer_full", "kind": "generate", "prompt": "recipe-scale", "route": "scale-writer", "version": "7"},
        {"id": "guard", "kind": "code", "role": "validate", "version": "1"},
        {"id": "changes", "kind": "code", "role": "derive", "version": "2"},
        {"id": "persist", "kind": "store", "target": "turso", "scope": "tenant", "version": "1"},
    ],
    "edges": [
        {"from": "classify", "to": "table"},
        {"from": "table", "to": "gate"},
        {"from": "gate", "to": "direct", "when": {"gate": "gate", "branch": "direct"}},
        {"from": "gate", "to": "agent", "when": {"gate": "gate", "branch": "writer"}},
        {"from": "agent", "to": "writer_quick", "when": {"gate": "agent", "branch": "quick"}},
        {"from": "agent", "to": "writer_full", "when": {"gate": "agent", "branch": "full"}},
        {"from": "writer_quick", "to": "guard"},
        {"from": "writer_full", "to": "guard"},
        {"from": "direct", "to": "changes"},
        {"from": "guard", "to": "changes"},
        {"from": "changes", "to": "persist"},
    ],
    "feedback": [
        {"id": "rating", "from": "surface:scaling-notes", "to": "gate", "source": "explicit", "scope": "user", "form": "score", "latency": "deferred", "score": {"name": "scale_feedback", "dataType": "NUMERIC"}},
        {"id": "flagged-lines", "from": "surface:scaling-notes", "to": "classify", "source": "explicit", "scope": "user", "form": "queue", "latency": "deferred", "promoteVia": "scale-line-review"},
    ],
}


# --- layout_pipeline


def test_layers_by_longest_path_and_keeps_every_node_inside_the_canvas():
    l = layout_pipeline(scale)
    at = lambda nid: next(n for n in l.nodes if n.id == nid)  # noqa: E731
    assert at("classify").layer == 0
    assert at("gate").layer == 2
    assert at("direct").layer == 3
    assert at("agent").layer == 3
    assert at("guard").layer == 5
    assert at("changes").layer == 6
    assert at("persist").layer == 7
    for n in l.nodes:
        assert n.x >= 0
        assert n.y >= 0
        assert n.x + n.w <= l.width
        assert n.y + n.h <= l.height
        assert abs(n.x) < float("inf") and abs(n.y) < float("inf")
    assert l.sinks == ["persist"]


def test_never_overlaps_two_nodes_in_one_layer():
    l = layout_pipeline(scale)
    for a in l.nodes:
        for b in l.nodes:
            if a is b or a.layer != b.layer:
                continue
            apart = a.x + a.w <= b.x or b.x + b.w <= a.x
            assert apart, f"{a.id} overlaps {b.id}"


def test_groups_feedback_edges_by_surface_below_the_graph():
    l = layout_pipeline(scale)
    assert len(l.surfaces) == 1
    s = l.surfaces[0]
    assert s.channel == "surface"
    assert s.name == "scaling-notes"
    assert [f["to"] for f in s.edges] == ["gate", "classify"]
    assert s.y > max(n.y + n.h for n in l.nodes)
    assert len(layout_pipeline(scale, feedback=False).surfaces) == 0


def test_works_without_order_and_drops_an_edge_to_an_unknown_node():
    l = layout_pipeline({**scale, "edges": [*scale["edges"], {"from": "persist", "to": "ghost"}]})
    assert len(l.edges) == len(scale["edges"])
    assert next(n for n in l.nodes if n.id == "persist").layer == 7


def test_runs_an_edge_that_skips_a_layer_down_a_rail_at_the_left_one_rail_per_fan():
    fan_out = layout_pipeline(
        {
            "id": "chain",
            "nodes": [{"id": i, "kind": "code", "role": "transform"} for i in ["a", "b", "c", "d", "e"]],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
                {"from": "c", "to": "d"},
                {"from": "d", "to": "e"},
                {"from": "a", "to": "d"},
                {"from": "a", "to": "e"},
                {"from": "b", "to": "e"},
            ],
        }
    )
    # a fans out (a→d, a→e) and owns a rail; b's one long edge joins e's fan-in rail
    assert sorted(fan_out.rails.keys()) == ["dst:e", "src:a"]
    for x in fan_out.rails.values():
        assert x < fan_out.graph_left
    assert all(n.y >= 20 for n in fan_out.nodes)

    fan_in = layout_pipeline(
        {
            "id": "fan-in",
            "nodes": [{"id": i, "kind": "code", "role": "transform"} for i in ["a", "b", "c", "z"]],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
                {"from": "c", "to": "z"},
                {"from": "a", "to": "z"},
                {"from": "b", "to": "z"},
            ],
        }
    )
    # a and b each have ONE long edge, both into z: they share z's rail
    assert list(fan_in.rails.keys()) == ["dst:z"]


def test_wraps_a_long_header_and_the_legend_for_a_narrow_graph():
    narrow = {
        "id": "narrow",
        "fact": "a_fact",
        "trigger": ["ingest", "cron", "on-demand"],
        "result": ["one", "two", "three", "four"],
        "decidedBy": "llm",
        "nodes": [{"id": "a", "kind": "read"}, {"id": "b", "kind": "store", "target": "turso", "scope": "tenant"}],
        "edges": [{"from": "a", "to": "b"}],
    }
    svg = render_pipeline_svg(narrow)
    header_start = svg.index('class="dp-header"')
    header = svg[header_start : svg.index("</g>", header_start)]
    assert len(re.findall(r"<text", header)) > 2
    legend = svg[svg.index('class="dp-legend"') :]
    ys = {re.sub(r".*y=", "", m) for m in re.findall(r'<text x="[\d.]+" y="[\d.]+"', legend)}
    assert len(ys) > 1


# --- node_subtitle


def test_names_the_one_thing_that_identifies_a_node():
    assert node_subtitle({"id": "x", "kind": "decide", "questions": "jev-q"}) == "decide · jev-q"
    assert node_subtitle({"id": "x", "kind": "decide", "probes": {"from": "w"}}) == "decide · probes ← w"
    assert node_subtitle({"id": "x", "kind": "generate", "prompt": "p"}) == "generate · p"
    assert node_subtitle({"id": "x", "kind": "code", "role": "gate", "thresholds": {"a": 1, "b": 2}}) == "gate · a, b"
    assert node_subtitle({"id": "x", "kind": "code", "role": "transform", "thresholds": {"a": 1}}) == "transform · 1 threshold"
    assert node_subtitle({"id": "x", "kind": "store", "target": "turso", "scope": "tenant"}) == "store · turso · tenant"
    assert node_subtitle({"id": "x", "kind": "read", "providesFeatures": True}) == "read · features"


# --- render_pipeline_svg

svg = render_pipeline_svg(scale, title="Scale")


def test_is_one_svg_with_every_node_edge_branch_and_feedback_edge_addressable():
    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>")
    assert not re.search(r"NaN|undefined", svg)
    for n in scale["nodes"]:
        assert f'data-node="{n["id"]}"' in svg
    for e in scale["edges"]:
        assert f'data-edge="{e["from"]}→{e["to"]}"' in svg
    for b in ["direct", "writer", "quick", "full"]:
        assert f'data-branch="{b}"' in svg
    for f in scale.get("feedback") or []:
        assert f'data-feedback="{f["id"]}"' in svg
    assert 'data-surface="surface:scaling-notes"' in svg
    assert re.search(r'<title id="[^"]+">Scale — decision pipeline</title>', svg)


def test_draws_a_gate_as_the_hexagon_and_a_survivable_node_dashed():
    assert re.search(r'data-node="gate"[^>]*data-role="gate">.*?<path d="M', svg, re.S)
    assert re.search(r'data-node="classify"[^>]*>.*?stroke-dasharray="4 3"', svg, re.S)
    assert not re.search(r'data-node="table"[^>]*>[^/]*?stroke-dasharray', svg, re.S)


def test_escapes_text_and_puts_the_details_in_a_tooltip():
    out = render_pipeline_svg({**scale, "nodes": [{"id": "a<b", "kind": "read", "describe": 'reads "x" & y'}], "edges": [], "feedback": []})
    assert "a&lt;b" in out
    assert "reads &quot;x&quot; &amp; y" in out
    assert "a<b" not in out


def test_is_theme_aware_inline_and_self_contained_when_standalone():
    assert "var(--dp-gate, #d97706)" in svg
    assert 'fill="currentColor"' in svg
    assert "<style>" not in svg
    alone = render_pipeline_svg(scale, standalone=True)
    assert "prefers-color-scheme: dark" in alone
    assert 'style="color:#1c1917;background:#ffffff"' in alone


def test_can_leave_the_header_and_legend_out():
    bare = render_pipeline_svg(scale, header=False, legend=False)
    assert 'class="dp-header"' not in bare
    assert 'class="dp-legend"' not in bare
    assert len(bare) < len(svg)
