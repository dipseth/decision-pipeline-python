from decision_pipeline.experiments import GuardPatch, ManifestPatch
from decision_pipeline.patch import apply_manifest_patch
from decision_pipeline.types import (
    DecideNode,
    Edge,
    EdgeGuard,
    GateNode,
    GenerateNode,
    is_gate_node,
)


def nodes():
    return {
        "classify": DecideNode(questions="jev-v1", version="1"),
        "gate": GateNode(run=lambda _a: "a", branches=["a", "b"], thresholds={"category": 0.7}, version="1"),
        "writer": GenerateNode(prompt="p1", route="r1", version="1"),
    }


def edges():
    return [
        Edge(from_="classify", to="gate"),
        Edge(from_="gate", to="writer", when=EdgeGuard(gate="gate", branch="b")),
    ]


def test_returns_a_copy_when_there_is_nothing_to_apply():
    original = nodes()
    result = apply_manifest_patch(original, edges(), None)
    assert result.applied == []
    assert result.nodes is not original
    assert apply_manifest_patch(original, edges(), ManifestPatch()).applied == []


def test_never_mutates_the_manifest_it_patches():
    original = nodes()
    apply_manifest_patch(original, edges(), ManifestPatch(thresholds={"gate": {"category": 0.6}}))
    gate = original["gate"]
    assert gate.thresholds["category"] == 0.7 if is_gate_node(gate) else False


def test_overrides_a_gate_threshold_and_reports_the_diff():
    result = apply_manifest_patch(nodes(), edges(), ManifestPatch(thresholds={"gate": {"category": 0.6}}))
    gate = result.nodes["gate"]
    assert gate.thresholds["category"] == 0.6 if is_gate_node(gate) else False
    assert result.applied == ["gate.thresholds.category=0.6"]


def test_formats_whole_number_thresholds_like_javascript():
    result = apply_manifest_patch(nodes(), edges(), ManifestPatch(thresholds={"gate": {"category": 1.0}}))
    assert result.applied == ["gate.thresholds.category=1"]


def test_swaps_a_prompt_a_route_and_a_jev_question_name():
    result = apply_manifest_patch(
        nodes(),
        edges(),
        ManifestPatch(
            prompts={"writer": "p2"},
            routes={"writer": "r2"},
            questions={"classify": "jev-v2"},
        ),
    )
    writer = result.nodes["writer"]
    classify = result.nodes["classify"]
    assert writer.prompt == "p2"
    assert writer.route == "r2"
    assert classify.questions == "jev-v2"


def test_rewrites_and_removes_an_edge_guard():
    rewritten = apply_manifest_patch(
        nodes(),
        edges(),
        ManifestPatch(guards=[GuardPatch(from_="gate", to="writer", when=EdgeGuard(gate="gate", branch="a"))]),
    )
    assert rewritten.edges[1].when == EdgeGuard(gate="gate", branch="a")

    removed = apply_manifest_patch(
        nodes(),
        edges(),
        ManifestPatch(guards=[GuardPatch(from_="gate", to="writer", when=None)]),
    )
    assert removed.edges[1].when is None
    assert removed.applied == ["gate->writer.when=none"]


def test_ignores_a_patch_aimed_at_the_wrong_node_kind_rather_than_corrupting_it():
    result = apply_manifest_patch(nodes(), edges(), ManifestPatch(prompts={"gate": "p2"}))
    assert result.applied == []
