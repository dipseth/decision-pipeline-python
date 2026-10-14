from decision_pipeline.graph import resolve_inbound, sink_nodes, topological_order
from decision_pipeline.types import Edge, EdgeGuard

edges = [
    Edge(from_="classify", to="table"),
    Edge(from_="table", to="gate"),
    Edge(from_="gate", to="direct", when=EdgeGuard(gate="gate", branch="direct")),
    Edge(from_="gate", to="writer", when=EdgeGuard(gate="gate", branch="writer")),
    Edge(from_="direct", to="changes"),
    Edge(from_="writer", to="changes"),
]
ids = ["classify", "table", "gate", "direct", "writer", "changes"]


def test_orders_parents_before_children():
    topo = topological_order(ids, edges)
    order, cycle = topo.order, topo.cycle
    assert cycle == []
    assert order.index("classify") < order.index("table")
    assert order.index("gate") < order.index("direct")
    assert order.index("writer") < order.index("changes")


def test_tie_breaks_by_declaration_order_so_a_runs_node_order_is_stable():
    order = topological_order(ids, edges).order
    assert order.index("direct") < order.index("writer")
    assert topological_order(ids, edges).order == order


def test_reports_the_nodes_in_a_cycle_instead_of_hanging():
    topo = topological_order(
        ["a", "b", "c"],
        [Edge(from_="a", to="b"), Edge(from_="b", to="c"), Edge(from_="c", to="a")],
    )
    assert topo.order == []
    assert sorted(topo.cycle) == ["a", "b", "c"]


ran = {"classify", "table", "gate"}


def test_runs_a_root_node_unconditionally():
    assert resolve_inbound(edges, "classify", set(), lambda _g: None).skip_reason is None


def test_opens_the_edge_whose_guard_matches_the_branch_taken():
    decision = resolve_inbound(edges, "direct", ran, lambda _g: "direct")
    assert decision.skip_reason is None
    assert [e.from_ for e in decision.open] == ["gate"]


def test_skips_the_other_branch_with_the_reason_never_silently():
    decision = resolve_inbound(edges, "writer", ran, lambda _g: "direct")
    assert decision.open == []
    assert 'gate took "direct", needed "writer"' in decision.skip_reason


def test_skips_downstream_of_a_skipped_node_and_says_which_one():
    decision = resolve_inbound(edges, "changes", {"direct"}, lambda _g: "direct")
    assert [e.from_ for e in decision.open] == ["direct"]

    orphaned = resolve_inbound(edges, "changes", set(), lambda _g: "direct")
    assert "direct skipped" in orphaned.skip_reason
    assert "writer skipped" in orphaned.skip_reason


def test_runs_a_join_node_when_any_inbound_edge_is_open():
    decision = resolve_inbound(edges, "changes", {"writer"}, lambda _g: "writer")
    assert decision.skip_reason is None


def test_finds_the_nodes_nothing_flows_out_of():
    assert sink_nodes(ids, edges) == ["changes"]
