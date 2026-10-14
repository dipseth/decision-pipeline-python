# Ported from src/bindings.test.ts: the pure parts (parse_input_ref, binding_problems,
# binding_sources, resolve_inputs), then the "inputs in a manifest" half (define/run).

import re

import pytest
from pydantic import BaseModel

from decision_pipeline import define_pipeline, run_pipeline, validate_pipeline
from decision_pipeline import testing as t
from decision_pipeline.bindings import (
    RefLiteral,
    RefSource,
    binding_problems,
    binding_sources,
    parse_input_ref,
    resolve_inputs,
)


def test_parses_node_sub_path_index_input_and_alternatives():
    assert parse_input_ref("evidence") == [RefSource(source="evidence", path=[])]
    assert parse_input_ref("evidence.row.top[2].slug") == [
        RefSource(source="evidence", path=["row", "top", 2, "slug"])
    ]
    assert parse_input_ref("$input.factor") == [RefSource(source="$input", path=["factor"])]
    assert parse_input_ref("maps | retry_maps.draft") == [
        RefSource(source="maps", path=[]),
        RefSource(source="retry_maps", path=["draft"]),
    ]


def test_takes_a_json_literal_as_the_last_alternative_only():
    assert parse_input_ref("decide.row | null") == [RefSource(source="decide", path=["row"]), RefLiteral(literal=None)]
    assert parse_input_ref('a | b | "none"') == [
        RefSource(source="a", path=[]),
        RefSource(source="b", path=[]),
        RefLiteral(literal="none"),
    ]
    assert parse_input_ref("x | -0.5") == [RefSource(source="x", path=[]), RefLiteral(literal=-0.5)]
    assert "only its default" in parse_input_ref("0")
    assert "must be the last" in parse_input_ref("a | 0 | b")


@pytest.mark.parametrize("bad", ["", "a..b", "a[x]", "$env.KEY", "a.b()", "1abc", "a | "])
def test_refuses_anything_that_is_not_a_plain_path(bad):
    assert isinstance(parse_input_ref(bad), str)


def test_allows_only_direct_inbound_nodes_and_input():
    inbound = {"evidence"}
    assert binding_problems("n", {"row": "evidence.row", "f": "$input.factor"}, inbound) == []
    problems = binding_problems(
        "n", {"row": "taxonomy.rows", "bad-name": "evidence", "x": "evidence..y"}, inbound
    )
    assert len(problems) == 3
    joined = "\n".join(problems)
    assert "no edge taxonomy -> n" in joined
    assert '"bad-name" is not an identifier' in joined


def test_lists_the_nodes_a_binding_set_reads():
    assert sorted(binding_sources({"a": "x.y", "b": "$input", "c": "y | z"})) == ["x", "y", "z"]


FROM = {"evidence": {"row": None, "cuisines": [{"slug": "thai"}]}, "maps": {"n": 1}}


def test_binds_null_as_a_value_undefined_falls_through_a_skipped_node_is_absent():
    assert resolve_inputs({"row": "evidence.row"}, FROM, {}) == {"row": None}
    assert resolve_inputs({"first": "evidence.cuisines[0].slug"}, FROM, {}) == {"first": "thai"}
    assert resolve_inputs({"d": "retry_maps | maps.n"}, FROM, {}) == {"d": 1}
    assert resolve_inputs({"d": "evidence.missing | maps"}, FROM, {}) == {"d": {"n": 1}}
    assert resolve_inputs({"d": "retry_maps"}, FROM, {}) == {"d": None}
    assert resolve_inputs({"f": "$input.factor"}, FROM, {"factor": 2}) == {"f": 2}
    assert resolve_inputs(None, FROM, {}) == {}
    assert resolve_inputs({"s": "decide.jevShare | 0", "r": "evidence.row | 1"}, FROM, {}) == {"s": 0, "r": None}


def test_walks_attributes_of_a_pydantic_input():
    from pydantic import BaseModel

    class In(BaseModel):
        id: str
        factor: int

    assert resolve_inputs({"f": "$input.factor", "m": "$input.nope | 7"}, {}, In(id="a", factor=2)) == {
        "f": 2,
        "m": 7,
    }


# ---------------------------------------------------------------------------
# In a manifest and a run (describe "inputs in a manifest")
# ---------------------------------------------------------------------------


class _BindIn(BaseModel):
    id: str
    factor: float


class _BindOut(BaseModel):
    text: str


def _spec(inputs):
    return {
        "id": "bind-demo",
        "fact": "demo",
        "input": _BindIn,
        "output": _BindOut,
        "trigger": ["on-demand"],
        "group": lambda i: f"demo-{i.id}",
        "result": ["writer", "plain"],
        "nodes": {
            "evidence": {"kind": "read", "load": lambda _a: {"row": {"p": 0.8}, "label": "from evidence"}, "version": "1"},
            "gate": {
                "kind": "code",
                "role": "gate",
                "inputs": inputs,
                "run": lambda args: "writer" if args.in_["p"] >= args.thresholds["p"] else "plain",
                "branches": ["writer", "plain"],
                "thresholds": {"p": 0.5},
                "version": "1",
            },
            "writer": {
                "kind": "generate",
                "prompt": "demo-writer",
                "route": "writer",
                "inputs": {"label": "gate.label", "factor": "$input.factor"},
                "version": "1",
            },
            "plain": {"kind": "code", "role": "derive", "run": lambda _a: {"text": "plain"}, "version": "1"},
            "persist": {"kind": "store", "target": "memory", "scope": "user", "version": "1"},
        },
        "edges": [
            {"from": "evidence", "to": "gate"},
            {"from": "gate", "to": "writer", "when": {"gate": "gate", "branch": "writer"}},
            {"from": "gate", "to": "plain", "when": {"gate": "gate", "branch": "plain"}},
            {"from": "writer", "to": "persist"},
            {"from": "plain", "to": "persist"},
        ],
        "feedback": [
            {
                "id": "rating",
                "from": "surface:demo",
                "to": "gate",
                "source": "explicit",
                "scope": "user",
                "form": "score",
                "latency": "deferred",
                "score": {"name": "demo_feedback", "dataType": "NUMERIC"},
            }
        ],
        "eval": {"dataset": "demo-eval", "gateMetric": "score_positive"},
    }


def test_a_binding_to_a_node_with_no_edge_into_this_one_does_not_load():
    problems = validate_pipeline(_spec({"p": "persist.p"}))
    assert re.search(r'node "gate": input "p" reads "persist", which has no edge persist -> gate', "\n".join(problems))


def test_round_trips_through_to_json():
    nodes = define_pipeline(_spec({"p": "evidence.row.p"})).to_json()["nodes"]
    assert next(n for n in nodes if n["id"] == "gate")["inputs"] == {"p": "evidence.row.p"}


async def test_bodies_read_args_in_and_ports_receive_request_in():
    pipeline = define_pipeline(_spec({"p": "evidence.row.p"}))
    seen = {}

    def writer(req):
        seen["in"] = req.in_
        return {"value": {"text": "written"}}

    ports = t.test_ports(generate={"writer": writer})
    result = await run_pipeline(pipeline, {"id": "r1", "factor": 2}, ports)
    assert result.output.text == "written"
    assert next(n for n in result.record.nodes if n.id == "gate").branch == "writer"
    # The gate passes its primary through, so `gate.label` is the evidence's label.
    assert dict(seen["in"]) == {"label": "from evidence", "factor": 2}
