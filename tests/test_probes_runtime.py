# Needs the core: define.py, run.py, testing.py. Ported from the pipeline tests in src/probes.test.ts.
import dataclasses

import pytest
from pydantic import BaseModel

from decision_pipeline import testing as dp_testing  # not imported by name: pytest would collect `test_ports`
from decision_pipeline.define import EvalSpec, PipelineSpec, define_pipeline, validate_pipeline
from decision_pipeline.errors import PipelineContractError
from decision_pipeline.ports import DecideResult, GenerateResult
from decision_pipeline.run import run_pipeline
from decision_pipeline.types import (
    CodeNode,
    DecideNode,
    Edge,
    EdgeGuard,
    FeedbackEdge,
    GateNode,
    GenerateNode,
    ProbeSpec,
    ReadNode,
    ScoreConfigSpec,
    StoreNode,
)


class In(BaseModel):
    id: str
    title: str


class Out(BaseModel):
    text: str


def spec(seen: dict | None = None) -> PipelineSpec:
    seen = {} if seen is None else seen

    def interpret(d, _args):
        seen["interpret"] = d
        return {"dish": (d.get("dish") or [0])[0]}

    def gate_run(args):
        seen["gate"] = args.probes
        return "writer" if args.primary["dish"] >= 0.5 else "plain"

    return PipelineSpec(
        id="probe-demo",
        fact="demo",
        input=In,
        output=Out,
        trigger=["on-demand"],
        group=lambda i: f"demo-{i.id}",
        result=["writer", "plain"],
        nodes={
            "evidence": ReadNode(load=lambda a: {"recipe": {"title": a.input.title}}, version="1"),
            "probe_gen": GenerateNode(prompt="demo-probes", route="cheap", on_failure="skip", version="1"),
            "classify": DecideNode(
                questions="jev-demo",
                probes=ProbeSpec(from_="probe_gen", max=2, paths=["recipe"]),
                interpret=interpret,
                version="1",
            ),
            "gate": GateNode(run=gate_run, branches=["writer", "plain"], thresholds={"dish": 0.5}, version="1"),
            "writer": GenerateNode(prompt="demo-writer", route="writer", version="1"),
            "plain": CodeNode(role="derive", run=lambda _a: {"text": "plain"}, version="1"),
            "persist": StoreNode(target="memory", scope="user", version="1"),
        },
        edges=[
            Edge(from_="evidence", to="probe_gen"),
            Edge(from_="evidence", to="classify"),
            Edge(from_="probe_gen", to="classify"),
            Edge(from_="classify", to="gate"),
            Edge(from_="gate", to="writer", when=EdgeGuard(gate="gate", branch="writer")),
            Edge(from_="gate", to="plain", when=EdgeGuard(gate="gate", branch="plain")),
            Edge(from_="writer", to="persist"),
            Edge(from_="plain", to="persist"),
        ],
        feedback=[
            FeedbackEdge(
                id="rating",
                from_="surface:demo",
                to="gate",
                source="explicit",
                scope="user",
                form="score",
                latency="deferred",
                score=ScoreConfigSpec(name="demo_feedback", data_type="NUMERIC"),
            )
        ],
        eval=EvalSpec(dataset="demo-eval", gate_metric="score_positive"),
    )


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


PROBES = [
    noul("Is `recipe.title` a fusion dish?"),
    noul("Is the moon full?"),
    {"type": "choice", "instructions": "Which tradition does `recipe.title` lean to?", "criteria": {"peru": "Peru", "china": "China"}},
]


# --- run_pipeline with probes ---


async def test_asks_validated_probes_in_the_same_decide_call_and_records_their_text():
    seen: dict = {}
    pipeline = define_pipeline(spec(seen))
    writer_probes = {}

    def writer(req):
        writer_probes["v"] = req.probes.get("classify")
        return GenerateResult(value={"text": "written"}, cost_usd=0.01)

    def classify(req):
        dists = {"dish": [0.8]}
        for i, k in enumerate(req.probes.keys()):
            dists[k] = [0.9] if i == 0 else [0.55, 0.45]
        return DecideResult(distributions=dists, distribution_options={"probe_1": ["peru", "china"]}, cost_usd=0)

    ports = dp_testing.test_ports(
        generate={"probe_gen": {"value": {"probes": PROBES}, "cost_usd": 0.001}, "writer": writer},
        decide={"classify": classify},
    )

    result = await run_pipeline(pipeline, In(id="r1", title="Chifa lomo saltado"), ports)
    record = result.record
    assert result.output.text == "written"

    call = ports.decide_calls[0]
    assert call.questions == "jev-demo"
    assert list(call.probes.keys()) == ["probe_0", "probe_1"]

    node = next(n for n in record.nodes if n.id == "classify")
    assert "tradition" in node.probes.asked["probe_1"]["instructions"]
    assert len(node.probes.dropped) == 1
    assert node.probes.dropped[0].index == 1
    assert "addresses no state path" in node.probes.dropped[0].reason
    assert list((node.distributions or {}).keys()) == ["dish", "probe_0", "probe_1"]

    # Evidence for the writer...
    assert [[a.key, a.options] for a in writer_probes["v"]] == [
        ["probe_0", ["true"]],
        ["probe_1", ["peru", "china"]],
    ]
    # ...and never for a gate or `interpret`.
    assert seen["gate"] == {}
    # TS: Object.isFrozen. Here: a read-only mapping.
    with pytest.raises(TypeError):
        seen["gate"]["x"] = 1
    assert list(seen["interpret"].keys()) == ["dish"]

    # Aggregates beside the static question's own features, no per-slot keys.
    assert record.features["classify.probes.asked"] == 2
    assert record.features["classify.probes.dropped"] == 1
    assert record.features["classify.dish.top"] == 0.8
    assert not any("probe_" in k for k in record.features)


async def test_still_asks_the_static_questions_when_the_writer_fails():
    pipeline = define_pipeline(spec())

    def boom(_req):
        raise RuntimeError("model down")

    ports = dp_testing.test_ports(
        generate={"probe_gen": boom, "writer": {"value": {"text": "w"}}},
        decide={"classify": {"distributions": {"dish": [0.9]}}},
    )
    result = await run_pipeline(pipeline, In(id="r1", title="t"), ports)
    assert ports.decide_calls[0].probes == {}
    node = next(n for n in result.record.nodes if n.id == "classify")
    assert node.probes.dropped[0].reason == "probe_gen did not run"


async def test_a_probe_only_decide_node_with_nothing_valid_to_ask_fails_through_its_on_failure_with_the_drops_recorded():
    base = spec()
    nodes = {
        **base.nodes,
        "classify": DecideNode(probes=ProbeSpec(from_="probe_gen", max=2, paths=["recipe"]), on_failure="skip", version="1"),
        "gate": dataclasses.replace(base.nodes["gate"], run=lambda _a: "plain"),
    }
    pipeline = define_pipeline(dataclasses.replace(base, nodes=nodes, edges=[*base.edges, Edge(from_="evidence", to="gate")]))
    ports = dp_testing.test_ports(generate={"probe_gen": {"value": [noul("Is it sunny?")]}}, decide={})
    result = await run_pipeline(pipeline, In(id="r1", title="t"), ports)
    assert len(ports.decide_calls) == 0
    assert "no probes to ask (1 dropped" in result.skipped["classify"]
    node = next(n for n in result.record.nodes if n.id == "classify")
    assert len(node.probes.dropped) == 1


# --- probe contract ---


def problems(patch) -> list[str]:
    s = spec()
    s = dataclasses.replace(s, nodes=dict(s.nodes), edges=list(s.edges))
    patch(s)
    return validate_pipeline(s)


def _any_contains(items: list[str], needle: str) -> bool:
    return any(needle in p for p in items)


def test_loads_the_demo_manifest_and_exports_probes_in_to_json():
    j = define_pipeline(spec()).to_json()
    node = next(n for n in j["nodes"] if n["id"] == "classify")
    assert node["probes"] == {"from": "probe_gen", "max": 2, "paths": ["recipe"]}


def test_requires_a_generate_writer_on_a_direct_edge():
    def not_generate(s):
        s.nodes["classify"] = dataclasses.replace(s.nodes["classify"], probes=ProbeSpec(from_="evidence", max=2))

    assert _any_contains(problems(not_generate), "probes.from must be a generate node")

    def no_edge(s):
        s.edges[:] = [e for e in s.edges if not (e.from_ == "probe_gen" and e.to == "classify")]

    assert _any_contains(problems(no_edge), "needs a direct edge probe_gen -> classify")


def test_refuses_a_decide_node_with_neither_questions_nor_probes_a_cached_probe_node_and_an_out_of_range_max():
    def bare(s):
        s.nodes["classify"] = DecideNode(version="1")

    assert _any_contains(problems(bare), "needs `questions`, `probes`, or both")

    def cached(s):
        s.nodes["classify"] = dataclasses.replace(s.nodes["classify"], cache="per-run")

    assert _any_contains(problems(cached), "may not be cached")

    base = spec()
    with pytest.raises(PipelineContractError):
        define_pipeline(
            dataclasses.replace(
                base,
                nodes={
                    **base.nodes,
                    "classify": DecideNode(questions="q", probes=ProbeSpec(from_="probe_gen", max=50), version="1"),
                },
            )
        )
