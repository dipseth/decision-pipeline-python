"""Port of src/learned.test.ts.

The pure tests come first. The ones that compile or run a spec need the core
modules (spec, run, testing, fixtures, jsonata) and sit in the last section;
they skip until those modules exist.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

import pytest

from decision_pipeline._compat import js_number, js_round_precision, maybe_await
from decision_pipeline.discover import (
    DiscoverAction,
    DiscoveredQuestion,
    DiscoverPorts,
    DiscoverRow,
    design_matrix,
    predict_linear,
    predict_logistic,
    question_id,
)
from decision_pipeline.hash import hash_to_unit
from decision_pipeline.learned import (
    BaseColumn,
    LearnedModel,
    QuestionColumn,
    attach_learned_gate,
    discover_for_spec,
    fit_learned_model,
    learned_expression,
    learned_importance,
    learned_question_set,
    score_learned,
)


def u(s: str) -> float:
    return hash_to_unit(s)


def close(a: float, b: float, digits: int = 2) -> bool:
    """vitest `toBeCloseTo`."""
    return abs(a - b) < 10 ** (-digits) / 2


def matches(obj: Any, expected: Mapping[str, Any]) -> bool:
    """vitest `toMatchObject` for dicts (or objects with attributes)."""
    for k, v in expected.items():
        got = obj.get(k) if isinstance(obj, Mapping) else getattr(obj, k, None)
        if isinstance(v, Mapping):
            if not matches(got, v):
                return False
        elif got != v:
            return False
    return True


QS = [
    DiscoveredQuestion(name="spicy", kind="presence", question="Is it spicy?", round=1),
    DiscoveredQuestion(name="heat", kind="intensity", question="How much heat does the method use?", round=2),
]


def model(target: str) -> LearnedModel:
    return LearnedModel(
        target=target,  # type: ignore[arg-type]
        columns=[
            BaseColumn(base="p"),
            QuestionColumn(question="spicy", kind="presence", encode="p"),
            QuestionColumn(question="heat", kind="intensity", encode="mean"),
            QuestionColumn(question="heat", kind="intensity", encode="sd"),
        ],
        w=[0.7, 1.3, -0.4, 0.25],
        b=-0.2 if target == "binary" else 88,
        mean=[0.5, 0.4, 2, 0.8],
        sd=[0.2, 0.3, 1.1, 0.4],
    )


# --- learned_expression ------------------------------------------------------


def test_matches_the_learner_it_was_fitted_with() -> None:
    rows = [DiscoverRow(id=f"r{i}", group=f"r{i}", label=1 if u(f"s{i}") > 0.5 else 0, base=[u(f"p{i}")], text="") for i in range(120)]
    answers = {
        question_id(QS[0]): [[u(f"s{i}")] for i in range(len(rows))],
        question_id(QS[1]): [[1 if k == math.floor(u(f"h{i}") * 5) else 0 for k in range(5)] for i in range(len(rows))],
    }
    for target in ("binary", "numeric"):
        m = fit_learned_model(rows=rows, base_names=["p"], questions=QS, answers=answers, target=target)
        X = design_matrix(rows, QS, answers).X
        predict = predict_logistic if target == "binary" else predict_linear
        for i, r in enumerate(rows[:10]):
            got = score_learned(m, {"spicy": answers[question_id(QS[0])][i], "heat": answers[question_id(QS[1])][i]}, {"p": r.base[0]})
            assert close(got, predict(m, X[i]), 9)  # type: ignore[arg-type]
        imp = learned_importance(m)
        assert sorted(imp.keys()) == ["heat", "p", "spicy"]
        assert imp["spicy"] > imp["p"]


# --- learned_question_set ----------------------------------------------------


def test_asks_presence_as_a_noul_and_intensity_as_a_five_level_score() -> None:
    s = learned_question_set(QS)
    assert matches(s["spicy"], {"type": "noul", "instructions": "Is it spicy?"})
    assert matches(s["heat"], {"type": "score"})
    assert len(s["heat"]["criteria"]) == 5


# --- Python-only: the generated text is the TS text --------------------------


def test_learned_expression_text_matches_the_ts_output() -> None:
    # Produced by src/learned.ts `learnedExpression(model("numeric"), { answers: "answers", base: "base" })`.
    expected = "\n".join(
        [
            "(",
            '  $b_p := $lookup(base, "p"); $b_p := $type($b_p) = "number" ? $b_p : 0.5;',
            '  $d_spicy := $lookup(answers, "spicy");',
            '  $d_heat := $lookup(answers, "heat");',
            "  $m_heat := $exists($d_heat) ? $sum($map($d_heat, function($p, $i) { $p * $i })) / $sum($d_heat) : null;",
            "  $s_heat := $exists($d_heat) ? $sqrt($max([0, $sum($map($d_heat, function($p, $i) { $p * $i * $i })) / $sum($d_heat) - $m_heat * $m_heat])) : null;",
            "  $z := 84.74393939 + 3.5 * $b_p + 4.333333333 * ($exists($d_spicy) ? $d_spicy[0] : 0.4) + (-0.3636363636) * ($m_heat != null ? $m_heat : 2) + 0.625 * ($s_heat != null ? $s_heat : 0.8);",
            '  { "score": $z }',
            ")",
        ]
    )
    assert learned_expression(model("numeric"), answers="answers", base="base") == expected


def test_learned_expression_refuses_bad_roots() -> None:
    with pytest.raises(ValueError, match="plain identifiers"):
        learned_expression(model("binary"), answers="a.b", base="base")
    with pytest.raises(ValueError, match="`base` root"):
        learned_expression(model("binary"), answers="answers")


# --- needs spec/run/testing/fixtures/jsonata (wave 2) ---


def _engine() -> Any:
    mod = pytest.importorskip("decision_pipeline.jsonata")
    return mod.jsonata_engine()


def _registry() -> Any:
    fx = pytest.importorskip("decision_pipeline.fixtures")
    reg = fx.demo_registry
    return reg() if callable(reg) else reg


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, Mapping) else getattr(obj, key)


def _plain(v: Any) -> Any:
    dump = getattr(v, "model_dump", None)
    return dump() if callable(dump) else v


def without_gate() -> dict[str, Any]:
    """The demo pipeline with its hand-written gate removed, so a learned one can take its place."""
    fx = pytest.importorskip("decision_pipeline.fixtures")
    spec = copy.deepcopy(fx.demo_spec())
    del spec["nodes"]["gate"]
    spec["edges"] = [e for e in spec["edges"] if e["from"] != "gate" and e["to"] != "gate"]
    spec["nodes"]["loud"] = {"kind": "code", "role": "derive", "call": "demo.shout@1", "args": {"title": "$input.title", "n": "$half($input.factor)"}, "version": "1"}
    spec["nodes"]["plain"] = {"kind": "code", "role": "derive", "expr": '{ "text": $input.title, "n": 0 }', "version": "1"}
    spec["feedback"] = [{**f, "to": "bundle"} for f in spec["feedback"]]
    return spec


ATTACH: dict[str, Any] = {
    "prefix": "disc",
    "after": "bundle",
    "base": "bundle",
    "questionsName": "jev-demo-disc",
    "gate": {"cut": 0.5, "above": "loud", "below": "plain", "routes": {"loud": ["loud"], "plain": ["plain"]}},
}


async def eval_expr(m: LearnedModel, answers: dict[str, list[float]], base: dict[str, float]) -> float:
    engine = _engine()
    out = await maybe_await(engine.compile(learned_expression(m, answers="answers", base="base")).evaluate({"answers": answers, "base": base}, {}))
    return _get(out, "score")  # type: ignore[no-any-return]


@pytest.mark.parametrize("target", ["binary", "numeric"])
async def test_computes_what_score_learned_computes(target: str) -> None:
    m = model(target)
    for i in range(20):
        heat = [u(f"a{i}"), u(f"b{i}"), u(f"c{i}"), u(f"d{i}"), u(f"e{i}")]
        answers = {"spicy": [u(f"s{i}")], "heat": heat}
        base = {"p": u(f"p{i}")}
        assert close(await eval_expr(m, answers, base), score_learned(m, answers, base), 6)


async def test_gives_a_missing_answer_or_base_column_its_training_mean_so_it_moves_nothing() -> None:
    m = model("numeric")
    at_means = score_learned(m, {"spicy": [0.4]}, {"p": 0.5})
    # heat missing: mean and sd both fall back; p at its mean; spicy at its mean.
    assert close(at_means, 88, 6)
    assert close(await eval_expr(m, {"spicy": [0.4]}, {}), 88, 6)


def test_publishes_the_presence_criteria_the_loop_asked_with() -> None:
    presence = {"true": "The note says so", "false": "It does not"}
    a = attach_learned_gate(without_gate(), model("binary"), QS, {**ATTACH, "presence": presence})
    b = attach_learned_gate(without_gate(), model("binary"), QS, ATTACH)
    assert matches(learned_question_set(QS, presence)["spicy"], {"criteria": presence})
    # Different wording is a different decide node version.
    assert a.spec["nodes"]["disc_ask"]["version"] != b.spec["nodes"]["disc_ask"]["version"]


async def test_adds_decide_derive_gate_and_the_patched_spec_compiles_and_runs() -> None:
    engine = _engine()
    registry = _registry()
    spec_mod = pytest.importorskip("decision_pipeline.spec")
    run_mod = pytest.importorskip("decision_pipeline.run")
    testing = pytest.importorskip("decision_pipeline.testing")

    m = model("binary")
    res = attach_learned_gate(without_gate(), m, QS, ATTACH)
    spec, ids, applied = res.spec, res.ids, res.applied
    assert ids == {"ask": "disc_ask", "score": "disc_score", "gate": "disc_gate"}
    assert len(applied) == 3
    assert matches(spec["nodes"]["disc_ask"], {"kind": "decide", "questions": "jev-demo-disc"})
    assert list(spec_mod.compile_pipeline_spec(spec, registry, engine).problems) == []

    answers = {"spicy": [0.9], "heat": [0, 0, 0.2, 0.5, 0.3]}
    expected = score_learned(m, answers, {"p": 0.8})
    ports = testing.test_ports(decide={"disc_ask": lambda _req: {"distributions": answers}})
    run = await run_mod.run_pipeline(spec_mod.pipeline_from_spec(spec, registry, engine), {"id": "r1", "title": "pad thai", "factor": 4}, ports)
    assert expected > 0.5
    assert run.route == "loud"
    assert _plain(run.output) == {"text": "PAD THAI", "n": 2}
    gate = next(n for n in run.record.nodes if _get(n, "id") == "disc_gate")
    assert f"score {js_number(js_round_precision(expected, 6))[:6]}" in _get(gate, "branch_reason")
    call = ports.decide_calls[0]
    assert _get(call, "node_id") == "disc_ask"
    assert _get(call, "questions") == "jev-demo-disc"


def test_replaces_its_own_fragment_on_a_refit_and_the_score_nodes_version_moves() -> None:
    engine = _engine()
    registry = _registry()
    spec_mod = pytest.importorskip("decision_pipeline.spec")

    first = attach_learned_gate(without_gate(), model("binary"), QS, ATTACH)
    refit = model("binary")
    refit.w = [0.1, 0.2, 0.3, 0.4]
    second = attach_learned_gate(first.spec, refit, QS, ATTACH)
    assert re.match(r"^replaced disc_ask, disc_score, disc_gate", second.applied[0])
    assert len([e for e in second.spec["edges"] if e["to"] == "disc_ask"]) == 1

    def v(s: dict[str, Any]) -> str:
        return spec_mod.pipeline_from_spec(s, registry, engine).nodes["disc_score"].version  # type: ignore[no-any-return]

    assert v(second.spec) != v(first.spec)


def test_refuses_a_model_that_reads_base_columns_when_no_base_is_given() -> None:
    with pytest.raises(ValueError, match="base"):
        attach_learned_gate(without_gate(), model("binary"), QS, {**ATTACH, "base": None})


async def test_runs_the_loop_fits_the_final_model_and_returns_a_challenger_spec_that_compiles() -> None:
    engine = _engine()
    registry = _registry()
    pytest.importorskip("decision_pipeline.spec")

    rows = [
        DiscoverRow(id=f"r{i}", group=f"r{i}", label=1 if u(f"s{i}") + 0.2 * u(f"n{i}") > 0.6 else 0, base=[u(f"p{i}")], text=f"row {i}")
        for i in range(160)
    ]

    async def author(_req: Any) -> list[DiscoverAction]:
        return [DiscoverAction(op="add", target="", name="spicy", kind="presence", question="Is it spicy?")]

    async def answer(rs: list[DiscoverRow], qs: list[DiscoveredQuestion]) -> dict[str, list[list[float]]]:
        return {q.name: [[u(f"s{r.id[1:]}")] for r in rs] for q in qs}

    out = await discover_for_spec(
        rows=rows,
        base_names=["p"],
        rounds=1,
        spec=without_gate(),
        attach=ATTACH,
        check={"registry": registry, "engine": engine},
        ports=DiscoverPorts(author=author, answer=answer),
    )
    assert out.problems == []
    assert [q.name for q in out.discovery.accepted] == ["spicy"]
    assert matches(out.question_set["spicy"], {"type": "noul"})
    assert out.model.columns == [BaseColumn(base="p"), QuestionColumn(question="spicy", kind="presence", encode="p")]
    assert out.model.w[1] > abs(out.model.w[0])
