"""Port of src/discover.test.ts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from decision_pipeline.discover import (
    DiscoverAction,
    DiscoverPorts,
    DiscoverRow,
    auc,
    cross_validate,
    discover_features,
    encode_answer,
    fit_logistic,
    fit_ridge,
    group_folds,
    predict_linear,
    predict_logistic,
    spearman,
)
from decision_pipeline.hash import hash_to_unit


def u(s: str) -> float:
    """Deterministic pseudo-random in [0,1) from a string."""
    return hash_to_unit(s)


def close(a: float, b: float, digits: int = 2) -> bool:
    """vitest `toBeCloseTo`."""
    return abs(a - b) < 10 ** (-digits) / 2


@dataclass(kw_only=True)
class PlantedRow(DiscoverRow):
    signal: float


def planted_rows(n: int) -> list[PlantedRow]:
    """Rows whose label depends on a hidden "signal" only the right question can see."""
    out = []
    for i in range(n):
        signal = u(f"s{i}")
        noise = u(f"n{i}")
        out.append(
            PlantedRow(
                id=f"r{i}",
                group=f"g{i // 2}",
                label=1 if signal + 0.3 * noise > 0.65 else 0,
                base=[u(f"b{i}")],
                text=f"row {i}",
                signal=signal,
            )
        )
    return out


# --- logistic learner --------------------------------------------------------


def test_recovers_a_planted_direction_and_predicts_better_than_chance() -> None:
    X = [[u(f"x{i}"), u(f"z{i}")] for i in range(400)]
    y = [1 if a > 0.5 else 0 for a, _ in X]
    m = fit_logistic(X, y, 0.1)
    assert m.w[0] > abs(m.w[1]) * 5
    assert auc(y, [predict_logistic(m, x) for x in X]) > 0.95


def test_encodes_intensity_as_mean_level_plus_spread_presence_as_p() -> None:
    assert encode_answer("presence", [0.8]) == [0.8]
    mean, sd = encode_answer("intensity", [0, 0, 1, 0, 0])
    assert close(mean, 2)
    assert close(sd, 0)


# --- grouped folds -----------------------------------------------------------


def test_never_splits_a_group_across_folds() -> None:
    groups = [f"g{i % 40}" for i in range(300)]
    f = group_folds(groups, 5, 0)
    seen: dict[str, int] = {}
    for i, g in enumerate(groups):
        if g in seen:
            assert f[i] == seen[g]
        seen[g] = f[i]
    assert len(set(f)) == 5


def test_cv_of_a_label_leaking_column_beats_base() -> None:
    rows = planted_rows(300)
    y = [r.label for r in rows]
    base = cross_validate([r.base for r in rows], y, [r.group for r in rows]).score
    with_signal = cross_validate([[*r.base, r.signal] for r in rows], y, [r.group for r in rows]).score
    assert with_signal.loss < base.loss
    assert with_signal.auc is not None and with_signal.auc > 0.85


# --- discover_features -------------------------------------------------------


async def test_keeps_the_informative_question_drops_the_flat_one_and_rejects_a_worse_revision() -> None:
    rows = planted_rows(240)
    scripted: list[list[DiscoverAction]] = [
        [
            DiscoverAction(op="add", target="", name="Signal", kind="presence", question="Does it show the signal?"),
            DiscoverAction(op="add", target="", name="Flat", kind="presence", question="Is it a row?"),
            DiscoverAction(op="add", target="", name="Noise", kind="intensity", question="How noisy is it?"),
        ],
        [DiscoverAction(op="revise", target="signal", name="signal", kind="presence", question="A worse wording")],
    ]
    requests: list[int] = []

    async def author(req):  # type: ignore[no-untyped-def]
        requests.append(req.round)
        if req.round == 2:
            assert "signal" in req.feedback
        return scripted[req.round - 1] if req.round - 1 < len(scripted) else []

    async def answer(rs, qs):  # type: ignore[no-untyped-def]
        out: dict[str, list[list[float]]] = {}
        for q in qs:
            col = []
            for i, r in enumerate(rs):
                s = r.signal
                if q.name == "flat":
                    col.append([0.5])
                elif q.name == "noise":
                    level = math.floor(u(f"q{i}") * 5)
                    col.append([1 if k == level else 0 for k in range(5)])
                else:
                    col.append([u(f"w{i}")] if q.round == 2 else [s])
            out[q.name] = col
        return out

    out = await discover_features(rows=rows, base_names=["base0"], rounds=2, ports=DiscoverPorts(author=author, answer=answer))
    assert requests == [1, 2]
    names = [q.name for q in out.accepted]
    assert "signal" in names
    assert "flat" not in names
    assert next(q for q in out.accepted if q.name == "signal").round == 1
    assert any(j.what == "reject" and j.name == "signal" for j in out.journal)
    assert out.history[0].loss < out.base_cv.loss


# --- numeric target ----------------------------------------------------------


@dataclass(kw_only=True)
class ScoredRow(DiscoverRow):
    quality: float


def scored(n: int) -> list[ScoredRow]:
    """A critic-style score: 80 + 12 × a hidden quality only the right question can see, plus noise."""
    out = []
    for i in range(n):
        quality = u(f"q{i}")
        out.append(
            ScoredRow(
                id=f"w{i}",
                group=f"w{i}",
                label=80 + 12 * quality + 2 * (u(f"e{i}") - 0.5),
                base=[u(f"b{i}")],
                text=f"note {i}",
                quality=quality,
            )
        )
    return out


def test_ridge_recovers_a_planted_slope() -> None:
    X = [[u(f"x{i}"), u(f"z{i}")] for i in range(300)]
    y = [3 + 5 * a for a, _ in X]
    m = fit_ridge(X, y, 1e-6)
    assert close(predict_linear(m, [0.5, 0.9]), 5.5, 3)
    assert abs(m.w[1]) < 1e-3


def test_spearman_is_1_for_a_monotone_map_and_minus_1_for_a_reversed_one() -> None:
    assert close(spearman([1, 2, 3, 4], [10, 20, 30, 45]), 1)
    assert close(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1)


async def test_scores_rounds_in_rmse_shows_the_author_the_label_range_and_keeps_the_question_that_sees_quality() -> None:
    rows = scored(200)
    first_examples = ""

    async def author(req):  # type: ignore[no-untyped-def]
        nonlocal first_examples
        first_examples = req.examples
        return [
            DiscoverAction(op="add", target="", name="quality", kind="intensity", question="How good does it sound?"),
            DiscoverAction(op="add", target="", name="noise", kind="presence", question="Is it odd?"),
        ]

    async def answer(rs, qs):  # type: ignore[no-untyped-def]
        def col(q):  # type: ignore[no-untyped-def]
            out = []
            for i, r in enumerate(rs):
                if q.name == "noise":
                    out.append([u(f"o{i}")])
                    continue
                level = min(4, math.floor(r.quality * 5))
                out.append([1 if k == level else 0 for k in range(5)])
            return out

        return {q.name: col(q) for q in qs}

    out = await discover_features(
        rows=rows,
        base_names=["base0"],
        target="numeric",
        rounds=1,
        examples=5,
        ports=DiscoverPorts(author=author, answer=answer),
    )
    assert out.base_cv.rmse is not None and out.base_cv.rmse > 3
    assert out.history[0].rmse is not None and out.history[0].rmse < 1.5
    assert out.history[0].spearman is not None and out.history[0].spearman > 0.9
    assert out.history[0].log_loss is None
    # Round 1 spans the range: lowest and highest label both shown.
    from decision_pipeline._compat import js_number

    labels = [r.label for r in rows]
    assert f"label {js_number(math.floor(min(labels) * 100 + 0.5) / 100)}" in first_examples
    assert f"label {js_number(math.floor(max(labels) * 100 + 0.5) / 100)}" in first_examples


async def test_refuses_non_0_1_labels_on_a_binary_target() -> None:
    async def author(_req):  # type: ignore[no-untyped-def]
        return []

    async def answer(_rs, _qs):  # type: ignore[no-untyped-def]
        return {}

    with pytest.raises(Exception, match="numeric"):
        await discover_features(rows=scored(10), base_names=["b"], ports=DiscoverPorts(author=author, answer=answer))
