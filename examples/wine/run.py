"""
TypeSafe's "Autoresearch feature discovery" cookbook, re-run on
decision-pipeline: the same 2,000 wine reviews (1,200 dev / 800 held out,
the cookbook's seeded sample), the same brief, Claude Sonnet 5 as the
author, Jev answering. Then `discover_for_spec` turns the result into a spec,
and the compiled pipeline scores held-out notes through a real decide port.

This is the Python twin of run.ts. Same CLI, same env vars, same cache file
(data/cache.jsonl, same keys), same output files, so the two replay each
other's Jev answers.

  python3 examples/wine/sample.py                  # once: data/wine.json
  TYPESAFE_API_KEY=... ANTHROPIC_API_KEY=... python/.venv/bin/python examples/wine/run.py --run r1

Two departures from the cookbook, on purpose:
  - the learner is ridge regression, not CatBoost, because the model has to
    be writable into the spec as one expression (learned.py);
  - the intensity rubric says "the text" (the package's INTENSITY_LEVELS),
    because the production decide node asks the same wording the loop did.

Every Jev answer is cached in data/cache.jsonl, and each run's loop in
data/runs/<run>/discovery.json, so a re-run replays without calling anything.

Standard library plus pydantic: the HTTP calls use urllib on a thread pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

from decision_pipeline import (
    INTENSITY_LEVELS,
    DiscoverAction,
    DiscoveredQuestion,
    DiscoverPorts,
    DiscoverResult,
    DiscoverRow,
    define_registry,
    design_matrix,
    discover_features,
    discover_for_spec,
    fit_ridge,
    learned_importance,
    learned_question_set,
    pipeline_from_spec,
    predict_linear,
    question_id,
    rmse,
    run_pipeline,
    score_learned,
    spearman,
)
from decision_pipeline import JEV_MODEL as DEFAULT_JEV_MODEL
from decision_pipeline.jsonata import jsonata_engine
from decision_pipeline.testing import test_ports

DATA = Path(__file__).resolve().parent / "data"
# The table in README.md was measured on jev-1.13.0; set JEV_MODEL=jev-1.13.0 to reproduce it.
JEV_MODEL = os.environ.get("JEV_MODEL") or DEFAULT_JEV_MODEL
AUTHOR_MODEL = os.environ.get("AUTHOR_MODEL") or "claude-sonnet-5"
USD_PER_JEV_TOKEN = 0.042e-6
N_DEV, N_TEST, EXAMPLES, PROPOSALS, WORKERS = 1200, 800, 60, 18, 16
PRESENCE = {"true": "The note states this or clearly implies it", "false": "The note gives no indication of this"}
SCORE_LEVELS = [
    "Faulty or unpleasant - the note is mostly criticism",
    "Barely acceptable - drinkable, with nothing to recommend it",
    "Simple and sound - correct, plain, forgettable",
    "Pleasant everyday wine - some appeal, little depth",
    "Good - clear varietal character, well made",
    "Very good - balanced, with something to say",
    "Excellent - complex and structured",
    "Outstanding - depth and length, built to age",
    "Superb - among the best of its type",
    "Profound - the note treats it as exceptional",
]

# ---------------------------------------------------------------- the author (Claude)

# The cookbook's brief, with the learner named truthfully.
_LEVELS = "\n".join(f"  {i}. {l}" for i, l in enumerate(INTENSITY_LEVELS))
BRIEF = f"""You are designing numeric features for a linear (ridge) regression model that
predicts the score a wine critic gave (an integer from 80 to 100) from the tasting note alone.
The model sees nothing but the features you design.

Return up to {PROPOSALS} actions. Each action is one of:

- {{"op": "add", "target": "", "name": ..., "kind": ..., "question": ...}}
  A new feature.
- {{"op": "revise", "target": <name of an existing feature>, "name": ..., "kind": ..., "question": ...}}
  Replace that feature's question with better wording. Use this when a feature measures the
  right thing badly: too narrow, too vague, or worded so nearly every note answers the same.
- {{"op": "drop", "target": <name of an existing feature>, "name": "", "kind": "intensity", "question": ""}}
  Remove a feature that is not earning its place.

`kind` is "intensity" for something with a degree, or "presence" for a yes/no fact.
`question` is what gets asked about one tasting note.

An "intensity" question is graded against this fixed five-level rubric, so word it so that the
levels make sense:
{_LEVELS}

A "presence" question is answered as the probability that it is true of the note.

Good features can be judged from the note's own words, vary from note to note, and carry
information about quality that the other features do not. Reviewers describe structure, fruit,
oak, length, complexity, and drinkability, and they also signal quality through word choice."""


class Action(BaseModel):
    op: Literal["add", "revise", "drop"]
    target: str
    name: str
    kind: Literal["intensity", "presence"]
    question: str


class Actions(BaseModel):
    actions: list[Action]


def need(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set")
    return v


def _post(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> tuple[int, str]:
    """One blocking POST. Returns (status, text); a transport error raises OSError."""
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode(), headers={"content-type": "application/json", **headers}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


async def author(req: Any) -> list[DiscoverAction]:
    """One Messages API call with a forced `submit` tool, so the actions come back as validated JSON."""
    accepted = "\n".join(f"- {q.name} ({q.kind}): {q.question}" for q in req.accepted)
    prompt = "\n".join([
        req.examples,
        f"\nThe features you have now. `add` must not duplicate one of these; `revise` and `drop` refer to one by name:\n{accepted}" if req.accepted else "",
        f"\nHow the model did with those features:\n{req.feedback}" if req.feedback else "",
    ])
    body = {
        "model": AUTHOR_MODEL,
        "max_tokens": 16000,
        "system": BRIEF,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"name": "submit", "description": "Submit this round's actions once.", "input_schema": Actions.model_json_schema()}],
        "tool_choice": {"type": "tool", "name": "submit"},
    }
    headers = {"x-api-key": need("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}
    for _ in range(3):
        status, text = await asyncio.to_thread(_post, "https://api.anthropic.com/v1/messages", headers, body, 600)
        if status != 200:
            print(f"author: HTTP {status} {text[:200]}", file=sys.stderr)
            continue
        block = next((b for b in json.loads(text)["content"] if b.get("type") == "tool_use"), None)
        try:
            return [DiscoverAction(**a.model_dump()) for a in Actions.model_validate((block or {}).get("input")).actions]
        except ValueError:
            continue
    raise RuntimeError("author made no valid submit call")


# ---------------------------------------------------------------- Jev, cached

CACHE = DATA / "cache.jsonl"
cache: dict[str, list[float]] = {}
usage = {"calls": 0, "tokens": 0}


def _js_json(v: Any) -> str:
    """JSON.stringify: compact, insertion order, non-ASCII left alone."""
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def key_of(note: str, q: dict[str, Any]) -> str:
    return hashlib.sha256(f"{JEV_MODEL}|{_js_json(q)}|{note}".encode()).hexdigest()[:32]


def parse_answer(q: dict[str, Any], a: Optional[dict[str, Any]]) -> list[float]:
    """presence -> [P(true)]; score -> one probability per level. The shape learned.py expects."""
    a = a or {}
    if q["type"] == "noul":
        v = a.get("noul")
        return [0.5 if v is None else v]
    levels = q["criteria"] if q["type"] == "score" else list(q["criteria"].keys())
    probs = a.get("probabilities") or {}
    out = []
    for k, lvl in enumerate(levels):
        v = probs.get(str(k))
        if v is None:
            v = probs.get(lvl)
        if v is None:
            v = 1 if a.get("score") == k else 0
        out.append(v)
    return out


async def jev(state: str, questions: dict[str, Any]) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {need('TYPESAFE_API_KEY')}"}
    body = {"model": JEV_MODEL, "state": state, "questions": questions}
    attempt = 0
    while True:
        err: Optional[str] = None
        status = 0
        try:
            status, text = await asyncio.to_thread(_post, "https://api.typesafe.ai/v1/systemone", headers, body, 60)
        except OSError as e:
            err = str(e)
        if err is None and status == 200:
            res = json.loads(text)
            usage["calls"] += 1
            usage["tokens"] += (res.get("usage") or {}).get("input_tokens") or 0
            return res["answers"]
        retryable = err is not None or status == 429 or status >= 500
        if not retryable or attempt >= 4:
            raise RuntimeError(f"Jev: {err if err is not None else f'HTTP {status} {text}'}")
        await asyncio.sleep(attempt + 1)
        attempt += 1


async def ask_note(note: str, qs: dict[str, Any]) -> dict[str, list[float]]:
    """One Jev request per note, carrying every question not already cached."""
    missing = {k: q for k, q in qs.items() if key_of(note, q) not in cache}
    if missing:
        answers = await jev(note, missing)
        with CACHE.open("a", encoding="utf-8") as f:
            for k, q in missing.items():
                key = key_of(note, q)
                p = parse_answer(q, answers.get(k))
                cache[key] = p
                f.write(_js_json({"key": key, "p": p}) + "\n")
    return {k: cache[key_of(note, q)] for k, q in qs.items()}


async def pool(items: list[Any], fn: Any) -> list[Any]:
    """Run `fn` over `items` with a fixed number of requests in flight."""
    out: list[Any] = [None] * len(items)
    state = {"next": 0, "done": 0}

    async def worker() -> None:
        while state["next"] < len(items):
            i = state["next"]
            state["next"] += 1
            out[i] = await fn(items[i])
            state["done"] += 1
            if state["done"] % 300 == 0:
                print(f"    {state['done']}/{len(items)}")

    await asyncio.gather(*(worker() for _ in range(WORKERS)))
    return out


async def answer_rows(notes: list[str], qs: list[DiscoveredQuestion]) -> dict[str, list[list[float]]]:
    """One source of wording for the loop AND the question set the spec publishes."""
    question_set = learned_question_set(qs, PRESENCE)
    got = await pool(notes, lambda note: ask_note(note, question_set))
    return {q.name: [g[q.name] for g in got] for q in qs}


# ---------------------------------------------------------------- stats


def mulberry(seed: int):
    """mulberry32, bit for bit the TS generator."""
    state = {"s": seed & 0xFFFFFFFF}
    M = 0xFFFFFFFF

    def imul(a: int, b: int) -> int:
        return (a * b) & M

    def rnd() -> float:
        s = (state["s"] + 0x6D2B79F5) & M
        state["s"] = s
        t = imul(s ^ (s >> 15), 1 | s)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296

    return rnd


def paired_gain(y: list[float], before: list[float], after: list[float]) -> dict[str, float]:
    """Paired bootstrap of the held-out RMSE change, after - before."""
    rnd = mulberry(0)
    n = len(y)
    draws: list[float] = []
    for _ in range(2000):
        s0 = s1 = 0.0
        for _k in range(n):
            i = int(rnd() * n)
            s0 += (y[i] - before[i]) ** 2
            s1 += (y[i] - after[i]) ** 2
        draws.append(math.sqrt(s1 / n) - math.sqrt(s0 / n))
    draws.sort()
    return {"diff": rmse(y, after) - rmse(y, before), "lo": draws[49], "hi": draws[1949]}


def to_exponential(x: float, digits: int) -> str:
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m}e{e[0]}{int(e[1:])}"


# ---------------------------------------------------------------- main


async def main() -> None:
    args = sys.argv[1:]

    def flag(n: str) -> Optional[str]:
        return args[args.index(n) + 1] if n in args and args.index(n) + 1 < len(args) else None

    run = flag("--run")
    if not run:
        raise RuntimeError("--run <name> required")
    rounds = int(float(flag("--rounds") or 5))
    out_dir = DATA / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (DATA / "wine.json").exists():
        raise RuntimeError("no data/wine.json: run python3 examples/wine/sample.py first")
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").split("\n"):
            if line:
                c = json.loads(line)
                cache[c["key"]] = c["p"]

    data = json.loads((DATA / "wine.json").read_text(encoding="utf-8"))
    all_rows = [DiscoverRow(id=f"w{i}", group=f"w{i}", label=data["points"][i], base=[], text=text) for i, text in enumerate(data["notes"])]
    dev, test = all_rows[:N_DEV], all_rows[N_DEV : N_DEV + N_TEST]
    y_dev, y_test = [r.label for r in dev], [r.label for r in test]
    print(f"dev {len(dev)}, held out {len(test)}; Jev {JEV_MODEL}, author {AUTHOR_MODEL}")

    # 1. The loop reads dev rows only.
    result_file = out_dir / "discovery.json"
    t0 = time.time()
    reused = result_file.exists()
    if reused:
        discovery = DiscoverResult.from_dict(json.loads(result_file.read_text(encoding="utf-8")))
    else:
        discovery = await discover_features(
            rows=dev, base_names=[], target="numeric", rounds=rounds, examples=EXAMPLES, max_actions=PROPOSALS,
            folds=5, repeats=3, min_spread=0.05, log=print,
            ports=DiscoverPorts(author=author, answer=lambda rows, qs: answer_rows([r.text for r in rows], qs)),
        )
        result_file.write_text(_js_json(discovery.to_dict()), encoding="utf-8")
    took = "" if reused else f", {(time.time() - t0) / 60:.1f} min"
    print(f"discovery: {len(discovery.accepted)} questions{' (reused)' if reused else took}")

    # 2. Held out, once: each round's question set fitted on dev and scored on test.
    snapshot_qs = list({question_id(q): q for qs in discovery.snapshots for q in qs}.values())
    print(f"answering {len(snapshot_qs)} questions on {len(test)} held-out notes")
    test_answers: dict[str, list[list[float]]] = {}
    for rnd_no in dict.fromkeys(q.round for q in snapshot_qs):
        qs = [q for q in snapshot_qs if q.round == rnd_no]
        got = await answer_rows([r.text for r in test], qs)
        for q in qs:
            test_answers[question_id(q)] = got[q.name]
    curve = []
    for i, qs in enumerate(discovery.snapshots):
        m = fit_ridge(design_matrix(dev, qs, discovery.answers).X, y_dev, 1)
        p = [predict_linear(m, x) for x in design_matrix(test, qs, test_answers).X]
        curve.append({"round": i + 1, "questions": len(qs), "dev": discovery.history[i].rmse, "test": rmse(y_test, p), "spearman": spearman(y_test, p), "p": p})
    gain = paired_gain(y_test, curve[0]["p"], curve[-1]["p"])

    # 3. Baselines: the dev mean, and asking Jev for the score outright.
    print("baseline: asking Jev for the score on all 2,000 notes")
    dev_mean = sum(y_dev) / len(y_dev)
    ask_q = {"type": "score", "instructions": "Judging only by what this tasting note says, how good is the wine?", "criteria": SCORE_LEVELS}

    async def ask_one(r: DiscoverRow) -> float:
        p = (await ask_note(r.text, {"quality": ask_q}))["quality"]
        t = sum(p) or 1
        return 80 + (20 * sum(k * v for k, v in enumerate(p))) / t / (len(SCORE_LEVELS) - 1)

    ask_dev = await pool(dev, ask_one)
    ask_test = await pool(test, ask_one)
    cal = fit_ridge([[v] for v in ask_dev], y_dev, 1e-6)
    ask_cal = [predict_linear(cal, [v]) for v in ask_test]

    # 4. The spec. A decision pipeline needs a gate and a human signal: the learned
    # score is cut at 90 (Wine Enthusiast's "outstanding"), and the critic's score is
    # the feedback edge.
    print("spec: discover_for_spec -> compile -> run on held-out notes")

    class WineInput(BaseModel):
        id: str
        note: str

    class WineOutput(BaseModel):
        score: float
        tier: Literal["notable", "everyday"]

    registry = define_registry(schemas={"wine.input": WineInput, "wine.output": WineOutput})
    engine = jsonata_engine()
    base: dict[str, Any] = {
        "id": "wine-score", "fact": "wine_tier", "input": "wine.input", "output": "wine.output", "trigger": ["on-demand"], "group": "wine-{id}",
        "result": ["notable", "everyday"],
        "nodes": {
            "note": {"kind": "code", "role": "transform", "inputs": {"note": "$input.note"}, "version": "1"},
            "notable": {"kind": "code", "role": "derive", "expr": '{ "score": wine_gate.score, "tier": "notable" }', "version": "1"},
            "everyday": {"kind": "code", "role": "derive", "expr": '{ "score": wine_gate.score, "tier": "everyday" }', "version": "1"},
            "persist": {"kind": "store", "target": "memory", "scope": "global", "version": "1"},
        },
        "edges": [{"from": "notable", "to": "persist"}, {"from": "everyday", "to": "persist"}],
        "feedback": [{"id": "critic", "from": "surface:wine", "to": "wine_gate", "source": "explicit", "scope": "user", "form": "score", "latency": "deferred", "score": {"name": "wine_critic_score", "dataType": "NUMERIC"}}],
        "eval": {"dataset": "wine-eval", "gateMetric": "score_positive"},
    }

    async def no_author(_req: Any) -> list[Any]:
        return []

    async def no_answer(_rows: Any, _qs: Any) -> dict[str, Any]:
        return {}

    out = await discover_for_spec(
        spec=base, rows=dev, base_names=[], target="numeric", discovery=discovery, ports=DiscoverPorts(author=no_author, answer=no_answer),
        attach={"prefix": "wine", "after": "note", "questionsName": "jev-wine-quality", "presence": PRESENCE,
                "gate": {"cut": 90, "above": "notable", "below": "everyday", "routes": {"notable": ["notable"], "everyday": ["everyday"]}}},
        check={"registry": registry, "engine": engine},
    )
    (out_dir / "spec.json").write_text(json.dumps(out.spec, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "question-set.json").write_text(json.dumps(out.question_set, indent=1, ensure_ascii=False), encoding="utf-8")
    if out.problems:
        raise RuntimeError("spec problems:\n  " + "\n  ".join(out.problems))

    # The decide port resolves the question-set name the spec carries, and asks Jev.
    pipeline = pipeline_from_spec(out.spec, registry, engine)
    final_p = curve[-1]["p"]
    max_diff, route_agree = 0.0, 0
    probe = test[:25]

    async def wine_ask(req: Any) -> dict[str, Any]:
        if req.questions != "jev-wine-quality":
            raise RuntimeError(f"unknown question set {req.questions}")
        return {"distributions": await ask_note(req.from_["note"]["note"], out.question_set)}

    for k, r in enumerate(probe):
        ports = test_ports(decide={"wine_ask": wine_ask})
        res = await run_pipeline(pipeline, {"id": r.id, "note": r.text}, ports)
        got = res.output
        score = got["score"] if isinstance(got, dict) else got.score
        offline = score_learned(out.model, {q.name: test_answers[question_id(q)][k] for q in discovery.accepted})
        max_diff = max(max_diff, abs(score - offline), abs(score - final_p[k]))
        if (res.route == "notable") == (offline >= 90):
            route_agree += 1

    # 5. Report.
    imp = sorted(learned_importance(out.model).items(), key=lambda kv: -kv[1])
    report = {
        "run": run, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", "jev": JEV_MODEL, "author": AUTHOR_MODEL, "jevUsage": usage,
        "heldOut": {
            "mean": {"rmse": rmse(y_test, [dev_mean] * len(y_test))},
            "askRaw": {"rmse": rmse(y_test, ask_test), "spearman": spearman(y_test, ask_test)},
            "askCalibrated": {"rmse": rmse(y_test, ask_cal), "spearman": spearman(y_test, ask_cal)},
            "rounds": [{k: v for k, v in c.items() if k != "p"} for c in curve],
            "round1ToFinal": gain,
        },
        "final": {
            "questions": len(discovery.accepted),
            "intensity": sum(q.kind == "intensity" for q in discovery.accepted),
            "presence": sum(q.kind == "presence" for q in discovery.accepted),
        },
        "importance": [{"question": k, "share": round(v * 1000) / 10} for k, v in imp[:12]],
        "spec": {"applied": out.applied, "pipelineVsOfflineMaxDiff": max_diff, "routeAgree": route_agree, "checkedRows": len(probe)},
        "journal": discovery.to_dict()["journal"],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    def row(name: str, r: float, s: Optional[float] = None) -> None:
        print(f"  {name:<46} {r:.3f}   {'' if s is None else f'{s:.3f}'}")

    h = report["heldOut"]
    print(f"\nheld out ({N_TEST} notes)                           RMSE    Spearman")
    row("predict the dev mean", h["mean"]["rmse"])
    row("ask Jev for the score, rescaled 80-100", h["askRaw"]["rmse"], h["askRaw"]["spearman"])
    row("ask Jev for the score, calibrated on dev", h["askCalibrated"]["rmse"], h["askCalibrated"]["spearman"])
    for c in curve:
        row(f"round {c['round']} ({c['questions']} questions), ridge", c["test"], c["spearman"])
    print(f"\nround 1 -> {len(curve)}: {'+' if gain['diff'] >= 0 else ''}{gain['diff']:.3f} points, 95% CI [{gain['lo']:.3f}, {gain['hi']:.3f}]")
    dev_cv = " -> ".join(f"{c['dev']:.3f}" for c in curve)
    print(f"dev CV RMSE per round: {dev_cv}")
    print(f"top questions: {', '.join(f'{k} {100 * v:.1f}%' for k, v in imp[:6])}")
    print(f"spec: {'; '.join(out.applied)}")
    print(f"compiled pipeline vs offline on {len(probe)} held-out notes: max |diff| {to_exponential(max_diff, 2)}, same route on {route_agree}/{len(probe)}")
    print(f"Jev this run: {usage['calls']} calls, {usage['tokens']} tokens (~${usage['tokens'] * USD_PER_JEV_TOKEN:.2f})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:  # noqa: BLE001
        print(f"RuntimeError: {e}" if isinstance(e, RuntimeError) else repr(e), file=sys.stderr)
        sys.exit(1)
