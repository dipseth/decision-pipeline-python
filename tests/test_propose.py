"""Port of src/propose.test.ts."""

from __future__ import annotations

import json
import re
from typing import Any

from decision_pipeline.fixtures import demo_registry as registry
from decision_pipeline.fixtures import demo_spec as base
from decision_pipeline.jsonata import jsonata_engine
from decision_pipeline.propose import AuthorRequest, SpecDiff, SpecVocabulary, diff_specs, extract_spec_json, propose_pipeline_spec, spec_vocabulary

engine = jsonata_engine()


def scripted(replies: list[Any]) -> tuple[Any, list[AuthorRequest]]:
    """An author that replies from a script, one entry per round, and keeps what it was asked."""
    seen: list[AuthorRequest] = []

    async def author(r: AuthorRequest) -> Any:
        seen.append(r)
        return replies[r.round - 1]

    return author, seen


def fenced(spec: Any) -> str:
    return "Raise the bar for loud.\n\n```json\n" + json.dumps(spec, separators=(",", ":")) + "\n```"


def stricter() -> dict[str, Any]:
    """The goal's answer: a stricter gate."""
    s = base()
    s["nodes"]["gate"] = {**s["nodes"]["gate"], "thresholds": {"p": 0.9}, "version": "2"}
    return s


# --- proposePipelineSpec ---


async def test_accepts_a_first_round_spec_that_compiles_and_changes_something() -> None:
    author, seen = scripted([fenced(stricter())])
    out = await propose_pipeline_spec(goal="only shout when p >= 0.9", registry=registry, engine=engine, author=author, base=base())
    assert out.problems == []
    assert out.pipeline is not None and out.pipeline.id == "spec-demo"
    assert len(out.rounds) == 1
    assert out.diff == SpecDiff(nodes={"added": [], "removed": [], "changed": ["gate"]}, edges={"added": [], "removed": []}, fields=[])
    # The system prompt carries the catalog; the first prompt carries the goal and the base.
    assert "demo.shout@1" in seen[0].system
    assert "only shout when p >= 0.9" in seen[0].prompt
    assert '"spec-demo"' in seen[0].prompt


async def test_feeds_every_compile_problem_back_and_accepts_the_repair() -> None:
    broken = stricter()
    broken["nodes"]["loud"] = {**broken["nodes"]["loud"], "call": "demo.whisper@1"}
    broken["nodes"]["plain"] = {**broken["nodes"]["plain"], "expr": "bundle.title"}
    author, seen = scripted([fenced(broken), fenced(stricter())])
    out = await propose_pipeline_spec(goal="stricter", registry=registry, engine=engine, author=author, base=base())
    assert out.pipeline is not None
    assert [len(r.problems) for r in out.rounds] == [2, 0]
    assert seen[1].round == 2
    assert re.search(r'unknown primitive "demo\.whisper@1"', seen[1].prompt)
    assert re.search(r"reads `bundle`.*read as `gate`", seen[1].prompt)
    assert '"demo.whisper@1"' in seen[1].prompt  # its previous spec, verbatim
    assert seen[1].system == seen[0].system


async def test_holds_the_bases_contract_and_the_hosts_vocabulary() -> None:
    drifted = stricter()
    drifted["fact"] = "shouting"
    drifted["nodes"]["persist"] = {"kind": "store", "target": "postgres", "scope": "user", "version": "1"}
    author, _ = scripted([fenced(drifted)])
    out = await propose_pipeline_spec(goal="g", registry=registry, engine=engine, author=author, base=base(), max_rounds=1)
    assert out.pipeline is None
    assert len(out.problems) == 2
    assert out.problems[0] == 'nodes.persist: unknown store target "postgres" — the host handles only "memory"'
    assert re.match(r'^fact: must stay "demo"', out.problems[1])


async def test_refuses_a_proposal_identical_to_the_base() -> None:
    author, _ = scripted([fenced(base())])
    out = await propose_pipeline_spec(goal="g", registry=registry, engine=engine, author=author, base=base(), max_rounds=1)
    assert out.problems == ["the proposal is identical to the base — change something toward the goal"]


async def test_repairs_a_reply_with_no_json_and_gives_up_after_max_rounds() -> None:
    author, seen = scripted(["I would raise the threshold.", "still no json", "nor here"])
    out = await propose_pipeline_spec(goal="g", registry=registry, engine=engine, author=author, base=base())
    assert len(out.rounds) == 3
    assert out.spec is None
    assert re.search(r"no JSON object found", out.problems[0])
    assert "(none — the reply held no JSON object)" in seen[1].prompt


async def test_without_a_base_only_the_compiler_judges() -> None:
    fresh = {**stricter(), "fact": "anything"}
    author, _ = scripted([fresh])
    out = await propose_pipeline_spec(goal="g", registry=registry, engine=engine, author=author)
    assert out.problems == []
    assert out.diff is None


# --- helpers ---


def test_extract_spec_json_prefers_the_last_fenced_block_and_reports_bad_json() -> None:
    assert extract_spec_json('```json\n{"a":1}\n```\ntext\n```json\n{"b":2}\n```') == {"value": {"b": 2}}
    assert extract_spec_json('here: {"a": 1} done') == {"value": {"a": 1}}
    assert re.search(r"does not parse", extract_spec_json("```json\n{nope}\n```")["problem"])
    assert re.search(r"not an object", extract_spec_json("```json\n[1]\n```")["problem"])


def test_spec_vocabulary_and_diff_specs() -> None:
    assert spec_vocabulary(base()) == SpecVocabulary(prompts=[], routes=[], questions=[], targets=["memory"])
    nxt = base()
    del nxt["nodes"]["check"]
    nxt["edges"] = [e for e in nxt["edges"] if e["to"] != "check" and e["from"] != "check"]
    nxt["edges"].extend([{"from": "loud", "to": "persist"}, {"from": "plain", "to": "persist"}])
    nxt["trigger"] = ["cron"]
    assert diff_specs(base(), nxt) == SpecDiff(
        nodes={"added": [], "removed": ["check"], "changed": []},
        edges={
            "added": ["loud -> persist", "plain -> persist"],
            "removed": ["loud -> check", "plain -> check", "check -> persist"],
        },
        fields=["trigger"],
    )
