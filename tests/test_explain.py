import copy

from decision_pipeline.explain import (
    EXPLAIN_LEVELS,
    Explanation,
    explain_brief,
    explain_cache_key,
    explain_pipeline,
    explain_prompt,
    explain_version,
    is_explain_level,
)

scale = {
    "id": "scale",
    "fact": "scaled_recipe",
    "trigger": ["on-demand"],
    "result": ["changes"],
    "decidedBy": "llm",
    "nodes": [
        {"id": "classify", "kind": "decide", "questions": "jev-scale-classify", "onFailure": "fallback", "version": "3", "timeoutMs": 4000},
        {"id": "gate", "kind": "code", "role": "gate", "branches": ["direct", "writer"], "thresholds": {"direct_line": {"env": "X", "default": 0.8}}, "version": "1"},
        {"id": "direct", "kind": "code", "role": "derive", "describe": "Arithmetic on the table", "version": "1"},
        {"id": "writer", "kind": "generate", "prompt": "recipe-scale", "route": "scale-writer", "version": "7"},
        {"id": "persist", "kind": "store", "target": "turso", "scope": "tenant", "version": "1"},
    ],
    "edges": [
        {"from": "classify", "to": "gate"},
        {"from": "gate", "to": "direct", "when": {"gate": "gate", "branch": "direct"}},
        {"from": "gate", "to": "writer", "when": {"gate": "gate", "branch": "writer"}},
        {"from": "direct", "to": "persist"},
        {"from": "writer", "to": "persist"},
    ],
    "feedback": [
        {"id": "rating", "from": "scaling-notes", "to": "gate", "source": "explicit", "scope": "tenant", "form": "score", "latency": "next-run"},
    ],
    "order": ["classify", "gate", "direct", "writer", "persist"],
}


def clone():
    return copy.deepcopy(scale)


# --- explain_version


def test_is_stable_across_calls_and_ignores_operational_knobs():
    a = clone()
    b = clone()
    b["nodes"][0]["timeoutMs"] = 9000
    b["nodes"][0]["cache"] = "run"
    assert explain_version(a) == explain_version(scale)
    assert explain_version(b) == explain_version(scale)


def test_changes_when_what_the_pipeline_does_changes():
    describe = clone()
    describe["nodes"][2]["describe"] = "Different words"
    threshold = clone()
    threshold["nodes"][1]["thresholds"] = {"direct_line": {"env": "X", "default": 0.5}}
    bump = clone()
    bump["nodes"][3]["version"] = "8"
    no_feedback = clone()
    no_feedback["feedback"] = []
    base = explain_version(scale)
    for changed in [describe, threshold, bump, no_feedback]:
        assert explain_version(changed) != base


# --- explain_cache_key


def test_names_pipeline_version_level_and_language_with_the_prompt_version_when_given():
    v = explain_version(scale)
    assert explain_cache_key(scale, level="eli5") == f"scale:{v}:eli5:en"
    assert explain_cache_key(scale, level="expert", language="pt-BR", prompt_version=4) == f"scale:{v}:expert:pt:p4"


# --- explain_brief


def test_walks_every_node_in_order_with_its_branch_and_the_feedback():
    brief = explain_brief(scale, title="Scale", summary="Resizes a recipe.")
    at = brief.index
    assert 'Pipeline: Scale (id "scale")' in brief
    assert "Establishes the fact: scaled_recipe" in brief
    for nid in scale.get("order") or []:
        assert f"- {nid} ·" in brief
    assert at("- classify") < at("- gate")
    assert at("- gate") < at("- persist")
    assert '(reached on branch "writer")' in brief
    assert "branches: direct | writer" in brief
    assert "thresholds: direct_line=0.8" in brief
    assert "on failure: fallback" in brief
    assert "stores to turso at tenant scope" in brief
    assert '"scaling-notes" → gate as a score' in brief


def test_says_so_when_there_is_no_feedback():
    none = clone()
    none["feedback"] = []
    assert "Feedback from people: none" in explain_brief(none)


# --- explain_prompt


def test_has_one_brief_for_every_level_and_level_specific_instructions():
    prompts = [explain_prompt(scale, level=level) for level in EXPLAIN_LEVELS]
    briefs = {p.user[p.user.index("Brief:") :] for p in prompts}
    assert len(briefs) == 1
    assert "five-year-old" in prompts[0].user
    assert "engineer" in prompts[3].user
    assert all(len(p.system) > 0 for p in prompts)


def test_asks_for_the_language_by_name():
    assert "Language: Spanish." in explain_prompt(scale, level="eli10", language_name="Spanish").user


# --- explain_pipeline


async def test_runs_the_hosts_generate_port_once_and_labels_the_result():
    seen: list[str] = []

    async def generate(p):
        seen.append(p.user)
        return "  It checks, then picks.  "

    result = await explain_pipeline(scale, level="eli15", generate=generate)
    assert len(seen) == 1
    assert result == Explanation(pipeline="scale", level="eli15", version=explain_version(scale), text="It checks, then picks.")


# --- is_explain_level


def test_accepts_the_four_levels_only():
    assert all(is_explain_level(level) for level in EXPLAIN_LEVELS)
    assert is_explain_level("eli99") is False
    assert is_explain_level(None) is False
