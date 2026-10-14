"""
A decision pipeline explained in plain words, from its own JSON (#321).

The diagram shows WHAT a pipeline is; this says what it MEANS, for a reader
who picked an audience on a scale — a five-year-old, a ten-year-old, a
fifteen-year-old, an engineer. Nothing here calls a model: `explain_prompt`
turns `manifest.to_json()` into a brief and the instructions for one level,
and the host supplies the model through the `generate` port of
`explain_pipeline`, the same way every other side effect is a port.

An explanation is worth caching per VERSION of the pipeline, not per
request: `explain_version` hashes the declarative half that changes what the
pipeline does (nodes, edges, gates, feedback), and ignores the operational
knobs that do not (timeouts, cache policy, eval dataset). A host keys its
store on `explain_cache_key`: pipeline · version · level · language, plus
whatever prompt version it resolved. The key is the same one the TypeScript
package computes for the same manifest.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

from ._compat import js_number, maybe_await
from .diagram import _get, _js_keys, _js_str, _truthy
from .hash import short_hash

__all__ = [
    "EXPLAIN_LEVELS",
    "EXPLAIN_SYSTEM_PROMPT",
    "ExplainPrompt",
    "Explanation",
    "LevelBrief",
    "explain_brief",
    "explain_cache_key",
    "explain_level_brief",
    "explain_pipeline",
    "explain_prompt",
    "explain_user_prompt",
    "explain_version",
    "is_explain_level",
]

# ---------------------------------------------------------------------------
# Levels — a scale, in order from simplest to fullest
# ---------------------------------------------------------------------------

EXPLAIN_LEVELS: tuple[str, ...] = ("eli5", "eli10", "eli15", "expert")


def is_explain_level(value: Any) -> bool:
    return isinstance(value, str) and value in EXPLAIN_LEVELS


@dataclass(frozen=True, kw_only=True)
class LevelBrief:
    #: Who is reading, in the second person for the model.
    audience: str
    #: How much to write.
    length: str
    #: What to keep and what to leave out at this level.
    rules: str


_LEVELS: dict[str, LevelBrief] = {
    "eli5": LevelBrief(
        audience="a curious five-year-old who has never used a computer",
        length="Three or four short sentences, one idea each.",
        rules=(
            "Use one everyday comparison (a kitchen, a game, a helper) and keep it the whole way through. "
            "No names of steps, no technical words, no numbers. Say what goes in, what comes out, and the one place where it has to choose."
        ),
    ),
    "eli10": LevelBrief(
        audience="a sharp ten-year-old who likes knowing how things work",
        length="One short paragraph, then a list of at most five steps in order.",
        rules=(
            "Plain words. You may name the steps in your own words, never by their identifiers. "
            "Explain each choice the pipeline makes as a question it asks itself. Mention that a person can correct it afterwards if the pipeline has a feedback surface."
        ),
    ),
    "eli15": LevelBrief(
        audience="a fifteen-year-old who has written a little code",
        length="Two or three short paragraphs and a step list.",
        rules=(
            "Distinguish the steps a model does from the steps plain code does, and say why that split exists. "
            "Name the gates and what each branch means. Say what happens when a step fails. Identifiers may appear in parentheses after the plain name."
        ),
    ),
    "expert": LevelBrief(
        audience="an engineer who will maintain this pipeline",
        length="As long as it needs to be, and no longer: headed sections, dense, no filler.",
        rules=(
            "Use the identifiers. Cover: the fact it establishes and what triggers it; every node in order with its kind, role, and failure policy; "
            "each gate's branches and the thresholds that decide them; the feedback edges and what they change; what is stored, where, at what scope. "
            "Point out anything a maintainer would want to know before changing it — ordering constraints, where a model is in the loop, what is reversible."
        ),
    ),
}


def explain_level_brief(level: str) -> LevelBrief:
    return _LEVELS[level]


# ---------------------------------------------------------------------------
# Version — what an explanation is ABOUT
# ---------------------------------------------------------------------------

# The declarative half of a node that changes its meaning. Knobs that do not are left out.
_MEANING_KEYS: tuple[str, ...] = (
    "id",
    "kind",
    "role",
    "describe",
    "version",
    "onFailure",
    "questions",
    "probes",
    "prompt",
    "route",
    "target",
    "scope",
    "branches",
    "thresholds",
    "providesFeatures",
    "acceptsFeatures",
    "batched",
    "submit",
    "tools",
    "inputs",
)


def _meaning_of_node(node: Mapping[str, Any]) -> dict[str, Any]:
    # TS keeps a key unless it is `undefined`. A Python `to_json()` writes None where
    # TS leaves a key out, so None counts as absent here — that keeps the hash equal
    # for the same manifest exported by either package.
    return {k: node[k] for k in _MEANING_KEYS if node.get(k) is not None}


def _or(value: Any, default: Any) -> Any:
    """TS `value ?? default`."""
    return default if value is None else value


def explain_version(pipeline: Mapping[str, Any]) -> str:
    """
    A short hash of what the pipeline DOES. Two manifests with the same version
    deserve the same explanation; a node version bump, a new branch, a changed
    threshold or a new feedback edge produce a new one. Timeouts, cache policy,
    eval dataset and experiments do not.
    """
    return short_hash(
        {
            "id": pipeline["id"],
            "fact": _or(pipeline.get("fact"), None),
            "trigger": _or(pipeline.get("trigger"), []),
            "result": _or(pipeline.get("result"), []),
            "decidedBy": _or(pipeline.get("decidedBy"), None),
            "nodes": [_meaning_of_node(n) for n in pipeline["nodes"]],
            "edges": [{"from": e["from"], "to": e["to"], "when": _or(e.get("when"), None)} for e in pipeline["edges"]],
            "feedback": [
                {
                    "from": f["from"],
                    "to": f["to"],
                    "form": _or(f.get("form"), None),
                    "promoteVia": _or(f.get("promoteVia"), None),
                }
                for f in _or(pipeline.get("feedback"), [])
            ],
        }
    )


def explain_cache_key(
    pipeline: Mapping[str, Any],
    *,
    level: str,
    language: Optional[str] = None,
    prompt_version: Union[str, int, float, None] = None,
) -> str:
    """`pipeline:version:level:language[:prompt]` — stable, readable in a table."""
    lang = _or(language, "en").lower().split("-")[0] or "en"
    base = f"{pipeline['id']}:{explain_version(pipeline)}:{level}:{lang}"
    return base if prompt_version is None else f"{base}:p{_js_str(prompt_version)}"


# ---------------------------------------------------------------------------
# The brief — the manifest, written out for a model
# ---------------------------------------------------------------------------


def _json_stringify(value: Any) -> str:
    """`JSON.stringify(value)` for JSON-like data: compact, JS numbers, JS key order."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return js_number(value) if not isinstance(value, float) or math.isfinite(value) else "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_json_stringify(v) for v in value) + "]"
    if isinstance(value, Mapping):
        keys = _js_keys(value)
        items = [(k, value[k] if k in value else value[int(k)]) for k in keys]
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_json_stringify(v)}" for k, v in items) + "}"
    return json.dumps(str(value), ensure_ascii=False)


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else _json_stringify(value)


def _node_line(node: Mapping[str, Any], branch_in: Optional[str]) -> str:
    g = node.get
    role = g("role")
    parts = [f"- {_js_str(g('id'))} · {_js_str(g('kind'))}{'/' + _js_str(role) if _truthy(role) else ''}"]
    if branch_in:
        parts.append(f'(reached on branch "{branch_in}")')
    if _truthy(g("describe")):
        parts.append(f"— {_js_str(g('describe'))}")
    facts: list[str] = []
    kind = g("kind")
    if kind == "decide" and _truthy(g("questions")):
        facts.append(f'asks a fast classifier the "{_js_str(g("questions"))}" questions')
    if kind == "decide" and _truthy(g("probes")):
        facts.append(f"extra questions written by {_js_str(_get(g('probes'), 'from'))}")
    if kind == "generate" and _truthy(g("prompt")):
        facts.append(f'a writer model on prompt "{_js_str(g("prompt"))}"')
    branches = g("branches")
    if role == "gate" and branches:
        facts.append(f"branches: {' | '.join(_js_str(b) for b in branches)}")
    thresholds = g("thresholds")
    if isinstance(thresholds, Mapping) and len(thresholds) > 0:
        items = []
        for k in _js_keys(thresholds):
            spec = thresholds[k] if k in thresholds else thresholds[int(k)]
            d = spec["default"] if isinstance(spec, Mapping) and "default" in spec else spec
            items.append(f"{k}={_as_text(d)}")
        facts.append(f"thresholds: {', '.join(items)}")
    if kind == "store":
        target = g("target")
        scope = g("scope")
        facts.append(f"stores to {'?' if target is None else _js_str(target)}{f' at {_js_str(scope)} scope' if _truthy(scope) else ''}")
    on_failure = g("onFailure")
    if _truthy(on_failure) and on_failure != "fail":
        facts.append(f"on failure: {_js_str(on_failure)}")
    if facts:
        parts.append(f"[{'; '.join(facts)}]")
    return " ".join(parts)


def _feedback_line(f: Mapping[str, Any]) -> str:
    pv = f.get("promoteVia")
    if isinstance(pv, str):
        via: Optional[str] = pv
    elif isinstance(pv, Mapping):
        queue = pv.get("queue")
        via = f"{_js_str(pv.get('lane'))}{f' ({_js_str(queue)})' if _truthy(queue) else ''}"
    else:
        via = None
    form = f.get("form")
    return f"- \"{_js_str(f.get('from'))}\" → {_js_str(f.get('to'))}{f' as a {_js_str(form)}' if _truthy(form) else ''}{f', promoted via {via}' if via else ''}"


def explain_brief(pipeline: Mapping[str, Any], *, title: Optional[str] = None, summary: Optional[str] = None) -> str:
    """
    The pipeline written out as a brief: what it establishes, when it runs,
    every node in topological order with the branch that reaches it, and the
    feedback edges. The same brief feeds every level; the level changes the
    instructions, not the facts.
    """
    by_id: dict[str, Mapping[str, Any]] = {}
    for n in pipeline["nodes"]:
        by_id[_js_str(n.get("id"))] = n
    order = pipeline.get("order")
    if order is None:
        order = list(by_id.keys())
    branch_into: dict[str, str] = {}
    for e in pipeline["edges"]:
        br = (e.get("when") or {}).get("branch")
        if _truthy(br):
            branch_into[e["to"]] = _js_str(br)
    flow = []
    for e in pipeline["edges"]:
        br = (e.get("when") or {}).get("branch")
        flow.append(f"{e['from']} → {e['to']}{f' [{br}]' if _truthy(br) else ''}")

    lines: list[str] = []
    lines.append(f'Pipeline: {_or(title, pipeline["id"])} (id "{pipeline["id"]}")')
    if summary:
        lines.append(f"One-line summary: {summary}")
    if _truthy(pipeline.get("fact")):
        lines.append(f"Establishes the fact: {pipeline['fact']}")
    if pipeline.get("trigger"):
        lines.append(f"Runs: {', '.join(pipeline['trigger'])}")
    if pipeline.get("result"):
        lines.append(f"Produces: {', '.join(pipeline['result'])}")
    if _truthy(pipeline.get("decidedBy")):
        lines.append(f"Highest-ranked decider: {pipeline['decidedBy']}")
    lines += ["", "Steps, in order:"]
    for nid in order:
        node = by_id.get(nid)
        if node is not None:
            lines.append(_node_line(node, branch_into.get(nid)))
    lines += ["", "Flow:", *(f"- {f}" for f in flow)]
    feedback = pipeline.get("feedback")
    if feedback:
        lines += ["", "Feedback from people, after a run:", *(_feedback_line(f) for f in feedback)]
    else:
        lines += ["", "Feedback from people: none — the run is final once stored."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

#: The system half. A host may manage its own copy (a prompt store) and pass
#: that version into `explain_cache_key`; this is the default.
EXPLAIN_SYSTEM_PROMPT = """You explain how an automated decision pipeline works to a reader at a chosen level.

A pipeline is a fixed sequence of steps. Some steps are plain code, some ask a small fast classifier a yes/no question, some ask a writing model to produce text, some are gates where code picks a branch, and one usually stores the result. People may send feedback afterwards, which changes future runs — a run never waits on a person.

You are given the pipeline as a brief and a level. Write for exactly that reader, in Markdown, with no preamble and no sign-off. Never invent a step, a threshold or a feedback path that is not in the brief; if the brief does not say, do not say. Prefer concrete verbs ("checks", "picks", "writes") to abstractions. Write in the language you are asked for."""


def explain_user_prompt(
    pipeline: Mapping[str, Any],
    *,
    level: str,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    language_name: Optional[str] = None,
) -> str:
    """The user half: the brief plus the level's instructions."""
    brief = _LEVELS[level]
    return "\n".join(
        [
            f"Reader: {brief.audience}.",
            f"Length: {brief.length}",
            f"Rules: {brief.rules}",
            f"Language: {_or(language_name, 'English')}.",
            "",
            "Brief:",
            explain_brief(pipeline, title=title, summary=summary),
        ]
    )


@dataclass(kw_only=True)
class ExplainPrompt:
    system: str
    user: str


def explain_prompt(
    pipeline: Mapping[str, Any],
    *,
    level: str,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    language_name: Optional[str] = None,
) -> ExplainPrompt:
    return ExplainPrompt(
        system=EXPLAIN_SYSTEM_PROMPT,
        user=explain_user_prompt(pipeline, level=level, title=title, summary=summary, language_name=language_name),
    )


# ---------------------------------------------------------------------------
# The port
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Explanation:
    pipeline: str
    level: str
    version: str
    #: Markdown.
    text: str


#: The host's model call: given system and user text, return the model's text (sync or async).
ExplainGenerate = Callable[[ExplainPrompt], Union[str, Awaitable[str]]]


async def explain_pipeline(
    pipeline: Mapping[str, Any],
    *,
    level: str,
    generate: ExplainGenerate,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    language_name: Optional[str] = None,
) -> Explanation:
    """
    Explain a pipeline at one level through the host's model. Pure apart from
    the port: the same inputs produce the same prompt, so a host can cache by
    `explain_cache_key` and never ask twice for one version.
    """
    prompt = explain_prompt(pipeline, level=level, title=title, summary=summary, language_name=language_name)
    text = (await maybe_await(generate(prompt))).strip()
    return Explanation(pipeline=pipeline["id"], level=level, version=explain_version(pipeline), text=text)
