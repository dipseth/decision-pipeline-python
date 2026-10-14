"""
Probes — Jev questions an LLM writes for THIS run.

A decide node's `questions` are structure: fixed in code, worded in
Langfuse (#318), the same every run. That is right for the questions a gate
thresholds on, and wrong for scrutiny — the question worth asking about a
Sichuan-Peruvian fusion recipe is not one anybody wrote down in advance.
A probe is a question an upstream `generate` node writes after looking at
the state, asked in the SAME Jev request as the static questions.

What makes that safe to put in a pipeline, and not just a prompt trick:

  validated   every probe is parsed against `probe_question_schema`, capped,
              de-duplicated, length-bounded and (with `paths`) required to
              point at a real part of the state. A bad probe is DROPPED
              with a reason on the record — it never reaches Jev.
  recorded    the probe TEXT lands on the run record beside its answer. A
              static question's text is recoverable from its prompt version;
              a probe's is not, so without this a replay cannot re-ask it.
  evidence    probe answers are never read by a gate. The runtime withholds
              them from gate bodies (`args.probes` is empty there) and from
              `interpret`, so a threshold can only ever be tuned against
              questions that exist on every run. They reach `generate` and
              non-gate `code` nodes, as evidence.
  bounded     features are aggregates (`<node>.probes.min_margin`, …), not
              one column per question — probe keys are slots (`probe_0`…),
              and slot 0 on one run means nothing on the next.

The loop back — probes whose answers track human labels becoming static
questions — is a `derived` feedback edge through a review lane, like any
other change to a pipeline's questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Mapping, NamedTuple, Optional, Sequence, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, ValidationError
from pydantic_core import PydanticCustomError

from .features import normalized_entropy, top_margin
from .types import ProbeAnswer, ProbeQuestion, ProbeSpec

__all__ = [
    "MAX_PROBES",
    "DEFAULT_PROBE_MAX_CHARS",
    "MAX_PROBE_OPTIONS",
    "PROBE_KEY_PREFIX",
    "ProbeAnswer",
    "ProbeQuestion",
    "ProbeSpec",
    "probe_key",
    "is_probe_key",
    "probe_question_schema",
    "ProbeDrop",
    "ProbeSet",
    "ProbeRecord",
    "collect_probes",
    "default_probe_options",
    "probe_answers",
    "split_probe_distributions",
    "probe_features",
]

#: Hard ceiling on probes per decide node, whatever a manifest asks for.
MAX_PROBES = 8
#: Default ceiling on a probe's text (instructions + criteria), in characters.
DEFAULT_PROBE_MAX_CHARS = 800
#: A Choice probe's option ceiling: a wide Choice is a classifier, not a probe.
MAX_PROBE_OPTIONS = 12

PROBE_KEY_PREFIX = "probe_"


def probe_key(slot: int) -> str:
    return f"{PROBE_KEY_PREFIX}{slot}"


def is_probe_key(key: str) -> bool:
    return key.startswith(PROBE_KEY_PREFIX)


# --- the schema ---------------------------------------------------------------

_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_SNAKE = re.compile(r"^[a-z0-9_]+$")


def _snake_key(key: str) -> str:
    if not _SNAKE.search(key):
        raise PydanticCustomError("snake_case_key", "option keys are snake_case")
    return key


def _choice_size(criteria: dict[str, str]) -> dict[str, str]:
    if len(criteria) < 2:
        raise PydanticCustomError("choice_min", "a choice needs at least two options")
    if len(criteria) > MAX_PROBE_OPTIONS:
        raise PydanticCustomError("choice_max", f"a choice allows at most {MAX_PROBE_OPTIONS} options")
    return criteria


def _score_size(levels: list[str]) -> list[str]:
    if len(levels) < 2:
        raise PydanticCustomError("score_min", "a score needs at least two levels")
    return levels


class _NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="ignore")
    true: _Text
    false: _Text


class _NoulProbe(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["noul"]
    instructions: _Text
    criteria: _NoulCriteria = None  # type: ignore[assignment]  # optional, but an explicit None is invalid
    #: Why the writer asked — for the hand read, never sent to Jev.
    why: str = None  # type: ignore[assignment]


class _ChoiceProbe(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["choice"]
    instructions: _Text
    criteria: Annotated[
        dict[Annotated[str, AfterValidator(_snake_key)], _Text],
        AfterValidator(_choice_size),
    ]
    why: str = None  # type: ignore[assignment]


class _ScoreProbe(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["score"]
    instructions: _Text
    criteria: Annotated[list[_Text], Field(max_length=MAX_PROBE_OPTIONS), AfterValidator(_score_size)]
    why: str = None  # type: ignore[assignment]


#: The three probe shapes, discriminated on `type`. `parse(x)` returns a model; `collect_probes`
#: turns it into the plain dict `ProbeQuestion` is.
probe_question_schema: TypeAdapter[Any] = TypeAdapter(
    Annotated[Union[_NoulProbe, _ChoiceProbe, _ScoreProbe], Field(discriminator="type")]
)

_TAGS = ("noul", "choice", "score")


def _question_dict(model: BaseModel) -> ProbeQuestion:
    # exclude_unset: an absent optional key stays absent, like zod.
    return model.model_dump(exclude_unset=True)


@dataclass(kw_only=True)
class ProbeDrop:
    #: Position in the generator's output; -1 when the output itself was unusable.
    index: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "reason": self.reason}


@dataclass(kw_only=True)
class ProbeSet:
    #: Slot key -> the probe, in the order the generator wrote them.
    asked: dict[str, ProbeQuestion]
    dropped: list[ProbeDrop]


@dataclass(kw_only=True)
class ProbeRecord:
    """How a run record keeps a decide node's probes."""

    from_: str
    asked: dict[str, ProbeQuestion]
    dropped: list[ProbeDrop]
    #: Asked, but the port returned no distribution for it.
    unanswered: list[str]

    def to_dict(self) -> dict[str, Any]:
        """The TS record shape (`from`, not `from_`)."""
        return {
            "from": self.from_,
            "asked": {k: dict(v) for k, v in self.asked.items()},
            "dropped": [d.to_dict() for d in self.dropped],
            "unanswered": list(self.unanswered),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProbeRecord":
        return cls(
            from_=d["from"] if "from" in d else d["from_"],
            asked=dict(d.get("asked") or {}),
            dropped=[x if isinstance(x, ProbeDrop) else ProbeDrop(index=x["index"], reason=x["reason"]) for x in d.get("dropped") or []],
            unanswered=list(d.get("unanswered") or []),
        )


def _js_len(s: str) -> int:
    """`String.prototype.length` counts UTF-16 code units."""
    return len(s.encode("utf-16-le")) // 2


def _probe_chars(q: ProbeQuestion) -> int:
    criteria = q.get("criteria")
    if criteria is None:
        items: list[str] = []
    elif isinstance(criteria, list):
        items = criteria
    else:
        items = list(criteria.values())
    return _js_len(q["instructions"]) + sum(_js_len(c) for c in items)


_BACKTICK = re.compile(r"`([^`]+)`")


def _backticked(s: str) -> list[str]:
    """Backticked spans in a string."""
    return [m.strip() for m in _BACKTICK.findall(s)]


def _addresses_path(instructions: str, paths: Sequence[str]) -> bool:
    return any(
        any(ref == p or ref.startswith(f"{p}.") or ref.startswith(f"{p}[") for p in paths)
        for ref in _backticked(instructions)
    )


def _normalize_for_dedupe(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _raw_list(raw: Any) -> Optional[list[Any]]:
    """The generator's output as a list: a bare array, or `{ probes }` / `{ questions }`."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("probes"), list):
            return raw["probes"]
        if isinstance(raw.get("questions"), list):
            return raw["questions"]
    return None


def _first_issue(err: ValidationError) -> str:
    issues = err.errors()
    if not issues:
        return "unparseable"
    first = issues[0]
    loc = [str(p) for p in first.get("loc", ())]
    # pydantic prefixes the matched union tag; zod's path does not have it.
    if loc and loc[0] in _TAGS:
        loc = loc[1:]
    return f"{'.'.join(loc) or '(root)'} {first.get('msg', 'invalid')}"


def collect_probes(raw: Any, spec: ProbeSpec) -> ProbeSet:
    """
    The generator's output → the probes that may be asked. Never throws: a
    generator that wrote garbage produces an empty set and a drop per item, so
    the run record says exactly what the model tried to ask and why it didn't.
    """
    asked: dict[str, ProbeQuestion] = {}
    dropped: list[ProbeDrop] = []
    items = _raw_list(raw)
    if items is None:
        dropped.append(ProbeDrop(index=-1, reason="generator output is not a list of probes"))
        return ProbeSet(asked=asked, dropped=dropped)
    max_ = min(spec.max, MAX_PROBES)
    max_chars = DEFAULT_PROBE_MAX_CHARS if spec.max_chars is None else spec.max_chars
    seen: set[str] = set()
    slot = 0

    for index, item in enumerate(items):
        try:
            q = _question_dict(probe_question_schema.validate_python(item))
        except ValidationError as err:
            dropped.append(ProbeDrop(index=index, reason=f"invalid: {_first_issue(err)}"))
            continue
        chars = _probe_chars(q)
        if chars > max_chars:
            dropped.append(ProbeDrop(index=index, reason=f"too long: {chars} > {max_chars} chars"))
            continue
        if spec.paths and not _addresses_path(q["instructions"], spec.paths):
            dropped.append(
                ProbeDrop(index=index, reason=f"addresses no state path (want one of: {', '.join(spec.paths)})")
            )
            continue
        norm = _normalize_for_dedupe(q["instructions"])
        if norm in seen:
            dropped.append(ProbeDrop(index=index, reason="duplicate"))
            continue
        if slot >= max_:
            dropped.append(ProbeDrop(index=index, reason=f"over max ({max_})"))
            continue
        seen.add(norm)
        asked[probe_key(slot)] = q
        slot += 1

    return ProbeSet(asked=asked, dropped=dropped)


def default_probe_options(q: ProbeQuestion) -> list[str]:
    """The option keys behind a probe's distribution when the port did not say."""
    t = q["type"]
    if t == "noul":
        return ["true"]
    if t == "choice":
        return list(q["criteria"].keys())
    return [str(i) for i, _ in enumerate(q["criteria"])]


class ProbeAnswers(NamedTuple):
    answers: list[ProbeAnswer]
    unanswered: list[str]


def probe_answers(
    asked: Mapping[str, ProbeQuestion],
    distributions: Mapping[str, list[float]],
    options: Optional[Mapping[str, list[str]]] = None,
) -> ProbeAnswers:
    """Pair each asked probe with its answer. Unanswered ones are listed, not invented. Unpacks as `(answers, unanswered)`."""
    answers: list[ProbeAnswer] = []
    unanswered: list[str] = []
    for key, question in asked.items():
        distribution = distributions.get(key)
        if not distribution:
            unanswered.append(key)
            continue
        opts = options.get(key) if options else None
        answers.append(
            ProbeAnswer(
                key=key,
                question=question,
                distribution=distribution,
                options=default_probe_options(question) if opts is None else opts,
            )
        )
    return ProbeAnswers(answers=answers, unanswered=unanswered)


class ProbeSplit(NamedTuple):
    fixed: dict[str, list[float]]
    probes: dict[str, list[float]]


def split_probe_distributions(distributions: Mapping[str, list[float]]) -> ProbeSplit:
    """Split a decide result's distributions into the static questions and the probes. Unpacks as `(fixed, probes)`."""
    fixed: dict[str, list[float]] = {}
    probes: dict[str, list[float]] = {}
    for key, p in distributions.items():
        (probes if is_probe_key(key) else fixed)[key] = p
    return ProbeSplit(fixed=fixed, probes=probes)


def probe_features(node_id: str, record: ProbeRecord, answers: Sequence[ProbeAnswer]) -> dict[str, float]:
    """
    Aggregates only — a probe slot means something different every run, so a
    per-slot column would be noise to any model trained on the record.

    `min_margin` is the one to watch: the least decided probe is where the
    writer found something the state does not settle.
    """
    out: dict[str, float] = {
        f"{node_id}.probes.asked": len(record.asked),
        f"{node_id}.probes.answered": len(answers),
        f"{node_id}.probes.dropped": len(record.dropped),
    }
    margins: list[float] = []
    entropies: list[float] = []
    for a in answers:
        # A Noul is one number: its "margin" is its distance from a coin flip.
        m = abs(2 * a.distribution[0] - 1) if len(a.distribution) == 1 else top_margin(a.distribution)
        if m is not None:
            margins.append(m)
        if len(a.distribution) > 1:
            h = normalized_entropy(a.distribution)
            if h is not None:
                entropies.append(h)
    if margins:
        out[f"{node_id}.probes.min_margin"] = min(margins)
    if entropies:
        out[f"{node_id}.probes.max_entropy"] = max(entropies)
        out[f"{node_id}.probes.mean_entropy"] = sum(entropies) / len(entropies)
    return out
