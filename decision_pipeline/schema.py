"""
Pydantic over the DECLARATIVE half of a manifest.

The other half — `run` bodies, `group()`, `key()` — is real code and always
will be, so it is checked structurally in define.py instead. The plain data
gets the same treatment a YAML file would have got, without a string->function
registry standing between an author and their own code.

Field names here are snake_case with camelCase aliases, so the same models
validate a dataclass manifest (define.py) and a JSON spec (spec.py).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ._compat import pydantic_problems
from .probes import MAX_PROBES
from .types import RANKS, SCOPES

ScopeLiteral = Literal["session", "device", "user", "associated", "tenant", "cohort", "global"]
RankLiteral = Literal["heuristic", "llm", "human", "admin"]
TriggerLiteral = Literal["ingest", "cron", "on-demand"]
OnFailureLiteral = Literal["fallback", "skip", "revert", "fail"]
CacheLiteral = Literal["none", "per-run", "per-key-forever"]
NodeKindLiteral = Literal["read", "decide", "generate", "code", "store"]
CodeRoleLiteral = Literal["transform", "validate", "derive", "gate"]
FeedbackFormLiteral = Literal["override", "feature", "queue", "derived", "score"]

assert set(SCOPES) == set(ScopeLiteral.__args__) and set(RANKS) == set(RankLiteral.__args__)  # type: ignore[attr-defined]

NonEmpty = Field(min_length=1)


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class _Strict(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ThresholdEnv(_Model):
    env: str = NonEmpty
    default: float


class ThresholdPromptConfig(_Model):
    prompt_config: str = Field(alias="promptConfig", min_length=1)
    default: float


ThresholdSchema = Union[float, ThresholdEnv, ThresholdPromptConfig]
threshold_schema: TypeAdapter[Any] = TypeAdapter(ThresholdSchema)


class ToolSelectableSchema(_Model):
    candidates: list[str] = Field(min_length=1)
    from_: str = Field(alias="from", min_length=1)
    gate: str = NonEmpty
    threshold: Optional[ThresholdSchema] = None


class ToolSpecSchema(_Model):
    static: Union[list[str], Literal["prompt-config"], None] = None
    selectable: Optional[ToolSelectableSchema] = None
    allow: Optional[list[str]] = None


class EdgeGuardSchema(_Model):
    gate: str = NonEmpty
    branch: str = NonEmpty


class EdgeSchema(_Model):
    from_: str = Field(alias="from", min_length=1)
    to: str = NonEmpty
    when: Optional[EdgeGuardSchema] = None


class ScoreCategorySchema(_Strict):
    label: str = NonEmpty
    value: float


class ScoreConfigSchema(_Strict):
    name: str = NonEmpty
    data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = Field(alias="dataType")
    categories: Optional[list[ScoreCategorySchema]] = None
    description: Optional[str] = None


class FeedbackEdgeSchema(_Model):
    id: str = NonEmpty
    from_: str = Field(alias="from", min_length=1)
    to: str = NonEmpty
    source: Literal["explicit", "implicit"]
    scope: ScopeLiteral
    form: FeedbackFormLiteral
    latency: Literal["immediate", "next-run", "deferred"]
    promote_via: Optional[str] = Field(default=None, alias="promoteVia", min_length=1)
    score: Optional[ScoreConfigSchema] = None
    #: A function; not validated beyond "present or not".
    key: Any = None


class ProbeSpecSchema(_Model):
    from_: str = Field(alias="from", min_length=1)
    max: int = Field(ge=1, le=MAX_PROBES)
    paths: Optional[list[str]] = Field(default=None, min_length=1)
    max_chars: Optional[int] = Field(default=None, alias="maxChars", ge=40)


class NodeDeclarationSchema(BaseModel):
    """The parts of a node pydantic can see. Bodies are checked in define.py."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    kind: NodeKindLiteral
    version: str = NonEmpty
    inputs: Optional[dict[str, str]] = None
    on_failure: Optional[OnFailureLiteral] = Field(default=None, alias="onFailure")
    timeout_ms: Optional[int] = Field(default=None, alias="timeoutMs", gt=0)
    cache: Optional[CacheLiteral] = None
    describe: Optional[str] = None
    # read
    provides_features: Optional[bool] = Field(default=None, alias="providesFeatures")
    # decide
    questions: Optional[str] = Field(default=None, min_length=1)
    state: Optional[list[str]] = None
    accepts_features: Optional[bool] = Field(default=None, alias="acceptsFeatures")
    batched: Optional[bool] = None
    probes: Optional[ProbeSpecSchema] = None
    # generate
    prompt: Optional[str] = Field(default=None, min_length=1)
    route: Optional[str] = Field(default=None, min_length=1)
    submit: Optional[str] = Field(default=None, min_length=1)
    tools: Optional[ToolSpecSchema] = None
    # code
    role: Optional[CodeRoleLiteral] = None
    thresholds: Optional[dict[str, ThresholdSchema]] = None
    branches: Optional[list[str]] = None
    # store
    target: Optional[str] = Field(default=None, min_length=1)
    scope: Optional[ScopeLiteral] = None
    ttl: Optional[str] = Field(default=None, min_length=1)


class EvalSchema(_Model):
    dataset: str = NonEmpty
    gate_metric: str = Field(alias="gateMetric", min_length=1)


class GuardPatchSchema(_Model):
    from_: str = Field(alias="from", min_length=1)
    to: str = NonEmpty
    when: Optional[EdgeGuardSchema] = None


class ManifestPatchSchema(_Model):
    thresholds: Optional[dict[str, dict[str, float]]] = None
    prompts: Optional[dict[str, str]] = None
    questions: Optional[dict[str, str]] = None
    routes: Optional[dict[str, str]] = None
    tools: Optional[dict[str, ToolSpecSchema]] = None
    guards: Optional[list[GuardPatchSchema]] = None


class ExperimentSchema(_Model):
    id: str = NonEmpty
    unit: ScopeLiteral
    mode: Literal["replay", "replay-redecide", "shadow", "split"]
    champion: str = NonEmpty
    arms: dict[str, ManifestPatchSchema]
    weights: Optional[dict[str, float]] = None
    enabled: Optional[bool] = None


class OverridableSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    scope: ScopeLiteral


class PrecedenceSchema(_Model):
    over: list[str]


class PipelineDeclarationSchema(_Model):
    """Everything pydantic can see about a manifest, in one schema."""

    id: str = NonEmpty
    fact: str = NonEmpty
    trigger: list[TriggerLiteral] = Field(min_length=1)
    nodes: dict[str, NodeDeclarationSchema]
    edges: list[EdgeSchema]
    feedback: list[FeedbackEdgeSchema]
    result: Union[str, list[str]]
    overridable: Optional[OverridableSchema] = None
    eval: EvalSchema
    experiments: Optional[list[ExperimentSchema]] = None
    decided_by: Optional[RankLiteral] = Field(default=None, alias="decidedBy")
    precedence: Optional[PrecedenceSchema] = None
    knowledge_version: Optional[str] = Field(default=None, alias="knowledgeVersion", min_length=1)


pipeline_declaration_schema: TypeAdapter[Any] = TypeAdapter(PipelineDeclarationSchema)


def declaration_problems(declaration: Any) -> list[str]:
    """Validate a declaration (dict) and return the flattened problems, empty when it parses."""
    try:
        pipeline_declaration_schema.validate_python(declaration)
        return []
    except ValidationError as err:
        return pydantic_problems(err)


def plain(value: Any) -> Any:
    """Dataclasses (and nested ones) → dicts with their field names, so pydantic can validate them. Callables pass through."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


# The TS name, for readers coming from the zod version.
zod_problems = pydantic_problems
