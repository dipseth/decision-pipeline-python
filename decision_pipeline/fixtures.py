"""
The doc's worked example — scale — as a runnable manifest with fake bodies,
plus a small JSON spec over a demo registry. The same shapes the TS package's
`test-fixtures.ts` has, so a test reads the same either side.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .define import DecisionPipeline, EvalSpec, PipelineSpec, define_pipeline
from .review import PIPELINE_REVIEW_QUEUE, define_review_lane
from .spec import define_primitive, define_registry
from .types import (
    ChangeRow,
    CodeNode,
    DecideNode,
    Edge,
    EdgeGuard,
    FeedbackEdge,
    GateNode,
    GenerateNode,
    NodeRunArgs,
    ScoreConfigSpec,
    StoreNode,
    ToolSelectable,
    ToolSpec,
)


class ScaleInput(BaseModel):
    recipe_id: str = Field(alias="recipeId")
    factor: float
    notes: str = ""
    substitution_count: float = Field(default=0, alias="substitutionCount")
    ingredients: list[str]
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Change(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    kind: str
    subject: str
    before: Any = None
    after: Any = None
    reason: str
    node_id: str = Field(alias="nodeId")


class ScaledRecipe(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    scaled_ingredients: list[str] = Field(alias="scaledIngredients")
    scaling_notes: list[str] = Field(alias="scalingNotes")
    factor: float
    changes: list[Change]


def interpret_classify(distributions: dict[str, list[float]], args: NodeRunArgs) -> dict[str, Any]:
    """`decide` returns distributions; `interpret` turns them into what the table needs."""
    lines = []
    for index, _ in enumerate(args.input.ingredients):
        p = distributions.get(f"line:{index}", [0])
        p0 = p[0] if p else 0
        lines.append({"index": index, "category": "linear" if p0 >= 0.5 else "structural", "confidence": p0})
    dish = distributions.get("dish") or [0]
    return {"lines": lines, "dishConfidence": dish[0] if dish else 0}


def build_scaling_table(args: NodeRunArgs) -> list[dict[str, Any]]:
    classification = args.primary
    rows = []
    for index, original in enumerate(args.input.ingredients):
        line = classification["lines"][index] if index < len(classification["lines"]) else None
        rows.append(
            {
                "index": index,
                "original": original,
                "linear": f"{_num(args.input.factor)}x {original}",
                "category": line["category"] if line else "structural",
                "confidence": line["confidence"] if line else 0,
            }
        )
    return rows


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def direct_gate(args: NodeRunArgs) -> dict[str, Any]:
    """The gate: code, never a model. Reads resolved NUMBERS, never the environment."""
    rows = args.primary
    low = 0 if len(rows) == 0 else min(r["confidence"] for r in rows)
    gate = args.thresholds.get("category", 0.7)
    if args.input.notes.strip() != "":
        return {"branch": "writer", "reason": "notes", "confidence": low}
    if args.input.substitution_count > 0:
        return {"branch": "writer", "reason": "substitutions", "confidence": low}
    return {"branch": "direct", "confidence": low} if low >= gate else {"branch": "writer", "reason": "low_line_confidence", "confidence": low}


def direct_scaled_recipe(args: NodeRunArgs) -> dict[str, Any]:
    rows = args.primary
    return {
        "scaledIngredients": [r["linear"] for r in rows],
        "scalingNotes": [f"{r['original']}: scaled {r['category']}" for r in rows],
        "factor": args.input.factor,
        "changes": [],
    }


def _field(obj: Any, name: str, alias: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(alias, obj.get(name))
    return getattr(obj, name)


def scale_guard(args: NodeRunArgs) -> Any:
    written = args.primary
    if len(_field(written, "scaled_ingredients", "scaledIngredients")) != len(args.input.ingredients):
        raise Exception("writer dropped a line")
    return written


def scale_changes(args: NodeRunArgs) -> dict[str, Any]:
    scaled = args.primary
    scaled_ingredients = _field(scaled, "scaled_ingredients", "scaledIngredients")
    changes = [
        {
            "kind": "scaled",
            "subject": args.input.ingredients[index] if index < len(args.input.ingredients) else "",
            "before": args.input.ingredients[index] if index < len(args.input.ingredients) else "",
            "after": after,
            "reason": f"factor {_num(args.input.factor)}",
            "nodeId": "changes",
        }
        for index, after in enumerate(scaled_ingredients)
    ]
    base = scaled if isinstance(scaled, dict) else scaled.model_dump(by_alias=True)
    return {**base, "changes": changes}


#: A lane, not a queue: it lands in the shared `pipeline-review` queue.
SCALE_LINE_LANE = define_review_lane(
    lane="scale-line-review",
    queue=PIPELINE_REVIEW_QUEUE,
    score_config={
        "name": "scale_line_verdict",
        "dataType": "CATEGORICAL",
        "categories": [{"label": "wrong", "value": 0}, {"label": "arguable", "value": 1}, {"label": "fine", "value": 2}],
        "description": "A line a cook flagged on the scaling notes panel. `wrong` = the category or the rounding is wrong for this ingredient; `arguable` = defensible but not what a cook would do; `fine` = the flag was a misread.",
    },
)


def build_scale_pipeline(experiments: Optional[list[Any]] = None) -> DecisionPipeline:
    return define_pipeline(
        PipelineSpec(
            id="scale",
            fact="scaled_recipe",
            input=ScaleInput,
            output=ScaledRecipe,
            trigger=["on-demand"],
            group=lambda i: f"cook-{i.recipe_id}",
            result=["changes"],
            decided_by="llm",
            nodes={
                "classify": DecideNode(questions="jev-scale-classify", state=["scale-categories"], accepts_features=True, interpret=interpret_classify, version="3"),
                "table": CodeNode(role="transform", run=build_scaling_table, version="2"),
                "gate": GateNode(
                    run=direct_gate,
                    branches=["direct", "writer"],
                    thresholds={"category": {"env": "RECIPES_SCALE_CATEGORY_GATE", "default": 0.7}, "tools": 0.5},
                    version="1",
                ),
                "direct": CodeNode(role="derive", run=direct_scaled_recipe, version="1"),
                "writer": GenerateNode(
                    prompt="recipe-scale",
                    route="scale-writer",
                    submit="submit_scaled_recipe",
                    tools=ToolSpec(
                        static="prompt-config",
                        selectable=ToolSelectable(candidates=["lookup_pan_conversion", "lookup_cooking_adjustment"], from_="classify", gate="gate"),
                        allow=["lookup_pan_conversion", "lookup_cooking_adjustment", "lookup_scaling_rule"],
                    ),
                    version="7",
                ),
                "guard": CodeNode(role="validate", run=scale_guard, on_failure="revert", version="1"),
                "changes": CodeNode(role="derive", run=scale_changes, version="2"),
                "persist": StoreNode(target="qdrant", scope="tenant", version="1"),
            },
            edges=[
                Edge(from_="classify", to="table"),
                Edge(from_="table", to="gate"),
                Edge(from_="gate", to="direct", when=EdgeGuard(gate="gate", branch="direct")),
                Edge(from_="gate", to="writer", when=EdgeGuard(gate="gate", branch="writer")),
                Edge(from_="direct", to="changes"),
                Edge(from_="writer", to="guard"),
                Edge(from_="guard", to="changes"),
                Edge(from_="changes", to="persist"),
            ],
            feedback=[
                FeedbackEdge(
                    id="rating",
                    from_="surface:scaling-notes",
                    to="gate",
                    source="explicit",
                    scope="user",
                    form="score",
                    latency="deferred",
                    score=ScoreConfigSpec(
                        name="scale_feedback",
                        data_type="NUMERIC",
                        description="The cook's rating of a scaled recipe (#317). bad 0 / okay 0.5 / good 1, on the run's trace; the comment carries the factor, the route and the flagged lines.",
                    ),
                ),
                FeedbackEdge(id="flagged-lines", from_="surface:scaling-notes", to="classify", source="explicit", scope="user", form="queue", latency="deferred", promote_via=SCALE_LINE_LANE),
            ],
            eval=EvalSpec(dataset="scale-eval", gate_metric="score_positive"),
            experiments=experiments,
        )
    )


def confident_distributions(n: int) -> dict[str, list[float]]:
    """Distributions that clear the 0.7 gate on every line."""
    out: dict[str, list[float]] = {"dish": [0.95]}
    for i in range(n):
        out[f"line:{i}"] = [0.92, 0.08]
    return out


def unsure_distributions(n: int) -> dict[str, list[float]]:
    """Distributions that do not."""
    out: dict[str, list[float]] = {"dish": [0.55]}
    for i in range(n):
        out[f"line:{i}"] = [0.52, 0.48]
    return out


# ---------------------------------------------------------------------------
# A small JSON spec over a demo registry (spec.py, propose.py)
# ---------------------------------------------------------------------------


class DemoInput(BaseModel):
    id: str
    title: str
    factor: float


class DemoOutput(BaseModel):
    text: str
    n: float


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ShoutArgs(BaseModel):
    title: str
    n: float


demo_registry = define_registry(
    primitives=[
        define_primitive(
            id="demo.load",
            version="1",
            describe="The recipe's evidence.",
            args=_NoArgs,
            run=lambda _a, ctx: {"row": {"p": 0.8, "top": [{"slug": "thai", "p": 0.8}]}, "title": ctx.input.title},
        ),
        define_primitive(
            id="demo.shout",
            version="1",
            describe="Upper-cases a title, n times.",
            args=_ShoutArgs,
            run=lambda a, _ctx: {"text": a.title.upper(), "n": a.n},
        ),
    ],
    functions={"half": {"describe": "x / 2", "fn": lambda x: x / 2}},
    schemas={"demo.input": DemoInput, "demo.output": DemoOutput},
)


def demo_spec() -> dict[str, Any]:
    return {
        "id": "spec-demo",
        "fact": "demo",
        "input": "demo.input",
        "output": "demo.output",
        "trigger": ["on-demand"],
        "group": "demo-{id}",
        "result": ["loud", "plain"],
        "nodes": {
            "evidence": {"kind": "read", "call": "demo.load@1", "version": "1"},
            "bundle": {"kind": "code", "role": "transform", "inputs": {"p": "evidence.row.p", "title": "evidence.title", "missing": "evidence.nope | 0"}, "version": "1"},
            "gate": {
                "kind": "code",
                "role": "gate",
                "inputs": {"p": "bundle.p"},
                "thresholds": {"p": 0.5},
                "branches": ["loud", "plain"],
                "rules": [
                    {"when": "p >= $t.p", "branch": "loud", "reason": "'p ' & $string(p)", "confidence": "p"},
                    {"branch": "plain", "reason": "'under'"},
                ],
                "version": "1",
            },
            "loud": {"kind": "code", "role": "derive", "call": "demo.shout@1", "args": {"title": "gate.title", "n": "$half($input.factor)"}, "version": "1"},
            "plain": {"kind": "code", "role": "derive", "expr": '{ "text": gate.title, "n": 0 }', "version": "1"},
            "check": {"kind": "code", "role": "validate", "assert": [{"that": "$string(loud.text ? loud.text : plain.text) != ''", "message": "empty text"}], "version": "1"},
            "persist": {"kind": "store", "target": "memory", "scope": "user", "version": "1"},
        },
        "edges": [
            {"from": "evidence", "to": "bundle"},
            {"from": "bundle", "to": "gate"},
            {"from": "gate", "to": "loud", "when": {"gate": "gate", "branch": "loud"}},
            {"from": "gate", "to": "plain", "when": {"gate": "gate", "branch": "plain"}},
            {"from": "loud", "to": "check"},
            {"from": "plain", "to": "check"},
            {"from": "check", "to": "persist"},
        ],
        "feedback": [
            {"id": "rating", "from": "surface:demo", "to": "gate", "source": "explicit", "scope": "user", "form": "score", "latency": "deferred", "score": {"name": "demo_feedback", "dataType": "NUMERIC"}}
        ],
        "eval": {"dataset": "demo-eval", "gateMetric": "score_positive"},
    }
