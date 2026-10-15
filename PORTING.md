# Porting conventions: TypeScript → Python

This package is a port of the TypeScript core in
[dipseth/decision-pipeline](https://github.com/dipseth/decision-pipeline) (`src/`). The TS source is
the reference: same module names, same function names (snake_case), same error messages, same
record schema. When in doubt, read the `.ts` file of the same name and match it. `UPSTREAM` at the
repo root names the TS commit this port mirrors; bump it when a delta is ported.

## Layout

| TS (`src/`)           | Python (`decision_pipeline/`) | Notes |
|-----------------------|-------------------------------|-------|
| `types.ts`            | `types.py`                    | dataclasses, `kw_only=True` |
| `eval.ts`             | `evals.py`                    | `eval` is a builtin |
| `test-fixtures.ts`    | `fixtures.py`                 | the scale pipeline and the demo spec |
| `testing.ts`          | `testing.py`                  | in-memory ports |
| `jsonata.ts`          | `jsonata.py`                  | needs the `jsonata` extra (`jsonata-python`) |
| `x.test.ts`           | `tests/test_x.py`             | one test module per TS test file; port **every** test, keep its name as the function name |
| everything else       | same name                     | |

## Naming

- Functions, fields and parameters are `snake_case`: `definePipeline` → `define_pipeline`,
  `onFailure` → `on_failure`, `acceptsFeatures` → `accepts_features`, `costUsd` → `cost_usd`.
- Two TS names are Python keywords. Mechanically: **`from` → `from_`**, **`in` → `in_`**
  (`Edge.from_`, `FeedbackEdge.from_`, `NodeRunArgs.from_`, `NodeRunArgs.in_`,
  `DecideRequest.from_` / `.in_`, `GenerateRequest.from_` / `.in_`).
- Classes keep their TS names: `DecisionPipeline`, `RunRecord`, `PipelineContractError`.
- Constants keep their names: `SCOPES`, `MAX_PROBES`, `BUILTIN_METRICS`.

## Wire formats stay camelCase

Anything that is JSON shared with the TS package keeps the TS keys exactly:

- the JSON pipeline spec (`spec.py`): `onFailure`, `promoteVia`, `dataType`, `gateMetric`,
  `acceptsFeatures`, `keyFromInput`, `knowledgeVersion`, `decidedBy`, `providesFeatures`, `maxChars`…
  (pydantic models with `alias=` / `populate_by_name=True`);
- `DecisionPipeline.to_json()` (the `PipelineJSON` shape);
- the hypothesis spec (`hypothesis.py`): `minEffect`, `recordedAt`…;
- `ManifestPatch` as data.

The run record (`RunRecord`, `RunRecordNode`, `RunLabel`) is already snake_case in TS and is
identical here. Goal: a TS run and a Python run of the same manifest produce the same record.

## Data types

- Plain data is `@dataclass(kw_only=True)`. Nodes: `ReadNode`, `DecideNode`, `GenerateNode`,
  `CodeNode`, `GateNode`, `StoreNode` (each has a fixed `kind`). `is_gate_node(n)` is
  `isinstance(n, GateNode)`.
- TS `Record<string, X>` → `dict[str, X]`. TS `X[]` → `list[X]`.
- TS `undefined` → `None`. Where the difference between "absent" and `null` matters
  (bindings, `pick_path`), use the `MISSING` sentinel from `_compat.py` *internally* and never
  let it escape a public function.
- TS `x ?? y` → `y if x is None else x`. TS `x || y` → mind the Python truthiness of `0`/`""`.
- Zod schemas → pydantic v2. A manifest's `input`/`output`, a primitive's `args` and a metric's
  `args` accept a pydantic `BaseModel` subclass, a `TypeAdapter`, or any object with `.parse()`;
  wrap with `as_schema()` from `_compat.py`, which gives `.parse(value)` and `.json_schema()`.
- `z.toJSONSchema(x)` → `as_schema(x).json_schema()`.
- `zodProblems(error)` → `pydantic_problems(error)` (`_compat.py`): `"path.to.field: message"`,
  `"(root): message"` for a root issue. Same strings as TS where pydantic's message allows.
- Timestamps: TS `new Date(ms).toISOString()` → `iso_from_ms(ms)` (`_compat.py`), which
  produces `2023-11-14T22:13:20.000Z` (milliseconds, `Z`). `Date.parse(iso)` → `ms_from_iso(iso)`.
- `clock()` returns **milliseconds** (like `Date.now()`), as a number.
- Small JSON shapes that are wire data, not config, stay plain dicts with a `TypedDict` for the
  type checker: `Box`, `BoxAnswer`, a `ProbeQuestion`. They land on the record as-is.
- TS `Math.round` rounds half up; Python's `round` rounds half to even. Where a rounded number is
  output (`box_to_pixels`), use `math.floor(v + 0.5)` so both languages give the same pixels.
- A TS callback that grew a trailing argument keeps working at its old arity here: `interpret` is
  called with `(distributions, args, boxes)` only when it accepts three positional arguments.

## Sync or async

The TS runtime is async. Here `run_pipeline`, `run_shadow`, `submit_feedback`,
`propose_pipeline_spec`, `evaluate_hypothesis`, `collect_trial`, `judge_trial`,
`discover_features`, `discover_for_spec`, `record_pipeline_eval_run` are `async def`.

Everything the **host** supplies may be sync **or** async: node bodies, `interpret`, every port,
every observer hook, `ThresholdSource.prompt_config`, `CachePort.get/set`. The runtime calls
`await maybe_await(fn(...))` (`_compat.py`). `TracePort.root/span` receive an `async` callback
and must await it.

## Errors and messages

Exception classes live in `errors.py` with the TS names. Error **messages** and `problems`
strings are copied from the TS source verbatim (the tests assert on them), with these
mechanical substitutions: `zod` → `pydantic` in prose, `definePipeline()` → `define_pipeline()`,
`\`promoteVia\`` stays as written when it names a JSON key, becomes `promote_via` when it names
a Python field.

Number formatting in messages: `x.toFixed(n)` → `f"{x:.{n}f}"`; `x.toPrecision(3)` →
`to_precision(x, 3)` (`_compat.py`); `x.toExponential(2)` → `f"{x:.2e}"`.

`console.warn` → `logging.getLogger("decision_pipeline").warning(...)`.

## Hashing

`stable_stringify` must produce the same string as the TS version for JSON-like data (sorted
keys at every depth, JS number formatting: `1` not `1.0`, `1e+21`, `1e-7`). `short_hash` and
`hash_to_unit` therefore match TS for the same data, and so do composite versions, cache keys,
fingerprints and experiment assignments.

Pydantic models and dataclasses are stringified through their fields. `bool` is checked
**before** `int` (Python's `True` is an `int`).

## Registries

`record.py` (gate metrics) and `review.py` (queues and lanes) are process-global, exactly like
TS, with the same `clear_*` / `snapshot_*` / `restore_*` test seams.

## Tests

`pytest`, `asyncio_mode = "auto"` (an `async def test_*` just works). Use the fakes in
`decision_pipeline.testing` and the fixtures in `decision_pipeline.fixtures` like the TS tests
use `./testing` and `./test-fixtures`. `vi.fn()` → a small recording callable. `expect(x).toEqual(y)`
→ `assert x == y` (dataclasses compare by value). `expect(fn).toThrow(/re/)` → `pytest.raises(…, match=…)`.
Do not skip a test because it is awkward to port; port it.
