---
name: decision-pipeline
description: Design, build, test and improve decision pipelines with the decision-pipeline library (TypeScript `@rivers/decision-pipeline` or the Python `decision_pipeline` package) - declared DAGs where a model estimates probabilities, a code gate decides, a feedback edge lands the human signal, and every change earns its place through a shadow run and a locked trial. Use when writing or reviewing a pipeline manifest or JSON spec, implementing ports (decide, generate, store, tracer, scores, queue), writing a hypothesis or trial, running feature discovery, debugging a PipelineContractError or a run record, or when someone asks how to structure an LLM feature so it can be traced, tuned and corrected.
---

# Decision pipelines

A decision pipeline is a small declared graph for one LLM feature. The stance, which every API in the
library enforces: **models estimate, code decides, people correct it later.** A `decide` node asks a
model for probabilities, never for the answer. A code **gate** turns probabilities plus recorded
thresholds into a branch. A generative model runs only on the branch that needs it. A **feedback edge**
declares where a human signal lands, on an earlier node, so the next run improves. One run leaves one
trace and one record that can replay, score or train.

The TypeScript library is in this repo (`src/`); the Python port is its sibling repo
[decision-pipeline-python](https://github.com/dipseth/decision-pipeline-python) (`decision_pipeline/`). Both expose the
same functions (camelCase vs snake_case), the same contract check, the same run record and the same JSON
wire formats, and they produce identical hashes, versions and fingerprints for the same manifest.
Read `references/typescript.md` or `references/python.md` for the exact API of the language in use, and
`references/json-formats.md` for specs, hypotheses and patches as JSON.

## 1. Decide whether it is a pipeline

Before writing nodes, answer three questions. If any answer is "no", it is a chain, not a decision
pipeline, and `definePipeline` / `define_pipeline` will refuse it:

1. **What is the one fact it produces?** (`fact`, and `result` names the node(s) that produce it)
2. **Where does code decide?** At least one `code` node with `role: "gate"`, reading a probability a
   `decide` node produced and a threshold declared as data.
3. **Where does a person correct it?** At least one `feedback` edge landing on an earlier node, with a
   score config (`form: "score"`) or a review lane (`form: "queue" | "derived"`).

If the feature really is "call a model, save the answer", say so and do not force a gate in.

## 2. The contract the loader enforces

`definePipeline` returns every violation at once as a `PipelineContractError` (`.problems`). Predict them
rather than discover them:

| Rule | Check |
|---|---|
| One named fact | `fact` and `result` required; `result` may not name a `store` node |
| A code gate | at least one `code` node with `role: "gate"`; only a gate may appear in an edge's `when.gate`; the gate declares `branches` and `thresholds` |
| A fallback per branch | every declared branch has an outbound edge `{ from: gate, to, when: { gate, branch } }` |
| A feedback edge | at least one; `to` is a real node; `score` form needs a `score` config; `queue`/`derived` need `promoteVia` naming a **registered lane** (`defineReviewLane`); `override`/`feature` only at scope `associated` or narrower |
| Review never inside a run | structural: there is no review node kind; review is a lane a feedback edge promotes into |
| Separability | `overridable` requires a `store` node, so the pipeline's own answer is persisted beside any human override |
| Acyclic graph | flow edges form a DAG |
| Named inputs | a node's `inputs` refs name direct inbound nodes (or `$input`); a renamed node breaks at load |
| Eval | `eval.gateMetric` is a registered gate metric (`score_positive` and `no_override_disagreement` ship) |
| Probes | `probes.from` is a `generate` node with a direct edge into the decide node; a decide node with probes is not cached |
| Tools | `tools.allow` bounds `static` and `selectable.candidates`; `selectable.from` is an upstream decide node, `selectable.gate` an upstream gate |
| Experiments | champion is an arm; two or more arms; an arm patch touches only declarative fields of nodes that have them; the assignment unit is not narrower than the widest scope a feedback edge writes to |

Rule 7, one run is one trace, is the runtime's: one root span, one child span per node that ran,
skipped nodes listed on the root with their reason.

## 3. Build one

Work in this order; each step is checkable before the next.

1. **Schemas.** Input and output as zod schemas (TS) or pydantic models / `TypeAdapter`s (Python).
2. **Nodes.** Five kinds, nothing else:

   | Kind | Job | Notes |
   |---|---|---|
   | `read` | load reference data (`load` body) | may `providesFeatures` |
   | `decide` | ask a model for probabilities (`questions` names a fixed question set; `interpret` turns distributions into a value) | never branches; may declare `probes` |
   | `generate` | ask a model to write (`prompt`, `route`, optional `tools`) | only on the branch that needs it |
   | `code` | `role: "transform" | "validate" | "derive"` with a `run` body, or `role: "gate"` with `branches` + `thresholds` | only a gate branches; a transform may also declare `thresholds` it applies |
   | `store` | persist the fact (`target`, `scope`) | its output is not the fact |

   Every node has a `version` string; bump it when the node's meaning changes. `onFailure` is
   `fail` (default), `skip`, `revert` (pass the input through, for validators) or `fallback` (run the
   `fallback` body). `timeoutMs` gives a node a deadline: past it the node fails with `NodeTimeoutError`
   (so `onFailure` decides), `args.signal` aborts, and a late answer is discarded. A body or port that
   throws `NodeSkipped("reason")` is recorded as skipped, not failed, whatever its `onFailure`.
   Thresholds are data: a number, `{ env, default }` or `{ promptConfig, default }`; a gate body reads
   `args.thresholds.<name>` and never the environment.
3. **Edges.** Plain `{ from, to }`, and guarded `{ from: gate, to, when: { gate, branch } }` out of a gate.
   A node whose every inbound guard is false is skipped with the reason recorded.
4. **Named inputs** instead of digging in `args.from`: `inputs: { row: "evidence.row", n: "$input.n", maps: "maps | retry_maps | null" }` → the body reads `args.in.row` (`args.in_` in Python). Checked at load.
5. **Feedback.** `{ id, from: "surface:<name>", to: <earlier node>, source, scope, form, latency, score? , promoteVia? }`. A 👍/👎 is `form: "score"` with a `score` config; a reviewer lane is `form: "queue"`; a proposed change to the pipeline's own questions is `form: "derived"` and must go through a lane.
6. **Eval.** `{ dataset, gateMetric }`.
7. **Define, then test with the in-memory ports** before any real port exists: script the model
   (`testPorts({ decide: { classify: { distributions: { refund: [0.93, 0.07] } } } })`), run, and assert
   on `route`, `output`, `record.nodes` (branch, `branch_reason`, thresholds, skipped reasons) and
   `storeWrites`. Flip the scripted probability and assert the other branch. No network, ever.
8. **Ports.** Implement only what the nodes need. A port returns something the run uses; an **observer**
   only watches and may never fail a run. `tracer` and `cache` are ports; a record sink and a cost
   meter are observers. An observer gets `onRunStart`, one `onNodeStart`/`onNodeEnd` pair per node
   considered (skips included), and exactly one of `onRunEnd` (the record) or `onRunFail` (what a
   failed run had spent). A failed run never reaches the record sink.
9. **Draw it and explain it** from `pipeline.toJSON()`: `renderPipelineSvg(json, { title })` is one
   dependency-free SVG (gates are hexagons, survivable nodes dashed, feedback is the dashed rail back up);
   `explainPipeline(json, { level: "eli5" | "eli10" | "eli15" | "expert" }, generate)` builds the brief
   and prompt, your model writes the words, and `explainCacheKey` keys it per pipeline version, level
   and language so it is written at most once.

Minimal manifests for both languages are in the reference files. Copy the shape, not the domain.

## 4. The decide port and Jev

The `decide` port receives one request per decide node and returns `distributions` (mandatory, keyed by
question) plus `distributionOptions`, optional `boxes`, `value` and `costUsd`. Any model that returns probabilities
works; the package was built around Jev (TypeSafe), a calibrated evaluator that answers many questions
about one state in one request. Rules learned in production:

- **One state, many questions, one request.** Never fan out N calls over the same material. Point each
  question at its part of the state with a backticked path.
- **Ask atomic facts and combine them in code.** "Is this the dish they want?" is several questions.
- **Tune each threshold against its own question type.** A yes/no (`noul`) probability is absolute; a
  `choice` is relative to its options.
- **Math, counting, dates and filtering go in code**, not in a question.
- **Pin the model version** (`jev-1.14.0`, the package's `JEV_MODEL`) once thresholds are tuned; moving
  model versions is a challenger, re-tune gates on its records. `jev-latest` moves to 1.14.0 a week after
  release, so a pipeline on the alias changes model under its thresholds.

The package ships the wire helpers so a port is just the HTTP call: `jevAnswersToDecideResult` /
`jev_answers_to_decide_result` (noul → `[p]`, choice/score → per option, box → `boxes`), `jevHeaders` (adds
`TypeSafe-Danger-Zone-Images: true` for an image or a Box; `TypeSafe-Danger-Zone-Long-Context: true` only when
asked) and `checkJevRequest` (a Box on a pre-1.14 model, a Box with criteria). Images work on every Jev, 1.13
included, anywhere in the JSON (state, instructions, criteria); every model is weaker on images than text,
so decompose image questions finely. All danger-zone features are dreamer-ready, not production-ready.

**Box (Jev 1.14, images)** answers *where* something is: four edges as image fractions plus a `confidence`.
It is an estimate, so it stays inside a `decide` node, no new node kind. The runtime checks each box, records
it as `boxes` on the node, mirrors its confidence into `distributions` as `[confidence]` (so a gate thresholds
it like a Noul), and adds `<node>.<q>.box_area/box_cx/box_cy` features. Code reads the edges from
`args.boxes.<decide>.<q>` (`$boxes` in JSON specs, or `interpret`'s third argument). A Box never says "not
found": ask a Noul beside it and gate on both. Cropping, padding (`padBox`), tiling (`tileBoxes`,
`boxFromTile`) and edge checks (`edgeCheckQuestions`, `expandBoxEdges`) are code; a redrawn box scored with
`boxIoU` is a natural `score` feedback edge. Probes stay noul/choice/score.

**Probes** let an upstream `generate` node write a few extra questions for *this* input
(`probes: { from, max, paths }` on the decide node). The runtime validates them (schema, cap of 8,
duplicates, length, a real state path), records their text on the run, and withholds their answers from
every gate, so a threshold is only ever tuned on questions that exist on every run. Forbid the writer from
re-asking the decision itself; ask for observable facts. Probes have to earn their cost in a trial like
any change.

## 5. Improve it safely

A live pipeline is the champion. Every change is a challenger:

1. **Propose** (`proposePipelineSpec`): a model writes a changed JSON spec toward a goal; the compiler
   returns every problem and the model repairs until it compiles. Vocabulary (prompts, routes, question
   sets, store targets) is bounded to what the host's ports dispatch on.
2. **Shadow** (`runShadow`): the challenger runs beside the champion on real inputs. The champion ships;
   the challenger's writes are captured, never stored; identical model calls are shared so you measure
   the pipeline, not model noise.
3. **Claim** (`compileHypothesis` + `lockHypothesis`): metric, estimand, test (`greater` / `less` /
   `equivalent` / `different`), guardrails, a sealed holdout split. Lock **before** data. Decision rule:
   supported when the whole interval clears the claim, refuted when it all misses, inconclusive otherwise;
   a `different` claim is never refuted (claim `equivalent` to say "no difference").
4. **Trial** (`lockTrial` → `registerTrial` → `collectTrial` → `judgeTrial`): paired rows on the holdout,
   one confirmatory read. A supported primary with a refuted guardrail is refuted.

**Experiments as patches** (`experiments[].arms`) change only declarative fields (thresholds, prompts,
questions, routes, tools, edge guards); a change that needs new code is a new node version, and one
that adds nodes goes through shadow instead. Arms are assigned by a deterministic hash before node 0.

**Discover** (`discoverForSpec`) changes what the pipeline *asks*: an author model proposes questions,
the answer port answers them for labelled rows, a linear model (logistic or ridge) learns, worst rows go
back to the author. It returns a patched spec with `<prefix>_ask` (decide), `<prefix>_score` (derive,
the model as one JSONata expression) and optionally `<prefix>_gate`. Publish the exact wording the loop
asked (`learnedQuestionSet`, same `presence` criteria), or production scores drift.

## 6. Reading a run

The `RunRecord` is the same for every pipeline: `version` (composite hash of node versions, prompt
versions, resolved thresholds, knowledge version and arms), `run_id` (this attempt's trace id) vs
`run_key` (the durable identity of the work, stable across retries), `fact_route` (branches joined,
`direct` or `writer/quick`), `fact_confidence`, `nodes[]` (per node: `input_hash`, `ms`, `cost_usd`,
`cost_reported`, `cached`, `skipped` reason, gate `branch` + `branch_reason` + `thresholds`, decide
`distributions` + `distribution_options` + `probes`, generate `tools`), and a flat numeric `features`
map (`<node>.<question>.margin/entropy/top`, probe aggregates, `nodes_ran`, `cost_usd`). A threshold can
be re-tuned from records alone, with zero model calls.

## 7. Pitfalls

- A gate body sees `args.probes = {}`; only non-gate nodes and `generate` requests get probe answers.
- `result` names the node that **produces** the fact, never the `store` node.
- `derived` and `queue` feedback need a registered lane; queues are capped per Langfuse project, lanes
  are free (`defineReviewLane({ lane, queue: PIPELINE_REVIEW_QUEUE, scoreConfig })`).
- `override` and `feature` feedback is only legal at scope `associated` or narrower.
- A shadow challenger never writes (`writesFact: false`), whatever its store node says.
- Changing a hypothesis after locking it makes the verdict `invalid`; give the new version a new id.
- Cross-language naming: TS `from` / `in` are Python `from_` / `in_`; everything else is snake_case.
  JSON specs, hypotheses, patches and `toJSON()` keep camelCase keys in both languages.
- Python hosts may pass sync or async callables everywhere; node bodies and ports are awaited either way.
