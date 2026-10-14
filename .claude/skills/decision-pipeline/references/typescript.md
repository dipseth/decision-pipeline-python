# TypeScript API (`@rivers/decision-pipeline`, `src/`)

Install: `npm install github:dipseth/decision-pipeline zod`. Ships TypeScript source; use `tsx`, Vite, Bun or
Next.js `transpilePackages`. Subpaths: `@rivers/decision-pipeline/testing` (in-memory ports) and
`@rivers/decision-pipeline/jsonata` (expression engine; bring `jsonata` yourself).

## A manifest

```ts
import { z } from "zod";
import { definePipeline, runPipeline, type DecidePort } from "@rivers/decision-pipeline";
import { testPorts } from "@rivers/decision-pipeline/testing";

const Ticket = z.object({ id: z.string(), text: z.string() });
const Reply = z.object({ text: z.string() });

export const triage = definePipeline({
  id: "triage", fact: "reply", input: Ticket, output: Reply,
  trigger: ["on-demand"], group: (t) => `ticket-${t.id}`, result: ["canned", "check"],
  nodes: {
    classify: { kind: "decide", questions: "is-refund-request", version: "1" },
    gate: {
      kind: "code", role: "gate", branches: ["canned", "writer"], thresholds: { sure: 0.8 },
      run: ({ primary, thresholds }) => {
        const p = (primary as Record<string, number[]>).refund?.[0] ?? 0;
        return p >= thresholds.sure! ? { branch: "canned", confidence: p }
                                     : { branch: "writer", reason: "not sure it's a refund", confidence: p };
      },
      version: "1",
    },
    canned: { kind: "code", role: "derive", run: () => ({ text: "Your refund is on its way." }), version: "1" },
    writer: { kind: "generate", prompt: "support-reply", route: "support-writer", version: "1" },
    check:  { kind: "code", role: "validate", run: ({ primary }) => primary, onFailure: "revert", version: "1" },
    save:   { kind: "store", target: "tickets-db", scope: "user", version: "1" },
  },
  edges: [
    { from: "classify", to: "gate" },
    { from: "gate", to: "canned", when: { gate: "gate", branch: "canned" } },
    { from: "gate", to: "writer", when: { gate: "gate", branch: "writer" } },
    { from: "writer", to: "check" }, { from: "canned", to: "save" }, { from: "check", to: "save" },
  ],
  feedback: [{ id: "thumbs", from: "surface:reply", to: "gate", source: "explicit", scope: "user",
               form: "score", latency: "deferred", score: { name: "reply_helpful", dataType: "BOOLEAN" } }],
  eval: { dataset: "triage-eval", gateMetric: "score_positive" },
});

const ports = testPorts({ decide: { classify: { distributions: { refund: [0.93, 0.07] } } } });
const { output, route, record } = await runPipeline(triage, { id: "t-42", text: "charged twice" }, ports);
```

## Surface, by module

| Module | What you call |
|---|---|
| `define` | `definePipeline(spec)`, `validatePipeline(spec): string[]`, `validatePatch`, `ancestorsOf`; `pipeline.toJSON()`, `pipeline.order` |
| `run` | `runPipeline(pipeline, input, ports?, options?)` → `{ output, runId, runKey, version, route, confidence, record, skipped, patchApplied, scopeRejections }`; `applyOverride(pipeline, output, ctx)` |
| `ports` | `PipelinePorts { tracer, decide, generate, store, scope, scores, queue, dataset, prompts, thresholds, toolRegistry, cache, records, observers, clock, knowledgeVersion }`; `RunOptions { trigger, runKey, parentRunId, forceArm, shadow, patch, signal, metadata, tags }`; request/result types `DecideRequest/Result`, `GenerateRequest/Result`, `StoreRequest`, `ScoreRequest`, `QueueRequest`, `DatasetRunRequest` |
| `testing` | `testPorts({ decide, generate, scope, prompts, registry, env, promptConfig, knowledgeVersion, traceId })`, `recordingTracer`, `scriptedDecide`, `scriptedGenerate`, `collectingStore`, `memoryCache`, `collectingRecords`, `collectingObserver`, `throwingObserver`, `collectingScores`, `collectingQueue`, `collectingDataset`, `memoryTrialStore`, `staticScope`, `staticPrompts`, `staticRegistry`, `steppingClock` |
| `feedback` | `submitFeedback(pipeline, { runId, runKey?, edgeId, value?, label?, comment?, rank?, sourceId, key?, payload?, modelValue?, nodeId?, metadata?, arm? }, ports)` → `{ edge, label, row, queued, scored, warnings }`; `enqueueForHandRead` |
| `review` | `defineReviewQueue`, `defineReviewLane({ lane, queue, scoreConfig })`, `PIPELINE_REVIEW_QUEUE`, `PIPELINE_HAND_READ_QUEUE`, `REVIEW_QUEUE_BUDGET` (3) |
| `record` | `RunRecord`, `RunRecordNode`, `RunLabel`, `defineGateMetric(name, fn)`, `aggregateGateMetric` |
| `eval` | `recordPipelineEvalRun(pipeline, { rows, traceId, experiment?, arm?, extraScores? }, ports)`; `evalRunName` |
| `probes` | `collectProbes`, `probeAnswers`, `probeFeatures`, `MAX_PROBES` (8), `probeQuestionSchema` |
| `bindings` | `parseInputRef`, `resolveInputs`, `bindingProblems` |
| `experiments` / `patch` | `ManifestPatch { thresholds, prompts, questions, routes, tools, guards }`, `assignArm`, `mergePatches`, `applyManifestPatch` |
| `spec` | `compilePipelineSpec(json, registry, engine)` → `{ pipeline, problems }`, `pipelineFromSpec` (throws), `defineRegistry({ primitives, functions, schemas, features })`, `definePrimitive({ id, version, describe, args, run })`, `describeRegistry`, `pipelineSpecJsonSchema()` |
| `jsonata` | `jsonataEngine({ timeoutMs?, maxDepth? })` |
| `propose` | `proposePipelineSpec({ goal, registry, engine, author, base?, vocabulary?, maxRounds? })` → `{ spec, pipeline, rounds, problems, diff }` |
| `shadow` | `runShadow({ champion, challenger, input, ports, options?, label, share? })` → `{ champion, challenger, challengerError, challengerStores, calls, comparison }`; `compareRuns` |
| `metrics` | `BUILTIN_METRICS` (36), `defineMetric`, `metricRegistry`, `metricCatalog`, `ObservationRow` |
| `hypothesis` | `compileHypothesis(json, { metrics?, engine? })`, `lockHypothesis`, `evaluateHypothesis(h, rows, { lock, now })`, `planHypothesis`, `splitOf`, `partitionRows`, `observationFromRun`, `observationsFromShadow` |
| `trial` | `lockTrial(h, { champion, challenger })`, `registerTrial(store, lock)`, `collectTrial({ hypothesis, store, champion, challenger, inputs, id, unit?, ports, truth?, meta?, concurrency? })`, `judgeTrial(h, store)`, `pipelineFingerprint` |
| `discover` / `learned` | `discoverFeatures({ rows, baseNames, target, rounds, ports: { author, answer } })`, `discoverForSpec({ spec, attach, discovery?, check?, ...})`, `attachLearnedGate`, `learnedQuestionSet`, `learnedExpression`, `scoreLearned`, `fitRidge`, `fitLogistic`, `designMatrix`, `INTENSITY_LEVELS` |
| `diagram` | `renderPipelineSvg(json, { title?, standalone?, header?, legend? })`, `layoutPipeline`, `nodeSubtitle` |
| `explain` | `explainPipeline(json, { level, title?, summary?, languageName? }, generate)`, `explainPrompt`, `explainBrief`, `explainVersion`, `explainCacheKey`, `EXPLAIN_LEVELS`, `EXPLAIN_SYSTEM_PROMPT` |
| `stats` | `wilsonInterval`, `newcombeInterval`, `newcombePairedInterval`, `mcnemarExact`, `clusterBootstrap`, `seededRandom`, `requiredSampleSize` |

## Node bodies

Every body receives `NodeRunArgs { input, from, in, primary, ctx, thresholds, node, branches, probes, signal }`.
`signal` is an `AbortSignal` that aborts when the run is cancelled or the node's `timeoutMs` passes; a
port should pass it to its HTTP call. Throw `NodeSkipped(reason)` to opt out of running (recorded as a skip);
a `NodeTimeoutError` is what `onFailure` sees past the deadline.
`primary` is the first inbound node's output (the pipeline input at a root). A gate returns a branch name
or `{ branch, reason?, confidence? }`; `confidence` becomes `fact_confidence`. `interpret(distributions, args)`
on a decide node turns raw distributions into the value downstream sees; the record keeps the distributions.

## A decide port backed by Jev

```ts
export const jevDecide: DecidePort = async (req) => {
  const probes = Object.fromEntries(Object.entries(req.probes).map(([k, { why, ...q }]) => [k, q]));
  const res = await fetch("https://api.typesafe.ai/v1/systemone", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${process.env.TYPESAFE_API_KEY}` },
    body: JSON.stringify({ model: "jev-1.13.0", state: buildState(req.input), questions: { ...loadQuestions(req.questions), ...probes } }),
  });
  const { answers, usage } = await res.json();
  const distributions: Record<string, number[]> = {}, distributionOptions: Record<string, string[]> = {};
  for (const [key, a] of Object.entries<any>(answers)) {
    if (typeof a.noul === "number") { distributions[key] = [a.noul]; distributionOptions[key] = ["true"]; }
    else if (a.probabilities) { distributionOptions[key] = Object.keys(a.probabilities); distributions[key] = Object.values(a.probabilities); }
  }
  return { distributions, distributionOptions, costUsd: usage.input_tokens * 0.042e-6 };
};
```

## Develop

`npm install && npm test` (vitest, no network) and `npm run type-check`. Fixtures: `src/test-fixtures.ts`
(the scale pipeline, the demo registry and spec). Tests for a new node or port follow `src/run.test.ts`.
