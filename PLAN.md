# Thymos — Build Plan

> Status: planning document. No code changes are implied by this file. It defines
> the order of work, definitions of done, and invariants that must hold across
> all phases. Edits to this file are decisions; treat it that way.

---

## 0. North Star (one paragraph)

Thymos is a **typed, effectful, deterministic compositional computation substrate
in zero-dependency C**. ML is the first morphism library on top, not the project
itself. The moat is the small algebra — typed values, declared effects,
content-addressed deterministic plans, audit-grade provenance — that survives
decades because it does not grow into a framework. Agent tools, models, quant,
actuary, and vulnerability research are all morphism packs over the same core.

---

## 1. Invariants (non-negotiable across every phase)

These are properties every phase must preserve. Violating one is grounds to
revert work.

- **Zero forced dependencies.** Core builds with a C compiler. Domain packs may
  optionally link extras (BLAS, libcurl, etc.) but core never requires them.
- **Flat file layout at the repo root.** No deep directory trees in core.
  Domain packs live in shallow folders (`models/`, `tools/`, etc.) and use the
  flat `th_*` naming scheme.
- **One naming convention.** Public types: `th_Pascal`. Functions:
  `th_snake_case`. Constants/enums: `TH_SCREAMING`. No exceptions.
- **Determinism by default.** Same graph + same inputs => byte-identical outputs
  unless the morphism declares a non-deterministic effect (e.g. `TH_EFF_RANDOM`,
  `TH_EFF_NET_IO`).
- **Effects are first-class.** Every morphism declares its effect set. Planner
  decisions (rewrite, reorder, fuse, schedule, cache) are constrained by them.
- **Content-addressable plans.** Compiled plans are hashes of their inputs +
  graph; identical sub-plans produce identical hashes.
- **Provenance is a compile output.** Every compile emits a machine-readable
  bundle: schemas, effects touched, capabilities used, memory plan, schedule.
- **Embeddability.** No global state that cannot be scoped via `th_Context`.
  Multiple independent runtimes must coexist in one process.
- **No hidden allocations in the hot loop.** Steady-state execution must run
  on a pre-planned memory layout.

---

## 2. Anti-goals (what Thymos explicitly does NOT do)

Listing these once so future-you (and agents) don't drift.

- Not a deep learning framework. Not competing with PyTorch on giant GPU
  training. Not an autodiff library by default (autodiff is a functor we may
  add for specific morphism packs).
- Not a Python project. No language bindings until C core is stable. Ever.
- Not another LLM inference engine. We *call* LLMs as tools; we don't run them.
- Not a data warehouse, not a database, not a streaming platform. We compute.
- No reinvention of LAPACK/BLAS internals; we wrap or link when needed.
- No build system maze. CMake stays minimal; flat target list.

---

## 3. Phase 0 — Core Substrate (precondition; user-driven)

This is the work you are finishing. Phases 1+ do not start until everything
here is real and tested. Treat each item as a gate.

### 3.1 Prelude (`prelude.h`)

Stable atoms used everywhere.

- `th_Error` (already present)
- `th_DType` (already present)
- `th_Id` — 32 or 64 bit stable id type
- `th_Shape` — rank + dims, fixed `TH_MAX_DIMS`
- `th_Type` — `{ th_Kind kind; th_DType dtype; th_Shape shape; uint32_t flags; }`
- `th_Hash` — 32-byte content-address digest type

### 3.2 Context (`ctx.h` / `ctx.c`)

Memory + execution + planner state.

- `th_Allocator` (default malloc, arena bump)
- `th_ExecutionPolicy` (`th_pfor` only initially)
- `th_Context` (already present)
- `th_MemoryPlan` — offsets, lifetimes, total bytes
- `th_ContentCache` — content-address -> result handle (optional, can stub)
- Implementations: `th_ctx_init`, `th_ctx_destroy`, `th_alloc`, `th_free`,
  `th_pfor` (single-threaded default, parallel later)

**DoD:** `ctx.c` compiles clean, no UB, has at least one unit test per public
function, arena and default modes both pass.

### 3.3 Tensor (`tensor.h`)

Runtime values for the tensor kind.

- `th_Tensor` (already present; lock the shape)
- `th_StridedView` — view over a tensor without owning storage
- `th_TensorDesc` — pure descriptor for IR (no data)
- Init / destroy / view / copy / fill / print
- Element-wise ops as **kernels only** (not graph nodes): add, sub, mul, div,
  fma, exp, log, relu, sigmoid, tanh, reduce_sum, reduce_max
- Linear ops: matmul (naive), transpose, gemm (call BLAS if available)

**DoD:** kernels work, no allocs in steady state, all ops respect the arena.

### 3.4 ACT IR (`act.h`)

The IR. This is the long-term moat — get the signatures right.

- `th_Object` — IR value with `th_Type` (carries kind, dtype, shape, flags)
- `th_Morphism` — typed transform with effect/capability flags + apply fn
- `th_Signature` — input/output type list; **must be concrete**, not empty
- `th_Category` — owned set of objects + morphisms + boundary ports
- `th_HyperGraph` — graph view over a category (for rewrite/analysis)
- `th_Compiler` — pass pipeline holder
- `th_ExecutionNode` — topo-ordered, memory-planned, dispatchable
- `th_EffectFlags`, `th_CapabilityFlags` — bitfields, enum constants

Compiler passes that must exist before Phase 1:

1. `pass_type_check` — every wire's producer/consumer types are compatible
2. `pass_effect_check` — effect propagation; barrier morphisms not reordered
3. `pass_capability_check` — required capabilities present in `th_Context`
4. `pass_topo_sort` — deterministic ordering (ties broken by stable id)
5. `pass_liveness` — when each object's storage can be reused
6. `pass_memory_plan` — produce `th_MemoryPlan`
7. `pass_content_address` — assign deterministic hash to every node and plan
8. `pass_emit_provenance` — write the provenance bundle

**DoD:** can build a 5-node category by hand, compile it, get a memory plan
and provenance bundle, execute it twice and verify byte-identical outputs.

### 3.5 Provenance bundle format

A small, stable, machine-readable schema (likely a length-prefixed binary +
optional JSON view). Fields:

- Graph hash
- Per-node hashes
- Effects touched
- Capabilities required
- Memory plan summary
- Schedule
- Tool/morphism versions
- Inputs schema hashes

**DoD:** identical builds on different machines produce identical bundles
modulo timestamps; timestamps are isolated into a separate section.

### 3.6 Minimal CLI demo

Single binary that builds a tiny graph, compiles it, runs it, prints output
plus a one-line provenance summary. This is the "easy command to get a
valuable run" promised in `README.md`.

**Phase 0 exit gate:** the demo runs, the provenance bundle exists, two runs
produce identical output and identical bundles.

---

## 4. Phase 1 — `tools/` Pack (first domain; biggest moat)

Why first: research validated that idempotency, deterministic replay, typed
effects, and pre-execution plan verification are unsolved problems in current
agent stacks. Thymos's substrate solves them structurally. Demonstrating this
first proves the thesis with the lowest dependency on rare expertise.

### 4.1 Files

Flat-file principle preserved.

- `tools/tools.h` — public API surface
- `tools/tool_http.c`, `tools/tool_http.h` — HTTP morphism (optional libcurl)
- `tools/tool_shell.c`, `tools/tool_shell.h` — shell morphism
- `tools/tool_file.c`, `tools/tool_file.h` — file morphism
- `tools/tool_llm.c`, `tools/tool_llm.h` — LLM call morphism (vendor-agnostic)
- `tools/plan.c`, `tools/plan.h` — plan, verifier, runner

### 4.2 Core structs

- `th_Tool` — a morphism with: name, input schema, output schema, effect set,
  capability set, cost estimate, deterministic flag, apply fn
- `th_Plan` — alias of `th_Category` with named entry/exit ports
- `th_PlanReport` — output of `th_plan_verify`: list of issues, budget summary,
  capability summary, effect summary
- `th_CapabilityGrant` — explicit list of permitted capabilities, attached to
  `th_Context`
- `th_Budget` — max cost, max latency, max IO bytes, max tool calls
- `th_Sandbox` — fs root, net allowlist, env allowlist
- `th_Retry`, `th_Fallback` — morphism combinators
- `th_Parallel`, `th_Sequence`, `th_Branch`, `th_Observe`, `th_Approve` —
  control-flow combinators expressed as morphisms

### 4.3 Public API (signatures, illustrative)

- `th_Error th_tool_register(th_Context*, const th_Tool*)`
- `th_Error th_plan_verify(th_Context*, const th_Plan*, const th_Budget*, const th_CapabilityGrant*, th_PlanReport*)`
- `th_Error th_plan_compile(th_Context*, const th_Plan*, th_ExecutionNode**, size_t*)`
- `th_Error th_plan_run(th_Context*, th_ExecutionNode*, size_t, const th_Object* inputs, th_Object* outputs)`
- `th_Error th_plan_hash(const th_Plan*, th_Hash*)`
- `th_Error th_plan_cache_lookup(th_Context*, const th_Hash*, th_Object* out)`

### 4.4 Invariants specific to tools

- Every `th_Tool` declares effects + capabilities. Unverified tools cannot
  enter a plan.
- Idempotency key = `th_hash(content_address(tool) || content_address(inputs))`.
- A plan that requires a capability not present in `th_Context` fails
  `th_plan_verify` before any tool runs.
- Cost estimates are summed at compile time; if `sum > Budget`, verify fails.
- `TH_EFF_NET_IO`, `TH_EFF_FILE_IO`, `TH_EFF_IRREVERSIBLE` cannot be reordered
  across barriers.
- `th_Approve` morphism forces a host callback before execution proceeds.

### 4.5 Definition of Done (Phase 1)

A demo binary that:

1. Registers 3 tools (e.g. http-get, file-write, llm-call).
2. Accepts a JSON plan (or builds one in C) describing a multi-step task.
3. Runs `th_plan_verify`, prints capability/effect/budget report.
4. Runs the plan. Second run hits the cache for unchanged sub-plans.
5. Replays the plan from the provenance bundle and produces identical outputs
   (modulo declared non-determinism).
6. Rejects a plan that tries to do file IO without the capability granted.

If a competing agent framework cannot reproduce step 5 or step 6 structurally,
Thymos has earned its existence.

---

## 5. Phase 2 — `models/` Pack (second domain)

Why second: piggybacks on existing `opt.h`, `tensor.h`, and the planner. The
single-sample inference overhead win (sklearn's per-call validation cost) is
a clean, measurable benchmark that proves the substrate's performance claim.

### 5.1 Files

- `models/models.h` — public API
- `models/linear.c`, `models/linear.h` — linear / logistic / GLM
- `models/tree.c`, `models/tree.h` — decision tree
- `models/gbm.c`, `models/gbm.h` — gradient boosted ensemble
- `models/mlp.c`, `models/mlp.h` — small MLP
- `models/pipeline.c`, `models/pipeline.h` — preprocess + model fused
- `models/calibration.c`, `models/calibration.h` — Platt, isotonic, conformal
- `models/drift.c`, `models/drift.h` — drift monitor as effectful morphism
- `models/linear_regression.h` — already present; subsume or rewrite

### 5.2 Core structs

- `th_LinearModel`, `th_GLM`, `th_LogisticModel`
- `th_Tree`, `th_GBMEnsemble`
- `th_MLP`
- `th_Pipeline` — ordered preprocessors + estimator as one compiled category
- `th_Calibrator` (Platt / isotonic)
- `th_Conformal` (split conformal predictor)
- `th_DriftMonitor`

### 5.3 What this pack must prove

- **Static compilation of preprocess+predict** that strips runtime validation
  and produces a flat memory plan.
- **Single-sample p99 latency materially lower than sklearn** on at least one
  realistic tabular pipeline (e.g. log-reg + scaling + one-hot).
- **Zero allocations in steady-state inference** for the same pipeline.
- **Train + predict express the same morphism category** so training-serving
  skew is structurally impossible.

### 5.4 Definition of Done (Phase 2)

1. Build a tabular pipeline (scale + encode + logistic regression) in Thymos.
2. Train it; serialize to a content-addressed bundle.
3. Load and run inference; benchmark p50/p99 single-sample latency vs sklearn.
4. Confirm zero allocations during the prediction loop.
5. Wrap the same pipeline as a `th_Tool`; expose to Phase 1's planner.

---

## 6. Phase 3+ — Sketches (do not start yet)

These are deliberately sparse. They become real only after Phases 0-2 lock.

### 6.1 `quant/`

- Time-aware effects (`as_of`, `data_version`).
- `BarSeries`, `OrderBook`, `PointInTimeJoin` as typed primitives.
- `BlackScholes`, `MCPricer`, `Greeks` (functorial sensitivities).
- Backtest = research category; live = production category; same morphisms.
- Audit artifact = provenance bundle directly consumable by compliance.

### 6.2 `actuary/`

- `CashFlowProjection` over a time-grid functor.
- `MortalityTable`, `LapseAssumption`, `DiscountCurve` as versioned objects.
- `Triangle`, `ChainLadder`, `Mack`, `BornhuetterFerguson`.
- `ScenarioSet` as monoidal product of base projection.
- IFRS 17 / Solvency II audit pack from provenance bundles.

### 6.3 `vuln/`

- Taint lattice as `th_EffectFlags` (source/sink/sanitizer).
- `AST`, `CFG`, `IR`, `Binary`, `Trace` as objects.
- Analysis passes as functors on category-of-programs.
- Symbolic + concrete execution as parallel functors.
- Diff between code versions as morphism on programs.

---

## 7. Conventions (apply from now on)

- **Headers** include only what they need; `thymos.h` is the user-facing
  umbrella.
- **No circular includes.** If two files need each other's types, extract
  the shared subset into `prelude.h`.
- **Public functions** return `th_Error`. Out-params are last.
- **No globals** other than `th_default_ctx`. Anything else lives in
  `th_Context`.
- **Unit tests** colocate next to source as `_test.c`, optional CMake target.
- **Benchmarks** live as `bench_*.c` and emit machine-readable output for
  later regression tracking.
- **No TODO without a date.** TODOs older than 90 days are bugs.

---

## 8. Order of work (one screen)

```
Phase 0  [you, now]      core substrate, IR, compiler passes, provenance, demo
Phase 1  [next]          tools/ pack: typed verifiable plans + deterministic replay
Phase 2  [after Phase 1] models/ pack: compiled inference pipelines + bench vs sklearn
Phase 3+ [later]         quant/, actuary/, vuln/ — sketched only until Phase 2 lands
```

A milestone is not done until its Definition of Done is met and its demo runs
on a clean checkout. Skipping a DoD is how the project becomes "another C ML
project that died."

---

## 9. Open decisions (track here, do not lose)

- Hash function for content-addressing (BLAKE3 preferred; pure-C dependency
  acceptable since vendored single-file).
- Provenance bundle wire format (binary + JSON projection vs JSON only).
- LLM tool transport (stdio subprocess vs HTTP; both eventually).
- Threadpool ownership (`th_ExecutionPolicy` vs separate `th_Threadpool`).
- Whether autodiff lives in `models/` or as a generic functor in `act.h`.
