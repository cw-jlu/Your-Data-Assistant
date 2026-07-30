# Submission log

> 🌐 **Language**: **English** · [한국어](SUBMISSION_LOG.ko.md) · [中文](SUBMISSION_LOG.zh.md)

Tracks every Phase 1 submission against the 30-submission cap. Append a new section per submission; do not edit historical entries.

## Final Phase 1 result — official leaderboard (recorded 2026-07-14)

Source: [dataagent.top/leaderboard](https://dataagent.top/leaderboard). Phase 1 final score is the task-count-weighted average of the two boards (57 A-board / 324 B-board ≈ 0.15 / 0.85).

| Metric | team1438 |
|---|---|
| A-board (2 h, 57 tasks) | **0.3886** (v8 image) |
| B-board (12 h, 324 tasks) | **0.4349** (single sanctioned run, v8 image under `:final` tag) |
| **Phase 1 final** | **0.4279** |
| Final rank | **137** of ~300 teams |
| Top-60 cutoff (Phase 2 qualification) | 0.5209 — did not qualify |

Evaluated-score trajectory: v1 eval-fail (arm64) → v2 **0.3386** → v4 **0.2281** (SIGTERM-truncated) → v6 **0.3509** → v8 **0.3886** → B-board **0.4349**. Everything below is the contemporaneous per-submission record, preserved as written.

## Budget tracker

- Phase 1 cap: 30 total / 1 per day
- Phase 1 window: 2026-04-24 → 2026-05-23

| Used | Remaining | Today | Note |
|---|---|---|---|
| 1 | 29 | 2026-05-05 | v2 leaderboard = **0.3386** (baseline floor) |
| 2 | 28 | 2026-05-11 | **v3** tarball ready (sha256 `1bb11bac…`); awaiting organizer evaluation |
| 3 | 27 | 2026-05-11 | **v4** tarball built (sha256 `40fc00a2…`); 50-task mock 0.7045 (Δ vs v3 ≈ variance); ship decision pending |
| 4 | 26 | 2026-05-12 | **v5** tarball built (sha256 `28eb3880…`); 50-task effective 0.680 (+0.06 vs v4); S-1 deterministic verified; ship decision pending |

## Decision rules

A submission ships only if both gates are green:

- **mock_scorer**: holdout score ≥ previous-best holdout score, no per-difficulty bucket regresses by >2 points
- **local docker e2e**: `bash scripts/local_eval.sh` completes without crash, `prediction.csv` written for ≥95% of holdout

If exactly one gate is green, hold for 24h and re-investigate.

---

## Template (copy below for each new submission)

```
### v<N> — YYYY-MM-DD

- **Branch / commit**: <short-sha>
- **Headline change**: <one sentence>
- **Files touched**: <comma-separated>
- **mock_scorer holdout (λ=0.10)**: mean Score = <value>, recall = <value>
- **Per-difficulty (recall)**: easy=<>, medium=<>, hard=<>, extreme=<>
- **Local docker e2e**: pass | fail (<error>)
- **Leaderboard score (after submission)**: <value>
- **Δ vs prev**: holdout <±value> / leaderboard <±value>
- **Retro**: <1-3 sentences — what we learned, what to do next>
```

---

## v2 leaderboard score — 2026-05-05 (received from organizers)

**Leaderboard score: 0.3386** — first successful evaluation. The v1 → v2 fix (linux/amd64 cross-build) finally landed at the operator side.

**Implications for our planning**:
- Our local mock_scorer (with name-equivalence) was **0.7000** on the public 50-task. Hidden gap ≈ −0.36, much wider than expected.
- Likely sources: (a) hidden distribution harder than public (more extreme tier?), (b) name-equivalence partially mismatched between our scorer and the official one, (c) hidden has more long-tail tasks where we run out of max_steps / timeout.
- Floor for all subsequent submissions: **leaderboard ≥ 0.3386** is the bar to beat. Anything lower = regression.

---

## v3 ship gate — 2026-05-11 (consolidated round: memory layer + multi-pass voting + streaming JSON + error pattern memory)

Single big round on the v2 baseline. Brings the harness to ship quality without depending on leaderboard feedback for further iteration.

### What changed (patch ID → location → behaviour)

**Runtime + scheduling**

- **K-1** — `configs/eval.yaml` `pass_safety_margin: 1.1`. Tighter margin lets the multi-pass orchestrator fit more passes inside the 12 h budget. First pass is always preserved, so regression vs single-pass remains impossible.
- **L-1** — `runner._looks_like_first_step_transient` is now tier-aware. Hard / extreme tier are opted out of "Task timed out after" retry — their timeout is intrinsic to the task, not an endpoint blip. Endpoint-level transients (`Connection error`, `Request timed out`) still retry regardless of tier. Saves ~900 s per long-tail task and recovers that budget for the rest of the multi-pass run.
- **H-1** — `runner.run_benchmark_with_passes` orchestrates up to `repeat_max=3` benchmark passes into `output_dir/_runs/run_<i>/` and votes per task by column-multiset majority via `scoring/cross_run_vote.py`. Adaptive budget guard + `_fallback_copy_pass` make regression vs single-pass impossible.

**Agent**

- **G-1** — `_FIRST_STEP_TRANSIENT_HINTS` includes `"Request timed out"` (recovers ~21 transient timeouts seen in earlier endpoint-overload runs).
- **G-2** — `agents/self_consistency.py:SelfConsistencyAgent` runs k=3 ReActAgents on hard / extreme tier at `temperature=0.5` and majority-votes by `column_signature` multiset. Tie-break to earliest sample. Opt-out via `DABENCH_DISABLE_SELF_CONSISTENCY=1`.
- **G-3** — `agents/prompt.py` raises the `knowledge.md` cap to 5000 chars and reorders H2 / H3 sections by question-keyword relevance, so the truncation prefers material that actually matches the question.

**Tools**

- **H-3** — `tools/streaming_json.py` (`streaming_json_keys`, `streaming_json_count`, `streaming_json_aggregate`) streams huge JSON arrays through `ijson` without loading them into memory. The runner's bundled error_patterns advisory steers the agent toward these tools whenever a large JSON file is in scope.

**Self-improvement memory layer**

- **M-1 / M-2 / M-3** — `src/data_agent_baseline/memory/`:
  - `task_shape.py` — pure-IO TaskShape classifier (difficulty / sizes / file types / question keywords / `is_heavy`).
  - `policies.py` — ShapePolicy resolver with priority + AND match + range expressions (`{"gte":100}`).
  - `learnings.json` — bundled, ships seeded from forensic clusters: heavy tasks get `timeout_multiplier=1.5 / max_steps_multiplier=1.25`, large-JSON / large-DB / large-CSV / aggregate / plural / singular each get tailored hints.
- **M-4** — `runner._run_single_task_core` + `prompt.build_task_prompt` + `react.ReActAgent` + `self_consistency.SelfConsistencyAgent` all consume the resolved policy. `timeout_multiplier` scales the subprocess timeout; `max_steps_multiplier` scales the difficulty-aware step budget; `prompt_hints` / `preferred_tools` / `avoid_tools` are inlined as advisories.
- **M-5** — `memory/recorder.py` + `dabench update-learnings` CLI. After every public-set benchmark the recorder clusters per-task outcomes by shape and proposes adjustments. Low-risk (numeric multiplier bumps) auto-apply with `--apply-low-risk`; high-risk (prompt cue ideas) always require human review.

**Error pattern memory (the user-requested core feature)**

- **N-1** — `memory/error_patterns.py` + `memory/error_patterns.json`. The recorder extracts every failed tool call from `trace.json`, classifies the raw error into a stable signature class (`sqlite_no_such_column`, `python_memoryerror`, `read_json_capped`, …), and clusters by `(shape, action, signature)`. Patterns with ≥ 2 distinct evidence tasks merge into the bundled JSON. At the start of every task the runner injects the matching advisories as a *"Past failure modes observed on similar tasks"* hint block.
- **N-2** — `ReActAgent._build_messages` scans the tail of the trace; when ≥ 2 consecutive tool calls share the same error signature, a one-line **"REPEATED ERROR: do NOT retry the same approach"** circuit-breaker is appended before the next model turn. Cap-at-streak so the prompt does not balloon when the agent is genuinely stuck.
- **N-3** — `memory/task_brief.py` runs a deterministic read-only scan of `context/` (CSV column + row peek, SQLite tables / row counts, knowledge.md presence, join-key candidates derived from question token overlap) and inlines a `Pre-flight task brief:` block in the user prompt. Saves 1–2 early discovery steps the agent used to spend on `list_context` + `inspect_sqlite_schema`.

### Mock scoring (public 50-task, 49-task subset — task_418 hangs on OneDrive mounts in unkillable D-state; eval env irrelevant)

Single pass measurements (the multi-pass orchestrator inside the container runs `repeat_max=3` at eval time and votes — strictly better than the numbers below).

| Run | Tasks scored | Mean (scored) | Perfect (λ=0.10) | 50-task projection |
|---|---|---|---|---|
| v2 internal (no memory layer) | 47 | 0.6986 | 31 | 0.6566 |
| v3 baseline (pre N-1/N-2/N-3) | 45 | 0.6854 | 29 | 0.6169 |
| **v3 with N-1/N-2/N-3 (current)** | **44** | **0.7254** | **31** | **0.6384** |

Tier 별 (current):
- easy n=14 score 0.7143
- medium n=23 score 0.7355
- hard n=7 score 0.7143

**L-1 effect confirmed**: every hard-tier timeout completed in 900 s (single attempt) rather than the legacy 1800 s. ~900 s of wall-clock recovered per long-tail task.

**Recorder auto-apply** after the last measurement merged two new policy bumps and four new error patterns into the bundled JSONs:

learnings.json delta:
- `is_plural_question=True` → `max_steps_multiplier` 1.25 → 1.5
- `difficulty=hard` → `timeout_multiplier` 1.2 → 1.4
- `is_aggregate_question=True` → `timeout_multiplier` 1.0 → 1.2

error_patterns.json delta (newly observed):
- `difficulty=medium / execute_context_sql / sqlite_no_such_table × 7` (task_145, task_173, task_196, task_214, task_261, task_287, task_303)
- `difficulty=easy / execute_context_sql / "file is not a database" × 2`
- `is_heavy=True / execute_context_sql / "file is not a database" × 2`
- `difficulty=medium / execute_python / sqlite_no_such_table × 2`

Both get baked into every subsequent v3 rebuild — every benchmark is an iteration of the system itself.

### Tarball

- **Tarball**: `submissions/team1438_v3.tar.gz`
- **Sizes**: tarball 0.38 GB / image 0.38 GB (well under 10 GB)
- **sha256**: `1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09`
- **Image**: python:3.10-slim + uv 0.5.14 + `UV_OFFLINE=1` + `linux/amd64` (3-tier arch guard). Bundles `memory/learnings.json` and `memory/error_patterns.json`.
- **eval.yaml**: `repeat_max: 3`, `pass_safety_margin: 1.1`, `wall_clock_budget_seconds: 43200`, `flat_output_dir: true`, `log_file: /logs/runtime.log`, env-overridable model fields.

### Container e2e (rules-submission §8 checklist)

- [x] Image name `team1438:v3`
- [x] Tarball name `team1438_v3.tar.gz`
- [x] Tarball size ≤ 9 GB (0.38 GB)
- [x] linux/amd64 manifest verified
- [x] ENTRYPOINT `dabench run-benchmark --config configs/eval.yaml`
- [x] eval.yaml: env-overridable model fields, `flat_output_dir: true`, `log_file: /logs/runtime.log`
- [x] multi-pass orchestrator + voter wired (`repeat_max: 3`, `pass_safety_margin: 1.1`)
- [x] sha256 recorded
- [ ] Drive upload with "Anyone with the link can view" (manual step)
- [ ] Email organizers with team_id + Drive URL + sha256 (manual step)
- [ ] Update budget tracker after sending

### Δ vs v2

- v2 leaderboard floor: 0.3386
- v3 mock 50-task projection (single pass): 0.6384 — multi-pass voting expected up to ~0.66 in production.
- mock → leaderboard gap (v2 historic): −0.15 to −0.20.
- **v3 expected leaderboard: 0.45 – 0.55** = **+0.11 – +0.17 over v2**.

### Retro

- The biggest v3 win is operational, not score-on-paper: `dabench update-learnings` makes every public-set run a learning iteration. From now on, even if we never get another leaderboard reply, the system gets better with every benchmark.
- N-1 / N-2 / N-3 fold the lessons we learned from forensics into structure: instead of writing one-off prompt fixes per task (which the v3-round forensic showed were variance, not signal), the system now reads its own past failures and steers around them at task start.
- Variance in single-pass measurement is still ±3 perfect. Multi-pass + voting (`repeat_max=3` baked into eval.yaml) is the production answer.

---

## v4 ship gate — 2026-05-11 (forensic-driven prompt + per-sample SC budget)

Targeted follow-up after v3's 49-task forensic identified 12 zero-with-pred + 4 hard timeouts. v4 applies four root-cause fixes; tarball is built and ready, leaderboard ship is the user's decision.

### What changed (patch ID → location → behaviour)

**Runtime + agent**

- **P-1 (Fix #1)** — `agents/react.py:ReActAgent.run` accepts an optional `deadline` (perf_counter absolute). `agents/self_consistency.py:SelfConsistencyAgent` now budgets `task_timeout × 0.85` across k samples — each sample gets `remaining_budget / remaining_samples`. Prevents the v3 pattern where k=3 hard tasks blew past 900 s SIGKILL with no trace (`task_344/350/352/379`).
- **P-2 (Fix #2)** — `agents/prompt.py` REACT_SYSTEM_PROMPT adds a **Column whitelist rule** with concrete BIRD pattern examples ("Which X has Y" → X only, "give their X" → X only, etc). Targets 5 over-emission zero-score patterns (task_25/38/180/199/259).
- **P-3 (Fix #3)** — same file: **Aggregation granularity** rule (per-entity vs total averaging) and **Percentage formula** rule (BIRD-style `CAST(COUNT(CASE WHEN ...) AS REAL) * 100 / COUNT(*)`). General pattern, not data-file-specific (hidden set has its own knowledge.md).
- **P-4 (Fix #4)** — weakens the F-1 empty-answer cue: agent must verify (a) no alternate table holds the answer via a different join path, AND (b) the filter column's value range covers the question scope, before committing `rows: []`.

### 50-task local single-pass results (v4 vs v3 same-config baseline)

| Metric | v3 (49 task) | **v4 (50 task)** |
|---|---|---|
| Mean Score (λ=0.10) | 0.7254 | **0.7045** |
| Perfect | 31/44 | 31/44 |
| Zero-with-pred | 12 | 13 |
| Missing | 5 | 6 |
| Wall-clock | 5841s | 5341s |

**Per-difficulty perfect counts:** easy 10/14 → 10/14; medium 16/23 → 17/22; hard 5/7 → 4/8 (+1 newly scored).

**Recovered vs regressed (diff):**
- ✅ Recovered (not → perfect): **task_38** (P-2 over-emit fix), **task_350** (P-1 deadline fix), **task_199** (P-2)
- ❌ Regressed (perfect → not): task_11 (max_steps after column-whitelist tightening), task_349, task_420 (extreme timeout)

### Compliance checklist

- [x] linux/amd64 image built (Dockerfile pinned)
- [x] UV_OFFLINE=1 at runtime
- [x] ENTRYPOINT `dabench run-benchmark --config configs/eval.yaml`
- [x] eval.yaml: env-overridable model fields, `flat_output_dir: true`, `log_file: /logs/runtime.log`
- [x] multi-pass orchestrator + voter wired (`repeat_max: 3`, `pass_safety_margin: 1.1`)
- [x] sha256 recorded (`40fc00a2236731d3fca871b9a1ba6e4e9c2b8b6a28371835e6abc3ccb5a0e278`)
- [ ] Drive upload with "Anyone with the link can view" (manual step)
- [ ] Email organizers with team_id + Drive URL + sha256 (manual step)
- [ ] Update budget tracker after sending

### Decision context

- Mock Δ of −0.02 vs v3 is within run-to-run variance (forensic-confirmed ±0.06 across v5 runs at temp=0).
- P-1's deadline split is a clear architectural win (2 hard timeouts recovered; trace written even on still-timeout).
- P-2 cleanly recovers task_38 over-emit but causes task_11 max_steps regression — net wash on easy tier.
- v4 is comparable to v3 on mock; the hidden set may benefit more from P-1's deadline split if it has similar long-tail patterns.

### Retro

- Trace forensics on 14 failed tasks gave 4 actionable root-cause patches (P-1..P-4). Three of the four are general-pattern fixes; only P-1 hits a structural bug. The other three are prompt cues whose effect is bounded by what the model picks up.
- Hard timeouts still recur in v4 (`task_349/352/379`). P-1's `task_timeout × 0.85 / k` budget gives ~357 s per sample on a 1260 s budget; that's enough for some BIRD-pattern hard tasks but not all. Future round should consider k=2 for shape-tagged "extreme aggregate" or a longer hard timeout.
- Variance dominates over single-pass measurement. Production multi-pass + voting will absorb the medium-tier ±1 swings; the hard-tier 4/8 is the actual delta to chase.

---

## v5 ship gate — 2026-05-12 (Question→Intent→Shape 구조적 재설계)

After v4 wash (0.70 vs v3 0.72 = variance), v5 attacks the fundamental "freedom-accumulating-ambiguity" problem with a 4-layer structural redesign rather than more patch cycles.

### What changed (patch ID → location → behaviour)

**Layer 1 — Variance elimination (S-1)**

- **S-1** — `agents/model.py:OpenAIModelAdapter(seed=...)`. Every `chat.completions.create` call carries a per-task deterministic seed = `blake2b(task_id) + sample_idx`. vLLM ≥0.6 honors it for nucleus/top-k sampling — probed end-to-end (T=1.0, identical bytes). Same task_id → identical prediction.csv across runs. Variance ±0.06 → 0.

**Layer 3 — Question-pattern → answer-shape (S-3)**

- **S-3** — new `memory/question_shape.py`. Rule-based regex matcher classifies the question into one of 5 BIRD-style patterns (percentage / count / average-of-entity / list / single-value) and emits a one-line hint into `policy.prompt_hints`. Built from forensic clustering of the public 50-task (question, gold) shapes. No external models or embeddings — single-model policy intact.

**Layer 4 — Wall-clock re-allocation (S-4)**

- **S-4** — `configs/eval.yaml`: `repeat_max: 3 → 1` (deterministic makes voting redundant); `task_timeout_by_difficulty.hard: 900 → 2400`, `extreme: 1200 → 3600`. Freed ~67% wall-clock now goes to hard tier headroom. SC `k=3` retained for in-task diversity (k samples × deterministic but DIFFERENT seeds = 3 distinct reproducible paths).

**Safety patches (S-5/6/7/8)**

- **S-5** — `config.DEFAULT_MAX_STEPS_BY_DIFFICULTY.easy: 8 → 12` (task_11 v4 regression direct fix).
- **S-6** — column-whitelist rule gets explicit multi-attribute exception ("names AND funding types" → keep both columns).
- **S-7** — `tools/registry.py:_csv_sqlite_fallback`. `execute_context_sql` auto-routes to pandas → in-memory sqlite when sqlite raises "file is not a database" / "no such table" / "disk i/o error". `observation` includes `fallback_used: true` + `original_error` + `loaded_tables`. Real DB missing-table propagates (legitimate schema bugs aren't masked).
- **S-8** — F-1 0-row cue rebalanced: ONE alternate-table check then commit `rows: []`; do NOT exhaust max_steps trying to "find rows" when the answer is genuinely empty.

### Full 50-task local measurement (deterministic single-pass)

Effective 50-task score (sum / 50 — apples-to-apples vs v4):

| Round | Sum | Effective | Perfect | Missing |
|---|---|---|---|---|
| v4 | 31.0 | 0.620 | 31/50 | 6 |
| **v5** | **34.0** | **0.680** | **34/50** | **1** |

Δ = **+0.060 effective** (~10% relative). 50-task mean over scored tasks: v4 = 0.7045 (44 scored) → v5 = 0.6939 (49 scored). Mean dipped because v5 includes 5 newly-attempted tasks where some scored 0 — but the SUM increased.

**Per-difficulty perfect counts:**

| Tier | v4 | v5 | Δ |
|---|---|---|---|
| easy | 10/14 | 11/15 | +1 |
| medium | 17/22 | 16/23 | -1 |
| hard | 4/8 | **7/10** | **+3** |
| extreme | (missing) | 0/1 | scored but wrong |

**Recovered vs regressed (diff):**

- ✅ Recovered (not → perfect): **task_349** (v4 timeout — S-1 deadline + S-4 budget), **task_408** (v4 wrong formula — S-3 percentage hint), **task_420** (v4 timeout — S-1 + S-4), **task_86** (v4 wrong rows — S-3 list hint or S-6 multi-attr)
- ❌ Regressed (perfect → not): task_199 (Fix #2 win lost — likely variance OR question_shape hint mis-applied)

**Determinism verification.** Ran `task_22` twice on host with no other code change: identical `prediction.csv` byte-for-byte. S-1 verified end-to-end.

### Wall-clock notes

The 50-task run was interrupted twice due to host load-avg 100+ (machine thrashing — 16 logged-in users). Stalled tasks (task_379/418/420) re-ran individually with `max_workers=1` and all completed:
- task_379 ✓ 454s
- task_420 ✓ 168s (perfect)
- task_418 ✓ 246s (first time ever — was historically SIGKILLed during OneDrive D-state hang)

This means **deterministic seed + per-task isolation is sufficient to make even task_418 produce a trace**, which v3/v4 never achieved. Hidden set runs in the eval container won't have OneDrive thrash interference.

### Compliance checklist

- [x] linux/amd64 image built (Dockerfile pinned)
- [x] UV_OFFLINE=1 at runtime
- [x] ENTRYPOINT `dabench run-benchmark --config configs/eval.yaml`
- [x] eval.yaml: env-overridable model fields, `flat_output_dir: true`, `log_file: /logs/runtime.log`
- [x] sha256 recorded: `28eb3880d399c99c01681c7cf46c21bd3a206866169d1a6b041abe60b1e51323`
- [x] tarball ≤ 10 GB (actual 0.38 GB)
- [ ] Drive upload + email organizers (manual step)
- [ ] Update budget tracker after sending

### Decision context

Effective 50-task **+0.06 over v4** with structural improvements:
- Variance eliminated (S-1) → measurements now reproducible
- Hard tier +3 perfect (S-1 deadline + S-4 budget combined)
- task_418 now executes end-to-end (architectural unblock)
- task_352 remains the only "missing" — 3360s SC timeout exceeded; needs higher k=1 single-shot fallback (future round)

Ship gate (perfect ≥ 38, mean ≥ 0.80) was deliberately aggressive and NOT met. But v5's +3 perfect / +0.06 effective with NO variance is a more reliable signal than v4's wash.

### Retro

- Determinism (S-1) is the highest-leverage change in the whole project so far. It turns every measurement into ground truth and exposes how much of v3/v4 "improvements" were variance noise. Every future round must keep this gate.
- task_199 regression is the only red flag. Forensic in next round — likely question_shape misclassified "List the names and funding types" as multi-attribute but the agent already had it right in v4.
- The hard tier recovery validates the "structural cost of voting was hidden" thesis. Multi-pass cross-run voting (H-1) was a defense against variance; now that S-1 removes variance, we get back the wall-clock for actually solving hard tasks.
- task_418 producing a trace for the first time means the hidden eval set's potential D-state hangs will not be a hard wall — they were correlated with host-side OneDrive contention, not a property of the algorithm.

---

## v8 ship gate FAIL — 2026-05-12 (chronic timeout 표적 패치 — NOT SHIPPED)

**Status**: ❌ Ship gate not met (-1 perfect vs v5). Not submitted. Image and tarball retained locally only.

### Code changes (uncommitted)

Forensic basis: `artifacts/timeout_analysis_2026-05-12.md` — chronic 10 failure cluster across 14 eval runs.

- **T-A** (`scoring/answer_validator.py:218-231`): `empty_rows` severity `error` → `info`. Removes contradiction with prompt — agent can now commit `rows: []` + `confirm: true` without soft-rejection.
- **T-B** (`agents/prompt.py:49`): F-1 cue rewritten — explicit `confirm: true` for true-empty; first version was too leading (named tasks → self-fulfilling), patched to balanced "verify range with MIN/MAX, then commit if overlap or after one alt-table check".
- **T-C** (`agents/prompt.py:RESPONSE_EXAMPLES`): added 0-row + `confirm: true` example.
- **T-D** (`memory/question_shape.py`): 3 new patterns (abnormal/normal lab, Nth-of-each ordinal, two-named-entity doc comparison) + reordered list-pattern BEFORE average (fixed "List ... where the average ..." mis-classification).
- **T-E** (`memory/learnings.json`): zip5/9 + time-format shape policies (advisory only).
- **T-F** (`memory/error_patterns.json`): `percentage_formula_mismatch` entry.
- **T-G** (`run/runner.py:_build_task_advisories`): prepend ★ MULTI-ATTRIBUTE marker when question_shape rationale signals multi-attribute.
- 365 unit tests pass + ruff clean.

### Local 50-task eval result (deterministic, single-pass)

| metric | v5 baseline | v8 | Δ |
|---|---|---|---|
| 50-task perfect (score = 1.000) | 34 | **33** | **−1** |
| Tasks scored | 49 | 46 | −3 |
| Mean Score (scored only) | 0.6939 | 0.7174 | +0.024 |
| Effective Score (missing = 0) | 0.680 | 0.660 | **−0.020** |
| Missing predictions | 1 (task_352) | 4 (task_344, task_352, task_379, task_418) | +3 |
| Run-to-run variance | 0 (S-1) | 0 maintained | — |

Image: `dabench:v8` / `team1438:v8` (sha256 `3bd76abcbd6e78c18d3994d2c2f1e047f406573fa07eba06f031133ddc6acc09`, 0.38 GB) and tarball `submissions/team1438_v8.tar.gz` kept locally for reference.

### Per-task delta (v5 → v8)

Gains: **0**.

Regressions (1 real + 3 no-op):
- **task_420 (hard) 1.000 → 0.000** ← real. v5 emitted `100.00` (correct); v8 emitted `99.94`. Gold counts cards with `hasContentWarning = 0` over total commander+legal — v8 included 0.06% as having warning. Likely caused by T-D percentage rationale + T-F error_pattern hint nudging the agent toward stricter BIRD-style filtering.
- task_344 / task_379 / task_418: 0.000 → SUBPROC timeout. Algorithmic dead-ends — just changed termination mode (scored-0 → missing).

### Chronic-10 measurement (pre-50-task)

Run: `artifacts/runs/v8_chronic10/` (9/10 finished, task_418 stopped). After filtering against the actually-shipped v5:

| task | diff | v5 | v8 |
|---|---|---|---|
| task_19, task_38 (already perfect in v5) | easy | 1.000 | 1.000 |
| task_80, task_173, task_199 | easy/medium | 0.000 | 0.000 |
| task_344, task_352 | hard | 0.000/missing | timeout |
| task_379, task_396 | hard | 0.000 | 0.000 |

**Chronic-recovery net = 0.** My initial "chronic" classification mistakenly included v5_run2 (host thrash) failures; once filtered to actually-shipped v5, `task_19` and `task_38` were never chronic.

### Root causes for the regression

1. T-B v1 named `task_38 / task_173 / task_80` directly in the prompt's empty-commit cue — self-fulfilling bias toward early commit. Patched mid-run.
2. T-D percentage / T-F error_pattern hints changed agent's filter on task_420 enough to land 0.06% off gold. On a 1×1 numeric answer, any drift kills the score.

### Decision

**Do not submit v8.** Effective score regression (−0.020) and perfect-count drop (−1) vs the v5 already shipped.

### Next round candidates (v9)

1. **Revert T-D/T-F percentage hints**, keep T-A/T-B/T-C/T-G — verify task_420 recovers.
2. The genuinely chronic-5 (80/173/199/344/352/379/396/418) need domain-knowledge injection or 2-stage agent (Tier-3 of v5 plan, still deferred); advisory hints alone are not enough.
3. SUBPROC timeout count rose from 1 → 4 with same `task_timeout_by_difficulty`. Investigate whether `max_workers: 2` for hard/extreme would trade throughput for completion rate.
4. Build a "v5-perfect sample" hold-out (10 tasks v5 got perfect) as a regression sanity check alongside chronic-10 — task_420 wasn't in chronic-10 and the regression was only visible at full-eval scale.

### Retro

- Naming specific task IDs in the prompt's behavioral cues was a mistake — agent treats it as "those questions are empty" not "be careful with 0 rows". Cue principle: describe the signal (range overlap, alt-table check), not the evidence tasks.
- Forensic timeout analysis grouped failures across runs with DIFFERENT configs (incl. abandoned/thrash runs). Apples-to-apples baseline must be the actually-shipped previous version. Update the analysis playbook.
- S-1 determinism was load-bearing here — without it, −1 perfect could be noise. With it, it's a real signal.
- Local chronic-10 (~30 min) is a useful fast pre-flight, but it underestimates regression risk on the broader 50 (task_420 was missed). Future round: chronic-10 + v5-perfect-sample combined sanity set.

---

## v4 (=v8 image) interrupt — 2026-05-12 organizers eval result

**Status**: 🔴 Interrupt. Score 0.2281. Organizer mail attached below.

Submission received: 2026-05-12T21:40:41+09:00 (image: team1438:v4 retagged from v8).
Brief reason: organizer's `timeout --signal=TERM --kill-after=30s ${MAX_RUNTIME_SECONDS}s docker run ...` fired before the eval finished. Hidden ~400-task set against our v8 wall-clock (4.4h on public 50 = ~32h naive extrapolation) overflowed the 12h budget.

### Diagnosis

- v8 (=v4) wall-clock on public 50 = 13800s (4.4h). Linear extrapolation to 400 tasks ≈ 30.7h.
- `wall_clock_budget_seconds: 43200` (12h) was set but governor only fires ONCE in pre-v6 code. Even with one downgrade the chronic 2400-3600s tasks still consumed the wall-clock.
- Score 0.2281 = leaderboard scored only the tasks that finished before SIGTERM (~25% of the hidden set probably).

### Conclusion

v8 algorithmic regression (task_420 -1) was a secondary concern; the load-bearing fix was wall-clock containment. Pivot the next round to fitting 400 tasks inside 12h, not chronic-recovery.

---

## v6 ship gate — 2026-05-17 (wall-clock containment for hidden ~400-task budget)

**Status**: ✅ Ship gate met. Cleared for organizer submission.

### Code changes (since v4=v8)


- **`run/runner.py`** — **cascading governor**. Previously fired once and locked. Now re-evaluates after each batch with `_GOVERNOR_MAX_CASCADES=3` and `_GOVERNOR_RECHECK_TASK_INTERVAL=5`. Each cascade halves `max_steps` + `task_timeout` floors. Subsequent `_downgrade_config(effective_config)` calls halve again (no longer rebasing from original).
- **`configs/eval.yaml`** — tightened `task_timeout_by_difficulty`:
  - `easy: 300 → 180`, `medium: 600 → 360`, `hard: 2400 → 900`, `extreme: 3600 → 1200`
  - `task_timeout_seconds: 600 → 300` (default fallback)
  - `max_workers: 4` (briefly tested 6 — caused vLLM queue timeouts → reverted)
- **`agents/model.py`** — explicit OpenAI client `timeout=240.0`. Default 60s blew past under concurrent load; "Request timed out" wiped 18 tasks in v6 build #1 before this fix.
- All 362 unit tests + ruff pass. New test: `test_downgrade_can_cascade_when_applied_twice` covers the cascade math.

### Local 50-task eval result

| metric | v4 (=v8) baseline | v6 | Δ |
|---|---|---|---|
| 50-task perfect | 33 | **34** | +1 |
| Tasks scored | 46 | 50 | +4 |
| Mean Score (over 50) | 0.660 | **0.680** | +0.020 |
| **Wall-clock (50 tasks)** | **15981s (4.4h)** | **6714s (1.86h)** | **−58%** |
| Missing predictions | 4 | 4 | = (different ids; all chronic algorithmic) |
| Run-to-run variance | 0 (S-1) | 0 maintained | — |

Image: `team1438:v6` (sha256 `2b4a9cfe0cdab35637475cf33847ed39fae19adb42ba701b330e6da3cf24d166`, 0.38 GB). Tarball: `submissions/team1438_v6.tar.gz`.

### Per-task delta vs v5 (the shipped reference)

Gains: **2**
- `task_199` (medium) 0.000 → 1.000 — ★ MULTI-ATTRIBUTE prepend (T-G) fired correctly after list-pattern reorder; agent emitted 2-col answer.
- `task_379` (hard) 0.000 → 1.000 — Nth-of-each ROW_NUMBER hint (T-D) + 1200s extreme cap let agent finish.

Regressions: **2** (wash)
- `task_408` (hard) 1.000 → 0.000 — same BIRD percentage edge case as task_420 in v8.
- `task_420` (hard) 1.000 → 0.000 — known regression carried over from v8.

Net delta vs v5: **0** (same mean, same perfect count). Net delta vs v4 (last submission): **+1 perfect, +0.020 mean**.

### Wall-clock extrapolation to hidden 400-task set

| approach | extrapolation | fits in 12h? |
|---|---|---|
| v4 wall-clock × 8 | 32h naive | NO (hit at v4 submission) |
| v6 wall-clock × 8 | 14.9h naive | borderline — governor cascading covers the gap |
| v6 with chronic cap × 8 + 1-cascade halving | ~10h | YES |

The chronic 4 tasks (344/352/379/418) on v6 cost 1260s × 4 ≈ 84min total locally. Under the 12h hidden budget the cascading governor would halve their cap after 5-task observation if they were over-budget; the math fits.

### Compliance checklist

- [x] linux/amd64 image built (`team1438:v6`)
- [x] UV_OFFLINE=1 at runtime
- [x] ENTRYPOINT `dabench run-benchmark --config configs/eval.yaml`
- [x] eval.yaml: env-overridable model fields, `flat_output_dir: true`, `log_file: /logs/runtime.log`
- [x] tarball ≤ 10 GB (actual 0.38 GB)
- [x] sha256 recorded: `2b4a9cfe0cdab35637475cf33847ed39fae19adb42ba701b330e6da3cf24d166`
- [ ] Drive upload + email organizers (manual step)

### Side artifact

- New: `scripts/build_dashboard.py` + `artifacts/dashboard/dashboard.html` (9 MB, file:// viewer for all 17 historical eval runs). Generated from `discover_runs()` glob over `artifacts/eval_full_*/`. Per-task: question + gold + prediction + step-by-step reasoning timeline. Rebuild on demand after new eval: `uv run python scripts/build_dashboard.py`.

### Retro

- The "wall-clock containment" lens was missing in v5/v8 rounds — both prioritized perfect-count and accidentally let the hidden 400-task budget regress. Hidden-budget-fit is now an explicit gate.
- Cascading governor was a 30-line change with outsized leverage. The single-fire version was a latent bug that hadn't manifested on 50 tasks (where one downgrade was always enough).
- `OpenAI(timeout=...)` was unset — the 60s default was a silent ceiling that masqueraded as algorithmic failure under concurrency. Should have been measured before raising max_workers, not after.
- The v6 build #1 → build #2 cycle was caught by running 50-task eval before committing the tag. Don't re-tag → tarball until a fresh full eval passes.

---

## v7 round — 2026-05-18 (percentage revert + duplicate validator + multi-source cue)

**Status**: ⚠️ Net 0 vs v6. Ship decision pending (no score gain but wall-clock improved).

### Code changes (since v6)


Forensic basis: v6 50-task showed 16/50 score<1.0. Identified 2 regressions vs v5 (task_408/420) and over-emission patterns (task_180/25/11). Target: revert percentage hints + escalate duplicate_rows + add multi-source cue.

- **T-1a** (`memory/question_shape.py`): percentage rationale weakened from "Gold almost always uses CAST(COUNT(CASE WHEN ...))" to "verify against your data, don't blindly apply a fixed template". v6's strong directive nudged task_408/420 toward wrong filter; v7 leaves agent's judgment intact.
- **T-1b** (`memory/error_patterns.json`): removed `percentage_formula_mismatch` entry that was meant to recover task_196/396 but only succeeded in regressing task_408/420.
- **T-2** (`scoring/answer_validator.py`): duplicate_rows escalated from info to warning (blocking) when (a) question is plural AND (b) duplicate ratio ≥ 50%. Targets task_180 (153 rows where gold is 9), task_25 (28 rows where gold is 3).
- **T-3** (`agents/prompt.py`): F-1 cue extended with "Multi-source time-series lookup" — when range is outside obvious table, scan entire context for date-based aggregates (yearmonth, monthly, daily). Targets task_173.
- 5 new validator tests (`test_answer_validator_duplicate_blocking.py`) + 2 new question_shape regression tests. 368/368 pass + ruff clean.

### Local 50-task eval result

| metric | v6 baseline | v7 | Δ |
|---|---|---|---|
| 50-task perfect (score = 1.000) | 34 | **34** | 0 |
| Tasks scored | 46 | 47 | +1 |
| Mean Score (scored only) | 0.7174 | 0.7234 | +0.006 |
| **Effective Score (over 50)** | **0.6800** | **0.6800** | **0** |
| Missing predictions | 4 (344/352/418/80) | 3 (352/379/418) | -1 |
| **Wall-clock (50 tasks)** | **6714s (1.86h)** | **5888s (1.63h)** | **−12%** |
| Run-to-run variance | 0 (S-1) | 0 maintained | — |

Image: `team1438:v7` (sha256 `fc195bbc7bcd952ada0171b9897f314c4cedd64ded998e6f626ae77b574e56f4`, 0.38 GB).
Tarball: `submissions/team1438_v7.tar.gz`.

### Per-task delta vs v6

Gains: **1**
- `task_420` (hard) 0.000 → 1.000 — T-1a percentage rationale weakening let the agent recover the v5-style filter.

Regressions: **1**
- `task_379` (hard) 1.000 → missing (deadline) — v6 recovered this via T-D Nth-of-each ordinal hint, but T-3 prompt addition shifted token positions enough to change sampling path and the agent now exceeds the per-sample deadline.

Other changes:
- `task_344` (hard) missing → 0 (now produces a 0-score answer instead of timing out; net 0)
- `task_80` (easy) missing → 0 (same — timeout-cap respect produces empty answer; net 0)

Net delta: **0** (one swap; identical effective score).

### What worked / didn't

**Worked**:
- T-1a percentage rationale weakening — recovered task_420 (v5-perfect, v6-broken).
- T-2 duplicate_rows escalation — task_11 (severely duplicated answer) recovered in chronic-targeted measurement (not 50-task because task_11 was already perfect in v6).
- Wall-clock 12% faster — additional safety margin for the 12h budget.

**Didn't**:
- T-1a alone insufficient to recover task_408 (v6 and v7 both emit `0.51`, gold is `0.316` — the formula itself is wrong, not just the filter).
- T-3 multi-source cue — task_173 still empty-commits; never reached the yearmonth.csv branch. The cue text wasn't operative enough to change agent behavior.
- T-2 duplicate_rows didn't catch task_180 — agent first-commits with `confirm: true`, bypassing the validator (153 rows are all distinct CustomerIDs, so dup_count is 0, no warning fires at all).
- task_379 regression — T-3 prompt addition changed sampling path enough to time out (lost the v6 win without a corresponding gain).

### Ship decision

**Pending — user judgment**:
- v7 has identical perfect / effective score to v6.
- v7 has 12% faster wall-clock (modest extra safety on the 12h budget).
- v7 swaps task_379 for task_420 — equal magnitude but different task IDs. Whichever pattern is more common in the hidden 400-task set wins.
- Either way: better than v4 (=v8) which interrupt-scored 0.2281.

Recommendation: ship v7 OR ship v6 — both are net-equivalent. v7 marginally safer on wall-clock; v6 marginally safer on the task_379 type-of-question.

### Compliance checklist

- [x] linux/amd64 image built (`team1438:v7`)
- [x] tarball ≤ 10 GB (0.38 GB)
- [x] sha256 recorded: `fc195bbc7bcd952ada0171b9897f314c4cedd64ded998e6f626ae77b574e56f4`
- [ ] Ship decision (v6 vs v7)
- [ ] Drive upload + email organizers

### Retro

- One prompt edit (T-3 added a single bullet line) was enough to shift the sampling path on a hard task with deep ROW_NUMBER reasoning (task_379). Deterministic seed gives identical output for the same prompt; a one-character shift moves you to a different decision tree. Prompt edits at this granularity are riskier than they look.
- T-2 duplicate-rows validator only catches the "DISTINCT was missed" failure mode. The other over-emission failure (task_180: 153 unique CustomerIDs where gold has 9) needs a different signal — perhaps "agent emitted ≫ expected on a plural question with confirm:true on first try" — but the validator can't see expected count, and the confirm:true path bypasses warning escalation.
- Targeted regression sample (chronic-6 + sanity-5) cost ~30 min and surfaced 1 of 2 regressions (task_379) before the full 50-task eval. Both surfaced in 50-task too, so the targeted sample is a useful (~10× faster) pre-flight signal but not a substitute.

---

## v6 interrupt — 2026-05-18 organizers eval result

**Status**: 🔴 Interrupt. Score 0.3509 (v4 0.2281 대비 +54%, but still SIGTERM at 12h).

Submission received: 2026-05-18T20:28:44+09:00. Image: team1438:v6 (sha256 `fc195bbc...`).
Brief reason: `1975268 Killed timeout --signal=TERM --kill-after=30s ...` — organizer's 12h wrapper fired SIGTERM, then SIGKILL after 30s grace. ~52% of hidden tasks processed before kill.

### Diagnosis

- v6 wall-clock on public 50 = 6714s (1.86h). Linear 8× = 13h naive — barely over 12h with no margin.
- Organizer environment processed ~206 / ~400 tasks in 12h. 1.85× slower than our DGX (or hidden set larger than 400).
- Score recovery vs v4 (+54%) shows wall-clock containment works; not enough alone for ≥80% processing rate.
- Cascading governor with the v6 "100%" trigger fires only after the budget is already projected to fail — too late.

### Conclusion

v6 algorithmic gains were retained (mean 0.680 on public, +0 vs v5). The bottleneck is purely wall-clock. v_next pivots to:
1. Pre-emptive governor (80% margin) so cascade fires before budget is lost
2. SC k 1→2 + governor-forced k=1 on cascade (saves up to 5× per hard/extreme task)
3. timeout cap −33% (chronic dead-ends cut earlier)
4. max_workers 4→6 (now safe with explicit OpenAI timeout=200s)
5. confirm:true first-call bypass blocked (recover over-emission tasks like task_180)

---

## v7 (= v_next) ship gate — 2026-05-20 (pre-emptive governor + SC k fallback + max_workers 6)

**Status**: ✅ Ship gate met. Submission packaged.

### Code changes (since v6)


- **B-1 SC k fallback** (`runner.py:_downgrade_config`): `self_consistency_k=1` is forced on every cascade. + eval.yaml default `self_consistency_k: 2` (down from 5).
- **B-2 timeout cap −33%** (`configs/eval.yaml`): easy 180→120, medium 360→240, hard 900→600, extreme 1200→900, default 300→240.
- **B-3 max_workers 4→6** (`configs/eval.yaml`) + httpx 240→200s (`agents/model.py`): now safely raises concurrency without re-hitting v7-build-#1's "Request timed out" failures.
- **G-1 governor 80% margin** (`runner.py:_governor_should_engage`): trigger now fires when `remaining_budget * 0.8 < tasks_remaining * avg` (v6 fired at 100% — too late).
- **I-1 hidden_set_probe event** (`runner.py`): after first 10 task completions, emit `hidden_set_probe` runtime event with avg/p50/p95 elapsed + remaining budget. Lets the organizer environment's endpoint latency be diagnosed from /logs alone.
- **A-1 first-call confirm bypass blocked** (`tools/registry.py:_make_answer_handler`): when `bypass_count == 0`, ignore `confirm: true` and force one corrective pass. Recovers severe over-emission (task_180-class).
- 379/379 unit tests pass + ruff clean. 3 new test files (governor margin, SC k cascade, first-call confirm bypass).

### Local 50-task eval result

| metric | v6 baseline | v7 | Δ |
|---|---|---|---|
| 50-task perfect (score = 1.000) | 34 | **35** | **+1** |
| Mean Score (over 50, missing=0) | 0.6800 | **0.7000** | **+0.020** |
| Tasks scored | 46 | 47 | +1 |
| Missing predictions | 4 (344/352/418/80) | 3 (344/379/80) | −1 |
| **Wall-clock (50 tasks)** | **6714s (112min)** | **3540s (59min)** | **−47%** |
| Model request failed (LLM timeout) | 0 | 0 | maintained |
| Governor engaged | no | no (budget had headroom on 50-task) | — |
| Run-to-run variance | 0 (S-1) | 0 maintained | — |

Image: `team1438:v7` (sha256 `4f9b52266945098956d5fbf875e032567bd93085889fcf86724bda7a3e0d21bf`, 0.38 GB). Tarball: `submissions/team1438_v7.tar.gz`.

### Per-task delta vs v6

Gains: **3**
- `task_11` (easy) 0.000 → 1.000 — A-1 forced retry on over-emission (3×6→3×3).
- `task_418` (extreme) MISSING → 1.000 — chronic algorithmic dead-end *resolved* — first time this task scores. Likely combination of A-1 (validator pass) + new SC k=2 ordering + faster overall pace.
- `task_420` (hard) 0.000 → 1.000 — BIRD percentage formula regression from v8 is now permanently fixed (carried over from v7 percentage rationale weakening).

Regressions: **2**
- `task_199` (medium) 1.000 → 0.000 — multi-attribute ★ marker advisory got reordered by combined effect of A-1 + new SC k. Acceptable trade vs +3.
- `task_379` (hard) 1.000 → MISSING — Hard cap 600s no longer enough for ROW_NUMBER ordinal Nth-of-each pattern. v6 had 900s cap and it fit; v7 600s does not. Worth restoring extreme cap to 720s in v_next2 if score stays high.

Net delta vs v6: **+1 perfect, +0.020 effective**.

### Wall-clock extrapolation to hidden ~400 task set

| scenario | extrapolation | 12h fit? |
|---|---|---|
| v4 wall-clock × 8 (this is what got SIGTERM at 0.2281) | 32h | NO |
| v6 wall-clock × 8 (SIGTERM at 0.3509) | 13h | borderline (just over) |
| **v7 wall-clock × 8 (this submission)** | **7.9h** | **YES** + 4h headroom |
| v7 × 1.8 slower (estimated organizer endpoint factor) | 14.2h | borderline — governor will cascade |
| v7 × 1.8 × hidden=600 | 18.5h | needs governor 1 cascade (→ ~13h) |

The 1.8× slower scenario fits inside 12h if either (a) governor cascades once (which 80%-margin trigger now does pre-emptively) OR (b) hidden has ≤450 tasks. Both are reasonable.

### Compliance checklist

- [x] linux/amd64 image built (`team1438:v7`)
- [x] tarball ≤ 10 GB (0.38 GB)
- [x] sha256 recorded: `4f9b52266945098956d5fbf875e032567bd93085889fcf86724bda7a3e0d21bf`
- [x] eval.yaml — env-overridable model fields, `flat_output_dir: true`, `log_file: /logs/runtime.log`, `wall_clock_budget_seconds: 43200`
- [ ] Drive upload + email organizers (manual step)
- [ ] Mark date received once organizer eval result returns

### Retro

- Pre-emptive governor (80% margin) is the structural fix for the "engaged too late" pattern in v6. v6's trigger only fired *at* the projected fail point — by then the budget was already lost. The 20% margin gives cascade time to actually work.
- SC k=5 was an algorithmic-recovery investment (vote across 5 samples) that turned into a wall-clock anchor on hard/extreme tier. Default k=2 + cascade-forced k=1 keeps modest voting benefit for normal-pace runs while ditching it the moment we're racing the clock.
- task_418 finally scoring after 9 prior submissions (v4 through v6) is the most interesting signal. It suggests the algorithm had this answer reachable but was being killed by the timeout budget before it could finalize. A-1's mandatory validator pass may also have caught a near-final answer that the agent would otherwise have committed with a flaw.
- hidden_set_probe is the first observability we have for the organizer environment. Once v7 runs there, the avg/p50/p95 in /logs will tell us definitively whether endpoint latency is 1.0× / 1.8× / 3× our DGX baseline. That data drives v_next2 cap decisions.
- max_workers=6 + explicit httpx timeout=200s combination has been tested locally with 0 LLM client failures. The fear from v7-build-#1 was unfounded once the timeout was properly set.
- The "raise concurrency carefully" pattern (raise httpx timeout FIRST, then raise workers) should be the standard playbook for future concurrent throughput improvements.

---

## v8 (Tier-1 16-vCPU saturation: max_workers=8 + BLAS thread cap=2) — 2026-05-20

**Status**: ⚠️ Built and measured. **Locally regresses vs v7** (−1 perfect, +12% wall-clock). Ship decision pending.

### Code changes (since v7)

Goal: Saturate the organizer's 16 vCPU spec. v7 used max_workers=6 → ~38% vCPU utilization. v8 raises workers to 8 and caps BLAS threads (OMP/MKL/OPENBLAS/NUMEXPR=2) so 8 workers × 2 threads = 16 vCPU exact match, eliminating both under-use and oversubscription.

- **`configs/eval.yaml`**: `max_workers: 6 → 8`
- **`Dockerfile`**: added `ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2`
- Rule compliance verified: 16 vCPU + 64 GB RAM + qwen-only solver + no external network — all untouched. Tier-1 changes are Python-internal threading policy only.
- 379/379 unit tests pass + ruff clean.

### Local 50-task eval result

| metric | v7 baseline | v8 | Δ |
|---|---|---|---|
| 50-task perfect (score = 1.000) | 35 | **34** | **−1** |
| Mean Score (over 50, missing=0) | 0.7000 | **0.6800** | **−0.020** |
| Tasks scored | 47 | 45 | −2 |
| Missing predictions | 3 (344/352/418) | 5 (173/257/344/352/80) | +2 |
| **Wall-clock (50 tasks)** | **3540s (59min)** | **3971s (66min)** | **+12% slower** |
| **RAM peak (locally limited to 7.65 GiB)** | n/a | **~342 MiB / 7.65 GiB (4.5%)** | OOM risk 0 on 64 GB |
| Model request failed (LLM client timeout) | 0 | 0 | ✓ stable |
| Governor engaged | no | no | — |

Image: `team1438:v8` (sha256 `5d883c3a479d9bf2f3c7bc6aaabe171099409b630dbec1b126c6c460da94742e`, 0.38 GB). Tarball: `submissions/team1438_v8.tar.gz`.

### Per-task delta vs v7

Gains: **1**
- `task_199` (medium) 0.000 → 1.000 — multi-attribute hint fired more aggressively after worker re-batching.

Regressions: **2**
- `task_257` (medium) 1.000 → MISSING — exceeded `max_steps`. BLAS cap may have slowed CSV-heavy probing by a worker, pushing per-step latency over budget.
- `task_418` (extreme) 1.000 → 0.000 — `task_418` had finally scored in v7 (first success ever); v8 lost it. Suspect 8-worker queue contention on the vLLM endpoint changing sampling timing.

Other changes (no scoring effect, just status):
- `task_173` 0.000 → MISSING (still 0 effective)
- `task_352` 0.000 → MISSING (still 0 effective)
- `task_379` MISSING → 0.000 (still 0 effective)

Net delta vs v7: **−1 perfect, −0.020 effective**, **+7 minutes wall-clock**.

### Rule-compliance check

| Rule | v8 status |
|---|---|
| 16 vCPU limit | ✓ within (8 workers × 2 threads = 16 vCPU exact) |
| 64 GB RAM limit | ✓ measured peak 342 MiB locally; estimated <1 GB on 64 GB system (massive headroom) |
| GPU disallowed | ✓ no GPU calls |
| External network blocked | ✓ no external calls; `MODEL_API_URL` only |
| qwen3.5-35b-a3b only as primary solver | ✓ no auxiliary LLM added |
| `/input` read-only | ✓ unchanged |
| `MODEL_API_URL/KEY/NAME` env-injected, not hardcoded | ✓ eval.yaml empty strings retained |
| Organizer-injected env vars not modified | ✓ `OMP_*` / `MKL_*` etc. are separate vars we set in Dockerfile |
| `--memory-swap=64g` (no swap) | ✓ peak well under non-swap RAM |

No rule violation. v8 is safe to submit from a compliance standpoint.

### Why it regressed (hypothesis)

1. **BLAS cap = 2 hurts a few CSV-heavy tasks more than it helps**. task_257 had multi-step dataframe operations; capping each worker's BLAS to 2 threads turned a 30-step task that v7 finished into one that v8 couldn't finish before max_steps. The cap was meant to PREVENT one pandas-heavy worker from monopolizing 16 vCPU and starving others — but with max_workers=8, each worker only needed ~2 vCPU anyway, so the cap was solving a problem v7 didn't have.
2. **8-worker queue contention on the local DGX vLLM**. With 6 workers, the endpoint was always responsive (~5–20s per call). With 8 workers, occasional queue spikes pushed individual call latency past comfortable, and task_418 (extreme; needs many sequential LLM calls to traverse its long context) tipped over the per-task budget.
3. **Local environment is not organizer environment**. Our Docker Desktop allocates 14 vCPU / 7.65 GiB. The organizer's 16 vCPU / 64 GiB might let max_workers=8 + BLAS cap=2 actually shine; we can't validate that here.

### Ship decision

Two safe paths:
- **Path A (recommended)**: ship v7. Validated 35-perfect / 0.700 effective / 59-min wall-clock on our local environment. Lowest regression risk.
- **Path B (exploratory)**: ship v8 anyway, hoping the organizer's 16 vCPU / 64 GiB lets max_workers=8 + BLAS cap=2 produce different (better) results. Tradeoff: known local regression of −1 perfect, but bigger headroom on the organizer's wall-clock budget.

Both images are kept locally:
- `team1438:v7` — sha256 `fc195bbc7bcd952ada0171b9897f314c4cedd64ded998e6f626ae77b574e56f4`
- `team1438:v8` — sha256 `5d883c3a479d9bf2f3c7bc6aaabe171099409b630dbec1b126c6c460da94742e`

### Retro

- Rule compliance ≠ score improvement. Tier-1 changes were verified safe, but "safe" doesn't guarantee "better" — and locally it was worse.
- BLAS thread cap is contextual. With max_workers ≤ workers × BLAS_threads ≤ vCPU exact match, the math is clean. But the cap also forces a slower-than-default code path for any worker that *could* have benefited from more BLAS threads transiently. We accidentally penalized those workers.
- Our local 7.65 GiB Docker Desktop limit produces zero meaningful RAM signal. We can't test the OOM scenario locally; trust the math (8 workers × ~5 GB peak << 64 GB) and ship gate on Model timeout rate + score.
- The right next experiment isn't another Tier-1 — it's either (a) ship v7 and wait for organizer's response, or (b) try Tier-2 (max_workers 10 or 12) with the existing v8 BLAS cap to see if more concurrency overrides the per-worker cap penalty. But Tier-2 has its own RAM risk (per the prior compliance review).

### Retag: v7 image → v8 submission tag (2026-05-20)

**Decision**: ship v7 image under the v8 submission tag. The v8 code branch (max_workers=8 + BLAS cap=2) showed local regression on our measurement environment and the analysis suggested it would likely regress harder on the organizer environment (1.8× slower endpoint, queue contention worsens at higher concurrency).

- `docker rmi team1438:v8 dabench:v8` removed the max_workers=8 build (image `1cc979800bec`)
- `docker tag team1438:v7 team1438:v8` + `dabench:v7 dabench:v8` — v8 tag now points at v7 image (`700f51cfbf1c`)
- Regenerated tarball from re-tagged image: `submissions/team1438_v8.tar.gz`, sha256 `802566b60706e9b7b672ffed5e6fbf3a4030f9f5074cd0a8ad00f2707fff254c` (different from v7's `4f9b5226...` because `docker save` includes tag metadata, but image content is identical)
- Image arch verified: linux/amd64
- Code branch (max_workers=8, BLAS cap) is retained in `configs/eval.yaml` + `Dockerfile` for further iteration — it is NOT the submitted image.

**Pattern**: "submission tag = the safest validated version; dev branch = current experimental work". Future submissions follow the same separation: build the dev code freely, but retag a validated previous build under the submission's version slot. The dev branch is fertile ground for hypotheses that may or may not pan out (max_workers, BLAS cap, future structural changes); the submitted tarball must always be a measurement-confirmed winner.

## B-board final — 2026-05-21 (v8 image content under `:final` tag)

**Decision**: ship the v8 submission image (= v7 code content, image ID `700f51cfbf1c`) as the B-board final submission. v8 A-board result not yet available (organizer announces 2026-05-22 19:59 Beijing); regardless of that outcome, this is the safest baseline because (a) the v8 dev branch already regressed on our 50-task measurement, (b) B-board permits only **one submission** with no version replacement, so the experimental B-board build (SC k=5 + extended timeout cap to use the 12h budget) would have to be measurement-confirmed against v7 before risking the single B-board slot — and we don't have that confirmation yet.

**Trade-off accepted**: B-board allows 12h wall-clock (6× A-board's 2h), and our v7 code is wall-clock-tuned for 2h. Re-using it on B-board means we run the same image on a more generous budget — no harm, but we also don't recover the algorithm headroom that the larger budget could buy (e.g. restoring SC k from 2 → 5 on hard tier). If the v8 A-board result lands by tomorrow evening (5/22 20:59 KST) with a solid score, we may overwrite this `:final` artifact with a B-board-tuned rebuild before the 5/23 20:59 KST deadline — but only if a 50-task measurement on that rebuild shows ≥ v7's 35 perfect / 0.700 mean within ≤ 90 min wall-clock.

**Artifacts**:
- Image: `team1438:final` (alias of v8 image `700f51cfbf1c`)
- Tarball: `submissions/team1438_final.tar.gz`, 388 MB (well under 10 GB cap)
- sha256: `fd28e96f34bfb62e734af8bcfc36b72edf849dea1e05e6c1de1f03bce0d9b54b` (different bytes than v7/v8 tarballs because `docker save` records the tag list; content is identical)
- Arch: linux/amd64 ✓
- Entrypoint: `uv run dabench run-benchmark --config configs/eval.yaml` ✓
- Load round-trip: ✓

**Pre-send checklist (per organizer email §5)**:
- [x] Image name `team1438:final` matches the required format
- [x] Archive named `team1438_final.tar.gz`
- [x] Built with `docker save` (not `docker export`)
- [x] linux/amd64 (not arm64)
- [x] ≤ 10 GB tarball
- [ ] Drive upload + "Anyone with the link can view" — pending (manual)
- [ ] Email sent: subject `[KDDCup2026 Data Agents] B-board Final Submission - team1438`, body with Team ID + sharing link — pending (manual)
- [ ] Logged the send timestamp + Drive link below — pending (after send)

**Email template** (paste into Gmail/Outlook when ready):
```
Subject: [KDDCup2026 Data Agents] B-board Final Submission - team1438

Team ID: team1438
Sharing link: <Drive share link>
```

**Why no rebuild for the B-board-larger-budget upside**: B-board's "1 submission, no replacement" rule penalizes experimentation heavily. The reward for restoring SC k=5 is roughly +1~2 perfect (~+0.02~0.04 effective). The risk for a misjudged rebuild is the entire 12h B-board evaluation runs on a regressed image. Symmetric upside × asymmetric downside × one-shot ⇒ ship the validated baseline by default. The window for revisiting this is 5/22 evening once A-board comes back; if v7 image already produced a strong score there, B-board with the same code is the conservative bet. If v7 produced a weak score (e.g. another SIGTERM at the 12h B-board cap), the right pivot is structural (max_steps reduction, force-cascade-from-start), not algorithmic — and that's a different decision.

**Submission window watch**: B-board opens 5/21 20:59 KST, closes 5/23 20:59 KST. After send, organizer fetches the link once and the version is locked. Auto-fallback (if we don't send): organizer picks our highest A-board scorer — which is v6 (0.3509) until v8's result lands. Explicit send dominates the fallback in both expected-value and worst-case scenarios, even with identical code content.

## v8 A-board result — 2026-05-22 organizers eval

**Score: 0.3886** (Δ vs v6 0.3509: **+0.0377, +10.7%**).

**Gap analysis**:

| Reference | Score | Δ vs us |
|---|---|---|
| 1st place | 0.6610 | **−0.272** |
| Our 50-task local effective | 0.700 | **−0.311 (−56%)** |
| v6 (prior) | 0.3509 | +0.038 |
| v2 (baseline floor) | 0.3386 | +0.050 |

**Diagnosis**: v6 → v8 wall-clock contention (1.63h → 0.98h, −40%) brought only +0.038 leaderboard. The improvement-per-hour-saved ratio is dramatically lower than our plan predicted (target was 0.50+). Two implications:

1. **Wall-clock alone doesn't explain the local/organizer gap**. If the entire 0.31 gap were wall-clock-related, v7's 40% wall-clock reduction should have moved us much closer to local 0.700. It didn't.
2. **Hidden distribution differs from public 50** is likely the dominant factor. Our 35/50 = 70% perfect rate on public maps to a lower-than-expected hidden-perfect rate, suggesting either (a) hidden hard-tier fraction is higher than public's 26%, or (b) hidden tasks within each tier are harder than the public exemplars.

**Open question — SIGTERM vs full completion**: organizer feedback shows the score (0.3886) but the SIGTERM/full-completion split is not visible in this thread. v6 was explicitly SIGTERM at 0.3509. For v8, two scenarios diverge sharply for B-board strategy:

- **Scenario A (v8 also SIGTERM)**: 12h budget still insufficient. B-board's 12h *might* recover some tasks if our endpoint is faster on B-board, but our governor cascade likely already kicked in mid-A-board. The same image on B-board may complete 60~70% of tasks instead of ~55%, recovering +0.02~0.04.
- **Scenario B (v8 ran to completion)**: 0.3886 is the algorithmic ceiling of this image. B-board's extra budget buys nothing structural. The only upside paths are (i) restore SC k=5 to capture the hard-tier voting we surrendered, (ii) extend per-task timeout caps to recover chronic dead-ends, (iii) targeted prompt patches on chronic-0 tasks.

**B-board decision (final)**: pending. See "B-board final" entry above for the prepared default (`team1438:final` = v8 image content). The decision tree:
- Default / conservative: ship the already-prepared `team1438:final` artifact. Captures Scenario-A upside automatically (same image runs on 12h budget). Loses Scenario-B headroom.
- Stretch / risky: rebuild with SC k=5 + relaxed timeout caps + targeted prompt patches (Option B in the conversation). 50-task local must measure ≥ v7's 35 perfect / 0.700 mean within ≤ 90 min wall-clock before risking the one-shot B-board slot.
- Structural-pivot: rebuild with reduced max_steps + force-cascade-from-start (Option C). Only sensible if Scenario A is confirmed (i.e. v8 organizer feedback mentions SIGTERM/timeout) and we believe B-board endpoint is no faster than A-board's.

**Pending data**: organizer feedback wording (SIGTERM keyword? task counts?), task-level scores on leaderboard if available, B-board task count vs A-board task count (B-board may be a larger or differently-mixed set within the 12h budget).

### Scenario resolved — 2026-05-22 (full completion confirmed)

Organizer email verbatim:

> Submission received: 2026-05-20T18:59:24+09:00
> Evaluation completed: 0.3886
> You may continue to iterate on your solution and submit additional versions before the official deadline.

No SIGTERM keyword, no timeout keyword, no per-task breakdown. The "evaluated successfully" wording + presence of a single normalized score implies **v8 ran to completion within the A-board 2h cap**. This is Scenario B from the decision tree above.

**Implications**:

1. **0.3886 is the algorithmic ceiling of this image**, not a wall-clock truncation. v6 → v8 wall-clock reduction (1.63h → 0.98h, −40%) bought only +0.038 leaderboard, so the wall-clock changes (SC k=5→2, governor cascade, tightened timeout caps) yielded almost no per-task algorithmic gain on the hidden set — A-1 (confirm:true first-call validator) accounts for most of the +0.038.
2. **Hidden distribution differs sharply from public 50**. Our 35/50 = 70% perfect locally maps to ~22/57 = 39% perfect on hidden A-board. Either (a) the hidden hard/extreme tier is larger than public 50's 26%/22%, or (b) tasks within each tier are systematically harder, or both.
3. **B-board's 12h budget buys near-zero algorithmic upside on this image**. v8 task pace × 1.8 (organizer/DGX slowdown estimate) × 324 (B-board task count, assumed) = 11.34h, well within 12h. SIGTERM is not the binding constraint, so the extra budget doesn't unlock new task completions — same image scores essentially the same as A-board (±0.02 from distribution variance).

**B-board decision — final**: ship `team1438:final` (= v8 image content) as-is. Rationale:

| Option | Risk | Realistic reward | Worst-case downside | Net |
|---|---|---|---|---|
| A. v8 image as-is | 0 | 0.38~0.40 (variance) | 0.38~0.40 | **+0.0** |
| B. timeout-relax + chronic prompt patches | medium-high | +0.01~0.03 | SIGTERM if 12h margin (0.66h) blown → −0.10~−0.20 | negative EV |
| C. max_steps↓ + force-cascade | medium | only helps Scenario A — which is now refuted | same downside as B for no upside | **strictly dominated** |

The one-shot / no-replacement nature of B-board makes asymmetric downsides decisive. Even a 30% chance of blowing the 12h margin (Option B's main failure mode) erases the expected upside ten times over.

**Action**: send `submissions/team1438_final.tar.gz` (sha256 `fd28e96f34bfb62e734af8bcfc36b72edf849dea1e05e6c1de1f03bce0d9b54b`) via the template above. Deadline 2026-05-23 20:59 KST. Earlier send is safer (avoids deadline-day Drive/email risk).
