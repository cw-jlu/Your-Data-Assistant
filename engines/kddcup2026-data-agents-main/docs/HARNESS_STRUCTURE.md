# Harness Structure — `data_agent_baseline`

> 🌐 **Language**: **English** · [한국어](HARNESS_STRUCTURE.ko.md) · [中文](HARNESS_STRUCTURE.zh.md)

> Living document — keep this in lock-step with the codebase; bump it whenever a patch round lands.
> Last updated: 2026-05-11 (v3 — consolidated round on top of v2: memory layer M-1~M-5, multi-pass voting H-1, streaming JSON H-3, error pattern memory N-1, repeat-error guard N-2, pre-flight task brief N-3, plus G-1~G-3 / K-1 / L-1 runtime polish)

ReAct agent harness for the KDD Cup 2026 DataAgent-Bench challenge. Seven logical layers + build/eval automation + visibility tools. Each layer is unaware of layers above it and depends only on layers below (single-direction dependency).

For data and execution sequence diagrams see [`SYSTEM_FLOW.md`](SYSTEM_FLOW.md). This document focuses on **structure (component graph + responsibility boundaries)**.

---

## 1. 7-Layer view (logical separation)

```mermaid
flowchart TB
    L7["<b>Layer 7 — Submission</b><br/>Docker tarball · Drive · email<br/>team1438:v&lt;N&gt; / team1438_v&lt;N&gt;.tar.gz"]
    L6["<b>Layer 6 — Scoring</b><br/>normalize + column-signature<br/>+ name-equivalence (rules §10)<br/>+ column_ablation"]
    L5["<b>Layer 5 — Tools</b><br/>16 tools: filesystem · sqlite · python_kernel<br/>dataframe_describe/head · _answer + validator<br/>format dispatcher (PDF/Excel/Parquet/Image/Archive)<br/>hierarchical inspect_file"]
    L4["<b>Layer 4 — Agent</b><br/>ReAct loop · JSON contract · parse-retry<br/>action_input coercion · difficulty-aware max_steps<br/>plan-then-execute · 0-row trap / self-verify prompts<br/><b>+ G-2 SelfConsistencyAgent (k=3 voting)</b><br/><b>+ N-2 repeat-error circuit-breaker</b>"]
    L3["<b>Layer 3 — Runtime</b><br/>per-task subprocess · ThreadPool batches<br/>cascading wall-clock governor (v6, max 3 halvings)<br/>SIGTERM trap · difficulty-aware task_timeout<br/>G-1/L-1 first-step retry<br/><b>+ H-1 multi-pass orchestrator (cross-run vote)</b>"]
    L2["<b>Layer 2 — Model</b><br/>OpenAIModelAdapter · JSON-mode probe<br/>retry expand (1→3, exponential backoff)<br/>explicit httpx timeout=240s (v6)<br/>env-injected MODEL_API_URL/KEY/NAME"]
    L1["<b>Layer 1 — Infra</b><br/>Docker · linux/amd64 (3-tier guard)<br/>UV_OFFLINE=1 · 16 vCPU/64 GB · 12h<br/>/input RO · /output RW · /logs RW"]

    L7 --> L6
    L6 --> L5
    L5 --> L4
    L4 --> L3
    L3 --> L2
    L2 --> L1
```

### Per-layer entry points

| Layer | Entry point |
|---|---|
| 7 | `scripts/build_submission.sh`, `submissions/team1438_v<N>.tar.gz` |
| 6 | `src/data_agent_baseline/scoring/{normalize,mock_scorer,answer_validator,column_ablation,holdout,cross_run_vote}.py` |
| 5 | `src/data_agent_baseline/tools/{registry,filesystem,sqlite,python_exec,python_kernel,streaming_json}.py` |
| 4 | `src/data_agent_baseline/agents/{react,prompt,model,runtime,self_consistency}.py` |
| 4m | `src/data_agent_baseline/memory/{task_shape,policies,learnings,recorder,error_patterns,task_brief}.py` (memory layer M-1~M-5 + error pattern N-1~N-3 + pre-flight brief) |
| 3 | `src/data_agent_baseline/run/runner.py` (`run_benchmark` + `run_benchmark_with_passes`) |
| 2 | `src/data_agent_baseline/agents/model.py:OpenAIModelAdapter` |
| 1 | `Dockerfile`, `configs/eval.yaml` (multi-pass enabled), `scripts/build_submission.sh` |

---

## 2. Component dependency graph (module level)

```mermaid
flowchart LR
    cli[cli.py<br/>Typer · 6 commands]

    subgraph cfg [config & schema]
        config[config.py<br/>env > YAML > default]
        schema[benchmark/schema.py<br/>PublicTask, AnswerTable]
        dataset[benchmark/dataset.py<br/>DABenchPublicDataset]
    end

    subgraph agent [agents/]
        react[react.py<br/>ReActAgent.run]
        prompt[prompt.py<br/>system + task + observation]
        model[model.py<br/>OpenAIModelAdapter]
        runtime[runtime.py<br/>StepRecord · AgentRuntimeState]
        sc[self_consistency.py<br/>SelfConsistencyAgent k=3]
    end

    subgraph tools [tools/]
        registry[registry.py<br/>16 tools + ToolRegistry]
        fs[filesystem.py<br/>csv/json/doc/pdf/excel/parquet/image/archive]
        sql[sqlite.py<br/>read-only SQL]
        kernel[python_kernel.py<br/>persistent IPython]
        pyexec[python_exec.py<br/>ephemeral fallback]
    end

    subgraph scoring [scoring/]
        normalize[normalize.py<br/>null/2dp HALF_UP/ISO/strip]
        mock[mock_scorer.py<br/>3-phase name-equiv]
        validator[answer_validator.py<br/>ValidationReport]
        ablation[column_ablation.py<br/>λ-agreement diagnostic]
        holdout[holdout.py<br/>blake2b 80/20 split]
        crv[cross_run_vote.py<br/>column-multiset majority]
    end

    subgraph run [run/]
        runner[runner.py<br/>run_benchmark + governor + SIGTERM<br/>+ run_benchmark_with_passes]
    end

    subgraph inspect [inspect/]
        traceview[trace_view.py<br/>render_trace]
        summarize[summarize.py<br/>summarize_run + diff]
    end

    cli --> config
    cli --> dataset
    cli --> runner
    cli --> traceview
    cli --> summarize

    runner --> react
    runner --> sc
    runner --> registry
    runner --> dataset
    runner --> model
    runner --> crv

    sc --> react
    sc --> normalize

    react --> prompt
    react --> registry
    react --> model
    react --> runtime
    prompt --> schema

    registry --> fs
    registry --> sql
    registry --> kernel
    registry --> pyexec
    registry --> normalize
    registry --> validator
    fs --> schema
    sql --> schema

    mock --> normalize
    summarize --> mock

    runner --> normalize

    classDef external fill:#fff3e0,stroke:#e65100;
    classDef harness fill:#e3f2fd,stroke:#0277bd;
    class cli,runner,react,registry harness;
```

---

## 3. External communication — single-channel policy

```mermaid
flowchart LR
    container["v3 container<br/>(team1438:v3)"]
    qwen["MODEL_API_URL<br/>(organizer's qwen3.5-35b-a3b)"]
    pypi["pypi.org<br/>~~uv side-effect~~"]
    other["Other LLM APIs<br/>(OpenAI/Anthropic/HF)"]

    container -->|allowed (rule §runtime §4)| qwen
    container -.->|<b>blocked by UV_OFFLINE=1</b>| pypi
    container -.->|<b>0 grep hits in src/</b>| other

    classDef allowed fill:#c8e6c9,stroke:#2e7d32;
    classDef blocked fill:#ffcdd2,stroke:#c62828;
    class qwen allowed;
    class pypi,other blocked;
```

Verification grep:
- direct imports of `requests` / `urllib` / `httpx` / `aiohttp` → **0 hits**
- `anthropic` / `cohere` / `together` / external LLM SDK → **0 hits**
- socket / curl / wget subprocess → **0 hits**
- hardcoded external domain URLs → **0 hits**
- `os.environ[MODEL_*] = …` env mutation → **0 hits**
- legitimate call: `openai` SDK in one location (`agents/model.py`) — only to `MODEL_API_URL`

---

## 4. Build + verify + submit pipeline

```mermaid
flowchart LR
    src[src/<br/>+ configs/eval.yaml]
    build["bash scripts/build_submission.sh v&lt;N&gt;<br/>docker buildx --platform linux/amd64"]
    img[(team1438:v&lt;N&gt;<br/>linux/amd64<br/>UV_OFFLINE=1)]
    tar[(submissions/team1438_v&lt;N&gt;.tar.gz<br/>≤ 10 GB)]
    eval["docker run --rm --platform linux/amd64<br/>(local_eval.sh or direct)"]
    out[artifacts/eval_full_v&lt;N&gt;/output/<br/>task_&lt;id&gt;/{prediction.csv, trace.json}]
    score["mock_scorer<br/>(λ ∈ {0.05, 0.10, 0.20})"]
    inspect["dabench inspect-trace<br/>dabench summarize-traces --diff"]
    drive[Google Drive<br/>+ organizer email]

    src --> build
    build --> img
    img --> tar
    img --> eval
    eval --> out
    out --> score
    out --> inspect
    tar --> drive
    score -->|on ship-gate pass| drive
```

### 3-tier arch guard

```mermaid
flowchart TB
    g1["① Dockerfile<br/>FROM --platform=linux/amd64"]
    g2["② Dockerfile<br/>RUN test &quot;$(uname -m)&quot; = &quot;x86_64&quot;<br/>(build-time fail)"]
    g3["③ build_submission.sh<br/>docker image inspect ... .Architecture<br/>(post-build fail)"]
    out_amd64[linux/amd64 manifest guaranteed]

    g1 --> g2
    g2 --> g3
    g3 --> out_amd64
```

---

## 5. ReAct agent: per-task processing flow

```mermaid
sequenceDiagram
    participant runner as Layer 3<br/>runner
    participant agent as Layer 4<br/>ReActAgent
    participant model as Layer 2<br/>ModelAdapter
    participant registry as Layer 5<br/>ToolRegistry
    participant validator as Layer 6<br/>answer_validator

    runner->>+agent: run(task)
    Note over agent: build_messages (system + task + history)
    loop step 1..max_steps_by_difficulty
        agent->>+model: complete(messages)
        Note over model: JSON-mode probe (once) + retry expand
        model-->>-agent: raw response
        Note over agent: parse_model_step + action_input coercion + parse-retry
        agent->>+registry: execute(action, action_input)
        alt action == "answer"
            registry->>+validator: validate_answer(answer, question)
            validator-->>-registry: ValidationReport
            alt has_blocking + bypass < 1
                registry-->>agent: is_terminal=False<br/>(soft-reject, observation has warnings)
            else
                registry-->>agent: is_terminal=True<br/>(committed)
            end
        else other tool
            registry-->>-agent: ToolExecutionResult
        end
        agent->>agent: append StepRecord
        Note over agent: pre-answer self-verify (F-3) on next answer
    end
    agent-->>-runner: AgentRunResult
    Note over runner: write trace.json + prediction.csv (normalized)
```

---

## 5b. Multi-pass orchestration (H-1)

When `repeat_max > 1`, `run_benchmark_with_passes` runs the same task set N times and uses cross-run vote to decide the final answer per task. The first pass is always preserved → even if the voter fails or budget runs out, the system never regresses below single-pass.

```mermaid
flowchart TB
    start["dabench run-benchmark<br/>(eval.yaml: repeat_max=3, pass_safety_margin=1.3)"]
    pass0["pass 0 (run_benchmark)<br/>output_dir/_runs/run_0/task_*/"]
    chk0{"remaining_budget<br/>&gt; last_pass × 1.3 ?"}
    pass1["pass 1<br/>output_dir/_runs/run_1/"]
    chk1{"remaining_budget<br/>&gt; last_pass × 1.3 ?"}
    pass2["pass 2<br/>output_dir/_runs/run_2/"]
    vote["cross_run_vote.vote_across_runs<br/>(column-multiset majority,<br/>tie-break to earliest pass)"]
    fallback["_fallback_copy_pass<br/>(copies pass 0 if voter raises)"]
    final[output_dir/task_&lt;id&gt;/<br/>prediction.csv (final)]

    start --> pass0 --> chk0
    chk0 -->|yes| pass1 --> chk1
    chk0 -->|no / repeat_max reached| vote
    chk1 -->|yes| pass2 --> vote
    chk1 -->|no / repeat_max reached| vote
    vote --> final
    vote -.exception.-> fallback --> final

    classDef pass fill:#e8f5e9,stroke:#2e7d32;
    classDef vote fill:#fff8e1,stroke:#f57c00;
    classDef safe fill:#ffebee,stroke:#c62828;
    class pass0,pass1,pass2 pass;
    class vote vote;
    class fallback safe;
```

**Voting key.** For each task's `prediction.csv`, compute per-column `scoring.normalize.column_signature` (a value multiset), then freeze the multiset of signatures via `frozenset(Counter(...).items())` for use as a hashable bucket key. This is **identical** to the column-multiset semantics the scorer uses.

**Tie-break.** When buckets have equal size, the bucket containing the earliest pass wins. Therefore **if every pass disagrees, pass 0 wins** = worst case degrades to single-pass behaviour.

**Budget guard.** At the end of each pass:
- If `repeat_max` reached → stop.
- If `wall_clock_budget_seconds` is None or ≤ 0 → unlimited (run all configured passes).
- Otherwise: if `(budget − elapsed_total) < last_pass_duration × pass_safety_margin` → don't start the next pass → first pass preserved.

**runtime.log events** (analyzable by visibility tools):
- `multi_pass_start` — repeat_max / margin / budget recorded
- `multi_pass_iteration_done` — pass_index + duration + elapsed_total
- `multi_pass_early_stop` — reason (remaining budget / margin etc.)
- `multi_pass_vote_failed` — fallback fired
- `multi_pass_end` — completed_passes / unanimous_tasks / split_tasks / missing_in_all

---

## 6. Visibility tools — post-mortem flow

```mermaid
flowchart LR
    subgraph artifacts [artifacts/eval_full_v&lt;N&gt;/]
        traces[output/task_&lt;id&gt;/trace.json]
        preds[output/task_&lt;id&gt;/prediction.csv]
        log[logs/runtime.log JSONL]
    end

    insp["dabench inspect-trace &lt;task_id&gt;<br/>(single-trace colored step view)"]
    summ["dabench summarize-traces<br/>(50-task categorization)"]

    traces --> insp
    traces --> summ
    preds --> summ

    insp --> single[Rich console panel:<br/>step-by-step thought/action/obs<br/>+ gold comparison]
    summ --> report[failure bucket<br/>+ soft-reject codes<br/>+ tool freq<br/>+ score by difficulty<br/>+ recovered/regressed (--diff)]

    classDef tool fill:#fff8e1,stroke:#f57c00;
    class insp,summ tool;
```

---

## 7. Rules compliance — single matrix

A bird's-eye view of which rule each layer satisfies. (Verbatim matrix is in [`../README.md`](../README.md) §3.)

| Rule category (skill) | Satisfying layer |
|---|---|
| `kddcup-rules-runtime` (mounts, env, network) | Layer 1 (Dockerfile, configs/eval.yaml) |
| `kddcup-rules-compute` (CPU/RAM/12h/SIGTERM/amd64) | Layer 1 + Layer 3 (governor + SIGTERM trap) |
| `kddcup-rules-model` (qwen mandated, no hardcode) | Layer 2 (env > YAML > default empty string) |
| `kddcup-rules-submission` (image naming / 10 GB / 1/day) | Layer 7 (build_submission.sh + SUBMISSION_LOG.md) |
| `kddcup-rules-output` (CSV / normalization / name-equiv) | Layer 6 (normalize + mock_scorer) + Layer 5 (`_answer` handler) |
| `kddcup-rules-prohibitions` (no external calls / no probing) | All layers (code grep + every submission a real improvement) |

---

## 8. Change log (per round)

| Round | Change | sha256 (tarball) |
|---|---|---|
| Phase 0 | normalize + mock_scorer + holdout (starter-kit rewrite) | — |
| Phase 1.0 | knowledge.md inject, persistent IPython kernel, dataframe prepass | — |
| Phase 2.0 | JSON-mode probe, difficulty max_steps, governor, parse-retry, plan-then-execute | — |
| Phase 3 substrate | answer_validator, conditional terminal, name-equiv, column_ablation, doc auto-inject | — |
| §3.1-§3.5 | format dispatcher (PDF/Excel/Parquet/Image/Archive), size-aware streaming, hierarchical inspect_file, domain sanity | — |
| **v1** (submitted) | First submission — eval failed due to arm64 manifest issue | (discarded) |
| **v2** (submitted) | First successful eval after linux/amd64 cross-build | leaderboard **0.3386** |
| **v3** (submitted) | Consolidated round on top of v2: **G-1** transient retry hint, **G-2** SelfConsistencyAgent k=3 (hard/extreme), **G-3** knowledge.md 5000-char + question-keyword H2/H3 reorder, **H-1** multi-pass orchestrator + cross-run vote (`repeat_max=3`, `pass_safety_margin=1.1`), **H-3** streaming JSON tools, **K-1** size-aware reader chunking, **L-1** tier-aware retry skip, **memory layer M-1~M-5** (TaskShape + ShapePolicy + learnings.json + recorder + `dabench update-learnings`), **N-1** error pattern memory (`error_patterns.json`), **N-2** in-loop repeat-error circuit-breaker, **N-3** pre-flight task brief | tarball `team1438_v3.tar.gz` sha256 `1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09` (mock 0.7254 single-pass, 31 perfect on the 49-task subset; multi-pass voting in production) |

---

## 9. Backlog / deferred

| Candidate | Effect | Why deferred |
|---|---|---|
| H-2 extend SC to medium tier | absorb medium-tier variance | H-1 cross-run vote provides equivalent effect more safely |
| H-3 long-tail extreme task fix (task_352/396/418) | consistently fail every run | needs streaming JSON parser + execute_python helper (medium effort) |
| Plan §C steps.jsonl streaming | per-task progress during eval | inspect-trace handles post-mortem |
| Plan §D token/latency metrics | LLM cost visibility | no score impact |
| OneDrive subprocess hang fix (task_418) | dodge SIGKILL-resistant D-state | likely irrelevant in eval env (no OneDrive mount) |

---

## 10. Update policy

This document is the **single source of truth for harness structure**. Update points:
- After a new round merges (update §8 table + relevant layer-diagram labels)
- New layer / module added (update §2 dependency graph)
- Compliance matrix changes (§7)
- Leaderboard score arrives (§8 sha256 / score columns)

Related docs (English):
- [`../README.md`](../README.md) — Project overview + verbatim rules-compliance matrix
- [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) — 7-Layer + multi-pass + voting unified one-screen view (v3 round)
- [`SYSTEM_FLOW.md`](SYSTEM_FLOW.md) — data flow / execution sequence (8+ mermaid)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Code walkthrough (human-oriented)
- [`DATA_ANALYSIS.md`](DATA_ANALYSIS.md) — 50-task statistics + failure modes
- [`SUBMISSION_LOG.md`](SUBMISSION_LOG.md) — Submission history + budget tracker
- `.claude/skills/kddcup-*` — 12 KDD Cup skill packs
