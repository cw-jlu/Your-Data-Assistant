# DataAgent-Bench — System Flow & Agent Diagrams

> 🌐 **Language**: **English** · [한국어](SYSTEM_FLOW.ko.md) · [中文](SYSTEM_FLOW.zh.md)

A diagram-driven document covering the **end-to-end system flow** of our ReAct agent harness for the KDD Cup 2026 DataAgent-Bench challenge. For per-module code see `[ARCHITECTURE.md](ARCHITECTURE.md)`; for a one-screen architecture see `[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)`; for measurements and scores see `[DATA_ANALYSIS.md](DATA_ANALYSIS.md)`; for submission history see `[SUBMISSION_LOG.md](SUBMISSION_LOG.md)`; operating manual at `[../CLAUDE.md](../CLAUDE.md)`.

> Last updated: 2026-05-11 (v3 round — agent G/H/K/L + memory M + error N patches consolidated). When changes land, update §10's mermaid blocks first.

---

## 1. Document map


| Question                                      | Reference doc                                                    |
| --------------------------------------------- | ---------------------------------------------------------------- |
| "Where does which function live in the code?" | `[ARCHITECTURE.md](ARCHITECTURE.md)`                             |
| "Public 50-task statistics + failure modes"   | `[DATA_ANALYSIS.md](DATA_ANALYSIS.md)`                           |
| "Submission history + budget + score trend"   | `[SUBMISSION_LOG.md](SUBMISSION_LOG.md)`                         |
| "vLLM endpoint capabilities"                  | `[qwen_endpoint_capabilities.md](qwen_endpoint_capabilities.md)` |
| **"Whole system in one frame + every flow"**  | **This document (SYSTEM_FLOW.md)**                               |
| Rules compliance (verbatim)                   | skills `kddcup-rules-`*                                          |


---

## 2. 7-Layer system architecture

```mermaid
flowchart TB
    classDef infra fill:#1f2937,color:#f3f4f6,stroke:#374151
    classDef model fill:#312e81,color:#f3f4f6,stroke:#4338ca
    classDef runtime fill:#0f766e,color:#f3f4f6,stroke:#0d9488
    classDef agent fill:#7c2d12,color:#f3f4f6,stroke:#9a3412
    classDef tools fill:#7e22ce,color:#f3f4f6,stroke:#9333ea
    classDef scoring fill:#a16207,color:#f3f4f6,stroke:#ca8a04
    classDef submit fill:#9f1239,color:#f3f4f6,stroke:#be123c

    L1["<b>Layer 1 — Infrastructure</b><br/>Docker (python:3.10-slim)<br/>/input RO · /output RW · /logs RW<br/>16 vCPU · 64 GB · 12h<br/>network: MODEL_API_URL only"]:::infra
    L2["<b>Layer 2 — Model</b><br/>OpenAIModelAdapter<br/>JSON-mode probe + transient retry (G-1)<br/>qwen3.5-35b-a3b (eval)<br/>env-injected URL/KEY/NAME"]:::model
    L3["<b>Layer 3 — Runtime</b><br/>per-task subprocess<br/>ThreadPool batches<br/>Cascading wall-clock governor (12h)<br/>SIGTERM trap · RuntimeLogger<br/><b>+ run_benchmark_with_passes (H-1)</b>"]:::runtime
    L4["<b>Layer 4 — Agent</b><br/>ReAct loop · JSON contract<br/>parse-retry · action_input coercion<br/>plan-then-execute<br/>difficulty-aware max_steps<br/><b>+ SelfConsistencyAgent k=3 (G-2)</b>"]:::agent
    L5["<b>Layer 5 — Tools</b><br/>filesystem · sqlite · python_kernel<br/>dataframe_describe/head<br/>knowledge.md (5000 chars, keyword reorder)<br/>+ doc/*.md auto-inject<br/>_answer (conditional terminal)"]:::tools
    L6["<b>Layer 6 — Scoring</b><br/>normalize.py (null/2dp/ISO/trim)<br/>column-signature multiset<br/>name-equivalence (rules §10)<br/>mock_scorer + holdout split<br/><b>+ cross_run_vote (H-1 voter)</b>"]:::scoring
    L7["<b>Layer 7 — Submission</b><br/>build_submission.sh<br/>team1438:v<N> / .tar.gz ≤ 10 GB<br/>local_eval.sh container reproduction<br/>Drive + email + SUBMISSION_LOG"]:::submit

    L7 --> L6 --> L5 --> L4 --> L3 --> L2 --> L1
```



Each layer is unaware of layers above it and depends only on layers below. Changes only require lock-step updates to adjacent layers.

---

## 3. Organizer eval moment — end-to-end flow

```mermaid
sequenceDiagram
    autonumber
    participant Judge as Organizer eval driver
    participant Container as our dabench container
    participant Runner as run/runner.py
    participant Agent as ReActAgent
    participant Tools as ToolRegistry
    participant LLM as Organizer Qwen endpoint
    participant FS as /input · /output · /logs

    Judge->>Container: docker run --network=host --cpus=16 --memory=64g<br/>-v /input:/input:ro -v /output:/output:rw -v /logs:/logs:rw<br/>-e MODEL_API_URL/KEY/NAME<br/>(wrapped by timeout 7200s A-board / 43200s B-board)
    Container->>Runner: ENTRYPOINT: dabench run-benchmark --config configs/eval.yaml
    Runner->>Runner: install SIGTERM trap
    Runner->>FS: open /logs/runtime.log (JSONL append)
    Runner->>FS: discover task_<N>/ from /input
    Runner-->>Runner: ThreadPool(max_workers=4) batches

    loop per batch (governor checked between batches)
        Runner->>Runner: _maybe_engage_governor()
        opt budget remaining < tasks_remaining * avg_observed
            Runner-->>Runner: _downgrade_config (max_steps ÷2, timeout ÷2)
            Runner->>FS: log governor_engaged event
        end
        loop per task in batch (each in subprocess)
            Runner->>Agent: ReActAgent.run(task)
            Agent->>Agent: build_task_prompt<br/>(knowledge.md + doc/*.md auto-injected)
        end
    end
```



---

## 3b. Multi-pass orchestration (H-1)

`run_benchmark_with_passes` (`run/runner.py`) is the **outer loop** that wraps the eval-time entrypoint — runs the same task set N times then voter selects the per-task majority answer to compose the single prediction tree the judge consumes.

```mermaid
flowchart TB
    Entry["dabench run-benchmark<br/>(eval.yaml: repeat_max=3, pass_safety_margin=1.1)"]
    Master["create_run_output_dir<br/>master_output_dir = /output (flat)"]
    LogStart[multi_pass_start log event]

    P0Begin["pass 0<br/>output_dir=/output/_runs/run_0/"]
    P0End[multi_pass_iteration_done #0]
    Guard0{remaining_budget > <br/>last_pass_duration × 1.1 ?}

    P1Begin["pass 1<br/>output_dir=/output/_runs/run_1/"]
    P1End[multi_pass_iteration_done #1]
    Guard1{remaining_budget > <br/>last_pass_duration × 1.1 ?}

    P2Begin["pass 2<br/>output_dir=/output/_runs/run_2/"]
    P2End[multi_pass_iteration_done #2]

    Stop[multi_pass_early_stop log event]
    Vote["cross_run_vote.vote_across_runs<br/>(column-multiset majority,<br/>tie-break to earliest pass)"]
    VoteFail[multi_pass_vote_failed log event]
    Fallback["_fallback_copy_pass(<br/>pass_outputs[0], master)"]
    EndLog[multi_pass_end log event]
    Final["/output/task_<id>/prediction.csv<br/>(judge scores this)"]

    Entry --> Master --> LogStart --> P0Begin --> P0End --> Guard0
    Guard0 -- yes --> P1Begin --> P1End --> Guard1
    Guard1 -- yes --> P2Begin --> P2End --> Vote
    Guard0 -- no --> Stop --> Vote
    Guard1 -- no --> Stop
    Vote -- ok --> EndLog --> Final
    Vote -.exception.-> VoteFail --> Fallback --> EndLog

    classDef pass fill:#312e81,color:#fff,stroke:#4338ca
    classDef vote fill:#7e22ce,color:#fff,stroke:#9333ea
    classDef safe fill:#7f1d1d,color:#fff,stroke:#b91c1c
    class P0Begin,P1Begin,P2Begin pass
    class Vote vote
    class Fallback safe
```



**Regression-safety guarantees**:

- Pass 0 *always* completes (it's the fallback) → guarantees ≥ single-pass results on any hidden set.
- On voter exception, `_fallback_copy_pass` graceful-degrades — the judge's `/output/task_<id>/prediction.csv` is never empty.
- Governor mid-pass: next pass not started → single voted output preserved.

---

## 4. ReAct agent step loop (inside one task)

```mermaid
flowchart TD
    start([Task start]) --> build_prompt[build_system_prompt + build_task_prompt<br/>knowledge.md + doc/*.md auto-injected]
    build_prompt --> resolve_max[resolve_max_steps task.difficulty<br/>easy=8 / medium=12 / hard=24 / extreme=32]
    resolve_max --> step_loop{step ≤ max_steps?}
    step_loop -- no --> max_fail([failure: max_steps exceeded])

    step_loop -- yes --> complete[_complete_and_parse]
    complete --> llm_call[model.complete<br/>response_format if probed OK]
    llm_call --> parse_try{parse_model_step OK?}
    parse_try -- raise --> retry{retry < 2?}
    retry -- yes --> add_correction[append corrective<br/>PARSE_RETRY_REMINDER]
    add_correction --> llm_call
    retry -- no --> error_step[StepRecord: __error__<br/>parse_attempts=3]
    error_step --> step_loop

    parse_try -- ok --> coerce[_coerce_action_input<br/>execute_python str→{code:str}<br/>read_csv str→{path:str}]
    coerce --> tool_exec[ToolRegistry.execute<br/>action, action_input]
    tool_exec --> append[StepRecord append]
    append --> terminal{is_terminal?}
    terminal -- yes --> done([AgentRunResult])
    terminal -- no --> step_loop
```



**Invariants (require lockstep changes):**

1. Model response is a single fenced `json` block, keys = {thought, action, action_input}
2. `action_input` is a dict (strings get wrapped by `_coerce_action_input`)
3. Only `_answer` is terminal — even with conditional terminal, eventually it commits
4. context paths are always relative (`resolve_context_path` rejects escape)
5. Every `finally` calls `cleanup_task` (persistent kernel + validation counter)

---

## 5. Conditional terminal — `_answer` flow

```mermaid
flowchart LR
    answer_call[/_answer call:<br/>columns + rows + ?confirm/] --> build[_build_answer_payload<br/>shape validation]
    build -- ValueError --> raise([raise — agent loop catches])
    build --> validate[validate_answer<br/>question · normalized table]
    validate --> check{has_blocking?<br/>error/warning severity}

    check -- 0 warnings --> commit_a[terminal commit<br/>status=submitted<br/>val_warns=0]

    check -- has warnings --> branch{confirm == True<br/>OR bypass_count >= 1?}
    branch -- yes --> commit_b[terminal commit<br/>status=submitted<br/>val_warns shown]
    branch -- no --> reject[is_terminal=False<br/>status=validation_blocking<br/>warnings + suggestion<br/>bypass_remaining=0]
    reject --> back[Agent re-emits<br/>or sets confirm=true]
    back --> answer_call

    commit_a --> persist[runner._write_task_outputs<br/>prediction.csv = normalized]
    commit_b --> persist
```



**Severity levels:**

- `error` — empty answer / empty column (must fix)
- `warning` — all-null column (fix recommended)
- `info` — singular question + multiple rows, dup rows, column_count_mismatch (advisory)

`info`-only cases are non-blocking and **commit on the first call**.

---

## 6. Wall-Clock Governor (Layer 3)

```mermaid
sequenceDiagram
    participant Runner as run_benchmark
    participant Governor as _governor_should_engage
    participant Config as effective_config
    participant Logger as RuntimeLogger

    Runner->>Runner: budget=43200s (12h)<br/>elapsed_per_task=[]
    loop each batch
        Runner->>Governor: should_engage?<br/>(remaining = budget − elapsed)
        Governor-->>Runner: yes if remaining < tasks_remaining * avg_seen
        opt engage
            Runner->>Config: _downgrade_config<br/>(max_steps ×0.5, timeout ×0.5)
            Runner->>Logger: log governor_engaged<br/>{elapsed, tasks_remaining, avg_seen}
        end
        Runner->>Runner: execute batch (effective_config)
        Runner->>Runner: append elapsed_per_task
    end
    Runner->>Logger: log benchmark_end<br/>(governor_engaged status)
```



**Floor:**

- max_steps ≥ 1 (`_GOVERNOR_MIN_MAX_STEPS`)
- task_timeout ≥ 60s (`_GOVERNOR_MIN_TIMEOUT_SECONDS`)
- **v6 cascading**: up to `_GOVERNOR_MAX_CASCADES = 3` halvings, re-evaluated every `_GOVERNOR_RECHECK_TASK_INTERVAL = 5` tasks. Each cascade applies on top of the previous (so timeout=900s → 450 → 225 → 112 with the floor clamping at 60).

**SIGTERM trap (main thread only):**

- Handler logs `sigterm_received` event + closes log_file
- Raises `SystemExit(143)` → `finally` blocks run (kernel cleanup etc.) before exit
- Graceful flush within the 30s window before SIGKILL

---

## 7. Scoring — 3-phase matching (Layer 6)

```mermaid
flowchart TD
    pred[/prediction.csv/] --> norm_p[normalize_answer_table]
    gold[/gold.csv/] --> norm_g[normalize_answer_table]
    norm_p --> sigs_p[_column_signatures<br/>frozenset of (val, count)]
    norm_g --> sigs_g[_column_signatures]

    sigs_p --> phase1{Phase 1<br/>direct one-to-one}
    sigs_g --> phase1
    phase1 -- match --> add_m1[matched += 1<br/>used_pred + used_gold]
    phase1 -- some unmatched --> phase2{Phase 2<br/>gold single ⇄ pred pair join<br/>rules §10 name-eq}

    phase2 -- match --> add_m2[matched += 1<br/>used_pred += 2<br/>used_gold + 1]
    phase2 -- some unmatched --> phase3{Phase 3<br/>pred single ⇄ gold pair join<br/>rules §10 name-eq reverse}

    phase3 -- match --> add_m3[matched += 2<br/>used_gold += 2<br/>used_pred + 1]
    phase3 -- done --> compute[compute Score]
    add_m1 --> phase2
    add_m2 --> phase3
    add_m3 --> compute

    compute --> formula[Recall = matched / gold_cols<br/>Extra = pred_cols − len used_pred<br/>Score = max 0, Recall − λ·Extra/pred_cols]
    formula --> output([TaskScore])
```



**Two-direction Phase 2/3:** `_pair_signature_matches` checks both `(idx_a, idx_b)` and `(idx_b, idx_a)` orderings — rules §10 ("FirstName + LastName" → "FirstName LastName") doesn't fix which column is first/last.

**ExtraCols accuracy:** `pred_cols - len(used_pred)` (the older `pred_cols - matched` over-counted in Phase 3).

---

## 8. Single-task data flow (input → answer)

```mermaid
flowchart LR
    subgraph input["context/ (RO)"]
        kn[knowledge.md]
        doc[doc/*.md]
        csv[csv/*.csv]
        json[json/*.json]
        db[db/*.db]
    end

    subgraph prompt["agents/prompt.py"]
        sys[REACT_SYSTEM_PROMPT<br/>+ tool catalog<br/>+ normalization rules]
        task_prompt[build_task_prompt]
    end

    subgraph agent_loop["ReActAgent.run"]
        adapter[OpenAIModelAdapter]
        loop_step[step loop max_steps]
    end

    subgraph tools["ToolRegistry"]
        list_ctx[list_context]
        df_describe[dataframe_describe]
        df_head[dataframe_head]
        sqlite_q[execute_context_sql]
        py_kernel[execute_python<br/>persistent kernel]
        ans[_answer<br/>conditional terminal]
    end

    subgraph normalize_layer["scoring/"]
        normalize[normalize_answer_table<br/>null/2dp/ISO/trim]
        validator[validate_answer]
    end

    subgraph output["/output/task_id/"]
        pred[prediction.csv]
        trace[trace.json]
    end

    kn --> task_prompt
    doc --> task_prompt
    sys --> adapter
    task_prompt --> adapter
    adapter --> loop_step
    loop_step --> list_ctx
    loop_step --> df_describe
    loop_step --> df_head
    loop_step --> sqlite_q
    loop_step --> py_kernel
    csv --> df_describe
    csv --> df_head
    csv --> py_kernel
    json --> py_kernel
    db --> sqlite_q

    loop_step --> ans
    ans --> normalize
    ans --> validator
    normalize --> pred
    validator --> ans
    loop_step --> trace
```



---

## 9. Skills ↔ Layer mapping

Which skill to consult when touching a layer:


| Layer                    | Primary skills                                             |
| ------------------------ | ---------------------------------------------------------- |
| Layer 1 — Infrastructure | rules-runtime, rules-compute                               |
| Layer 2 — Model          | rules-model                                                |
| Layer 3 — Runtime        | rules-compute                                              |
| Layer 4 — Agent          | overview, agent, strategy                                  |
| Layer 5 — Tools          | overview, agent, dataset                                   |
| Layer 6 — Scoring        | scoring, rules-output                                      |
| Layer 7 — Submission     | submission, rules-submission, rules-prohibitions, strategy |


---

## 10. v2 → v3 submission cadence (current location marked)

```mermaid
gantt
    title Submission cadence — Phase 1 deadline 2026-05-23
    dateFormat YYYY-MM-DD
    axisFormat %m/%d

    section v2 (measured floor)
    Phase 1.0 + 2.0 + hotfix merge :done, m1, 2026-04-26, 3d
    holdout 0.7000 (after name-eq) :done, v2_meas, 2026-04-29, 1d
    leaderboard submission 0.3386  :done, v2_sub, 2026-04-29, 1d

    section v3 (consolidated round)
    agent patches G-1/G-2/G-3 + L-1 :done, v3a, 2026-04-30, 3d
    multi-pass H-1 + streaming JSON H-3 :done, v3b, after v3a, 2d
    size-aware reader K-1 + tier-aware retry :done, v3c, after v3b, 1d
    memory layer M-1~M-5 (TaskShape + recorder) :done, v3d, after v3c, 2d
    error pattern N-1~N-3 (cross-task + circuit-breaker + brief) :done, v3e, after v3d, 1d
    smoke + tarball v3 (sha256 1bb11bac…) :done, v3f, after v3e, 1d
    submit + leaderboard measurement :crit, v3_sub, after v3f, 1d
```



**Current (2026-05-11):** v3 build complete (`team1438_v3.tar.gz`, sha256 `1bb11bac…`). Multi-pass + cross-run vote, memory layer, and error pattern aggregation are all baked in and validated by 49-task smoke (0.7254 mean, 31 perfect). v3 has been submitted; leaderboard measurement is pending the 4-5 day organizer delay.

---

## 11. Ablation diagnostic flow (`mock_scorer --ablate`)

```mermaid
flowchart TD
    start([mock_scorer --ablate]) --> read[_read_all_tables<br/>prediction.csv + gold.csv pair]
    read --> per_task{per task}
    per_task --> minimum{pred_cols ≥ 2?}
    minimum -- no --> skip[skip]
    minimum -- yes --> keep_score[score_keep<br/>for each λ]
    minimum -- yes --> drop_each[for each candidate column<br/>compute score after drop]

    drop_each --> compare{score_drop > score_keep<br/>across ALL λ?}
    compare -- yes --> proposal[ColumnAblationProposal<br/>task_id, column_name, score_delta_per_λ]
    compare -- no --> skip2[skip]

    proposal --> aggregate[aggregate by frequency<br/>across all tasks]
    skip --> done([report])
    skip2 --> done
    keep_score --> done
    aggregate --> done
```



**Use case:** finding columns that systematically hurt the score across λ values. Rare on the public set (most predictions have ≤ 5 columns).

---

## 12. Entry-point one-line mapping


| Question                             | Answer                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| How is a single task run?            | `cli.py:run_task_command` → `runner.run_single_task` → `ReActAgent.run`                                 |
| How is a benchmark run?              | `cli.py:run_benchmark_command` → `runner.run_benchmark_with_passes` → `runner.run_benchmark` (per pass) |
| How are predictions scored?          | `mock_scorer.py:score_run` → `score_one` → 3-phase matching                                             |
| How is multi-pass voting computed?   | `runner.run_benchmark_with_passes` → `cross_run_vote.vote_across_runs`                                  |
| How is a Docker submission built?    | `scripts/build_submission.sh` → `Dockerfile` (`linux/amd64`) → `gzip` → sha256                          |
| Eval container ENTRYPOINT?           | `Dockerfile`: `uv run dabench run-benchmark --config configs/eval.yaml`                                 |
| `multi_pass_`* events written where? | `runner.run_benchmark_with_passes` → `RuntimeLogger.log_event` → `/logs/runtime.log`                    |


---

## 13. Predicting downstream impact for the next change

When changing some layer X, here's what is likely affected:


| Layer changed                                        | Direct impact                         | Lockstep update                      | Test                                    |
| ---------------------------------------------------- | ------------------------------------- | ------------------------------------ | --------------------------------------- |
| Layer 1 (Dockerfile / eval.yaml)                     | image build, eval container behaviour | scripts/build_submission.sh, configs | rebuild + smoke `local_eval.sh`         |
| Layer 2 (model.py)                                   | LLM call shape, JSON-mode             | agents/prompt.py contract            | `tests/agents/test_model_retry.py`      |
| Layer 3 (runner.py)                                  | parallelism, governor, multi-pass     | configs/eval.yaml                    | `tests/run/test_runner_*.py`            |
| Layer 4 (react.py / prompt.py / self_consistency.py) | step loop, JSON contract, voting      | tools/registry.py descriptions       | `tests/agents/test_*`                   |
| Layer 5 (tools/)                                     | observation shape, terminal logic     | agents/prompt.py tool catalog        | `tests/tools/test_*`                    |
| Layer 6 (scoring/)                                   | normalization, scorer logic, voter    | _answer handler, summarize-traces    | `tests/scoring/test_*`                  |
| Layer 7 (build / submission)                         | image manifest, naming                | docs/SUBMISSION_LOG.md               | `bash scripts/build_submission.sh v<N>` |


The Korean / Chinese full historical narrative versions of this document are linked at the top.