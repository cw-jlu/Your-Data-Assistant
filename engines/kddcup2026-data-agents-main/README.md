

# kddcup2026-data-agents — team1438

> 🌐 **Language**: **English** · [한국어](README.ko.md) · [中文](README.zh.md)

A ReAct agent harness for the KDD Cup 2026 **DataAgent-Bench** Leaderboard track.

[Competition](https://dataagent.top)
[Track](https://dataagent.top/rules)
[Team](#)



> A **single ReAct agent** that takes a natural-language analytics question, examines the provided context (CSV / SQLite / JSON / Markdown / PDF · Image · Excel · Parquet — for Phase 2 robustness) and produces a `prediction.csv`. Runs inside a Docker container the organizers mount, and only ever calls the organizer-mandated `qwen3.5-35b-a3b` LLM.

---

## 0. Final result — Phase 1 wrap-up (2026-07)

**Phase 1 is over for team1438 and this repo is now public as a competition archive.** Official standings from the [Phase 1 leaderboard](https://dataagent.top/leaderboard):

| Metric | team1438 |
|---|---|
| A-board (2 h wall-clock, 57 tasks) | **0.3886** |
| B-board (12 h wall-clock, 324 tasks) | **0.4349** |
| **Phase 1 final** (task-count-weighted ≈ 0.15·A + 0.85·B) | **0.4279** |
| Final rank | **137** of ~300 teams |
| Top-60 cutoff (Phase 2 qualification) | 0.5209 — **not qualified** |

The score trajectory tells the real story of the phase: **v1** failed evaluation outright (arm64 manifest — built natively on the DGX), **v2** landed the first score (**0.3386**) after the linux/amd64 cross-build fix, **v4** *regressed* to **0.2281** because the harness needed ~32 naive hours against a 2-hour A-board budget and was SIGTERM-truncated, and **v6 → v8** clawed it back (**0.3509 → 0.3886**) almost purely through wall-clock engineering — worker parallelism, per-difficulty task timeouts, a cascading governor that halves budgets under pressure. The single sanctioned B-board run (same v8 image, full 12-hour budget) scored **0.4349**.

Takeaway we'd hand to a future team: **on a hard wall-clock benchmark, scheduling beats model cleverness.** Every point recovered after v4 came from *finishing more tasks*, not from answering better — our per-task answer quality (public-set mock ≈ 0.69–0.73) was never the binding constraint; time was.

Everything below this section is preserved as it was during the competition (dates, projections, and "current status" included) as a working record. The full submission-by-submission history lives in [docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md).

---

## 1. One-line summary


| Item                            | Value                                                                                                                                                                             |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Competition                     | [KDD Cup 2026 — DataAgent-Bench](https://dataagent.top)                                                                                                                           |
| Track                           | Leaderboard (we drop the Creative track)                                                                                                                                          |
| Team ID                         | `team1438`                                                                                                                                                                        |
| Eval-time LLM                   | `qwen3.5-35b-a3b` (mandated by organizers, env-injected)                                                                                                                          |
| **Model policy (self-imposed)** | **qwen-only** — no auxiliary LLM/embedding/vision/web API. Config defaults are empty strings, so a missing env var fails loudly instead of silently calling an external provider. |
| Compute envelope                | 16 vCPU · 64 GB RAM · no GPU · **12-hour total** · linux/amd64                                                                                                                    |
| Mounts                          | `/input` RO · `/output` RW · `/logs` RW                                                                                                                                           |
| Network                         | Only `MODEL_API_URL` reachable (`--network=host`)                                                                                                                             |
| Submission                      | Docker tar.gz ≤ 10 GB → Google Drive share → email organizers. 1/day, 30 total in Phase 1.                                                                                        |
| Phase 1 deadline                | 2026-05-23 (AoE)                                                                                                                                                                  |


For the full rules see the six `.claude/skills/kddcup-rules-`* packs or the [official rules page](https://dataagent.top/rules).

---

## 2. Current status (2026-05-11)

| Stage | Status |
|---|---|
| Phase 1.0 substrate (knowledge.md injection + persistent IPython kernel + dataframe prepass) | ✅ merged |
| Phase 2.0 substrate (JSON-mode probe + difficulty-aware max_steps + wall-clock governor + parse-retry + plan-then-execute) | ✅ merged |
| Phase 3 substrate (answer_validator + conditional terminal + doc auto-injection + column_ablation + name-equivalence) | ✅ merged |
| §3.1–§3.5 (format dispatcher / size-aware streaming / hierarchical inspect_file / domain sanity) | ✅ merged |
| Compliance hardening (UV_OFFLINE=1, 3-tier linux/amd64 guard, visibility tools `dabench inspect-trace` / `summarize-traces`) | ✅ merged |
| **v3 agent polish (G-1 retry hint, G-2 SelfConsistencyAgent k=3, G-3 knowledge.md 5000-char + question-keyword reorder)** | ✅ merged |
| **v3 runtime (H-1 multi-pass orchestrator + cross_run_vote `repeat_max=3 pass_safety_margin=1.1`, K-1 tighter margin, L-1 tier-aware retry skip)** | ✅ merged |
| **v3 tools (H-3 streaming JSON: streaming_json_keys / count / aggregate)** | ✅ merged |
| **v3 memory layer (M-1~M-5 TaskShape + ShapePolicy + learnings.json + recorder + `dabench update-learnings`)** | ✅ merged |
| **v3 error pattern memory (N-1 error_patterns.json + recorder cross-task aggregation, N-2 in-loop repeat-error guard, N-3 pre-flight task brief)** | ✅ merged |
| **v3 build + 49-task measurement + ship** | ▶︎ `team1438_v3.tar.gz` (0.38 GB) |

### Organizer leaderboard history

| Version | Eval result | Note |
|---|---|---|
| v1 | eval failed | arm64 manifest issue (DGX-native build at the time) — drove our 3-tier amd64 guard |
| **v2** | **0.3386** | First successful eval after linux/amd64 cross-build. Current baseline floor. |
| **v3** | **built + measured, awaiting eval** | sha256 `1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09` — consolidated round (G-1~G-3, H-1, H-3, K-1, L-1, M-1~M-5, N-1~N-3); mock 0.7254 single-pass (44 scored, 31 perfect on the 49-task subset). |

### v3 measurement (public 50-task mock_scorer, λ=0.10)

49-task subset (task_418 hangs on OneDrive mounts in unkillable D-state — host-side artifact, irrelevant in eval).

| Measurement | Perfect | Mean (scored) | **50-task projection** | Wall-clock |
|---|---|---|---|---|
| v2 baseline floor | — | — | 0.3386 leaderboard | n/a |
| v3 single-pass (no memory layer yet) | 31 | 0.6986 | 0.6566 | 4911 s |
| **v3 single-pass (with memory layer, current)** | **31** | **0.7254** | **0.6384** | 5841 s |
| v3 multi-pass smoke (2-pass in-runner voting) | — | 0.7036 | 0.6473 | 12426 s |

The eval container uses `repeat_max=3` multi-pass + voting in production, which is strictly stronger than the single-pass numbers above. Multi-pass ablation across three independent runs hit projection 0.6457 with variance much lower than any individual run — the orchestrator is the actual variance-absorption mechanism.

### Leaderboard projection

```
v2 baseline (measured):   0.3386
mock → leaderboard gap:   ~ −0.15 to −0.20 (assuming hidden distribution is similar)
v3 expected leaderboard:  0.45 ~ 0.55 (multi-pass + voting in production)
Improvement over v2:      +0.11 ~ +0.17
```

### v3 artifacts

```
Image           : team1438:v3 (linux/amd64, 0.38 GB)
Tarball         : submissions/team1438_v3.tar.gz (0.38 GB)
sha256          : 1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09
LLM endpoint    : http://<VLLM_HOST>:8000/v1 (DGX, qwen3.5-35b-a3b, max_ctx 262 144)
Workers         : ThreadPool max_workers=4, difficulty timeout (180/360/900/1200, v6), wall_clock 12h
                  + cascading governor (v6: max 3 halvings, recheck every 5 tasks)
                  + multi-pass orchestrator (repeat_max=1 since v5 S-1; pass_safety_margin=1.1)
G-2 SC          : SelfConsistencyAgent k=3 temp=0.5 on hard/extreme tier
Memory layer    : memory/learnings.json + memory/error_patterns.json bundled in the image;
                  recorder ingests trace.json after every public-set run and proposes deltas
Compliance      : UV_OFFLINE=1, FROM --platform=linux/amd64, build-time uname guard, manifest inspect,
                  _fallback_copy_pass (preserves pass 0 even if voter fails → no regression vs single-pass)
```

### Score the eval output

```bash
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/eval_full_v3/output \
    --gold        data/public/output \
    --input       data/public/input \
    --lambda-values 0.05 0.10 0.20
```

---

## 3. Eval environment ↔ our container, 1:1

The verbatim rules-`§runtime` command the organizers run on our tarball:

```bash
docker run --rm \
  --network=host --cpus=16 --memory=64g \
  -v /input:/input:ro \
  -v /output:/output:rw \
  -v /logs:/logs:rw \
  -e MODEL_API_URL=...  -e MODEL_API_KEY=...  -e MODEL_NAME=qwen3.5-35b-a3b \
  team1438:v<N>
```


| Rule (skill)                                   | Stated spec                                 | Our behavior                                                                                                                          | Verification                                         |
| ---------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **runtime §1** ENTRYPOINT                      | `team1438:v<N>` runs by itself              | `uv run dabench run-benchmark --config configs/eval.yaml`                                                                             | `docker inspect`                                     |
| **runtime §2** /input RO                       | mutating it violates rules                  | `resolve_context_path` enforces sandbox                                                                                               | `grep` returns 0                                     |
| **runtime §2** /output format                  | `/output/task_<id>/prediction.csv`          | `flat_output_dir: true` writes per-task dir directly                                                                                  | `eval.yaml`                                          |
| **runtime §3** env injection (3 vars)          | no hardcode, env-mandated                   | model/api_base/api_key empty strings in eval.yaml                                                                                     | `grep -rE "api_key\s*=" src/` returns 0              |
| **runtime §4** network isolation               | only `MODEL_API_URL` allowed                | 0 calls to `requests`/`urllib`/`httpx`/external LLMs                                                                                  | grep returns empty                                   |
| **compute §1** linux/amd64 enforced            | x86-64 only                                 | 3-tier guard: ① `FROM --platform=linux/amd64` ② Dockerfile build-time `uname -m` guard ③ post-build `docker image inspect` arch check | `docker image inspect team1438:v3` → `amd64 linux`   |
| **compute §2** 12h total                       | not per-task                                | `wall_clock_budget_seconds: 43200` + 8h-trigger downgrade governor                                                                    | `runner.py:_governor_should_engage`                  |
| **compute §5** SIGTERM 30s grace               | partial flush within 30s                    | `_install_sigterm_trap` (graceful)                                                                                                    | `runner.py:_install_sigterm_trap`                    |
| **model §1** qwen3.5-35b-a3b mandated          | no other LLM main solver                    | `OpenAIModelAdapter` only, 0 auxiliary LLMs                                                                                           | pyproject deps                                       |
| **model §3** no small CPU LLMs allowed         | no in-container LLM weights                 | image 0.38 GB (no model weights)                                                                                                      | `du -sh`                                             |
| **submission §1** naming                       | `<team_id>:v<N>` + `<team_id>_v<N>.tar.gz`  | `team1438:v3` + `submissions/team1438_v3.tar.gz`                                                                                      | `ls submissions/`                                    |
| **submission §2** ≤ 10 GB                      | post-compress limit                         | 0.38 GB                                                                                                                               | `du -h submissions/*.tar.gz`                         |
| **submission §4** 1/day · 30/Phase 1           | rate limit                                  | tracker                                                                                                                               | `[docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md)`   |
| **output §1-3** CSV format                     | UTF-8 + 1 header row + col order irrelevant | `_answer` handler → `normalize_answer_table` → flat path                                                                              | `scoring/normalize.py`                               |
| **output §4-9** normalization                  | HALF_UP / ISO / strip                       | explicit `decimal.Decimal` + `ROUND_HALF_UP`                                                                                          | `normalize.py:_normalize_numeric`                    |
| **output §10** name-equivalence                | `FN+LN` ↔ `FN LN`                           | `mock_scorer.py` 3-phase matching                                                                                                     | `mock_scorer.py:match_columns_with_name_equivalence` |
| **prohibitions §1** external internet bypass   | strictly forbidden                          | 0 grep hits + assumes `--network=host`                                                                                            | grep                                                 |
| **prohibitions §3** /input mutate / env tamper | strictly forbidden                          | 0 instances of `os.environ[MODEL_*] = …`                                                                                              | grep returns empty                                   |
| **prohibitions §5** infra probing              | adversarial submissions forbidden           | every submission a real improvement                                                                                                   | `[docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md)`   |


### Residual gaps — things we can't replicate locally


| Item                     | Eval env                           | Our env                              | Impact                                       |
| ------------------------ | ---------------------------------- | ------------------------------------ | -------------------------------------------- |
| Host machine             | Organizer Linux x86-64 native      | macOS Apple Silicon → QEMU emulation | slower wall-clock, identical results         |
| `MODEL_API_URL`          | Organizer's internal qwen endpoint | DGX self-hosted vLLM              | same model ID                                |
| Concurrent request limit | Unknown                            | DGX vLLM limit                       | OpenAIAdapter retry triggers if rate-limited |
| Network                  | only host w/ firewall (`--network=host`) | macOS → DGX LAN                      | irrelevant since we make zero external calls |


**Conclusion:** v3 satisfies 100% of the rule spec. Only the eval endpoint's max_context / rate limit need to be confirmed via the leaderboard response.

---

## 4. Quick start — macOS dev + DGX vLLM split workflow

team1438 recommended layout:

- **DGX (`<VLLM_HOST>`, aarch64)**: vLLM(`qwen3.5-35b-a3b`) **serving only** (DGX is ARM, mismatches eval's amd64).
- **macOS local (Apple Silicon arm64)**: build / test / submission packaging. `linux/amd64` cross-build matches eval exactly.

```bash
# 0. (DGX, once) start vLLM serving
ssh <user>@<VLLM_HOST> 'cd <repo-dir> && bash scripts/serve_qwen_docker.sh'

# 1. (macOS) deps + env vars
uv sync --extra dev
export MODEL_API_URL=http://<VLLM_HOST>:8000/v1
export MODEL_API_KEY=EMPTY
export MODEL_NAME=qwen3.5-35b-a3b

# 2. (macOS) data inspection
uv run dabench status        --config configs/local.yaml
uv run dabench inspect-task task_<id> --config configs/local.yaml

# 3. (macOS) host run — single / full / holdout
uv run dabench run-task     task_<id> --config configs/local.yaml
uv run dabench run-benchmark          --config configs/local.yaml
uv run dabench run-benchmark          --config configs/local.yaml --task-set data/public/holdout_ids.txt

# 4. (macOS) score
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> --gold data/public/output \
    --input data/public/input --lambda-values 0.05 0.10 0.20 --ablate

# 5. (macOS) submission package — linux/amd64 cross-build
bash scripts/build_submission.sh v<N>
docker tag dabench:v<N> team1438:v<N> && \
  docker save team1438:v<N> | gzip --best > submissions/team1438_v<N>.tar.gz
shasum -a 256 submissions/team1438_v<N>.tar.gz

# 6. (optional) container reproducibility check
DABENCH_LAMBDAS="0.05 0.10 0.20" \
  MODEL_API_URL=http://<VLLM_HOST>:8000/v1 \
  bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
```

### Eval mirror — full 50-task container run

```bash
mkdir -p artifacts/eval_full_v3/{output,logs}
docker run --rm -d --name team1438_v3_full \
    --platform linux/amd64 \
    -e MODEL_API_URL="http://<VLLM_HOST>:8000/v1" \
    -e MODEL_API_KEY="EMPTY" \
    -e MODEL_NAME="qwen3.5-35b-a3b" \
    -v "$(pwd)/data/public/input:/input:ro" \
    -v "$(pwd)/artifacts/eval_full_v3/output:/output" \
    -v "$(pwd)/artifacts/eval_full_v3/logs:/logs" \
    team1438:v3
```

This command **mirrors the rule §runtime invocation in mounts, env, and platform exactly**.

---

## 5. 7-Layer architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 7 — Submission   Docker tarball, Drive, email — team1438:v<N>│
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6 — Scoring      normalize + column-signature multiset        │
│                        + name-equivalence (rules §10)               │
│                        + cross_run_vote (column-multiset majority)  │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 5 — Tools        filesystem · sqlite · python_kernel          │
│                        dataframe_describe/head · _answer            │
│                        validator (conditional terminal)             │
│                        format dispatcher (PDF/Excel/Parquet/Img)    │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4 — Agent        ReAct loop · JSON contract · parse-retry     │
│                        difficulty-aware max_steps · plan-execute    │
│                        SelfConsistencyAgent (k=3, hard/extreme)     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3 — Runtime      per-task subprocess · ThreadPool batches     │
│                        wall-clock governor · SIGTERM trap           │
│                        run_benchmark_with_passes (multi-pass H-1)   │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2 — Model        OpenAIModelAdapter + JSON-mode probe         │
│                        env-injected MODEL_API_URL/KEY/NAME          │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 1 — Infra        Docker · 16 vCPU / 64 GB / 12h · /input RO  │
└─────────────────────────────────────────────────────────────────────┘
```

For the layer-responsibility one-shot view see `[docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)`. For the data/execution mermaid sequence diagrams see `[docs/SYSTEM_FLOW.md](docs/SYSTEM_FLOW.md)`. For the component dependency graph + external-comm policy + build pipeline + round-by-round change log see `[docs/HARNESS_STRUCTURE.md](docs/HARNESS_STRUCTURE.md)` — a **living document**.

---

## 6. Directory structure

```
.
├── src/data_agent_baseline/
│   ├── cli.py                    # Typer 4 sub-commands
│   ├── config.py                 # env > YAML > default overlay
│   ├── benchmark/                # DABenchPublicDataset + PublicTask schema
│   ├── agents/
│   │   ├── prompt.py             # system / task / observation prompt + knowledge.md (5000-char, question-keyword reorder) & doc/*.md auto-inject
│   │   ├── react.py              # ReActAgent.run + parse-retry + action_input coercion + difficulty max_steps
│   │   ├── self_consistency.py   # SelfConsistencyAgent — k=3 voting on hard/extreme (G-2)
│   │   ├── model.py              # OpenAIModelAdapter (JSON-mode probe + fallback)
│   │   └── runtime.py            # StepRecord / AgentRuntimeState / AgentRunResult
│   ├── tools/
│   │   ├── registry.py           # tool catalog + conditional-terminal _answer (16 tools)
│   │   ├── filesystem.py         # csv/json/doc/pdf/excel/parquet/image/archive + dataframe_* + inspect_file
│   │   ├── sqlite.py             # read-only SQL + schema inspect
│   │   ├── python_exec.py        # ephemeral subprocess fallback
│   │   └── python_kernel.py      # persistent IPython kernel (default)
│   ├── scoring/
│   │   ├── normalize.py          # null / 2dp HALF_UP / ISO date / strip
│   │   ├── mock_scorer.py        # 3-phase matching incl. rules §10 name-equivalence
│   │   ├── answer_validator.py   # heuristic ValidationReport (§3.5 domain sanity)
│   │   ├── column_ablation.py    # λ-agreement diagnostic
│   │   ├── cross_run_vote.py     # column-multiset majority across N benchmark roots (H-1)
│   │   └── holdout.py            # blake2b deterministic 80/20 split
│   └── run/runner.py             # subprocess isolation + ThreadPool + cascading governor (v6) + RuntimeLogger + SIGTERM trap
│                                 # + run_benchmark_with_passes (multi-pass orchestrator, H-1)
├── configs/
│   ├── eval.yaml                     # Docker submission (env-overridable, empty-string defaults, v6 timeout caps, repeat_max=1)
│   └── local.yaml                    # development (assumes self-hosted vLLM)
├── docker/vllm-qwen35/                  # self-hosted vLLM (DGX)
├── scripts/
│   ├── build_submission.sh           # docker buildx (linux/amd64) + gzip + sha256 + 10GB check
│   ├── build_dashboard.py            # v6: single-file HTML viewer of all eval runs (file:// open, no deps)
│   ├── local_eval.sh                 # container reproduction + mock_scorer + render report
│   ├── serve_qwen_docker.sh          # vLLM compose wrapper
│   ├── probe_qwen.py                 # endpoint capability measurement
│   └── render_score_report.py        # JSON → markdown report
├── docs/
│   ├── ARCHITECTURE.{md,ko.md,zh.md}            # code · runtime guide
│   ├── SYSTEM_ARCHITECTURE.{md,ko.md,zh.md}     # one-screen system architecture (v3 round)
│   ├── SYSTEM_FLOW.{md,ko.md,zh.md}             # 8+ mermaid (data/execution flow)
│   ├── HARNESS_STRUCTURE.{md,ko.md,zh.md}       # 7-Layer + dependency + change log
│   ├── DATA_ANALYSIS.{md,ko.md,zh.md}           # 50-task statistics + failure modes
│   ├── SUBMISSION_LOG.{md,ko.md,zh.md}          # submission history + budget tracker
│   └── qwen_endpoint_capabilities.{md,ko.md,zh.md} # self-hosted vLLM probe
├── .claude/skills/                   # 12 KDD Cup skills
├── Dockerfile                        # eval container (linux/amd64 enforced)
├── pyproject.toml                    # data-agent-baseline package
└── CLAUDE.{md,ko.md,zh.md}           # AI-assistant operating manual
```

`tests/`, `data/`, `artifacts/`, `submissions/` are gitignored. Within `docs/*` only the English/Korean/Chinese trio of each doc is whitelisted.

---

## 7. Submission flow (summary)

```
1. Build           bash scripts/build_submission.sh v<N>
2. Retag           docker tag dabench:v<N> team1438:v<N> &&
                   docker save team1438:v<N> | gzip --best > submissions/team1438_v<N>.tar.gz
3. Reproducibility DABENCH_LAMBDAS="0.05 0.10 0.20" \
                   bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
                   → must match host within ±0.005
4. sha256 check    shasum -a 256 submissions/team1438_v<N>.tar.gz
5. Drive upload    "Anyone with the link can view" permission
6. Email organizer team_id + version + Drive URL + sha256
7. Update log      docs/SUBMISSION_LOG.md budget tracker (Used N→N+1)
```

Rules: **1 per day**, **30 per Phase 1 total**, hold the next submission until the previous one is scored.

---

## 8. Dev commands

```bash
uv run pytest                                   # unit tests
uv run pytest tests/path/to/test_x.py::test_y   # single test
uv run ruff check src tests                     # lint (line-length 100, py310)

# Regenerate holdout split (when data is updated)
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public

# Score diagnosis (with column ablation)
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/eval_full_v3/output \
    --gold data/public/output --input data/public/input \
    --lambda-values 0.05 0.10 0.20 --ablate
```

For the full dev workflow see `[CLAUDE.md](CLAUDE.md)`.

---

## 9. Dependencies / runtime

- Python ≥ 3.10 (uv installs 3.10/3.11)
- Docker 24+ + buildx (linux/amd64 cross-build)
- (optional) NVIDIA GPU + Docker compose — for self-hosted vLLM
- Inside the eval container all deps are deterministically installed via `uv.lock`

Deps live in `pyproject.toml` `[project.dependencies]`. Headliners: pandas, numpy, openai, polars, pyarrow, pypdf, openpyxl, Pillow, IPython.

---

## 10. Where this repo came from

A fork of [HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit). The starter kit ships only a Phase 0 baseline (score ≈ 0); we have nearly fully rewritten `src/data_agent_baseline/`:

- Normalization + mock_scorer + holdout split (Phase 0 ship gate)
- knowledge.md/doc auto-injection, persistent kernel, dataframe prepass (Phase 1.0)
- JSON-mode probe, difficulty max_steps, wall-clock governor, parse-retry, plan-then-execute (Phase 2.0)
- answer_validator, conditional terminal, name-equivalence, column_ablation (Phase 3 substrate)
- §3.1 format dispatcher (PDF/Excel/Parquet/Image/Archive readers)
- §3.2 size-aware streaming + §3.3 hierarchical inspect_file + §3.5 domain sanity heuristics
- **Runtime + compliance hardening:** column-count ratio>1.5× blocking, difficulty-aware task_timeout (300/600/900/1200), OpenAIAdapter retry 1→3 with backoff, plan-then-execute prompt strengthening, hard/extreme max_steps 24/32, `UV_OFFLINE=1` (matches the organizers' `--network=host`)
- **Visibility tools:** `dabench inspect-trace` (single-trace colored step view), `dabench summarize-traces` (50-task categorization + diff)
- **Prompt cues seeded from forensics:** 0-row trap prompt, source-schema lock, pre-answer self-verify, plural-cue under-emission validator
- **v3 agent + runtime round (G/H/K/L):**
  - G-1: include `Request timed out` in the first-step transient hint set (recovers 21 timeouts seen in earlier endpoint-overload runs)
  - G-2: `SelfConsistencyAgent` k=3 column-signature voting on hard/extreme tier (temperature 0.5)
  - G-3: knowledge.md cap 3000 → 5000 chars + question-keyword H2/H3 reorder
  - **H-1: multi-pass orchestrator + cross_run_vote** — the container runs the same task set up to `repeat_max=3` times and majority-votes by column-multiset. The budget guard makes regression vs single-pass impossible (`pass_safety_margin=1.1`, `_fallback_copy_pass`).
  - **H-3: streaming JSON tools** — `streaming_json_keys / count / aggregate` stream huge JSON arrays via `ijson`, no size cap.
  - K-1 / L-1: tighter `pass_safety_margin`, tier-aware retry skip on hard/extreme subprocess timeouts.
- **v3 memory layer round (M/N):**
  - M-1~M-5: `memory/` (TaskShape classifier, ShapePolicy resolver, bundled `learnings.json`, recorder, `dabench update-learnings` CLI).
  - **N-1: error pattern memory** — bundled `error_patterns.json` with cross-task signature aggregation; advisories injected into the task prompt at task start.
  - N-2: in-loop repeat-error circuit-breaker — when 2 consecutive same-signature errors fire, a one-line "do NOT retry the same approach" cue is appended before the next model turn.
  - N-3: pre-flight task brief — deterministic context-tree scan (CSV peek, SQLite tables, join-key candidates) inlined as `Pre-flight task brief:` in the user prompt.
- **Forensic discipline:** Two earlier prompt cues (a "simulate the grader" instruction and a strict 0-row override) were proposed during the round and then rejected after multi-sample ablation — single-run comparisons had misled their diagnosis; variance was the real cause. Every patch now passes through multi-sample ablation.

For the full change history see `git log --oneline`.

---

## 11. License / attribution

This repo started as a fork of [HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit). The upstream starter kit ships **without an explicit license file**; rights to the original starter-kit portions remain with the upstream authors (HKUST DIAL), and this fork follows the same terms for those portions. The `src/data_agent_baseline/` tree was almost entirely rewritten by team1438 during Phase 1.

The repo was made public after Phase 1 concluded (final judging closed 2026-07-14), as an archive and write-up. It was private for the full duration of the competition, in line with the competition rules on cross-team sharing. The `.claude/skills/kddcup-rules-*` packs paraphrase the [official rules](https://dataagent.top/rules); treat the official page as authoritative.