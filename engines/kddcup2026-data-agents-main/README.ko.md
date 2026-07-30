

# kddcup2026-data-agents — team1438

> 🌐 **Language**: [English](README.md) · **한국어** · [中文](README.zh.md)

KDD Cup 2026 **DataAgent-Bench** Leaderboard 트랙 도전을 위한 ReAct 에이전트 하네스.

[대회](https://dataagent.top)
[트랙](https://dataagent.top/rules)
[팀](#)



> 자연어 분석 질문을 받아 컨텍스트(CSV / SQLite / JSON / Markdown / PDF·이미지·Excel·Parquet — Phase 2 방어용)를 분석한 뒤 `prediction.csv`를 생성하는 **단일 ReAct 에이전트**. 운영진이 마운트한 Docker 컨테이너에서 동작하며 LLM은 운영진이 강제하는 `qwen3.5-35b-a3b`만 사용한다.

---

## 0. 최종 결과 — Phase 1 마무리 (2026-07)

**team1438의 Phase 1은 종료되었고, 이 레포는 대회 아카이브로 공개되었다.** [Phase 1 공식 리더보드](https://dataagent.top/leaderboard) 기준:

| 항목 | team1438 |
|---|---|
| A-board (2시간 wall-clock, 57 task) | **0.3886** |
| B-board (12시간 wall-clock, 324 task) | **0.4349** |
| **Phase 1 최종** (task 수 가중 ≈ 0.15·A + 0.85·B) | **0.4279** |
| 최종 순위 | 약 300팀 중 **137위** |
| Top-60 컷 (Phase 2 진출) | 0.5209 — **미진출** |

점수 궤적이 이 라운드의 실제 이야기다: **v1**은 평가 자체가 실패했고(arm64 manifest — 당시 DGX 네이티브 빌드), **v2**가 linux/amd64 크로스 빌드 수정 후 첫 점수(**0.3386**)를 기록했다. **v4**는 2시간 A-board 예산에 대해 naive 기준 ~32시간이 필요한 하네스가 SIGTERM으로 잘리며 **0.2281**로 *역행*했고, **v6 → v8**은 워커 병렬화·난이도별 task timeout·압박 시 예산을 반감하는 cascading governor 등 순수한 wall-clock 엔지니어링으로 **0.3509 → 0.3886**을 회복했다. 단 한 번뿐인 B-board 제출(동일 v8 이미지, 12시간 예산 전체)은 **0.4349**를 기록했다.

미래의 팀에게 전할 교훈: **하드 wall-clock 벤치마크에서는 스케줄링이 모델의 영리함을 이긴다.** v4 이후 회복한 모든 점수는 *더 많은 task를 끝낸 것*에서 왔지, 더 잘 답한 것에서 오지 않았다 — task당 답변 품질(public-set mock ≈ 0.69–0.73)은 한 번도 병목이 아니었고, 시간이 병목이었다.

이 섹션 아래의 모든 내용은 대회 당시의 작업 기록 그대로 보존한다(날짜·예측치·"현재 상태" 포함). 제출별 전체 히스토리는 [docs/SUBMISSION_LOG.ko.md](docs/SUBMISSION_LOG.ko.md) 참고.

---

## 1. 한 줄 요약


| 항목             | 값                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| 대회             | [KDD Cup 2026 — DataAgent-Bench](https://dataagent.top)                                                     |
| 트랙             | Leaderboard (Creative 트랙은 포기)                                                                               |
| 팀 ID           | `team1438`                                                                                                  |
| 평가 시 LLM       | `qwen3.5-35b-a3b` (운영진 강제, 환경변수 주입)                                                                         |
| **모델 정책 (자체)** | **qwen 단일** — 보조 LLM/임베딩/vision/web API 모두 미사용. config 기본값은 빈 문자열로 두어 env 누락 시 명시적 에러 (silent fallback 차단). |
| 컴퓨트 envelope   | 16 vCPU · 64 GB RAM · GPU 없음 · **12시간 합계** · linux/amd64                                                    |
| 마운트            | `/input` RO · `/output` RW · `/logs` RW                                                                     |
| 네트워크           | `MODEL_API_URL` 외 차단 (`--network=eval_net`)                                                                 |
| 제출             | Docker tar.gz ≤ 10 GB → Google Drive 공유 → 운영진 이메일. 하루 1회 / Phase 1 총 30회                                    |
| Phase 1 마감     | 2026-05-23 (AoE)                                                                                            |


자세한 룰은 `.claude/skills/kddcup-rules-`* 6종 또는 [공식 룰 페이지](https://dataagent.top/rules) 참고.

---

## 2. 현재 상태 (2026-05-11)


| 단계 | 상태 |
|---|---|
| Phase 1.0 substrate (knowledge.md 인젝션 + persistent IPython kernel + dataframe prepass) | ✅ 머지 |
| Phase 2.0 substrate (JSON-mode probe + 난이도별 max_steps + wall-clock governor + parse-retry + plan-then-execute) | ✅ 머지 |
| Phase 3 substrate (answer_validator + conditional terminal + doc auto-injection + column_ablation + name-equivalence) | ✅ 머지 |
| §3.1–§3.5 (format dispatcher / size-aware streaming / hierarchical inspect_file / domain sanity) | ✅ 머지 |
| 컴플라이언스 강화 (UV_OFFLINE=1, 3중 linux/amd64 가드, 가시성 도구 `dabench inspect-trace` / `summarize-traces`) | ✅ 머지 |
| **v3 agent 라운드 (G-1 retry hint, G-2 SelfConsistencyAgent k=3, G-3 knowledge.md 5000자 + question-keyword reorder)** | ✅ 머지 |
| **v3 runtime 라운드 (H-1 multi-pass orchestrator + cross_run_vote `repeat_max=3 pass_safety_margin=1.1`, K-1 좁은 margin, L-1 tier-aware retry skip)** | ✅ 머지 |
| **v3 tools 라운드 (H-3 streaming JSON: streaming_json_keys / count / aggregate)** | ✅ 머지 |
| **v3 메모리 레이어 (M-1~M-5 TaskShape + ShapePolicy + learnings.json + recorder + `dabench update-learnings`)** | ✅ 머지 |
| **v3 error pattern 메모리 (N-1 error_patterns.json + recorder cross-task aggregation, N-2 in-loop repeat-error guard, N-3 pre-flight task brief)** | ✅ 머지 |
| **v3 빌드 + 49-task 측정 + ship** | ▶︎ `team1438_v3.tar.gz` (0.38 GB) |

### 운영진 leaderboard 이력

| 버전 | 평가 결과 | 비고 |
|---|---|---|
| v1 | 평가 fail | arm64 manifest issue (당시 DGX native build) — 이후 3중 amd64 가드 도입 |
| **v2** | **0.3386** | linux/amd64 cross-build 후 첫 성공 평가. 현재 baseline floor. |
| **v3** | **빌드 + 측정 완료, 평가 대기** | sha256 `1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09` — 통합 라운드 (G-1~G-3, H-1, H-3, K-1, L-1, M-1~M-5, N-1~N-3); mock 0.7254 single-pass (44 scored, 49-task 부분집합에서 31 perfect). |

### v3 측정 (공개 50-task mock_scorer, λ=0.10)

49 task 기준 (task_418 hang 제외; OneDrive 마운트 D-state 이슈 — eval 환경에선 미발생).

| 측정 | Perfect | Mean (scored) | **50-task 환산** | Wall-clock |
|---|---|---|---|---|
| v2 baseline floor | — | — | 0.3386 leaderboard | n/a |
| v3 single-pass (메모리 레이어 전) | 31 | 0.6986 | 0.6566 | 4911 s |
| **v3 single-pass (메모리 레이어 포함, current)** | **31** | **0.7254** | **0.6384** | 5841 s |
| v3 multi-pass smoke (in-runner 2-pass voting) | — | 0.7036 | 0.6473 | 12426 s |

eval 컨테이너는 production에서 `repeat_max=3` multi-pass + voting 사용 — 위 single-pass 숫자보다 엄격히 좋음. 3 run 독립 ablation 결과 분포는 0.6457로 수렴, single run 변동성보다 훨씬 안정.

### Leaderboard 추정

```
v2 baseline (실측):       0.3386
mock → leaderboard gap:   약 −0.15 ~ −0.20 (hidden 분포 가정)
v3 기대 leaderboard:      0.45 ~ 0.55 (production multi-pass + voting)
v2 floor 대비:            +0.11 ~ +0.17 개선
```

### v3 산출물

```
이미지       : team1438:v3 (linux/amd64, 0.38 GB)
Tarball      : submissions/team1438_v3.tar.gz (0.38 GB)
sha256       : 1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09
LLM endpoint : http://<VLLM_HOST>:8000/v1 (DGX, qwen3.5-35b-a3b, max_ctx 262 144)
워커         : ThreadPool max_workers=4, difficulty timeout(300/600/900/1200), wall_clock 12h
                + multi-pass orchestrator (repeat_max=3, pass_safety_margin=1.1)
G-2 SC       : SelfConsistencyAgent k=3 temp=0.5 on hard/extreme tier
메모리 레이어 : memory/learnings.json + memory/error_patterns.json bundled;
                recorder가 매 public-set run trace.json을 흡수 → delta 제안
가드 환경    : UV_OFFLINE=1, FROM --platform=linux/amd64, build-time uname guard, manifest inspect,
                _fallback_copy_pass (voter 실패 시에도 pass 0 보존 → single-pass와 회귀 불가)
```

### 평가 후 점수 측정

```bash
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/eval_full_v3/output \
    --gold        data/public/output \
    --input       data/public/input \
    --lambda-values 0.05 0.10 0.20
```

---

## 3. 평가 환경 ↔ 우리 컨테이너 1:1 대조

운영진이 우리 tarball을 받아 실행하는 룰 §runtime verbatim 명령:

```bash
docker run --rm \
  --network=eval_net --cpus=16 --memory=64g \
  -v /input:/input:ro \
  -v /output:/output:rw \
  -v /logs:/logs:rw \
  -e MODEL_API_URL=...  -e MODEL_API_KEY=...  -e MODEL_NAME=qwen3.5-35b-a3b \
  team1438:v<N>
```


| 룰 (skill)                              | 룰 명시 spec                                  | 동작                                                                                                                 | 검증 근거                                                    |
| -------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| **runtime §1** ENTRYPOINT              | `team1438:v<N>` 자체 실행 가능                   | `uv run dabench run-benchmark --config configs/eval.yaml`                                                          | `docker inspect`                                         |
| **runtime §2** /input RO               | 수정 시 룰 위반                                  | `resolve_context_path` sandbox 강제                                                                                  | `grep` 0건                                                |
| **runtime §2** /output 형식              | `/output/task_<id>/prediction.csv`         | `flat_output_dir: true` → task별 dir 직접 작성                                                                          | `eval.yaml`                                              |
| **runtime §3** env 주입 (3개)             | 하드코드 금지, env 강제                            | eval.yaml의 model/api_base/api_key 빈 문자열                                                                            | `grep -rE "api_key\s*=" src/` 0건                         |
| **runtime §4** 네트워크 격리                 | `MODEL_API_URL`만 허용                        | `requests`/`urllib`/`httpx`/외부 LLM 호출 0건                                                                           | grep 빈 결과                                                |
| **compute §1** linux/amd64 강제          | x86-64 only                                | 3중 가드: ① `FROM --platform=linux/amd64` ② Dockerfile build-time `uname -m` 가드 ③ 빌드 후 `docker image inspect` arch 검증 | `docker image inspect team1438:v3` → `amd64 linux`       |
| **compute §2** 12h 합계                  | per-task 아님                                | `wall_clock_budget_seconds: 43200` + 8h trigger downgrade governor                                                 | `runner.py:_governor_should_engage`                      |
| **compute §5** SIGTERM 30s grace       | 30초 안에 partial flush                       | `_install_sigterm_trap` (graceful)                                                                                 | `runner.py:_install_sigterm_trap`                        |
| **model §1** qwen3.5-35b-a3b 강제        | 다른 LLM 메인 솔버 금지                            | `OpenAIModelAdapter` 단일, 보조 LLM 0개                                                                                 | pyproject 의존성                                            |
| **model §3** CPU에 작은 LLM 명시 금지         | 컨테이너 내 다른 LLM 가중치 금지                       | image 0.38 GB (model weights 미포함)                                                                                  | `du -sh`                                                 |
| **submission §1** 네이밍                  | `<team_id>:v<N>` + `<team_id>_v<N>.tar.gz` | `team1438:v3` + `submissions/team1438_v3.tar.gz`                                                                   | `ls submissions/`                                        |
| **submission §2** ≤ 10 GB              | 압축 후 한도                                    | 0.38 GB                                                                                                            | `du -h submissions/*.tar.gz`                             |
| **submission §4** 1/일 · 30/Phase 1     | 빈도 한도                                      | 누적 트래킹                                                                                                             | `[docs/SUBMISSION_LOG.ko.md](docs/SUBMISSION_LOG.ko.md)` |
| **output §1-3** CSV 형식                 | UTF-8 + 헤더 1행 + 컬럼순서 무관                    | `_answer` 핸들러 → `normalize_answer_table` → flat path 작성                                                            | `scoring/normalize.py`                                   |
| **output §4-9** 정규화                    | HALF_UP / ISO / strip                      | `decimal.Decimal` + `ROUND_HALF_UP` 명시적 사용                                                                         | `normalize.py:_normalize_numeric`                        |
| **output §10** name-equivalence        | `FN+LN` ↔ `FN LN`                          | `mock_scorer.py` 3-phase matching                                                                                  | `mock_scorer.py:match_columns_with_name_equivalence`     |
| **prohibitions §1** 외부 인터넷 우회          | 절대 금지                                      | 코드 grep 0건 + `--network=eval_net` 가정                                                                               | grep                                                     |
| **prohibitions §3** /input 수정 / env 변조 | 절대 금지                                      | `os.environ[MODEL_*] = …` 0건                                                                                       | grep 빈 결과                                                |
| **prohibitions §5** 인프라 probing        | adversarial submission 금지                  | 모든 제출 진짜 개선용                                                                                                       | `[docs/SUBMISSION_LOG.ko.md](docs/SUBMISSION_LOG.ko.md)` |


### 잔여 차이 — 해소 불가능한 점


| 항목              | 평가 환경                   | 우리 환경                            | 영향                                         |
| --------------- | ----------------------- | -------------------------------- | ------------------------------------------ |
| 호스트 머신          | 운영진 Linux x86-64 native | macOS Apple Silicon → QEMU 에뮬레이션 | 속도 느림, 결과 동일                               |
| `MODEL_API_URL` | 운영진 내부 qwen endpoint    | DGX self-hosted vLLM          | 모델 ID 동일                                   |
| 동시 요청 한도        | 운영진 rate limit 미상       | DGX vLLM 한도                      | 평가 시 rate limit 걸리면 OpenAIAdapter retry 발동 |
| 네트워크            | `eval_net`만 도달          | macOS → DGX LAN                  | 우리 코드 외부 호출 0건이라 무관                        |


**결론:** v3 컨테이너는 룰 spec을 100% 만족. 운영진 endpoint의 max_context / rate limit만 leaderboard 응답으로 검증 필요.

---

## 4. 빠른 시작 — macOS dev + DGX vLLM 분리 워크플로

team1438 권장 배치:

- **DGX (`<VLLM_HOST>`, aarch64)**: vLLM(`qwen3.5-35b-a3b`) **서빙만**.
- **macOS local (Apple Silicon arm64)**: 빌드 / 테스트 / 제출 packaging. `linux/amd64` cross-build로 평가 환경 정확 일치.

```bash
# 0. (DGX, 한 번) vLLM 서빙 시작
ssh <user>@<VLLM_HOST> 'cd <repo-dir> && bash scripts/serve_qwen_docker.sh'

# 1. (macOS) 의존성 + 환경 변수
uv sync --extra dev
export MODEL_API_URL=http://<VLLM_HOST>:8000/v1
export MODEL_API_KEY=EMPTY
export MODEL_NAME=qwen3.5-35b-a3b

# 2. (macOS) 데이터 가시성
uv run dabench status        --config configs/local.yaml
uv run dabench inspect-task task_<id> --config configs/local.yaml

# 3. (macOS) host run — 단일 / 전체 / holdout
uv run dabench run-task     task_<id> --config configs/local.yaml
uv run dabench run-benchmark          --config configs/local.yaml
uv run dabench run-benchmark          --config configs/local.yaml --task-set data/public/holdout_ids.txt

# 4. (macOS) 점수 측정
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> --gold data/public/output \
    --input data/public/input --lambda-values 0.05 0.10 0.20 --ablate

# 5. (macOS) 제출 패키지 — linux/amd64 cross-build
bash scripts/build_submission.sh v<N>
docker tag dabench:v<N> team1438:v<N> && \
  docker save team1438:v<N> | gzip --best > submissions/team1438_v<N>.tar.gz
shasum -a 256 submissions/team1438_v<N>.tar.gz

# 6. (선택) container 재현 검증
DABENCH_LAMBDAS="0.05 0.10 0.20" \
  MODEL_API_URL=http://<VLLM_HOST>:8000/v1 \
  bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
```

### 평가 환경 미러 — 50 task 전체 컨테이너 실행

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

이 명령은 룰 §runtime의 형태와 **마운트·env·platform이 정확히 동일**.

---

## 5. 7-Layer 아키텍처

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

레이어 책임 한 장면 시각화는 `[docs/SYSTEM_ARCHITECTURE.ko.md](docs/SYSTEM_ARCHITECTURE.ko.md)`. 데이터·실행 흐름 mermaid 8종은 `[docs/SYSTEM_FLOW.ko.md](docs/SYSTEM_FLOW.ko.md)`. 컴포넌트 dependency graph + 외부 통신 정책 + 빌드 파이프라인 mermaid + 라운드 변경 이력은 `[docs/HARNESS_STRUCTURE.ko.md](docs/HARNESS_STRUCTURE.ko.md)` — **살아있는 문서**.

---

## 6. 디렉토리 구조

```
.
├── src/data_agent_baseline/
│   ├── cli.py                    # Typer 4 서브커맨드
│   ├── config.py                 # env > YAML > default 오버레이
│   ├── benchmark/                # DABenchPublicDataset + PublicTask schema
│   ├── agents/
│   │   ├── prompt.py             # system / task / observation prompt + knowledge.md (5000자, question keyword reorder) & doc/*.md auto-inject
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
│   │   └── holdout.py            # blake2b 결정적 80/20 split
│   └── run/runner.py             # subprocess 격리 + ThreadPool + governor + RuntimeLogger + SIGTERM trap
│                                 # + run_benchmark_with_passes (multi-pass orchestrator, H-1)
├── configs/
│   ├── eval.yaml                     # Docker submission (env-overridable, 빈 문자열 디폴트, repeat_max=3)
│   └── local.yaml                    # 개발용 (자체 vLLM 가정)
├── docker/vllm-qwen35/                  # 자체 호스팅 vLLM (DGX)
├── scripts/
│   ├── build_submission.sh           # docker buildx (linux/amd64) + gzip + sha256 + 10GB 검증
│   ├── local_eval.sh                 # 컨테이너 재현 + mock_scorer + render report
│   ├── serve_qwen_docker.sh          # vLLM compose wrapper
│   ├── probe_qwen.py                 # endpoint capability 측정
│   └── render_score_report.py        # JSON → markdown 보고서
├── docs/
│   ├── ARCHITECTURE.{md,ko.md,zh.md}            # 코드·런타임 가이드
│   ├── SYSTEM_ARCHITECTURE.{md,ko.md,zh.md}     # 한 장면 시스템 architecture (v3 라운드)
│   ├── SYSTEM_FLOW.{md,ko.md,zh.md}             # mermaid 8종+ (데이터·실행 흐름)
│   ├── HARNESS_STRUCTURE.{md,ko.md,zh.md}       # 7-Layer + dependency + 변경 이력
│   ├── DATA_ANALYSIS.{md,ko.md,zh.md}           # 50-task 통계 + 실패 모드
│   ├── SUBMISSION_LOG.{md,ko.md,zh.md}          # 제출 이력 + budget tracker
│   └── qwen_endpoint_capabilities.{md,ko.md,zh.md} # 자체 vLLM probe
├── .claude/skills/                   # 12 KDD Cup skills
├── Dockerfile                        # 평가 컨테이너 (linux/amd64 강제)
├── pyproject.toml                    # data-agent-baseline 패키지
└── CLAUDE.{md,ko.md,zh.md}           # AI 보조도구용 운영 매뉴얼
```

`tests/`, `data/`, `artifacts/`, `submissions/`는 gitignore. `docs/*` 중 영어/한국어/중국어 3개 언어 파일만 화이트리스트.

---

## 7. 제출 흐름 (요약)

```
1. 빌드          bash scripts/build_submission.sh v<N>
2. retag         docker tag dabench:v<N> team1438:v<N> &&
                 docker save team1438:v<N> | gzip --best > submissions/team1438_v<N>.tar.gz
3. 재현 검증     DABENCH_LAMBDAS="0.05 0.10 0.20" \
                 bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
                 → host와 ±0.005 이내 일치해야
4. sha256 확인   shasum -a 256 submissions/team1438_v<N>.tar.gz
5. Drive 업로드   "Anyone with the link can view" 권한
6. 운영진 이메일  team_id + version + Drive URL + sha256
7. 로그 갱신      docs/SUBMISSION_LOG.ko.md budget tracker (Used N→N+1)
```

룰: **하루 1회**, **Phase 1 총 30회**, 직전 평가 완료까지 다음 제출 보류.

---

## 8. 개발 명령

```bash
uv run pytest                                   # 단위 테스트
uv run pytest tests/path/to/test_x.py::test_y   # 단일 테스트
uv run ruff check src tests                     # lint (line-length 100, py310)

# Holdout split 재생성 (데이터 갱신 시)
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public

# 점수 진단 (column ablation 추가)
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/eval_full_v3/output \
    --gold data/public/output --input data/public/input \
    --lambda-values 0.05 0.10 0.20 --ablate
```

자세한 dev 워크플로우는 `[CLAUDE.ko.md](CLAUDE.ko.md)`.

---

## 9. 의존성 / 실행 환경

- Python ≥ 3.10 (uv가 3.10/3.11 설치)
- Docker 24+ + buildx (linux/amd64 cross-build)
- (선택) NVIDIA GPU + Docker compose — 자체 vLLM 호스팅 시
- 평가 컨테이너 안에선 모든 deps가 `uv.lock` 결정적 install

deps는 `pyproject.toml` `[project.dependencies]` 참고. 대표: pandas, numpy, openai, polars, pyarrow, pypdf, openpyxl, Pillow, IPython.

---

## 10. 이 레포는 어디서 왔는가

[HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit)의 fork. starter kit은 Phase 0 베이스라인(점수 ≈ 0)만 제공하고 우리는 `src/data_agent_baseline/`을 거의 전면 재작성:

- 정규화 + mock_scorer + holdout split (Phase 0 ship gate)
- knowledge.md/doc auto-injection, persistent kernel, dataframe prepass (Phase 1.0)
- JSON-mode probe, 난이도 max_steps, wall-clock governor, parse-retry, plan-then-execute (Phase 2.0)
- answer_validator, conditional terminal, name-equivalence, column_ablation (Phase 3 substrate)
- §3.1 format dispatcher (PDF/Excel/Parquet/Image/Archive readers)
- §3.2 size-aware streaming + §3.3 hierarchical inspect_file + §3.5 domain sanity heuristics
- **Runtime + 컴플라이언스 강화:** column-count ratio>1.5× blocking, difficulty-aware task_timeout (300/600/900/1200), OpenAIAdapter retry 1→3 with backoff, plan-then-execute prompt 강화, hard/extreme max_steps 24/32, `UV_OFFLINE=1` (`--network=eval_net` 대응)
- **가시성 도구:** `dabench inspect-trace` (단일 trace 컬러 step view), `dabench summarize-traces` (50-task 카테고리화 + diff)
- **forensic 시드 prompt cue:** 0-row trap, source-schema lock, pre-answer self-verify, plural-cue under-emission validator
- **v3 agent + runtime 라운드 (G / H / K / L):**
  - G-1: `Request timed out` 을 first-step transient hint에 포함 (이전 endpoint overload run에서 21건 timeout 회수)
  - G-2: `SelfConsistencyAgent` k=3 column-signature voting on hard/extreme tier (temperature 0.5)
  - G-3: knowledge.md cap 3000 → 5000자 + question keyword H2/H3 reorder
  - **H-1: multi-pass orchestrator + cross_run_vote** — 컨테이너가 같은 task set을 최대 `repeat_max=3`회 풀고 column-multiset majority로 voting. budget guard로 회귀 불가 (`pass_safety_margin=1.1`, `_fallback_copy_pass`).
  - **H-3: streaming JSON tools** — `streaming_json_keys / count / aggregate`가 `ijson` 스트림으로 거대 JSON 처리, size cap 없음.
  - K-1 / L-1: 더 좁은 `pass_safety_margin`, hard/extreme subprocess timeout retry skip (tier-aware).
- **v3 메모리 라운드 (M / N):**
  - M-1~M-5: `memory/` (TaskShape classifier, ShapePolicy resolver, bundled `learnings.json`, recorder, `dabench update-learnings` CLI).
  - **N-1: error pattern 메모리** — bundled `error_patterns.json`이 cross-task signature aggregation 누적; task 시작 시 prompt에 advisory inject.
  - N-2: in-loop repeat-error 회로 차단기 — 같은 signature 에러 2번 연속이면 다음 model turn 앞에 "do NOT retry the same approach" cue.
  - N-3: pre-flight task brief — `context/`의 결정적 read-only 스캔 (CSV peek, SQLite table, join key 후보) → user prompt에 `Pre-flight task brief:` inline.
- **forensic 규율:** 라운드 중 검토된 두 prompt cue ("simulate the grader" / 엄격한 0-row override)는 multi-sample ablation 후 폐기 — 단일-run 비교가 진단을 오도했고 variance가 진짜 원인. 이후 모든 patch는 multi-sample ablation 필수.

상세 변경 내역은 `git log --oneline` 참고.

---

## 11. 라이선스 / 출처

이 레포는 [HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit)의 fork로 시작했다. upstream starter kit에는 **명시적 라이선스 파일이 없으며**, 원본 starter-kit 부분의 권리는 upstream 저자(HKUST DIAL)에게 있고 이 fork도 해당 부분에 대해 동일한 조건을 따른다. `src/data_agent_baseline/` 트리는 Phase 1 동안 team1438이 거의 전면 재작성했다.

이 레포는 Phase 1 종료(최종 심사 2026-07-14 마감) 이후 아카이브·회고 목적으로 공개되었다. 대회 기간 전체 동안은 팀 간 공유 금지 룰에 따라 비공개였다. `.claude/skills/kddcup-rules-*` 팩은 [공식 룰](https://dataagent.top/rules)을 요약·의역한 것이므로 공식 페이지를 기준으로 삼을 것.