# CLAUDE.md

> 🌐 **Language**: [English](CLAUDE.md) · **한국어** · [中文](CLAUDE.zh.md)

이 파일은 본 레포에서 코드를 다룰 때 Claude Code (claude.ai/code) 가 따라야 할 가이드를 제공한다.

## Project

KDD Cup 2026 DABench / DataAgent-Bench 챌린지를 위한 ReAct 베이스라인 (`data-agent-baseline`, package `data_agent_baseline`). `data/public/input/`에서 task를 읽어 LLM 기반 ReAct 에이전트를 per-task 도구로 실행하고, task별로 `prediction.csv` + `trace.json`을 작성한다.

Python ≥3.10, `uv` 관리. CLI 진입점 `dabench`는 `[project.scripts]`에 정의되어 있고 `data_agent_baseline.cli:main`으로 resolve된다.

## 자주 쓰는 명령

> **macOS dev + DGX vLLM 분리 워크플로 (team1438):** vLLM은 DGX (`<VLLM_HOST>:8000`) 서빙 전용. 빌드 / 테스트 / 제출 packaging은 macOS local. 아래 명령들은 모두 macOS에서 실행하고 LLM endpoint는 env로 주입한다 — `export MODEL_API_URL=http://<VLLM_HOST>:8000/v1; export MODEL_API_KEY=local; export MODEL_NAME=qwen3.5-35b-a3b`. `Dockerfile`은 `--platform=linux/amd64` 핀이 박혀 있어 Apple Silicon에서도 운영진 amd64 환경 호환 이미지가 나온다. 자세한 단계는 [`README.ko.md`](README.ko.md) §3.

```bash
uv sync                                                            # uv.lock에서 deps 설치
uv sync --extra dev                                                # pytest + ruff 포함
uv run dabench status        --config configs/local.yaml           # 경로 + 데이터셋 존재 여부
uv run dabench inspect-task task_<id> --config configs/local.yaml  # task 메타데이터 + context tree
uv run dabench run-task     task_<id> --config configs/local.yaml  # 단일 task end-to-end
uv run dabench run-benchmark          --config configs/local.yaml  # 전체 task
uv run dabench run-benchmark          --config configs/local.yaml --task-set data/public/holdout_ids.txt  # holdout만
uv run dabench run-benchmark          --config configs/local.yaml --limit 5  # 스모크
uv run ruff check src                                              # lint (line-length 100, py310)
uv run pytest                                                      # tests/ (gitignored, 로컬 실행)
uv run pytest tests/path/to/test_x.py::test_name                   # 단일 테스트

# Holdout split (data/public/input/에 데이터 떨군 후 1회):
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public

# Local mock scoring (gold 대비):
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> --gold data/public/output \
    --input data/public/input --lambda-values 0.05 0.10 0.20

# Docker submission 빌드 + 점수:
bash scripts/build_submission.sh v3               # → submissions/dabench_v3.tar.gz
bash scripts/local_eval.sh v3                     # 이미지에 대한 full holdout
bash scripts/local_eval.sh v3 data/public/holdout_ids.txt  # holdout 부분집합만
```

추적되는 configs: `configs/eval.yaml` (Docker submission, env-overridable) 와 `configs/local.yaml` (로컬 vLLM 가정 dev 기본값 — 새 dev config 작성 시 시작점). 다른 모든 `configs/*` 는 gitignored. 추적되는 docs: `docs/ARCHITECTURE.md`, `docs/SYSTEM_FLOW.md`, `docs/DATA_ANALYSIS.md`, `docs/SUBMISSION_LOG.md`, `docs/HARNESS_STRUCTURE.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/qwen_endpoint_capabilities.md` 와 그 `.ko.md` / `.zh.md` 변형. 다른 docs는 gitignored.

## Architecture

런타임은 엄격한 layered pipeline이다. 한 레이어를 건드리면 위·아래 레이어 모두 다시 읽어야 한다.

`**cli.py` (Typer)** — 4개 subcommand 선언하고 progress UI orchestrate. `PROJECT_ROOT`를 `Path(__file__).resolve().parents[2]`로 정의; YAML의 상대경로는 `config.py`가 이 root 기준으로 resolve.

`**config.py`** — YAML을 frozen dataclass (`AppConfig` → `DatasetConfig | AgentConfig | RunConfig`)로 로드. 우선순위는 **env > YAML > default**. 평가 컨테이너는 이를 의존: 운영진이 `MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME`을 runtime에 주입하면 우리 YAML이 이를 가리면 안 됨. 다른 env knob들: `DABENCH_INPUT_DIR`, `DABENCH_OUTPUT_DIR`, `DABENCH_RUN_ID`, `DABENCH_MAX_WORKERS`, `DABENCH_TASK_TIMEOUT`, `DABENCH_FLAT_OUTPUT`, `DABENCH_LOG_FILE`, `DABENCH_LAMBDA` (mock scorer), `DABENCH_REPEAT_MAX`, `DABENCH_PASS_SAFETY_MARGIN` (multi-pass), `DABENCH_DISABLE_SELF_CONSISTENCY`. 모든 path는 absolute `Path`로 반환. `run_id`는 정규화: 빈 문자열 → `None` → 자동 생성 UTC timestamp; `runner.py`의 resolver가 `.`, `..`, `/` 또는 `\` 포함 시 reject.

`**benchmark/dataset.py` + `schema.py`** — `DABenchPublicDataset`이 `task_<N>` 디렉토리 발견, 숫자 suffix로 정렬, 각 `task.json`이 *정확히* `{task_id, difficulty, question}` 키를 가지는지 검증. 키 불일치 시 raise — 스키마를 silently 확장하지 말 것. `PublicTask`는 `task_dir`와 `context_dir` 둘 다 노출; 도구는 `context_dir`에서만 읽어야 함.

`**agents/*`* — ReAct 루프:

- `model.py` — `ModelAdapter` Protocol; `OpenAIModelAdapter`는 OpenAI-호환 `api_base`에 대해 `chat.completions.create` 호출. `ScriptedModelAdapter`는 테스트용.
- `prompt.py` — system / task / observation prompt builder. Contract는 엄격: 모델은 **정확히 한 개의 JSON object** (keys `thought`, `action`, `action_input`)를 단일 ```json fenced block에 wrapping해서 반환해야 함. Contract 변경 시 `react.parse_model_step`도 lockstep 업데이트 필요. **G-3:** knowledge.md cap 5000 chars + question keyword H2/H3 reorder.
- `react.py` — `ReActAgent.run`이 `max_steps`까지 iterate, 모델 응답 파싱, `ToolRegistry.execute`로 dispatch, observation을 next user message로 feed. `is_terminal=True` 반환 시 종료 (only `answer`). Parser exception은 잡혀서 `__error__` step record로 surface — agent가 같은 task 안에서 회복 가능.
- `self_consistency.py` (G-2) — `SelfConsistencyAgent`는 hard/extreme tier에서 새 OpenAIModelAdapter (temperature=0.5)로 k=3 ReActAgent를 인스턴스화하고 `column_signature` multiset majority vote.
- `runtime.py` — `StepRecord`, `AgentRuntimeState`, `AgentRunResult` dataclass (trace.json 직렬화).

`**tools/registry.py**` — agent의 도구 표면의 단일 진리 출처. `create_default_tool_registry()`가 names → `ToolSpec` (description + JSON schema example) + names → `ToolHandler` callable 와이어링. `describe_for_prompt()`가 모델이 실제로 보는 spec text라 prompt의 일부; 변경하면 agent 행동 영향. 새 도구 추가 = handler function + `ToolSpec` entry + (terminal이면) `ToolExecutionResult(is_terminal=True, answer=..., normalized_answer=...)` 반환. `_answer` handler는 raw `AnswerTable` (LLM literal 출력)와 `normalized_answer` (`scoring.normalize.normalize_answer_table` 통과) **둘 다** emit; `runner._write_task_outputs`가 default로 normalized variant를 `prediction.csv`에 작성.

`**tools/filesystem.py`, `tools/sqlite.py`, `tools/python_exec.py`** — 모든 도구 입력은 *`context/` 하위 상대경로*. `resolve_context_path`가 이 invariant 강제; absolute path나 context root 탈출 path 절대 허용 안 함. SQL 실행은 read-only.

`**run/runner.py**` — 5가지 관심사가 layered:

1. **Run dir 생성** (`create_run_output_dir(..., flat=...)`) — `flat=False`는 `mkdir(exist_ok=False)` + `<run_id>/` wrap (충돌 reject). `flat=True` (eval mode)는 `output_root/task_<id>/...` 직접 작성, 사전 존재 dir 허용; 내부적으로 trace metadata용 run_id 합성. `config.run.flat_output_dir` / `DABENCH_FLAT_OUTPUT`로 wired.
2. **Per-task timeout 격리** (`_run_single_task_with_timeout`) — `task_timeout_seconds > 0`일 때 task가 `multiprocessing.Process` + queue에서 실행; 부모가 timeout 시 자식 kill. `task_timeout_seconds <= 0`은 in-process. **부수효과:** timeout mode에서 각 task는 자식 안에서 자체 `OpenAIModelAdapter`와 `ToolRegistry`를 생성; `model=` / `tools=` override 못 넘김. runner는 override가 제공되면 single-process mode로 fallback (max_workers=1 강제) — `ScriptedModelAdapter` 테스트용.
3. **Parallelism** — `run_benchmark`는 `>1` workers에서 `ThreadPoolExecutor(max_workers=...)` 사용 (각 worker가 task용 자체 subprocess spawn), override 제공 또는 `max_workers == 1`일 때 공유 model/tools serial loop. `--task-set <file>` (또는 `task_filter` kwarg)로 task id list로 제한 — holdout-only iteration용.
4. **Multi-pass orchestrator (H-1)** — `run_benchmark_with_passes`가 `repeat_max > 1`일 때 같은 task set을 N번 풀고 cross-run vote. `output_dir/_runs/run_<i>/`에 per-pass 작성, 끝에 `cross_run_vote`로 `output_dir/task_<id>/`에 voted final 작성. `pass_safety_margin × last_pass_duration` budget guard로 회귀 안전성 보장; voter 실패 시 `_fallback_copy_pass`로 pass 0 복사.
5. **Runtime log** (`RuntimeLogger`) — `config.run.log_file` 설정 시 runner가 JSON Lines 이벤트 (`benchmark_start`, `task_done`, `benchmark_end`, multi-pass 이벤트 5종) append. eval rules는 `/logs/runtime.log` 요구; `configs/eval.yaml`이 자동 wire.

`execute_python` (`tools/python_exec.py`)도 hard 30s timeout (`tools/registry.py`의 `EXECUTE_PYTHON_TIMEOUT_SECONDS`)로 subprocess-isolated, `task.context_dir`로 chdir, stdout/stderr fd-level capture. Nested multiprocessing: `task_timeout_seconds > 0`이고 agent가 `execute_python` 호출하면 process-in-a-process. 그걸 견디지 못하는 debugger에서 benchmark 실행 X.

`**scoring/`** — official DataAgent-Bench scorer 로컬 미러. 대회 룰 `Score = Recall − λ·(ExtraCols/PredictedCols)`, column-signature matching이 컬럼 이름과 row 순서 무시 (정규화 후: numeric → 2 decimals, dates → ISO 8601, nulls → `""`, strings → trimmed case-sensitive).

- `scoring/normalize.py` — `normalize_value`, `normalize_column`, `normalize_answer_table`, `column_signature`. policy 미지정 시 per-cell type 자동 감지. `_answer`에 wired되어 raw alongside normalized variant 생산.
- `scoring/mock_scorer.py` — `predictions/`와 `gold/` root 주어지면 per-task `score`, `recall`, `matched/gold/predicted` columns 계산. CLI는 λ-sensitivity sweep용 `--lambda-values` 지원. leaderboard λ는 비공개; default 0.10, column-ablation은 `{0.05, 0.10, 0.20}` 일치 게이트.
- `scoring/cross_run_vote.py` (H-1) — N개 benchmark output root 사이의 column-multiset majority voter. CLI + library. `vote_across_runs(prediction_roots, output_dir)`이 host-side ensemble 측정 (컨테이너 내부의 in-runner voting과 동일 알고리즘) 가능.
- `scoring/holdout.py` — 결정적 80/20 hash split (`blake2b(salt + task_id)`), `train_ids.txt` / `holdout_ids.txt` 생산. holdout이 prompt iteration의 유일한 honest signal; prompt 디자인에 leak시키지 말 것.

**Submission 흐름.** `Dockerfile` + `configs/eval.yaml` + `scripts/build_submission.sh`가 `submissions/dabench_<version>.tar.gz` 생산. tarball이 10 GB cap 초과 시 build script abort. `scripts/local_eval.sh`가 `host.docker.internal:8000`의 self-hosted vLLM과 `data/public/input/`에 대해 빌드된 image 실행하고 mock scorer로 prediction 파이핑.

## Dataset 레이아웃

```
data/public/input/task_<id>/
  task.json    # {task_id, difficulty, question} — 정확한 키셋, 추가 X
  context/     # CSV / JSON / SQLite / 텍스트 파일; 모든 도구 path는 여기 기준 상대경로
data/public/output/task_<id>/gold.csv  # 공개 demo gold (별도 트리)
```

Hidden test set은 `input/`만 ship — `output/` 존재에 의존하는 코드 절대 작성 X.

## Run 출력

```
artifacts/runs/<run_id>/
  <task_id>/
    trace.json        # 전체 step history + answer + failure_reason
    prediction.csv    # agent가 answer 도구 호출했을 때만 존재
  summary.json        # benchmark run에만
```

`run_id` default는 UTC timestamp `YYYYMMDDTHHMMSSZ`. `artifacts/` 트리 전체가 gitignored, `artifacts/.gitkeep` 만 예외.

## gitignored 항목 (레포에 없을 거라 기대 X)

`data/`, `tests/`, `evaluation/`, `artifacts/*` (`artifacts/.gitkeep` 제외), `submissions/`, `vllm_logs/`, 대부분의 `docs/*` (whitelist된 7개 docs × 3개 언어 변형 제외: ARCHITECTURE / SYSTEM_FLOW / SYSTEM_ARCHITECTURE / DATA_ANALYSIS / SUBMISSION_LOG / HARNESS_STRUCTURE / qwen_endpoint_capabilities), 대부분의 `configs/*` (`eval.yaml`, `local.yaml` 제외). `.codex`도 ignored. 위 트리에 새 추적 파일 추가 시 `.gitignore`에 `!path/to/file` 추가.

## Plan & 대회 컨텍스트

기억할 핵심 룰:

- 평가 시 LLM은 `MODEL_API_URL`/`MODEL_API_KEY`/`MODEL_NAME` 통해 `qwen3.5-35b-a3b`로 잠금; 하드코드 X.
- **Single-model 정책 (team1438):** 룰은 기술적으로 보조 모델 (embedding, retrieval) 허용 (하드웨어 예산 안에서)이지만 우리 팀은 qwen 메인 솔버만 사용 — harness 어떤 레이어에도 보조 LLM, sentence-transformers, FAISS, ONNX vision/embedding, 외부 web/vision API 없음. `AgentConfig.model`/`api_base`/`api_key` default가 빈 문자열이라 env 누락 시 silently 외부 provider 부르지 않고 명시 fail.
- 컴퓨트 envelope: 16 vCPU / 64 GB RAM / GPU 없음 / 모든 task **합 12시간**. LLM endpoint 외 네트워크 차단.
- 제출 cap: 1/일, 30/Phase-1. 모든 제출은 [`docs/SUBMISSION_LOG.ko.md`](docs/SUBMISSION_LOG.ko.md)에 추적.
- 마운트: `/input` (RO), `/output` (RW), `/logs` (RW). `configs/eval.yaml`이 이 경로들과 `flat_output_dir: true`로 wired.
