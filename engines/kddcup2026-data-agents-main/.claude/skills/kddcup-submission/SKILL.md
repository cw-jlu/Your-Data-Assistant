---
name: kddcup-submission
description: Docker 빌드, 평가 컨테이너 wiring, env 오버레이, /input·/output·/logs 마운트, build_submission/local_eval 스크립트, 그리고 우리 환경 vs 운영진 환경의 차이를 다룬다. "Docker 어떻게 빌드해", "env 변수 어떻게 받아", "submission 어떻게 만들어", "/output 경로 안 맞아", "configs/eval.yaml", "flat_output_dir", "로컬에서 평가 시뮬레이션", "vLLM 띄우기" 같은 질문에서 트리거.
---

# KDD Cup 2026 — Submission Pipeline

Docker 이미지 빌드부터 운영진 평가 컨테이너까지의 전체 wiring. 대회 규칙은 `kddcup-overview`, 코드 내부는 `kddcup-agent`, 점수 검증은 `kddcup-scoring` 참조.

---

## 1. 세 가지 실행 모드

같은 코드베이스가 세 가지 컨텍스트에서 돈다. 모드에 따라 `config` 동작과 출력 경로가 다르다.

| Mode | 트리거 | 출력 경로 | run_id wrapper | LLM endpoint |
|---|---|---|---|---|
| **로컬 개발** | `uv run dabench run-task ... --config configs/local.yaml` | `artifacts/runs/<run_id>/<task_id>/` | 있음 | DGX vLLM (LAN) |
| **로컬 Docker 스모크** | `bash scripts/local_eval.sh <ver> [task_set]` | `artifacts/sandbox/<ver>/output/<task_id>/` | 없음 (flat) | DGX vLLM via env |
| **운영진 평가** | 운영진이 우리 tar.gz를 docker run | `/output/<task_id>/` (mount) | 없음 (flat) | 운영진 Qwen endpoint |

핵심 토글은 `flat_output_dir`. **Eval mode = True** (운영진이 `/output`을 빈 채로 마운트하므로 `<run_id>/` 래퍼를 만들면 안 됨). 로컬 개발은 False (run_id 래퍼로 버전 비교).

`configs/eval.yaml`이 wired:
```yaml
flat_output_dir: true
log_file: /logs/runtime.log
```

---

## 2. 환경변수 오버레이

`config.py:load_app_config()`의 우선순위:

```
env > YAML > dataclass default
```

이 순서가 핵심이다. **운영진은 `MODEL_API_URL` 등을 컨테이너에 주입하므로, YAML이 env를 덮으면 안 된다.** `configs/eval.yaml`은 이런 필드를 빈 문자열로 두어 명시적으로 env에 양보.

| dataclass | 필드 | env | 비고 |
|---|---|---|---|
| `AgentConfig` | `model` | `MODEL_NAME` | 운영진 → `qwen3.5-35b-a3b` |
| `AgentConfig` | `api_base` | `MODEL_API_URL` | 운영진 endpoint |
| `AgentConfig` | `api_key` | `MODEL_API_KEY` | |
| `DatasetConfig` | `root_path` | `DABENCH_INPUT_DIR` | 운영진 `/input` |
| `RunConfig` | `output_dir` | `DABENCH_OUTPUT_DIR` | 운영진 `/output` |
| `RunConfig` | `run_id` | `DABENCH_RUN_ID` | 빈 문자열 → None → auto UTC |
| `RunConfig` | `max_workers` | `DABENCH_MAX_WORKERS` | ThreadPool 크기 |
| `RunConfig` | `task_timeout_seconds` | `DABENCH_TASK_TIMEOUT` | per-task subprocess |
| `RunConfig` | `flat_output_dir` | `DABENCH_FLAT_OUTPUT` | `1/true/yes` → True |
| `RunConfig` | `log_file` | `DABENCH_LOG_FILE` | `/logs/runtime.log` |

스코어링 측:
- `DABENCH_LAMBDA` — mock_scorer 기본 λ
- `DABENCH_LAMBDAS` — `local_eval.sh`의 다중 λ (공백 구분)

`run_id` 정규화: 빈 문자열 → None → auto-generated UTC `YYYYMMDDTHHMMSSZ`. `runner.py`가 `.`, `..`, `/`, `\` 포함된 값을 거부.

---

## 3. 마운트 (운영진 평가)

```
/input/                    (read-only)
└─ task_<id>/              # 우리는 모든 디렉토리를 순회
   ├─ task.json
   └─ context/

/output/                   (read-write)
└─ task_<id>/
   └─ prediction.csv       # required output

/logs/                     (read-write)
└─ runtime.log             # JSONL — 우리가 mirror
```

운영진 docker run 형태 (rules에 명시):
```bash
docker run --rm \
  --network=eval_net --cpus=16 --memory=64g \
  -v /input:/input:ro -v /output:/output:rw -v /logs:/logs:rw \
  -e MODEL_API_URL=... -e MODEL_API_KEY=... -e MODEL_NAME=qwen3.5-35b-a3b \
  <team_id>:v<N>
```

---

## 4. `Dockerfile` 구조 (현재)

```dockerfile
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ...
WORKDIR /app

# 최소 시스템 deps (every byte counts)
RUN apt-get install -y --no-install-recommends ca-certificates curl

# uv pinned
RUN pip install --no-cache-dir uv==0.5.14

# 캐싱 최대화 — lock + 메타 먼저
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 프로젝트 소스
COPY src ./src
COPY configs/eval.yaml ./configs/eval.yaml
RUN uv sync --frozen --no-dev

# CLI 로드 스모크
RUN uv run dabench --help > /dev/null

ENTRYPOINT ["uv", "run", "dabench", "run-benchmark", "--config", "configs/eval.yaml"]
```

**원칙:**
- ≤ 9 GB (1GB 여유 — 10GB 한도)
- `--frozen`으로 결정성 보장
- README.md는 `pyproject.toml`이 참조하므로 반드시 COPY
- deps와 프로젝트 install을 분리 → 소스만 변하면 deps 레이어 캐시 살림

---

## 5. 빌드·평가 스크립트

`scripts/build_submission.sh`:
```bash
bash scripts/build_submission.sh v1
# → submissions/dabench_v1.tar.gz
# 10 GB 초과 시 fail
# SUBMISSION_LOG.md 자동 갱신 (계획)
```

`scripts/local_eval.sh`:
```bash
# 전체 50 태스크
bash scripts/local_eval.sh v1

# 홀드아웃 10개만 (제출 직전 검증)
bash scripts/local_eval.sh v1 data/public/holdout_ids.txt

# 멀티 λ
DABENCH_LAMBDAS="0.05 0.10 0.20" bash scripts/local_eval.sh v1
```

내부 동작: 빌드된 image를 `host.docker.internal:8000`의 vLLM에 연결, `/input`/`/output`/`/logs`를 sandbox dir로 마운트, mock_scorer로 채점, λ sensitivity 출력.

---

## 6. 로컬 vLLM 서빙 (DGX 또는 회사 GPU)

```bash
# DGX에서 실행
bash scripts/serve_qwen_docker.sh         # 시작
bash scripts/serve_qwen_docker.sh --probe-only
bash scripts/serve_qwen_docker.sh --stop
```

**왜 자체 호스팅인가?** 운영진이 `qwen3.5-35b-a3b`를 강제하므로 로컬 개발 모델 = 평가 모델로 맞추는 게 가장 큰 무기다. GPT-4o로 프롬프트 튜닝하면 평가 시 무용. 가중치 차이가 있다면 `Qwen/Qwen3-30B-A3B-Instruct-2507`로 alias하고 SUBMISSION_LOG에 기록.

자세한 capabilities (max context, JSON-mode 지원, latency, throughput)는 `docs/qwen_endpoint_capabilities.md`에 기록.

---

## 7. 운영진 환경 시뮬레이션 — 한 번에

```bash
# 1. 빌드
bash scripts/build_submission.sh v1

# 2. 로컬 docker run (운영진 환경 모방)
docker run --rm \
  -e MODEL_API_URL=http://host.docker.internal:8000/v1 \
  -e MODEL_API_KEY=local \
  -e MODEL_NAME=qwen3.5-35b-a3b \
  -v $(pwd)/data/public/input:/input:ro \
  -v $(pwd)/artifacts/sandbox/v1/output:/output \
  -v $(pwd)/artifacts/sandbox/v1/logs:/logs \
  dabench:v1

# 3. 채점
bash scripts/local_eval.sh v1 data/public/holdout_ids.txt
```

**Pre-submission ship gate (모두 통과해야 함):**
1. local_eval.sh가 holdout 무에러 완주
2. mock_scorer가 0~1 사이 숫자 출력 (0이면 정규화/path 버그)
3. Docker 이미지 ≤ 9 GB
4. env-var override 확인 — 의도적 잘못된 YAML + 정상 env로 실행 → env가 이김

---

## 8. `/logs/runtime.log` 포맷 (JSONL)

운영진이 디버그용으로 요구. 매 라인 한 이벤트:

```json
{"ts":"2026-04-27T10:39:05Z","run_id":"...","event":"benchmark_start"}
{"ts":"...","run_id":"...","event":"task_done","task_id":"task_74","succeeded":true,"elapsed_seconds":12.3,"failure_reason":null,"wrote_prediction":true}
{"ts":"...","run_id":"...","event":"benchmark_end","task_count":5,"succeeded_task_count":5}
```

`run/runner.py:RuntimeLogger` 담당. `config.run.log_file` 설정 시 자동 활성. eval.yaml은 `/logs/runtime.log`로 와이어드.

---

## 9. Submission 워크플로우

```bash
# 1. 변경 통합 → mock + holdout 그린 확인
uv run dabench run-benchmark --config configs/local.yaml \
  --task-set data/public/holdout_ids.txt
uv run python -m data_agent_baseline.scoring.mock_scorer \
  --predictions artifacts/runs/<run_id> --gold data/public/output \
  --input data/public/input --lambda-values 0.05 0.10 0.20

# 2. 컨테이너로 동일 결과 재현
bash scripts/build_submission.sh v<N>
bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt

# 3. 운영진 제출
# - submissions/dabench_v<N>.tar.gz 를 Google Drive에 업로드
# - "Anyone with the link can view" 권한
# - 공식 이메일로 링크 전송
# - docs/SUBMISSION_LOG.md 갱신: (날짜, 변경 요약, holdout, leaderboard, 회고)
```

---

## 10. 우리 환경 vs 운영진 환경 — 같이 봐야 하는 차이

| 항목 | 우리 (DGX vLLM) | 운영진 평가 |
|---|---|---|
| LLM | DGX의 `Qwen3-30B-A3B-Instruct-2507`을 `qwen3.5-35b-a3b` alias로 노출 | 진짜 `qwen3.5-35b-a3b` 가중치 |
| Max context | 32K (YaRN 미적용) | 262144 토큰 (`--max-model-len 262144`, 운영진 명시) |
| 네트워크 | LAN 자유 | `MODEL_API_URL`만 (`--network=host` + 방화벽) |
| 컴퓨트 | 풍족 | 16 vCPU / 64GB / **A-board 2h** 또는 **B-board 12h** |
| `/input` 데이터 | 50개 공개셋 | **A-board 57 task** (easy 10 / med 15 / hard 30 / extreme 2) 또는 **B-board 324 task** (easy 68 / med 136 / hard 115 / extreme 5) |
| `/output` | sandbox dir | 평가 서버의 빈 디렉토리 |
| `flat_output_dir` | `local_eval.sh`는 True (eval.yaml 사용) | True |
| Endpoint latency | 우리 DGX baseline (~5-20s per LLM call) | ~1.8× slower 추정 (v6 SIGTERM 역산) |
| MODEL_API_KEY | `local` 또는 임의 문자열 | **`"EMPTY"`** (운영진 명시) |

**가장 큰 미지수:**
1. JSON-mode (`response_format`) 지원 여부
2. Max context 윈도우
3. λ 정확값 (우리 추정 0.10)

---

## 11. 자주 깨지는 지점

1. **`flat_output_dir` 미적용** — 운영진 채점기는 `/output/task_<id>/prediction.csv`를 못 찾음. eval.yaml이 책임지는데 새 config 만들 때 누락 가능
2. **`set -euo pipefail` + macOS의 없는 `ip` 명령** — `local_eval.sh`가 침묵 종료 (이미 fix: `a3b42f2`)
3. **Dockerfile + `uv sync`** — pyproject가 `readme = "README.md"`를 요구 → COPY 필요 (이미 fix)
4. **Submission tarball 10GB 한도** — `build_submission.sh`가 빌드 후 사이즈 fail
5. **API key 하드코드 금지** — eval.yaml은 빈 문자열 유지. 변경하면 룰 위반 가능성
6. **Network 차단** — 빌드 시점에 모든 deps 동결. 런타임 다운로드 시도하면 즉시 실패

---

## 12. 진입점 한 줄 매핑

| 다루는 것 | 파일 |
|---|---|
| Dockerfile | 루트 `Dockerfile` |
| Eval 컨테이너 config | `configs/eval.yaml` (gitignore 화이트리스트) |
| 로컬 dev config | `configs/local.yaml` (gitignore 화이트리스트) |
| 빌드 스크립트 | `scripts/build_submission.sh` |
| 로컬 평가 스크립트 | `scripts/local_eval.sh` |
| vLLM 서빙 | `scripts/serve_qwen_docker.sh` |
| Env 오버레이 | `src/data_agent_baseline/config.py:load_app_config` |
| `flat_output_dir` 분기 | `src/data_agent_baseline/run/runner.py:create_run_output_dir` |
| RuntimeLogger (JSONL) | `src/data_agent_baseline/run/runner.py:RuntimeLogger` |
| 제출 로그 | `docs/SUBMISSION_LOG.md` (gitignore 화이트리스트) |
| Endpoint capabilities | `docs/qwen_endpoint_capabilities.md` |
