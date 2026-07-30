---
name: kddcup-rules-runtime
description: 운영진 평가 컨테이너의 런타임 spec — 마운트 (/input RO, /output RW, /logs RW), 주입 환경변수 (MODEL_API_URL/KEY/NAME), 네트워크 격리. "마운트 어디", "env var 뭐 들어와", "/input 쓰기 가능?", "외부 인터넷 되는지" 같은 런타임 환경 컴플라이언스 질문에서 트리거. 우리 코드가 어떻게 받아쓰는지(operational)는 kddcup-submission 참조.
---

# Rules — Runtime Environment

원천: https://dataagent.top/rules. 이 스킬은 **룰이 명시한 런타임 환경**만 다룬다 — 우리 코드 wiring은 `kddcup-submission`.

---

## 1. 컨테이너 실행 명령 (운영진 측)

운영진 SIGTERM 이메일에서 직접 확인된 실제 명령 (2026-05):

```bash
timeout --signal=TERM --kill-after=30s "${MAX_RUNTIME_SECONDS}s" \
  docker run --rm --name "${CONTAINER_NAME}" \
  --network=host --cpus=16 --memory=64g --memory-swap=64g \
  -v "${PRIVATE_DATA_DIR}/input:/input:ro" \
  -v "${EVAL_RESULT_OUTPUT_DIR}/output:/output:rw" \
  -v "${EVAL_RESULT_OUTPUT_DIR}/logs:/logs:rw" \
  -e MODEL_API_URL="${MODEL_API_URL}" \
  -e MODEL_API_KEY="EMPTY" \
  -e MODEL_NAME="${MODEL_NAME}" \
  "${DOCKER_IMAGE}"
```

**`${MAX_RUNTIME_SECONDS}`**:
- A-board: **7200** (2시간)
- B-board: **43200** (12시간)

**`MODEL_API_KEY="EMPTY"`** — 실제 API key 값은 `"EMPTY"` 문자열. 우리 코드는 빈/`EMPTY` 토큰을 OpenAI 클라이언트에 그대로 넘겨주면 됨 (vLLM 같은 self-hosted는 token 검증 안 함).

`--network=host` — 컨테이너가 호스트 네트워크 namespace 사용 (이전 추정 `eval_net`은 잘못). 그러나 host 외부로의 통신은 방화벽으로 차단되며, `MODEL_API_URL` 도메인만 접근 허용.

각 플래그가 곧 룰이다 — 변경 불가.

---

## 2. 마운트 (3개)

| 마운트 | 권한 | 용도 |
|---|---|---|
| `/input` | **Read-Only** | 태스크 데이터. 수정 시도 = 즉시 룰 위반 |
| `/output` | Read-Write | 우리 결과물 (`prediction.csv`)을 task별 디렉토리에 작성 |
| `/logs` | Read-Write | 디버그용. 우리는 JSONL `/logs/runtime.log`로 mirror |

**`/input` 구조 (룰 명시):**
```
/input/
  └─ task_<id>/
     ├─ task.json
     └─ context/    (csv/, db/, json/, doc/, knowledge.md — 가변)
```

**`/output` 형식 (필수):**
```
/output/
  └─ task_<id>/
     └─ prediction.csv
```

`/output/<task_id>/prediction.csv`가 채점 대상. 다른 위치에 쓰면 **그 태스크 점수 0**. 자세한 prediction.csv 포맷은 `kddcup-rules-output`.

---

## 3. 주입 환경변수 (3개)

평가 시점에 운영진이 컨테이너에 주입:

| 변수 | 의미 |
|---|---|
| `MODEL_API_URL` | 운영진 내부 Qwen 엔드포인트 (OpenAI Chat Completions 호환) |
| `MODEL_API_KEY` | 운영진 발급 인증 토큰 |
| `MODEL_NAME` | 값: `"qwen3.5-35b-a3b"` |

**규칙:**
- 우리 코드는 이 셋을 **반드시 환경변수에서 읽어야 한다**
- **하드코딩 금지** — `configs/eval.yaml`이 이 필드를 빈 문자열로 두는 이유
- 우리 `config.py`는 env > YAML > default 우선순위로 자동 오버레이

**위반 예시:** `agent.api_base = "http://my-server:8000"`처럼 YAML에 박아두면 룰 위반 가능성. eval.yaml은 **절대 채우지 말 것**.

---

## 4. 네트워크 격리

| 통로 | 가능 |
|---|---|
| `MODEL_API_URL` | ✅ |
| **그 외 모든 외부 접속** | ❌ (완전 차단) |

**시사점:**
- 빌드 시점에 모든 deps 동결 (`uv sync --frozen`)
- 런타임 패키지 다운로드 시도 = 즉시 실패
- 외부 API 호출 (다른 LLM, HuggingFace Hub 등) = 즉시 실패 + 룰 위반
- DNS 자체가 막혔을 가능성 — `pip install`, `curl` 등 시도 금지

`--network=host`로 호스트 네트워크 namespace를 공유하지만, 외부 트래픽은 방화벽으로 차단되며 `MODEL_API_URL`만 도달 가능. inter-container 통신·host port 직접 접근도 모두 금지.

---

## 5. 컨테이너 구조 — Read-Only 트리

```
/input/                    (read-only)  ← 수정 시 룰 위반
  └─ task_<id>/
     ├─ task.json
     └─ context/

/output/                   (read-write) ← 결과 작성
  └─ task_<id>/
     └─ prediction.csv

/logs/                     (read-write) ← 디버그
```

**금지:**
- `/input`에 파일 생성·수정·삭제 시도
- `/input`을 임시 작업 공간으로 활용 (mkdir, touch 등)
- 주입 env var 삭제·변조 (`del os.environ["MODEL_API_KEY"]`)

작업용 임시 공간이 필요하면 `/tmp` (컨테이너 내 일반 tmpfs) 또는 `/output` 사용.

---

## 6. 룰 위반 트리거 (런타임 측)

| 행위 | 결과 |
|---|---|
| `MODEL_API_URL` 외 외부 endpoint 호출 시도 | 룰 위반 |
| API 정보를 코드/YAML/Dockerfile에 하드코드 | 룰 위반 가능성 |
| `/input` 수정 | 룰 위반 |
| 주입된 env var 변조 | 룰 위반 |
| 평가 인프라 probing (마운트 외 디렉토리 접근, 호스트 정보 수집 등) | 룰 위반 |

자세한 금지 행위는 `kddcup-rules-prohibitions`.

---

## 7. 컴플라이언스 체크리스트 (제출 전)

- [ ] `configs/eval.yaml`의 `api_base`/`api_key`/`model`이 빈 문자열인가
- [ ] env-var override가 실제 동작하는가 (잘못된 YAML + 정상 env로 docker run → env가 이김)
- [ ] `/output/task_<id>/prediction.csv` 경로로 결과가 떨어지는가 (`flat_output_dir: true` 확인)
- [ ] 컨테이너 안에서 외부 호출 시도하는 코드 없는가 (deps 다운로드, 분석 라이브러리의 telemetry 등)
- [ ] `/input`을 RO로만 다루는가 (`resolve_context_path`가 강제)
