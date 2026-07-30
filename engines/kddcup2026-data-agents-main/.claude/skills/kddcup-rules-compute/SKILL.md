---
name: kddcup-rules-compute
description: 운영진 평가 컨테이너의 하드웨어 한도 — 16 vCPU, 64 GB RAM, GPU 없음. A-board 2시간 / B-board 12시간 wall-clock cap, SIGTERM→30초→SIGKILL. "메모리 얼마", "GPU 쓸 수 있어?", "타임아웃 어떻게 와", "A-board와 B-board 시간 차이", "x86이야 ARM이야" 같은 컴퓨트 한도 컴플라이언스 질문에서 트리거.
---

# Rules — Compute Limits

원천: https://dataagent.top/rules (2026-05 갱신 직접 확인). 운영진 평가 컨테이너의 하드웨어 envelope. 위반 시 OOM kill / SIGKILL / 점수 누락.

---

## 1. 핵심 한도 (룰 명시)

| 항목 | 값 |
|---|---|
| **CPU** | **16 vCPU 코어 (x86-64)** |
| **RAM** | **64 GB** |
| **GPU** | **없음** |
| **A-board wall-clock** | **2시간 합계** (57 task, per-evaluation) |
| **B-board wall-clock** | **12시간 합계** (324 task, per-evaluation) |
| **종료 시그널** | **SIGTERM → 30초 후 SIGKILL** |

**아키텍처:** x86-64. ARM 빌드는 거부될 수 있다 — Mac M-series에서 `docker buildx build --platform linux/amd64` 강제.

---

## 2. A-board 2h / B-board 12h — 가장 큰 함정

룰 명시 (2026-05 업데이트):
- "A-board: Maximum 2 hours wall-clock per evaluation run"
- "B-board: Maximum 12 hours wall-clock per evaluation run"

**한 태스크 단위가 아니라 모든 태스크 합계**다. 즉:
- **A-board 57 task / 2시간 / 4 workers = 평균 task당 ~150s**
- **B-board 324 task / 12시간 / 4 workers = 평균 task당 ~133s**
- 둘 다 비슷한 압박 — Easy를 빠르게 끝내고 Hard/Extreme에 시간 몰아주는 wall-clock governor 필수
- 마지막 태스크가 SIGTERM 시점에 진행 중이면 그 30초 안에 정리해야 함 — 그 후 SIGKILL

**제출 시점에 따라 budget이 다름**:
- Phase 1 4/24 ~ 5/21 사이 제출 → **A-board 2시간 budget**
- 5/21 ~ 5/23 B-board 제출 → **12시간 budget**
- 우리가 받은 SIGTERM 메일들 (v4, v6)은 모두 A-board 기간이므로 **2h budget**이었을 가능성 매우 높음

**우리 대응:** `run/runner.py`의 cascading wall-clock governor (v6 도입, v7 80% margin 추가). 평균 task 시간 × 남은 task > 남은 budget × 0.8 일 때 자동 발동 → max_steps + timeout 절반. 최대 3회 cascade.

---

## 3. 64GB RAM — OOM 트리거

룰 명시: "OOM termination if exceeded"

**위협 시나리오:**
- 0.5GB CSV 8개를 동시에 pandas로 로드 → 즉시 RAM 초과
- ThreadPool에 max_workers=8로 큰 task 동시 실행 → 메모리 swarm
- persistent IPython kernel이 누적 namespace를 안 비우면 leak

**대응:**
- 큰 CSV는 `dataframe_describe` / `dataframe_head` 같은 prepass로 압축 (Phase 1)
- `max_workers` 보수적으로 — 4~6 권장 (config 기본 4)
- task별 subprocess 격리(`task_timeout_seconds > 0`)가 자연스러운 메모리 제한 — 자식 프로세스 종료 시 RAM 회수

---

## 4. GPU 없음 — CPU-only 인퍼런스

| 영향 | 대응 |
|---|---|
| 임베딩 모델은 GPU에서 못 돔 | ONNX 양자화 / 작은 모델 (sentence-transformers MiniLM 등) CPU 인퍼런스 |
| FAISS·ANN | CPU 모드로 충분 (작은 corpus) |
| OCR (pytesseract 등) | CPU OK, 그러나 PDF 텍스트 우선 추출이 빠름 |

**금지:** GPU 의존 워크로드(`torch.cuda.*`)를 메인 솔버로 쓰면 컨테이너에서 fail. dev 환경에서만 작동하고 평가 시 0점 위험.

---

## 5. 종료 시그널 시퀀스

```
A-board 2h 또는 B-board 12h 경과
  ↓
SIGTERM (graceful shutdown 신호)
  ↓ 우리에게 30초 주어짐
  ├─ runner.py SIGTERM trap이 잡음 (v6 구현됨)
  ├─ `sigterm_received` 이벤트 /logs/runtime.log에 기록
  └─ SystemExit(143) raise → finally 블록 cleanup
  ↓ 30초 후
SIGKILL (강제 종료, 복구 불가)
```

**SIGTERM trap 구현 완료** (`runner.py:_install_sigterm_trap`). v4 0.2281 / v6 0.3509 평가 결과의 SIGTERM은 모두 trap이 작동해 partial output은 보존됨 (단, 진행 중이던 task의 prediction.csv는 안 써짐).

---

## 6. CPU 16 vCPU — 병렬화 정책

| 작업 | 권장 |
|---|---|
| ThreadPool tasks | `max_workers ∈ [4, 8]` (각 worker가 subprocess 추가 spawn) |
| pandas / numpy | BLAS thread 자동 (대부분 4 thread per op로 throttle) |
| persistent kernel | task당 1워커 — 동시성은 ThreadPool에서 |

`DABENCH_MAX_WORKERS` env로 조정. **너무 늘리면 RAM 폭증** + IO 경합. 4가 균형.

---

## 7. 컴플라이언스 체크리스트

- [ ] Docker 이미지가 `linux/amd64` 플랫폼인가
- [ ] 12h 시뮬레이션이 11h 이내 (1h 여유)에 완료되는가
- [ ] 동시 RAM 사용 피크가 50GB 미만인가 (memory profiler로 측정)
- [ ] `max_workers`가 4~6 사이인가
- [ ] GPU 의존 코드가 없는가 (`grep -r "cuda\|gpu" src/`)
- [ ] SIGTERM trap이 있는가 (계획) — 30초 안에 partial flush

---

## 8. 자주 깨지는 지점

1. **Docker buildx 누락** — Mac에서 빌드 시 ARM이 들어가면 평가 시 immediate fail
2. **pandas read_csv가 dtype 추론으로 메모리 폭증** — `dtype=str`로 보수적 로딩
3. **multiprocessing fork on macOS** — 평가 환경(Linux)과 다르게 동작 → Linux 컨테이너에서 시뮬레이션 필수
4. **12h sim을 안 돌려본 채 제출** — 8h 시점 거버너가 동작하는지 확인 안 됨
5. **SIGTERM 무시** — graceful shutdown 없이 SIGKILL 받으면 진행 중 태스크 통째로 0점

운영 측 wiring은 `kddcup-submission`, 우리 작업 우선순위는 `kddcup-strategy`.
