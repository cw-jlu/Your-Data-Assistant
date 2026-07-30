---
name: kddcup-overview
description: KDD Cup 2026 DataAgent-Bench 대회의 전반적 규칙·일정·상금·모델/컴퓨트 제약을 빠르게 참조한다. 대회 개요, 트랙 구성, 제출 한도, 평가 환경, 마감일, 검증된 사실 vs 미확인 사실을 묻는 질문에 트리거. "대회가 뭐야", "마감 언제야", "Phase 2가 뭐지", "GPU 쓸 수 있어?", "상금 얼마", "트랙 몇 개" 같은 질문에서 사용.
---

# KDD Cup 2026 DataAgent-Bench — Overview

이 스킬은 챌린지의 **운영 규칙과 환경 제약**을 한눈에 보여준다. 점수식·데이터셋 구조·제출 절차·우리 코드 아키텍처는 별도 스킬:
- `kddcup-scoring` — 채점식, 정규화 규칙
- `kddcup-dataset` — task.json / context/ 구조
- `kddcup-submission` — Docker 빌드·env·mount
- `kddcup-agent` — ReAct 에이전트 내부
- `kddcup-strategy` — 우승 플랜, 제출 예산

원천: https://dataagent.top + https://dataagent.top/rules (직접 fetch 확인). 플랜 원본은 `~/.claude/plans/fuzzy-yawning-porcupine.md`, 시스템 가이드는 `docs/ARCHITECTURE.md`.

---

## 1. 챌린지 한 줄 요약

운영진이 비공개 hidden 태스크를 마운트한 Docker 컨테이너를 실행하면, 우리 컨테이너 안의 ReAct 에이전트가 자연어 질문을 읽고 컨텍스트(CSV/SQLite/JSON/Markdown/PDF)를 분석해 `prediction.csv`를 `/output/task_<id>/`에 떨어뜨린다. 운영진이 컬럼-시그니처 매칭으로 채점한다.

---

## 2. 트랙 구성

| 트랙 | 단계 | 비고 |
|---|---|---|
| **Phase 1 (단일 main 트랙)** | 자동 채점 → 공개 리더보드 | 우리가 노리는 트랙 |
| **Phase 2 Leaderboard 서브트랙** | Phase 1 통과 팀만, 더 어려운 데이터 (이미지·비디오 포함) | 통과 시 자동 진입 |
| **Phase 2 Creative 서브트랙** | 시스템 디자인·사용성 위원회 심사 | **포기** — Leaderboard만 노림 |

---

## 3. 일정 (모두 AoE / UTC-12)

| 단계 | 기간 |
|---|---|
| Launch + 데모 데이터 공개 | 2026-03-15 ~ 03-18 |
| 등록 (**703팀** 등록 완료, 1307명 참가자) | 2026-03-22 ~ 04-23 |
| 평가 시스템 freeze (workflow migration) | 2026-04-30 ~ 05-02 |
| **Phase 1 A-board** | **2026-04-24 ~ 05-21 (23:59 AoE)** |
| **Phase 1 B-board 제출** | **2026-05-21 ~ 05-23 (EoE AoE)** — 이메일 직접 제출 |
| Phase 1 B-board 내부 평가 | 2026-05-22 ~ 05-26 (EoD AoE) |
| 자격 심사 (Top 60 확정) | 2026-05-26 ~ 05-28 (AoE) |
| **Phase 2 경쟁** | **2026-05-28 ~ 06-30** |
| 최종 심사 (manual checks) | 2026-07-01 ~ 07-14 |
| 수상 알림 | 2026-07-15 (AoE) |
| **KDD 2026 발표** | **2026-08-09** |

오늘이 Phase 1 어느 시점인지는 매번 확인 (CLAUDE.md `currentDate` 참고).

**Phase 1 board 구조** (2026-05 업데이트):
- **A-board** (57 task, **2시간 wall-clock 평가**): 매 제출마다 자동 채점 → 단계적 리더보드 피드백
- **B-board** (324 task, **12시간 wall-clock 평가**): Phase 1 마지막 3일에 이메일로 직접 1회 제출
- Phase 1 최종 점수: A/B-board 가중 평균 (데이터 비율 기준)

**Top 60** 통과 시 Phase 2 진입:
- Top 1–40: Leaderboard 또는 Creative 선택
- Top 41–60: Creative만 가능

---

## 4. 상금 (Leaderboard 트랙, 단위: USD ≈ CNY/6.91)

| 순위 | 상금 |
|---|---|
| 챔피언 | $6,000 |
| 준우승 | $4,000 |
| 3위 | $2,500 |
| 4–10위 | 각 $350 |

Creative 트랙은 별도 (1위 $1,200 / 2위 $800 / 3위 $350) — 우리는 출품하지 않음.

---

## 5. 모델 제약 (강제)

| 항목 | 값 |
|---|---|
| **평가 시점 LLM** | `qwen3.5-35b-a3b` (운영진 일괄 배포) |
| 엔드포인트 | OpenAI Chat Completions 호환 |
| 주입 방식 | 환경변수 — `MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME` |
| 보조 모델 | 임베딩·소형 검색 LM **허용** (CPU 인퍼런스 한정) |
| 금지 | 다른 LLM을 메인 솔버로 쓰면 실격. CPU 로컬 인퍼런스 우회 명시적 금지 |

**개발 시점**: 다른 LLM 자유. 단 **API 정보는 절대 하드코드 금지** — env에서 읽어야 한다 (`configs/eval.yaml`이 빈 문자열로 와이어드).

---

## 6. 컴퓨트 envelope (제출 컨테이너 한정)

| 항목 | 값 |
|---|---|
| CPU | 16 vCPU (x86-64) |
| RAM | **64 GB** (초과 시 OOM kill) |
| GPU | **없음** |
| **A-board wall-clock** | **2시간** (per-evaluation 합계, 57 task) |
| **B-board wall-clock** | **12시간** (per-evaluation 합계, 324 task) |
| 종료 시그널 | SIGTERM → 30초 후 SIGKILL |
| 네트워크 | `MODEL_API_URL` **외 전부 차단** |

**시사점:**
- 외부 패키지 다운로드 불가 → Docker 빌드 시점에 모든 deps 동결
- GPU 없음 → 임베딩 모델은 ONNX/CPU
- **A-board 2h가 가장 빡빡한 제약** — 57 task / 4 workers = 평균 task당 ~150s만 허용. wall-clock governor + timeout cap 필수.
- B-board 12h는 324 task에 12h × 4 workers / 324 = task당 ~133s. A-board와 비슷한 압박.
- 단순 비례: 50-task public 측정 wall-clock을 A-board용으로 환산 시 1.14× (57/50), B-board용 6.48× (324/50)

---

## 7. 제출 규칙

| 항목 | 값 |
|---|---|
| 형식 | Docker 이미지 → `docker save | gzip` → `.tar.gz` |
| **사이즈 한도** | **≤ 10 GB** (1GB 여유 남기는 게 안전) |
| 이미지 네이밍 | `<team_id>:v<N>` (이미지) / `<team_id>_v<N>.tar.gz` (아카이브) |
| 전달 | Google Drive 공유 링크 → 운영진 이메일 |
| 권한 | "Anyone with the link can view" |
| **빈도** | **하루 1회** |
| **Phase 1 총량** | **30회** |
| 직렬성 | 직전 평가 완료까지 다음 제출 보류 |
| 이메일 | (운영진 공지 참고; 공식 사이트 / Discord에서 확인) |

**시사점:** submission은 **신성한 자원**. mock_scorer + holdout 둘 다 그린일 때만 제출. 우리는 30회 중 ~7회만 실사용 예정 (자세한 분배는 `kddcup-strategy`).

---

## 8. 컨테이너 마운트 + 입출력 (요약)

```
/input/                    (read-only)
└─ task_<id>/
   ├─ task.json            # {task_id, difficulty, question} — exact key set
   └─ context/             # csv/, db/, json/, doc/, knowledge.md (가변)

/output/                   (read-write)
└─ task_<id>/
   └─ prediction.csv       # 우리가 작성 — UTF-8 CSV, header + rows

/logs/                     (read-write)
└─ runtime.log             # JSONL, 운영진 디버그용
```

자세한 데이터 포맷·예시는 `kddcup-dataset`. 우리 컨테이너 wiring은 `kddcup-submission`.

---

## 9. 공식 페이지 링크

| 자료 | URL |
|---|---|
| 공식 | https://dataagent.top |
| 룰 | https://dataagent.top/rules |
| 스타터 킷 | https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit |
| Discord | https://discord.com/invite/7eFwJQN3Fx |
| Phase 1 데모 데이터 | Google Drive (공식 사이트 링크 참조) |

---

## 10. 검증된 사실 vs 아직 모르는 것

**검증됨 (rules 페이지 직접 확인, 2026-05 갱신):**
- 모델·env·mount·네트워크 정책
- 16 vCPU / 64GB
- **A-board 2h / B-board 12h** wall-clock budget
- 제출 1/일, Phase 1 30회 한도, 10GB tarball
- 점수식 골격 (`Recall − λ·Extra/Pred`), 정규화 규칙
- **Phase 1 hidden 분포 (공식 발표)**:
  - A-board 57 task: easy 10 (17.5%) / medium 15 (26.3%) / hard 30 (52.6%) / extreme 2 (3.5%)
  - B-board 324 task: easy 68 (21.0%) / medium 136 (41.9%) / hard 115 (35.5%) / extreme 5 (1.5%)
  - **Total 381 task** (이전 추정 "약 400"의 실제 값)
- 데이터 modality: csv/json/db/doc + knowledge.md. Phase 2에서 image/video 추가

**미확인 — Discord/이슈로 직접 질의 필요:**
- λ 정확값 (우리 추정 0.10)
- 운영진 엔드포인트의 실제 latency (우리 DGX 대비 몇 배)
- qwen 엔드포인트의 `response_format={"type":"json_object"}` 또는 `guided_json` 지원 여부
- B-board 평가 인프라가 A-board와 동일한지 (CPU 클라스, 큐 부하 등)

---

## 11. 의도적으로 안 하는 것

| 안 하는 것 | 이유 |
|---|---|
| Creative 서브트랙 출품 | 인력 부족, Leaderboard 집중 |
| 모델 fine-tuning | 평가 시 운영진 호스팅 모델만 허용 → fine-tune 무용 |
| Adversarial schema probing (의도적 minimal/max 제출로 gold schema 역추정) | 룰 위반 가능성 + 30회 예산 낭비 |
| 다른 LLM을 메인 솔버로 사용 | 즉시 실격 |

전략 전체는 `kddcup-strategy` 참조.
