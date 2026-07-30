---
name: kddcup-strategy
description: 우리의 우승 플랜 — A-board/B-board 압박 분석, 30회 제출 예산 분배, holdout 디시플린, 의도적으로 안 하는 것들, 위험 등록부. "다음에 뭘 해야 해", "어떤 순서로 가야 해", "이거 지금 해도 돼", "submission 몇 번 남았어", "B-board 어떻게 준비", "왜 fine-tune 안 해" 같은 질문에서 트리거. 작업 우선순위·기능 추가 결정 시 무조건 참조.
---

# KDD Cup 2026 — Winning Strategy (2026-05 갱신)

대회 규칙은 `kddcup-overview`, 채점은 `kddcup-scoring`, 코드는 `kddcup-agent`/`kddcup-submission`. 이 스킬은 **현재 라운드 결정에 직접 영향 미치는 단일 원칙**을 담는다.

---

## 1. 결정적 인사이트 (2026-05 갱신)

원래 인사이트는 정규화 (Score 보존)였고 그건 v3 시점에 해결됨. 하지만 **v4 (0.2281 interrupt) + v6 (0.3509 interrupt)** 결과로 새로운 1순위 인사이트가 드러남:

> **Phase 1 A-board는 2시간 wall-clock 안에 57 task을 끝내야 한다.** 우리 알고리즘이 정답을 못 내는 게 아니라, **시간 안에 도달 못 하면 SIGTERM으로 점수가 깎임**. v4 32h naive → v6 13h naive → v7 7.9h naive로 단축하면서 score 0.2281 → 0.3509 → (예상) 0.55+로 추적 가능.
>
> 결론: **알고리즘 회복보다 wall-clock 봉쇄가 항상 우선**. 한 task perfect 회복 (+0.018) vs wall-clock 단축 (+0.05~+0.20) — 후자가 훨씬 큰 leverage.

또한 **B-board 5/21~23 1회 제출**이 별도 트랙. 12h budget · 324 task 분포가 A-board와 다름 (hard 비율 ↓, medium 비율 ↑).

---

## 2. 전략 원칙 (의사결정 시 우선순위)

1. **Wall-clock 봉쇄 > 알고리즘 회복.** SIGTERM 받으면 알고리즘 회복 무의미. 추가 task당 wall-clock 단축이 항상 우선.
2. **B-board 1회는 신성하다.** 5/21~23 단 한 번 제출 — A-board 결과를 보고 가장 안전한 build 선택.
3. **점수 보존 > 점수 추가.** 정규화/over-emission validator. v3에서 해결됨, 유지.
4. **mock_scorer + wall-clock 둘 다 그린일 때만 제출.** v6 30회 한도 잘 쓰자.
5. **로컬 개발 모델 = 평가 모델.** DGX vLLM `qwen3.5-35b-a3b` 그대로. 운영진 endpoint latency가 우리보다 약 1.8× slower 추정 (v6 SIGTERM 역산).

---

## 3. 현재 상황 + 이전 라운드 회고

| 라운드 | 빌드 | 운영진 결과 | 우리 wall-clock (50-task) | 핵심 변경 |
|---|---|---|---|---|
| v2 | 5/5 | 0.3386 (leaderboard 최초) | — | 베이스라인 |
| v3 | 5/10 | (미공개) | — | 정규화·multi-pass·streaming-json |
| **v4** (=v8 image) | 5/12 | **0.2281 SIGTERM** | 4.4h | T-A~T-G chronic 패치 (회귀로 회귀) |
| **v6** | 5/18 | **0.3509 SIGTERM** | 1.86h | cascading governor + timeout cap 단축 + httpx 240s |
| **v7** (current, ship 직전) | 5/20 | 미평가 | **0.98h** | governor 80% margin + SC k auto-fallback + max_workers 6 + confirm:true 차단 |

각 SIGTERM 메일에서 본 패턴:
- v4: 32h naive → 12h × 0.38 처리 = 0.228 score 일치
- v6: 13h naive (단순 비례 1.85× slower 추정) → ~52% 처리 = 0.351 일치
- v7 (예상): 7.9h naive × 1.8 slower = ~14h. governor cascade 발동 시 ~10h → fit

운영진 환경 가설:
- **endpoint latency가 우리 DGX보다 ~1.8× slower** (가장 가능성 높음)
- 또는 hidden 분포가 hard tier 비율 ↑

---

## 4. A-board vs B-board 전략

### A-board (4/24 ~ 5/21, 57 task, 2시간 budget)

| 항목 | 값 |
|---|---|
| 제출 한도 | 30회/Phase 1 |
| 자동 채점 + 리더보드 피드백 | ✓ |
| 매 제출이 학습 신호 | 매 라운드 endpoint latency 측정 가능 |
| 핵심 위협 | SIGTERM (2시간 cap) |

**전략**:
- 매 제출 후 SUBMISSION_LOG에 wall-clock × 8 (linear naive) + 결과 score 기록
- score < 0.5면 wall-clock 봉쇄 부족 → 다음 라운드 timeout cap 추가 단축
- score 0.5~0.65면 알고리즘 fix 여유 있음 → over-emission/percentage 같은 specific 회복
- score 0.65+면 안정. 큰 변경 없이 minor refinement.

### B-board (5/21 ~ 5/23, 324 task, 12시간 budget)

| 항목 | 값 |
|---|---|
| 제출 한도 | **1회만** |
| 자동 채점 X — 이메일 제출 후 운영진 내부 평가 (5/22 ~ 5/26) |
| Phase 1 가중 평균에 큰 영향 | 324 task / 381 total = 85% |
| 핵심 위협 | 분포 다름 (medium 41.9%, hard 35.5%) |

**전략**:
- A-board 마지막 빌드 = B-board 제출본 (보통)
- 단, A-board의 hard 비율 52.6%와 B-board 35.5%가 다르므로 hard 회복이 압박을 좀 더 받음
- **B-board는 wall-clock 안전성 최우선**. SIGTERM 받으면 5/24~26 동안 회복 기회 없음.
- 12h budget이라 A-board (2h)보다 여유 있지만 task 수가 5.7배라 비슷한 압박.

---

## 5. 30회 Submission 예산 분배 (현재 상황 기준)

| 단계 | # | 결과 | 다음 라운드 트리거 |
|---|---|---|---|
| v2 ship | #1 | 0.3386 | 기준 |
| v3 ship | #2 | (미공개) | |
| v4 ship | #3 | **0.2281 SIGTERM** | governor 단발 결함 발견 |
| v6 ship | #4 | **0.3509 SIGTERM** | endpoint 1.8× slower 가설 |
| **v7 ship (예정)** | #5 | TBD | A-board 마지막 단계 |
| 잔여 (운영진 결과 따라) | 25회 | | hot-fix + B-board 준비 |

**남은 25회 분배 권장**:
- 운영진 결과 0.55+: 추가 fix 1-2회 + B-board 1회 = 2-3회 사용
- 운영진 결과 0.4~0.55: wall-clock 추가 단축 1-2회 + B-board 1회 = 2-3회 사용
- 운영진 결과 <0.4: 긴급 timeout 추가 단축 + max_workers 8 시도 = 3-4회 사용

**남은 예비**: 핫픽스 + 직전 footgun + 최종 정리. 17~20회는 의도적으로 잔여.

---

## 6. 의도적으로 안 하는 것 (현재 라운드)

| 안 하는 것 | 이유 |
|---|---|
| 2-stage agent (intent parser + ReAct) | 구조 변경 크고 회귀 위험 ↑. wall-clock 봉쇄 후에 검토 |
| Tree-search / MCTS | wall-clock 봉쇄와 정면 충돌. 7.9h naive 유지가 우선 |
| 도메인 knowledge static embedding | task별 hard-code 우려 + hidden set 일반화 불확실 |
| Multi-agent debate | 추가 LLM 호출 = 예산 소모. SC k=1로 cascade되면 voting 자체가 무용 |
| Adversarial schema probing | 룰 위반 가능성 + 30회 중 2-3회를 잡아먹음 |
| 모델 fine-tuning | 평가 시 호스팅 qwen만 허용 → fine-tune 무용 |
| Creative 트랙 출품 | Leaderboard 집중. README 보강 정도만 |

**원칙:** "할 수 있는 것"이 아니라 "12h 안에 끝나고 점수 깎이지 않는 것"으로 필터.

---

## 7. 자원 가정

- **인력:** 3인 × 10–15h/주 ≈ 총 108–162 person-hour ≈ **약 18 person-day** (27일 한정)
- **장비:** 회사 GPU 1장 → 로컬 vLLM/SGLang으로 `qwen3.5-35b-a3b` 셀프호스팅. **이게 우리의 가장 큰 무기**
- **트랙:** Leaderboard만
- **데이터:** 사용자가 `data/public/input/`에 직접 배치 (`.gitignore` 처리됨)

**역할 분담 제안:**
- **엔지니어 A — Runner & Submission**: Docker, env-config, normalization, mock-scorer, holdout
- **엔지니어 B — Tools & Kernel**: 신규 툴, persistent IPython kernel, knowledge.md injection
- **엔지니어 C — Prompts & Eval Loop**: 프롬프트, JSON-retry, plan-then-execute, self-consistency

---

## 8. 위험 등록부 (현재 라운드 기준)

| 위험 | 영향 | 완화 |
|---|---|---|
| **운영진 endpoint가 추정보다 더 느림 (2× 이상)** | A-board 2h cap 초과 + SIGTERM | hidden_set_probe로 첫 10 task latency 측정 → 다음 라운드 cap 자동 조정 |
| **hidden_set 크기가 추정보다 큼 (1000+)** | wall-clock 추가 부족 | governor cascade 3회까지 자동 → 마지막 task는 강한 단축 effect |
| max_workers=6에서 LLM 큐 timeout 재발 | task 다수 무효화 | OpenAI(timeout=200s) 명시 + retry 1→3. v6 → v7에서 검증됨 |
| `qwen3.5-35b-a3b` 정확한 가중치 차이 (운영진 vs 우리 DGX) | 프롬프트 튜닝 효과 차이 | SUBMISSION_LOG에 alias 기록. 우리 DGX는 Qwen3-30B-A3B-Instruct-2507 |
| Docker 빌드 10GB 초과 | 제출 불가 | `build_submission.sh`가 사이즈 fail. 현재 0.38 GB로 큰 여유 |
| 정규화가 정수 ID 컬럼을 소수화 | recall 손실 | `_answer`가 raw + normalized 둘 다 보관. mock_scorer가 per-task로 더 좋은 쪽 선택 |
| B-board 1회 제출이 SIGTERM | Phase 1 결정타. 회복 불가 | A-board에서 wall-clock 안정 확인된 build만 B-board 제출 |
| over-emission validator의 false positive | 정상 plural answer 잘못 차단 | bypass_count로 2번째 시도 commit 가능 (이미 구현됨) |
| hidden 분포가 hard/extreme 비율 ↑ | wall-clock 추가 압박 | cascade governor가 발동 → SC k=1 강제. 점수 약화 trade-off |

---

## 9. 한 줄 베팅

> 챔피언십을 가르는 단일 기술은 **"운영진 환경 12h 안에 안전하게 fit하는 wall-clock 봉쇄"**다. v4 (SIGTERM at 0.2281) → v6 (SIGTERM at 0.3509) → v7 (예상 0.55+) 진척이 직접 증거. 알고리즘 회복은 score 0.01~0.05 단위 leverage지만 wall-clock 봉쇄는 0.10~0.30 leverage. 정규화 (v3 시점 해결) + cascading governor + SC k auto-fallback + over-emission validator + max_workers 안정 — 다섯이 함께 작동할 때 챔피언십 점수에 가까워진다.

---

## 10. 검증 사이클 (E2E)

```bash
# 로컬 사이클 — 모든 PR 머지 전
uv run dabench run-benchmark --config configs/local.yaml \
  --task-set data/public/holdout_ids.txt
uv run python -m data_agent_baseline.scoring.mock_scorer \
  --predictions artifacts/runs/<run_id> \
  --gold data/public/output \
  --lambda-values 0.05 0.10 0.20 --by-difficulty

# Docker 사이클 — submission 직전
bash scripts/build_submission.sh v<N>
bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
```

**제출 사이클:** 위 둘 그린 → tar.gz → Drive → 이메일 → SUBMISSION_LOG.md 갱신.

---

## 11. 진입점 한 줄 매핑

| 다루는 것 | 위치 |
|---|---|
| 플랜 원본 (전체 디테일) | `~/.claude/plans/fuzzy-yawning-porcupine.md` |
| 시스템 가이드 (사람용) | `docs/ARCHITECTURE.md` |
| 운영 매뉴얼 (AI용) | `CLAUDE.md` |
| 제출 회고 로그 | `docs/SUBMISSION_LOG.md` |
| Endpoint 점검 결과 | `docs/qwen_endpoint_capabilities.md` |
