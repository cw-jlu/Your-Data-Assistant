---
name: kddcup-scoring
description: KDD Cup 2026 채점 함수, 정규화 규칙, 그리고 우리 로컬 미러 스코어러 사용법을 다룬다. "점수가 어떻게 계산돼", "λ 값", "정규화 어떻게 해", "mock_scorer 어떻게 써", "컬럼 시그니처 매칭", "왜 점수가 0인지", "ablation해야 하나" 같은 질문에서 트리거. 점수에 영향을 주는 코드 변경 전·후에 무조건 참조.
---

# KDD Cup 2026 — Scoring & Normalization

채점 공식, 컬럼-시그니처 매칭 메커니즘, 그리고 우리 로컬 mock_scorer / column-ablation / holdout 도구. 이 스킬을 모르고 점수 관련 코드를 만지면 안 된다.

---

## 1. 핵심 공식

```
Score = Recall − λ · (ExtraCols / PredictedCols)

Recall        = MatchedCols / GoldCols
ExtraCols     = max(PredictedCols − MatchedCols, 0)
PredictedCols = 우리 prediction.csv의 컬럼 수
GoldCols      = gold.csv의 컬럼 수
```

- **per-task 점수 범위:** `[0, 1]` (음수는 0으로 clip)
- **최종 순위:** 모든 태스크 평균 점수 — 동점 시 가장 빠른 제출 timestamp 우선
- **λ 값:** 비공개. 우리 mock_scorer 기본값 **0.10**, sensitivity sweep은 `{0.05, 0.10, 0.20}`

**시사점:** Recall 우대, Extra column은 약하게 페널티. **의심되면 컬럼을 포함**. 단 데이터로 정당화 못할 때는 빼라.

---

## 2. 컬럼 매칭 — "Column Signature"

채점기는 **컬럼 이름·순서를 무시한다**. 각 컬럼을 정규화된 값들의 multiset(frozenset of sorted values)으로 시그니처를 만들어, 시그니처가 동일한 컬럼끼리 greedy 1:1 bipartite로 매칭한다.

| 동작 | 채점 영향 |
|---|---|
| 컬럼 이름 변경 (`name` → `superhero_name`) | 영향 없음 |
| 행 순서 셔플 | 영향 없음 |
| 컬럼 순서 셔플 | 영향 없음 |
| 한 셀 정규화 누락 (예: `3.14159` 그대로) | 그 컬럼 시그니처 mismatch → recall 손실 |
| 추가 빈 컬럼 | extra penalty 작게 발생 |

**즉, 정규화가 곧 점수다.** 알고리즘 개선보다 정규화 누락이 더 크게 점수를 깎는다.

---

## 3. 정규화 규칙 (운영진 spec)

`scoring/normalize.py`의 `normalize_value` / `normalize_column` / `normalize_answer_table`이 이 표를 그대로 구현.

| 입력 타입 | 변환 |
|---|---|
| `null`, `None`, `nan`, `""` (대소문자 무관) | `""` |
| 숫자 (parseable) | `f"{x:.2f}"` (소수점 2자리, ROUND_HALF_UP). `4200000 → "4200000.00"` |
| 날짜 (parseable) | ISO-8601 `YYYY-MM-DD`. `"2024-3-1" → "2024-03-01"` |
| 날짜시간 | ISO-8601, UTC면 `Z` 접미사. timezone 없으면 그대로 |
| 그 외 문자열 | `str.strip()` (앞뒤 공백·`\r\n` 제거, **case-sensitive**) |
| 이름 | `"FirstName" + "LastName"` 와 `"FirstName LastName"` 모두 인정 |

**자주 깨지는 곳:**
- ID 같은 정수 컬럼이 `f"{x:.2f}"`로 강제 소수화돼서 mismatch — `_answer`는 raw + normalized 둘 다 trace에 남기고, mock_scorer가 per-task로 더 점수 높은 쪽을 선택
- 시간대 정보가 있는데 `Z` 접미사 누락
- "trim 후 case-sensitive" — `"Apple"`과 `"apple"`은 다른 값

---

## 4. 우리 로컬 도구

### 4.1 mock_scorer

`scoring/mock_scorer.py`. 공식 채점 공식을 로컬에서 그대로 미러.

```bash
# 단일 λ
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> \
    --gold data/public/output \
    --input data/public/input

# λ sensitivity sweep (column ablation 결정용)
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> \
    --gold data/public/output \
    --input data/public/input \
    --lambda-values 0.05 0.10 0.20
```

`DABENCH_LAMBDA` env로 기본 λ 오버라이드 가능. `--by-difficulty`로 난이도별 분해.

### 4.2 holdout split

`scoring/holdout.py`. `blake2b(salt + task_id)` 기반 **결정적** 80/20 분할:

```bash
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input \
    --output-dir data/public
# → train_ids.txt (40), holdout_ids.txt (10)
```

**철칙:** holdout 10개는 **프롬프트 튜닝에 절대 사용 금지** (제출 직전 검증 한 번만). 일상 이터레이션은 train 40개 또는 `data/public/smoke_ids.txt` (5개).

### 4.3 column-ablation (Phase 3)

`scoring/column_ablation.py` (계획상 신규). 답변 컬럼이 6개 이상이고 일부가 의심스러울 때 "drop k" 후보들에 대해 mock_scorer를 λ ∈ {0.05, 0.10, 0.20} 모두로 실행. **세 λ 모두에서 drop이 우월**할 때만 컬럼 제거. λ 미스에 강건한 구조.

---

## 5. 코드 안에서 점수 보존하는 곳

`tools/registry.py:_answer` (terminal 핸들러)

`_answer`는 **두 가지 답을 모두 emit**한다:
- `AnswerTable` — LLM이 만든 raw 출력
- `normalized_answer` — `normalize_answer_table`을 통과한 버전

`run/runner.py:_write_task_outputs(prefer_normalized=True)`가 normalized 버전을 `prediction.csv`로 기록. raw는 `trace.json.answer`로 보존되어 디버깅 / 사후분석에 쓰인다.

---

## 6. 점수 디버깅 플레이북

증상별 1차 가설:

| 증상 | 가장 흔한 원인 | 1차 점검 |
|---|---|---|
| 한 태스크 점수 = 0인데 답은 그럴듯 | 정규화 누락 (소수점 / 날짜) | `scoring/normalize.py` 단위 테스트로 그 셀만 통과시켜보기 |
| 모든 태스크 점수 0 | flat_output_dir 미적용 → `prediction.csv` 경로 mismatch | `kddcup-submission` 참조 |
| 점수가 holdout vs leaderboard에서 큰 갭 | 분포 차 또는 λ 추정 오류 | submission #1, #2, #3 갭 측정 |
| Phase 3 변경 후 점수 회귀 | column-ablation이 단일 λ로 결정 | sweep 모드로 재검증 |
| ID 컬럼이 mismatch | 정수형 ID가 `.00`으로 강제됨 | `_answer`의 raw vs normalized 비교 |

---

## 7. "한 줄 베팅"

> 챔피언십을 가르는 단일 기술은 **"공식 정규화에 맞춘 컬럼-시그니처 self-consistency 투표"**다. 채점식이 컬럼-시그니처를 본다는 사실 + qwen3.5-35b-a3b의 stochasticity를 자산으로 전환하는 기법이 정확히 일치한다. 단 정규화 레이어가 안 깔리면 투표는 노이즈만 증폭한다 — 그래서 순서가 중요. (`kddcup-strategy` Phase 0 → 4)
