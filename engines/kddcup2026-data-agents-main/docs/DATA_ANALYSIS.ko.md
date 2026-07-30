# DataAgent-Bench — 데이터 분석 보고서

> 🌐 **Language**: [English](DATA_ANALYSIS.md) · **한국어** · [中文](DATA_ANALYSIS.zh.md)

KDD Cup 2026 DataAgent-Bench 챌린지의 공개 50개 태스크에 대한 정량 분석과, v2 시점(2026-04-29)까지의 holdout 실측 결과를 한 문서로 정리한 자료다. 다음 라운드(v3+)에서 어디를 건드려야 점수가 오를지의 근거.

> 이 문서가 다루는 것: **데이터셋 자체 + 우리 에이전트가 그 위에서 어떻게 동작했는가**.
> 시스템 동작 원리는 `[ARCHITECTURE.ko.md](ARCHITECTURE.ko.md)`, 제출 이력·점수 추이는 `[SUBMISSION_LOG.ko.md](SUBMISSION_LOG.ko.md)`, 운영 매뉴얼은 `[../CLAUDE.ko.md](../CLAUDE.ko.md)`.

---

## 1. 한눈에 보기


| 항목         | 값                                                                   |
| ---------- | ------------------------------------------------------------------- |
| 공개 태스크 수   | 50                                                                  |
| 분할         | train 40 / holdout 10 / smoke 5 (`scoring/holdout.py`로 결정적 분할)      |
| 난이도 분포     | easy 15 / medium 23 / hard 11 / extreme 1                           |
| 컨텍스트 파일 종류 | `.csv`, `.json`, `.db` (SQLite), `.md` 만 — PDF/Excel/Parquet/이미지 없음 |
| 항상 존재하는 자료 | `context/knowledge.md` (50/50 — 도메인 사전·테이블 설명)                      |
| 최대 컨텍스트 크기 | task_257 = **441 MB** (medium)                                      |
| 답안 일반 형태   | 컬럼 1~3개, 행 중앙값 1, 최대 140                                            |


핵심 시사점:

- 모든 공개 태스크에 `knowledge.md`가 있다 — 이걸 자동 인젝션하지 않으면 도메인 컨텍스트 손실(이미 v2에서 인젝션 적용).
- 정답이 1×1 셀(룩업형)이 다수 — 정규화 한 번 어긋나면 그 태스크는 0점.
- Phase 1 데이터엔 PDF/Excel/이미지가 없지만 운영진 hidden set은 multi-modal 명시 → 리더는 v3 이후 추가 후보.

---

## 2. 난이도 × 컨텍스트 형태 (50 태스크)

### 2.1 난이도 분포


| Difficulty | n   | 비중  | 룰상 정의                            |
| ---------- | --- | --- | -------------------------------- |
| easy       | 15  | 30% | 정형 파일 + 지식 문서                    |
| medium     | 23  | 46% | 정형 + DB + 문서                     |
| hard       | 11  | 22% | 멀티 소스 + 비정형(≈10K~128K tokens)    |
| extreme    | 1   | 2%  | ultra-long inputs (>128K tokens) |


extreme 1개뿐이지만 hidden set에서 비중이 늘 가능성 — runner의 `max_steps_by_difficulty` 매핑(extreme=28)은 그 시나리오 대비.

### 2.2 컨텍스트 서브트리 등장 빈도 (50중)


| 서브트리                 | 빈도     | 용도                     |
| -------------------- | ------ | ---------------------- |
| `csv/`               | 36     | tabular                |
| `json/`              | 30     | 구조화 데이터 (자주 nested 객체) |
| `db/` (SQLite `.db`) | 27     | 정규화 테이블 — JOIN/집계용     |
| `doc/`               | 12     | 보조 문서 (`.md`)          |
| `**knowledge.md`**   | **50** | 도메인 사전 — 100% 존재       |


조합 패턴(50중 빈도 상위):

- CSV+JSON (12) — 외래키로 lookup
- CSV+DB (10) — DB 스키마 → CSV로 추가 fact
- JSON+DB (6) — DB로 정규화, JSON으로 nested
- CSV+JSON+DB (4) — 세 소스 조합 (medium+ 위주)

### 2.3 파일 확장자 합계


| 확장자     | 파일 수  | 비고                         |
| ------- | ----- | -------------------------- |
| `.md`   | 64    | knowledge.md 50 + doc/* 14 |
| `.csv`  | 40    |                            |
| `.json` | 37    |                            |
| `.db`   | 27    | SQLite                     |
| 그 외     | **0** | PDF/Excel/Parquet/이미지 없음   |


---

## 3. 컨텍스트 크기 분포 — 메모리 부하 지표


| Difficulty | n   | min    | median | max        |
| ---------- | --- | ------ | ------ | ---------- |
| easy       | 15  | 17 KB  | 287 KB | **58 MB**  |
| medium     | 23  | 38 KB  | 1.4 MB | **441 MB** |
| hard       | 11  | 41 KB  | 256 KB | **267 MB** |
| extreme    | 1   | 376 KB | 376 KB | 376 KB     |


**대형 컨텍스트 (≥50 MB) 8개:**


| task_id  | difficulty | 크기     |
| -------- | ---------- | ------ |
| task_257 | medium     | 441 MB |
| task_250 | medium     | 384 MB |
| task_330 | hard       | 267 MB |
| task_259 | medium     | 182 MB |
| task_249 | medium     | 166 MB |
| task_243 | medium     | 137 MB |
| task_420 | hard       | 59 MB  |
| task_38  | easy       | 58 MB  |


**시사점:**

- 32K 컨텍스트 윈도우(우리 vLLM)에 원본 적재 불가능. → `dataframe_describe`/`dataframe_head` 압축 prepass 필수 (v2에 도입).
- 매 호출 reload 시 30s 한도 IO에 잡힘. → persistent IPython kernel(v2 도입)이 task_249/task_250 같은 대형 케이스에서 직접 ROI.

---

## 4. 정답(`gold.csv`) 형태 분포


| 형태    | 빈도            |
| ----- | ------------- |
| 컬럼 1개 | 40 / 50 (80%) |
| 컬럼 2개 | 7 / 50        |
| 컬럼 3개 | 3 / 50        |


행 수: min 1, **median 1**, p90 7, max 140.

**시사점:**

- 답이 단일 값(1×1)인 케이스가 가장 흔함 → easy의 lookup-style 질문이 다수.
- column-ablation(Phase 3 계획)은 컬럼 6+개에서만 발동하므로 공개셋에선 거의 안 굴러감 → hidden set의 wide-table 케이스에 대비해 코드만 깔아둔다.
- 행 140짜리 long-list (task_180)도 존재 — 답안이 long list가 되는 케이스에 대해 정규화 cost가 비례 증가.

---

## 5. 질문(natural language) 길이


| 통계     | 값    |
| ------ | ---- |
| min    | 30자  |
| median | 90자  |
| max    | 144자 |
| mean   | 88자  |


질문이 짧다 — 모델이 의도를 추론해야 하는 폭이 넓음. 시스템 프롬프트가 답변 형식(정규화, 컬럼 시그니처)을 명시해야 점수 손실 최소화 (v2 도입).

---

## 6. v2 holdout 실측 (2026-04-29)

### 6.1 점수 요약 (`λ ∈ {0.05, 0.10, 0.20}`)


| λ        | mean Score | mean Recall |
| -------- | ---------- | ----------- |
| 0.05     | 0.6300     | 0.6333      |
| **0.10** | **0.6267** | **0.6333**  |
| 0.20     | 0.6200     | 0.6333      |


세 λ에서 Score가 거의 평행 이동 — extra-column 페널티의 영향이 작다는 뜻 (우리 답이 컬럼 over-emission을 잘 안 한다).

### 6.2 난이도별 (Container, λ=0.10)


| difficulty | n   | score      | recall |
| ---------- | --- | ---------- | ------ |
| easy       | 3   | 0.6667     | 0.6667 |
| medium     | 5   | **0.8000** | 0.8000 |
| hard       | 2   | **0.1333** | 0.1667 |
| extreme    | 0   | —          | —      |


**해석:**

- **medium 0.80** — knowledge.md 인젝션 + dataframe prepass + persistent kernel이 가장 효과 큰 구간. 이 카테고리는 v3에서 추가 ROI 추출 어렵다(이미 plateau 가까움).
- **easy 0.67** — 1×1 lookup인데도 1/3이 0점. 정규화 또는 over-emission 의심.
- **hard 0.13** — 가장 큰 leak. 다음 라운드 핵심 타겟.

### 6.3 holdout 10 태스크 per-row 측정

각 태스크의 컨텍스트 모양 + 답 매칭 패턴:


| task_id  | diff   | ctx 서브트리    | size       | steps | gold (c×r) | pred (c×r) | shape match | 종합                         |
| -------- | ------ | ----------- | ---------- | ----- | ---------- | ---------- | ----------- | -------------------------- |
| task_11  | easy   | json        | 0.5 MB     | 5     | 3 × 3      | 3 × 6      | **rows ↑**  | 컬럼 시그니처 깨짐 (over-emission) |
| task_27  | easy   | json        | 24 KB      | 5     | 3 × 1      | 3 × 1      | OK          | 매칭 추정                      |
| task_75  | easy   | csv+json    | 0.5 MB     | 8     | 1 × 1      | 1 × 1      | OK          | 매칭 추정                      |
| task_169 | medium | csv+db      | 9 MB       | 7     | 1 × 1      | 1 × 1      | OK          | 매칭                         |
| task_249 | medium | db+json     | 174 MB     | 7     | 2 × 1      | 2 × 1      | OK          | 매칭 (대형 컨텍스트 OK)            |
| task_250 | medium | csv+db+json | **403 MB** | 6     | 1 × 1      | 1 × 1      | OK          | 매칭 (kernel/dataframe 효과)   |
| task_261 | medium | csv+db+json | 0.13 MB    | 6     | 1 × 1      | 1 × 1      | OK          | 매칭                         |
| task_303 | medium | db+json     | 0.24 MB    | 9     | 1 × 1      | 1 × 1      | OK          | 매칭                         |
| task_355 | hard   | csv+doc     | 41 KB      | 6     | **3 × 1**  | 2 × 1      | **cols ↓**  | 컬럼 누락 (under-emission)     |
| task_408 | hard   | db+doc      | 1.1 MB     | 8     | 1 × 1      | 1 × 1      | OK          | 값 mismatch (정규화/잘못된 답) 추정  |


**모든 10 task가 `_answer`까지 도달**(succeeded=True) — 즉 0/10 fail에서 10/10 답 제출로 회복. 이제 점수 leak은 "도달은 했지만 답이 틀림"에서 발생.

---

## 7. 실패 패턴 분류

`v2` holdout에서 점수 0(또는 부분 점수)인 태스크를 살펴보면 세 가지 모드가 분리된다.

### 7.1 Over-emission (행 ↑) — task_11

`gold` 3×3, 우리 `pred` 3×6. 같은 3컬럼인데 행 6개. 채점기는 정규화 후 sorted-value multiset으로 시그니처 만들므로 추가 행이 들어가면 multiset이 mismatch → 컬럼 매칭 0. 즉 **3개 짜리 정답에 추가 데이터를 6개 합쳐서 제출**한 패턴. easy 카테고리에서 이 한 태스크가 평균을 깎음.

**대응 후보:**

- 시스템 프롬프트에 "답에는 질문이 요구하는 정확한 행 수만 포함, 추가 데이터 X" 강조 추가.
- `answer_validator`(Phase 3 계획)에서 `gold_rows` 모를 때라도 "쿼리 결과를 그대로 반환할 때 dedup·필터 누락" 패턴 경고.

### 7.2 Under-emission (컬럼 ↓) — task_355

`gold` 3×1 vs `pred` 2×1. 컬럼 한 개를 빠뜨림. 채점기에서 매칭 가능한 컬럼 수가 ≤2 → recall ≤2/3, λ-페널티는 0. hard 카테고리의 한 태스크가 0~0.33 사이에서 끝남.

**대응 후보:**

- 시스템 프롬프트에 "When in doubt about including a column, INCLUDE" 정책은 이미 있음(v2). 다만 hard 태스크에선 모델이 멀티-소스 결합에서 1개 컬럼을 잊어버림.
- `answer_validator`에서 question 안의 명사구(예: "list their A, B, and C") 수와 답 컬럼 수가 일치하는지 휴리스틱 체크.

### 7.3 값 mismatch (정규화·계산) — task_408

`gold` 1×1, `pred` 1×1, 컬럼 모양 일치인데 점수는 0(또는 부분)으로 추정. 이게 가장 까다로운 모드: shape는 맞췄는데 값 자체가 다름. 원인 가능성:

- 잘못된 SQL/Python 계산 결과
- 정규화 — 숫자 ROUND_HALF_UP vs HALF_EVEN, 날짜 포맷, trimming 누락
- doc/ 안의 비정형 단서를 모델이 잘못 해석

**대응 후보:**

- trace.json 직접 분석으로 어떤 step에서 잘못 추론했는지 보기 — task_408 trace는 8 step이고 marathon-style 추론. 잘못된 도메인 가정이 있을 가능성.
- `dataframe_describe` 결과에 numeric 컬럼의 ROUND_HALF_UP 정규화 미리보기 추가.

---

## 8. 강점·약점 한 페이지 요약

### 강점 (v2 시점)

1. **Medium 0.80** — knowledge.md 자동 인젝션 + persistent kernel + dataframe_describe가 시너지 확보.
2. **대형 컨텍스트 (≥100 MB) 다 통과** — task_249 (174 MB), task_250 (403 MB) 모두 정답 매칭. persistent kernel 도입의 정당성 입증.
3. **Easy + medium의 1×1 lookup** — JSON-mode probe + plan-then-execute 프롬프트가 단일 값 추출에서 안정.
4. **모든 task가 `_answer` 호출** — 10/10 도달. `_coerce_action_input` 핫픽스 이후 step 낭비 없음.

### 약점 (v3+ 타겟)


| 우선순위    | 영역                    | 증거                                  | 후보                                       |
| ------- | --------------------- | ----------------------------------- | ---------------------------------------- |
| 🔴 High | hard 0.13             | task_355 컬럼 누락, task_408 값 mismatch | answer_validator + question parsing 휴리스틱 |
| 🟠 Mid  | easy 1/3 점수 0         | task_11 over-emission               | "exact row count" 프롬프트 강조                |
| 🟡 Low  | hidden set의 PDF/Excel | 공개셋엔 0건이지만 룰에 multi-modal 명시        | Phase 1.5 계획대로 reader 추가                 |
| 🟡 Low  | extreme(>128K)        | 공개셋엔 1개, 376 KB로 작음                 | YaRN/RoPE 확장 재검토는 우리 vLLM 책임             |


---

## 9. 다음 라운드(v3) 작업 후보 — ROI 우선

1. **answer_validator + conditional terminal** (Phase 3) — `_answer` 직전에 (a) numeric 컬럼 dtype 일관성, (b) question 명사구 수 vs 답 컬럼 수 비교, (c) 의심스러운 셀 warning. warning 있으면 observation으로 반환 → agent가 수정 재제출.
2. **hard tier trace 디버깅** — task_355, task_408 외에 train 40에서 hard 11개 trace 분석. 공통 실패 모드 파악 후 시스템 프롬프트에 1~2줄 추가.
3. **column-signature self-consistency** (Phase 4 게이트) — k=3, temperature 0→0.5, multiset 투표. hard 태스크의 stochastic 오답을 다수결로 흡수. 12h 예산 시뮬 통과 시에만.
4. **extra-row 가드** — `_coerce_action_input` 옆에 "rows 비정상 길이 경고" 휴리스틱 (gold 모르는 상태에선 task별 보수적 cap).

각 변경 후 holdout 10개로 회귀 측정 → mean Score ≥ v2 + 3점일 때만 v3 제출.

---

## 10. 데이터 갱신 절차

이 문서의 §2~~§5는 **공개 50 태스크 정적 분석**이므로 데이터셋이 바뀌지 않는 한 stable. §6~~§8은 **매 제출마다 갱신** 필요:

```bash
# 1. 새 run 실행
uv run dabench run-benchmark --config configs/local.yaml \
  --task-set data/public/holdout_ids.txt

# 2. mock_scorer
uv run python -m data_agent_baseline.scoring.mock_scorer \
  --predictions artifacts/runs/<run_id> \
  --gold data/public/output --input data/public/input \
  --lambda-values 0.05 0.10 0.20

# 3. per-task shape 추출 (이 문서 §6.3 표 갱신용)
.venv/bin/python -c "
import json, pathlib
run_dir = pathlib.Path('artifacts/runs/<run_id>')
for trace in sorted(run_dir.glob('task_*/trace.json')):
    d = json.loads(trace.read_text())
    print(d['task_id'], d.get('succeeded'), len(d.get('steps', [])))
"
```

레포지토리 `.gitignore`는 이 문서를 화이트리스트(`!docs/DATA_ANALYSIS.md`)로 처리해 git tracking — 팀 공유용.