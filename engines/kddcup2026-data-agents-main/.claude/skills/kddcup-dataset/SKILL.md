---
name: kddcup-dataset
description: 태스크 디렉토리 구조, task.json 스키마, context/ 트리, 공개 50개 태스크 실측 통계(난이도·크기·확장자)를 참조한다. "태스크 어떻게 생겼어", "context 안에 뭐 있어", "어떤 파일 포맷 지원해야 해", "Phase 1 데이터 얼마나 커", "knowledge.md가 뭐야", "extreme 태스크가 몇 개야" 같은 질문에서 트리거.
---

# KDD Cup 2026 — Dataset & Task Format

데이터셋 레이아웃, 입력/출력 스키마, 그리고 공개 50개 태스크에 대한 실측 통계. 새 툴을 추가하거나 컨텍스트 처리 로직을 짤 때 먼저 읽는다. 채점 규칙은 `kddcup-scoring`, 점수에 영향 없는 시스템적 wiring은 `kddcup-submission` 참조.

---

## 1. 디렉토리 레이아웃

```
data/public/input/task_<id>/                # 우리 로컬, 평가 시에는 /input/task_<id>/
├── task.json                               # 메타 (질문, 난이도)
└── context/                                # 자료실 (모든 툴이 여기로만 접근)
    ├── csv/                                # 0..N개 .csv
    │   └── *.csv
    ├── db/                                 # 0..N개 SQLite
    │   └── *.db
    ├── json/                               # 0..N개 .json
    │   └── *.json
    ├── doc/                                # 보조 문서 — 현재 .md만
    │   └── *.md
    └── knowledge.md                        # 도메인 지식 — 거의 항상 존재

data/public/output/task_<id>/gold.csv       # 공개 데모만 (히든은 없음)
```

**평가 시 마운트:** `/input` (RO), `/output` (RW), `/logs` (RW). 자세한 wiring은 `kddcup-submission`.

**규칙:** 모든 툴 입력 경로는 `context/`를 루트로 한 **상대경로**. 절대경로 / `..` escape는 `tools/filesystem.resolve_context_path`가 거부.

---

## 2. `task.json` 스키마 (strict)

키 집합이 **정확히 `{task_id, difficulty, question}`**. 추가 키가 들어오면 `DABenchPublicDataset`(`benchmark/dataset.py`)이 raise.

```json
{
  "task_id": "task_269",
  "difficulty": "medium",
  "question": "What are the names of the superheroes with the power of death touch?"
}
```

- `difficulty ∈ {"easy", "medium", "hard", "extreme"}`
- `question`: 영어 자연어. 공개셋 평균 90자, 범위 30~144자

**의도된 동작:** strict 검증. 하지만 hidden 셋이 미묘하게 다를 가능성이 있으므로 `kddcup-strategy` 위험 등록부에서 `lenient 플래그` 옵션을 명시.

---

## 3. `context/` 안에 들어 있는 것 (공개 50개 실측)

| 서브트리 | 존재 빈도 (50개 중) | 비고 |
|---|---|---|
| `csv/` | 36 | tabular |
| `db/` (SQLite) | 27 | `.db` 파일, `tools/sqlite.py`로 read-only SQL |
| `json/` | 30 | structured |
| `doc/` | 12 | 현재는 `.md`만 |
| **`knowledge.md`** | **50 / 50** | **항상 존재** — 도메인 지식 |

**시사점:**
- `knowledge.md`는 100% 존재 → 시스템 프롬프트에 자동 인젝션해도 손실 없음 (Phase 1)
- `csv/`+`db/` 동시 등장 흔함 → SQL과 pandas 둘 다 필요
- `doc/`는 24%에서만 — 그러나 hidden 셋에서 PDF/DOCX로 확장될 가능성 (rules에 multi-modal 명시)

---

## 4. 파일 확장자 (공개 50개 합계)

| 확장자 | 파일 수 |
|---|---|
| `.md` | 64 (knowledge.md 50 + doc/*.md 14) |
| `.csv` | 40 |
| `.json` | 37 |
| `.db` | 27 |
| 그 외 (.pdf, .xlsx, .parquet, .png 등) | **0** |

**Phase 1 dataset 한정**으로는 PDF/Excel/Parquet/PNG 리더가 불필요. 그러나 운영진 rules가 multi-modal을 명시했고 Phase 2는 확실히 등장할 것이므로 plan은 reader 추가를 포함한다 (Phase 1 단계).

---

## 5. 난이도 분포 (공개 50개)

| Difficulty | n | 비중 |
|---|---|---|
| easy | 15 | 30% |
| medium | 23 | 46% |
| hard | 11 | 22% |
| extreme | 1 | 2% |

extreme이 1개뿐이지만 hidden 셋에서 비중이 늘 수 있다. 운영진 spec상 extreme = "ultra-long inputs (>128K tokens)".

---

## 6. 컨텍스트 크기 분포 (난이도별, bytes)

| Difficulty | n | min | median | max |
|---|---|---|---|---|
| easy | 15 | 17 KB | 287 KB | **58 MB** |
| medium | 23 | 38 KB | 1.4 MB | **441 MB** |
| hard | 11 | 41 KB | 256 KB | **267 MB** |
| extreme | 1 | 376 KB | 376 KB | 376 KB |

**대형 컨텍스트 (≥50MB) 8개 — 인덱스로 메모.**

| task_id | difficulty | 크기 |
|---|---|---|
| task_257 | medium | 441 MB |
| task_250 | medium | 384 MB |
| task_330 | hard | 267 MB |
| task_259 | medium | 182 MB |
| task_249 | medium | 166 MB |
| task_243 | medium | 137 MB |
| task_420 | hard | 59 MB |
| task_38 | easy | 58 MB |

**시사점:**
- 32K 컨텍스트 윈도우(우리 vLLM)에 원본 적재 불가 → `dataframe_describe`/`dataframe_head` 같은 압축 prepass 필수
- 0.5GB 짜리 CSV를 매 호출마다 다시 읽으면 30s 한도가 IO에 잡아먹힘 → persistent IPython kernel (Phase 1)이 ROI 큼

---

## 7. Gold answer 형태 (공개 50개)

| 형태 | 빈도 |
|---|---|
| 컬럼 1개 | 40 / 50 (80%) |
| 컬럼 2개 | 7 / 50 |
| 컬럼 3개 | 3 / 50 |
| 행 수 min=1, **median=1**, p90=7, max=140 | |

**시사점:**
- 답이 단일 값(1행 1컬럼)인 케이스가 가장 흔함 — easy 태스크의 lookup-style
- 다중 행은 exception, 그러나 task_180(9행), 140행 같은 long-list도 존재
- column-ablation은 컬럼 6개 이상일 때만 발동하므로 공개셋에서는 거의 안 굴러감 → hidden 셋의 wide-table 케이스에 대비해 코드만 깔아둠

---

## 8. 출력 — `prediction.csv`

표 1장. 첫 행은 헤더(채점 시 무시), 이후 행은 값. UTF-8.

```csv
superhero_name
Black Flash
Blackwulf
Hela
```

**작성 시점:**
1. 에이전트가 `_answer(columns=..., rows=...)` 호출
2. `_answer`가 `normalize_answer_table()` 통과 — raw + normalized 둘 다 trace에 보존
3. `runner._write_task_outputs(prefer_normalized=True)`가 normalized 버전을 `prediction.csv`로 기록

채점 시 컬럼명/순서 무시 — 자세한 메커니즘은 `kddcup-scoring`.

---

## 9. 출력 — `trace.json` (디버깅용)

```json
{
  "task_id": "task_269",
  "succeeded": true,
  "answer": {"columns": [...], "rows": [...]},
  "normalized_answer": {"columns": [...], "rows": [...]},
  "failure_reason": null,
  "e2e_elapsed_seconds": 28.4,
  "steps": [
    {
      "step_index": 1,
      "thought": "...",
      "action": "list_context",
      "action_input": {"max_depth": 4},
      "raw_response": "```json\n{...}\n```",
      "observation": {"ok": true, "tool": "list_context", "content": {...}}
    }
  ]
}
```

`__error__` step은 LLM 출력 파싱 실패 시 기록 — 프롬프트 디버깅에 필수.

---

## 10. 출력 — `summary.json` (벤치마크 단위)

```json
{
  "run_id": "20260427T103905Z",
  "task_count": 5,
  "succeeded_task_count": 5,
  "max_workers": 4,
  "flat_output_dir": false,
  "tasks": [{"task_id": "task_74", "succeeded": true, ...}]
}
```

`run-benchmark` 명령에서만 작성. `run-task` 단일은 만들지 않음.

---

## 11. 검사·탐색 명령

```bash
# 데이터셋 가시성 + task 수 확인
uv run dabench status --config configs/local.yaml

# 단일 태스크 메타 + context 트리 보기
uv run dabench inspect-task task_269 --config configs/local.yaml

# Holdout 재생성 (한 번만)
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public
```

데이터셋이 비어 있으면 `data/public/input/`에 직접 배치 후 `status`. 가시성 확인 후 `holdout.py`.

---

## 12. 코드 진입점 한 줄 매핑

| 다루는 것 | 파일 |
|---|---|
| 디렉토리 발견 + 키 검증 | `src/data_agent_baseline/benchmark/dataset.py` |
| `PublicTask` 데이터클래스 | `src/data_agent_baseline/benchmark/schema.py` |
| Context 경로 sandbox | `src/data_agent_baseline/tools/filesystem.py:resolve_context_path` |
| Holdout 분할 | `src/data_agent_baseline/scoring/holdout.py` |
| 정규화 (출력 시) | `src/data_agent_baseline/scoring/normalize.py` |

---

## 13. Hidden 셋이 다를 수 있는 항목 (방어적 설계)

- 파일 확장자 (PDF/XLSX/PNG/PARQUET 등장 가능)
- `task.json`에 추가 키 (strict → lenient 플래그)
- knowledge.md 위치/이름 (절대 가정 금지 — 동적 감지)
- 난이도 분포 (extreme이 더 많을 수 있음)
- 답 컬럼 수 (wide-table 케이스 대비)
