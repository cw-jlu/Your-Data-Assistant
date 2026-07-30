---
name: kddcup-rules-output
description: prediction.csv 출력 형식과 정규화 spec을 룰 페이지 verbatim으로 다룬다 — UTF-8 CSV, 헤더, 컬럼 순서 무관, null/numeric/date/datetime/string 정규화. "prediction.csv 어떻게 써", "헤더 필요해?", "어디에 저장해", "정규화 규칙", "소수점 몇 자리", "날짜 포맷", "case 무시야 case-sensitive야" 같은 출력 컴플라이언스 질문에서 트리거. 채점 메커니즘은 kddcup-scoring.
---

# Rules — Output Format & Normalization

원천: https://dataagent.top/rules. `prediction.csv`의 정확한 형식, 그리고 운영진이 채점 직전에 적용하는 정규화. 우리 답이 의미적으로 맞아도 정규화에서 한 셀 어긋나면 그 컬럼 시그니처가 깨져 점수 0.

채점 공식·매칭 메커니즘은 `kddcup-scoring`. 이 스킬은 **출력 형식 룰 그 자체**.

---

## 1. 출력 위치 (룰 명시)

```
/output/task_<id>/prediction.csv
```

- **task별로 한 디렉토리** + **그 안에 `prediction.csv` 한 파일**
- 다른 위치(예: `/output/prediction_<id>.csv`)에 쓰면 채점 안 됨 → 그 태스크 0점
- `flat_output_dir: true` (`configs/eval.yaml`)이 이 경로 보장. wiring 자세한 내용은 `kddcup-rules-runtime` + `kddcup-submission`

**미작성 허용:** 우리가 답을 못 내면 prediction.csv를 안 쓸 수도 있다 (그 태스크는 0점). 룰 위반은 아님.

---

## 2. CSV 형식 (룰 명시)

| 항목 | 값 |
|---|---|
| 인코딩 | **UTF-8** |
| 형식 | 표준 CSV |
| 헤더 행 | **있음** (단, 컬럼명은 채점에 영향 없음) |
| 데이터 행 | 1행 이상 (또는 0행도 가능 — 빈 결과) |

**예시 (룰 페이지 형태):**

```csv
superhero_name
Black Flash
Blackwulf
Hela
```

**컬럼명 의의:**
- 채점기는 **컬럼명을 무시**하고 정렬된 값 시그니처만 본다
- 그러나 헤더는 **반드시 있어야 한다** (없으면 첫 데이터 행이 헤더로 오해됨)
- 의미 있는 헤더를 쓰는 게 trace.json 디버깅에 도움

---

## 3. 컬럼 순서 — 무관 (Unordered)

룰 verbatim:

> "Column order irrelevant; scoring matches unordered column value vectors"

| 동작 | 채점 영향 |
|---|---|
| 컬럼 순서 셔플 | 영향 없음 |
| 행 순서 셔플 | 영향 없음 (정규화 후 정렬) |
| 컬럼 이름 변경 | 영향 없음 |

**시사점:** 우리는 답을 정렬할 필요 없다. ORDER BY 누락이 점수에 영향 X.

---

## 4. 정규화 규칙 (룰 verbatim) — Pre-Scoring

운영진이 **채점 직전에** prediction.csv와 gold.csv 양쪽에 동일하게 적용. 즉 우리가 정규화를 안 해도 채점은 같지만, 우리 컬럼 시그니처가 맞춰지도록 **선행 정규화**가 안전하다.

| 타입 | 룰 |
|---|---|
| **Null** | 빈 문자열, `"null"`, `"none"`, `"nan"` (case-insensitive) → `""` |
| **Numeric** | 소수점 2자리 (ROUND_HALF_UP). 예: `4200000` → `"4200000.00"` |
| **Date** | ISO 8601 (`YYYY-MM-DD`). 예: `"2024-3-1"` → `"2024-03-01"` |
| **DateTime** | UTC면 `Z` 접미사. timezone 없으면 ISO 형식 그대로 |
| **String** | 앞뒤 공백·`\r\n` 제거, **case-sensitive 비교** |
| **Names** | `"FirstName" + "LastName"`와 `"FirstName LastName"` **둘 다 인정** |

`scoring/normalize.py`가 이 표를 그대로 구현. 자세한 함수는 `kddcup-scoring` §3.

---

## 5. Null 정규화 — 흔한 실수

룰: `null`, `none`, `nan`, `""` (대소문자 무관) → `""`

**흔한 비호환 입력:**

| 입력 | 정규화 후 |
|---|---|
| `None` (Python) | `""` |
| `NaN` (numpy/pandas) | `""` |
| `"NULL"` | `""` |
| `"NaN"` | `""` |
| `"none"` | `""` |
| 빈 문자열 `""` | `""` |
| **`"NA"`** | **그대로 `"NA"`** (룰에 명시 안 됨!) |
| **`"-"` 또는 `"--"`** | **그대로** (자주 데이터에서 null 표현되지만 정규화 안 됨) |

**시사점:** `pandas.read_csv`의 기본 `na_values`에 `"NA"`, `"-"`가 포함되지만, 룰 정규화는 명시된 4개만 처리. 우리가 답을 만들 때 `pd.to_csv()` 그대로 쓰지 말고 — **null 표현은 빈 문자열로 통일**.

---

## 6. Numeric 정규화 — ROUND_HALF_UP

룰 명시 예시: `4200000` → `"4200000.00"`

**핵심:**
- `f"{x:.2f}"`이 Python 기본인데 ROUND_HALF_EVEN (banker's rounding) — `2.5` → `"2.50"`이지만 `0.125` → `"0.12"` (HALF_EVEN), 룰은 HALF_UP이므로 `"0.13"` 기대
- `decimal.Decimal` + `quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)` 권장

**예시:**

| 입력 | HALF_EVEN (Python `:.2f`) | **HALF_UP (룰)** |
|---|---|---|
| `0.125` | `"0.12"` | **`"0.13"`** |
| `0.135` | `"0.14"` | `"0.14"` |
| `2.5` | `"2.50"` | `"2.50"` |

**우리 `normalize.py`는 HALF_UP을 명시적으로 구현 필요.**

**정수 vs 소수:** 룰은 모든 numeric을 `.00`으로 변환. ID 같은 정수도 강제 변환 → mismatch 위험. 대응: `_answer`에서 raw + normalized 둘 다 emit, mock_scorer가 per-task로 더 점수 높은 쪽 선택.

---

## 7. Date 정규화 — ISO 8601

룰: `"2024-3-1"` → `"2024-03-01"`

**핵심:**
- ISO 8601 = `YYYY-MM-DD`
- 단일 자릿수 month/day는 0 패딩
- 구분자는 `-` (슬래시 `/`나 점 `.` 아님)

**파싱 흐름 (우리 `normalize_value`):**
1. `dateutil.parser.parse(s)`로 시도
2. 성공하면 `dt.strftime("%Y-%m-%d")`
3. 실패하면 string 처리로 fall-through

**경계 케이스:**
- `"March 1, 2024"` — dateutil이 `2024-03-01`로 파싱 가능
- `"01/03/2024"` — 모호 (DD/MM vs MM/DD). dateutil 기본은 MM/DD (US). 운영진 spec 미상 → 가능하면 source 데이터의 컨벤션 보존
- `"2024-03"` — 일 누락. dateutil은 day=1 가정 → `2024-03-01`. 의도치 않은 매칭 가능

---

## 8. DateTime 정규화

룰 verbatim:

> "DateTime — UTC timezone (Z suffix); no timezone: preserve ISO format"

| 입력 | 정규화 후 |
|---|---|
| `"2024-01-15T10:30:00Z"` | `"2024-01-15T10:30:00Z"` (그대로) |
| `"2024-01-15T10:30:00+09:00"` | UTC로 변환 후 `Z` (e.g. `"2024-01-15T01:30:00Z"`) |
| `"2024-01-15T10:30:00"` (timezone 없음) | `"2024-01-15T10:30:00"` (보존) |
| `"2024-01-15 10:30:00"` (공백 구분자) | ISO format으로 변환 — 정확한 변환 룰은 모호 |

**시사점:** datetime 컬럼은 source의 timezone 정보를 보존하거나 명시적으로 UTC 변환. **혼합하면 시그니처 mismatch.**

---

## 9. String 정규화

룰 verbatim:

> "Strings — strip leading/trailing whitespace & \r\n; case-sensitive comparison"

| 동작 | 영향 |
|---|---|
| 앞뒤 공백 제거 (`str.strip()`) | OK |
| `\r\n` 제거 | OK |
| case 보존 | `"Apple"`과 `"apple"`은 다른 값 |
| **내부** 공백 압축 | **안 함** — `"Hello  World"`와 `"Hello World"`는 다른 값 |

**시사점:**
- 답을 만들 때 `s.strip()` 적용
- LLM이 모르고 `s.lower()` 시키면 시그니처 mismatch
- pandas `read_csv`의 `keep_default_na=True`로 빈 문자열이 NaN으로 변환되는 함정 — `keep_default_na=False, na_values=[""]` 권장

---

## 10. Names — 두 형태 모두 인정

룰 verbatim:

> "Names — 'FirstName' + 'LastName' OR 'FirstName LastName' both accepted"

**해석:** 사람 이름 컬럼에 한해, 두 컬럼으로 분리(`FirstName`, `LastName`)하든 한 컬럼에 합치(`FirstName LastName`)든 채점기가 동일하게 인정.

**예시:**

```csv
# 형태 A — 둘 다 인정
first_name,last_name
John,Doe

# 형태 B — 둘 다 인정
full_name
John Doe
```

**경계 케이스 — 명시 안 된 부분:**
- `"DOE, JOHN"` (성, 이름 콤마 형태) — 명시 X, 위험
- `"John D."` (이니셜) — 명시 X, 위험
- `"John Middle Doe"` (middle name) — 명시 X. `FirstName LastName`만 표현하면 매칭 가능하지만 데이터 손실

**원칙:** 가능하면 source가 사용한 형태 그대로 유지.

---

## 11. 우리 정규화 흐름 (코드)

```
LLM이 _answer(columns, rows) 호출
  ↓
tools/registry.py:_answer_handler
  ↓
scoring/normalize.py:normalize_answer_table(raw)
  ↓
ToolExecutionResult(answer=raw, normalized_answer=normalized)
  ↓
runner._write_task_outputs(prefer_normalized=True)
  ↓
/output/task_<id>/prediction.csv (normalized 버전)
```

raw는 `trace.json.answer`에 보존 — 디버깅용.

---

## 12. 컴플라이언스 체크리스트

- [ ] `prediction.csv`가 UTF-8인가 (BOM 없는 게 안전)
- [ ] 헤더 행이 첫 줄에 있는가
- [ ] 빈 셀이 `""`로 표현되는가 (`pd.to_csv(na_rep="")` 또는 직접 작성)
- [ ] 숫자가 ROUND_HALF_UP으로 소수점 2자리인가
- [ ] 날짜가 `YYYY-MM-DD`인가
- [ ] datetime의 timezone 일관성 (UTC면 `Z`, 없으면 없음)
- [ ] 문자열 내부 공백이 보존되는가 (잘못된 압축 금지)
- [ ] case가 source와 일치하는가
- [ ] 컬럼 헤더가 의미 있는가 (디버깅 도움)

---

## 13. 자주 깨지는 지점

1. **`pd.to_csv()` 기본 출력에 NaN이 `""`가 아니라 `"nan"`** — `na_rep=""` 명시
2. **Python `:.2f`는 HALF_EVEN** — `decimal.Decimal` + `ROUND_HALF_UP` 사용
3. **dateutil parser의 ambiguous date 처리** — `dayfirst` 파라미터로 컨텍스트 컨벤션 매치
4. **string `.lower()` 호출** — 룰은 case-sensitive. LLM 프롬프트에 "preserve case" 명시
5. **timezone-naive datetime을 임의로 UTC로 변환** — 룰은 timezone 없으면 보존. 멋대로 `Z` 붙이면 mismatch
6. **Excel-style 날짜 (`44927` serial)** — pandas 읽을 때 변환 필수, 안 하면 정규화 후에도 `"44927.00"` 같은 무의미값
