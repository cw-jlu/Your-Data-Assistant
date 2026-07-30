---
name: kddcup-agent
description: ReAct 에이전트 내부 — step 루프, JSON 응답 컨트랙트, 8개 툴 + answer terminal, 모델 어댑터, 프롬프트 빌더를 다룬다. "에이전트가 어떻게 동작해", "툴 추가하려면", "프롬프트 어디서 바꿔", "JSON 파싱 어디", "max_steps 어떻게 정해", "ReAct 루프", "모델 어댑터" 같은 질문에서 트리거. 에이전트 코드(`agents/`, `tools/`)를 만질 때 무조건 참조.
---

# KDD Cup 2026 — Agent Internals

ReAct 에이전트의 step 루프, 모델 응답 컨트랙트, 툴 표면, 프롬프트 빌더. 코드 진입점은 `src/data_agent_baseline/agents/` + `tools/`. 채점은 `kddcup-scoring`, 데이터는 `kddcup-dataset`, 컨테이너 wiring은 `kddcup-submission` 참조.

---

## 1. 한 태스크 내부 — step 루프

```
build_prompt (system + task) ──▶ for step in 1..max_steps:
                                  call_llm (chat.completions.create)
                                  parse_model_step (fenced ```json 추출)
                                    ├─ ParseError → observation = __error__, 다음 step
                                    └─ ok → dispatch
                                          ├─ 정상 툴 → ToolRegistry.execute → StepRecord 추가
                                          └─ "answer" → normalize → terminal
                                              ↓
                                     AgentRunResult (raw + normalized 답 모두 보존)
                                              ↓
                                     runner._write_task_outputs (prediction.csv = normalized)
```

진입: `agents/react.py:ReActAgent.run`. 데이터 흐름 시각화는 `docs/ARCHITECTURE.md` §5.

---

## 2. JSON 응답 컨트랙트 — **변경 시 lockstep 갱신 필수**

모델은 매 step에 **반드시** 다음 형태의 응답을 emit:

````
```json
{
  "thought": "...",
  "action": "tool_name",
  "action_input": {"k": "v"}
}
```
````

- 단일 fenced block, 주변에 prose 금지
- `action`은 `ToolRegistry`에 등록된 툴 이름 또는 `"answer"`
- `action_input`은 그 툴의 JSON schema에 맞는 dict

**컨트랙트의 단일 진실 원천 (Single Source of Truth):**
- 시스템 프롬프트: `agents/prompt.py:build_system_prompt`
- 파싱: `agents/react.py:parse_model_step`

**둘은 lockstep으로 유지**. 한쪽만 바꾸면 silently 전체 점수가 0이 된다.

**파싱 실패 처리:**
- 예외를 catch → step record에 `observation={"__error__": "..."}` 저장
- `raw_response`는 통째로 보존 → 디버깅에 필수
- 다음 step으로 진행 (모델이 자기 실수를 보고 회복할 기회)
- Phase 2 계획: parse-retry (corrective user message 1번, max 2회 — step 카운트에 포함 X)

---

## 3. 툴 표면 — 8개 + answer

`tools/registry.py:create_default_tool_registry()`가 단일 진입점. 각 툴은 `ToolSpec`(설명+JSON 예시) + `ToolHandler`(callable) 쌍.

| Tool | 파일 | 입력 | terminal? |
|---|---|---|---|
| `list_context` | `tools/filesystem.py` | `max_depth` | no |
| `read_csv` | `tools/filesystem.py` | `path`, `max_rows` | no |
| `read_json` | `tools/filesystem.py` | `path`, `max_chars` | no |
| `read_doc` | `tools/filesystem.py` | `path`, `max_chars` | no |
| `inspect_sqlite_schema` | `tools/sqlite.py` | `path` | no |
| `execute_context_sql` | `tools/sqlite.py` | `path`, `sql`, `limit` | no (read-only) |
| `execute_python` | `tools/python_exec.py` | `code` | no (30s subprocess) |
| `_answer` | `tools/registry.py` | `columns`, `rows` | **YES** |

**중요한 invariant:**
1. 모든 path는 `context/`를 루트로 한 **상대경로** — `resolve_context_path`가 강제
2. SQL은 **read-only** (모든 mutating 키워드 차단)
3. `execute_python`은 30s wall-clock + subprocess 격리, `task.context_dir`로 chdir, stdout/stderr fd-level 캡처
4. **`_answer`만 terminal** — 다른 툴은 무조건 `is_terminal=False`

**`describe_for_prompt()`가 곧 모델이 보는 툴 카탈로그**다. 스펙 텍스트 변경 = 프롬프트 변경 = 에이전트 동작 변경. 함부로 만지지 말고 holdout 재측정.

---

## 4. `_answer` terminal — 점수 보존의 마지막 관문

```python
# tools/registry.py
def _answer_handler(columns, rows):
    raw = AnswerTable(columns=columns, rows=rows)
    normalized = normalize_answer_table(raw)
    return ToolExecutionResult(
        is_terminal=True,
        answer=raw,
        normalized_answer=normalized,
        # ... observation 등
    )
```

- **두 가지 변형 모두 emit** — `runner._write_task_outputs(prefer_normalized=True)`가 normalized를 `prediction.csv`로 기록
- raw는 `trace.json.answer`에서 사후분석에 활용
- Phase 3 계획: validator를 통과시켜 conditional terminal로 변경 (warning 있으면 observation으로 반환 → 에이전트가 수정 후 재제출)

---

## 5. 새 툴 추가 절차

1. **handler 함수 작성** — `def _my_tool(...) -> ToolExecutionResult: ...`
2. **`ToolSpec` 작성** — 설명 + JSON schema 예시
3. **`tools/registry.py:create_default_tool_registry()`에 등록**
4. (terminal이라면) `is_terminal=True`로 `ToolExecutionResult` 반환
5. **holdout으로 회귀 테스트** — 새 툴 등장으로 인한 프롬프트 길이 변화가 다른 태스크 점수를 깎을 수 있음

---

## 6. 모델 어댑터

`agents/model.py`:

| 어댑터 | 용도 |
|---|---|
| `OpenAIModelAdapter` | 실 모델. `chat.completions.create` against any OpenAI-compatible `api_base`. 평가 시점에 운영진 endpoint 사용 |
| `ScriptedModelAdapter` | 테스트용. 미리 정해둔 응답을 순서대로 emit |

**JSON-mode 분기 (Phase 2 계획):**
- 초기화 시 1회 probe로 `response_format={"type":"json_object"}` 지원 여부 결정
- 지원하면 native JSON-mode, 미지원하면 fenced-block fallback

---

## 7. 프롬프트 빌더 — 3종

`agents/prompt.py`:

| 빌더 | 역할 |
|---|---|
| `build_system_prompt` | JSON 컨트랙트 정의, 툴 카탈로그 (`describe_for_prompt`), 정규화 규칙 (Phase 2부터), scoring policy (Phase 3) |
| `build_task_prompt` | 첫 user 메시지 — 난이도, 질문, context 트리, knowledge.md (Phase 1부터 자동 인젝션) |
| `build_observation_prompt` | 매 툴 실행 후 다음 user 메시지 — observation을 JSON으로 |

**프롬프트 변경 원칙:**
1. 단일 태스크에서만 검증하지 마라 — holdout 점수 회귀 가능
2. 시스템 프롬프트의 JSON 예시는 `parse_model_step`이 받을 형식과 정확히 일치
3. 토큰 예산 고려 — 32K 컨텍스트(우리 vLLM)에 task prompt + observation history + 답이 모두 들어가야 함

---

## 8. `max_steps` — 난이도별 차등 (Phase 2 계획)

기본: 16. Phase 2부터:

| difficulty | max_steps |
|---|---|
| easy | 8 |
| medium | 12 |
| hard | 20 |
| extreme | 28 |

`run/runner.py`의 wall-clock governor가 12h 예산 잔여에 따라 동적으로 하향 조정.

---

## 9. 자주 깨지는 지점

1. **JSON parse silent skip** — 현재 카운트가 step에 포함됨. Phase 2에서 retry로 변경
2. **execute_python timeout 30s** — 0.5GB CSV는 매번 다시 못 읽음 → persistent kernel (Phase 1) 필요
3. **Nested multiprocessing** — `task_timeout_seconds > 0` × `execute_python` = process-in-process. 디버거 X. 테스트는 `ScriptedModelAdapter` + 단일 프로세스 모드로
4. **`context/` escape** — 절대경로 전달 시 즉시 raise. 의도된 동작
5. **System prompt 토큰 폭증** — 새 툴 추가가 매번 +수백 토큰 → context 윈도우 압박
6. **Observation 압축 누락** — 큰 CSV를 그대로 observation으로 돌려보내면 한 번에 전체 윈도우 소진

---

## 10. 데이터클래스 (trace.json 직렬화)

`agents/runtime.py`:

```python
@dataclass(frozen=True)
class StepRecord:
    step_index: int
    thought: Optional[str]
    action: Optional[str]
    action_input: Optional[dict]
    raw_response: str
    observation: dict

@dataclass(frozen=True)
class AgentRuntimeState:
    task_id: str
    messages: list[dict]
    steps: list[StepRecord]

@dataclass(frozen=True)
class AgentRunResult:
    task_id: str
    succeeded: bool
    answer: Optional[AnswerTable]
    normalized_answer: Optional[AnswerTable]
    failure_reason: Optional[str]
    steps: list[StepRecord]
    e2e_elapsed_seconds: float
```

이 구조가 `trace.json` 형식의 단일 진실 원천. 변경 시 `runner._write_task_outputs`도 같이.

---

## 11. 진입점 한 줄 매핑

| 다루는 것 | 파일 |
|---|---|
| step 루프 + 파싱 + 에러 회복 | `agents/react.py` |
| 시스템·태스크·observation 프롬프트 | `agents/prompt.py` |
| OpenAI/Scripted 어댑터 | `agents/model.py` |
| 데이터클래스 | `agents/runtime.py` |
| 툴 카탈로그 | `tools/registry.py` |
| FS 툴 + 경로 sandbox | `tools/filesystem.py` |
| SQLite read-only | `tools/sqlite.py` |
| 30s subprocess Python | `tools/python_exec.py` |
