# Qwen3.5-35B-A3B 엔드포인트 capability

> 🌐 **Language**: [English](qwen_endpoint_capabilities.md) · **한국어** · [中文](qwen_endpoint_capabilities.zh.md)

_마지막 측정: **2026-04-27T06:21:15Z**_
_Endpoint: `http://127.0.0.1:8000/v1` · Served model: `qwen3.5-35b-a3b`_

평가 컨테이너는 `MODEL_API_URL`, `MODEL_API_KEY="EMPTY"` (문자열 그대로), `MODEL_NAME=qwen3.5-35b-a3b` env 변수로 운영진의 공식 Qwen endpoint와 통신한다. 우리는 같은 weight를 self-hosted vLLM으로 서빙해서 로컬 dev에서의 prompt 튜닝이 leaderboard로 1:1 transfer되도록 mirror한다.

**운영진 공식 vLLM serve 명령** (rules 페이지, 2026-05):

```bash
vllm serve <model_path> \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --served-model-name qwen3.5-35b-a3b \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
```

핵심 사실:
- **Max context: 262144 토큰** (~256K). 이전 probe의 32753은 우리 로컬 한도, 운영진 endpoint 아님.
- **Reasoning parser `qwen3`**: `<think>…</think>` 블록 분리 처리.
- **Tool calling 활성화** (`qwen3_coder` parser). 우리는 자체 JSON 파싱 사용해서 활용 X.
- **Seed 1024 고정** (서버 측). endpoint 자체는 우리 per-request seed 없으면 결정적이지 않음.

endpoint가 바뀔 때마다 (새 vLLM 버전 / 새 weight / 새 flag) `bash scripts/serve_qwen_docker.sh --probe-only`를 재실행하고 갱신된 표를 commit한다.

## Probe 체크리스트

| 항목 | 값 | 검증 방법 |
|---|---|---|
| OpenAI 호환 endpoint | yes | `GET http://127.0.0.1:8000/v1/models` |
| `chat.completions.create` 텍스트 응답 | yes (0.314s) | "say pong" 단순 프롬프트 |
| 안정적 fenced ```json 출력 | 10/10 (100.0%) | `parse_model_step` 스타일 프롬프트 × N회 |
| `response_format={"type":"json_object"}` 수용 | yes | format 지정 요청 |
| `extra_body.guided_json` 수용 (vLLM grammar) | no (JSONDecodeError: Expecting value: line 1 column 1 (char 0)) | full action schema 요청 |
| 최대 입력 context 토큰 (단일 turn) | ~32753 tokens | 4xx 나올 때까지 점진적 filler |
| Latency p50 (짧은 프롬프트 → 8 tok) | 0.187s | 20회 호출 median |
| Latency p95 (짧은 프롬프트 → 8 tok) | 0.203s | 동일 |
| 동시성: 8 parallel 요청 | 14.1 req/s (846.0 req/min) | 32-request burst |
| Tokenizer cost를 `usage`에 보고 | yes | API 응답 usage 블록 introspect |

## 우리 시스템에 미치는 영향

- **JSON-mode 지원?** → YES — Phase 2는 fenced-block 파서 빼고 response_format/guided_json 사용 가능.
- **Max context (운영진 endpoint) = 262144 토큰** (rules 페이지). 우리 로컬 probe 32753은 우리 vLLM 설정 한도일 뿐; 두 환경 모두에서 안전하려면 agent prompt를 32K 기준으로 budget.
- **Throughput 예산**: 운영진 A-board 2h cap (57 task), B-board 12h (324 task). 우리 측정 14.1 req/s 기준 이론 ceiling은 A/B 각각 101,520 / 609,120 LLM 호출 — 모두 충분히 큼. 실제 제약은 per-task wall-clock.
- **Latency**: 평균 0.186s × steps_per_task × tasks. 운영진 endpoint는 우리 DGX보다 ~1.8× slower 추정 (v6 SIGTERM forensic). 우리 50-task wall-clock 30min 이하면 A-board 2h fit 안전 마진.
- **HTTP client timeout (v6)**: `OpenAIModelAdapter`가 `OpenAI(timeout=240.0)` 명시. 기본 60s는 silent ceiling — 동시 요청 부하 시 vLLM 큐가 60s 초과하면 모델이 답할 수 있어도 클라이언트가 abort. v6 build #1에서 18 task가 "Request timed out"으로 손실된 후 명시 timeout 추가.

## Raw probe 데이터

```json
{
  "probed_at": "2026-04-27T06:21:15Z",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "qwen3.5-35b-a3b",
  "basic_chat": {
    "ok": true,
    "elapsed_seconds": 0.314,
    "content": "pong",
    "usage": {
      "completion_tokens": 2,
      "prompt_tokens": 15,
      "total_tokens": 17,
      "completion_tokens_details": null,
      "prompt_tokens_details": null
    }
  },
  "json_mode": {
    "supported": true,
    "parsed": {
      "ok": true
    },
    "raw": "{\"ok\": true}"
  },
  "guided_json": {
    "supported": false,
    "reason": "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
  },
  "fenced_json": {
    "samples": 10,
    "successes": 10,
    "rate": 1.0,
    "failure_examples": []
  },
  "context_probe": [
    {
      "target_tokens": 4096,
      "ok": true,
      "elapsed_seconds": 1.275,
      "prompt_tokens": 4081
    },
    {
      "target_tokens": 16384,
      "ok": true,
      "elapsed_seconds": 5.342,
      "prompt_tokens": 16369
    },
    {
      "target_tokens": 32768,
      "ok": true,
      "elapsed_seconds": 13.691,
      "prompt_tokens": 32753
    },
    {
      "target_tokens": 65536,
      "ok": false,
      "error": "BadRequestError: Error code: 400 - {'error': {'message': \"This model's maximum context length is 32768 tokens. However, your request has 65521 input tokens. Please reduce the length of the input messages. (parameter=i"
    }
  ],
  "latency_serial": {
    "samples": 20,
    "p50_seconds": 0.187,
    "p95_seconds": 0.203,
    "mean_seconds": 0.186,
    "min_seconds": 0.166,
    "max_seconds": 0.203
  },
  "concurrency_probe": {
    "concurrency": 8,
    "requests": 32,
    "wall_seconds": 2.269,
    "throughput_rps": 14.1,
    "throughput_rpm": 846.0,
    "latency_p50_seconds": 0.555,
    "latency_p95_seconds": 0.688
  }
}
```

## 운영진 측 미해결 질문

Discord / GitHub issues에서 답이 들어올 때마다 추적:

1. **`Score = Recall − λ·(ExtraCols/PredictedCols)` 의 λ 값** — 기본 0.10 가정; column-ablation은 {0.05, 0.10, 0.20}로 게이트.
2. **`knowledge.md` 표준 포맷 + 항상 존재 여부** — 항상 존재? free-form markdown? 고정 섹션?
3. **평가용 Qwen endpoint의 capability (위와 비교)** — `response_format={"type":"json_object"}`, `guided_json`, rate limit 확인.
4. **Phase 1 task 개수** — per-task wall-clock 예산 산정용.
5. **`task.json` schema 안정성** — strict `{task_id, difficulty, question}`인가, 확장 가능한가?
