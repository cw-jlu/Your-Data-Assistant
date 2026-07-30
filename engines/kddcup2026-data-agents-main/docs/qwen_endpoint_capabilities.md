# Qwen3.5-35B-A3B endpoint capabilities

> 🌐 **Language**: **English** · [한국어](qwen_endpoint_capabilities.ko.md) · [中文](qwen_endpoint_capabilities.zh.md)

_Last probed: **2026-04-27T06:21:15Z**_
_Endpoint: `http://127.0.0.1:8000/v1` · Served model: `qwen3.5-35b-a3b`_

The eval container talks to the official Qwen endpoint via env vars
`MODEL_API_URL`, `MODEL_API_KEY="EMPTY"` (literal), `MODEL_NAME=qwen3.5-35b-a3b`.
We mirror those semantics with a self-hosted vLLM serving the same weights so
prompt-tuning on local dev transfers 1:1 to the leaderboard.

**Organizer's official vLLM serve command** (rules page, 2026-05):

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

Key facts from this spec:
- **Max context: 262144 tokens** (~256K). Earlier probe found 32753 — that was our local probe limit, not the organizer endpoint. Hard / Extreme tasks with full document context fit comfortably.
- **Reasoning parser `qwen3`** wraps reasoning chains in `<think>…</think>` blocks separate from the response.
- **Tool calling enabled** via `qwen3_coder` parser. We don't use it (we parse our own JSON), but the endpoint supports it.
- **Seed 1024 fixed** at the server side — the endpoint itself is not deterministic without our per-request `seed`.

Re-run `bash scripts/serve_qwen_docker.sh --probe-only` whenever the endpoint changes
(new vLLM version, new weights, new flags) and commit the regenerated table.

## Probe checklist

| Capability | Value | How tested |
|---|---|---|
| Endpoint OpenAI-compatible | yes | `GET http://127.0.0.1:8000/v1/models` |
| `chat.completions.create` returns text | yes (0.314s) | trivial "say pong" prompt |
| Stable fenced ```json output | 10/10 (100.0%) | `parse_model_step`-style prompt × N calls |
| `response_format={"type":"json_object"}` accepted | yes | request with format set |
| `extra_body.guided_json` accepted (vLLM grammar) | no (JSONDecodeError: Expecting value: line 1 column 1 (char 0)) | request with full action schema |
| Max input context tokens (single turn) | ~32753 tokens | progressive filler until 4xx |
| Latency p50 (short prompt → 8 tok) | 0.187s | 20-call median |
| Latency p95 (short prompt → 8 tok) | 0.203s | same |
| Concurrency: 8 parallel requests | 14.1 req/s (846.0 req/min) | 32-request burst |
| Tokenizer cost reported in `usage` | yes | introspect API response usage block |

## Implications for our system

- **JSON-mode supported?** → YES — Phase 2 can drop the fenced-block parser and use response_format/guided_json.
- **Max context (organizer endpoint) = 262144 tokens** (rules page). Our local probe shows ~32753 because we serve a smaller-context build; budget the agent prompt against 32K to stay safe across both environments.
- **Throughput budget**: organizer A-board has 2h cap (57 task), B-board 12h (324 task). At our measured 14.1 req/s, theoretical ceiling is 101,520 / 609,120 LLM calls per A/B run respectively — far above need. Real constraint is per-task wall-clock, not throughput.
- **Latency**: avg 0.186s × steps_per_task × tasks. Organizer endpoint estimated ~1.8× slower than our DGX (v6 SIGTERM forensic). Targeting 50-task local wall-clock ≤30min keeps us safely inside A-board 2h after scale-up.
- **HTTP client timeout (v6)**: `OpenAIModelAdapter` sets `OpenAI(timeout=240.0)`. Default 60s was a silent ceiling — under concurrent load the vLLM queue could push individual requests past 60s and the client would abort even when the model would have answered. v6 build #1 hit this and lost 18 tasks to "Request timed out" before the explicit timeout was added.

## Raw probe data

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

## Open questions for organizers

Track answers as we get them via Discord / GitHub issues:

1. **λ value in `Score = Recall − λ·(ExtraCols/PredictedCols)`** — defaulting to 0.10; column-ablation gates against {0.05, 0.10, 0.20}.
2. **`knowledge.md` standard format and presence guarantee** — always present? Free-form markdown? Fixed sections?
3. **Eval Qwen endpoint capabilities (vs ours above)** — confirm `response_format={"type":"json_object"}`, `guided_json`, rate limit.
4. **Phase 1 task count** — for per-task wall-clock budget.
5. **`task.json` schema stability** — strictly `{task_id, difficulty, question}` or could extend?
