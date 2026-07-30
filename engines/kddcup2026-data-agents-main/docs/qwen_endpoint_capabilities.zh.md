# Qwen3.5-35B-A3B 端点能力 (capabilities)

> 🌐 **Language**: [English](qwen_endpoint_capabilities.md) · [한국어](qwen_endpoint_capabilities.ko.md) · **中文**

_最近一次探测: **2026-04-27T06:21:15Z**_
_Endpoint: `http://127.0.0.1:8000/v1` · 模型: `qwen3.5-35b-a3b`_

评测容器通过环境变量 `MODEL_API_URL`、`MODEL_API_KEY="EMPTY"`(字面值)、`MODEL_NAME=qwen3.5-35b-a3b` 与官方 Qwen 端点通信。我们用 self-hosted vLLM 服务相同权重以镜像该语义,这样本地开发的 prompt 调优可以 1:1 迁移到 leaderboard。

**官方 vLLM serve 命令** (rules 页面, 2026-05):

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

要点:
- **最大上下文: 262144 tokens** (~256K)。之前探测的 32753 是我们本地的限制,不是组织者端点。
- **Reasoning parser `qwen3`**: 把 reasoning 链包在 `<think>…</think>` 中分离。
- **Tool calling 启用** (`qwen3_coder` parser)。我们用自己的 JSON 解析,所以不使用。
- **Seed 1024 服务器端固定**。端点本身在没有我们 per-request seed 时不是确定性的。

每当端点改变(新 vLLM 版本、新权重、新 flag)时重新执行 `bash scripts/serve_qwen_docker.sh --probe-only`,并提交重新生成的表格。

## 探测清单

| 能力 | 值 | 测试方法 |
|---|---|---|
| 端点 OpenAI 兼容 | yes | `GET http://127.0.0.1:8000/v1/models` |
| `chat.completions.create` 返回文本 | yes (0.314s) | 简单的 "say pong" 提示 |
| 稳定的 fenced ```json 输出 | 10/10 (100.0%) | `parse_model_step` 风格提示 × N 次调用 |
| `response_format={"type":"json_object"}` 被接受 | yes | 设置 format 的请求 |
| `extra_body.guided_json` 被接受 (vLLM grammar) | no (JSONDecodeError: Expecting value: line 1 column 1 (char 0)) | 带完整 action schema 的请求 |
| 最大输入上下文 token (单轮) | ~32753 tokens | 渐进式 filler 直到 4xx |
| Latency p50 (短提示 → 8 tok) | 0.187s | 20 次调用中位数 |
| Latency p95 (短提示 → 8 tok) | 0.203s | 同上 |
| 并发: 8 个并行请求 | 14.1 req/s (846.0 req/min) | 32-request burst |
| Tokenizer 成本在 `usage` 中报告 | yes | API 响应的 usage 块 introspect |

## 对我们系统的影响

- **支持 JSON-mode?** → YES — Phase 2 可以丢弃 fenced-block 解析器,改用 response_format/guided_json。
- **最大上下文 (官方 endpoint) = 262144 tokens** (rules 页面)。我们本地 probe 的 32753 是我们 vLLM 配置的限制;两个环境都安全的话,agent prompt 按 32K 预算。
- **吞吐预算**: 官方 A-board 2h cap (57 task)、B-board 12h (324 task)。按我们测量的 14.1 req/s,理论上限 A/B 分别 101,520 / 609,120 次 LLM 调用 — 都远超需求。真正约束是 per-task wall-clock。
- **Latency**: 平均 0.186s × steps_per_task × tasks。官方 endpoint 估计比我们 DGX 慢 ~1.8× (v6 SIGTERM 反推)。我们 50-task wall-clock ≤30min 时 A-board 2h 容纳是安全的。
- **HTTP client timeout (v6)**: `OpenAIModelAdapter` 显式设置 `OpenAI(timeout=240.0)`。默认 60s 是隐性上限 — 并发负载下 vLLM 队列超过 60s 时,即使模型本可返回,客户端也会中止。v6 build #1 因此损失了 18 个 task 才加上显式 timeout。

## 原始探测数据

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

## 给主办方的待解决问题

通过 Discord / GitHub issues 跟踪获得的答复:

1. **`Score = Recall − λ·(ExtraCols/PredictedCols)` 中的 λ 值** — 默认 0.10;column-ablation 在 {0.05, 0.10, 0.20} 上做 gate。
2. **`knowledge.md` 标准格式与存在性保证** — 总是存在?自由格式 markdown?固定章节?
3. **评测 Qwen 端点能力 (与上述相比)** — 确认 `response_format={"type":"json_object"}`、`guided_json`、rate limit。
4. **Phase 1 任务数量** — 用于 per-task wall-clock 预算计算。
5. **`task.json` schema 稳定性** — 严格 `{task_id, difficulty, question}` 还是可扩展?
