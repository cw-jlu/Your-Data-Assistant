"""Probe an OpenAI-compatible Qwen endpoint and write a capabilities report.

Self-contained — depends only on stdlib + the `openai` package. The output
schema mirrors the checklist in `docs/qwen_endpoint_capabilities.md`, so the
report this script writes can be committed verbatim once filled in.

Run from `serve_qwen_docker.sh`, or directly:

    python scripts/probe_qwen.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model qwen3.5-35b-a3b \
        --report docs/qwen_endpoint_capabilities.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import APIError, AsyncOpenAI, BadRequestError, OpenAI
except ImportError:  # pragma: no cover
    print("openai package required: pip install 'openai>=1.40'", file=sys.stderr)
    raise


# ---- individual probes ------------------------------------------------------


def probe_basic_chat(client: OpenAI, model: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=8,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - t0
    return {
        "ok": True,
        "elapsed_seconds": round(elapsed, 3),
        "content": (resp.choices[0].message.content or "").strip(),
        "usage": resp.usage.model_dump() if resp.usage else None,
    }


def probe_json_mode(client: OpenAI, model: str) -> dict[str, Any]:
    """response_format={'type':'json_object'} — vLLM-supported on Qwen3."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": 'Reply with exactly {"ok": true} and nothing else.',
                },
                {"role": "user", "content": "Confirm."},
            ],
            response_format={"type": "json_object"},
            max_tokens=32,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(content)
            return {"supported": True, "parsed": parsed, "raw": content}
        except json.JSONDecodeError:
            return {"supported": False, "reason": "non-JSON response", "raw": content}
    except (BadRequestError, APIError) as exc:
        return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}


def probe_guided_json(client: OpenAI, model: str) -> dict[str, Any]:
    """vLLM-specific structured output via extra_body.guided_json."""
    schema = {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string"},
            "action_input": {"type": "object"},
        },
        "required": ["thought", "action", "action_input"],
    }
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Plan one ReAct step in JSON."},
                {"role": "user", "content": "List files under context/."},
            ],
            extra_body={"guided_json": schema},
            max_tokens=256,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        ok = all(k in parsed for k in ("thought", "action", "action_input"))
        return {"supported": ok, "parsed": parsed, "raw_preview": content[:300]}
    except (BadRequestError, APIError) as exc:
        return {"supported": False, "reason": f"{type(exc).__name__}: {exc}"}
    except json.JSONDecodeError as exc:
        return {"supported": False, "reason": f"JSONDecodeError: {exc}"}


def probe_fenced_json(client: OpenAI, model: str, samples: int) -> dict[str, Any]:
    """Project's actual contract: a single ```json fenced block.

    Measures how often the model honors the contract under a vanilla prompt —
    this is what `parse_model_step` consumes.
    """
    sys_prompt = (
        "You must reply with exactly one ```json fenced block containing "
        '{"thought": str, "action": str, "action_input": object}. '
        "No prose outside the fence."
    )
    successes = 0
    failures: list[str] = []
    for i in range(samples):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Decide step {i} for inspecting a CSV."},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
        except APIError as exc:
            failures.append(f"call {i}: {type(exc).__name__}: {exc}")
            continue
        if "```json" in content and "```" in content.split("```json", 1)[1]:
            inner = content.split("```json", 1)[1].split("```", 1)[0].strip()
            try:
                parsed = json.loads(inner)
                if all(k in parsed for k in ("thought", "action", "action_input")):
                    successes += 1
                    continue
            except json.JSONDecodeError:
                pass
        failures.append(f"call {i}: bad fenced block — {content[:120]!r}")
    return {
        "samples": samples,
        "successes": successes,
        "rate": round(successes / samples, 3) if samples else None,
        "failure_examples": failures[:3],
    }


def probe_context_length(client: OpenAI, model: str, targets: list[int]) -> list[dict[str, Any]]:
    """Send progressively longer prompts; stop at first 4xx/5xx."""
    results: list[dict[str, Any]] = []
    for target in targets:
        # ~4 chars/token is a rough lower bound for English fillers; we slightly
        # under-shoot so vLLM accepts the request and reports actual prompt_tokens.
        filler_words = max(target - 32, 1)
        filler = ("the " * filler_words).rstrip()
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Ignore this filler and reply with: ok\n\n{filler}",
                    }
                ],
                max_tokens=8,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            results.append(
                {
                    "target_tokens": target,
                    "ok": True,
                    "elapsed_seconds": round(elapsed, 3),
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                }
            )
        except APIError as exc:
            results.append(
                {
                    "target_tokens": target,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            break
    return results


def probe_latency_serial(client: OpenAI, model: str, samples: int) -> dict[str, Any]:
    latencies: list[float] = []
    for i in range(samples):
        t0 = time.perf_counter()
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Reply with the number {i}."}],
            max_tokens=8,
            temperature=0.0,
        )
        latencies.append(time.perf_counter() - t0)
    p95 = (
        round(statistics.quantiles(latencies, n=20)[18], 3) if samples >= 20 else None
    )
    return {
        "samples": samples,
        "p50_seconds": round(statistics.median(latencies), 3),
        "p95_seconds": p95,
        "mean_seconds": round(statistics.mean(latencies), 3),
        "min_seconds": round(min(latencies), 3),
        "max_seconds": round(max(latencies), 3),
    }


async def _one_call(aclient: AsyncOpenAI, model: str, idx: int) -> float:
    t0 = time.perf_counter()
    await aclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Echo: {idx}"}],
        max_tokens=8,
        temperature=0.0,
    )
    return time.perf_counter() - t0


async def _concurrency_run(
    base_url: str, api_key: str, model: str, concurrency: int, requests: int
) -> dict[str, Any]:
    aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def runner(idx: int) -> float:
        async with sem:
            return await _one_call(aclient, model, idx)

    t0 = time.perf_counter()
    latencies = await asyncio.gather(*[runner(i) for i in range(requests)])
    total = time.perf_counter() - t0
    p95 = (
        round(statistics.quantiles(latencies, n=20)[18], 3) if requests >= 20 else None
    )
    return {
        "concurrency": concurrency,
        "requests": requests,
        "wall_seconds": round(total, 3),
        "throughput_rps": round(requests / total, 2),
        "throughput_rpm": round(requests / total * 60, 1),
        "latency_p50_seconds": round(statistics.median(latencies), 3),
        "latency_p95_seconds": p95,
    }


def probe_concurrency(
    base_url: str, api_key: str, model: str, concurrency: int, requests: int
) -> dict[str, Any]:
    return asyncio.run(_concurrency_run(base_url, api_key, model, concurrency, requests))


# ---- report writer ----------------------------------------------------------


def render_report(data: dict[str, Any]) -> str:
    """Render a markdown report compatible with the checklist headings."""
    ts = data["probed_at"]
    base = data["base_url"]
    model = data["model"]
    basic = data["basic_chat"]
    json_mode = data["json_mode"]
    guided = data["guided_json"]
    fenced = data["fenced_json"]
    ctx = data["context_probe"]
    lat = data["latency_serial"]
    conc = data["concurrency_probe"]

    max_ok_tokens = max((r.get("prompt_tokens") or r["target_tokens"] for r in ctx if r["ok"]), default=0)
    json_mode_cell = "yes" if json_mode["supported"] else f"no ({json_mode.get('reason', 'unknown')})"
    guided_cell = "yes" if guided["supported"] else f"no ({guided.get('reason', 'unknown')})"

    lines = [
        f"# Qwen3.5-35B-A3B endpoint capabilities",
        "",
        f"_Last probed: **{ts}**_  ",
        f"_Endpoint: `{base}` · Served model: `{model}`_",
        "",
        "The eval container talks to the official Qwen endpoint via env vars",
        "`MODEL_API_URL`, `MODEL_API_KEY`, `MODEL_NAME=qwen3.5-35b-a3b`. We mirror those",
        "semantics with a self-hosted vLLM serving the same weights so prompt-tuning on",
        "local dev transfers 1:1 to the leaderboard.",
        "",
        "Re-run `bash scripts/serve_qwen_docker.sh --probe-only` whenever the endpoint changes",
        "(new vLLM version, new weights, new flags) and commit the regenerated table.",
        "",
        "## Probe checklist",
        "",
        "| Capability | Value | How tested |",
        "|---|---|---|",
        f"| Endpoint OpenAI-compatible | yes | `GET {base}/models` |",
        f"| `chat.completions.create` returns text | yes ({basic['elapsed_seconds']}s) | trivial \"say pong\" prompt |",
        f"| Stable fenced ```json output | {fenced['successes']}/{fenced['samples']} ({(fenced['rate'] or 0) * 100:.1f}%) | `parse_model_step`-style prompt × N calls |",
        f"| `response_format={{\"type\":\"json_object\"}}` accepted | {json_mode_cell} | request with format set |",
        f"| `extra_body.guided_json` accepted (vLLM grammar) | {guided_cell} | request with full action schema |",
        f"| Max input context tokens (single turn) | ~{max_ok_tokens} tokens | progressive filler until 4xx |",
        f"| Latency p50 (short prompt → 8 tok) | {lat['p50_seconds']}s | {lat['samples']}-call median |",
        f"| Latency p95 (short prompt → 8 tok) | {lat['p95_seconds']}s | same |",
        f"| Concurrency: {conc['concurrency']} parallel requests | {conc['throughput_rps']} req/s ({conc['throughput_rpm']} req/min) | {conc['requests']}-request burst |",
        f"| Tokenizer cost reported in `usage` | {'yes' if basic.get('usage') else 'no'} | introspect API response usage block |",
        "",
        "## Implications for our system",
        "",
        f"- **JSON-mode supported?** → {'YES — Phase 2 can drop the fenced-block parser and use response_format/guided_json.' if json_mode['supported'] else 'NO — keep the fenced-block parser; Phase 2 adds JSON parse-retry only.'}",
        f"- **Max context vs Extreme tasks** → {'≥128K, can ingest extreme contexts directly.' if max_ok_tokens >= 128000 else f'~{max_ok_tokens} tokens; must compress with dataframe_describe before passing extreme contexts.'}",
        f"- **Throughput budget**: 12h × 60 × {conc['throughput_rpm']} req/min ≈ {int(12 * 60 * conc['throughput_rpm']):,} LLM calls. Drives `run.max_workers` and self-consistency `k`.",
        f"- **Latency**: avg {lat['mean_seconds']}s × steps_per_task × tasks → keep ≤ 11h with 1h safety.",
        "",
        "## Raw probe data",
        "",
        "```json",
        json.dumps(data, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Open questions for organizers",
        "",
        "Track answers as we get them via Discord / GitHub issues:",
        "",
        "1. **λ value in `Score = Recall − λ·(ExtraCols/PredictedCols)`** — defaulting to 0.10; column-ablation gates against {0.05, 0.10, 0.20}.",
        "2. **`knowledge.md` standard format and presence guarantee** — always present? Free-form markdown? Fixed sections?",
        "3. **Eval Qwen endpoint capabilities (vs ours above)** — confirm `response_format={\"type\":\"json_object\"}`, `guided_json`, rate limit.",
        "4. **Phase 1 task count** — for per-task wall-clock budget.",
        "5. **`task.json` schema stability** — strictly `{task_id, difficulty, question}` or could extend?",
        "",
    ]
    return "\n".join(lines)


def write_report(report_path: Path, data: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(data), encoding="utf-8")


# ---- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", required=True, help="served-model-name as exposed by the endpoint")
    ap.add_argument("--report", default="docs/qwen_endpoint_capabilities.md")
    ap.add_argument("--latency-samples", type=int, default=20)
    ap.add_argument("--fenced-samples", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--concurrency-requests", type=int, default=32)
    ap.add_argument(
        "--context-targets",
        default="4096,16384,32768,65536,131072",
        help="comma-separated prompt-token targets to probe (stop at first failure)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    targets = [int(t) for t in args.context_targets.split(",") if t.strip()]
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print("[probe] basic chat...", flush=True)
    basic = probe_basic_chat(client, args.model)
    print(f"  → {basic['content']!r} in {basic['elapsed_seconds']}s")

    print("[probe] response_format JSON mode...", flush=True)
    json_mode = probe_json_mode(client, args.model)
    print(f"  → supported={json_mode['supported']}")

    print("[probe] extra_body.guided_json (vLLM grammar)...", flush=True)
    guided = probe_guided_json(client, args.model)
    print(f"  → supported={guided['supported']}")

    print(f"[probe] fenced ```json contract over {args.fenced_samples} calls...", flush=True)
    fenced = probe_fenced_json(client, args.model, args.fenced_samples)
    print(f"  → {fenced['successes']}/{fenced['samples']} ({(fenced['rate'] or 0) * 100:.1f}%)")

    print(f"[probe] context length scan: {targets}...", flush=True)
    ctx_results = probe_context_length(client, args.model, targets)
    for r in ctx_results:
        flag = "ok" if r["ok"] else f"FAIL ({r.get('error', '?')})"
        print(f"  {r['target_tokens']:>8d} tokens: {flag}")

    print(f"[probe] serial latency × {args.latency_samples}...", flush=True)
    lat = probe_latency_serial(client, args.model, args.latency_samples)
    print(f"  → p50={lat['p50_seconds']}s p95={lat['p95_seconds']}")

    print(
        f"[probe] concurrency={args.concurrency} × {args.concurrency_requests} requests...",
        flush=True,
    )
    conc = probe_concurrency(
        args.base_url, args.api_key, args.model, args.concurrency, args.concurrency_requests
    )
    print(f"  → {conc['throughput_rps']} req/s, p95={conc['latency_p95_seconds']}s")

    data: dict[str, Any] = {
        "probed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_url": args.base_url,
        "model": args.model,
        "basic_chat": basic,
        "json_mode": json_mode,
        "guided_json": guided,
        "fenced_json": fenced,
        "context_probe": ctx_results,
        "latency_serial": lat,
        "concurrency_probe": conc,
    }

    report_path = Path(args.report)
    write_report(report_path, data)
    print(f"\n[probe] report written → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
