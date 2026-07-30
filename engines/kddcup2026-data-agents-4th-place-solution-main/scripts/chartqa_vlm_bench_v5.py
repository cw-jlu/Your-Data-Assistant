"""ChartQA bench v5 — v4 mix prompt + max_tokens=16K + HTTP retry.

Goal: collapse raw acc onto clean acc by removing the two infra failure modes:
  - HTTP timeout → exponential-backoff retry (max 3)
  - thinking exhaustion → max_tokens=16384 (= ample for thinking + answer)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"


PROMPT_V5 = """Read the chart and answer the question.

Decide the answer type first:
- LABEL: the question asks WHICH item / WHAT category / NAME the X / a yes-no.
  Return the EXACT text from the chart's axis labels, legend, or annotations,
  including any trailing symbols (*, ?, etc.).
- NUMBER: the question asks HOW MANY / WHAT VALUE / a percentage / a ratio.
  Return only the digits (no units, no %).

Optional tool: if the question requires counting a condition across many
items, computing a ratio/average, or precise arithmetic, you MAY emit ONE
fenced block to execute:
```python
# any code; print() the value you need
```
After tool stdout is returned, finalize the answer.

After your reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


def encode_image(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def exec_python(code: str, timeout: int = 10) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["python3", path],
            capture_output=True, text=True, timeout=timeout,
            env={"PATH": "/usr/bin:/bin"},
        )
        out = (r.stdout or "")[:2000]
        err = (r.stderr or "")[:500]
        if r.returncode != 0:
            return f"[ERROR rc={r.returncode}]\n{err}\nstdout:\n{out}"
        return out.strip() or "[no stdout]"
    except subprocess.TimeoutExpired:
        return "[ERROR] timeout after 10s"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def chat_with_retry(messages: list, max_tokens: int, max_retries: int = 3, http_timeout: int = 180,
                    temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "CF-Access-Client-Id": CF_ID,
        "CF-Access-Client-Secret": CF_SECRET,
    }
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{API_BASE}/chat/completions",
                              headers=headers, json=body, timeout=http_timeout)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            sleep = min(2 ** attempt * 2, 30)  # 2, 4, 8, 16 capped at 30
            time.sleep(sleep)
    raise last_exc  # unreachable


def parse_final_answer(content: str) -> str | None:
    if not content:
        return None
    m = _ANSWER_LINE_RE.search(content)
    if m:
        return m.group(1).strip()
    return None


def parse_number(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").rstrip("%")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    p_num = parse_number(pred)
    g_num = parse_number(gold)
    if g_num is not None and p_num is not None:
        if abs(g_num) < 1e-9:
            return abs(p_num) < 1e-6
        return abs(p_num - g_num) / abs(g_num) <= tol
    return pred.strip().lower() == gold.strip().lower()


def run_one(idx: int, ex: dict, max_tokens: int, max_turns: int = 3) -> dict:
    t0 = time.time()
    try:
        q = ex["query"]
        gold = ex["label"][0] if isinstance(ex["label"], list) else ex["label"]
        b64 = encode_image(ex["image"])

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{PROMPT_V5}\n\nQuestion: {q}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        tool_calls_made = 0
        final = None
        last_content = ""
        finish_reason = None

        for turn in range(max_turns):
            r = chat_with_retry(messages, max_tokens=max_tokens)
            msg = r["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            last_content = content
            finish_reason = r["choices"][0]["finish_reason"]
            final = parse_final_answer(content)
            if final is not None:
                break
            code_blocks = _PYTHON_BLOCK_RE.findall(content)
            if not code_blocks:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                    "Output the final answer as: ANSWER: <value>"})
                continue
            out = exec_python(code_blocks[0])
            tool_calls_made += 1
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                f"TOOL stdout:\n{out}\n\n"
                f"Output ANSWER: <value> on the last line."})

        ok = final is not None and relaxed_match(final, gold)
        return {
            "idx": idx, "q": q[:120], "gold": gold,
            "pred": (final or "")[:120],
            "raw": last_content[:300],
            "tool_calls": tool_calls_made,
            "turns": turn + 1,
            "ok": ok, "elapsed": time.time() - t0,
            "finish_reason": finish_reason,
        }
    except Exception as e:
        return {"idx": idx, "error": str(e)[:160], "ok": False, "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--out", default="artifacts/chartqa_bench_v5_retry.json")
    args = ap.parse_args()

    print(f"loading ChartQA test...", flush=True)
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), args.n)
    examples = [{"idx": i, **{k: ds[i][k] for k in ("query", "label", "image")}} for i in idxs]
    print(f"sampled {len(examples)} items, workers={args.workers}, max_tokens={args.max_tokens}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, e["idx"], e, args.max_tokens, args.max_turns): e for e in examples}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            tool = r.get("tool_calls", 0)
            print(f"[{i:3d}/{len(examples)}] {mark} idx={r['idx']} "
                  f"gold={str(r.get('gold',''))[:25]} pred={str(r.get('pred',''))[:25]} "
                  f"tools={tool} acc={acc:.3f} {err}", flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for x in results if x.get("ok"))
    n_err = sum(1 for x in results if x.get("error"))
    n_tool = sum(r.get("tool_calls", 0) for r in results if not r.get("error"))
    print()
    print(f"=== DONE n={len(results)} accuracy={n_ok/len(results):.4f} errors={n_err} "
          f"total_tool_calls={n_tool} elapsed={elapsed:.1f}s ===")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": len(results), "accuracy": n_ok / len(results),
        "errors": n_err, "elapsed_seconds": elapsed,
        "prompt_version": "v5_v4mix_plus_16k_retry",
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
