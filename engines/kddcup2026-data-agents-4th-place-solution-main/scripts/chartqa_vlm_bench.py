"""ChartQA test bench against gpuhost Qwen3.5-35B-A3B (multimodal).

Samples N items from ChartQA test, sends image + question, computes
relaxed accuracy (= numeric ≤5% error or string EM, lowered case).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
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


def encode_image(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(question: str, b64: str, max_tokens: int = 1024) -> dict:
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}\n\nAnswer with the value ONLY, no explanation. For numbers, return just the digits (no units, no %)."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "CF-Access-Client-Id": CF_ID,
        "CF-Access-Client-Secret": CF_SECRET,
    }
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


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


def run_one(idx: int, ex: dict, max_tokens: int) -> dict:
    t0 = time.time()
    try:
        q = ex["query"]
        gold = ex["label"][0] if isinstance(ex["label"], list) else ex["label"]
        b64 = encode_image(ex["image"])
        r = ask(q, b64, max_tokens=max_tokens)
        msg = r["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        thinking = (msg.get("reasoning") or "")
        ok = relaxed_match(content, gold)
        return {
            "idx": idx, "q": q[:120], "gold": gold, "pred": content[:120],
            "ok": ok, "elapsed": time.time() - t0,
            "thinking_chars": len(thinking),
            "finish_reason": r["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"idx": idx, "error": str(e)[:160], "ok": False, "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="artifacts/chartqa_bench.json")
    args = ap.parse_args()

    # Load all and sample
    print(f"loading ChartQA test...", flush=True)
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    print(f"loaded {len(ds)} examples", flush=True)
    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), args.n)

    examples = [{"idx": i, **{k: ds[i][k] for k in ("query", "label", "image")}} for i in idxs]
    print(f"sampled {len(examples)} items, running with workers={args.workers}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, e["idx"], e, args.max_tokens): e for e in examples}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            running = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:3d}/{len(examples)}] {mark} idx={r['idx']} "
                  f"gold={str(r.get('gold',''))[:30]} pred={str(r.get('pred',''))[:30]} "
                  f"acc={running:.3f} {err}", flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for x in results if x.get("ok"))
    n_err = sum(1 for x in results if x.get("error"))
    print()
    print(f"=== DONE n={len(results)} accuracy={n_ok/len(results):.4f} errors={n_err} elapsed={elapsed:.1f}s ===")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": len(results),
        "accuracy": n_ok / len(results),
        "errors": n_err,
        "elapsed_seconds": elapsed,
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
