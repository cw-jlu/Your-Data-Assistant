"""TableVQA-Bench (terryoo) single-shot bench on gpuhost Qwen3.5-35B-A3B.

Splits: fintabnetqa (250), vtabfact, vwtq, vwtq_syn — covers financial tables,
fact-checking, Wikipedia tables. Same harness pattern as chartqa_vlm_bench_v5.

Default: stratified sample across the 4 splits.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
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

PROMPT = """Read the table image and answer the question.

Decide the answer type first:
- LABEL/TEXT: the question asks WHICH item / WHAT name / a yes-no (= for
  TabFact-style fact-checking, answer 1=true / 0=false).
  Return the EXACT text from the table cells (verbatim including symbols,
  units if printed in the cell).
- NUMBER: the question asks HOW MANY / WHAT VALUE / a percentage.
  Return only the digits (no units, no %).
- LIST: the question asks for multiple values. Return them comma-separated
  in the order they appear in the table.

After your reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


def encode_image(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def chat_with_retry(messages: list, max_tokens: int, max_retries: int = 3,
                    http_timeout: int = 180, temperature: float = 0.6) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
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
            sleep = min(2 ** attempt * 2, 30)
            time.sleep(sleep)
    raise last_exc


def parse_final_answer(content: str) -> str | None:
    if not content:
        return None
    m = _ANSWER_LINE_RE.search(content)
    return m.group(1).strip() if m else None


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


def match(pred: str, gold: str, tol: float = 0.02) -> bool:
    if pred is None:
        return False
    p_num = parse_number(pred)
    g_num = parse_number(gold)
    if g_num is not None and p_num is not None:
        if abs(g_num) < 1e-9:
            return abs(p_num) < 1e-6
        return abs(p_num - g_num) / abs(g_num) <= tol
    # text: case-insensitive contains/equals
    pl, gl = pred.strip().lower(), gold.strip().lower()
    return pl == gl or gl in pl


def run_one(split: str, idx: int, ex: dict, max_tokens: int) -> dict:
    t0 = time.time()
    try:
        q = ex["question"]
        gold = str(ex.get("gt", "")).strip()
        b64 = encode_image(ex["image"])
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f"{PROMPT}\n\nQuestion: {q}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}]
        r = chat_with_retry(messages, max_tokens=max_tokens)
        msg = r["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        final = parse_final_answer(content)
        ok = match(final or "", gold)
        return {
            "split": split, "idx": idx, "q": q[:120], "gold": gold[:120],
            "pred": (final or "")[:120], "raw": content[:280],
            "ok": ok, "elapsed": time.time() - t0,
            "finish_reason": r["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"split": split, "idx": idx, "error": str(e)[:160], "ok": False,
                "elapsed": time.time() - t0}


def stratified_sample(splits: list[str], n_per: int, seed: int) -> list[tuple[str, int, dict]]:
    random.seed(seed)
    out = []
    for s in splits:
        ds = load_dataset("terryoo/TableVQA-Bench", split=s)
        idxs = random.sample(range(len(ds)), min(n_per, len(ds)))
        for i in idxs:
            ex = ds[i]
            out.append((s, i, {"question": ex["question"], "gt": ex.get("gt"), "image": ex["image"]}))
        print(f"[load] {s}: {len(ds)} → sampled {len(idxs)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=12)  # 12 * 4 = 48
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="artifacts/tablevqa_bench.json")
    ap.add_argument("--splits", nargs="+", default=["fintabnetqa", "vtabfact", "vwtq", "vwtq_syn"])
    args = ap.parse_args()

    items = stratified_sample(args.splits, args.n_per_split, args.seed)
    print(f"sampled total {len(items)} items, workers={args.workers}, T={args.temperature}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, s, i, ex, args.max_tokens): (s, i)
                for (s, i, ex) in items}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:3d}/{len(items)}] {mark} {r['split']:<12} idx={r['idx']:<5} "
                  f"gold={str(r.get('gold',''))[:25]:<25} pred={str(r.get('pred',''))[:25]:<25} "
                  f"acc={acc:.3f} {err}", flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for x in results if x.get("ok"))
    n_err = sum(1 for x in results if x.get("error"))

    # Per-split breakdown
    per_split = {}
    for s in args.splits:
        sub = [r for r in results if r["split"] == s]
        if sub:
            n_s_ok = sum(1 for r in sub if r.get("ok"))
            per_split[s] = {"n": len(sub), "acc": n_s_ok/len(sub),
                            "n_err": sum(1 for r in sub if r.get("error"))}

    print()
    print(f"=== DONE n={len(results)} accuracy={n_ok/len(results):.4f} errors={n_err} elapsed={elapsed:.1f}s ===")
    for s, st in per_split.items():
        print(f"   {s}: n={st['n']} acc={st['acc']:.3f} err={st['n_err']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": len(results), "accuracy": n_ok / len(results),
        "errors": n_err, "elapsed_seconds": elapsed,
        "per_split": per_split,
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
