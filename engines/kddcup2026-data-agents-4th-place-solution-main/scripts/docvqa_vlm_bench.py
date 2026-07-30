"""DocVQA bench on gpuhost Qwen3.5-35B-A3B (lmms-lab/DocVQA validation).

DocVQA = document image QA (= scanned reports, forms, charts in PDFs).
Standard metric: ANLS (Average Normalized Levenshtein Similarity). We use
a simpler relaxed match for POC; ANLS would require Levenshtein dependency.
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

PROMPT = """Read the document image and answer the question.

Decide the answer type first:
- TEXT/LABEL: the question asks WHICH item / WHAT name / a yes-no.
  Return the EXACT text from the document verbatim (preserve case, symbols,
  punctuation as printed).
- NUMBER: the question asks HOW MANY / WHAT VALUE / a percentage / a date.
  Return only the digits (no units, no %, dates as MM/DD/YYYY if shown).
- LIST: if multiple values, comma-separate in source order.

Keep the answer SHORT (a few words at most). Do not paraphrase.

After your reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


def encode_image(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def chat_with_retry(messages: list, max_tokens: int = 16384, max_retries: int = 3,
                    http_timeout: int = 180, temperature: float = 0.6) -> dict:
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.95, "top_k": 20,
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
            time.sleep(min(2 ** attempt * 2, 30))
    raise last_exc


def parse_final_answer(content: str) -> str | None:
    if not content:
        return None
    m = _ANSWER_LINE_RE.search(content)
    return m.group(1).strip() if m else None


def parse_number(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip("%")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def match_any(pred: str, golds: list[str], tol: float = 0.02) -> bool:
    """DocVQA: gold is a list of acceptable answers (alternative phrasings)."""
    if not pred:
        return False
    p_num = parse_number(pred)
    p_norm = normalize(pred)
    for g in golds:
        if not g:
            continue
        g_num = parse_number(g)
        if g_num is not None and p_num is not None:
            if abs(g_num) < 1e-9 and abs(p_num) < 1e-6:
                return True
            if abs(g_num) >= 1e-9 and abs(p_num - g_num) / abs(g_num) <= tol:
                return True
        g_norm = normalize(g)
        if p_norm == g_norm or g_norm in p_norm or p_norm in g_norm:
            return True
    return False


def run_one(idx: int, ex: dict, max_tokens: int) -> dict:
    t0 = time.time()
    try:
        q = ex["question"]
        golds = ex.get("answers") or []
        if isinstance(golds, str):
            golds = [golds]
        b64 = encode_image(ex["image"])
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f"{PROMPT}\n\nQuestion: {q}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}]
        r = chat_with_retry(messages, max_tokens=max_tokens)
        msg = r["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        final = parse_final_answer(content)
        ok = match_any(final or "", golds)
        return {
            "idx": idx, "q": q[:120],
            "gold": "|".join(golds)[:140],
            "pred": (final or "")[:120], "raw": content[:280],
            "ok": ok, "elapsed": time.time() - t0,
            "finish_reason": r["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"idx": idx, "error": str(e)[:160], "ok": False,
                "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="artifacts/docvqa_bench.json")
    args = ap.parse_args()

    print(f"loading DocVQA validation...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print(f"loaded {len(ds)} examples", flush=True)
    random.seed(args.seed)
    idxs = random.sample(range(len(ds)), args.n)
    examples = [{"idx": i, **{k: ds[i][k] for k in ("question", "answers", "image")}} for i in idxs]
    print(f"sampled {len(examples)}, workers={args.workers}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, e["idx"], e, args.max_tokens): e for e in examples}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:3d}/{len(examples)}] {mark} idx={r['idx']:<6} "
                  f"gold={str(r.get('gold',''))[:30]:<30} pred={str(r.get('pred',''))[:30]:<30} "
                  f"acc={acc:.3f} {err}", flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for x in results if x.get("ok"))
    n_err = sum(1 for x in results if x.get("error"))
    print()
    print(f"=== DONE n={len(results)} accuracy={n_ok/len(results):.4f} errors={n_err} elapsed={elapsed:.1f}s ===")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": len(results), "accuracy": n_ok / len(results),
        "errors": n_err, "elapsed_seconds": elapsed,
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
