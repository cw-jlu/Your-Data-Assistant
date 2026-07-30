"""Aggregation-only bench with sub-agent calculation routing.

Setup: curated 10+ questions across benches that require aggregation
(count under condition, ratio, average, comparison, sum). Test 2 conditions:
  A. Single-shot: VLM answers directly (= our baseline universal prompt)
  B. Sub-agent: Stage1 VLM extracts data → Stage2 Python computes the answer

Compares whether explicit calculation routing beats single-shot.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
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


# ===== Curated aggregation tasks =====
# Mix of confirmed-fail cases from our prior benches + a few more aggregation
# Qs. We treat these by their (bench, idx) tuple.

CURATED = [
    # ChartQA (= confirmed aggregation failures)
    ("chartqa", 102,  "0.08"),   # largest dark red bar value (= percent ambiguity)
    ("chartqa", 383,  "1"),      # count years where X<60%
    ("chartqa", 1083, "5.546"),  # ratio Service/Product
    ("chartqa", 571,  "9.1"),    # average (= passed before, sanity)
    ("chartqa", 1378, "4.7"),    # average (= passed before)
    # TableVQA aggregation Qs
    ("tablevqa/fintabnetqa", 57,  "11,832,000"),  # value with $thousands
    ("tablevqa/vwtq",        665, "2"),            # number of connections (count)
    ("tablevqa/vwtq",        718, "20"),           # count
    ("tablevqa/vwtq",        429, "3"),            # count
    # DocVQA aggregation
    ("docvqa", 4139, "2,200.72"),  # value (= passed before, sanity)
]


PROMPT_SINGLE = """Look at the data and answer the question.

Decide the answer type first:
- TEXT/LABEL: exact text from the source.
- NUMBER: digits only (no units, no %).
- LIST: comma-separated.
- MULTIPLE_CHOICE: letter only.

After reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


PROMPT_STAGE1_EXTRACT = """Look at the data and answer in JSON only.

The user wants to compute an aggregation. First, identify what raw values
from the source are needed, then specify the operation. Do NOT do the
computation yourself; we will run Python.

Output a single JSON object on the last line:
{"values": [<float>, ...], "operation": "<sum|count|avg|ratio|min|max|other>", "expression": "<python expression using the values list>", "answer_format": "<number|percent|currency>"}

Example output line:
{"values": [10.5, 20.0, 30.5], "operation": "sum", "expression": "sum(values)", "answer_format": "number"}

Output ONLY the JSON line, no other text after it."""


PROMPT_STAGE2_FINAL = """Based on the computed result and the original question,
format the final answer.

After reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


# ===== HTTP & parsing =====

_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def chat(messages: list, max_tokens: int = 4096, temperature: float = 0.6,
         max_retries: int = 3) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0.0}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_KEY}",
               "CF-Access-Client-Id": CF_ID, "CF-Access-Client-Secret": CF_SECRET}
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{API_BASE}/chat/completions",
                              headers=headers, json=body, timeout=180)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt * 2, 30))


def parse_final(content: str) -> str | None:
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


def match_num(pred: str, gold: str, tol: float = 0.05) -> bool:
    p = parse_number(pred)
    g = parse_number(gold)
    if p is None or g is None:
        return (pred or "").strip().lower() == gold.strip().lower()
    if abs(g) < 1e-9:
        return abs(p) < 1e-6
    return abs(p - g) / abs(g) <= tol


def encode_image(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ===== Data loaders =====

_LOADED: dict = {}


def load_item(bench: str, idx: int) -> dict | None:
    """Returns {question, gold, image} for the given (bench, idx)."""
    if bench == "chartqa":
        if "chartqa" not in _LOADED:
            _LOADED["chartqa"] = load_dataset("HuggingFaceM4/ChartQA", split="test")
        ds = _LOADED["chartqa"]
        ex = ds[idx]
        return {"question": ex["query"],
                "gold": ex["label"][0] if isinstance(ex["label"], list) else ex["label"],
                "image": ex["image"]}
    elif bench.startswith("tablevqa/"):
        split = bench.split("/", 1)[1]
        key = f"tablevqa/{split}"
        if key not in _LOADED:
            _LOADED[key] = load_dataset("terryoo/TableVQA-Bench", split=split)
        ds = _LOADED[key]
        ex = ds[idx]
        return {"question": ex["question"], "gold": str(ex.get("gt","")), "image": ex["image"]}
    elif bench == "docvqa":
        if "docvqa" not in _LOADED:
            _LOADED["docvqa"] = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        ds = _LOADED["docvqa"]
        ex = ds[idx]
        golds = ex.get("answers") or []
        if isinstance(golds, str): golds = [golds]
        return {"question": ex["question"], "gold": golds[0] if golds else "",
                "image": ex["image"]}
    return None


# ===== Strategies =====

def strategy_single(item: dict) -> tuple[str, str]:
    """Single-shot VLM answer."""
    b64 = encode_image(item["image"])
    msg = [{"role": "user", "content": [
        {"type": "text", "text": f"{PROMPT_SINGLE}\n\nQuestion: {item['question']}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
    r = chat(msg)
    raw = (r["choices"][0]["message"].get("content") or "").strip()
    return parse_final(raw) or "", raw


def safe_eval(values: list, expr: str) -> float | None:
    """Evaluate expr (= 'sum(values) / 2' etc.) in restricted namespace."""
    try:
        ns = {"values": values, "sum": sum, "min": min, "max": max, "len": len,
              "abs": abs, "round": round}
        result = eval(expr, {"__builtins__": {}}, ns)
        return float(result)
    except Exception:
        return None


def strategy_subagent(item: dict) -> tuple[str, str]:
    """Stage1: VLM extracts values+expression; Stage2: Python computes; Stage3: VLM formats final."""
    b64 = encode_image(item["image"])

    # Stage 1: extract
    msg1 = [{"role": "user", "content": [
        {"type": "text", "text": f"{PROMPT_STAGE1_EXTRACT}\n\nQuestion: {item['question']}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
    r1 = chat(msg1, max_tokens=2048)
    raw1 = (r1["choices"][0]["message"].get("content") or "").strip()

    # Parse JSON
    jm = _JSON_OBJ_RE.findall(raw1)
    if not jm:
        return "", f"[Stage1 no JSON]\n{raw1[:200]}"
    parsed = None
    for s in reversed(jm):  # try last JSON first
        try:
            parsed = json.loads(s)
            break
        except json.JSONDecodeError:
            continue
    if not parsed:
        return "", f"[Stage1 JSON parse fail]\n{raw1[:200]}"

    values = parsed.get("values", [])
    expr = parsed.get("expression", "")
    if not isinstance(values, list) or not expr:
        return "", f"[Stage1 missing values/expr]\n{json.dumps(parsed)[:200]}"

    # Stage 2: compute
    computed = safe_eval(values, expr)
    if computed is None:
        return "", f"[Stage2 eval fail] values={values} expr={expr}"

    # Stage 3: format (= simple, since computed is already the answer for most aggregation Qs)
    # For most cases, we just return the computed value directly with simple formatting.
    fmt = parsed.get("answer_format", "number")
    if fmt == "percent":
        return f"{computed}", f"[Stage1 JSON] {json.dumps(parsed)[:200]}\n[Stage2 result] {computed}"
    if fmt == "currency":
        return f"{computed:.2f}", f"[Stage1 JSON] {json.dumps(parsed)[:200]}\n[Stage2 result] {computed}"
    return str(computed), f"[Stage1 JSON] {json.dumps(parsed)[:200]}\n[Stage2 result] {computed}"


# ===== Bench runner =====

def run_one(bench: str, idx: int, gold_str: str, strategy_name: str) -> dict:
    t0 = time.time()
    item = load_item(bench, idx)
    if not item:
        return {"id": f"{bench}/{idx}", "strategy": strategy_name,
                "error": "load failed", "ok": False, "elapsed": time.time() - t0}
    try:
        if strategy_name == "single":
            pred, raw = strategy_single(item)
        else:
            pred, raw = strategy_subagent(item)
        ok = match_num(pred, gold_str)
        return {"id": f"{bench}/{idx}", "strategy": strategy_name,
                "q": item["question"][:120], "gold": gold_str,
                "pred": pred[:120], "raw": raw[:400],
                "ok": ok, "elapsed": time.time() - t0}
    except Exception as e:
        return {"id": f"{bench}/{idx}", "strategy": strategy_name,
                "error": str(e)[:160], "ok": False, "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="artifacts/aggregation_subagent_bench.json")
    args = ap.parse_args()

    tasks = []
    for bench, idx, gold in CURATED:
        for strat in ("single", "subagent"):
            tasks.append((bench, idx, gold, strat))
    print(f"{len(CURATED)} curated questions × 2 strategies = {len(tasks)} runs", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, b, i, g, s): (b, i, s) for (b, i, g, s) in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            print(f"[{i:2d}/{len(tasks)}] {mark} {r['id']:<28} {r['strategy']:<10} "
                  f"gold={r.get('gold','')[:15]:<15} pred={r.get('pred','')[:20]:<20} {err}", flush=True)

    elapsed = time.time() - t_start

    # Per-strategy summary
    for strat in ("single", "subagent"):
        sub = [r for r in results if r["strategy"] == strat]
        n_ok = sum(1 for r in sub if r.get("ok"))
        print(f"  {strat:<10}: {n_ok}/{len(sub)} = {n_ok/len(sub):.3f}")

    # Per-question side-by-side
    print()
    print("Per-question single vs subagent:")
    for bench, idx, gold in CURATED:
        s = next((r for r in results if r["id"] == f"{bench}/{idx}" and r["strategy"] == "single"), None)
        a = next((r for r in results if r["id"] == f"{bench}/{idx}" and r["strategy"] == "subagent"), None)
        mark_s = "✓" if s and s.get("ok") else "✗"
        mark_a = "✓" if a and a.get("ok") else "✗"
        print(f"  {mark_s} single  vs {mark_a} subagent  | {bench}/{idx} gold={gold:<15} "
              f"s_pred={(s.get('pred','')[:18] if s else 'NA'):<18} a_pred={a.get('pred','')[:18] if a else 'NA'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n": len(results), "elapsed": elapsed,
                                    "results": results}, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
