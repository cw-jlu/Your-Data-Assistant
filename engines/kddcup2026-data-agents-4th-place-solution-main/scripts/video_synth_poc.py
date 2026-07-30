"""Video QA POC — synthetic frame sequences mimicking 'data video' tasks.

Since Phase 2 'data videos' have no public bench equivalent, we generate
synthetic frame sequences (= a chart evolving over time, a table being
populated, etc.) and ask the model temporal questions.

This validates that:
  - gpuhost endpoint accepts multi-image (= frame list) input
  - the model can do temporal reasoning across frames
  - our prompt style transfers from image-QA to video-QA
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
from PIL import Image, ImageDraw

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"

PROMPT = """You are watching a sequence of frames from a short data-analysis
video. Each image is a consecutive frame.

Decide the answer type first:
- NUMBER: return digits only (no units).
- TEXT: return the exact label/word.

After reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


def synth_growing_bar(n_frames: int, values: list[int]) -> list[Image.Image]:
    """A bar chart where the bar's height changes over frames."""
    out = []
    for v in values:
        img = Image.new("RGB", (320, 200), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([60, 200 - v * 1.5, 140, 200], fill="steelblue")
        d.text((60, 5), f"Value: {v}", fill="black")
        out.append(img)
    return out


def synth_table_population(n_rows_per_frame: list[int]) -> list[Image.Image]:
    """A table that gains rows over frames."""
    rows_data = [("Alice", 10), ("Bob", 25), ("Carol", 15), ("Dave", 40), ("Eve", 30)]
    out = []
    for n in n_rows_per_frame:
        img = Image.new("RGB", (320, 200), "white")
        d = ImageDraw.Draw(img)
        d.text((10, 5), "Name      Score", fill="black")
        for i, (name, score) in enumerate(rows_data[:n]):
            d.text((10, 25 + i * 22), f"{name:<10}{score}", fill="black")
        out.append(img)
    return out


def make_tasks(seed: int) -> list[dict]:
    """Generate synthetic video-QA tasks. Universal patterns only — no test leak."""
    random.seed(seed)
    tasks = []

    # T1: growing bar — final value
    vals = [10, 30, 60, 90]
    tasks.append({
        "id": "bar_final",
        "frames": synth_growing_bar(4, vals),
        "q": "What is the final value of the bar in the last frame?",
        "gold": "90",
    })
    # T2: growing bar — count frames where value > 50
    tasks.append({
        "id": "bar_count_above_50",
        "frames": synth_growing_bar(4, vals),
        "q": "In how many frames is the value greater than 50?",
        "gold": "2",
    })
    # T3: table growth — final row count
    tasks.append({
        "id": "table_count",
        "frames": synth_table_population([1, 3, 5]),
        "q": "How many rows does the table have in the last frame?",
        "gold": "5",
    })
    # T4: table growth — Eve's score
    tasks.append({
        "id": "table_eve_score",
        "frames": synth_table_population([1, 3, 5]),
        "q": "What score does Eve have in the table?",
        "gold": "30",
    })
    # T5: bar change direction
    vals2 = [80, 50, 20, 5]
    tasks.append({
        "id": "bar_decreasing",
        "frames": synth_growing_bar(4, vals2),
        "q": "Across the frames, is the bar increasing or decreasing? Answer one word.",
        "gold": "decreasing",
    })
    # T6: bar — sum of last 2 values
    vals3 = [10, 20, 35, 45]
    tasks.append({
        "id": "bar_sum_last2",
        "frames": synth_growing_bar(4, vals3),
        "q": "Sum the values shown in the last two frames.",
        "gold": "80",
    })
    # T7: table — which name appears first (= row 1)
    tasks.append({
        "id": "table_first_name",
        "frames": synth_table_population([1, 3, 5]),
        "q": "What name is in the first row?",
        "gold": "Alice",
    })
    # T8: bar growth amount frame-to-frame
    tasks.append({
        "id": "bar_first_to_last_delta",
        "frames": synth_growing_bar(4, vals),
        "q": "By how much did the bar grow from the first frame to the last?",
        "gold": "80",  # 90-10
    })
    return tasks


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def chat_with_retry(messages: list, max_tokens: int = 4096, max_retries: int = 3,
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
    s = s.strip().replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def match(pred: str, gold: str, tol: float = 0.05) -> bool:
    if not pred:
        return False
    p_num = parse_number(pred)
    g_num = parse_number(gold)
    if g_num is not None and p_num is not None:
        if abs(g_num) < 1e-9:
            return abs(p_num) < 1e-6
        return abs(p_num - g_num) / abs(g_num) <= tol
    return pred.strip().lower() == gold.strip().lower()


def run_task(task: dict) -> dict:
    t0 = time.time()
    try:
        content = [{"type": "text", "text": f"{PROMPT}\n\nQuestion: {task['q']}"}]
        for i, frame in enumerate(task["frames"], 1):
            b64 = encode_image(frame)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        messages = [{"role": "user", "content": content}]
        r = chat_with_retry(messages)
        msg = r["choices"][0]["message"]
        cont = (msg.get("content") or "").strip()
        final = parse_final_answer(cont)
        ok = match(final or "", task["gold"])
        return {
            "id": task["id"], "q": task["q"][:120], "gold": task["gold"],
            "pred": (final or "")[:120], "raw": cont[:280],
            "n_frames": len(task["frames"]),
            "ok": ok, "elapsed": time.time() - t0,
            "finish_reason": r["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"id": task["id"], "error": str(e)[:160], "ok": False,
                "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="artifacts/video_synth_poc.json")
    args = ap.parse_args()

    tasks = make_tasks(args.seed)
    print(f"generated {len(tasks)} synthetic video tasks, workers={args.workers}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:2d}/{len(tasks)}] {mark} id={r['id']:<25} "
                  f"gold={str(r.get('gold',''))[:15]:<15} pred={str(r.get('pred',''))[:25]:<25} "
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
