"""Video-MME bench v2 — native video_url (mp4 base64 data URI).

v1 lesson: manual 8-frame sampling gave clean 0.60. Qwen3.5 default is fps=2
(60 frames for 30s short). Switching to native video_url lets the model
internally sample at its preferred rate.
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

VIDEO_CACHE = Path("/tmp/vme_videos")
VIDEO_CACHE.mkdir(parents=True, exist_ok=True)

PROMPT = """Watch the video and answer the multiple-choice question.

After your reasoning, output ONLY the letter on the last line as:
ANSWER: <A|B|C|D>"""


def yt_download(video_id: str, url: str) -> Path | None:
    out = VIDEO_CACHE / f"{video_id}.mp4"
    if out.is_file() and out.stat().st_size > 0:
        return out
    try:
        r = subprocess.run(
            ["python3", "-m", "yt_dlp", "-q", "--no-warnings",
             "-f", "best[height<=480]/best", "--max-filesize", "60M",
             "-o", str(out), url],
            capture_output=True, text=True, timeout=120,
        )
        return out if out.is_file() and out.stat().st_size > 0 else None
    except subprocess.TimeoutExpired:
        return None


_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*([A-D])", re.IGNORECASE | re.MULTILINE)


def chat_with_retry(messages: list, max_tokens: int = 8192, max_retries: int = 2,
                    http_timeout: int = 300, temperature: float = 0.6) -> dict:
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


def parse_letter(content: str) -> str | None:
    if not content:
        return None
    m = _ANSWER_LINE_RE.search(content)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([A-D])\b", content[-100:])
    return m2.group(1).upper() if m2 else None


def run_one(qid: str, item: dict) -> dict:
    t0 = time.time()
    try:
        path = yt_download(item["videoID"], item["url"])
        if not path:
            return {"qid": qid, "error": "download failed", "ok": False,
                    "elapsed": time.time() - t0}

        with open(path, "rb") as f:
            mp4_b64 = base64.b64encode(f.read()).decode()

        options_txt = "\n".join(item["options"])
        text = (f"{PROMPT}\n\nQuestion: {item['question']}\n\n"
                f"Options:\n{options_txt}")
        content = [
            {"type": "text", "text": text},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{mp4_b64}"}},
        ]

        r = chat_with_retry([{"role": "user", "content": content}])
        msg = r["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        pred = parse_letter(raw)
        gold = item["answer"].strip().upper()
        ok = pred == gold
        return {
            "qid": qid, "videoID": item["videoID"],
            "video_bytes": path.stat().st_size,
            "q": item["question"][:120], "gold": gold, "pred": pred or "",
            "raw": raw[:280], "ok": ok, "elapsed": time.time() - t0,
            "finish_reason": r["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"qid": qid, "error": str(e)[:160], "ok": False,
                "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--duration", default="short", choices=["short", "medium", "long"])
    ap.add_argument("--out", default="artifacts/videomme_bench_v2_native.json")
    args = ap.parse_args()

    print(f"sampling {args.n} {args.duration} from Video-MME...", flush=True)
    ds = load_dataset("lmms-lab/Video-MME", split="test", streaming=True)
    items = []
    for ex in ds:
        if ex["duration"] != args.duration:
            continue
        items.append({
            "qid": ex["question_id"], "videoID": ex["videoID"], "url": ex["url"],
            "question": ex["question"], "options": ex["options"], "answer": ex["answer"],
        })
        if len(items) >= args.n * 3:
            break

    random.seed(args.seed)
    random.shuffle(items)
    items = items[:args.n]
    print(f"picked {len(items)} items", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, it["qid"], it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:2d}/{len(items)}] {mark} qid={r['qid']:<10} "
                  f"gold={r.get('gold','?')} pred={r.get('pred','?')} "
                  f"bytes={r.get('video_bytes','?')} acc={acc:.3f} {err}", flush=True)

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
        "mode": "native_video_url_base64",
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
