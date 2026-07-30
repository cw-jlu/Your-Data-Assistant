"""Video-MME (short) bench on gpuhost Qwen3.5-35B-A3B via frame-list.

Approach:
  1. Sample N "short" videos from lmms-lab/Video-MME (= multiple-choice).
  2. yt-dlp each to local mp4 (cached).
  3. cv2 extract n_frames frames evenly across the video.
  4. Send frames as image list with the question + options.
  5. Compare predicted letter to gold (A/B/C/D).
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

import cv2
import requests
from datasets import load_dataset
from PIL import Image

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"

VIDEO_CACHE = Path("/tmp/vme_videos")
VIDEO_CACHE.mkdir(parents=True, exist_ok=True)

PROMPT = """You are watching a sequence of frames sampled evenly from a video.
Answer the multiple-choice question by selecting the best option.

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
            env={**os.environ},
        )
        if out.is_file() and out.stat().st_size > 0:
            return out
        print(f"  [DL FAIL] {video_id}: rc={r.returncode} stderr={r.stderr[-200:]}", flush=True)
        return None
    except subprocess.TimeoutExpired:
        return None


def extract_frames(video_path: Path, n_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // n_frames)
    frames = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total - 1))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # downscale to limit token budget
        img.thumbnail((640, 360))
        frames.append(img)
    cap.release()
    return frames


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*([A-D])", re.IGNORECASE | re.MULTILINE)


def chat_with_retry(messages: list, max_tokens: int = 8192, max_retries: int = 3,
                    http_timeout: int = 240, temperature: float = 0.6) -> dict:
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
    # fallback: any leading letter
    m2 = re.search(r"\b([A-D])\b", content[-100:])
    return m2.group(1).upper() if m2 else None


def run_one(qid: str, item: dict, n_frames: int) -> dict:
    t0 = time.time()
    try:
        # Download video if needed
        path = yt_download(item["videoID"], item["url"])
        if not path:
            return {"qid": qid, "error": "download failed", "ok": False, "elapsed": time.time() - t0}

        frames = extract_frames(path, n_frames)
        if not frames:
            return {"qid": qid, "error": "frame extract failed", "ok": False, "elapsed": time.time() - t0}

        # Build prompt
        options_txt = "\n".join(item["options"])
        text = (f"{PROMPT}\n\nQuestion: {item['question']}\n\n"
                f"Options:\n{options_txt}\n\n"
                f"({len(frames)} frames sampled evenly across the video)")
        content = [{"type": "text", "text": text}]
        for fr in frames:
            b64 = encode_image(fr)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        r = chat_with_retry([{"role": "user", "content": content}])
        msg = r["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        pred = parse_letter(raw)
        gold = item["answer"].strip().upper()
        ok = pred == gold
        return {
            "qid": qid, "videoID": item["videoID"], "n_frames": len(frames),
            "q": item["question"][:120], "gold": gold,
            "pred": pred or "", "raw": raw[:280],
            "ok": ok, "elapsed": time.time() - t0,
            "finish_reason": r["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"qid": qid, "error": str(e)[:160], "ok": False,
                "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of questions")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)  # video is heavy
    ap.add_argument("--duration", default="short", choices=["short", "medium", "long"])
    ap.add_argument("--out", default="artifacts/videomme_bench.json")
    args = ap.parse_args()

    print(f"sampling {args.n} {args.duration} videos from Video-MME...", flush=True)
    ds = load_dataset("lmms-lab/Video-MME", split="test", streaming=True)
    items = []
    for ex in ds:
        if ex["duration"] != args.duration:
            continue
        items.append({
            "qid": ex["question_id"], "videoID": ex["videoID"], "url": ex["url"],
            "question": ex["question"], "options": ex["options"], "answer": ex["answer"],
        })
        if len(items) >= args.n * 3:  # over-sample, in case some downloads fail
            break

    random.seed(args.seed)
    random.shuffle(items)
    items = items[:args.n]
    print(f"picked {len(items)} items", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, it["qid"], it, args.n_frames): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:2d}/{len(items)}] {mark} qid={r['qid']:<10} "
                  f"gold={r.get('gold','?')} pred={r.get('pred','?')} "
                  f"frames={r.get('n_frames','?')} acc={acc:.3f} {err}", flush=True)

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
        "n_frames": args.n_frames, "duration": args.duration,
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
