"""Universal VLM bench — mixed ChartQA / TableVQA / DocVQA / Video-MME with
ONE shared prompt. Tests whether a single prompt generalizes across
modalities (image vs video, chart vs table vs document vs scene).

Goal: validate that Phase 2 can be served by ONE harness that doesn't
know in advance what kind of data is incoming.
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
from PIL import Image

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"

VIDEO_CACHE = Path("/tmp/vme_videos")
VIDEO_CACHE.mkdir(parents=True, exist_ok=True)


# ONE prompt covering all modalities, no source-specific hints.
UNIVERSAL_PROMPT = """Look at the data (images or video) and answer the question.

Decide the answer type first:
- TEXT/LABEL: the question asks WHICH item / WHAT name / a yes-no.
  Return the EXACT text visible in the source (verbatim including symbols,
  case, trailing markers like *, ?, units).
  For true/false questions, output exactly "1" for true, "0" for false.
- NUMBER: the question asks HOW MANY / WHAT VALUE / a percentage / ratio.
  Return only the digits (no units, no %).
- LIST: multiple values, comma-separated in source order.
- MULTIPLE_CHOICE: if the question has labelled options (A/B/C/D), output
  ONLY the letter.

After your reasoning, output ONLY the final answer on the last line as:
ANSWER: <value>"""


def encode_image(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=85)
    else:
        img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_ANSWER_LINE_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"ANSWER\s*:\s*([A-D])\b", re.IGNORECASE | re.MULTILINE)


def chat_with_retry(messages: list, max_tokens: int = 8192, max_retries: int = 3,
                    http_timeout: int = 240, temperature: float = 0.6) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_KEY}",
               "CF-Access-Client-Id": CF_ID, "CF-Access-Client-Secret": CF_SECRET}
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


def parse_final(content: str) -> str | None:
    if not content:
        return None
    m = _ANSWER_LINE_RE.search(content)
    return m.group(1).strip() if m else None


def parse_letter(content: str) -> str | None:
    m = _LETTER_RE.search(content)
    if m:
        return m.group(1).upper()
    if content:
        m2 = re.search(r"\b([A-D])\b", content[-100:])
        return m2.group(1).upper() if m2 else None
    return None


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


def match(pred: str, golds: list[str], tol: float = 0.05) -> bool:
    if not pred:
        return False
    p_num = parse_number(pred)
    pl = pred.strip().lower().rstrip(".")
    for g in golds:
        if not g:
            continue
        gl = g.strip().lower().rstrip(".")
        g_num = parse_number(g)
        if g_num is not None and p_num is not None:
            if abs(g_num) < 1e-9 and abs(p_num) < 1e-6:
                return True
            if abs(g_num) >= 1e-9 and abs(p_num - g_num) / abs(g_num) <= tol:
                return True
        if pl == gl or gl in pl or pl in gl:
            return True
    return False


# --- per-source loaders ---

def load_chartqa(n: int, seed: int) -> list[dict]:
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    random.seed(seed)
    idxs = random.sample(range(len(ds)), n)
    out = []
    for i in idxs:
        ex = ds[i]
        out.append({"source": "chartqa", "qid": f"chartqa-{i}", "image": ex["image"],
                    "video_path": None, "question": ex["query"],
                    "golds": [ex["label"][0] if isinstance(ex["label"], list) else ex["label"]],
                    "format": "value"})
    return out


def load_tablevqa(n_per: int, seed: int) -> list[dict]:
    out = []
    for split in ["fintabnetqa", "vtabfact", "vwtq", "vwtq_syn"]:
        ds = load_dataset("terryoo/TableVQA-Bench", split=split)
        random.seed(seed)
        idxs = random.sample(range(len(ds)), min(n_per, len(ds)))
        for i in idxs:
            ex = ds[i]
            out.append({"source": f"tablevqa/{split}", "qid": f"table-{split}-{i}",
                        "image": ex["image"], "video_path": None,
                        "question": ex["question"],
                        "golds": [str(ex.get("gt", "")).strip()],
                        "format": "value"})
    return out


def load_docvqa(n: int, seed: int) -> list[dict]:
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    random.seed(seed + 1)  # different seed to avoid overlap
    idxs = random.sample(range(len(ds)), n)
    out = []
    for i in idxs:
        ex = ds[i]
        golds = ex.get("answers") or []
        if isinstance(golds, str):
            golds = [golds]
        out.append({"source": "docvqa", "qid": f"docvqa-{i}", "image": ex["image"],
                    "video_path": None, "question": ex["question"],
                    "golds": [str(g) for g in golds], "format": "value"})
    return out


def load_videomme(n: int, seed: int) -> list[dict]:
    ds = load_dataset("lmms-lab/Video-MME", split="test", streaming=True)
    items = []
    for ex in ds:
        if ex["duration"] != "short":
            continue
        items.append(ex)
        if len(items) >= n * 3:
            break
    random.seed(seed + 2)
    random.shuffle(items)
    out = []
    for ex in items:
        if len(out) >= n:
            break
        vid = VIDEO_CACHE / f"{ex['videoID']}.mp4"
        if not vid.is_file():
            try:
                subprocess.run(
                    ["python3", "-m", "yt_dlp", "-q", "--no-warnings",
                     "-f", "best[height<=480]/best", "--max-filesize", "60M",
                     "-o", str(vid), ex["url"]],
                    capture_output=True, text=True, timeout=120)
            except Exception:
                continue
        if not vid.is_file():
            continue
        options_txt = "\n".join(ex["options"])
        q = f"{ex['question']}\n\nOptions:\n{options_txt}"
        out.append({"source": "video-mme", "qid": f"vme-{ex['question_id']}",
                    "image": None, "video_path": vid,
                    "question": q, "golds": [ex["answer"].strip().upper()],
                    "format": "multiple_choice"})
    return out


def run_one(item: dict) -> dict:
    t0 = time.time()
    try:
        content = [{"type": "text", "text": f"{UNIVERSAL_PROMPT}\n\nQuestion: {item['question']}"}]
        if item["image"] is not None:
            b64 = encode_image(item["image"], fmt="PNG")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}})
        elif item["video_path"] is not None:
            with open(item["video_path"], "rb") as f:
                mp4 = base64.b64encode(f.read()).decode()
            content.append({"type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{mp4}"}})
        else:
            return {"qid": item["qid"], "source": item["source"],
                    "error": "no image or video", "ok": False, "elapsed": time.time() - t0}

        r = chat_with_retry([{"role": "user", "content": content}])
        msg = r["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        if item["format"] == "multiple_choice":
            pred = parse_letter(raw)
            ok = pred is not None and pred == item["golds"][0]
        else:
            pred = parse_final(raw)
            ok = match(pred or "", item["golds"])
        return {"qid": item["qid"], "source": item["source"],
                "question": item["question"][:120],
                "gold": "|".join(item["golds"])[:80], "pred": (pred or "")[:80],
                "raw": raw[:280], "ok": ok, "elapsed": time.time() - t0,
                "finish_reason": r["choices"][0]["finish_reason"]}
    except Exception as e:
        return {"qid": item["qid"], "source": item["source"],
                "error": str(e)[:160], "ok": False, "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chartqa", type=int, default=10)
    ap.add_argument("--n-tablevqa-per", type=int, default=3)  # 3*4=12
    ap.add_argument("--n-docvqa", type=int, default=10)
    ap.add_argument("--n-videomme", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="artifacts/universal_bench.json")
    args = ap.parse_args()

    print("loading mixed items...", flush=True)
    items = []
    items += load_chartqa(args.n_chartqa, args.seed)
    items += load_tablevqa(args.n_tablevqa_per, args.seed)
    items += load_docvqa(args.n_docvqa, args.seed)
    items += load_videomme(args.n_videomme, args.seed)
    random.seed(args.seed)
    random.shuffle(items)
    print(f"total {len(items)} items, workers={args.workers}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            acc = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:3d}/{len(items)}] {mark} {r['source']:<22} "
                  f"gold={str(r.get('gold',''))[:25]:<25} pred={str(r.get('pred',''))[:25]:<25} "
                  f"acc={acc:.3f} {err}", flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for x in results if x.get("ok"))
    n_err = sum(1 for x in results if x.get("error"))

    # per-source breakdown
    per_src = {}
    for r in results:
        s = r["source"]
        per_src.setdefault(s, {"n": 0, "ok": 0, "err": 0})
        per_src[s]["n"] += 1
        if r.get("ok"):
            per_src[s]["ok"] += 1
        if r.get("error"):
            per_src[s]["err"] += 1

    print()
    print(f"=== DONE n={len(results)} accuracy={n_ok/len(results):.4f} errors={n_err} elapsed={elapsed:.1f}s ===")
    for s, st in sorted(per_src.items()):
        print(f"   {s:<22} n={st['n']} acc={st['ok']/st['n']:.3f} err={st['err']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": len(results), "accuracy": n_ok / len(results), "errors": n_err,
        "elapsed_seconds": elapsed, "per_source": per_src,
        "results": results,
    }, indent=2, default=str))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
