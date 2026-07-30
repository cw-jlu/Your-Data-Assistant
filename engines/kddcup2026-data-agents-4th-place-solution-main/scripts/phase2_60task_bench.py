"""Phase 2 demo (60 tasks) bench — exp_142_phase2_baseline v1.

Settings:
  - att = 1 (single attempt, no vote)
  - workers = 12 parallel
  - max_steps = 64 (matches Phase 1 spec)
  - T = 0.6, top_p = 0.95, top_k = 20, presence_penalty = 0 (Qwen3.5 official)
  - Inline video attach (= mp4 base64 in PLAN turn, no sub-agent)
  - knowledge.md preamble (= Phase 1 same)

Per-task artifacts:
  artifacts/runs/exp_142_phase2_baseline_NNN/<task_id>/
    ├── prediction.csv
    ├── result.json    (= compare metrics + finish reason + elapsed)
    └── trace.json     (= full message history + actions + SQL results)

Run-level:
  summary.json with mean exact-match accuracy + per-task pass/fail
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Use Phase 1 v11's DuckDB unified data layer to handle CSV / SQLite / JSON.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from experiments.exp_137_math_advisor.tools import duckdb_unified  # noqa: E402

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"

DEMO_ROOT = Path("data/phase2_demo/demo_samples_phase2")
RUNS_ROOT = Path("artifacts/runs")
EXP_NAME = "exp_142_phase2_baseline"


SYSTEM_PROMPT = """You are a data analyst agent. Solve the task using the
provided knowledge guide, structured files, and optional briefing video.

Decision steps:
0. If any video file is provided in context/, watch it FIRST to extract
   task-specific criteria (= thresholds, time windows, scope filters,
   business phrases). Quote them verbatim before deciding columns.
1. Read the knowledge guide to understand schema and field semantics.
2. Plan the answer columns and aggregation.
3. Use execute_sql to query the SQLite database at context/db/sub_db.sqlite
   (read-only). Iterate until the result matches the question.
4. When confident, call answer_csv with the final CSV result.

Output ONE JSON action per turn, wrapped in a single ```json fenced block:
  {"action": "execute_sql", "action_input": {"sql": "SELECT ..."}}
  or
  {"action": "answer_csv", "action_input": {"csv": "col1,col2\\nval1,val2"}}

The answer_csv VALUE must include a header row and all data rows. Do not
wrap the CSV in code fences.
"""


_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def encode_video(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def chat(messages: list, max_tokens: int = 8192, temperature: float = 0.6,
         max_retries: int = 3, http_timeout: int = 300) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0.0}
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


def parse_action(content: str) -> dict | None:
    m = _JSON_BLOCK_RE.search(content or "")
    if m:
        raw = m.group(1).strip()
    else:
        m2 = re.search(r"\{[^{}]*\"action\"[^{}]*\}", content or "", re.DOTALL)
        if not m2:
            return None
        raw = m2.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def execute_sql(db_path: Path, sql: str, max_rows: int = 50) -> str:
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
            out = io.StringIO()
            w = csv.writer(out)
            if cols:
                w.writerow(cols)
            for r in rows:
                w.writerow(r)
            text = out.getvalue().strip()
            extra = f"\n(truncated at {max_rows} rows)" if len(rows) == max_rows else ""
            return f"OK:\n{text}{extra}"
    except sqlite3.Error as e:
        return f"SQL ERROR: {e}"


def parse_csv_text(text: str) -> tuple[list[str], list[list[str]]]:
    if not text:
        return [], []
    rdr = csv.reader(io.StringIO(text))
    rows = list(rdr)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def compare_csv(pred_text: str, gold_text: str) -> dict:
    p_cols, p_rows = parse_csv_text(pred_text)
    g_cols, g_rows = parse_csv_text(gold_text)
    def norm(s):
        return s.strip().lower()
    p_cells = set(tuple(norm(c) for c in r) for r in p_rows)
    g_cells = set(tuple(norm(c) for c in r) for r in g_rows)
    return {
        "pred_cols": p_cols, "gold_cols": g_cols,
        "pred_rows": len(p_rows), "gold_rows": len(g_rows),
        "rows_intersect": len(p_cells & g_cells),
        "rows_only_pred": len(p_cells - g_cells),
        "rows_only_gold": len(g_cells - p_cells),
        "exact_match": p_cells == g_cells and p_cols == g_cols,
    }


def find_db_path(task_dir: Path) -> Path | None:
    cands = sorted((task_dir / "context").glob("**/*.sqlite"))
    if cands:
        return cands[0]
    cands = sorted((task_dir / "context").glob("**/*.db"))
    return cands[0] if cands else None


def run_one(task_id: str, run_dir: Path, max_steps: int) -> dict:
    t0 = time.time()
    task_dir = DEMO_ROOT / "input" / task_id
    gold_path = DEMO_ROOT / "output" / task_id / "gold.csv"
    out_dir = run_dir / task_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = {"task_id": task_id, "messages": [], "steps": []}

    try:
        meta = json.load(open(task_dir / "task.json"))
        question = meta["question"]
        knowledge = (task_dir / "context" / "knowledge.md").read_text() if (task_dir / "context" / "knowledge.md").is_file() else ""
        db_path = find_db_path(task_dir)
        if not db_path:
            return {"task_id": task_id, "ok": False, "error": "no db found",
                    "elapsed": time.time() - t0}
        video_files = sorted((task_dir / "context").glob("video/*.mp4"))
        video_path = video_files[0] if video_files else None

        content = [{"type": "text",
                    "text": f"# Knowledge Guide\n\n{knowledge}\n\n# Question\n\n{question}"}]
        if video_path:
            content.append({"type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{encode_video(video_path)}"}})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        trace["has_video"] = video_path is not None
        trace["db_path"] = str(db_path.relative_to(task_dir))

        pred_csv = None
        finish = None
        for step in range(1, max_steps + 1):
            t_step = time.time()
            r = chat(messages)
            msg = r["choices"][0]["message"]
            raw = (msg.get("content") or "").strip()
            thinking = (msg.get("reasoning") or "")
            finish = r["choices"][0]["finish_reason"]
            action = parse_action(raw)

            step_info = {
                "step": step,
                "elapsed": time.time() - t_step,
                "finish_reason": finish,
                "thinking_chars": len(thinking),
                "content_chars": len(raw),
                "action": action.get("action") if action else None,
            }

            if not action:
                step_info["error"] = "parse_action failed"
                step_info["raw_tail"] = raw[-200:]
                trace["steps"].append(step_info)
                break

            a_type = action["action"]
            a_input = action.get("action_input", {})

            if a_type == "execute_sql":
                sql = a_input.get("sql", "")
                step_info["sql"] = sql[:400]
                result = execute_sql(db_path, sql)
                step_info["result_head"] = result[:400]
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",
                                 "content": f"TOOL execute_sql result:\n{result}\n\nContinue."})
            elif a_type == "answer_csv":
                pred_csv = a_input.get("csv", "").strip()
                step_info["pred_csv_head"] = pred_csv[:300]
                trace["steps"].append(step_info)
                break
            else:
                step_info["error"] = f"unknown action: {a_type}"
                trace["steps"].append(step_info)
                break

            trace["steps"].append(step_info)

        elapsed = time.time() - t0
        gold_text = gold_path.read_text() if gold_path.is_file() else ""

        result = {
            "task_id": task_id, "elapsed": elapsed,
            "n_steps": len(trace["steps"]),
            "finish_reason": finish,
            "has_video": video_path is not None,
            "pred_csv_received": pred_csv is not None,
        }
        if pred_csv is not None:
            (out_dir / "prediction.csv").write_text(pred_csv)
            cmp = compare_csv(pred_csv, gold_text)
            result.update({"ok": cmp["exact_match"], "compare": cmp})
        else:
            result["ok"] = False
            result["error"] = "no answer_csv emitted"

        trace["gold"] = gold_text[:500]
        trace["pred"] = (pred_csv or "")[:500]
        (out_dir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as e:
        trace["fatal_error"] = str(e)
        (out_dir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        return {"task_id": task_id, "ok": False, "error": str(e)[:200],
                "elapsed": time.time() - t0}


def next_run_id() -> str:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in RUNS_ROOT.glob(f"{EXP_NAME}_*")
                      if p.name.startswith(f"{EXP_NAME}_"))
    next_n = 1
    if existing:
        last_n = int(existing[-1].rsplit("_", 1)[-1])
        next_n = last_n + 1
    return f"{EXP_NAME}_{next_n:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 60 tasks")
    args = ap.parse_args()

    task_ids = sorted([p.name for p in (DEMO_ROOT / "input").iterdir() if p.is_dir()],
                      key=lambda s: int(s.split("_")[1]))
    if args.limit > 0:
        task_ids = task_ids[:args.limit]

    run_id = next_run_id()
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {EXP_NAME} run_id={run_id} ===")
    print(f"  tasks: {len(task_ids)}, workers={args.workers}, max_steps={args.max_steps}")
    print(f"  T=0.6, top_p=0.95, top_k=20, pp=0; attempts=1")
    print(f"  output: {run_dir}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        futs = {exr.submit(run_one, tid, run_dir, args.max_steps): tid for tid in task_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✓" if r.get("ok") else "✗"
            err = (r.get("error") or "")[:50]
            running = sum(1 for x in results if x.get("ok")) / len(results)
            print(f"[{i:3d}/{len(task_ids)}] {mark} {r['task_id']:<10} "
                  f"steps={r.get('n_steps','?')} t={r.get('elapsed',0):.0f}s "
                  f"acc={running:.3f} {err}", flush=True)

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("ok"))
    summary = {
        "run_id": run_id, "n_tasks": len(results),
        "n_perfect": n_ok, "accuracy": n_ok / len(results) if results else 0,
        "elapsed_seconds": elapsed,
        "config": {"workers": args.workers, "max_steps": args.max_steps,
                   "n_attempts": 1, "temperature": 0.6, "model": MODEL,
                   "endpoint": API_BASE},
        "per_task": [{k: v for k, v in r.items() if k != "compare"} for r in results],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"=== DONE accuracy={n_ok/len(results):.4f} perfect={n_ok}/{len(results)} "
          f"elapsed={elapsed:.0f}s ===")
    print(f"saved {run_dir}/summary.json")


if __name__ == "__main__":
    main()
