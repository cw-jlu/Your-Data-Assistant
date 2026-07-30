"""Phase 2 spike — end-to-end smoke test on task_1 (video + db + knowledge).

Minimal ReAct loop:
  user message (= prompt + knowledge.md + question + video_url) → model
    → action: execute_sql / answer_csv
  ↻ until answer_csv or max_steps reached

Compares prediction against gold.csv visually + cell-wise.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"

DEMO_ROOT = Path("data/phase2_demo/demo_samples_phase2")


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


def encode_video(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def chat(messages: list, max_tokens: int = 8192, temperature: float = 0.6) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0.0}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_KEY}",
               "CF-Access-Client-Id": CF_ID, "CF-Access-Client-Secret": CF_SECRET}
    r = requests.post(f"{API_BASE}/chat/completions",
                      headers=headers, json=body, timeout=300)
    r.raise_for_status()
    return r.json()


_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def parse_action(content: str) -> dict | None:
    m = _JSON_BLOCK_RE.search(content or "")
    if not m:
        # try inline JSON object
        m2 = re.search(r"\{[^{}]*\"action\"[^{}]*\}", content or "", re.DOTALL)
        if not m2:
            return None
        raw = m2.group(0)
    else:
        raw = m.group(1).strip()
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
    rdr = csv.reader(io.StringIO(text))
    rows = list(rdr)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def compare_csv(pred_text: str, gold_text: str) -> dict:
    p_cols, p_rows = parse_csv_text(pred_text)
    g_cols, g_rows = parse_csv_text(gold_text)
    # Normalize cells: strip + lowercase for text
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


def run_task(task_id: str, max_steps: int = 8) -> dict:
    task_dir = DEMO_ROOT / "input" / task_id
    gold_path = DEMO_ROOT / "output" / task_id / "gold.csv"

    task_meta = json.load(open(task_dir / "task.json"))
    question = task_meta["question"]
    knowledge = (task_dir / "context" / "knowledge.md").read_text()
    db_path = task_dir / "context" / "db" / "sub_db.sqlite"
    if not db_path.is_file():
        # discover any .sqlite/.db
        cands = list((task_dir / "context").glob("**/*.sqlite")) + list((task_dir / "context").glob("**/*.db"))
        if cands:
            db_path = cands[0]
        else:
            return {"error": "no sqlite/db found"}

    video_files = sorted((task_dir / "context").glob("video/*.mp4"))
    video_b64 = encode_video(video_files[0]) if video_files else None

    print(f"=== {task_id} ===", flush=True)
    print(f"  question: {question[:140]}", flush=True)
    print(f"  knowledge: {len(knowledge)} chars, db: {db_path.name}, "
          f"video: {video_files[0].name if video_files else 'none'}", flush=True)

    content = [{"type": "text",
                "text": f"# Knowledge Guide\n\n{knowledge}\n\n# Question\n\n{question}"}]
    if video_b64:
        content.append({"type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    pred_csv = None
    for step in range(1, max_steps + 1):
        t0 = time.time()
        r = chat(messages)
        elapsed = time.time() - t0
        msg = r["choices"][0]["message"]
        raw = (msg.get("content") or "").strip()
        thinking = (msg.get("reasoning") or "")
        action = parse_action(raw)
        finish = r["choices"][0]["finish_reason"]
        print(f"  step {step}: finish={finish} elapsed={elapsed:.1f}s "
              f"thinking={len(thinking)} content={len(raw)}", flush=True)

        if not action:
            print(f"    [parse fail] raw={raw[:200]}", flush=True)
            break

        a_type = action.get("action")
        a_input = action.get("action_input", {})
        if a_type == "execute_sql":
            sql = a_input.get("sql", "")
            print(f"    [SQL] {sql[:200]}", flush=True)
            result = execute_sql(db_path, sql)
            print(f"    [RESULT] {result[:300]}", flush=True)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"TOOL execute_sql result:\n{result}\n\nContinue."})
        elif a_type == "answer_csv":
            pred_csv = a_input.get("csv", "").strip()
            print(f"    [ANSWER]\n{pred_csv[:500]}", flush=True)
            break
        else:
            print(f"    [unknown action] {a_type}", flush=True)
            break

    gold_text = gold_path.read_text() if gold_path.is_file() else ""
    if pred_csv is None:
        return {"task_id": task_id, "ok": False, "error": "no answer_csv emitted",
                "gold": gold_text}

    cmp = compare_csv(pred_csv, gold_text)
    print(f"\n=== Compare ===", flush=True)
    print(f"  pred cols: {cmp['pred_cols']}", flush=True)
    print(f"  gold cols: {cmp['gold_cols']}", flush=True)
    print(f"  pred rows: {cmp['pred_rows']}, gold rows: {cmp['gold_rows']}", flush=True)
    print(f"  intersect: {cmp['rows_intersect']}, only_pred: {cmp['rows_only_pred']}, "
          f"only_gold: {cmp['rows_only_gold']}", flush=True)
    print(f"  exact_match: {cmp['exact_match']}", flush=True)
    print(f"\nGold:\n{gold_text}", flush=True)
    print(f"\nPred:\n{pred_csv}", flush=True)
    return {"task_id": task_id, "ok": cmp["exact_match"], "compare": cmp,
            "pred": pred_csv, "gold": gold_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_1")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    out = run_task(args.task, max_steps=args.max_steps)
    Path("artifacts/phase2_spike_result.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/phase2_spike_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
