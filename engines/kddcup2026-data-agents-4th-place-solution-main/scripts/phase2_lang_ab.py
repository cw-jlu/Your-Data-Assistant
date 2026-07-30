"""Phase 2 prompt-language A/B: en vs zh system prompt on the same task.

Single-ReAct spike (NOT the full phased harness) so the ONLY variable is the
system-prompt language. Uses the official column-recall λ=0.5 scorer.

Usage:
  uv run python scripts/phase2_lang_ab.py --task task_2 --lang zh
  → prints: SCORE_JSON: {"task":"task_2","lang":"zh","score":0.0,"steps":N,...}
"""
from __future__ import annotations

import argparse, base64, csv, io, json, os, re, time
from pathlib import Path

import requests

CF_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
API_KEY = os.environ.get("AGENT_API_KEY", "")
API_BASE = os.environ.get("AGENT_API_BASE", "").rstrip("/")
MODEL = "qwen3.5-35b-a3b"
DEMO = Path("data/phase2_demo/demo_samples_phase2")

SYSTEM_EN = """You are a data analyst agent. Solve the task using the provided knowledge guide, structured files, and optional briefing video.

Decision steps:
0. If any video file is in context/, watch it FIRST to extract task-specific criteria (thresholds, time windows, scope filters, business phrases). Quote them verbatim before deciding columns.
1. Read the knowledge guide for schema and field semantics.
2. Plan answer columns and aggregation. For "find / list / show / 查 / 找" retrieval intents, return the RAW rows — do NOT SUM, aggregate, or drop NULL rows unless explicitly asked. If a join fans out (constant value repeated), use COUNT(DISTINCT key).
3. Use execute_sql against the data (read-only). Iterate until the result matches the question.
4. When confident, call answer_csv with the final CSV (header + all rows).

Output ONE JSON action per turn in a single ```json fenced block:
  {"action":"execute_sql","action_input":{"sql":"SELECT ..."}}
  {"action":"answer_csv","action_input":{"csv":"col\\nval1\\nval2"}}
Do not wrap the CSV in code fences."""

SYSTEM_ZH = """你是一个数据分析智能体。请使用提供的知识指南、结构化文件以及可选的简报视频来完成任务。

决策步骤：
0. 如果 context/ 中存在视频文件，请先观看视频，提取任务特定的标准（阈值、时间范围、筛选口径、业务术语），并在确定列之前逐字引用这些标准。
1. 阅读知识指南，理解数据表结构（schema）和字段语义。
2. 规划答案的列和聚合方式。对于"查一下 / 找一下 / 列出 / 展示 / find / list / show"这类检索型问题，请返回原始行——除非明确要求，否则不要使用 SUM、聚合或删除 NULL 行。如果连接（join）导致行数膨胀（同一数值重复出现），请使用 COUNT(DISTINCT 主键)。
3. 使用 execute_sql 查询数据（只读）。反复迭代，直到结果与问题相符。
4. 确信无误后，调用 answer_csv 提交最终的 CSV（表头 + 所有数据行）。

每轮只输出一个 JSON 动作，包裹在单个 ```json 代码块中：
  {"action":"execute_sql","action_input":{"sql":"SELECT ..."}}
  {"action":"answer_csv","action_input":{"csv":"列名\\n值1\\n值2"}}
CSV 内容不要再用代码块包裹。"""

_JSON_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def chat(messages, max_tokens=8192, temperature=0.6, max_retries=3):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
    H = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}",
         "CF-Access-Client-Id": CF_ID, "CF-Access-Client-Secret": CF_SECRET}
    for a in range(max_retries + 1):
        try:
            r = requests.post(f"{API_BASE}/chat/completions", headers=H, json=body, timeout=300)
            r.raise_for_status(); return r.json()
        except Exception:
            if a >= max_retries: raise
            time.sleep(min(2 ** a * 2, 30))


def parse_action(content):
    m = _JSON_RE.search(content or "")
    raw = m.group(1).strip() if m else None
    if not raw:
        m2 = re.search(r"\{.*\"action\".*\}", content or "", re.DOTALL)
        raw = m2.group(0) if m2 else None
    if not raw: return None
    try: return json.loads(raw)
    except Exception: return None


def find_db(ctx):
    for ext in ("*.sqlite", "*.db"):
        c = sorted(ctx.glob(f"**/{ext}"))
        if c: return c[0]
    return None


def exec_sql(db, sql, max_rows=50):
    import sqlite3
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows)
            out = io.StringIO(); w = csv.writer(out)
            if cols: w.writerow(cols)
            for r in rows: w.writerow(r)
            t = out.getvalue().strip()
            return f"OK ({len(rows)} rows{'+ (truncated)' if len(rows)==max_rows else ''}):\n{t}"
    except Exception as e:
        return f"SQL ERROR: {e}"


def official_score(pred_path, gold_path):
    import sys
    sys.path.insert(0, "src")
    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    if not gold_path.is_file(): return 0.0
    try:
        ev = _evaluate_task(task_id=pred_path.parent.name, prediction_path=pred_path,
                            gold_path=gold_path, options=EvaluationOptions())
        return float(ev.official_score_lambda_0_5)
    except Exception:
        return 0.0


def run(task_id, lang, max_steps=12):
    tdir = DEMO / "input" / task_id
    meta = json.load(open(tdir / "task.json"))
    q = meta["question"]
    kn = (tdir / "context" / "knowledge.md").read_text() if (tdir / "context" / "knowledge.md").is_file() else ""
    db = find_db(tdir / "context")
    if not db: return {"task": task_id, "lang": lang, "score": 0.0, "error": "no db"}
    vids = sorted((tdir / "context").glob("video/*.mp4"))

    sysmsg = SYSTEM_ZH if lang == "zh" else SYSTEM_EN
    head = "# 知识指南\n\n" if lang == "zh" else "# Knowledge Guide\n\n"
    qhead = "\n\n# 问题\n\n" if lang == "zh" else "\n\n# Question\n\n"
    content = [{"type": "text", "text": f"{head}{kn[:40000]}{qhead}{q}"}]
    if vids:
        content.append({"type": "video_url", "video_url":
            {"url": f"data:video/mp4;base64,{base64.b64encode(vids[0].read_bytes()).decode()}"}})
    messages = [{"role": "system", "content": sysmsg}, {"role": "user", "content": content}]

    pred = None; n_steps = 0
    for step in range(max_steps):
        n_steps = step + 1
        r = chat(messages)
        raw = (r["choices"][0]["message"].get("content") or "").strip()
        act = parse_action(raw)
        if not act: break
        if act.get("action") == "execute_sql":
            res = exec_sql(db, act["action_input"].get("sql", ""))
            messages.append({"role": "assistant", "content": raw})
            cont = "继续。" if lang == "zh" else "Continue."
            messages.append({"role": "user", "content": f"TOOL execute_sql:\n{res}\n\n{cont}"})
        elif act.get("action") == "answer_csv":
            pred = act["action_input"].get("csv", "").strip(); break
        else: break

    outdir = Path(f"/tmp/lang_ab/{task_id}_{lang}")
    outdir.mkdir(parents=True, exist_ok=True)
    score = 0.0
    if pred:
        (outdir / "prediction.csv").write_text(pred)
        score = official_score(outdir / "prediction.csv", DEMO / "output" / task_id / "gold.csv")
    return {"task": task_id, "lang": lang, "score": round(score, 3),
            "steps": n_steps, "answered": pred is not None, "has_video": bool(vids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--max-steps", type=int, default=12)
    a = ap.parse_args()
    res = run(a.task, a.lang, a.max_steps)
    print("SCORE_JSON: " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
