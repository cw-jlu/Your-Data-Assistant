"""Pre-compute LLM-judged FK relationships for each public DABench task.

Cache layout: artifacts/schema_fk_cache/<task_id>.json

Per task we feed the LLM:
  - For each table (CSV / SQLite / JSON-array): name + columns (with dtype + ~5 sample distinct values)
  - The task question (so the LLM can prioritize FKs relevant to the asked relationship)

We ask the LLM to return a JSON object with:
  {
    "joins": [
      {"from": "fileA::tableA.colX", "to": "fileB::tableB.colY",
       "kind": "fk_child_to_parent | fk_parent_to_child | duplicate_copy",
       "confidence": "high|med|low",
       "reason": "..." }
    ],
    "orphan_columns": ["fileX.colZ", ...]  # columns unlikely to participate in any join, may help the agent skip them
  }

Cache invalidation: re-run this script whenever schema understanding rules
change. Cache files are JSON; run.py reads them at preamble-build time.

Usage:
    uv run python scripts/precompute_schema_fk.py                    # all tasks
    uv run python scripts/precompute_schema_fk.py --task task_64     # single
    uv run python scripts/precompute_schema_fk.py --force            # overwrite cache
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/...` invocation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import ModelMessage, OpenAIModelAdapter

load_dotenv()

CACHE_DIR = Path("artifacts/schema_fk_cache")
DEFAULT_DATA_ROOT = Path("data/public/input")

_SAMPLE_VALUES_PER_COLUMN = 5
_MAX_VALUES_TO_READ = 1000  # for distinct-value sampling per column


@dataclass(frozen=True, slots=True)
class _ColInfo:
    file: str
    table: str | None
    column: str
    dtype: str
    sample_values: list[str]


def _summarize(values: list[str]) -> list[str]:
    seen: list[str] = []
    seenset: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seenset:
            continue
        seen.append(s if len(s) < 80 else s[:77] + "...")
        seenset.add(s)
        if len(seen) >= _SAMPLE_VALUES_PER_COLUMN:
            break
    return seen


def _csv_columns(path: Path) -> list[_ColInfo]:
    import pandas as pd

    out: list[_ColInfo] = []
    try:
        df = pd.read_csv(path, nrows=_MAX_VALUES_TO_READ, low_memory=False)
    except Exception:
        return out
    for col in df.columns:
        try:
            vals = df[col].dropna().astype(str).tolist()
        except Exception:
            continue
        out.append(
            _ColInfo(
                file=path.name,
                table=None,
                column=str(col),
                dtype=str(df[col].dtype),
                sample_values=_summarize(vals),
            )
        )
    return out


def _sqlite_columns(path: Path) -> list[_ColInfo]:
    out: list[_ColInfo] = []
    try:
        conn = sqlite3.connect(path)
    except Exception:
        return out
    try:
        cur = conn.cursor()
        tables = [
            r[0]
            for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        for table in tables:
            try:
                cols = cur.execute(f'PRAGMA table_info("{table}")').fetchall()
            except Exception:
                continue
            for ci in cols:
                col_name, col_type = ci[1], ci[2] or "?"
                try:
                    rows = cur.execute(
                        f'SELECT "{col_name}" FROM "{table}" LIMIT {_MAX_VALUES_TO_READ}'
                    ).fetchall()
                    vals = [str(r[0]) for r in rows if r[0] is not None]
                except Exception:
                    vals = []
                out.append(
                    _ColInfo(
                        file=path.name,
                        table=table,
                        column=col_name,
                        dtype=col_type,
                        sample_values=_summarize(vals),
                    )
                )
    finally:
        conn.close()
    return out


def _json_columns(path: Path) -> list[_ColInfo]:
    out: list[_ColInfo] = []
    try:
        # Stream-safe partial read for huge JSONs.
        with path.open("rb") as fh:
            head = fh.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
        # Try full parse first; fall back to a head-only parse for huge arrays.
        try:
            obj = json.loads(head)
        except json.JSONDecodeError:
            # Probably a truncated array head — find the last complete object.
            depth = 0
            last_complete = -1
            in_str = False
            esc = False
            for i, ch in enumerate(head):
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        last_complete = i + 1
            if last_complete < 0:
                return out
            try:
                obj = json.loads(head[: head.index("[") + 1] + head[head.index("[") + 1 : last_complete] + "]")
            except Exception:
                return out
    except Exception:
        return out
    if not isinstance(obj, list) or not obj or not isinstance(obj[0], dict):
        return out
    keys = list(obj[0].keys())
    for k in keys:
        vals = []
        for r in obj[: _MAX_VALUES_TO_READ]:
            if isinstance(r, dict) and k in r:
                vals.append(str(r[k]))
        out.append(
            _ColInfo(
                file=path.name,
                table=None,
                column=k,
                dtype=type(obj[0][k]).__name__,
                sample_values=_summarize(vals),
            )
        )
    return out


def collect_columns(task_dir: Path) -> list[_ColInfo]:
    cols: list[_ColInfo] = []
    context_dir = task_dir / "context"
    if not context_dir.exists():
        return cols
    for p in sorted(context_dir.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            cols.extend(_csv_columns(p))
        elif suf in {".db", ".sqlite", ".sqlite3"}:
            cols.extend(_sqlite_columns(p))
        elif suf == ".json":
            cols.extend(_json_columns(p))
    return cols


def _format_inventory(cols: list[_ColInfo]) -> str:
    """Compact human/LLM readable inventory."""
    # Group by file::table
    groups: dict[tuple[str, str | None], list[_ColInfo]] = {}
    for c in cols:
        groups.setdefault((c.file, c.table), []).append(c)
    lines = []
    for (file, table), gcols in groups.items():
        loc = f"{file}::{table}" if table else file
        lines.append(f"### {loc}")
        for c in gcols:
            samples = ", ".join(repr(v) for v in c.sample_values) or "(no samples)"
            lines.append(f"  - {c.column} :: {c.dtype} :: samples=[{samples}]")
    return "\n".join(lines)


_SYSTEM_PROMPT = """You are a data-engineering expert analyzing a multi-file dataset to identify foreign-key relationships and irrelevant columns. The dataset may include CSV files, SQLite databases (with multiple tables), and JSON arrays. Foreign keys may be IMPLICIT (no DDL declares them) — you must infer them from column names, dtypes, and overlapping sample values.

Output a single JSON object with this exact structure:
{
  "joins": [
    {
      "from": "<fileA>[::<tableA>].<colX>",
      "to":   "<fileB>[::<tableB>].<colY>",
      "kind": "fk_child_to_parent" | "fk_parent_to_child" | "duplicate_copy",
      "confidence": "high" | "med" | "low",
      "reason": "<short justification>"
    }
  ],
  "orphan_columns": ["<file[::table].col>", ...]
}

Rules:
- "fk_child_to_parent": the FROM column's values are a subset of TO's (e.g. orders.user_id → users.id). Always express the child→parent direction in this kind.
- "duplicate_copy": same data exposed in two formats (e.g. CSV mirror of a SQLite table). Mark these to warn the agent against double-counting.
- "orphan_columns": columns clearly unrelated to any join graph (e.g. free-text comments, scoring metrics) that the agent likely won't need for joins. Be conservative — only list when you're confident. Empty list is fine.
- Quote column locations EXACTLY as shown in the inventory header (case-sensitive).
- Prefer "high" confidence only when names match AND values overlap meaningfully.
- DO NOT invent columns that don't appear in the inventory.
- Output JSON only, no surrounding prose. No markdown fences."""


def _build_user_prompt(task_id: str, question: str, inventory: str) -> str:
    return f"""# Task
task_id: {task_id}
question: {question}

# Column inventory (file::table.column :: dtype :: sample distinct values)
{inventory}

Now identify FK joins, duplicate copies, and orphan columns relevant to answering the question. Return JSON only."""


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks defensively."""
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Orphan close
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_json(text: str) -> dict | None:
    """Extract first balanced { ... } JSON object from text."""
    text = _strip_thinking(text)
    # Strip markdown fences if any
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith(("json", "JSON")):
            text = text[4:].lstrip("\n")
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first { ... } balanced
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None


def precompute_one(
    task_id: str, dataset: DABenchPublicDataset, model: OpenAIModelAdapter, *, force: bool
) -> tuple[str, dict | None, str]:
    cache_path = CACHE_DIR / f"{task_id}.json"
    if cache_path.exists() and not force:
        return task_id, None, "cached"
    task = dataset.get_task(task_id)
    cols = collect_columns(task.task_dir)
    if len(cols) < 2:
        # Nothing to join.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"joins": [], "orphan_columns": [], "_meta": {"reason": "fewer_than_2_columns"}}
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return task_id, payload, "trivial"

    inventory = _format_inventory(cols)
    user_prompt = _build_user_prompt(task_id, task.question, inventory)

    messages = [
        ModelMessage(role="system", content=_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_prompt),
    ]
    t0 = time.time()
    try:
        # Disable thinking mode — we want fast, structured output.
        raw = model.complete(messages, enable_thinking=False)
    except Exception as exc:
        return task_id, None, f"error: {exc!r}"
    dt = time.time() - t0
    parsed = _extract_json(raw)
    if parsed is None:
        # Save the raw failure for debugging (still write cache so we don't redo).
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"joins": [], "orphan_columns": [], "_meta": {"parse_error": True, "raw": raw[:2000]}},
                ensure_ascii=False,
                indent=2,
            )
        )
        return task_id, None, f"parse_error in {dt:.1f}s"

    parsed.setdefault("joins", [])
    parsed.setdefault("orphan_columns", [])
    parsed["_meta"] = {"latency_s": round(dt, 2), "n_columns": len(cols)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    return task_id, parsed, f"ok in {dt:.1f}s ({len(parsed.get('joins', []))} joins)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="single task_id (e.g. task_64). default: all")
    ap.add_argument("--force", action="store_true", help="overwrite cache")
    ap.add_argument("--workers", type=int, default=8, help="parallel LLM calls")
    args = ap.parse_args()

    api_base = os.environ.get("AGENT_API_BASE", "")
    api_key = os.environ.get("AGENT_API_KEY", "")
    cf_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    cf_sec = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    if not (api_base and api_key):
        print("Missing AGENT_API_BASE / AGENT_API_KEY in env (.env)")
        sys.exit(2)
    headers = {}
    if cf_id and cf_sec:
        headers["CF-Access-Client-Id"] = cf_id
        headers["CF-Access-Client-Secret"] = cf_sec

    model = OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=api_base,
        api_key=api_key,
        temperature=0.6,
        extra_headers=headers,
        max_tokens=8192,
        enable_thinking=False,
    )

    dataset = DABenchPublicDataset(root_dir=DEFAULT_DATA_ROOT)
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = dataset.list_task_ids()
    print(f"precomputing FK for {len(task_ids)} tasks (force={args.force}, workers={args.workers})")
    print(f"cache dir: {CACHE_DIR}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(precompute_one, tid, dataset, model, force=args.force): tid for tid in task_ids
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                _, _, status = fut.result()
            except Exception as exc:
                status = f"exception: {exc!r}"
            print(f"  {tid}: {status}")
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
