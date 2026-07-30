"""PoC for a phase-2 schema-linker sub-agent.

The agent sees only the task question, source inventory, and visible SQL
catalog. It does not see gold answers, previous runs, scores, or traces.
Gold labels are loaded only after inference for offline evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_154_v1_audio_asr.tools.duckdb_unified import get_catalog
from experiments.exp_155_phase_tool_visibility.prefix_cache import with_prefix_cache_header


INPUT_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"
GOLD_PATH = REPO / "artifacts" / "schema_linking_gold" / "phase2_demo_schema_link_gold_v0.jsonl"
RUNS_ROOT = REPO / "artifacts" / "runs"

SYSTEM_PROMPT = """You are a schema linker for a data-answering task.

Given a user question, source inventory, and visible SQL catalog, identify the
tables and columns likely needed to answer the question.

Rules:
- Do not answer the question.
- Internally identify the requested entity, metric, output fields, filters,
  grouping fields, and ordering fields before choosing tables.
- Use only exact table names and column names from the provided SQL catalog.
- Include columns needed for filters, joins, grouping, ordering, or output.
- Include only likely relevant tables/columns; prefer an empty list over guessing.
- Return the minimal schema set, not alternative or backup tables.
- Use at most 4 SQL tables and at most 12 SQL columns total unless the question
  clearly requires more.
- Do not include a table only because it has join keys such as id/code/name.
  Each selected table must contain at least one non-key column that directly
  matches a requested entity, metric, filter, grouping, ordering, or output.
- Preserve entity grain and metric meaning. Do not substitute a nearby table or
  related metric when the requested entity or metric is different.
- For example, organization/company attributes are not product attributes, and
  benchmark growth is not NAV/return growth.
- If a document/video source appears to be the named dataset or carries the
  requested metric while SQL has only a nearby-domain substitute, use
  non_sql_sources and leave schema_links empty or partial.
- Include identifier/code columns for joins when two selected SQL tables refer
  to the same entity and a code/key column is visible.
- If needed evidence appears to come from a document or video rather than SQL,
  add it to non_sql_sources.
- Return JSON only.

Output schema:
{
  "schema_links": [
    {"table": "exact_table_name", "columns": ["exact_column_name"]}
  ],
  "non_sql_sources": [
    {"type": "doc" | "video", "source": "file-or-short-source", "reason": "short reason"}
  ],
  "notes": "one short note"
}
"""


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "schema-linker-agent-poc",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers=headers,
        enable_thinking=False,
        max_tokens=900,
    )


def next_out_dir() -> Path:
    i = 1
    while (RUNS_ROOT / f"schema_linker_agent_poc_{i:03d}").exists():
        i += 1
    out = RUNS_ROOT / f"schema_linker_agent_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def clip(value: Any, max_chars: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_question(task_id: str) -> str:
    return str(json.loads((INPUT_ROOT / task_id / "task.json").read_text(encoding="utf-8"))["question"])


def context_inventory(task_id: str) -> dict[str, Any]:
    context = INPUT_ROOT / task_id / "context"
    inventory: dict[str, Any] = {"files": {}}
    for kind in ("csv", "json", "doc", "video", "db"):
        folder = context / kind
        if folder.exists():
            inventory["files"][kind] = sorted(p.name for p in folder.iterdir() if p.is_file())
    knowledge = context / "knowledge.md"
    if knowledge.exists():
        lines = knowledge.read_text(encoding="utf-8", errors="replace").splitlines()
        inventory["knowledge_head"] = "\n".join(lines[:10])
    return inventory


def compact_catalog(task_id: str) -> list[dict[str, Any]]:
    catalog = get_catalog(INPUT_ROOT / task_id / "context")
    return [
        {
            "table": row["view"],
            "source": row.get("source", ""),
            "kind": row.get("kind", ""),
            "columns": row.get("columns", []),
        }
        for row in catalog
    ]


def parse_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in response: {raw[:300]}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    return data


def normalize_prediction(data: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    table_cols = {row["table"]: set(row["columns"]) for row in catalog}
    raw_links = data.get("schema_links") or []
    valid_links: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    all_tables: set[str] = set()
    all_columns: set[tuple[str, str]] = set()

    if not isinstance(raw_links, list):
        raw_links = []
        invalid.append({"reason": "schema_links_not_list", "value": data.get("schema_links")})

    for item in raw_links:
        if not isinstance(item, dict):
            invalid.append({"reason": "link_not_object", "value": item})
            continue
        table = str(item.get("table") or "").strip()
        cols = item.get("columns") or []
        if not isinstance(cols, list):
            cols = []
            invalid.append({"reason": "columns_not_list", "table": table, "value": item.get("columns")})
        cols = [str(col).strip() for col in cols if str(col).strip()]
        if table:
            all_tables.add(table)
        for col in cols:
            all_columns.add((table, col))
        if table not in table_cols:
            invalid.append({"reason": "missing_table", "table": table, "columns": cols})
            continue
        missing = [col for col in cols if col not in table_cols[table]]
        if missing:
            invalid.append({"reason": "missing_columns", "table": table, "columns": missing})
        valid_cols = [col for col in cols if col in table_cols[table]]
        if valid_cols:
            valid_links.append({"table": table, "columns": sorted(set(valid_cols), key=valid_cols.index)})

    non_sql = data.get("non_sql_sources") or []
    if not isinstance(non_sql, list):
        non_sql = []
    source_types: set[str] = set()
    clean_sources: list[dict[str, str]] = []
    for item in non_sql:
        if not isinstance(item, dict):
            continue
        typ = str(item.get("type") or "").strip().lower()
        if typ not in {"doc", "video"}:
            continue
        source_types.add(typ)
        clean_sources.append(
            {
                "type": typ,
                "source": clip(item.get("source", ""), 160),
                "reason": clip(item.get("reason", ""), 240),
            }
        )

    return {
        "schema_links": valid_links,
        "invalid_schema_links": invalid,
        "all_pred_tables": sorted(all_tables),
        "all_pred_columns": sorted([{"table": t, "column": c} for t, c in all_columns], key=lambda x: (x["table"], x["column"])),
        "non_sql_sources": clean_sources,
        "non_sql_types": sorted(source_types),
        "notes": clip(data.get("notes", ""), 400),
    }


def call_schema_linker(model: OpenAIModelAdapter, task_id: str) -> dict[str, Any]:
    payload = {
        "question": load_question(task_id),
        "source_inventory": context_inventory(task_id),
        "sql_catalog": compact_catalog(task_id),
    }
    raw = model.complete(
        [
            ModelMessage(role="system", content=SYSTEM_PROMPT),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ],
        enable_thinking=False,
        max_tokens=900,
    )
    parsed = parse_json(raw)
    normalized = normalize_prediction(parsed, payload["sql_catalog"])
    return {"raw": raw, "parsed": parsed, "normalized": normalized, "payload": payload}


def load_gold() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows[item["task_id"]] = item
    return rows


def gold_sets(item: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    tables: set[str] = set()
    columns: set[tuple[str, str]] = set()
    for link in item.get("schema_links") or []:
        table = str(link["table"])
        tables.add(table)
        for col in link.get("columns") or []:
            columns.add((table, str(col)))
    types: set[str] = set()
    for source in item.get("non_sql_evidence") or []:
        text = str(source).lower()
        if "video" in text:
            types.add("video")
        if text.startswith("doc/") or "doc/" in text or ".pdf" in text or ".md" in text:
            types.add("doc")
    return tables, columns, types


def pred_sets(result: dict[str, Any]) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    normalized = result["normalized"]
    pred_tables = set(normalized.get("all_pred_tables") or [])
    pred_columns = {
        (str(item["table"]), str(item["column"]))
        for item in normalized.get("all_pred_columns") or []
    }
    pred_types = set(normalized.get("non_sql_types") or [])
    return pred_tables, pred_columns, pred_types


def prf(pred: set[Any], gold: set[Any]) -> dict[str, Any]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_one(task_id: str, result: dict[str, Any], gold_by_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold = gold_by_task[task_id]
    pred_tables, pred_columns, pred_types = pred_sets(result)
    gold_tables, gold_columns, gold_types = gold_sets(gold)
    return {
        "task_id": task_id,
        "gold_status": gold.get("label_status"),
        "table": prf(pred_tables, gold_tables),
        "column": prf(pred_columns, gold_columns),
        "non_sql_type": prf(pred_types, gold_types),
        "invalid_schema_link_count": len(result["normalized"].get("invalid_schema_links") or []),
        "pred_tables": sorted(pred_tables),
        "gold_tables": sorted(gold_tables),
        "pred_non_sql_types": sorted(pred_types),
        "gold_non_sql_types": sorted(gold_types),
    }


def aggregate(evals: list[dict[str, Any]]) -> dict[str, Any]:
    def sum_counts(key: str) -> dict[str, Any]:
        tp = sum(e[key]["tp"] for e in evals)
        fp = sum(e[key]["fp"] for e in evals)
        fn = sum(e[key]["fn"] for e in evals)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    exact_table = sum(e["table"]["fp"] == 0 and e["table"]["fn"] == 0 for e in evals)
    exact_column = sum(e["column"]["fp"] == 0 and e["column"]["fn"] == 0 for e in evals)
    exact_non_sql = sum(e["non_sql_type"]["fp"] == 0 and e["non_sql_type"]["fn"] == 0 for e in evals)
    return {
        "n": len(evals),
        "table_micro": sum_counts("table"),
        "column_micro": sum_counts("column"),
        "non_sql_type_micro": sum_counts("non_sql_type"),
        "table_exact_rate": round(exact_table / len(evals), 4) if evals else None,
        "column_exact_rate": round(exact_column / len(evals), 4) if evals else None,
        "non_sql_type_exact_rate": round(exact_non_sql / len(evals), 4) if evals else None,
        "invalid_schema_link_count": sum(e["invalid_schema_link_count"] for e in evals),
        "status_counts": dict(Counter(e["gold_status"] for e in evals)),
    }


def parse_tasks(value: str, limit: int = 0) -> list[str]:
    if value:
        tasks = [part.strip() for part in value.split(",") if part.strip()]
    else:
        tasks = [f"task_{i}" for i in range(1, 61)]
    if limit:
        tasks = tasks[:limit]
    return tasks


def run_task(task_id: str) -> dict[str, Any]:
    model = make_model()
    started = time.time()
    error = None
    result: dict[str, Any] | None = None
    try:
        result = call_schema_linker(model, task_id)
    except Exception as exc:
        error = repr(exc)
    return {
        "task_id": task_id,
        "elapsed_sec": round(time.time() - started, 3),
        "error": error,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="", help="Comma-separated task ids; default: all 60.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not GOLD_PATH.exists():
        raise SystemExit(f"gold labels not found: {GOLD_PATH}")

    tasks = parse_tasks(args.tasks, args.limit)
    out_dir = next_out_dir()
    (out_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    gold_by_task = load_gold()

    rows: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_task, task_id): task_id for task_id in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            task_id = row["task_id"]
            if row["result"] is not None and row["error"] is None:
                ev = evaluate_one(task_id, row["result"], gold_by_task)
                evals.append(ev)
                print(
                    f"[{i:02d}/{len(tasks)}] {task_id} "
                    f"table_f1={ev['table']['f1']:.4f} col_f1={ev['column']['f1']:.4f} "
                    f"non_sql_f1={ev['non_sql_type']['f1']:.4f} "
                    f"invalid={ev['invalid_schema_link_count']} {row['elapsed_sec']}s",
                    flush=True,
                )
            else:
                print(f"[{i:02d}/{len(tasks)}] {task_id} ERROR {row['error']}", flush=True)

    rows.sort(key=lambda r: int(r["task_id"].split("_")[-1]))
    evals.sort(key=lambda r: int(r["task_id"].split("_")[-1]))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": tasks,
        "workers": args.workers,
        "errors": [r for r in rows if r["error"]],
        "aggregate": aggregate(evals),
        "evals": evals,
    }
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_id",
                "gold_status",
                "table_precision",
                "table_recall",
                "table_f1",
                "column_precision",
                "column_recall",
                "column_f1",
                "non_sql_precision",
                "non_sql_recall",
                "non_sql_f1",
                "invalid_schema_link_count",
                "pred_tables",
                "gold_tables",
                "pred_non_sql_types",
                "gold_non_sql_types",
            ]
        )
        for ev in evals:
            writer.writerow(
                [
                    ev["task_id"],
                    ev["gold_status"],
                    ev["table"]["precision"],
                    ev["table"]["recall"],
                    ev["table"]["f1"],
                    ev["column"]["precision"],
                    ev["column"]["recall"],
                    ev["column"]["f1"],
                    ev["non_sql_type"]["precision"],
                    ev["non_sql_type"]["recall"],
                    ev["non_sql_type"]["f1"],
                    ev["invalid_schema_link_count"],
                    "; ".join(ev["pred_tables"]),
                    "; ".join(ev["gold_tables"]),
                    "; ".join(ev["pred_non_sql_types"]),
                    "; ".join(ev["gold_non_sql_types"]),
                ]
            )

    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
