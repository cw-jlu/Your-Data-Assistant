"""PoC for tri-agent source-role routing.

Runs three independent sub-agents (table/doc/video), each classifying only one
source as primary/secondary/no. Inputs are source-only snapshots from the task
context: no traces, gold answers, previous router output, or prediction previews.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from kobushi_core.model import ModelMessage, OpenAIModelAdapter
from experiments.exp_152_final_sql_guard.keyframe_cache import ensure_task_keyframes
from experiments.exp_152_final_sql_guard.prefix_cache import with_prefix_cache_header


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
RUNS_ROOT = REPO / "artifacts" / "runs"
LABEL_PATH = REPO / "artifacts" / "analysis" / "source_router_manual_labels_phase2_demo60.tsv"

ROLES = {"primary", "secondary", "no"}
SOURCES = ("table", "doc", "video")

BASE_INSTRUCTIONS = """Return ONLY compact JSON:
{"role":"primary|secondary|no","reason":"one short sentence"}

Role definitions:
primary = this source provides the final answer values, rows, counts, ranking, or literals.
secondary = this source provides filters, thresholds, selected options, date scope, entity IDs, schema hints, validation, or literal materialization only.
no = this source is not needed or no usable evidence for this source is available.
"""

TABLE_PROMPT = """You classify the TABLE role for a data-answering task.

Classify only the table/structured-data source.

Rules:
- TABLE is primary when structured rows are used to compute, filter, aggregate, or enumerate the final answer.
- TABLE is secondary when it only maps IDs, validates another source, or materializes literals from doc/video.
- A proposed SQL with no real FROM clause (literal SELECT/UNION/VALUES) is not table-primary.
- If table evidence is a similarly named but wrong-domain table and doc/video gives the actual values, TABLE is secondary.
- Use only table_source. Do not infer from missing document or video content.
""" + BASE_INSTRUCTIONS

DOC_PROMPT = """You classify the DOC role for a data-answering task.

Classify only prose, markdown, PDF, and document evidence.

Rules:
- DOC is primary when document text provides the final answer values, rows, counts, or required literal formatting.
- DOC is primary when the structured table is absent/wrong-domain and the document contains the needed records.
- DOC is secondary when it only maps an ID, explains schema, or provides context.
- DOC is no when no document evidence is needed or available.
- Use only doc_source. Do not infer from table data or video keyframes.
""" + BASE_INSTRUCTIONS

VIDEO_PROMPT = """You classify the VIDEO role for a data-answering task.

Classify only video/keyframe evidence.

Rules:
- VIDEO is primary when the attached keyframes fully display the requested final value/list/count/ranking after all filters.
- If the question asks only for the top/highest item, full lower-ranked rows are not required when the video shows the top row and maps its label.
- VIDEO is secondary when it only gives filters, thresholds, selected options, date scope, Top-N, grouping, schema hints, or partial examples.
- VIDEO is not primary for a filtered count if the displayed subtotal includes excluded or below-threshold buckets.
- Buckets marked below threshold/低于门槛/not included/不纳入 cannot be included in a primary video count.
- For bucket counts, verify every included bucket satisfies the filter shown in the keyframes.
- VIDEO is not primary when requested output fields are hidden or missing and another source resolves them.
- Use only the attached keyframes. Do not infer from table data or document text.
""" + BASE_INSTRUCTIONS

PROMPTS = {
    "table": TABLE_PROMPT,
    "doc": DOC_PROMPT,
    "video": VIDEO_PROMPT,
}


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "source-router-tri-agent-poc",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers=headers,
    )


def next_out_dir() -> Path:
    i = 1
    while (RUNS_ROOT / f"source_router_tri_agent_poc_{i:03d}").exists():
        i += 1
    out = RUNS_ROOT / f"source_router_tri_agent_poc_{i:03d}"
    out.mkdir(parents=True)
    return out


def task_question(task_id: str) -> str:
    return json.loads((INPUT_ROOT / task_id / "task.json").read_text(encoding="utf-8"))["question"]


def clip(text: Any, max_chars: int = 1000) -> str:
    value = str(text or "").strip().replace("\n", " | ")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def make_task(task_id: str) -> PublicTask:
    task_dir = INPUT_ROOT / task_id
    record = TaskRecord(task_id=task_id, difficulty="", question=task_question(task_id))
    assets = TaskAssets(task_dir=task_dir, context_dir=task_dir / "context")
    return PublicTask(record=record, assets=assets)


def _safe_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sample_csv(path: Path, context: Path, *, max_rows: int = 3) -> dict[str, Any]:
    sample_rows: list[dict[str, str]] = []
    row_count = 0
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            if len(sample_rows) < max_rows:
                sample_rows.append({k: clip(v, 80) for k, v in row.items()})
    return {
        "path": _safe_rel(path, context),
        "columns": columns,
        "row_count": row_count,
        "sample_rows": sample_rows,
    }


def _json_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        sample = value[:2]
        fields: list[str] = []
        for row in sample:
            if isinstance(row, dict):
                fields.extend(str(k) for k in row.keys())
        return {
            "shape": "list",
            "row_count": len(value),
            "fields": sorted(set(fields))[:80],
            "sample_preview": clip(json.dumps(sample, ensure_ascii=False), 1800),
        }
    if isinstance(value, dict):
        return {
            "shape": "object",
            "keys": list(value.keys())[:80],
            "preview": clip(json.dumps(value, ensure_ascii=False), 2200),
        }
    return {
        "shape": type(value).__name__,
        "preview": clip(json.dumps(value, ensure_ascii=False), 1200),
    }


def _sample_json(path: Path, context: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        summary = _json_summary(data)
    except Exception as exc:
        summary = {"error": repr(exc)}
    return {"path": _safe_rel(path, context), **summary}


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sample_sqlite(path: Path, context: Path, *, max_rows: int = 2) -> dict[str, Any]:
    out: dict[str, Any] = {"path": _safe_rel(path, context), "tables": []}
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        table_names = [
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type in ('table','view') order by name"
            ).fetchall()
        ]
        for name in table_names:
            columns = [row[1] for row in con.execute(f"pragma table_info({_quote_ident(name)})").fetchall()]
            try:
                row_count = int(con.execute(f"select count(*) from {_quote_ident(name)}").fetchone()[0])
            except Exception:
                row_count = None
            sample_rows = []
            try:
                for row in con.execute(f"select * from {_quote_ident(name)} limit {int(max_rows)}").fetchall():
                    sample_rows.append({k: clip(row[k], 80) for k in row.keys()})
            except Exception:
                pass
            out["tables"].append(
                {
                    "name": name,
                    "columns": columns,
                    "row_count": row_count,
                    "sample_rows": sample_rows,
                }
            )
        con.close()
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def table_source(task: PublicTask) -> dict[str, Any]:
    context = task.context_dir
    csv_files = sorted((context / "csv").glob("*.csv")) if (context / "csv").exists() else []
    json_files = sorted((context / "json").glob("*.json")) if (context / "json").exists() else []
    db_files = sorted((context / "db").glob("*.sqlite")) if (context / "db").exists() else []
    return {
        "source_kind": "table",
        "source_inventory": {
            "has_table": bool(csv_files or json_files or db_files),
            "csv_count": len(csv_files),
            "json_count": len(json_files),
            "sqlite_count": len(db_files),
        },
        "csv": [_sample_csv(p, context) for p in csv_files],
        "json": [_sample_json(p, context) for p in json_files],
        "sqlite": [_sample_sqlite(p, context) for p in db_files],
    }


def _read_doc_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
        except Exception as exc:
            return f"[PDF read error: {exc!r}]"
    return path.read_text(encoding="utf-8", errors="replace")


def doc_source(task: PublicTask, *, max_chars_per_doc: int = 3200) -> dict[str, Any]:
    context = task.context_dir
    docs = sorted((context / "doc").glob("*")) if (context / "doc").exists() else []
    knowledge = context / "knowledge.md"
    if knowledge.exists():
        docs = [knowledge] + docs
    items: list[dict[str, Any]] = []
    for path in docs:
        if not path.is_file():
            continue
        text = _read_doc_text(path)
        items.append(
            {
                "path": _safe_rel(path, context),
                "suffix": path.suffix.lower(),
                "size": path.stat().st_size,
                "total_chars": len(text),
                "text_preview": clip(text, max_chars_per_doc),
                "truncated": len(text) > max_chars_per_doc,
            }
        )
    return {
        "source_kind": "doc",
        "source_inventory": {"has_doc": bool(items), "doc_count": len(items)},
        "documents": items,
    }


def _max_video_frames() -> int:
    try:
        return max(1, int(os.environ.get("EXP152_KEYFRAME_MAX_IMAGES", "18")))
    except ValueError:
        return 18


def video_source(task: PublicTask) -> dict[str, Any]:
    context = task.context_dir
    video_files = sorted((context / "video").glob("*")) if (context / "video").exists() else []
    result = ensure_task_keyframes(task)
    frames: list[Path] = []
    if result is not None:
        all_frames = sorted(result.frames_dir.glob("*.jpg"))
        max_frames = _max_video_frames()
        frames = all_frames[:max_frames]
    return {
        "source_kind": "video",
        "source_inventory": {"has_video": bool(video_files), "video_count": len(video_files)},
        "video_files": [_safe_rel(p, context) for p in video_files],
        "keyframe_paths": [str(p) for p in frames],
        "keyframe_names": [p.name for p in frames],
    }


def build_payload(task_id: str) -> dict[str, Any]:
    task = make_task(task_id)
    return {
        "question": task.question,
        "source_payloads": {
            "table": {"question": task.question, "table_source": table_source(task)},
            "doc": {"question": task.question, "doc_source": doc_source(task)},
            "video": {"question": task.question, "video_source": video_source(task)},
        },
    }


def source_payload_for(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source}")
    return payload["source_payloads"][source]


def parse_role(raw: str) -> dict[str, str]:
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object in response: {raw[:200]}")
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("response is not an object")
    role = str(data.get("role", "")).strip()
    reason = str(data.get("reason", "")).strip()
    if role not in ROLES:
        raise ValueError(f"bad role: {role!r}")
    if not reason:
        raise ValueError("missing reason")
    return {"role": role, "reason": clip(reason, 500)}


def classify_source(
    model: OpenAIModelAdapter,
    *,
    source: str,
    payload: dict[str, Any],
) -> tuple[dict[str, str], str]:
    source_payload = source_payload_for(source, payload)
    if source == "video":
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "question": source_payload.get("question", ""),
                        "video_source": {
                            k: v
                            for k, v in source_payload.get("video_source", {}).items()
                            if k != "keyframe_paths"
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        for frame_path in source_payload.get("video_source", {}).get("keyframe_paths", []):
            frame = Path(frame_path)
            if not frame.exists():
                continue
            content.append({"type": "text", "text": f"Keyframe: {frame.name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(frame.read_bytes()).decode()}"
                    },
                }
            )
        raw = model.complete(
            [
                ModelMessage(role="system", content=PROMPTS[source]),
                ModelMessage(role="user", content=content),
            ],
            enable_thinking=False,
            max_tokens=180,
        )
        return parse_role(raw), raw

    raw = model.complete(
        [
            ModelMessage(role="system", content=PROMPTS[source]),
            ModelMessage(role="user", content=json.dumps(source_payload, ensure_ascii=False)),
        ],
        enable_thinking=False,
        max_tokens=180,
    )
    return parse_role(raw), raw


def load_manual_labels() -> dict[str, dict[str, str]]:
    if not LABEL_PATH.exists():
        return {}
    with LABEL_PATH.open(encoding="utf-8", newline="") as f:
        return {row["task_id"]: row for row in csv.DictReader(f, delimiter="\t")}


def default_tasks() -> list[str]:
    preferred = [
        "task_7",
        "task_20",
        "task_24",
        "task_26",
        "task_34",
        "task_38",
        "task_40",
        "task_51",
        "task_57",
        "task_60",
    ]
    return [task_id for task_id in preferred if (INPUT_ROOT / task_id / "task.json").exists()]


def final_sources(roles: dict[str, dict[str, str]]) -> list[str]:
    return [source for source in SOURCES if roles.get(source, {}).get("role") == "primary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="", help="Comma-separated task ids. Default: focused PoC set.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else default_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    labels = load_manual_labels()
    out_dir = next_out_dir()
    model = make_model()
    rows: list[dict[str, Any]] = []
    for i, task_id in enumerate(tasks, 1):
        payload = build_payload(task_id)
        roles: dict[str, dict[str, str]] = {}
        raw: dict[str, str] = {}
        errors: dict[str, str] = {}
        started = time.time()
        for source in SOURCES:
            try:
                result, source_raw = classify_source(model, source=source, payload=payload)
                roles[source] = result
                raw[source] = source_raw
            except Exception as exc:
                errors[source] = repr(exc)
                roles[source] = {"role": "no", "reason": f"error: {exc!r}"}
        predicted_final = final_sources(roles)
        manual = labels.get(task_id)
        row = {
            "task_id": task_id,
            "roles": roles,
            "final_answer_source": predicted_final,
            "errors": errors,
            "raw": raw,
            "elapsed": time.time() - started,
            "payload": payload,
            "source_payloads": {source: source_payload_for(source, payload) for source in SOURCES},
            "manual": manual,
        }
        rows.append(row)
        partial_summary = []
        for partial in rows:
            partial_roles = partial["roles"]
            partial_summary.append(
                {
                    "task_id": partial["task_id"],
                    "table": partial_roles["table"]["role"],
                    "doc": partial_roles["doc"]["role"],
                    "video": partial_roles["video"]["role"],
                    "final_answer_source": partial["final_answer_source"],
                    "manual_final_source": (partial["manual"] or {}).get("final_source"),
                    "errors": partial["errors"],
                }
            )
        (out_dir / "partial_results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "partial_summary.json").write_text(
            json.dumps(partial_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manual_final = manual["final_source"] if manual else "?"
        ok = manual_final == "+".join(predicted_final)
        print(
            f"[{i:02d}/{len(tasks)}] {task_id} "
            f"table={roles['table']['role']} doc={roles['doc']['role']} video={roles['video']['role']} "
            f"final={'+'.join(predicted_final) or 'none'} manual={manual_final} ok={ok}",
            flush=True,
        )

    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = []
    for row in rows:
        roles = row["roles"]
        summary.append(
            {
                "task_id": row["task_id"],
                "table": roles["table"]["role"],
                "doc": roles["doc"]["role"],
                "video": roles["video"]["role"],
                "final_answer_source": row["final_answer_source"],
                "manual_final_source": (row["manual"] or {}).get("final_source"),
                "manual_roles": (
                    {
                        "table": row["manual"]["table_role"],
                        "doc": row["manual"]["doc_role"],
                        "video": row["manual"]["video_role"],
                    }
                    if row["manual"]
                    else None
                ),
                "elapsed": row["elapsed"],
                "errors": row["errors"],
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if labels:
        final_ok = 0
        role_ok = Counter()
        role_total = Counter()
        for item in summary:
            manual = item.get("manual_roles")
            if not manual:
                continue
            if item["manual_final_source"] == "+".join(item["final_answer_source"]):
                final_ok += 1
            for source in SOURCES:
                role_total[source] += 1
                if item[source] == manual[source]:
                    role_ok[source] += 1
        metrics = {
            "n": len(summary),
            "final_source_accuracy": final_ok / len(summary) if summary else 0.0,
            "role_accuracy": {
                source: role_ok[source] / role_total[source] if role_total[source] else 0.0
                for source in SOURCES
            },
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
