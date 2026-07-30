"""PoC for a lightweight phase-2 domain router sub-agent.

The router sees only a task question and a compact context inventory. It does
not see task IDs, gold answers, traces, predictions, or previous run scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
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
from experiments.exp_152_final_sql_guard.prefix_cache import with_prefix_cache_header


LABELS = {"fund", "stock", "macro", "ehr", "other"}

SYSTEM_PROMPT = """Classify the data task domain from the question and context inventory.

Output exactly one token, one of: fund, stock, macro, ehr, other.

Definitions:
fund = public/mutual fund data, fund products, fund managers, fund companies,
returns, risk, benchmark growth, mf_*, 公募基金, 基金经理, 基金管理人.
stock = A-share/listed company/stock/shareholder/corporate action data, lc_*,
qt_*, A股, 股票, 股东, 配股, 公司档案.
macro = macroeconomic/monetary/statistical data, ed_*, in_*,
fm_depositreserveratio, GDP, monetary authority, deposits, retail sales,
import/export, tax, PMI.
ehr = clinical patient-record or hospital EHR data: patients, visits,
admissions, diagnoses, lab tests, vital signs, treatments, medications,
allergies, microbiology, ICU records, eICU/MIMIC-style tables or IDs such as
patientunitstayid, hadm_id, icustay_id, ADMISSIONS, LABEVENTS, PRESCRIPTIONS,
vitalperiodic, treatment, microlab, allergy.
other = any domain outside the four families above, including education,
sports, card games, bank accounts, superheroes, toxicology, software,
racing, or general business data.

If terms from multiple domains appear, choose the domain of the main schema
and primary entities in the context inventory, not an incidental related table.
Prefer context inventory over question wording.
Output only the token.
"""


def make_model() -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        "domain-router-poc",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.0,
        extra_headers=headers,
        enable_thinking=False,
        max_tokens=128,
    )


def clip(value: Any, max_chars: int = 700) -> str:
    text = str(value or "").strip().replace("\n", " | ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _sqlite_tables(path: Path) -> list[str]:
    try:
        con = sqlite3.connect(path)
        rows = con.execute(
            "select name from sqlite_master where type in ('table','view') order by name"
        ).fetchall()
        con.close()
        return [str(row[0]) for row in rows]
    except Exception:
        return []


def context_inventory(task_dir: Path) -> dict[str, Any]:
    context = task_dir / "context"
    out: dict[str, Any] = {"files": {}, "sqlite_tables": []}
    if not context.exists():
        return out

    for kind in ("csv", "json", "doc", "video"):
        folder = context / kind
        if folder.exists():
            out["files"][kind] = sorted(p.name for p in folder.iterdir() if p.is_file())

    knowledge = context / "knowledge.md"
    if knowledge.exists():
        lines = knowledge.read_text(encoding="utf-8", errors="replace").splitlines()
        out["knowledge_head"] = "\n".join(lines[:8])

    db_folder = context / "db"
    if db_folder.exists():
        sqlite_files = sorted(db_folder.glob("*.sqlite"))
        out["files"]["db"] = [p.name for p in sqlite_files]
        for path in sqlite_files:
            out["sqlite_tables"].append(
                {"db": path.name, "tables": _sqlite_tables(path)[:120]}
            )

    return out


def load_question(task_dir: Path) -> str:
    data = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    return str(data.get("question") or "")


def expected_phase2(task_id: str, previous_result: dict[str, Any]) -> str | None:
    for row in previous_result.get("results", []):
        if row.get("task") == task_id:
            return row.get("expected")
    return None


def normalize(raw: str) -> str:
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    token = token.strip("`'\".,:;")
    return token if token in LABELS else f"invalid:{clip(raw, 80)}"


def classify_task(
    *,
    model: OpenAIModelAdapter,
    dataset: str,
    task_dir: Path,
    expected: str | None,
) -> dict[str, Any]:
    start = time.time()
    task_id = task_dir.name
    payload = {
        "question": load_question(task_dir),
        "context_inventory": context_inventory(task_dir),
    }
    error = None
    raw = ""
    pred = ""
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=SYSTEM_PROMPT),
                ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
            enable_thinking=False,
            max_tokens=128,
        )
        pred = normalize(raw)
    except Exception as exc:
        error = repr(exc)
        pred = "error"
    return {
        "dataset": dataset,
        "task": task_id,
        "expected": expected,
        "pred": pred,
        "ok": expected is None or pred == expected,
        "raw": raw,
        "error": error,
        "elapsed_sec": round(time.time() - start, 3),
    }


def iter_tasks(input_root: Path) -> list[Path]:
    return sorted(
        [p for p in input_root.iterdir() if p.is_dir() and (p / "task.json").exists()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.startswith("task_") else p.name,
    )


def run_dataset(
    *,
    dataset: str,
    input_root: Path,
    expected_by_task: dict[str, str | None],
    workers: int,
) -> dict[str, Any]:
    model = make_model()
    task_dirs = iter_tasks(input_root)
    results: list[dict[str, Any]] = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                classify_task,
                model=model,
                dataset=dataset,
                task_dir=task_dir,
                expected=expected_by_task.get(task_dir.name),
            )
            for task_dir in task_dirs
        ]
        for fut in as_completed(futures):
            row = fut.result()
            results.append(row)
            status = "ok" if row["ok"] else "BAD"
            print(
                f"[{dataset}] {row['task']}: {row['pred']} "
                f"expected={row['expected']} {status} {row['elapsed_sec']}s",
                flush=True,
            )
    results.sort(key=lambda r: int(r["task"].split("_")[-1]))
    expected_rows = [r for r in results if r["expected"] is not None]
    correct = sum(1 for r in expected_rows if r["ok"])
    return {
        "dataset": dataset,
        "input_root": str(input_root),
        "n": len(results),
        "expected_n": len(expected_rows),
        "correct": correct,
        "accuracy": correct / len(expected_rows) if expected_rows else None,
        "elapsed_sec": round(time.time() - start, 3),
        "pred_counts": dict(Counter(r["pred"] for r in results)),
        "expected_counts": dict(Counter(r["expected"] for r in expected_rows)),
        "mismatches": [r for r in expected_rows if not r["ok"]],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dataset",
        choices=["phase2", "phase1-public", "both"],
        default="both",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "artifacts" / "domain_router_poc_other_001.json",
    )
    args = parser.parse_args()

    previous = json.loads(
        (REPO / "artifacts" / "domain_router_poc_001.json").read_text(encoding="utf-8")
    )
    datasets: list[tuple[str, Path, dict[str, str | None]]] = []
    if args.dataset in {"phase2", "both"}:
        root = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"
        expected = {p.name: expected_phase2(p.name, previous) for p in iter_tasks(root)}
        datasets.append(("phase2", root, expected))
    if args.dataset in {"phase1-public", "both"}:
        root = REPO / "data" / "public" / "input"
        expected = {p.name: "other" for p in iter_tasks(root)}
        datasets.append(("phase1-public", root, expected))

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": SYSTEM_PROMPT,
        "workers": args.workers,
        "datasets": [],
    }
    for dataset, root, expected in datasets:
        payload["datasets"].append(
            run_dataset(
                dataset=dataset,
                input_root=root,
                expected_by_task=expected,
                workers=args.workers,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
