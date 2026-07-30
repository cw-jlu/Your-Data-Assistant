"""Task-domain router for exp159 finance-domain prompt injection.

The router sees only the task question and a compact context inventory. It does
not see task IDs, gold answers, previous predictions, traces, or scores.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage


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
other = any domain outside the four families above.

If terms from multiple domains appear, choose the domain of the main schema
and primary entities in the context inventory, not an incidental related table.
Prefer context inventory over question wording.
Output only the token.
"""


@dataclass(frozen=True, slots=True)
class DomainRouteResult:
    domain: str
    raw: str
    elapsed_sec: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def context_inventory(task: PublicTask) -> dict[str, Any]:
    context_dir = Path(task.context_dir)
    out: dict[str, Any] = {"files": {}, "sqlite_tables": []}
    if not context_dir.exists():
        return out

    for kind in ("csv", "json", "doc", "video"):
        folder = context_dir / kind
        if folder.exists():
            out["files"][kind] = sorted(p.name for p in folder.iterdir() if p.is_file())

    knowledge = context_dir / "knowledge.md"
    if knowledge.exists():
        lines = knowledge.read_text(encoding="utf-8", errors="replace").splitlines()
        out["knowledge_head"] = "\n".join(lines[:8])

    db_folder = context_dir / "db"
    if db_folder.exists():
        sqlite_files = sorted(db_folder.glob("*.sqlite"))
        out["files"]["db"] = [p.name for p in sqlite_files]
        for path in sqlite_files:
            out["sqlite_tables"].append(
                {"db": path.name, "tables": _sqlite_tables(path)[:120]}
            )

    return out


def _normalize(raw: str) -> str:
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    token = token.strip("`'\".,:;")
    return token if token in LABELS else "other"


def classify_domain(task: PublicTask, model: ModelAdapter) -> DomainRouteResult:
    start = time.time()
    payload = {
        "question": task.question,
        "context_inventory": context_inventory(task),
    }
    try:
        raw = model.complete(
            [
                ModelMessage(role="system", content=SYSTEM_PROMPT),
                ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
            enable_thinking=False,
            max_tokens=128,
        )
        return DomainRouteResult(
            domain=_normalize(raw),
            raw=raw,
            elapsed_sec=time.time() - start,
        )
    except Exception as exc:
        return DomainRouteResult(
            domain="other",
            raw="",
            elapsed_sec=time.time() - start,
            error=repr(exc),
        )
