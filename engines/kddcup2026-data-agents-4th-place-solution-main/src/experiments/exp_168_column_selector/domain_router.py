"""Task-domain router for exp166 finance-domain prompt injection.

The router sees only the task question and a compact context inventory. It does
not see task IDs, gold answers, previous predictions, traces, or scores.
"""
from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter


# EHR is refined to its source family (eICU vs MIMIC) deterministically from the
# schema: MIMIC uses UPPERCASE tables + D_* dictionaries; eICU uses lowercase
# clinical tables (lab/treatment/...) keyed on patientunitstayid.
_MIMIC_TABLES = {
    "ADMISSIONS", "PATIENTS", "LABEVENTS", "PROCEDURES_ICD", "DIAGNOSES_ICD",
    "PRESCRIPTIONS", "ICUSTAYS", "CHARTEVENTS", "MICROBIOLOGYEVENTS",
    "D_ICD_PROCEDURES", "D_ICD_DIAGNOSES", "D_LABITEMS", "D_ITEMS",
}
_EICU_TABLES = {
    "lab", "patient", "treatment", "diagnosis", "medication", "allergy",
    "microlab", "intakeoutput", "vitalperiodic", "cost", "apacheapsvar",
}

# Finance routing is deterministic: the demo author writes the source-db id into
# the knowledge guide as `ccks_(stock|fund|macro)`. This token is present in
# EVERY finance task, including the BULL-cn renamed editions whose tables carry
# Chinese display names with no ASCII prefix (e.g. task_4 公司分红 / lc_dividend).
# Table prefixes are a backup that only fire on the *-origin editions. Neither
# signal is a gold/answer literal -- both are universal schema/source vocabulary.
_CCKS_RE = re.compile(r"ccks_(stock|fund|macro)")
_PREFIX_RE = re.compile(r"^(lc|qt|mf|ed|in|fm)_")
_PREFIX_DOMAIN = {
    "lc": "stock", "qt": "stock",
    "mf": "fund",
    "ed": "macro", "in": "macro", "fm": "macro",
}
# Table-prefix routing is a thin signal (no Chinese, ASCII-prefix only), so it
# only fires when a single finance domain owns >= this many tables -- one stray
# `lc_*`/`in_*` table is not enough to classify.
_PREFIX_MIN_HITS = 3


# Require several fingerprint tables before committing to an EHR source. The
# eICU vocabulary is generic English (lab/patient/cost/treatment/...) and a
# SINGLE match collides with non-EHR schemas (e.g. a stray `Patient.md` in a
# general task). Measured separation is wide: a real eICU schema exposes ~10 of
# these tables and a real MIMIC schema ~13, while incidental collisions land at
# exactly 1 -- so this floor removes every observed false match with ample
# margin to true sources.
_EHR_MIN_HITS = 3


def _ehr_source_from_names(names: set[str]) -> str | None:
    """Return 'mimic'/'eicu' only when >= _EHR_MIN_HITS fingerprint tables match.

    Note this is a card-selection refinement applied AFTER the LLM has already
    judged the task to be EHR -- it never routes on its own. Returns None (->
    generic EHR primer) when the source is ambiguous.
    """
    if not names:
        return None
    upper = {n.upper() for n in names}
    lower = {n.lower() for n in names}
    mimic_hits = len({n for n in upper if n in _MIMIC_TABLES or n.startswith("D_")})
    if mimic_hits >= _EHR_MIN_HITS:
        return "mimic"
    eicu_hits = len(_EICU_TABLES & lower)
    if eicu_hits >= _EHR_MIN_HITS:
        return "eicu"
    return None


def _collect_schema_names(context_dir: Path) -> set[str]:
    """All table / file basenames the schema exposes (csv/json/doc + sqlite)."""
    names: set[str] = set()
    for kind in ("csv", "json", "doc"):
        folder = context_dir / kind
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    names.add(f.name.rsplit(".", 1)[0])
    db_folder = context_dir / "db"
    if db_folder.exists():
        for path in sorted(db_folder.glob("*.sqlite")):
            names.update(_sqlite_tables(path))
    return names


def rule_classify(task: PublicTask) -> tuple[str, str]:
    """Fully deterministic domain classification -- no LLM.

    Returns (domain, signal) with domain in fund/stock/macro/eicu/mimic/other.
    Every signal requires a strong match so unknown hidden-set schemas fall to
    'other' (which injects no domain note = baseline behaviour) rather than a
    wrong note:
      1. ccks_(stock|fund|macro) source-id token in knowledge.md  -> finance
      2. EHR schema fingerprint with >= _EHR_MIN_HITS tables       -> eicu/mimic
      3. >= _PREFIX_MIN_HITS finance-prefixed tables of one domain -> finance
      4. multiple finance families named -> the most frequent one
      5. otherwise -> other
    """
    context_dir = Path(task.context_dir)
    if not context_dir.exists():
        return "other", "no_context"

    knowledge = context_dir / "knowledge.md"
    ktext = (
        knowledge.read_text(encoding="utf-8", errors="replace")
        if knowledge.exists()
        else ""
    )
    fam_counts = Counter(_CCKS_RE.findall(ktext))
    if len(fam_counts) == 1:
        return next(iter(fam_counts)), "ccks_token"

    names = _collect_schema_names(context_dir)

    # EHR fingerprint is safe to route deterministically now that it requires
    # >= _EHR_MIN_HITS tables (generic 1-word collisions fall through).
    ehr = _ehr_source_from_names(names)
    if ehr:
        return ehr, "ehr_schema"

    prefix_counts: Counter[str] = Counter()
    for n in names:
        m = _PREFIX_RE.match(n.lower())
        if m:
            prefix_counts[_PREFIX_DOMAIN[m.group(1)]] += 1
    if prefix_counts:
        dom, cnt = prefix_counts.most_common(1)[0]
        if cnt >= _PREFIX_MIN_HITS:
            return dom, "prefix"

    if fam_counts:  # several finance families named; pick the most frequent
        return fam_counts.most_common(1)[0][0], "ccks_token_multi"
    return "other", "none"


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


def classify_domain(
    task: PublicTask, model: ModelAdapter | None = None
) -> DomainRouteResult:
    """Classify the task domain deterministically (no LLM).

    `model` is accepted for backward compatibility with the runner and ignored;
    every branch is a rule (see rule_classify). 'other' = no domain note.
    """
    start = time.time()
    domain, signal = rule_classify(task)
    return DomainRouteResult(
        domain=domain,
        raw=f"rule:{signal}",
        elapsed_sec=time.time() - start,
    )
