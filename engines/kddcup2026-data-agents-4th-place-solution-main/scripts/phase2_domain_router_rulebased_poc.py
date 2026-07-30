#!/usr/bin/env python3
"""PoC: rule-based (LLM-free) domain classification for Phase-2 tasks.

Question under test: can we classify each task's domain (fund/stock/macro/
eicu/mimic) deterministically, without an LLM call?

Signal layers (cheap, local, no model):
  1. full knowledge.md scan for the source id token `ccks_(stock|fund|macro)`
     -- this is the demo author's own source label and is present in EVERY
     finance task, including the BULL-cn renamed editions whose tables carry no
     ASCII prefix (e.g. task_4 公司分红 instead of lc_dividend).
  2. table / file prefix (lc|qt -> stock, mf -> fund, ed|in|fm -> macro) --
     backup that fires only on the *-origin editions.
  3. EHR schema detection (eicu lowercase clinical tables / mimic UPPERCASE +
     D_* dictionaries) -- EHR knowledge.md is an empty "Step3 skipped" stub, so
     it has no ccks token and must be resolved from the schema.

These signals carry NO gold/answer literals -- `ccks_stock` is a database id, the
prefixes and EHR table names are universal schema vocabulary. No test-data leak.

Run:  python3 scripts/phase2_domain_router_rulebased_poc.py [INPUT_DIR]
"""
from __future__ import annotations

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path(
    "data/phase2_demo/demo_samples_phase2/input"
)

_CCKS_RE = re.compile(r"ccks_(stock|fund|macro)")
_PREFIX_RE = re.compile(r"^(lc|qt|mf|ed|in|fm)_")
_PREFIX_DOMAIN = {
    "lc": "stock", "qt": "stock",
    "mf": "fund",
    "ed": "macro", "in": "macro", "fm": "macro",
}
# A single finance-prefixed table is a thin signal; require this many of one
# domain before classifying by prefix.
_PREFIX_MIN_HITS = 3

_MIMIC_TABLES = {
    "ADMISSIONS", "PATIENTS", "LABEVENTS", "PROCEDURES_ICD", "DIAGNOSES_ICD",
    "PRESCRIPTIONS", "ICUSTAYS", "CHARTEVENTS", "MICROBIOLOGYEVENTS",
    "D_ICD_PROCEDURES", "D_ICD_DIAGNOSES", "D_LABITEMS", "D_ITEMS",
}
_EICU_TABLES = {
    "lab", "patient", "treatment", "diagnosis", "medication", "allergy",
    "microlab", "intakeoutput", "vitalperiodic", "cost", "apacheapsvar",
}


def _schema_names(context: Path) -> set[str]:
    names: set[str] = set()
    for kind in ("csv", "json", "doc"):
        d = context / kind
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    names.add(f.stem)
    dbd = context / "db"
    if dbd.exists():
        for s in dbd.glob("*.sqlite"):
            try:
                con = sqlite3.connect(s)
                names.update(
                    r[0] for r in con.execute(
                        "select name from sqlite_master where type in ('table','view')"
                    )
                )
                con.close()
            except Exception:
                pass
    return names


# Require >= _EHR_MIN_HITS fingerprint tables: eICU words (lab/patient/cost/...)
# are generic and a single match collides with non-EHR schemas. Real eICU ~10
# hits, real MIMIC ~13, incidental collisions == 1.
_EHR_MIN_HITS = 3


def _detect_ehr_source(names: set[str]) -> str | None:
    if not names:
        return None
    upper = {n.upper() for n in names}
    lower = {n.lower() for n in names}
    mimic_hits = len({n for n in upper if n in _MIMIC_TABLES or n.startswith("D_")})
    if mimic_hits >= _EHR_MIN_HITS:
        return "mimic"
    if len(_EICU_TABLES & lower) >= _EHR_MIN_HITS:
        return "eicu"
    return None


def rule_classify(context: Path) -> tuple[str, str]:
    """Fully deterministic domain classification (no LLM).

    Returns (domain, signal); domain in fund/stock/macro/eicu/mimic/other.
    Every signal requires a strong match, so unknown schemas fall to 'other'
    (= no domain note injected) rather than a wrong one.
      1. ccks_(stock|fund|macro) source-id token  -> finance (specific id)
      2. EHR fingerprint with >= _EHR_MIN_HITS     -> eicu/mimic
      3. >= _PREFIX_MIN_HITS finance-prefixed cols -> finance
      4. multiple finance families named           -> most frequent
      5. otherwise                                 -> other
    """
    km = context / "knowledge.md"
    ktext = km.read_text(encoding="utf-8", errors="replace") if km.exists() else ""

    fam_counts = Counter(_CCKS_RE.findall(ktext))
    if len(fam_counts) == 1:
        return next(iter(fam_counts)), "ccks_token"

    names = _schema_names(context)

    ehr = _detect_ehr_source(names)
    if ehr:
        return ehr, "ehr_schema"

    prefix_counts: Counter = Counter()
    for n in names:
        m = _PREFIX_RE.match(n.lower())
        if m:
            prefix_counts[_PREFIX_DOMAIN[m.group(1)]] += 1
    if prefix_counts:
        dom, cnt = prefix_counts.most_common(1)[0]
        if cnt >= _PREFIX_MIN_HITS:
            return dom, "prefix"

    if fam_counts:
        return fam_counts.most_common(1)[0][0], "ccks_token_multi"
    return "other", "none"


def main() -> int:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    if not input_dir.exists():
        print(f"input dir not found: {input_dir}", file=sys.stderr)
        return 2

    tasks = sorted(
        (p for p in input_dir.glob("task_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )

    rows = []
    by_domain: dict[str, int] = {}
    by_signal: dict[str, int] = {}
    for t in tasks:
        domain, signal = rule_classify(t / "context")
        rows.append((t.name, domain, signal))
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_signal[signal] = by_signal.get(signal, 0) + 1

    print(f"{'task':8} {'domain':8} {'signal':16}")
    print("-" * 34)
    for name, domain, signal in rows:
        flag = "  <-- other (no note)" if domain == "other" else ""
        print(f"{name:8} {domain:8} {signal:16}{flag}")

    total = len(rows)
    other = by_domain.get("other", 0)
    resolved = total - other
    print("\n=== coverage ===")
    print(f"classified to a domain (no LLM): {resolved}/{total} "
          f"({100*resolved/total:.0f}%)")
    print(f"-> other (no note injected): {other}/{total}")
    print("by domain :", ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items())))
    print("by signal :", ", ".join(f"{k}={v}" for k, v in sorted(by_signal.items())))
    # false-positive guard: any finance/EHR routing on a non-BULL/non-EHR set is suspect
    fin = sum(by_domain.get(d, 0) for d in ("fund", "stock", "macro", "eicu", "mimic"))
    print(f"domain-routed: {fin}/{total}  (on a general set this should be ~0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
