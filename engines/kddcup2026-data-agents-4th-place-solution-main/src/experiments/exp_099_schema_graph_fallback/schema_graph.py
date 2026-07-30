"""LLM-derived schema relationships injected into the preamble.

Replaces the earlier Jaccard/MinHash heuristic. The LLM judgment is
pre-computed once per task by `scripts/precompute_schema_fk.py` and cached
under `artifacts/schema_fk_cache/<task_id>.json`. This module just reads
the cache and formats the hints for the agent.

Why LLM not Jaccard (lessons from exp_099_jaccard_v1):
- Jaccard is symmetric; real FKs are asymmetric (child → parent containment).
  Jaccard underestimated containment, e.g. reported `connected.atom_id ↔
  atom.atom_id` as Jaccard=0.50 even though every connected.atom_id ⊆ atom.
- Jaccard cannot tell duplicate-copy (CSV mirror of SQLite table) apart
  from genuine FK; both look like Jaccard≈1.0. The LLM tags duplicates
  separately so the agent doesn't double-count rows.
- Jaccard cannot drop semantically-irrelevant columns; the LLM produces an
  orphan-columns hint that's directly useful for output column scoping.

Cache fields (schema in scripts/precompute_schema_fk.py):
  joins[]:           [{from, to, kind, confidence, reason}]
  orphan_columns[]:  ["fileX[::tableX].col"]
"""
from __future__ import annotations

import json
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask

CACHE_DIR = Path("artifacts/schema_fk_cache")

_MAX_JOINS_REPORTED = 12
_MAX_ORPHANS_REPORTED = 15


def _load_cache(task_id: str) -> dict | None:
    p = CACHE_DIR / f"{task_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_schema_graph_section(task: PublicTask) -> str:
    """Format LLM-judged schema relationships from cache as a preamble section.

    Returns empty string if cache missing or no useful hints.
    """
    # exp_099 v4 (2026-05-07): hint section disabled.
    #
    # All three tested formats hurt vs the no-hint baseline:
    #   v1 (Jaccard MinHash):           -0.099 vs single-attempt floor
    #   v2 (LLM full + orphan):         -0.027 (orphan removed answer-cols)
    #   v3 (LLM FK + duplicate, n=15):  -0.286 on the 7 tasks WITH hint
    #                                   (+0.075 on the 8 tasks WITHOUT hint)
    #
    # Failure mode: imperative-tone hints ("do NOT double-count", "FK joins")
    # bias the agent toward forced joins or hint-defined output, even when
    # the question doesn't need that schema relationship. The runtime ReAct
    # agent already does fine schema linking when given the raw data —
    # adding a precomputed summary on top only introduces a cheaper-but-
    # less-context-aware second opinion that the agent can't easily ignore.
    #
    # Schema-only fallback in preamble.py is preserved (and was responsible
    # for the +0.075 improvement on the no-hint cohort): when a single huge
    # file would push the preamble over budget, swap it for a column-list
    # summary instead of dropping it entirely. That intervention adds
    # information without dictating behavior.
    _ = task  # quiet unused warning
    return ""
