#!/usr/bin/env python3
"""PoC: MiniLM retrieval of BULL (question -> gold_columns) for Phase-2 tasks.

For each Phase-2 demo task we (1) route its domain with the deterministic
domain router, (2) embed the task question with all-MiniLM-L6-v2, (3) cosine
top-k against the matching BULL domain corpus (bull_{lang}_{domain}.csv), and
(4) show the retrieved (question, gold_columns) plus the task's ACTUAL gold
columns from the demo gold.csv -- so we can eyeball how usable the retrieved
column suggestions are.

This is the exp_140 retrieval-ICL pattern re-pointed from BIRD to BULL, for
column selection instead of SQL patterns.

Usage:
  .venv/bin/python scripts/phase2_bull_icl_retrieval_poc.py [--task task_4] [--k 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "src")
from experiments.exp_166_domain_rules.domain_router import classify_domain  # noqa: E402

REPO = Path(".")
DEMO_IN = REPO / "data/phase2_demo/demo_samples_phase2/input"
DEMO_OUT = REPO / "data/phase2_demo/demo_samples_phase2/output"
BULL_DIR = REPO / "data/external/bull_gold_columns"
CACHE_DIR = REPO / "artifacts/bull_icl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FINANCE = {"fund", "stock", "macro"}

# a representative default sample spanning {en,cn} x {fund,stock,macro}
DEFAULT_TASKS = ["task_4", "task_60", "task_21", "task_45", "task_2", "task_18"]


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
    return _MODEL


def _load_bull(lang: str, domain: str) -> tuple[list[dict], np.ndarray]:
    """Return (rows, embeddings) for one BULL corpus, building+caching as needed."""
    csv_path = BULL_DIR / f"bull_{lang}_{domain}.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = CACHE_DIR / f"{lang}_{domain}.npy"
    if emb_path.exists():
        embs = np.load(emb_path)
        if embs.shape[0] == len(rows):
            return rows, embs
    print(f"  [embedding {len(rows)} {lang}/{domain} questions ...]", flush=True)
    embs = _model().encode(
        [r["question"] for r in rows],
        normalize_embeddings=True, batch_size=128, show_progress_bar=False,
    ).astype(np.float32)
    np.save(emb_path, embs)
    return rows, embs


def _demo_gold_columns(task_id: str) -> list[str]:
    gold = DEMO_OUT / task_id / "gold.csv"
    if not gold.exists():
        return []
    with open(gold, encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    return header


def _route(task_id: str) -> tuple[str, str]:
    ctx = DEMO_IN / task_id / "context"
    q = json.load(open(DEMO_IN / task_id / "task.json"))["question"]
    domain = classify_domain(SimpleNamespace(question=q, context_dir=str(ctx))).domain
    return q, domain


def run_task(task_id: str, k: int) -> None:
    q, domain = _route(task_id)
    lang = "cn" if _has_cjk(q) else "en"
    print("=" * 96)
    print(f"{task_id}  domain={domain}  lang={lang}")
    print(f"  Q: {q[:140]}")
    demo_cols = _demo_gold_columns(task_id)
    print(f"  DEMO gold columns: {demo_cols}")
    if domain not in FINANCE:
        print("  -> non-finance domain, no BULL corpus; skipping retrieval.")
        return

    rows, embs = _load_bull(lang, domain)
    qv = _model().encode([q], normalize_embeddings=True).astype(np.float32)[0]
    sims = embs @ qv
    top = np.argsort(-sims)[:k]
    demo_set = {c.strip().lower() for c in demo_cols}
    print(f"  --- top-{k} BULL neighbours ({lang}/{domain}) ---")
    for rank, i in enumerate(top, 1):
        cols = json.loads(rows[i]["gold_columns"])
        overlap = demo_set & {c.strip().lower() for c in cols}
        mark = f"  <=overlap {sorted(overlap)}" if overlap else ""
        print(f"   {rank}. sim={sims[i]:.3f}  cols={cols}{mark}")
        print(f"       q: {rows[i]['question'][:110]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", help="task id(s); repeatable")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    tasks = args.task or DEFAULT_TASKS
    for t in tasks:
        run_task(t, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
