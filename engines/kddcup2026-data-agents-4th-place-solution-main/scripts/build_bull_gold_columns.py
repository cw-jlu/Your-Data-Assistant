#!/usr/bin/env python3
"""Build a (question, SQL, gold_columns) dataset from the external BULL corpus.

BULL (FinSQL / CCKS) ships question + gold SQL pairs over three Chinese-finance
databases (ccks_fund / ccks_stock / ccks_macro), in both a Chinese (BULL-cn) and
an English (BULL-en) question edition. The gold result columns are obtained by
EXECUTING each gold SQL against the actual sqlite database and reading the
result header names from cursor.description -- this yields the canonical column
identifiers (e.g. `SecuAbbr`, `FundTypeName`) exactly as they appear in a result
table, which is the signal a column advisor / selector needs.

This is external training data (not the KDD test set); no DABench literal is
introduced. Output is split into one CSV per (language, domain) ->
  bull_cn_fund.csv  bull_cn_stock.csv  bull_cn_macro.csv
  bull_en_fund.csv  bull_en_stock.csv  bull_en_macro.csv
each with columns: question, SQL, gold_columns (JSON array string).

Usage:
  python3 scripts/build_bull_gold_columns.py \
      [--bull-root /tmp/phase2_source_check/BULL_full/BULL] \
      [--out-dir data/external/bull_gold_columns]
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

DEFAULT_BULL_ROOT = Path("/tmp/phase2_source_check/BULL_full/BULL")
DEFAULT_OUT_DIR = Path("data/external/bull_gold_columns")

# question-edition files to ingest (relative to BULL root), tagged by language
SPLITS = [
    ("cn", "BULL-cn/dev.json"), ("cn", "BULL-cn/train.json"),
    ("en", "BULL-en/dev.json"), ("en", "BULL-en/train.json"),
]
# db_name -> short domain label used in the output filename
DOMAIN_OF = {"ccks_fund": "fund", "ccks_stock": "stock", "ccks_macro": "macro"}
DB_NAMES = tuple(DOMAIN_OF)

# Drop rows whose output has >= this many columns: in BULL these are all
# `select *` "return every column" queries, which are noise for a column
# advisor (the gold is "all columns", not a chosen projection).
DEFAULT_MAX_COLS = 10


def _connect_dbs(bull_root: Path) -> dict[str, sqlite3.Connection]:
    """Open one read-only-ish connection per database (cn edition; table/column
    names are identical in the en edition, so it serves both question sets)."""
    cons: dict[str, sqlite3.Connection] = {}
    for db in DB_NAMES:
        path = bull_root / "database_cn" / db / f"{db}.sqlite"
        if path.exists():
            con = sqlite3.connect(path)
            con.text_factory = str
            cons[db] = con
    return cons


def _result_columns(con: sqlite3.Connection, sql: str) -> list[str] | None:
    """Return the result header names of a SELECT, or None if it does not run.

    Only the prepared statement's description is read; rows are not fetched, so
    this stays cheap even for large joins/aggregations.
    """
    try:
        cur = con.execute(sql)
        desc = cur.description
        cur.close()
        if not desc:
            return None
        return [d[0] for d in desc]
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bull-root", type=Path, default=DEFAULT_BULL_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-cols", type=int, default=DEFAULT_MAX_COLS,
                    help="drop rows whose gold_columns has >= this many entries "
                         "(BULL: all such rows are `select *`)")
    args = ap.parse_args()

    if not args.bull_root.exists():
        raise SystemExit(f"BULL root not found: {args.bull_root}")
    cons = _connect_dbs(args.bull_root)
    if not cons:
        raise SystemExit(f"no sqlite databases under {args.bull_root}/database_cn")

    # one bucket + dedup set per (language, domain)
    buckets: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    seen: dict[tuple[str, str], set[tuple[str, str]]] = {}
    n_total = n_skip_dup = n_skip_exec = n_skip_db = n_skip_wide = 0

    for lang, split in SPLITS:
        path = args.bull_root / split
        if not path.exists():
            print(f"  (skip missing {split})")
            continue
        for rec in json.load(open(path, encoding="utf-8")):
            n_total += 1
            question = (rec.get("question") or "").strip()
            sql = (rec.get("sql_query") or "").strip()
            db = rec.get("db_name")
            domain = DOMAIN_OF.get(db)
            if not question or not sql or domain is None:
                if domain is None:
                    n_skip_db += 1
                continue
            bucket = (lang, domain)
            key = (question, sql)
            if key in seen.setdefault(bucket, set()):
                n_skip_dup += 1
                continue
            con = cons.get(db)
            if con is None:
                n_skip_db += 1
                continue
            cols = _result_columns(con, sql)
            if not cols:
                n_skip_exec += 1
                continue
            if len(cols) >= args.max_cols:
                n_skip_wide += 1
                continue
            seen[bucket].add(key)
            buckets.setdefault(bucket, []).append(
                (question, sql, json.dumps(cols, ensure_ascii=False))
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"out dir: {args.out_dir}")
    grand = 0
    for (lang, domain), rows in sorted(buckets.items()):
        out = args.out_dir / f"bull_{lang}_{domain}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["question", "SQL", "gold_columns"])
            w.writerows(rows)
        grand += len(rows)
        print(f"  {out.name:22} {len(rows):5} rows")
    print(f"total written  : {grand}")
    print(f"  read={n_total} dup={n_skip_dup} no-db={n_skip_db} "
          f"exec-err={n_skip_exec} wide(>={args.max_cols}cols)={n_skip_wide}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
