"""Build per-DB domain reference from BIRD database_description CSVs.

For each BIRD DB, parse `database_description/<table>.csv` and emit a structured
markdown file at `data/external/domain_db/<db_id>.md`. Each column becomes one
fact entry with:
  - Table name
  - Original column name + display name
  - data_format
  - column_description
  - value_description (= contains thresholds, value mappings, commonsense evidence)

This file is consumed by the `consult_domain_db` agent tool at runtime via
keyword retrieval — it serves as an external schema reference for the task's DB.

Usage:
    python scripts/build_domain_db.py \\
        --bird-root data/external/bird/dev_20240627/dev_databases \\
        --out-dir data/external/domain_db
"""
from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path


def _read_csv_safe(path: Path) -> list[dict[str, str]]:
    """Read CSV with BOM stripped from headers, falling back through encodings."""
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return []
    # Strip BOM in case latin-1 fallback was used and left it literal.
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("﻿", "")
    rows = list(csv.DictReader(io.StringIO(text)))
    # Some CSVs from BIRD have a literal "ï»¿" prefix on the first header when
    # read via latin-1; normalize that too.
    fixed: list[dict[str, str]] = []
    for r in rows:
        fixed.append({(k.replace("ï»¿", "").strip() if k else k): v for k, v in r.items()})
    return fixed


def _clean(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip())


def _format_value_desc(text: str) -> str:
    """Format multi-line value_description for readability.

    Preserves embedded line breaks for "Normal range" / value enumerations.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Newlines between "K: V" entries are useful info — keep them as " | ".
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
    cleaned = re.sub(r"\n+", " | ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_db_doc(db_dir: Path) -> str:
    db_id = db_dir.name
    desc_dir = db_dir / "database_description"
    if not desc_dir.exists():
        return f"# Domain Reference: {db_id}\n\n(no database_description available)\n"

    out: list[str] = [f"# Domain Reference: {db_id}", ""]
    out.append(
        "Structured schema reference extracted from the database's official "
        "column descriptions. Each entry below describes one field: its type, "
        "what it represents, and any value-level notes (= ranges, codes, "
        "format hints). Use this as a lookup when interpreting question terms."
    )
    out.append("")

    csv_files = sorted(desc_dir.glob("*.csv"))
    for csv_file in csv_files:
        table_name = csv_file.stem
        rows = _read_csv_safe(csv_file)
        if not rows:
            continue
        out.append(f"## Table: {table_name}")
        out.append("")
        for row in rows:
            orig = _clean(row.get("original_column_name") or "")
            disp = _clean(row.get("column_name") or "")
            col_desc = _clean(row.get("column_description") or "")
            dtype = _clean(row.get("data_format") or "")
            val_desc = _format_value_desc(row.get("value_description") or "")

            field = orig or disp or "?"
            if not (col_desc or val_desc or dtype):
                continue  # skip fully-empty rows

            out.append(f"### Field: {table_name}.{field}")
            if disp and disp != orig:
                out.append(f"- Display name: {disp}")
            if dtype:
                out.append(f"- Type: {dtype}")
            if col_desc:
                out.append(f"- Description: {col_desc}")
            if val_desc:
                out.append(f"- Value notes: {val_desc}")
            out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bird-root",
        type=Path,
        default=Path("data/external/bird/dev_20240627/dev_databases"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/external/domain_db"),
    )
    args = ap.parse_args()

    if not args.bird_root.is_dir():
        print(f"error: bird-root not found: {args.bird_root}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    db_dirs = sorted(p for p in args.bird_root.iterdir() if p.is_dir())
    print(f"Processing {len(db_dirs)} databases…")
    total_chars = 0
    for db_dir in db_dirs:
        doc = build_db_doc(db_dir)
        out_path = args.out_dir / f"{db_dir.name}.md"
        out_path.write_text(doc)
        n_fields = doc.count("### Field:")
        total_chars += len(doc)
        print(f"  {db_dir.name:<28} → {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}"
              f"  ({n_fields} fields, {len(doc)} chars)")
    print(f"\nTotal output: {total_chars} chars across {len(db_dirs)} DBs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
