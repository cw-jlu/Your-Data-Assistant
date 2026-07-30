"""Extract domain reference thresholds from BIRD database_description/*.csv.

For each BIRD DB, walk its database_description directory and produce a
structured markdown reference containing:
  - Numeric thresholds (= "Normal range: X < N < Y")
  - Value mappings (= "M = male", "1 = most severe")
  - Code interpretations
  - Special data formats

Output: data/external/bird_thresholds/{db_id}.md (= one per DB)

Usage:
    python scripts/build_bird_threshold_refs.py \\
        --bird-root data/external/bird/dev_20240627/dev_databases \\
        --out-dir data/external/bird_thresholds
"""
from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path


# Regex patterns for evidence extraction from value_description.
_NORMAL_RANGE_RE = re.compile(
    r"Normal\s+range\s*:?\s*([^\n;|]+?)(?:\n|$|;)", re.IGNORECASE
)
_COMMONSENSE_RE = re.compile(
    r"Commonsense\s+evidence\s*:?\s*\n?\s*([^\n]+(?:\n[^\n]+){0,5})", re.IGNORECASE
)
# Value mapping: "1 = severe", "'M' = male", "+ refers to inpatient"
_MAPPING_RE = re.compile(
    r"['\"\b]?([\w+\-]+)['\"\b]?\s+(?:=|means|refers to|denotes|stands for)\s+([^;\n]+?)(?:\n|;|$)",
    re.IGNORECASE,
)
# Format hints — \b prevents matching "format" inside "information".
_FORMAT_RE = re.compile(
    r"\b(?:format|stored as)\b\s*:?\s*([^\n;]+?)(?:\n|;|$)", re.IGNORECASE
)


def _read_csv_safe(path: Path) -> list[dict[str, str]]:
    """Read CSV with utf-8-sig or latin-1 fallback."""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows


def _extract_thresholds(value_desc: str) -> list[str]:
    """Pull out Normal range expressions."""
    out = []
    for m in _NORMAL_RANGE_RE.finditer(value_desc):
        clean = m.group(1).strip().rstrip(".,;")
        # Remove residual quotes / weird chars
        clean = re.sub(r"\s+", " ", clean)
        if clean and len(clean) < 200:
            out.append(clean)
    return out


def _extract_value_mappings(value_desc: str) -> list[tuple[str, str]]:
    """Pull out 'X = Y' / 'X means Y' / 'X refers to Y' pairs."""
    out: list[tuple[str, str]] = []
    for m in _MAPPING_RE.finditer(value_desc):
        key = m.group(1).strip("'\"`")
        val = m.group(2).strip().rstrip(".,;")
        # Skip too-long values (= probably free text, not a mapping)
        if len(val) > 100 or len(key) > 30:
            continue
        # Skip obvious false positives
        if key.lower() in {"the", "this", "that", "it"}:
            continue
        if not key:
            continue
        out.append((key, val))
    return out


def _extract_format_hints(value_desc: str) -> list[str]:
    """Pull out 'format: X' / 'stored as X' hints."""
    out = []
    for m in _FORMAT_RE.finditer(value_desc):
        clean = m.group(1).strip().rstrip(".,;")
        if clean and len(clean) < 100:
            out.append(clean)
    return out


def build_db_reference(db_dir: Path) -> str:
    """Build markdown reference for one BIRD DB."""
    db_id = db_dir.name
    desc_dir = db_dir / "database_description"
    if not desc_dir.exists():
        return f"# Domain Reference: {db_id}\n\n(no database_description available)"

    sections = [f"# Domain Reference: {db_id}", ""]
    sections.append(
        "Authoritative threshold definitions, value mappings, and code "
        "interpretations for the schema. Use these when knowledge.md or "
        "narrative docs are ambiguous about boundaries (= e.g., 'normal' "
        "vs 'abnormal' lab values, gender codes, status codes)."
    )
    sections.append("")

    csv_files = sorted(desc_dir.glob("*.csv"))
    for csv_file in csv_files:
        table_name = csv_file.stem
        rows = _read_csv_safe(csv_file)
        if not rows:
            continue

        thresholds: list[str] = []
        mappings: list[str] = []
        formats: list[str] = []

        for row in rows:
            col_orig = (row.get("original_column_name") or "").strip()
            col_name = (row.get("column_name") or "").strip()
            col_desc = (row.get("column_description") or "").strip()
            val_desc = (row.get("value_description") or "").strip()
            if not val_desc:
                continue

            display_col = col_orig or col_name or "?"

            # Thresholds
            for rng in _extract_thresholds(val_desc):
                thresholds.append(f"- **{display_col}**: Normal range {rng}")

            # Value mappings (= but skip if range already extracted to avoid dup)
            if not _extract_thresholds(val_desc):
                vmaps = _extract_value_mappings(val_desc)
                if vmaps:
                    pretty = ", ".join(f"`{k}` = {v}" for k, v in vmaps[:6])
                    mappings.append(f"- **{display_col}**: {pretty}")

            # Format hints
            for fmt in _extract_format_hints(val_desc):
                formats.append(f"- **{display_col}**: format `{fmt}`")

        if not (thresholds or mappings or formats):
            continue

        sections.append(f"## Table: {table_name}")
        sections.append("")
        if thresholds:
            sections.append("### Thresholds (= reference normal ranges)")
            sections.extend(thresholds)
            sections.append("")
        if mappings:
            sections.append("### Value mappings")
            sections.extend(mappings)
            sections.append("")
        if formats:
            sections.append("### Format hints")
            sections.extend(formats)
            sections.append("")

    return "\n".join(sections)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bird-root",
        type=Path,
        default=Path("data/external/bird/dev_20240627/dev_databases"),
        help="Directory containing BIRD database subdirs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/external/bird_thresholds"),
        help="Output directory for per-DB markdown reference files",
    )
    args = ap.parse_args()

    if not args.bird_root.is_dir():
        print(f"error: bird-root not found: {args.bird_root}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    db_dirs = sorted(p for p in args.bird_root.iterdir() if p.is_dir())
    print(f"Processing {len(db_dirs)} BIRD databases…")
    total_th = 0
    total_map = 0
    for db_dir in db_dirs:
        md = build_db_reference(db_dir)
        out_path = args.out_dir / f"{db_dir.name}.md"
        out_path.write_text(md)
        n_th = md.count("Normal range")
        n_map = md.count("Value mappings")
        n_size = len(md)
        total_th += n_th
        total_map += n_map
        print(f"  {db_dir.name:<28} → {out_path.relative_to(Path.cwd()) if out_path.is_relative_to(Path.cwd()) else out_path}"
              f"  ({n_th} thresholds, {n_map} mapping sections, {n_size} chars)")
    print()
    print(f"Total thresholds extracted: {total_th}")
    print(f"Total mapping sections: {total_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
