#!/usr/bin/env python3
"""Dump schema + 5 random sample rows for every CSV/JSON/SQLite table in demo_samples_phase2."""

import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "demo_samples_phase2" / "input"
random.seed(42)


def schema_key(cols: list[str]) -> str:
    return "|".join(sorted(cols))


def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    if len(df) <= n:
        return df.to_dict(orient="records")
    return df.sample(n, random_state=42).to_dict(orient="records")


def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False, nrows=500)
    except Exception as e:
        print(f"  [WARN] CSV read failed: {path} — {e}")
        return None


def load_json(path: Path) -> pd.DataFrame | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data[:500])
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return pd.DataFrame(v[:500])
            return pd.DataFrame([data])
        return None
    except Exception as e:
        print(f"  [WARN] JSON read failed: {path} — {e}")
        return None


def collect_file_tables(root: Path) -> dict:
    """Return {table_name: {source, columns, dtypes, sample_rows, row_count, files}}."""
    tables: dict[str, dict] = {}
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        context = task_dir / "context"
        if not context.exists():
            continue

        csv_dir = context / "csv"
        if csv_dir.exists():
            for f in sorted(csv_dir.glob("*.csv")):
                name = f.stem
                if name in tables:
                    tables[name]["files"].append(str(f.relative_to(root)))
                    continue
                df = load_csv(f)
                if df is None:
                    continue
                tables[name] = {
                    "source": "csv",
                    "columns": list(df.columns),
                    "dtypes": {c: str(df[c].dtype) for c in df.columns},
                    "row_count": len(df),
                    "sample_rows": sample_rows(df),
                    "files": [str(f.relative_to(root))],
                }

        json_dir = context / "json"
        if json_dir.exists():
            for f in sorted(json_dir.glob("*.json")):
                name = f.stem
                if name in tables:
                    tables[name]["files"].append(str(f.relative_to(root)))
                    continue
                df = load_json(f)
                if df is None or df.empty:
                    continue
                tables[name] = {
                    "source": "json",
                    "columns": list(df.columns),
                    "dtypes": {c: str(df[c].dtype) for c in df.columns},
                    "row_count": len(df),
                    "sample_rows": sample_rows(df),
                    "files": [str(f.relative_to(root))],
                }

    return tables


def collect_sqlite_tables(root: Path) -> dict:
    """Return {table_name: {source, columns, dtypes, sample_rows, row_count, files}} from SQLite."""
    tables: dict[str, dict] = {}

    for db_path in sorted(root.rglob("*.sqlite")):
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tbl_names = [r[0] for r in cursor.fetchall()]
            for tbl in tbl_names:
                if tbl in tables:
                    tables[tbl]["files"].append(str(db_path.relative_to(root)))
                    continue
                try:
                    cursor.execute(f"PRAGMA table_info([{tbl}])")
                    info = cursor.fetchall()
                    columns = [r[1] for r in info]
                    dtypes = {r[1]: r[2] for r in info}

                    cursor.execute(f"SELECT COUNT(*) FROM [{tbl}]")
                    row_count = cursor.fetchone()[0]

                    df = pd.read_sql_query(f"SELECT * FROM [{tbl}] ORDER BY RANDOM() LIMIT 5", conn)
                    tables[tbl] = {
                        "source": "sqlite",
                        "columns": columns,
                        "dtypes": dtypes,
                        "row_count": row_count,
                        "sample_rows": df.to_dict(orient="records"),
                        "files": [str(db_path.relative_to(root))],
                    }
                except Exception as e:
                    print(f"  [WARN] SQLite table [{tbl}] in {db_path.name}: {e}")
            conn.close()
        except Exception as e:
            print(f"  [WARN] SQLite open failed: {db_path} — {e}")

    return tables


def merge_tables(file_tables: dict, sqlite_tables: dict) -> dict:
    merged = {}
    all_names = sorted(set(file_tables) | set(sqlite_tables))
    for name in all_names:
        sources = []
        if name in file_tables:
            sources.append(file_tables[name])
        if name in sqlite_tables:
            sources.append(sqlite_tables[name])
        primary = sources[0]
        all_files = []
        all_source_types = set()
        for s in sources:
            all_files.extend(s["files"])
            all_source_types.add(s["source"])
        merged[name] = {
            "source_types": sorted(all_source_types),
            "columns": primary["columns"],
            "dtypes": primary["dtypes"],
            "num_columns": len(primary["columns"]),
            "row_count_sample_file": primary["row_count"],
            "sample_rows": primary["sample_rows"],
            "num_files": len(all_files),
        }
    return merged


def print_report(merged: dict):
    print("=" * 100)
    print(f"TOTAL UNIQUE TABLES: {len(merged)}")
    print("=" * 100)

    by_source = defaultdict(int)
    for info in merged.values():
        for s in info["source_types"]:
            by_source[s] += 1
    print("\nBy source type:")
    for s, c in sorted(by_source.items()):
        print(f"  {s}: {c} tables")

    col_counts = [info["num_columns"] for info in merged.values()]
    print(f"\nColumn count range: {min(col_counts)} — {max(col_counts)}")
    print(f"Median columns: {sorted(col_counts)[len(col_counts) // 2]}")

    print("\n" + "=" * 100)
    print("SCHEMA + SAMPLES PER TABLE")
    print("=" * 100)

    for name in sorted(merged.keys()):
        info = merged[name]
        print(f"\n{'─' * 80}")
        print(f"TABLE: {name}")
        print(
            f"  Sources: {', '.join(info['source_types'])}  |  "
            f"Columns: {info['num_columns']}  |  "
            f"Rows (sample file): {info['row_count_sample_file']}  |  "
            f"Appears in {info['num_files']} files"
        )
        print("  Schema:")
        for col in info["columns"]:
            dtype = info["dtypes"].get(col, "?")
            print(f"    {col:50s}  {dtype}")
        print(f"  Sample rows ({min(5, len(info['sample_rows']))}):")
        for i, row in enumerate(info["sample_rows"][:5]):
            compact = {
                k: (v if not isinstance(v, float) or not pd.isna(v) else None)
                for k, v in row.items()
            }
            print(f"    [{i}] {json.dumps(compact, ensure_ascii=False, default=str)[:300]}")


def main():
    print("Scanning", DATA_ROOT)
    print()
    file_tables = collect_file_tables(DATA_ROOT)
    print(f"Collected {len(file_tables)} tables from CSV/JSON")
    sqlite_tables = collect_sqlite_tables(DATA_ROOT)
    print(f"Collected {len(sqlite_tables)} tables from SQLite")
    merged = merge_tables(file_tables, sqlite_tables)
    print_report(merged)

    out_path = DATA_ROOT.parent / "all_schemas.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\nFull schema+samples written to: {out_path}")


if __name__ == "__main__":
    main()
