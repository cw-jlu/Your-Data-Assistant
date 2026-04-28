import sqlite3
import csv
import json
from pathlib import Path
from collections import defaultdict

def get_data_roadmap(context_dir: Path) -> str:
    sources: dict[str, list[str]] = {}
    lines = ["=== DATA SCHEMA KNOWLEDGE GRAPH ==="]
    for db_path in sorted(context_dir.rglob("*.db")):
        rel = db_path.relative_to(context_dir)
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cursor.fetchall():
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                cols = cursor.fetchall()
                sources[f"{rel}::{table_name}"] = [c[1] for c in cols]
                lines.append(f"\n[DB] {rel} → Table '{table_name}': {', '.join([f'{c[1]}({c[2]})' for c in cols])}")
            conn.close()
        except Exception: pass
    for csv_path in sorted(context_dir.rglob("*.csv")):
        rel = csv_path.relative_to(context_dir)
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                header = next(csv.reader(f), None)
                if header:
                    sources[str(rel)] = header
                    lines.append(f"\n[CSV] {rel}: {', '.join(header)}")
        except Exception: pass
    for json_path in sorted(context_dir.rglob("*.json")):
        rel = json_path.relative_to(context_dir)
        try:
            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                keys = []
                if isinstance(data, dict): keys = list(data.keys())
                elif isinstance(data, list) and data and isinstance(data[0], dict): keys = list(data[0].keys())
                if keys:
                    sources[str(rel)] = keys
                    lines.append(f"\n[JSON] {rel}: {', '.join(keys)}")
        except Exception: pass
    col_to_srcs = defaultdict(list)
    for src, cols in sources.items():
        for col in cols: col_to_srcs[col.lower().strip()].append(src)
    hints = [f"  '{col}' links: {' <--> '.join(srcs)}" for col, srcs in col_to_srcs.items() if len(srcs) > 1]
    if hints: lines.append("\n[JOIN HINTS]"); lines.extend(hints)
    for k_name in ["knowledge.md", "Knowledge.md"]:
        k_path = context_dir / k_name
        if k_path.exists():
            try:
                text = k_path.read_text(encoding="utf-8", errors="replace")
                defs = [f"  {l.strip()}" for l in text.splitlines() if l.strip() and (l.strip().startswith(("-", "*")) or ":" in l or l.strip().startswith("##"))]
                if defs: lines.append(f"\n[KNOWLEDGE] {k_name}:"); lines.extend(defs[:30])
            except Exception: pass
            break
    return "\n".join(lines)
