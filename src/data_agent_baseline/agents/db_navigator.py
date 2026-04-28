"""
Schema 知识图谱导航器。
在基础的 Schema 扫描之上，增加：
1. 跨文件关联识别（同名字段 → JOIN 路径）
2. knowledge.md 业务定义提取
"""
import sqlite3
import csv
import json
from pathlib import Path
from collections import defaultdict


def get_data_roadmap(context_dir: Path) -> str:
    """
    扫描 context 目录下的 .db, .csv, .json 文件，
    构建 Schema 知识图谱并生成结构化路线图。
    """
    # 收集所有数据源的字段信息
    sources: dict[str, list[str]] = {}  # "source_name" -> [col1, col2, ...]
    lines = ["=== DATA SCHEMA KNOWLEDGE GRAPH ==="]

    # --- 1. 扫描 DB ---
    db_files = sorted(context_dir.rglob("*.db"))
    for db_path in db_files:
        rel = db_path.relative_to(context_dir)
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (table_name,) in cursor.fetchall():
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                cols = cursor.fetchall()
                col_names = [c[1] for c in cols]
                col_desc = [f"{c[1]}({c[2]})" for c in cols]
                source_key = f"{rel}::{table_name}"
                sources[source_key] = col_names
                lines.append(f"\n[DB] {rel} → Table '{table_name}'")
                lines.append(f"  Columns: {', '.join(col_desc)}")
            conn.close()
        except Exception as e:
            lines.append(f"\n[DB] {rel} (Error: {e})")

    # --- 2. 扫描 CSV ---
    csv_files = sorted(context_dir.rglob("*.csv"))
    for csv_path in csv_files:
        rel = csv_path.relative_to(context_dir)
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    sources[str(rel)] = header
                    lines.append(f"\n[CSV] {rel}")
                    lines.append(f"  Columns: {', '.join(header)}")
        except Exception as e:
            lines.append(f"\n[CSV] {rel} (Error: {e})")

    # --- 3. 扫描 JSON ---
    json_files = sorted(context_dir.rglob("*.json"))
    for json_path in json_files:
        rel = json_path.relative_to(context_dir)
        try:
            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                keys = []
                if isinstance(data, dict):
                    keys = list(data.keys())
                elif isinstance(data, list) and data and isinstance(data[0], dict):
                    keys = list(data[0].keys())
                if keys:
                    sources[str(rel)] = keys
                    lines.append(f"\n[JSON] {rel}")
                    lines.append(f"  Keys: {', '.join(keys)}")
        except Exception as e:
            lines.append(f"\n[JSON] {rel} (Error: {e})")

    # --- 4. 发现跨源关联 (JOIN Hints) ---
    col_to_sources: dict[str, list[str]] = defaultdict(list)
    for src, cols in sources.items():
        for col in cols:
            col_lower = col.lower().strip()
            col_to_sources[col_lower].append(src)

    join_hints = []
    for col, srcs in col_to_sources.items():
        if len(srcs) > 1:
            join_hints.append(f"  '{col}' links: {' <--> '.join(srcs)}")

    if join_hints:
        lines.append("\n[JOIN HINTS] Shared columns across sources:")
        lines.extend(join_hints)

    # --- 5. 提取 knowledge.md 摘要 ---
    for k_name in ["knowledge.md", "Knowledge.md"]:
        k_path = context_dir / k_name
        if k_path.exists():
            try:
                text = k_path.read_text(encoding="utf-8", errors="replace")
                # 提取定义性内容（以 - 或 * 开头的行，或含有 ":" 的行）
                defs = []
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped and (
                        stripped.startswith("- ") or
                        stripped.startswith("* ") or
                        ":" in stripped or
                        stripped.startswith("##")
                    ):
                        defs.append(f"  {stripped}")
                    if len(defs) >= 30:  # 控制长度
                        break
                if defs:
                    lines.append(f"\n[KNOWLEDGE] {k_name} (key definitions):")
                    lines.extend(defs)
            except Exception:
                pass
            break

    return "\n".join(lines)
