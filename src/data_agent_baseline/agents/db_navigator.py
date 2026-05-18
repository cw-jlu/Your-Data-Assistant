import sqlite3
import json
import re
from pathlib import Path
from collections import defaultdict

def _scan_databases(context_dir: Path):
    """扫描数据库的表名、字段和物理主外键。"""
    db_info = []
    for db_file in context_dir.rglob("*.db"):
        rel_path = db_file.relative_to(context_dir)
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            table_details = {}
            for table in tables:
                cursor.execute(f"PRAGMA foreign_key_list({table});")
                fks = [{"from": row[3], "to_table": row[2], "to_column": row[4]} for row in cursor.fetchall()]
                cursor.execute(f"PRAGMA table_info({table});")
                cols = [row[1] for row in cursor.fetchall()]
                # 获取行数（COUNT(*)在SQLite上很快）
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM [{table}];")
                    row_count = cursor.fetchone()[0]
                except Exception:
                    row_count = None
                table_details[table] = {"columns": cols, "foreign_keys": fks, "row_count": row_count}
            
            db_info.append({"path": str(rel_path.as_posix()), "tables": table_details})
            conn.close()
        except Exception:
            pass
    return db_info

def _get_csv_json_schemas(context_dir: Path):
    """获取 CSV 和 JSON 的字段名清单。"""
    csv_files = []
    for f in context_dir.rglob("*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(f, nrows=0)
            csv_files.append({"path": str(f.relative_to(context_dir).as_posix()), "columns": df.columns.tolist()})
        except Exception:
            csv_files.append({"path": str(f.relative_to(context_dir).as_posix()), "columns": []})
            
    json_files = []
    for f in context_dir.rglob("*.json"):
        json_files.append({"path": str(f.relative_to(context_dir).as_posix())})
    return csv_files, json_files

def get_data_roadmap(context_dir: Path, model=None):
    """
    生成包含关联关系但不含数据预览的 Roadmap。
    """
    # 1. 基础结构扫描
    dbs = _scan_databases(context_dir)
    csvs, jsons = _get_csv_json_schemas(context_dir)

    # 主动扫描 knowledge 文档：knowledge.md 包含业务规则和定义，必须优先阅读
    knowledge_docs = [
        str(f.relative_to(context_dir).as_posix())
        for f in sorted(context_dir.rglob("*.md"))
        if "knowledge" in f.name.lower()
    ]
    
    other_docs = [
        str(f.relative_to(context_dir).as_posix())
        for f in sorted(context_dir.rglob("*.md"))
        if "knowledge" not in f.name.lower()
    ] + [
        str(f.relative_to(context_dir).as_posix())
        for f in sorted(context_dir.rglob("*.txt"))
    ]

    # 2. 同名字段统计：出现在 2+ 个文件中的字段，全部列出
    all_columns = defaultdict(set)
    for db in dbs:
        for t_meta in db["tables"].values():
            for col in t_meta["columns"]:
                all_columns[col.lower()].add(db["path"])
    for csv_f in csvs:
        for col in csv_f["columns"]:
            all_columns[col.lower()].add(csv_f["path"])

    shared_fields = {
        k: sorted(v)
        for k, v in sorted(all_columns.items(), key=lambda x: -len(x[1]))
        if len(v) > 1
    }

    # 3. 构造 Roadmap
    roadmap = {
        "IMPORTANT": (
            "READ knowledge_docs FIRST before writing any query or code. "
            "They contain business definitions, thresholds, and domain rules that determine correct answers."
        ) if knowledge_docs else None,
        "data_assets": {
            "knowledge_docs": knowledge_docs,
            "doc_files": other_docs,
            "databases": dbs,
            "csv_files": csvs,
            "json_files": jsons
        },
        "relationships": {
            "shared_fields": shared_fields,
            "note": (
                "These fields share the same name across multiple files. "
                "This is a NAME HINT only — do NOT assume they can be JOINed directly. "
                "Always verify the actual join semantics in knowledge_docs before using them as join keys."
            )
        }
    }

    # 去掉 IMPORTANT 为 None 的情况（没有 knowledge 文档时不注入）
    if not knowledge_docs:
        del roadmap["IMPORTANT"]

    return "=== DATA ROADMAP (SCHEMA & RELATIONS ONLY) ===\n" + json.dumps(roadmap, indent=2, ensure_ascii=False)
