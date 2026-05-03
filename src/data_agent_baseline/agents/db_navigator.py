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
                table_details[table] = {"columns": cols, "foreign_keys": fks}
            
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
    
    # 2. 关系提取 (保留核心能力)
    # 这里我们利用文件名和字段名进行简单的启发式关联，或调用 LLM 进行一次性关系总结
    # 为保持精简和速度，我们主要列出物理外键和同名字段
    all_columns = defaultdict(list)
    for db in dbs:
        for t_name, t_meta in db["tables"].items():
            for col in t_meta["columns"]:
                all_columns[col.lower()].append(f"{db['path']}.{t_name}")
    for csv in csvs:
        for col in csv["columns"]:
            all_columns[col.lower()].append(csv["path"])

    potential_matches = {k: v for k, v in all_columns.items() if len(v) > 1 and k in ("id", "customerid", "driverid", "productid")}

    # 3. 构造 Roadmap
    roadmap = {
        "data_assets": {
            "databases": dbs,
            "csv_files": csvs,
            "json_files": jsons
        },
        "relationships": {
            "potential_id_joins": potential_matches,
            "note": "Use read_doc('knowledge.md') for complex business join rules."
        }
    }
    
    return "=== DATA ROADMAP (SCHEMA & RELATIONS ONLY) ===\n" + json.dumps(roadmap, indent=2)
