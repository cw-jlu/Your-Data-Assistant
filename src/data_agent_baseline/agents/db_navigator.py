import json
import re
import sqlite3
from pathlib import Path
from collections import defaultdict
from data_agent_baseline.agents.model import ModelMessage

def _scan_databases(context_dir: Path):
    """扫描数据库，提取物理 Schema 和外键关系。"""
    dbs = []
    sources = {}
    fk_relations = []
    for db_path in sorted(context_dir.rglob("*")):
        if db_path.suffix.lower() not in [".db", ".sqlite"]:
            continue
        rel = str(db_path.relative_to(context_dir))
        db_info = {"path": rel, "tables": []}
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
            for table in tables:
                cursor.execute(f"PRAGMA table_info('{table}');")
                cols = cursor.fetchall()
                col_desc = [f"{c[1]}({c[2]})" for c in cols]
                sources[f"{rel}.{table}"] = [c[1] for c in cols]
                
                cursor.execute(f"SELECT COUNT(*) FROM '{table}';")
                row_count = cursor.fetchone()[0]
                db_info["tables"].append({
                    "name": table,
                    "columns": col_desc,
                    "row_count": row_count
                })
                
                cursor.execute(f"PRAGMA foreign_key_list('{table}');")
                for fk in cursor.fetchall():
                    fk_relations.append(f"{rel}.{table}.{fk[3]} -> {rel}.{fk[2]}.{fk[4]}")
            conn.close()
            dbs.append(db_info)
        except Exception:
            pass
    return dbs, sources, fk_relations

def _scan_csv(context_dir: Path):
    csvs = []
    sources = {}
    for csv_path in sorted(context_dir.rglob("*.csv")):
        rel = str(csv_path.relative_to(context_dir))
        try:
            import csv
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header:
                    sources[rel] = header
                    row_count = sum(1 for _ in reader)
                    csvs.append({"path": rel, "columns": header, "row_count": row_count})
        except Exception:
            pass
    return csvs, sources

def _scan_json(context_dir: Path):
    jsons = []
    sources = {}
    for json_path in sorted(context_dir.rglob("*.json")):
        if "task.json" in json_path.name: continue
        rel = str(json_path.relative_to(context_dir))
        try:
            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                head = f.read(1024 * 10)
                keys = sorted(list(set(re.findall(r'"([^"]+)":', head))))
                keys = [k for k in keys if k not in ["records", "table", "data", "items"]]
                if keys:
                    sources[rel] = keys
                    jsons.append({"path": rel, "keys": keys})
        except Exception:
            pass
    return jsons, sources

def get_data_roadmap(context_dir: Path, model=None) -> str:
    """构建包含物理和逻辑关系的深度知识图谱 (JSON)。"""
    db_list, db_srcs, explicit_fks = _scan_databases(context_dir)
    csv_list, csv_srcs = _scan_csv(context_dir)
    json_list, json_srcs = _scan_json(context_dir)
    
    all_sources = {**db_srcs, **csv_srcs, **json_srcs}
    
    # 1. 物理层关系：同名列发现 (Potential Joins)
    col_to_srcs = defaultdict(list)
    for src, cols in all_sources.items():
        for col in cols:
            col_to_srcs[col.lower().strip()].append(src)
    potential_joins = [f"{col}: {' <-> '.join(srcs)}" for col, srcs in col_to_srcs.items() if len(srcs) > 1]

    # 2. 逻辑层关系：从 knowledge.md 提取隐式关联
    implicit_relations = []
    business_rules = []
    full_knowledge_text = ""
    for k_name in ["knowledge.md", "Knowledge.md"]:
        k_path = context_dir / k_name
        if k_path.exists():
            try:
                full_knowledge_text = k_path.read_text(encoding="utf-8", errors="replace")
                if model:
                    prompt_tpl_path = Path(__file__).parent / "prompts" / "kg_relation_extraction.txt"
                    if prompt_tpl_path.exists():
                        prompt_tpl = prompt_tpl_path.read_text(encoding="utf-8")
                    else:
                        prompt_tpl = "Extract relations and rules from:\n{text}"
                    
                    prompt = prompt_tpl.format(text=full_knowledge_text[:8000])
                    resp = model.complete([ModelMessage(role="user", content=prompt)])
                    # 简单解析输出
                    lines = resp.strip().splitlines()
                    for line in lines:
                        if "->" in line:
                            implicit_relations.append(line.strip("-*• "))
                        elif line.strip() and not line.startswith(("RELATIONS", "RULES")):
                            business_rules.append(line.strip("-*• "))
            except Exception:
                pass
            break

    # 3. 整合 JSON 路线图
    roadmap = {
        "data_assets": {
            "databases": db_list,
            "csv_files": csv_list,
            "json_files": json_list
        },
        "relationships": {
            "explicit_foreign_keys": explicit_fks,
            "implicit_logic_joins": implicit_relations,
            "potential_name_matches": potential_joins
        },
        "business_rules": business_rules,
        "documentation": [str(f.relative_to(context_dir)) for f in sorted(context_dir.rglob("*.md")) if f.name.lower() != "knowledge.md"]
    }
    
    return "=== DATA ROADMAP (JSON-KG) ===\n" + json.dumps(roadmap, ensure_ascii=False, indent=2)
