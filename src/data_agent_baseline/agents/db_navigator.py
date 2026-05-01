"""
Schema 知识图谱导航器 (v2)。
升级内容：
1. 主外键关系提取 (PRAGMA foreign_key_list)
2. 行数统计 + 样本数据
3. knowledge.md 深度解析（业务定义）
4. 其他 MD 文件的 LLM 语义抽取
5. 强/弱关联分级 (FK vs 同名字段)
"""
import sqlite3
import csv
import json
from pathlib import Path
from collections import defaultdict


def _scan_databases(context_dir: Path) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """扫描所有 .db 文件，提取表结构、外键、行数和样本。"""
    lines = []
    sources: dict[str, list[str]] = {}
    fk_relations: list[str] = []

    for db_path in sorted(context_dir.rglob("*.db")):
        rel = db_path.relative_to(context_dir)
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [t[0] for t in cursor.fetchall()]

            for table_name in tables:
                # 列信息
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                cols = cursor.fetchall()
                col_names = [c[1] for c in cols]
                col_desc = [f"{c[1]}({c[2]})" for c in cols]
                sources[f"{rel}::{table_name}"] = col_names

                # 行数
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
                    row_count = cursor.fetchone()[0]
                except Exception:
                    row_count = "?"

                lines.append(f"\n[DB] {rel} → '{table_name}' ({row_count} rows)")
                lines.append(f"  Columns: {', '.join(col_desc)}")

                # 样本数据（前 3 行，每行截断 200 字符）
                try:
                    cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 3")
                    samples = cursor.fetchall()
                    if samples:
                        sample_strs = [str(row)[:200] for row in samples]
                        lines.append(f"  Sample: {'; '.join(sample_strs)}")
                except Exception:
                    pass

                # 外键关系
                try:
                    cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
                    fks = cursor.fetchall()
                    for fk in fks:
                        fk_table = fk[2]
                        fk_from = fk[3]
                        fk_to = fk[4]
                        fk_relations.append(
                            f"  [FK] {table_name}.{fk_from} → {fk_table}.{fk_to}"
                        )
                except Exception:
                    pass

            conn.close()
        except Exception as e:
            lines.append(f"\n[DB] {rel} (Error: {e})")

    return lines, sources, fk_relations


def _scan_csv(context_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """扫描 CSV 文件表头、行数和样本。"""
    lines = []
    sources: dict[str, list[str]] = {}
    for csv_path in sorted(context_dir.rglob("*.csv")):
        rel = csv_path.relative_to(context_dir)
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    sources[str(rel)] = header
                    
                    # 获取样本和行数
                    sample_rows = []
                    total_rows = 0
                    for row in reader:
                        if len(sample_rows) < 3:
                            sample_rows.append(str(row)[:200])
                        total_rows += 1
                    
                    lines.append(f"\n[CSV] {rel} ({total_rows} rows)")
                    lines.append(f"  Columns: {', '.join(header)}")
                    if sample_rows:
                        lines.append(f"  Sample: {'; '.join(sample_rows)}")
        except Exception:
            pass
    return lines, sources


def _scan_json(context_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """扫描 JSON 文件，智能识别列表结构并提取样本。"""
    lines = []
    sources: dict[str, list[str]] = {}
    for json_path in sorted(context_dir.rglob("*.json")):
        rel = json_path.relative_to(context_dir)
        try:
            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                
                records = []
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    # 尝试查找名为 records, data, items 的列表
                    for key in ["records", "data", "items"]:
                        if key in data and isinstance(data[key], list):
                            records = data[key]
                            break
                    if not records:
                        # 如果没有找到标准列表，就用顶层键作为列名
                        keys = list(data.keys())
                        sources[str(rel)] = keys
                        lines.append(f"\n[JSON] {rel}: {', '.join(keys)}")
                        continue
                
                if records and isinstance(records[0], dict):
                    keys = list(records[0].keys())
                    sources[str(rel)] = keys
                    
                    sample_strs = [str(r)[:200] for r in records[:3]]
                    lines.append(f"\n[JSON] {rel} ({len(records)} records)")
                    lines.append(f"  Columns: {', '.join(keys)}")
                    lines.append(f"  Sample: {'; '.join(sample_strs)}")
                elif records:
                    lines.append(f"\n[JSON] {rel} ({len(records)} items)")
                    lines.append(f"  Sample: {str(records[:3])[:500]}")
        except Exception:
            pass
    return lines, sources


def _find_join_hints(sources: dict[str, list[str]], fk_relations: list[str]) -> list[str]:
    """发现跨源关联，区分强关联(FK)和弱关联(同名字段)。"""
    hints = []

    # 强关联：外键
    if fk_relations:
        hints.append("\n[FOREIGN KEY RELATIONSHIPS] (strong links)")
        hints.extend(fk_relations)

    # 弱关联：同名字段
    col_to_srcs: dict[str, list[str]] = defaultdict(list)
    for src, cols in sources.items():
        for col in cols:
            col_to_srcs[col.lower().strip()].append(src)

    shared = [
        f"  '{col}' in: {' <--> '.join(srcs)}"
        for col, srcs in col_to_srcs.items()
        if len(srcs) > 1
    ]
    if shared:
        hints.append("\n[SHARED COLUMNS] (potential joins)")
        hints.extend(shared)

    return hints


def _parse_knowledge_md(context_dir: Path, model=None) -> list[str]:
    """使用 LLM 深度解析 knowledge.md，提取结构化的业务规则和映射。"""
    lines = []
    for k_name in ["knowledge.md", "Knowledge.md"]:
        k_path = context_dir / k_name
        if k_path.exists():
            try:
                text = k_path.read_text(encoding="utf-8", errors="replace")
                
                if model:
                    from data_agent_baseline.agents.model import ModelMessage
                    prompt = (
                        "You are a Senior Data Engineer. Analyze the following business knowledge document "
                        "and extract structured business logic to guide a data analysis agent.\n\n"
                        f"Document Content:\n{text[:5000]}\n\n"
                        "Extract the following into clear bullet points:\n"
                        "1. Categorical Mappings: Map business terms to specific table fields and values (e.g., 'Severe' -> Table.Field = Value).\n"
                        "2. Business Formulas: Extract KPIs and calculation rules.\n"
                        "3. Thresholds: Extract any numeric limits mentioned.\n"
                        "4. Join Rules: Note which fields link different tables.\n"
                        "Output ONLY the bullet points, no conversational filler."
                    )
                    response = model.complete([ModelMessage(role="user", content=prompt)])
                    if response and response.strip():
                        lines.append(f"\n[BUSINESS LOGIC GRAPH] (extracted from {k_name}):")
                        for bullet in response.strip().splitlines():
                            b = bullet.strip()
                            if b:
                                lines.append(f"  {b}")
                
                # 保留原始的一些关键行作为参考 (如 SQL 示例)
                raw_refs = []
                for line in text.splitlines():
                    s = line.strip()
                    if not s: continue
                    if "SELECT" in s.upper() or "WHERE" in s.upper() or s.startswith("###"):
                        raw_refs.append(f"  {s}")
                
                if raw_refs:
                    lines.append(f"\n[RAW REFERENCES] (key examples from {k_name}):")
                    lines.extend(raw_refs[:30]) # 限制参考行数

            except Exception as e:
                lines.append(f"\n[KNOWLEDGE] {k_name} (Error during LLM parse: {e})")
            break
    return lines


def _extract_doc_semantics_with_llm(model, context_dir: Path) -> list[str]:
    """用 LLM 从非 knowledge.md 的 MD/TXT 文件中抽取关键业务概念。"""
    if model is None:
        return []

    from data_agent_baseline.agents.model import ModelMessage

    lines = []
    doc_files = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))

    for doc_path in doc_files:
        if doc_path.name.lower() == "knowledge.md":
            continue
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")[:3000]
            if len(text.strip()) < 50:
                continue

            rel = doc_path.relative_to(context_dir)
            prompt = (
                f"Document: {rel}\n\n{text}\n\n"
                "Summarize the key data-related concepts in this document in 3-5 bullet points. "
                "Focus on: column definitions, business rules, thresholds, categories, and data relationships. "
                "Output ONLY bullet points, no introduction."
            )
            response = model.complete([ModelMessage(role="user", content=prompt)])
            if response and response.strip():
                lines.append(f"\n[DOC SEMANTICS] {rel}:")
                for bullet in response.strip().splitlines():
                    b = bullet.strip()
                    if b:
                        lines.append(f"  {b}")
        except Exception:
            pass

    return lines


def get_data_roadmap(context_dir: Path, model=None) -> str:
    """
    构建 Schema 知识图谱 (v2)：
    1. DB: 表结构 + 外键 + 行数 + 样本
    2. CSV/JSON: 表头/键
    3. 强/弱关联分级
    4. knowledge.md 深度解析
    5. 其他文档 LLM 语义抽取
    """
    all_lines = ["=== DATA SCHEMA KNOWLEDGE GRAPH ==="]
    all_sources: dict[str, list[str]] = {}

    # 1. 结构化数据扫描
    db_lines, db_sources, fk_relations = _scan_databases(context_dir)
    csv_lines, csv_sources = _scan_csv(context_dir)
    json_lines, json_sources = _scan_json(context_dir)

    all_lines.extend(db_lines)
    all_lines.extend(csv_lines)
    all_lines.extend(json_lines)
    all_sources.update(db_sources)
    all_sources.update(csv_sources)
    all_sources.update(json_sources)

    # 2. 关联发现
    join_hints = _find_join_hints(all_sources, fk_relations)
    all_lines.extend(join_hints)

    # 3. knowledge.md 业务定义
    knowledge_lines = _parse_knowledge_md(context_dir, model=model)
    all_lines.extend(knowledge_lines)

    # 4. 其他文档 LLM 语义抽取
    doc_lines = _extract_doc_semantics_with_llm(model, context_dir)
    all_lines.extend(doc_lines)

    return "\n".join(all_lines)
    join_hints = _find_join_hints(all_sources, fk_relations)
    all_lines.extend(join_hints)

    # 3. knowledge.md 业务定义
    knowledge_lines = _parse_knowledge_md(context_dir, model=model)
    all_lines.extend(knowledge_lines)

    # 4. 可用文档列表 (不调用 LLM，加快初始化)
    all_lines.extend(_list_available_docs(context_dir))

    return "\n".join(all_lines)
