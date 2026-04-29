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
    """扫描 CSV 文件表头。"""
    lines = []
    sources: dict[str, list[str]] = {}
    for csv_path in sorted(context_dir.rglob("*.csv")):
        rel = csv_path.relative_to(context_dir)
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                header = next(csv.reader(f), None)
                if header:
                    sources[str(rel)] = header
                    lines.append(f"\n[CSV] {rel}: {', '.join(header)}")
        except Exception:
            pass
    return lines, sources


def _scan_json(context_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """扫描 JSON 文件顶层键。"""
    lines = []
    sources: dict[str, list[str]] = {}
    for json_path in sorted(context_dir.rglob("*.json")):
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
                    lines.append(f"\n[JSON] {rel}: {', '.join(keys)}")
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


def _parse_knowledge_md(context_dir: Path) -> list[str]:
    """深度解析 knowledge.md，提取业务定义和规则。"""
    lines = []
    for k_name in ["knowledge.md", "Knowledge.md"]:
        k_path = context_dir / k_name
        if k_path.exists():
            try:
                text = k_path.read_text(encoding="utf-8", errors="replace")
                defs = []
                for line in text.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    if (
                        s.startswith(("- ", "* ", "•")) or
                        ":" in s or
                        s.startswith("##") or
                        "=" in s or
                        s.startswith("|")
                    ):
                        defs.append(f"  {s}")
                    if len(defs) >= 50:
                        break
                if defs:
                    lines.append(f"\n[KNOWLEDGE] {k_name} (business definitions):")
                    lines.extend(defs)
            except Exception:
                pass
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
                for bullet in response.strip().splitlines()[:7]:
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
    knowledge_lines = _parse_knowledge_md(context_dir)
    all_lines.extend(knowledge_lines)

    # 4. 其他文档 LLM 语义抽取
    doc_lines = _extract_doc_semantics_with_llm(model, context_dir)
    all_lines.extend(doc_lines)

    return "\n".join(all_lines)
