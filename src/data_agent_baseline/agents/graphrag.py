"""
GraphRAG 导航器。
利用 LLM 对非结构化文档进行 Entity-Relation 三元组抽取，
结合 DB/CSV Schema 构建统一语义图。
"""
import sqlite3
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


def _extract_schema_nodes(context_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """提取所有结构化数据的 Schema 节点。返回 (lines, sources)。"""
    lines = []
    sources: dict[str, list[str]] = {}

    for db_path in sorted(context_dir.rglob("*.db")):
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
                key = f"{rel}::{table_name}"
                sources[key] = col_names
                lines.append(f"[DB] {rel} → '{table_name}': {', '.join(col_desc)}")
            conn.close()
        except Exception as e:
            lines.append(f"[DB] {rel} (Error: {e})")

    for csv_path in sorted(context_dir.rglob("*.csv")):
        rel = csv_path.relative_to(context_dir)
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                header = next(csv.reader(f), None)
                if header:
                    sources[str(rel)] = header
                    lines.append(f"[CSV] {rel}: {', '.join(header)}")
        except Exception as e:
            lines.append(f"[CSV] {rel} (Error: {e})")

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
                    lines.append(f"[JSON] {rel}: {', '.join(keys)}")
        except Exception as e:
            lines.append(f"[JSON] {rel} (Error: {e})")

    return lines, sources


def _find_join_hints(sources: dict[str, list[str]]) -> list[str]:
    """发现跨源同名字段。"""
    col_to_srcs: dict[str, list[str]] = defaultdict(list)
    for src, cols in sources.items():
        for col in cols:
            col_to_srcs[col.lower().strip()].append(src)
    hints = []
    for col, srcs in col_to_srcs.items():
        if len(srcs) > 1:
            hints.append(f"  '{col}' links: {' <--> '.join(srcs)}")
    return hints


def _extract_triplets_with_llm(model, context_dir: Path) -> list[tuple[str, str, str]]:
    """
    调用 LLM 从 MD/TXT 文档中提取 (Subject, Predicate, Object) 三元组。
    """
    from data_agent_baseline.agents.model import ModelMessage

    triplets = []
    doc_files = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))

    for doc_path in doc_files:
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if len(text.strip()) < 50:
                continue

            rel = doc_path.relative_to(context_dir)
            prompt = (
                "Extract key entities and their relationships from the following text. "
                "Focus on definitions, classification, numerical thresholds, and constraints. "
                "Output ONLY a JSON list of triplets: [[\"Subject\", \"Predicate\", \"Object\"], ...]. "
                "Keep it concise (max 20 triplets). No extra text.\n\n"
                f"Text from {rel}:\n{text}"
            )

            response = model.complete([ModelMessage(role="user", content=prompt)])
            json_match = re.search(r"\[\s*\[.*?\]\s*\]", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                for t in parsed:
                    if isinstance(t, list) and len(t) == 3:
                        triplets.append((str(t[0]), str(t[1]), str(t[2])))
        except Exception:
            pass

    return triplets


def build_graphrag_roadmap(context_dir: Path, model=None) -> str:
    """
    构建 GraphRAG 路线图：
    1. 结构化 Schema 扫描
    2. 跨源 JOIN 发现
    3. LLM 文档三元组提取（如果模型可用）
    """
    lines = ["=== GRAPHRAG KNOWLEDGE GRAPH ==="]

    # 1. Schema 节点
    schema_lines, sources = _extract_schema_nodes(context_dir)
    if schema_lines:
        lines.append("\n[SCHEMA NODES]")
        lines.extend(schema_lines)

    # 2. JOIN 提示
    join_hints = _find_join_hints(sources)
    if join_hints:
        lines.append("\n[JOIN HINTS]")
        lines.extend(join_hints)

    # 3. LLM 三元组提取
    if model:
        triplets = _extract_triplets_with_llm(model, context_dir)
        if triplets:
            lines.append(f"\n[SEMANTIC TRIPLETS] ({len(triplets)} facts extracted)")
            for s, p, o in triplets[:30]:
                lines.append(f"  ({s}) --[{p}]--> ({o})")

    return "\n".join(lines)
