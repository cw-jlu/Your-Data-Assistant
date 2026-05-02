import json
import re
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_agent_baseline.agents.model import ModelMessage


def _scan_databases(context_dir: Path) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """扫描所有 sqlite/db 文件，提取表结构和样本。"""
    lines = []
    sources: dict[str, list[str]] = {}
    fk_relations = []

    for db_path in sorted(context_dir.rglob("*")):
        if db_path.suffix.lower() not in [".db", ".sqlite"]:
            continue
        
        rel = db_path.relative_to(context_dir)
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
            
            if tables:
                lines.append(f"\n[DATABASE] {rel}")
                for table in tables:
                    # 获取列信息
                    cursor.execute(f"PRAGMA table_info('{table}');")
                    cols = cursor.fetchall()
                    col_names = [c[1] for c in cols]
                    sources[f"{rel}.{table}"] = col_names
                    
                    # 获取行数
                    cursor.execute(f"SELECT COUNT(*) FROM '{table}';")
                    row_count = cursor.fetchone()[0]
                    
                    # 紧凑输出：Table (row_count rows): col1, col2, ...
                    col_str = ", ".join(col_names)
                    lines.append(f"  - {table} ({row_count} rows): {col_str}")
                    
                    # 获取外键
                    cursor.execute(f"PRAGMA foreign_key_list('{table}');")
                    for fk in cursor.fetchall():
                        fk_relations.append(f"  {rel}.{table}.{fk[3]} -> {rel}.{fk[2]}.{fk[4]}")
            conn.close()
        except Exception:
            pass
    return lines, sources, fk_relations


def _scan_csv(context_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """扫描所有 CSV 文件，提取表头。"""
    lines = []
    sources: dict[str, list[str]] = {}
    for csv_path in sorted(context_dir.rglob("*.csv")):
        rel = csv_path.relative_to(context_dir)
        try:
            import csv
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header:
                    sources[str(rel)] = header
                    col_str = ", ".join(header)
                    lines.append(f"[CSV] {rel}: {col_str}")
        except Exception:
            pass
    return lines, sources


def _scan_json(context_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
    """扫描所有 JSON 文件，优化大文件处理。"""
    lines = []
    sources: dict[str, list[str]] = {}
    for json_path in sorted(context_dir.rglob("*.json")):
        if "task.json" in json_path.name: continue
        rel = json_path.relative_to(context_dir)
        try:
            # 对于较大的 JSON 文件，不进行完整加载，只读取前 10KB 尝试解析
            file_size = json_path.stat().st_size
            if file_size > 1024 * 1024: # > 1MB
                with json_path.open("r", encoding="utf-8", errors="replace") as f:
                    head = f.read(1024 * 10) # 读取前 10KB
                    keys = sorted(list(set(re.findall(r'"([^"]+)":', head))))
                    keys = [k for k in keys if k not in ["records", "table", "data", "items"]]
                    if keys:
                        sources[str(rel)] = keys
                        key_str = ", ".join(keys)
                        lines.append(f"[JSON] {rel} (Large): {key_str}")
                        continue

            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                records = []
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    for key in ["records", "data", "items"]:
                        if key in data and isinstance(data[key], list):
                            records = data[key]
                            break
                    if not records:
                        keys = list(data.keys())
                        sources[str(rel)] = keys
                        key_str = ", ".join(keys)
                        lines.append(f"[JSON] {rel}: {key_str}")
                        continue
                
                if records and isinstance(records[0], dict):
                    keys = list(records[0].keys())
                    sources[str(rel)] = keys
                    key_str = ", ".join(keys)
                    lines.append(f"[JSON] {rel}: {key_str}")
        except Exception:
            pass
    return lines, sources


def _find_join_hints(sources: dict[str, list[str]], fk_relations: list[str]) -> list[str]:
    """发现跨源关联，区分强关联(FK)和弱关联(同名字段)。"""
    hints = []
    if fk_relations:
        hints.append("\n[FOREIGN KEY RELATIONSHIPS] (strong links)")
        hints.extend(fk_relations)

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
    """使用 LLM 深度解析 knowledge.md。"""
    lines = []
    for k_name in ["knowledge.md", "Knowledge.md"]:
        k_path = context_dir / k_name
        if k_path.exists():
            try:
                text = k_path.read_text(encoding="utf-8", errors="replace")
                if model:
                    prompt = (
                        "Analyze the following business knowledge document and extract structured business logic.\n"
                        f"Content:\n{text[:5000]}\n\n"
                        "Extract: 1. Categorical Mappings, 2. Business Formulas, 3. Thresholds, 4. Join Rules.\n"
                        "Output ONLY bullet points."
                    )
                    response = model.complete([ModelMessage(role="user", content=prompt)])
                    if response and response.strip():
                        lines.append(f"\n[BUSINESS LOGIC] (extracted from {k_name}):")
                        for bullet in response.strip().splitlines():
                            if bullet.strip(): lines.append(f"  {bullet.strip()}")
                
                raw_refs = []
                for line in text.splitlines():
                    s = line.strip()
                    if "SELECT" in s.upper() or "WHERE" in s.upper() or s.startswith("###"):
                        raw_refs.append(f"  {s}")
                if raw_refs:
                    lines.append(f"\n[RAW REFERENCES] (key snippets):")
                    lines.extend(raw_refs[:20])
            except Exception: pass
            break
    return lines


def _extract_doc_semantics_with_llm(model, context_dir: Path) -> list[str]:
    """并行化调用 LLM 生成非 knowledge.md 的文档摘要。"""
    if model is None:
        return []

    doc_files = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))
    targets = [f for f in doc_files if f.name.lower() != "knowledge.md"]
    if not targets:
        return []

    # 加载 Prompt 模板
    prompt_tpl_path = Path(__file__).parent / "prompts" / "doc_summary_prompt.txt"
    if prompt_tpl_path.exists():
        prompt_tpl = prompt_tpl_path.read_text(encoding="utf-8")
    else:
        prompt_tpl = "Summarize the key data-related concepts in this document:\n{text}"

    def summarize_one(doc_path):
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if len(text.strip()) < 50: return None
            rel = doc_path.relative_to(context_dir)
            prompt = prompt_tpl.format(text=text)
            summary = model.complete([ModelMessage(role="user", content=prompt)])
            if summary and summary.strip():
                # 仅取第一行或将其压缩为一行，保持 Roadmap 轻量化
                compact_summary = summary.strip().replace("\n", " ").strip()
                if len(compact_summary) > 200:
                    compact_summary = compact_summary[:197] + "..."
                return [f"- {rel}: {compact_summary}"]
        except Exception:
            return None
        return None

    all_doc_lines = []
    # 使用线程池并发调用 LLM
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(summarize_one, f): f for f in targets}
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_doc_lines.extend(res)
    
    return all_doc_lines


def get_data_roadmap(context_dir: Path, model=None) -> str:
    """构建完整的 Schema 知识图谱。"""
    all_lines = ["=== DATA SCHEMA KNOWLEDGE GRAPH ==="]
    all_sources: dict[str, list[str]] = {}

    # 1. 扫描各种数据源
    db_lines, db_sources, fk_relations = _scan_databases(context_dir)
    csv_lines, csv_sources = _scan_csv(context_dir)
    json_lines, json_sources = _scan_json(context_dir)

    all_lines.extend(db_lines)
    all_lines.extend(csv_lines)
    all_lines.extend(json_lines)
    all_sources.update(db_sources)
    all_sources.update(csv_sources)
    all_sources.update(json_sources)

    # 2. 发现表间关系
    all_lines.extend(_find_join_hints(all_sources, fk_relations))

    # 3. 提取业务规则 (knowledge.md)
    all_lines.extend(_parse_knowledge_md(context_dir, model=model))

    # 4. 提取文档语义 (并行摘要)
    all_lines.extend(_extract_doc_semantics_with_llm(model, context_dir))

    return "\n".join(all_lines)
