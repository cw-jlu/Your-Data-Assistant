"""
GraphRAG 导航器 (v2)。
升级内容：
1. 改进的 LLM 抽取 Prompt（归一化关系类型 + 置信度）
2. 实体去重（基于 lowercase 合并）
3. 按置信度排序，只注入高质量三元组
"""
import json
import re
from pathlib import Path


STANDARDIZED_RELATIONS = [
    "HAS_PROPERTY", "CONTAINS", "RELATES_TO", "IS_TYPE_OF",
    "HAS_VALUE", "EQUALS", "GREATER_THAN", "LESS_THAN",
    "BELONGS_TO", "DEFINED_AS", "MEASURED_BY", "CLASSIFIED_AS",
]

EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Rules:
1. Normalize entity names (e.g., "F1" → "Formula 1", abbreviations → full form)
2. Use standardized relationship types from this list when possible:
   {relations}
3. Include a confidence score (0.0-1.0) for each triplet
4. Deduplicate: merge synonyms into canonical forms
5. Focus on data-relevant facts: definitions, thresholds, categories, constraints

Output ONLY a JSON array: [["Subject", "Predicate", "Object", confidence], ...]
Max 20 triplets. No extra text.

Text from {source}:
{text}"""


def _extract_triplets_with_llm(
    model, context_dir: Path, min_confidence: float = 0.5
) -> list[tuple[str, str, str, float]]:
    """
    调用 LLM 从非 knowledge.md 的 MD/TXT 文件中抽取高质量三元组。
    """
    from data_agent_baseline.agents.model import ModelMessage

    raw_triplets: list[tuple[str, str, str, float]] = []
    doc_files = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))

    for doc_path in doc_files:
        if doc_path.name.lower() == "knowledge.md":
            continue
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if len(text.strip()) < 50:
                continue

            rel = doc_path.relative_to(context_dir)
            prompt = EXTRACTION_PROMPT.format(
                relations=", ".join(STANDARDIZED_RELATIONS),
                source=str(rel),
                text=text,
            )

            response = model.complete([ModelMessage(role="user", content=prompt)])

            # 解析 JSON 数组
            json_match = re.search(r"\[\s*\[.*?\]\s*\]", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                for t in parsed:
                    if isinstance(t, list) and len(t) >= 3:
                        conf = float(t[3]) if len(t) >= 4 else 0.7
                        raw_triplets.append((str(t[0]), str(t[1]), str(t[2]), conf))
        except Exception:
            pass

    return raw_triplets


def _deduplicate_entities(
    triplets: list[tuple[str, str, str, float]]
) -> list[tuple[str, str, str, float]]:
    """
    基于 lowercase 匹配进行实体合并。
    保留首次出现的大小写形式作为 canonical。
    """
    canonical: dict[str, str] = {}
    for s, p, o, conf in triplets:
        s_key = s.lower().strip()
        o_key = o.lower().strip()
        if s_key not in canonical:
            canonical[s_key] = s.strip()
        if o_key not in canonical:
            canonical[o_key] = o.strip()

    deduped = []
    seen = set()
    for s, p, o, conf in triplets:
        cs = canonical[s.lower().strip()]
        co = canonical[o.lower().strip()]
        key = (cs.lower(), p.lower(), co.lower())
        if key not in seen:
            seen.add(key)
            deduped.append((cs, p, co, conf))

    return deduped


def get_semantic_triplets(
    model, context_dir: Path,
    min_confidence: float = 0.5,
    max_triplets: int = 25,
) -> str:
    """
    提取、去重、排序三元组，返回格式化字符串。
    """
    raw = _extract_triplets_with_llm(model, context_dir, min_confidence)
    if not raw:
        return ""

    # 去重
    deduped = _deduplicate_entities(raw)

    # 按置信度排序
    deduped.sort(key=lambda x: x[3], reverse=True)

    # 过滤低置信度 + 限制数量
    filtered = [t for t in deduped if t[3] >= min_confidence][:max_triplets]
    if not filtered:
        return ""

    lines = [f"\n=== SEMANTIC TRIPLETS ({len(filtered)} high-confidence facts) ==="]
    for s, p, o, conf in filtered:
        lines.append(f"  ({s}) --[{p}]--> ({o})  [conf: {conf:.1f}]")

    return "\n".join(lines)
