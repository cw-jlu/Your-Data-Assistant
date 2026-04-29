import sqlite3
import csv
import json
import re
from pathlib import Path

def _extract_triplets_with_llm(model, context_dir: Path) -> list[tuple[str, str, str]]:
    from data_agent_baseline.agents.model import ModelMessage
    triplets = []
    doc_files = list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt"))
    for doc_path in doc_files:
        # 跳过知识库文档
        if doc_path.name.lower() == "knowledge.md": continue
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")[:4000]
            if len(text.strip()) < 50: continue
            rel = doc_path.relative_to(context_dir)
            prompt = (
                f"Text from {rel}:\n{text}\n\n"
                "Extract key entities and their relationships. Focus on definitions and constraints. "
                "Output ONLY a JSON list: [[\"S\", \"P\", \"O\"], ...]. Max 15 triplets."
            )
            resp = model.complete([ModelMessage(role="user", content=prompt)])
            match = re.search(r"\[\s*\[.*?\]\s*\]", resp, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                for t in parsed:
                    if isinstance(t, list) and len(t) == 3: triplets.append(tuple(t))
        except Exception: pass
    return triplets

def get_semantic_triplets(model, context_dir: Path) -> str:
    triplets = _extract_triplets_with_llm(model, context_dir)
    if not triplets: return ""
    lines = ["\n=== SEMANTIC TRIPLETS (from Documents) ==="]
    for s, p, o in triplets[:30]:
        lines.append(f"  ({s}) --[{p}]--> ({o})")
    return "\n".join(lines)
