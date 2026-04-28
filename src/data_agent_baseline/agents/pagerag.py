import re
from pathlib import Path

def _chunk_markdown(text: str, max_chunk_chars: int = 1500) -> list[dict]:
    chunks = []
    parts = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    for part in parts:
        part = part.strip()
        if not part: continue
        title_match = re.match(r'^(#{1,3})\s+(.+)', part)
        title = title_match.group(2).strip() if title_match else "(untitled)"
        if len(part) > max_chunk_chars:
            paras = part.split('\n\n')
            sub = ""
            for p in paras:
                if len(sub) + len(p) > max_chunk_chars and sub:
                    chunks.append({"title": title, "content": sub.strip()})
                    sub = p
                else: sub += "\n\n" + p
            if sub.strip(): chunks.append({"title": title, "content": sub.strip()})
        else: chunks.append({"title": title, "content": part})
    return chunks

def _relevance(query: str, content: str) -> float:
    qw = set(re.findall(r'\w+', query.lower()))
    cw = set(re.findall(r'\w+', content.lower()))
    return len(qw & cw) / len(qw) if qw else 0.0

class PageRAGNavigator:
    def __init__(self, context_dir: Path):
        self.context_dir = context_dir
        self.chunks = []
        self._index_documents()

    def _index_documents(self):
        for ext in ["*.md", "*.txt"]:
            for fpath in sorted(self.context_dir.rglob(ext)):
                # 跳过已由 KG 处理的 knowledge.md
                if fpath.name.lower() == "knowledge.md": continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    rel = fpath.relative_to(self.context_dir)
                    for c in _chunk_markdown(text):
                        c["source"] = str(rel)
                        self.chunks.append(c)
                except Exception: pass

    def get_catalog(self) -> str:
        if not self.chunks: return ""
        lines = ["\n=== DOCUMENT CATALOG (PageRAG) ==="]
        srcs = {}
        for i, c in enumerate(self.chunks):
            if c["source"] not in srcs: srcs[c["source"]] = []
            srcs[c["source"]].append(f"  Page {i}: {c['title']}")
        for s, p in srcs.items():
            lines.append(f"\n[DOC] {s}"); lines.extend(p)
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.chunks: return ""
        scored = sorted([(c, _relevance(query, c["content"])) for c in self.chunks], key=lambda x: x[1], reverse=True)
        lines = [f"\n=== RETRIEVED PAGES (Top {top_k}) ==="]
        for c, s in scored[:top_k]:
            if s > 0:
                lines.append(f"\n--- [{c['source']}] {c['title']} ---")
                lines.append(c["content"][:1500])
        return "\n".join(lines) if len(lines) > 1 else ""
