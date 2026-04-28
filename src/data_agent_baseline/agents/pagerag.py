"""
PageRAG 导航器。
将长文档（MD/TXT）按段落/章节分块，在 Agent 查询时返回最相关的"页"。
"""
import re
from pathlib import Path


def _chunk_markdown(text: str, max_chunk_chars: int = 1500) -> list[dict]:
    """
    将 Markdown 文本按标题（## / ###）分块。
    每块包含 title 和 content。
    """
    chunks = []
    # 按一级/二级/三级标题分割
    parts = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 提取标题
        title_match = re.match(r'^(#{1,3})\s+(.+)', part)
        title = title_match.group(2).strip() if title_match else "(untitled)"
        content = part

        # 如果内容太长，进一步按段落切分
        if len(content) > max_chunk_chars:
            paragraphs = content.split('\n\n')
            sub_chunk = ""
            for para in paragraphs:
                if len(sub_chunk) + len(para) > max_chunk_chars and sub_chunk:
                    chunks.append({"title": title, "content": sub_chunk.strip()})
                    sub_chunk = para
                else:
                    sub_chunk += "\n\n" + para
            if sub_chunk.strip():
                chunks.append({"title": title, "content": sub_chunk.strip()})
        else:
            chunks.append({"title": title, "content": content})

    return chunks


def _simple_relevance(query: str, chunk_content: str) -> float:
    """
    基于关键词重叠的简单相关性评分。
    不依赖任何外部向量库。
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    chunk_words = set(re.findall(r'\w+', chunk_content.lower()))
    if not query_words:
        return 0.0
    overlap = query_words & chunk_words
    return len(overlap) / len(query_words)


class PageRAGNavigator:
    def __init__(self, context_dir: Path):
        self.context_dir = context_dir
        self.chunks: list[dict] = []
        self._index_documents()

    def _index_documents(self):
        """预扫描所有 MD/TXT 文件，建立分块索引。"""
        for ext in ["*.md", "*.txt"]:
            for fpath in sorted(self.context_dir.rglob(ext)):
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    rel = fpath.relative_to(self.context_dir)
                    file_chunks = _chunk_markdown(text)
                    for chunk in file_chunks:
                        chunk["source"] = str(rel)
                    self.chunks.extend(file_chunks)
                except Exception:
                    pass

    def get_initial_context(self) -> str:
        """返回文档的目录概览（类似于书的目录页）。"""
        if not self.chunks:
            return ""

        lines = ["=== DOCUMENT INDEX (PageRAG) ==="]
        seen_sources = {}
        for i, chunk in enumerate(self.chunks):
            src = chunk["source"]
            if src not in seen_sources:
                seen_sources[src] = []
            seen_sources[src].append(f"  Page {i}: {chunk['title']}")

        for src, pages in seen_sources.items():
            lines.append(f"\n[DOC] {src}")
            lines.extend(pages)

        lines.append("\nUse the task question keywords to identify which pages are relevant.")
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """根据查询返回最相关的 top_k 个分块。"""
        if not self.chunks:
            return "(No documents indexed)"

        scored = [(c, _simple_relevance(query, c["content"])) for c in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        lines = [f"=== RETRIEVED PAGES (top {top_k}) ==="]
        for chunk, score in top:
            lines.append(f"\n--- [{chunk['source']}] {chunk['title']} (relevance: {score:.2f}) ---")
            lines.append(chunk["content"][:2000])  # 限制每块最大长度

        return "\n".join(lines)
