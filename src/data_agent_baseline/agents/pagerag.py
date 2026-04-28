"""
PageRAG 导航器 (v2)。
升级内容：
1. 双层切分（标题优先 → 段落 fallback）
2. BM25 评分替代简单关键词交集
3. Top-K 可配置（从全局 rag_top_k 读取）
4. 无文档时优雅降级
"""
import math
import re
from collections import Counter
from pathlib import Path


# ========== BM25 评分器 ==========

class BM25Scorer:
    """Okapi BM25 信息检索评分算法。不依赖任何外部库。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avg_dl = sum(len(doc) for doc in corpus) / max(self.N, 1)

        # 计算每个词出现在多少篇文档中 (document frequency)
        self.doc_freqs: dict[str, int] = {}
        for doc in corpus:
            for word in set(doc):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

    def score(self, query_words: list[str], doc_index: int) -> float:
        doc = self.corpus[doc_index]
        doc_len = len(doc)
        tf = Counter(doc)
        total = 0.0
        for q in query_words:
            if q not in tf:
                continue
            df = self.doc_freqs.get(q, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            freq = tf[q]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            total += idf * numerator / denominator
        return total


# ========== 分块逻辑 ==========

def _tokenize(text: str) -> list[str]:
    """简易分词：按非字母数字字符切分并 lowercase。"""
    return re.findall(r'\w+', text.lower())


def _chunk_document(text: str, source: str, max_chunk_chars: int = 1500) -> list[dict]:
    """
    双层切分策略：
    1. 优先按 Markdown 标题（# / ## / ###）切分
    2. 若无标题，则按段落（双换行）切分
    """
    chunks = []

    # 第一优先级：标题切分
    header_parts = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    header_parts = [p.strip() for p in header_parts if p.strip()]

    if len(header_parts) > 1:
        # 有标题结构
        for part in header_parts:
            title_match = re.match(r'^(#{1,3})\s+(.+)', part)
            title = title_match.group(2).strip() if title_match else "(section)"
            _split_and_append(chunks, part, title, source, max_chunk_chars)
    else:
        # 无标题：按段落切分
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if not paragraphs:
            return chunks

        current_chunk = ""
        chunk_idx = 0
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_chunk_chars and current_chunk:
                chunk_idx += 1
                chunks.append({
                    "title": f"(paragraph block {chunk_idx})",
                    "content": current_chunk.strip(),
                    "source": source,
                })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

        if current_chunk.strip():
            chunk_idx += 1
            chunks.append({
                "title": f"(paragraph block {chunk_idx})",
                "content": current_chunk.strip(),
                "source": source,
            })

    return chunks


def _split_and_append(chunks, content, title, source, max_chars):
    """若单个 chunk 过长，进一步按段落细分。"""
    if len(content) <= max_chars:
        chunks.append({"title": title, "content": content, "source": source})
    else:
        paras = content.split('\n\n')
        sub = ""
        sub_idx = 0
        for p in paras:
            if len(sub) + len(p) > max_chars and sub:
                sub_idx += 1
                chunks.append({
                    "title": f"{title} (part {sub_idx})",
                    "content": sub.strip(),
                    "source": source,
                })
                sub = p
            else:
                sub += "\n\n" + p
        if sub.strip():
            sub_idx += 1
            chunks.append({
                "title": f"{title} (part {sub_idx})" if sub_idx > 1 else title,
                "content": sub.strip(),
                "source": source,
            })


# ========== 主类 ==========

class PageRAGNavigator:
    def __init__(self, context_dir: Path, top_k: int = 5):
        self.context_dir = context_dir
        self.top_k = top_k
        self.chunks: list[dict] = []
        self._bm25: BM25Scorer | None = None
        self._index_documents()

    def _index_documents(self):
        """扫描所有非 knowledge.md 的 MD/TXT 文件并建立 BM25 索引。"""
        for ext in ["*.md", "*.txt"]:
            for fpath in sorted(self.context_dir.rglob(ext)):
                if fpath.name.lower() == "knowledge.md":
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    rel = str(fpath.relative_to(self.context_dir))
                    file_chunks = _chunk_document(text, rel)
                    self.chunks.extend(file_chunks)
                except Exception:
                    pass

        # 构建 BM25 索引
        if self.chunks:
            corpus = [_tokenize(c["content"]) for c in self.chunks]
            self._bm25 = BM25Scorer(corpus)

    def get_catalog(self) -> str:
        """返回文档目录概览。无文档时返回空字符串。"""
        if not self.chunks:
            return ""

        lines = ["\n=== DOCUMENT INDEX (PageRAG) ==="]
        seen: dict[str, list[str]] = {}
        for i, c in enumerate(self.chunks):
            src = c["source"]
            if src not in seen:
                seen[src] = []
            seen[src].append(f"  Page {i}: {c['title']}")

        for src, pages in seen.items():
            lines.append(f"\n[DOC] {src}")
            lines.extend(pages)

        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """使用 BM25 检索最相关的 top_k 个分块。"""
        k = top_k or self.top_k
        if not self.chunks or not self._bm25:
            return ""

        query_words = _tokenize(query)
        if not query_words:
            return ""

        scored = [
            (self.chunks[i], self._bm25.score(query_words, i))
            for i in range(len(self.chunks))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 只取正分的结果
        top = [(c, s) for c, s in scored[:k] if s > 0]
        if not top:
            return ""

        lines = [f"\n=== RETRIEVED PAGES (Top {len(top)}, BM25) ==="]
        for chunk, score in top:
            lines.append(f"\n--- [{chunk['source']}] {chunk['title']} (score: {score:.2f}) ---")
            lines.append(chunk["content"][:2000])

        return "\n".join(lines)
