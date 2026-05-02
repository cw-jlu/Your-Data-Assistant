import math
import re
import numpy as np
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from data_agent_baseline.agents.model import ModelMessage


# ========== BM25 评分器 ==========

class BM25Scorer:
    """Okapi BM25 信息检索评分算法。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avg_dl = sum(len(doc) for doc in corpus) / max(self.N, 1)

        self.doc_freqs: dict[str, int] = {}
        for doc in corpus:
            for word in set(doc):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

    def score(self, query_words: list[str], doc_index: int) -> float:
        if not self.corpus: return 0.0
        doc = self.corpus[doc_index]
        doc_len = len(doc)
        tf = Counter(doc)
        total = 0.0
        for q in query_words:
            if q not in tf: continue
            df = self.doc_freqs.get(q, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            freq = tf[q]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            total += idf * numerator / denominator
        return total


# ========== 分块逻辑 ==========

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def _chunk_document(text: str, source: str, max_chunk_chars: int = 1500) -> list[dict]:
    """按 Markdown 标题或段落进行物理切分。"""
    chunks = []
    header_parts = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    header_parts = [p.strip() for p in header_parts if p.strip()]

    if len(header_parts) > 1:
        for part in header_parts:
            title_match = re.match(r'^(#{1,3})\s+(.+)', part)
            title = title_match.group(2).strip() if title_match else "(section)"
            _split_and_append(chunks, part, title, source, max_chunk_chars)
    else:
        paragraphs = text.split("\n\n")
        current_title = "(content)"
        for p in paragraphs:
            if not p.strip(): continue
            _split_and_append(chunks, p, current_title, source, max_chunk_chars)
    return chunks


def _split_and_append(chunks: list[dict], text: str, title: str, source: str, max_chars: int):
    if len(text) <= max_chars:
        chunks.append({"title": title, "content": text, "source": source})
        return
    start = 0
    sub_idx = 1
    while start < len(text):
        end = start + max_chars
        content = text[start:end]
        chunks.append({
            "title": f"{title} (part {sub_idx})" if sub_idx > 1 else title,
            "content": content,
            "source": source
        })
        start = end
        sub_idx += 1


# ========== PageRAG 导航器 ==========

class PageRAGNavigator:
    def __init__(self, context_dir: Path, top_k: int = 5, model=None):
        self.context_dir = context_dir
        self.doc_dir = context_dir / "doc"
        self.top_k = top_k
        self.model = model
        
        self.pages: list[dict] = [] 
        self.all_chunks: list[dict] = [] 
        self._title_embeddings: np.ndarray | None = None
        self._bm25_tags: BM25Scorer | None = None
        
        if self.doc_dir.exists() and self.doc_dir.is_dir():
            self._index_documents()

    def _index_documents(self):
        """并行扫描文档并向量化标题。"""
        doc_files = sorted(self.doc_dir.rglob("*.md"))
        if not doc_files: return

        def process_doc(fpath):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                rel = str(fpath.relative_to(self.context_dir))
                chunks = _chunk_document(text, rel)
                return {"source": rel, "chunks": chunks}
            except Exception: return None

        # 并行化处理文档
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_doc, doc_files))
            for res in results:
                if res:
                    self.pages.append(res)
                    self.all_chunks.extend(res["chunks"])

        # 向量化 Chunk 标题
        if self.model and self.all_chunks:
            chunk_titles = [c["title"][:200] for c in self.all_chunks]
            embs = self.model.embed(chunk_titles)
            if embs:
                mat = np.array(embs, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                self._title_embeddings = mat / norms

        # 构建 BM25 索引
        if self.all_chunks:
            corpus = [_tokenize(c["content"]) for c in self.all_chunks]
            self._bm25_tags = BM25Scorer(corpus)

    def get_catalog(self) -> str:
        if not self.pages: return ""
        lines = ["\n=== [PageRAG] Documentation Catalog ==="]
        for p in self.pages:
            lines.append(f"- {p['source']} ({len(p['chunks'])} sections)")
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        top_r = top_k or self.top_k
        k_retrieve = top_r * 2
        if not self.all_chunks: return ""

        query_words = _tokenize(query)
        v_scores = np.zeros(len(self.all_chunks))
        if self.model and self._title_embeddings is not None:
            q_emb_list = self.model.embed([query])
            if q_emb_list:
                q_emb = np.array(q_emb_list[0], dtype=np.float32)
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 0:
                    v_scores = np.dot(self._title_embeddings, q_emb / q_norm)
        
        b_scores = np.zeros(len(self.all_chunks))
        if self._bm25_tags:
            for i in range(len(self.all_chunks)):
                b_scores[i] = self._bm25_tags.score(query_words, i)

        def get_ranks(scores):
            idx = np.argsort(scores)[::-1]
            ranks = np.zeros(len(scores))
            for r, i in enumerate(idx): ranks[i] = r + 1
            return ranks

        v_ranks = get_ranks(v_scores)
        b_ranks = get_ranks(b_scores)
        rrf_k = 60
        rrf_scores = np.zeros(len(self.all_chunks))
        for i in range(len(self.all_chunks)):
            v_s = 1.0 / (rrf_k + v_ranks[i]) if v_scores[i] > 0 else 0
            b_s = 1.0 / (rrf_k + b_ranks[i]) if b_scores[i] > 0 else 0
            rrf_scores[i] = v_s + b_s

        top_k_indices = np.argsort(rrf_scores)[::-1][:k_retrieve]
        candidates = [i for i in top_k_indices if rrf_scores[i] > 0]
        if not candidates: return ""

        final_list = candidates[:top_r]
        if self.model and len(candidates) > 1:
            try:
                p = f"Query: '{query}'. Rate relevance (0-10) for these sections. Output comma-separated scores only.\n"
                for idx, c_idx in enumerate(candidates):
                    c = self.all_chunks[c_idx]
                    p += f"{idx+1}. [{c['source']}-{c['title']}]: {c['content'][:200]}\n"
                resp = self.model.complete([ModelMessage(role="user", content=p)])
                scores = [int(s) for s in re.findall(r'\d+', resp)]
                if len(scores) >= len(candidates):
                    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                    final_list = [c for c, s in scored if s >= 5][:top_r]
                    if not final_list: final_list = candidates[:1]
            except Exception: pass

        lines = ["\n=== [PageRAG] Relevant Documentation (Reranked) ==="]
        for i in final_list:
            c = self.all_chunks[i]
            lines.append(f"\n--- [{c['source']}] {c['title']} ---\n{c['content']}")
        return "\n".join(lines)
