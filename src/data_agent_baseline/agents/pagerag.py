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
import numpy as np
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


def _chunk_document(text: str, source: str, model=None, max_chunk_chars: int = 1500) -> list[dict]:
    """
    动态目录感知切分策略：
    1. 检查是否存在原生 Markdown 标题（# / ## / ###）
    2. 若无标题，如果有大模型，则调用大模型生成目录结构并按此逻辑切分
    3. fallback: 按段落换行切分
    """
    chunks = []
    # 第一优先级：检查是否有原生标题
    header_parts = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    header_parts = [p.strip() for p in header_parts if p.strip()]

    if len(header_parts) > 1:
        # 有原生标题结构
        for part in header_parts:
            title_match = re.match(r'^(#{1,3})\s+(.+)', part)
            title = title_match.group(2).strip() if title_match else "(section)"
            _split_and_append(chunks, part, title, source, max_chunk_chars)
    else:
        # 无原生标题：先让模型生成目录及对应的“起始锚点文本”，然后根据锚点进行物理切分
        # 这种方式保证了“完全不改原文一字一句”
        if model and 200 < len(text) < 15000:
            from data_agent_baseline.agents.model import ModelMessage
            try:
                prompt = (
                    "Please analyze the following document and identify 3-5 logical sections. "
                    "For each section, provide a concise header and the EXACT first 30 characters of that section's text. "
                    "Format your output exactly as:\n"
                    "Header: [Section Title]\n"
                    "Start Text: [First 30 chars]\n\n"
                    f"Document:\n{text[:10000]}"
                )
                resp = model.complete([ModelMessage(role="user", content=prompt)])
                
                # 解析模型返回的锚点
                splits = []
                sections = re.findall(r"Header:\s*(.*?)\nStart Text:\s*(.*?)(?:\n|$)", resp, re.DOTALL)
                
                current_search_pos = 0
                for title, start_phrase in sections:
                    title = title.strip()
                    start_phrase = start_phrase.strip()
                    # 在原文中查找起始锚点
                    idx = text.find(start_phrase, current_search_pos)
                    if idx != -1:
                        splits.append((idx, title))
                        current_search_pos = idx + len(start_phrase)
                
                if len(splits) > 1:
                    # 根据索引进行物理切分
                    splits.sort() # 确保有序
                    for i in range(len(splits)):
                        start_idx, title = splits[i]
                        end_idx = splits[i+1][0] if i + 1 < len(splits) else len(text)
                        part_content = text[start_idx:end_idx].strip()
                        if part_content:
                            _split_and_append(chunks, part_content, title, source, max_chunk_chars)
                    return chunks
            except Exception:
                pass

        # Fallback: 按段落切分
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
                    "title": f"Paragraph Section {chunk_idx}",
                    "content": current_chunk.strip(),
                    "source": source,
                })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

        if current_chunk.strip():
            chunk_idx += 1
            chunks.append({
                "title": f"Paragraph Section {chunk_idx}",
                "content": current_chunk.strip(),
                "source": source,
            })

    return chunks


def _split_and_append(chunks, content, title, source, max_chars):
    """若单个 chunk 过长，进一步按段落细分。如果单段依然超长，强制按字符切断。"""
    if len(content) <= max_chars:
        chunks.append({"title": title, "content": content, "source": source})
    else:
        paras = content.split('\n\n')
        if len(paras) == 1:
            paras = content.split('\n')
            
        sub = ""
        sub_idx = 0
        for p in paras:
            # 防御：极端情况下连单行都超过最大限制，强制硬切分
            if len(p) > max_chars:
                # 先把缓存的 sub 推入
                if sub.strip():
                    sub_idx += 1
                    chunks.append({"title": f"{title} (part {sub_idx})", "content": sub.strip(), "source": source})
                    sub = ""
                # 硬切分超长段落
                for i in range(0, len(p), max_chars):
                    sub_part = p[i:i+max_chars]
                    if sub_part.strip():
                        sub_idx += 1
                        chunks.append({"title": f"{title} (part {sub_idx})", "content": sub_part.strip(), "source": source})
                continue

            if len(sub) + len(p) > max_chars and sub:
                sub_idx += 1
                chunks.append({
                    "title": f"{title} (part {sub_idx})",
                    "content": sub.strip(),
                    "source": source,
                })
                sub = p
            else:
                sub += ("\n\n" + p) if sub else p
                
        if sub.strip():
            sub_idx += 1
            chunks.append({
                "title": f"{title} (part {sub_idx})" if sub_idx > 1 else title,
                "content": sub.strip(),
                "source": source,
            })



class PageRAGNavigator:
    def __init__(self, context_dir: Path, top_k: int = 5, model=None):
        self.context_dir = context_dir
        self.doc_dir = context_dir / "doc"
        self.top_k = top_k
        self.model = model
        
        self.pages: list[dict] = [] 
        self.all_chunks: list[dict] = [] 
        
        self._page_embeddings: np.ndarray | None = None
        self._title_embeddings: np.ndarray | None = None
        self._bm25_tags: BM25Scorer | None = None
        
        # 仅当存在 doc 文件夹时才进行 PageRAG
        if self.doc_dir.exists() and self.doc_dir.is_dir():
            self._index_documents()
            if self.model:
                self._generate_tags()

    def _index_documents(self):
        """扫描 doc/*.md 文件，生成 Page 摘要并仅对标题进行向量化。"""
        doc_files = sorted(self.doc_dir.rglob("*.md"))
        if not doc_files:
            return

        for fpath in doc_files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                rel = str(fpath.relative_to(self.context_dir))
                
                # --- Step 1: Page 级摘要 (作为大标题) ---
                summary = ""
                if self.model and len(text) > 100:
                    from data_agent_baseline.agents.model import ModelMessage
                    prompt = (
                        "Provide a very concise summary (max 50 words) of this document's purpose. "
                        f"Document ({fpath.name}):\n{text[:4000]}"
                    )
                    try:
                        summary = self.model.complete([ModelMessage(role="user", content=prompt)]).strip()
                    except Exception:
                        summary = f"Documentation file: {fpath.name}"
                else:
                    summary = f"Documentation file: {fpath.name}"

                # --- Step 2: Chunk 级无损切分 ---
                file_chunks = _chunk_document(text, rel, model=self.model)
                
                page_data = {
                    "source": rel,
                    "summary": summary,
                    "chunks": file_chunks,
                    "id": len(self.pages)
                }
                self.pages.append(page_data)
                
                for c in file_chunks:
                    c["page_id"] = page_data["id"]
                    self.all_chunks.append(c)
            except Exception:
                pass

        # --- Step 3: 仅对“标题”建立向量索引 (Page Summary + Chunk Title) ---
        if self.model and self.all_chunks:
            try:
                # 1. Page 级摘要向量
                page_summaries = [p["summary"] for p in self.pages]
                page_embs = self.model.embed(page_summaries)
                if page_embs:
                    mat = np.array(page_embs, dtype=np.float32)
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-10
                    self._page_embeddings = mat / norms

                # 2. Chunk 级标题向量 (只向量化 Title，不向量化 Content)
                chunk_titles = [c["title"] for c in self.all_chunks]
                title_embs = self.model.embed(chunk_titles)
                if title_embs:
                    mat = np.array(title_embs, dtype=np.float32)
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    norms[norms == 0] = 1e-10
                    self._title_embeddings = mat / norms
            except Exception:
                pass

    def _generate_tags(self):
        """对标题和摘要提取关键词用于 BM25 补充。"""
        if not self.all_chunks: return
        corpus_tags = []
        for c in self.all_chunks:
            # 仅提取标题的关键词
            tags = _tokenize(c["title"])
            corpus_tags.append(tags)
        self._bm25_tags = BM25Scorer(corpus_tags)

    def get_catalog(self) -> str:
        if not self.pages:
            return ""
        lines = ["\n=== [PageRAG] Documentation Catalog (from context/doc/) ==="]
        for p in self.pages:
            lines.append(f"\n[Page {p['id']}: {p['source']}]")
            lines.append(f"Summary: {p['summary']}")
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """
        基于 RRF 的融合检索与 LLM Rerank
        """
        top_r = top_k or self.top_k
        k_retrieve = top_r * 2
        
        if not self.pages or not self.all_chunks:
            return ""

        query_words = _tokenize(query)

        # 1. 计算所有 chunk 的 Vector 和 BM25 score
        vector_scores = np.zeros(len(self.all_chunks))
        if self.model and self._title_embeddings is not None:
            q_emb_list = self.model.embed([query])
            if q_emb_list:
                q_emb = np.array(q_emb_list[0], dtype=np.float32)
                q_norm = np.linalg.norm(q_emb)
                if q_norm > 0:
                    q_emb = q_emb / q_norm
                    vector_scores = np.dot(self._title_embeddings, q_emb)
        
        bm25_scores = np.zeros(len(self.all_chunks))
        if self._bm25_tags:
            for i in range(len(self.all_chunks)):
                bm25_scores[i] = self._bm25_tags.score(query_words, i)

        # 2. RRF (Reciprocal Rank Fusion)
        def get_ranks(scores):
            sorted_indices = np.argsort(scores)[::-1]
            ranks = np.zeros(len(scores))
            for rank, idx in enumerate(sorted_indices):
                ranks[idx] = rank + 1
            return ranks

        vector_ranks = get_ranks(vector_scores)
        bm25_ranks = get_ranks(bm25_scores)

        rrf_k = 60
        rrf_scores = np.zeros(len(self.all_chunks))
        for i in range(len(self.all_chunks)):
            v_score = 1.0 / (rrf_k + vector_ranks[i]) if vector_scores[i] > 0 else 0
            b_score = 1.0 / (rrf_k + bm25_ranks[i]) if bm25_scores[i] > 0 else 0
            rrf_scores[i] = v_score + b_score

        top_k_indices = np.argsort(rrf_scores)[::-1][:k_retrieve]
        candidates = [i for i in top_k_indices if rrf_scores[i] > 0]
        if not candidates:
            return ""

        # 3. LLM Rerank
        final_list = candidates[:top_r]
        if self.model and len(candidates) > 1:
            try:
                from data_agent_baseline.agents.model import ModelMessage
                
                prompt = f"Given the user query: '{query}', rate the relevance of the following document sections on a scale of 0 to 10.\n"
                prompt += "Output ONLY a comma-separated list of scores in the exact same order, e.g. '8, 2, 10'.\n\n"
                for idx, c_idx in enumerate(candidates):
                    chunk = self.all_chunks[c_idx]
                    prompt += f"Section {idx+1} [{chunk['source']} - {chunk['title']}]:\n{chunk['content'][:300]}...\n\n"
                
                resp = self.model.complete([ModelMessage(role="user", content=prompt)])
                
                scores_str = re.findall(r'\d+', resp)
                if len(scores_str) == len(candidates):
                    llm_scores = [int(s) for s in scores_str]
                    scored_candidates = sorted(zip(candidates, llm_scores), key=lambda x: x[1], reverse=True)
                    final_list = [c for c, s in scored_candidates if s >= 5][:top_r]
                    if not final_list:
                        final_list = candidates[:1]
            except Exception:
                final_list = candidates[:top_r]

        if not final_list:
            return ""

        # 4. Result Formatting
        lines = [f"\n=== [PageRAG] Relevant Documentation (Reranked) ==="]
        for i in final_list:
            chunk = self.all_chunks[i]
            lines.append(f"\n--- [{chunk['source']}] Section: {chunk['title']} ---")
            lines.append(chunk["content"])

        return "\n".join(lines)
