"""
Wiki retrieval - BM25 search with content-hash cache invalidation.

Fixes:
  - Cache invalidation uses content hash, not page count
  - Pre-computed term frequency dicts (no O(n) count() per query term)
  - Title tokens pre-computed and cached
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.wiki.engine import WikiEngine, WikiPage, extract_summary


@dataclass
class RetrievalResult:
    page: WikiPage
    score: float
    path: str


class WikiRetriever:

    K1 = 1.2
    B = 0.75
    TITLE_BOOST = 2.0
    TAG_BOOST = 1.5
    EXACT_TITLE_BOOST = 5.0

    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "and", "but", "or", "if", "this",
        "that", "these", "those", "it", "its", "what", "which", "who",
        "whom", "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "his", "she", "her", "they", "them", "their",
    })

    def __init__(self, engine: WikiEngine) -> None:
        self.engine = engine
        self._cache_hash: str = ""
        self._n: int = 0
        self._avg_dl: float = 0.0
        self._doc_freqs: Counter[str] = Counter()
        # Per-doc: (title_lower, path, title_tokens, tf_dict, doc_length, page)
        self._docs: list[tuple[str, str, set[str], dict[str, int], int, WikiPage]] = []

    def _ensure_cache(self) -> None:
        current_hash = self.engine.get_content_hash()
        if self._cache_hash == current_hash and self._docs:
            return

        pages_data = self.engine.get_all_page_texts()
        self._docs = []
        all_doc_freqs: Counter[str] = Counter()
        total_len = 0

        for title, path, text in pages_data:
            tokens = self._tokenize(text)
            if not tokens:
                continue
            page = self.engine.load_page_by_title(title)
            if page is None:
                continue

            token_set = set(tokens)
            tf = Counter(tokens)
            dl = len(tokens)
            total_len += dl

            self._docs.append((
                title.lower(), path, set(self._tokenize(title)),
                dict(tf), dl, page,
            ))
            for t in token_set:
                all_doc_freqs[t] += 1

        self._n = len(self._docs)
        self._avg_dl = total_len / self._n if self._n > 0 else 0
        self._doc_freqs = all_doc_freqs
        self._cache_hash = current_hash

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self._ensure_cache()
        if not self._docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        query_lower = query.lower()
        n = self._n
        avg_dl = self._avg_dl
        k1 = self.K1
        b = self.B

        # Pre-compute IDF for query terms
        idf_cache: dict[str, float] = {}
        for qt in query_set:
            df = self._doc_freqs.get(qt, 0)
            idf_cache[qt] = math.log((n - df + 0.5) / (df + 0.5) + 1)

        results: list[RetrievalResult] = []

        for title_lower, path, title_tokens, tf_dict, dl, page in self._docs:
            # BM25 score
            score = 0.0
            for qt in query_set:
                tf = tf_dict.get(qt, 0)
                if tf == 0:
                    continue
                idf = idf_cache[qt]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm

            if score < 0.1:
                continue

            # Title boost
            if query_lower in title_lower or title_lower in query_lower:
                score += self.EXACT_TITLE_BOOST
            else:
                title_overlap = len(query_set & title_tokens)
                if title_overlap > 0:
                    score += self.TITLE_BOOST * title_overlap / len(query_set)

            # Tag boost
            if page.tags:
                tag_tokens: set[str] = set()
                for tag in page.tags:
                    tag_tokens.update(self._tokenize(tag))
                tag_overlap = len(query_set & tag_tokens)
                if tag_overlap > 0:
                    score += self.TAG_BOOST * tag_overlap / len(query_set)

            results.append(RetrievalResult(page=page, score=round(score, 4), path=path))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def build_context_string(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        parts = [
            "=== Relevant Knowledge from Wiki ===",
            f"The following {len(results)} wiki page(s) may help with this task:",
            "",
        ]
        for i, r in enumerate(results, 1):
            page = r.page
            summary = extract_summary(page.content, max_len=300)
            parts.append(f"--- Wiki Page {i}: [{page.page_type}] {page.title} (relevance: {r.score}) ---")
            if page.tags:
                parts.append(f"Tags: {', '.join(page.tags)}")
            if summary:
                parts.append(f"Summary: {summary}")
            if len(page.content) < 1500:
                parts.append(f"Content:\n{page.content}")
            parts.append("")

        return "\n".join(parts)

    def invalidate_cache(self) -> None:
        self._cache_hash = ""

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        return [t for t in tokens if len(t) > 1 and t not in self._STOP_WORDS]
