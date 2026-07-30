"""Schema-linking via cell value matching (= Agentar-Scale-SQL style).

Faithful reproduction of Agentar-Scale-SQL's two-pronged cell value retrieval:
  - Dense:  MiniLM-L6-v2 embeddings + cosine top-k (= semantic match)
  - Sparse: BM25 over distinct cell values (= lexical/exact-favoring match)

Pipeline (per task):
  1. scan_columns(context_dir) → list[(view, col, value)]
     (= AgentAR filtering: TEXT cols only, skip *_id/url/email/date/address/etc.)
  2. build_index(records) → dense embeddings + BM25 index
  3. retrieve(question) → top-k cell values from each, dedupe by (view, col, value)
  4. format_value_hints(hits) → preamble block

The index is built per task at runtime since DABench tasks have isolated schemas.
The MiniLM model is module-level cached (shared with bird_icl.py).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np

# ---------------------------------------------------------------------------
# Tunable parameters (best-practice defaults; mirrors AgentAR)
# ---------------------------------------------------------------------------
MAX_DISTINCT_PER_COL = 5_000       # = cap to bound scan cost
MAX_VALUE_LEN_DENSE = 256          # = AgentAR's max_str_len for dense
MAX_VALUE_LEN_SPARSE = 40          # = AgentAR's BM25 filter (= short values)
MIN_VALUE_LEN = 1
TOP_K_DENSE = 10                   # = pre-fusion top-k from dense
TOP_K_SPARSE = 10                  # = pre-fusion top-k from BM25
TOP_K_OUT = 12                     # = final hits surfaced in preamble
MIN_DENSE_SIM = 0.45               # = below this, dense hits are dropped
MIN_BM25_SCORE = 1.0               # = BM25 absolute floor

# Column-name skip rules (= AgentAR-style)
_SKIP_NAME_SUBSTR = (
    "_id", " id", "url", "email", "web", "time", "date",
    "address", "phone", "uuid", "hash", "guid",
)


@dataclass(frozen=True, slots=True)
class ValueRecord:
    view: str
    column: str
    value: str  # = original-case display value


@dataclass(frozen=True, slots=True)
class ValueHit:
    view: str
    column: str
    value: str
    score: float
    source: str  # = "dense" or "bm25"


# ---------------------------------------------------------------------------
# Column-value scan (= same skip rules as AgentAR)
# ---------------------------------------------------------------------------
def _should_skip_col(view: str, col: str) -> bool:
    low_col = col.lower()
    low_view = view.lower()
    # ID detection: case-insensitive, catches "raceId" / "circuitId" / "race_id" /
    # "race id" / bare "id". Min length 4 avoids false positives like "kid".
    if low_col == "id":
        return True
    if low_col.endswith("_id") or low_col.endswith(" id"):
        return True
    if len(low_col) >= 4 and low_col.endswith("id"):
        return True
    for kw in _SKIP_NAME_SUBSTR:
        if kw in low_col:
            return True
    if low_view in {"sqlite_sequence", "sqlite_master"}:
        return True
    return False


def scan_columns(context_dir: Path) -> list[ValueRecord]:
    """Enumerate distinct string cell values for every text-y column in the task."""
    from experiments.exp_140_agentar_icl.tools.duckdb_unified import (
        get_catalog,
        get_connection,
    )

    out: list[ValueRecord] = []
    try:
        catalog = get_catalog(context_dir)
        conn = get_connection(context_dir)
    except Exception:
        return out

    seen: set[tuple[str, str, str]] = set()
    for entry in catalog:
        view = entry.get("view")
        if not view or not entry.get("columns"):
            continue
        for col in entry["columns"]:
            if _should_skip_col(view, col):
                continue
            try:
                q = (
                    f"SELECT DISTINCT CAST(\"{col}\" AS VARCHAR) AS v "
                    f"FROM \"{view}\" "
                    f"WHERE \"{col}\" IS NOT NULL "
                    f"LIMIT {MAX_DISTINCT_PER_COL + 1}"
                )
                rows = conn.execute(q).fetchall()
            except Exception:
                continue
            if len(rows) > MAX_DISTINCT_PER_COL:
                continue  # = too cardinal, likely freetext / PK
            for (v,) in rows:
                if not isinstance(v, str):
                    continue
                if not (MIN_VALUE_LEN <= len(v) <= MAX_VALUE_LEN_DENSE):
                    continue
                key = (view, col, v)
                if key in seen:
                    continue
                seen.add(key)
                out.append(ValueRecord(view=view, column=col, value=v))
    return out


# ---------------------------------------------------------------------------
# Index state (per-task, cached by context_dir identity)
# ---------------------------------------------------------------------------
class _IndexCache:
    # RLock because get() acquires the lock and then calls build() which calls
    # get_model() — also wanting the lock. A plain Lock would deadlock.
    _lock = RLock()
    _cache: dict[str, "_TaskIndex"] = {}
    _model = None  # = SentenceTransformer; shared across tasks

    @classmethod
    def get_model(cls):
        with cls._lock:
            if cls._model is None:
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                from sentence_transformers import SentenceTransformer
                # Force CPU explicitly — the host's CUDA 12.4 driver is too
                # old for the bundled torch CUDA, and llama.cpp already owns
                # the GPU at inference time. CPU is fast enough for 100-1000
                # value tasks (= <5s embed each).
                cls._model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                )
        return cls._model

    @classmethod
    def get(cls, context_dir: Path) -> "_TaskIndex":
        key = str(context_dir.resolve())
        with cls._lock:
            idx = cls._cache.get(key)
            if idx is None:
                idx = _TaskIndex.build(context_dir)
                cls._cache[key] = idx
            return idx


@dataclass(frozen=True, slots=True)
class _TaskIndex:
    records: list[ValueRecord]
    embeddings: np.ndarray | None  # = (N, 384) L2-normalized, or None if empty
    bm25_index: object | None      # = bm25s.BM25 or None if empty
    bm25_corpus_tokens: object | None  # = list[list[str]] (= original tokens for matching)

    @classmethod
    def build(cls, context_dir: Path) -> "_TaskIndex":
        records = scan_columns(context_dir)
        if not records:
            return cls(records=[], embeddings=None, bm25_index=None, bm25_corpus_tokens=None)

        # Dense embeddings — embed lowercased values for case-insensitive match.
        model = _IndexCache.get_model()
        texts_dense = [r.value.lower() for r in records]
        embs = model.encode(
            texts_dense,
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # BM25 — index only values short enough (= AgentAR's len<=40 rule).
        import bm25s
        bm25_corpus = [r.value.lower() for r in records if len(r.value) <= MAX_VALUE_LEN_SPARSE]
        bm25_map_to_records = [i for i, r in enumerate(records) if len(r.value) <= MAX_VALUE_LEN_SPARSE]
        if bm25_corpus:
            tokens = bm25s.tokenize(bm25_corpus, show_progress=False)
            bm25 = bm25s.BM25()
            bm25.index(tokens, show_progress=False)
            # Stash mapping by attaching as an attribute (bm25s indexes are mutable).
            bm25._map_to_records = bm25_map_to_records  # type: ignore[attr-defined]
        else:
            bm25 = None
        return cls(
            records=records,
            embeddings=embs,
            bm25_index=bm25,
            bm25_corpus_tokens=None,
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _dense_topk(idx: _TaskIndex, question: str, k: int) -> list[ValueHit]:
    if idx.embeddings is None or len(idx.records) == 0:
        return []
    model = _IndexCache.get_model()
    q_vec = model.encode(
        [question.lower()],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]
    sims = idx.embeddings @ q_vec
    k_pick = min(k * 2, len(sims))
    top_idx = np.argpartition(-sims, k_pick - 1)[:k_pick]
    top_idx = top_idx[np.argsort(-sims[top_idx])][:k]
    hits: list[ValueHit] = []
    for i in top_idx:
        sim = float(sims[int(i)])
        if sim < MIN_DENSE_SIM:
            break
        r = idx.records[int(i)]
        hits.append(ValueHit(
            view=r.view, column=r.column, value=r.value,
            score=sim, source="dense",
        ))
    return hits


def _bm25_topk(idx: _TaskIndex, question: str, k: int) -> list[ValueHit]:
    if idx.bm25_index is None:
        return []
    import bm25s
    q_tokens = bm25s.tokenize([question.lower()], show_progress=False)
    docs, scores = idx.bm25_index.retrieve(q_tokens, k=min(k, len(idx.bm25_index.scores["data"])
                                                            if hasattr(idx.bm25_index, "scores") else k),
                                            show_progress=False)
    # docs shape: (1, k_actual), scores shape: (1, k_actual)
    mapping = idx.bm25_index._map_to_records  # type: ignore[attr-defined]
    hits: list[ValueHit] = []
    for doc_id, score in zip(docs[0].tolist(), scores[0].tolist()):
        if score < MIN_BM25_SCORE:
            continue
        rec_idx = mapping[int(doc_id)]
        r = idx.records[rec_idx]
        hits.append(ValueHit(
            view=r.view, column=r.column, value=r.value,
            score=float(score), source="bm25",
        ))
    return hits


def retrieve(
    question: str,
    context_dir: Path,
    *,
    top_k_dense: int = TOP_K_DENSE,
    top_k_sparse: int = TOP_K_SPARSE,
    top_k_out: int = TOP_K_OUT,
) -> list[ValueHit]:
    """Two-prong retrieval: dense + BM25, deduplicated and capped."""
    idx = _IndexCache.get(context_dir)
    if not idx.records:
        return []
    dense_hits = _dense_topk(idx, question, top_k_dense)
    bm25_hits = _bm25_topk(idx, question, top_k_sparse)

    # Dedupe by (view, col, value); BM25 wins when both surface the same value
    # (= lexical confirmation is stronger evidence than semantic alone).
    seen: dict[tuple[str, str, str], ValueHit] = {}
    for h in bm25_hits + dense_hits:  # BM25 first so dense doesn't overwrite
        key = (h.view, h.column, h.value)
        if key not in seen:
            seen[key] = h
    # Order: BM25 by score desc, then dense by sim desc
    ordered = sorted(seen.values(),
                     key=lambda h: (0 if h.source == "bm25" else 1, -h.score))
    return ordered[:top_k_out]


def format_value_hints(hits: list[ValueHit]) -> str | None:
    """Format AgentAR-style cell value retrieval hits as a preamble block."""
    if not hits:
        return None
    lines: list[str] = [
        "# VALUE HINTS (= cell values that match the question)",
        (
            "Below are cell values discovered in your data that semantically or "
            "lexically match the question. Each entry shows `view`.`column` = 'value' "
            "and the source (= dense semantic vs BM25 lexical). Use these to write "
            "WHERE filters with the EXACT cell value the question refers to.\n"
        ),
    ]
    for h in hits:
        tag = "BM25" if h.source == "bm25" else "DENSE"
        lines.append(
            f"  - [{tag} {h.score:.2f}] `{h.view}`.`{h.column}` = '{h.value}'"
        )
    return "\n".join(lines)


def build_value_hints_block(question: str, context_dir: Path) -> str | None:
    """One-shot: build index, retrieve, format."""
    hits = retrieve(question, context_dir)
    return format_value_hints(hits)
