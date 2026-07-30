"""BIRD-train ICL retrieval (= Agentar-Scale-SQL style).

For each DABench question, retrieve top-k semantically similar (question, SQL,
evidence) tuples from BIRD train (9428 examples) using MiniLM-L6-v2 cosine.

The retrieved examples are formatted as an "ICL EXAMPLES" preamble block that
the agent reads as SQL pattern guidance — not copy-paste templates, since the
schemas differ.

Artifacts (built once by scripts/build_bird_icl_index.py):
  artifacts/bird_icl/embeddings.npy   (= (9428, 384), L2-normalized float32)
  artifacts/bird_icl/metadata.parquet (= db_id, question, evidence, SQL)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
EMB_PATH = REPO / "artifacts" / "bird_icl" / "embeddings.npy"
META_PATH = REPO / "artifacts" / "bird_icl" / "metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Default ICL parameters (= best-practice starting points; tunable).
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIM = 0.40  # = filter out obviously remote matches (= empirical)
DEFAULT_MAX_SQL_CHARS = 320  # = p95 SQL length in BIRD train
DEFAULT_MAX_EVIDENCE_CHARS = 200


@dataclass(frozen=True, slots=True)
class IclExample:
    db_id: str
    question: str
    evidence: str
    sql: str
    sim: float


class _IndexState:
    """Module-level singleton — load embeddings and model exactly once per process."""

    _lock = Lock()
    _embs: np.ndarray | None = None
    _meta: pl.DataFrame | None = None
    _model: object | None = None  # SentenceTransformer instance

    @classmethod
    def ensure_loaded(cls) -> tuple[np.ndarray, pl.DataFrame, object]:
        with cls._lock:
            if cls._embs is None or cls._meta is None or cls._model is None:
                if not EMB_PATH.exists() or not META_PATH.exists():
                    raise FileNotFoundError(
                        f"BIRD ICL index missing — run scripts/build_bird_icl_index.py first.\n"
                        f"  expected: {EMB_PATH}\n  expected: {META_PATH}"
                    )
                cls._embs = np.load(EMB_PATH)
                cls._meta = pl.read_parquet(META_PATH)
                # Silence transformers noise at startup.
                os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(MODEL_NAME)
            return cls._embs, cls._meta, cls._model


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def retrieve(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_sim: float = DEFAULT_MIN_SIM,
) -> list[IclExample]:
    """Return top-k BIRD train examples ranked by cosine similarity to question."""
    embs, meta, model = _IndexState.ensure_loaded()
    q_vec = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
    sims = (embs @ q_vec[0]).astype(np.float32)  # = cosine since both normalized
    # argpartition then sort the top slice for speed
    k_pick = min(top_k * 4, len(sims))  # = pull more, then filter by min_sim
    top_idx = np.argpartition(-sims, k_pick - 1)[:k_pick]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out: list[IclExample] = []
    for i in top_idx:
        sim = float(sims[i])
        if sim < min_sim:
            break
        row = meta.row(int(i), named=True)
        out.append(
            IclExample(
                db_id=str(row["db_id"]),
                question=str(row["question"]),
                evidence=str(row["evidence"] or ""),
                sql=str(row["SQL"]),
                sim=sim,
            )
        )
        if len(out) >= top_k:
            break
    return out


def format_icl_block(
    examples: list[IclExample],
    *,
    max_sql_chars: int = DEFAULT_MAX_SQL_CHARS,
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS,
) -> str | None:
    """Format examples as a preamble block. Returns None if the list is empty."""
    if not examples:
        return None
    parts: list[str] = [
        "# ICL EXAMPLES (= prior SQL patterns retrieved by question similarity)",
        "If sim ≥ 0.9 and your schema matches, adapt the example's columns and JOIN structure directly. For lower-sim examples, use them as structural references for aggregation/filter idioms.",
    ]
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"## Example {i} (db={ex.db_id}, sim={ex.sim:.2f})\n"
            f"  Q: {_truncate(ex.question, 240)}\n"
            f"  Hint: {_truncate(ex.evidence, max_evidence_chars)}\n"
            f"  SQL: {_truncate(ex.sql, max_sql_chars)}\n"
        )
    return "\n".join(parts)


def retrieve_and_format(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_sim: float = DEFAULT_MIN_SIM,
) -> tuple[str | None, list[IclExample]]:
    """Convenience: retrieve and format in one call."""
    examples = retrieve(question, top_k=top_k, min_sim=min_sim)
    return format_icl_block(examples), examples
