"""Build an embedding index over BIRD train (= 9428 examples) for ICL retrieval.

Output:
  artifacts/bird_icl/embeddings.npy   (= float32, shape (9428, 384), L2-normalized)
  artifacts/bird_icl/metadata.parquet (= db_id, question, evidence, SQL)

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (= Agentar-Scale-SQL default).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SRC_PARQUET = ROOT / "data" / "external" / "bird" / "train" / "bird_train.parquet"
SRC_DEV_JSON = ROOT / "data" / "external" / "bird" / "dev_20240627" / "dev.json"
OUT_DIR = ROOT / "artifacts" / "bird_icl"
EMB_PATH = OUT_DIR / "embeddings.npy"
META_PATH = OUT_DIR / "metadata.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    if not SRC_PARQUET.exists():
        print(f"missing: {SRC_PARQUET}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train = pl.read_parquet(SRC_PARQUET).select(["db_id", "question", "evidence", "SQL"])
    df_train = df_train.with_columns(pl.lit("train").alias("split"))
    print(f"[load] {df_train.height} examples from {SRC_PARQUET.name}")

    if SRC_DEV_JSON.exists():
        import json as _json
        dev_raw = _json.loads(SRC_DEV_JSON.read_text())
        df_dev = pl.DataFrame([
            {"db_id": d["db_id"], "question": d["question"],
             "evidence": d.get("evidence", ""), "SQL": d["SQL"], "split": "dev"}
            for d in dev_raw
        ])
        print(f"[load] {df_dev.height} examples from {SRC_DEV_JSON.name} (= ⚠️ may overlap DABench gold)")
        df = pl.concat([df_train, df_dev], how="vertical")
    else:
        df = df_train
    n = df.height
    print(f"[load] total {n} examples (= train + dev combined)")

    print(f"[embed] loading {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"[embed] model loaded in {time.time()-t0:.1f}s, dim={model.get_embedding_dimension()}")

    questions = df["question"].to_list()
    t0 = time.time()
    embs = model.encode(
        questions,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,  # = L2 normalize so cosine == dot product
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[embed] encoded {n} questions in {time.time()-t0:.1f}s, shape={embs.shape}")

    np.save(EMB_PATH, embs)
    df.write_parquet(META_PATH)
    print(f"[save] {EMB_PATH} ({EMB_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"[save] {META_PATH} ({META_PATH.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
