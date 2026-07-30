"""Download (= cache) data-VQA benchmark datasets from HF.

We don't run anything yet; just snapshot to HF cache so later
benches can spin up fast.
"""
from __future__ import annotations

import sys
import time

from datasets import load_dataset

# (hf_id, split, friendly_name)
TARGETS = [
    ("HuggingFaceM4/ChartQA",             "test",  "ChartQA test"),
    ("terryoo/TableVQA-Bench",            None,    "TableVQA-Bench"),     # config-dependent
    ("vidore/docvqa_test_subsampled",     "test",  "DocVQA test subsampled"),
    ("niups/PlotQA-test",                 "test",  "PlotQA test"),
    ("SpursgoZmy/MMTab",                  None,    "MMTab"),
]


def try_load(name: str, split: str | None, label: str) -> str:
    t0 = time.time()
    try:
        if split is None:
            ds = load_dataset(name)
        else:
            ds = load_dataset(name, split=split)
        n = sum(len(s) for s in ds.values()) if hasattr(ds, "values") else len(ds)
        return f"  ✓ {label} ({name}): {n} examples in {time.time()-t0:.1f}s"
    except Exception as e:
        return f"  ✗ {label} ({name}): {type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    print(f"caching {len(TARGETS)} VLM benches sequentially")
    for name, split, label in TARGETS:
        msg = try_load(name, split, label)
        print(msg, flush=True)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
