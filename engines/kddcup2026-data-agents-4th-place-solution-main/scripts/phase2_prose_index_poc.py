"""PoC runner for Qwen sub-agent prose indexing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_144_modality.prose_indexer import build_doc_index, index_cache_path


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--doc", required=True, help="context-relative doc path, e.g. doc/x.pdf")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-lines", type=int, default=None)
    args = ap.parse_args()

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    task = ds.get_task(args.task)
    index = build_doc_index(
        task,
        args.doc,
        force=args.force,
        max_lines=args.max_lines,
    )
    out_path = index_cache_path(task, args.doc)
    print(f"saved {out_path}")
    print(json.dumps({
        "sections": len(index.get("sections", [])),
        "records": len(index.get("records", [])),
        "top_records_for_question": len(index.get("top_records_for_question", [])),
        "warnings": index.get("warnings", []),
    }, ensure_ascii=False, indent=2))
    for row in index.get("top_records_for_question", [])[:12]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
