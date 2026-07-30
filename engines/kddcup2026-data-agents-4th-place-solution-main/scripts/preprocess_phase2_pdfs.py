"""Pre-extract Phase 2 PDF docs into a text cache for exp_144 prose PoC."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_144_modality.pdf_text_cache import cache_root, write_pdf_cache


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task_ids; default all")
    ap.add_argument("--force", action="store_true", help="re-extract even if cache fingerprint matches")
    args = ap.parse_args()

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = sorted([t.task_id for t in ds.iter_tasks()], key=lambda x: int(x.split("_")[1]))

    n_pdf = 0
    total_chars = 0
    print(f"pdf_text_cache_root={cache_root()}")
    for tid in task_ids:
        task = ds.get_task(tid)
        pdfs = sorted(task.context_dir.rglob("*.pdf"))
        if not pdfs:
            continue
        for pdf in pdfs:
            result = write_pdf_cache(task, pdf, force=args.force)
            n_pdf += 1
            total_chars += result.text_chars
            print(
                f"{tid} {result.source_rel} pages={result.page_count} "
                f"text_chars={result.text_chars} → {result.text_path}"
            )
    print(f"done tasks={len(task_ids)} pdfs={n_pdf} total_text_chars={total_chars}")


if __name__ == "__main__":
    main()
