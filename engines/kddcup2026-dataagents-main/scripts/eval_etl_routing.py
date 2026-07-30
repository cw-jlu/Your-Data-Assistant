"""Evaluate ETL document routing: given a question, can the LLM correctly select
which doc files to extract (vs skip)?

Usage:
    uv run python scripts/eval_etl_routing.py --config configs/react.local.yaml [--limit N]

For each task with ≥2 doc files, the script:
1. Builds a lightweight profile per doc (filename + sampled paragraphs + schema from knowledge.md)
2. Asks the LLM: which files contain data needed to answer the question?
3. Compares against a deterministic oracle (gold.csv column names matched to knowledge.md sections)
4. Reports recall (did the router keep the necessary files?) and savings (how many were skipped?)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.etl._detect import detect_prose_files  # noqa: E402
from agents.etl._router import (  # noqa: E402
    FileProfile,
    build_file_profile,
    empty_file_profile,
    read_prose_text,
    route_file_profiles,
)
from agents.etl.knowledge import km_table_fields  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _make_adapter(cfg: dict):
    from agents.llm.openai import OpenAIModelAdapter

    agent_cfg = cfg["agent"]
    return OpenAIModelAdapter(
        model=agent_cfg["model"],
        api_base=agent_cfg["api_base"],
        api_key=agent_cfg["api_key"],
        temperature=0.1,
        max_tokens=512,
        enable_thinking=False,
        thinking_mode="disabled" if agent_cfg.get("backend_kind") == "deepseek" else None,
        backend_kind=agent_cfg.get("backend_kind"),
        max_retries=2,
        timeout=120,
    )


def _oracle_relevant_stems(
    task_dir: Path, gold_path: Path, km_text: str, prose_files: list[Path]
) -> list[str]:
    """Determine which prose files are actually needed based on gold.csv columns and knowledge.md mapping."""
    if not gold_path.exists():
        return [p.stem for p in prose_files]

    # Read gold column names
    with open(gold_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        gold_header = next(reader, None)

    # Also read gold values for matching
    gold_values: set[str] = set()
    with open(gold_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            gold_values.update(v.strip().lower() for v in row if v.strip())

    relevant: set[str] = set()

    for pf in prose_files:
        km_fields = km_table_fields(km_text, pf.stem) if km_text else {}

        # Strategy 1: if gold header matches a governance field, this file is relevant
        if gold_header and km_fields:
            km_lower = {k.lower() for k in km_fields}
            if any(h.lower() in km_lower for h in gold_header if h.strip()):
                relevant.add(pf.stem)
                continue

        # Strategy 2: check if gold values appear in the file sample
        text = read_prose_text(pf)
        text_lower = text[:50000].lower()
        matches = sum(1 for v in list(gold_values)[:20] if v in text_lower and len(v) > 3)
        if matches >= 3:
            relevant.add(pf.stem)

    # If no match found, assume all are relevant (conservative)
    return list(relevant) if relevant else [p.stem for p in prose_files]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate ETL document routing")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks (0=all)")
    parser.add_argument("--data-root", default=None, help="Override data root path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task sampling")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--api-base", default=None, help="Override API base URL")
    parser.add_argument("--api-key", default=None, help="Override API key")
    parser.add_argument("--backend", default=None, help="Override backend_kind")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.model:
        cfg["agent"]["model"] = args.model
    if args.api_base:
        cfg["agent"]["api_base"] = args.api_base
    if args.api_key:
        cfg["agent"]["api_key"] = args.api_key
    if args.backend:
        cfg["agent"]["backend_kind"] = args.backend
    adapter = _make_adapter(cfg)

    data_root = (
        Path(args.data_root) if args.data_root else PROJECT_ROOT / cfg["dataset"]["root_path"]
    )
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    # Find tasks with gold and output directories
    input_dir = data_root
    output_dir = data_root.parent / "output" if (data_root.parent / "output").exists() else None

    # Collect tasks
    from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task_dirs = sorted(input_dir.iterdir()) if input_dir.is_dir() else []
    task_dirs = [d for d in task_dirs if d.is_dir() and (d / "task.json").exists()]

    random.seed(args.seed)
    if args.limit > 0 and len(task_dirs) > args.limit:
        task_dirs = random.sample(task_dirs, args.limit)

    # Results
    results = []
    total_docs = 0
    total_selected = 0
    total_oracle_relevant = 0
    correct_recalls = 0
    tasks_evaluated = 0

    for td in task_dirs:
        task_json = json.loads((td / "task.json").read_text())
        task_id = task_json["task_id"]
        question = task_json["question"]

        # Build a minimal PublicTask for detect_prose_files
        task = PublicTask(
            record=TaskRecord(task_id=task_id, question=question, difficulty="hard"),
            assets=TaskAssets(task_dir=td, context_dir=td / "context"),
        )

        prose_files = detect_prose_files(task)
        if len(prose_files) < 2:
            # Still record, but skip routing
            result = {
                "task_id": task_id,
                "question": question[:80],
                "total_docs": len(prose_files),
                "selected": [p.stem for p in prose_files],
                "oracle": [p.stem for p in prose_files],
                "recall": 1.0,
                "savings": 0.0,
                "missed": [],
                "extra": [],
                "route_time_s": 0.0,
            }
            results.append(result)
            tasks_evaluated += 1
            total_docs += len(prose_files)
            total_selected += len(prose_files)
            total_oracle_relevant += len(prose_files)
            correct_recalls += 1
            logger.info("[SKIP] %s | docs=%d (no routing needed)", task_id, len(prose_files))
            continue

        # Read knowledge.md
        km_path = td / "context" / "knowledge.md"
        km_text = km_path.read_text(encoding="utf-8") if km_path.exists() else ""

        def _safe_build_profile(
            pf: Path, *, km_text: str = km_text, question: str = question
        ) -> FileProfile:
            try:
                return build_file_profile(pf, km_text, question, adapter=adapter)
            except Exception as e:
                logger.warning("Failed to build profile for %s: %s", pf.name, e)
                return empty_file_profile(pf)

        with ThreadPoolExecutor(max_workers=min(len(prose_files), 8)) as pool:
            futs = {pool.submit(_safe_build_profile, pf): pf for pf in prose_files}
            profiles: list[FileProfile] = []
            for fut in as_completed(futs):
                profiles.append(fut.result())
        # Restore original file order
        stem_order = {pf.stem: i for i, pf in enumerate(prose_files)}
        profiles.sort(key=lambda p: stem_order.get(p["stem"], 999))

        # Route
        t0 = time.time()
        try:
            selected_stems = route_file_profiles(adapter, question, profiles)
        except Exception as e:
            logger.error("Router failed for %s: %s", task_id, e)
            selected_stems = [p["stem"] for p in profiles]
        route_time = time.time() - t0

        # Oracle
        gold_path = output_dir / task_id / "gold.csv" if output_dir else None
        if gold_path and gold_path.exists():
            oracle_stems = _oracle_relevant_stems(td, gold_path, km_text, prose_files)
        else:
            oracle_stems = [p.stem for p in prose_files]

        # Evaluate
        oracle_set = set(oracle_stems)
        selected_set = set(selected_stems)
        all_stems = {p.stem for p in prose_files}

        recall = len(oracle_set & selected_set) / max(1, len(oracle_set))
        savings = 1.0 - len(selected_set) / max(1, len(all_stems))
        missed = oracle_set - selected_set
        extra = selected_set - oracle_set

        result = {
            "task_id": task_id,
            "question": question[:80],
            "total_docs": len(prose_files),
            "selected": sorted(selected_set),
            "oracle": sorted(oracle_set),
            "recall": recall,
            "savings": savings,
            "missed": sorted(missed),
            "extra": sorted(extra),
            "route_time_s": round(route_time, 2),
        }
        results.append(result)

        total_docs += len(prose_files)
        total_selected += len(selected_set)
        total_oracle_relevant += len(oracle_set)
        if recall >= 1.0:
            correct_recalls += 1
        tasks_evaluated += 1

        status = "OK" if recall >= 1.0 else "MISS"
        logger.info(
            "[%s] %s | docs=%d sel=%d oracle=%d recall=%.2f savings=%.0f%% time=%.1fs%s",
            status,
            task_id,
            len(prose_files),
            len(selected_set),
            len(oracle_set),
            recall,
            savings * 100,
            route_time,
            f" MISSED={sorted(missed)}" if missed else "",
        )

    # Summary
    print("\n" + "=" * 70)
    print("ETL ROUTING EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Tasks evaluated (with ≥2 doc files): {tasks_evaluated}")
    print(f"Total doc files across tasks:         {total_docs}")
    print(f"Total selected by router:             {total_selected}")
    print(f"Total oracle-relevant:                {total_oracle_relevant}")
    print(
        f"Perfect recall (all needed selected): {correct_recalls}/{tasks_evaluated} "
        f"({correct_recalls / max(1, tasks_evaluated) * 100:.0f}%)"
    )
    print(f"Avg savings (files skipped):          {1 - total_selected / max(1, total_docs):.0%}")
    print(f"Avg oracle density (needed/total):    {total_oracle_relevant / max(1, total_docs):.0%}")
    print()

    # Per-task details
    print(f"{'Task':<10} {'Docs':>4} {'Sel':>4} {'Orac':>4} {'Recall':>7} {'Save%':>6} {'Missed'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['task_id']:<10} {r['total_docs']:>4} {len(r['selected']):>4} "
            f"{len(r['oracle']):>4} {r['recall']:>7.2f} {r['savings'] * 100:>5.0f}% "
            f"{r['missed'] if r['missed'] else ''}"
        )

    # Save detailed results
    out_path = PROJECT_ROOT / "artifacts" / "etl_routing_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results: {out_path}")


if __name__ == "__main__":
    main()
