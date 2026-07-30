#!/usr/bin/env python3
"""Trace viewer v3 — v2 matrix viewer generalised to ANY task root.

v2 hard-codes the Phase-2 demo input/gold roots, so runs over other task sets
(e.g. the external eval set data/external/eval_set_v1) render with empty
question/gold/prediction. v3 makes the input+gold roots configurable and, when
a manifest.json is present, tags each task column with its source
(phase1 / bull / ehrsql) so you can see at a glance where failures cluster.

Run (external eval set is the default root):
  uv run python scripts/trace_viewer_v3.py --port 8777

Point at any DABench-layout root (expects <root>/input + <root>/output):
  uv run python scripts/trace_viewer_v3.py --task-root data/phase2_demo/demo_samples_phase2

All other flags (--host, --require-cf-access, --auth-token, ...) pass through to v2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import trace_viewer as v1  # noqa: E402
import trace_viewer_v2 as v2  # noqa: E402

REPO = SCRIPT_DIR.parent
# Default to the Phase-2 demo set (the familiar 60-task view). Point at any other
# DABench-layout root with --task-root (e.g. data/external/eval_set_v1).
DEFAULT_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"


def _apply_roots(input_root: Path, gold_root: Path) -> None:
    """Repoint EVERY root v1/v2 use so questions/gold/prediction resolve for the
    chosen task set. Both phase slots are set to the same root to avoid the
    phase-1 fallback resolving a colliding task_id (e.g. task_20) to the wrong set."""
    v1.PHASE2_INPUT_ROOT = input_root
    v1.PHASE2_GOLD_ROOT = gold_root
    v1.PHASE1_INPUT_ROOT = input_root
    v1.PHASE1_GOLD_ROOT = gold_root


def _install_source_tags(manifest_path: Path) -> None:
    """Wrap v2.get_matrix so each task column carries its manifest source and the
    tooltip/question is prefixed with [source]. No frontend edit needed — v2 already
    renders meta.question as the column tooltip."""
    if not manifest_path.is_file():
        return
    src = {m["task_id"]: m.get("source") for m in json.loads(manifest_path.read_text())}
    _orig = v2.get_matrix

    def get_matrix_v3():
        mat = _orig()
        for tm in mat.get("tasks", []):
            s = src.get(tm.get("task_id"))
            tm["source"] = s
            if s and tm.get("question"):
                tm["question"] = f"[{s}] {tm['question']}"
        return mat

    v2.get_matrix = get_matrix_v3  # handler calls the v2-module global


def _install_union_task_columns() -> None:
    """Pin the matrix column list to a startup SNAPSHOT of the configured root's
    tasks (60 for the Phase-2 demo).

    The per-run scoring wrapper (`_install_per_run_scoring`) temporarily repoints
    v1.PHASE2_INPUT_ROOT at each run's own `_task_root` while building cells. If a
    matrix column query (`v2._phase2_task_ids`) races that window while an eval_set
    run (40 tasks) is rendering, it would read the 40-task root and drop the
    task_41..task_60 columns. Snapshotting the configured root once — and returning
    it regardless of the live root — makes the column list immune to that race.

    Eval-set runs (task_1..task_40 ⊂ task_1..task_60) still render correctly in the
    shared columns; their per-task gold/prediction resolve via `_task_root`."""
    snapshot = sorted(set(v2._phase2_task_ids()), key=v1._task_sort_key)

    def pinned_ids():
        return snapshot

    v2._phase2_task_ids = pinned_ids


def _run_roots(run_dir: Path, default_input: Path, default_gold: Path):
    marker = run_dir / "_task_root"
    if marker.is_file():
        r = REPO / marker.read_text().strip()
        if (r / "input").is_dir():
            return r / "input", r / "output"
    return default_input, default_gold


def _install_per_run_scoring(default_input: Path, default_gold: Path) -> None:
    """The matrix live-scores predictions vs gold when a run has no summary.json
    (still running). Route that scoring through EACH run's own gold by wrapping the
    two run-dir-aware helpers so an eval_set_v1 run isn't scored against Phase-2 gold."""
    import threading
    lock = threading.Lock()

    def wrap(orig):
        def w(run_dir, *a, **k):
            inp, gold = _run_roots(Path(run_dir), default_input, default_gold)
            with lock:
                saved = (v1.PHASE2_INPUT_ROOT, v1.PHASE2_GOLD_ROOT,
                         v1.PHASE1_INPUT_ROOT, v1.PHASE1_GOLD_ROOT)
                v1.PHASE2_INPUT_ROOT, v1.PHASE2_GOLD_ROOT = inp, gold
                v1.PHASE1_INPUT_ROOT, v1.PHASE1_GOLD_ROOT = inp, gold
                try:
                    return orig(run_dir, *a, **k)
                finally:
                    (v1.PHASE2_INPUT_ROOT, v1.PHASE2_GOLD_ROOT,
                     v1.PHASE1_INPUT_ROOT, v1.PHASE1_GOLD_ROOT) = saved
        return w

    v2._build_cells = wrap(v2._build_cells)
    v1._run_summary = wrap(v1._run_summary)


def _install_per_run_root(default_input: Path, default_gold: Path) -> None:
    """Resolve each run's question/gold/prediction from the root THAT RUN used.

    A run that ran over a non-default task set writes `<run>/_task_root` (a repo-
    relative path). The task-detail endpoint reads it and resolves against that
    root, so e.g. an eval_set_v1 run shows BULL/EHR gold while Phase-2 runs still
    show Phase-2 gold — in the same viewer, no collision."""
    import threading
    _orig = v1.get_task_detail
    _lock = threading.Lock()

    def wrapped(run_id: str, task_id: str):
        inp, gold = default_input, default_gold
        marker = v1.RUNS_ROOT / run_id / "_task_root"
        if marker.is_file():
            r = REPO / marker.read_text().strip()
            if (r / "input").is_dir():
                inp, gold = r / "input", r / "output"
        with _lock:
            saved = (v1.PHASE2_INPUT_ROOT, v1.PHASE2_GOLD_ROOT,
                     v1.PHASE1_INPUT_ROOT, v1.PHASE1_GOLD_ROOT)
            v1.PHASE2_INPUT_ROOT, v1.PHASE2_GOLD_ROOT = inp, gold
            v1.PHASE1_INPUT_ROOT, v1.PHASE1_GOLD_ROOT = inp, gold
            try:
                return _orig(run_id, task_id)
            finally:
                (v1.PHASE2_INPUT_ROOT, v1.PHASE2_GOLD_ROOT,
                 v1.PHASE1_INPUT_ROOT, v1.PHASE1_GOLD_ROOT) = saved

    v1.get_task_detail = wrapped


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True, description="Trace viewer v3 (configurable task root).")
    ap.add_argument("--task-root", default=str(DEFAULT_ROOT),
                    help="DABench-layout root with <root>/input + <root>/output (default: eval_set_v1)")
    ap.add_argument("--input-root", default=None, help="override <task-root>/input")
    ap.add_argument("--gold-root", default=None, help="override <task-root>/output")
    ap.add_argument("--manifest", default=None, help="manifest.json for source tags (default: <task-root>/manifest.json)")
    # everything else is forwarded to v2.main
    args, rest = ap.parse_known_args()

    root = Path(args.task_root)
    input_root = Path(args.input_root) if args.input_root else root / "input"
    gold_root = Path(args.gold_root) if args.gold_root else root / "output"
    manifest = Path(args.manifest) if args.manifest else root / "manifest.json"

    _apply_roots(input_root, gold_root)
    _install_source_tags(manifest)
    _install_union_task_columns()
    _install_per_run_root(input_root, gold_root)
    _install_per_run_scoring(input_root, gold_root)

    print(f"[v3] input_root={input_root}")
    print(f"[v3] gold_root ={gold_root}")
    print(f"[v3] manifest  ={manifest} ({'ok' if manifest.is_file() else 'none'})")

    # hand the remaining args to v2.main (port/host/auth/...). Default port 8777.
    if not any(a == "--port" for a in rest):
        rest += ["--port", "8777"]
    sys.argv = [sys.argv[0], *rest]
    v2.main()


if __name__ == "__main__":
    main()
