#!/usr/bin/env python3
"""Compute a normalized diff between two experiment directories.

Each experiment lives in its own copy of src/experiments/exp_NNN_<slug>/, so a
plain `diff -urN` between them is dominated by module-rename noise (every
`from experiments.exp_NNN_<base>.foo import X` line shows as a change).

This script normalizes the package name to `experiments.EXP` in both
directories before diffing, so the resulting patch only contains real edits:
prompt text, preamble lines, tool registrations, runner logic, etc.

Usage:
    python scripts/compute_exp_diff.py <base_exp> <target_exp> [-o <out_path>]

If -o is given, the diff is written there. Otherwise it's printed to stdout.
Exit code: 0 if any diff produced, 0 also when identical (empty diff is fine).
Non-zero only on missing dirs / IO errors.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / "src" / "experiments"

CANONICAL = "EXP"


def normalize_dir(src: Path, exp_name: str, dst: Path) -> None:
    """Copy src/ to dst/, replacing `experiments.<exp_name>` with `experiments.EXP`
    in all .py / .yaml / .md files. Skips __pycache__ and binary blobs.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Replace `experiments.<exp_name>` (no trailing identifier chars) and the
    # bare directory name `exp_NNN_<slug>` when it appears in module strings.
    pattern_pkg = re.compile(rf"\bexperiments\.{re.escape(exp_name)}\b")

    for path in dst.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".cfg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = pattern_pkg.sub(f"experiments.{CANONICAL}", text)
        if new != text:
            path.write_text(new, encoding="utf-8")


def compute_diff(base: str, target: str) -> str:
    base_src = EXP_ROOT / base
    tgt_src = EXP_ROOT / target
    if not base_src.is_dir():
        raise SystemExit(f"base experiment dir not found: {base_src}")
    if not tgt_src.is_dir():
        raise SystemExit(f"target experiment dir not found: {tgt_src}")

    with tempfile.TemporaryDirectory(prefix="expdiff_") as tmp:
        tmp_path = Path(tmp)
        base_norm = tmp_path / f"BASE_{base}"
        tgt_norm = tmp_path / f"TGT_{target}"
        normalize_dir(base_src, base, base_norm)
        normalize_dir(tgt_src, target, tgt_norm)

        # Use stable labels so the patch reads as base→target without temp paths.
        cmd = [
            "diff",
            "-urN",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            f"--label=a/{base}",
            f"--label=b/{target}",
            str(base_norm),
            str(tgt_norm),
        ]
        # diff returns 1 when files differ — that's expected.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            raise SystemExit(
                f"diff failed rc={proc.returncode} stderr={proc.stderr.strip()}"
            )

        # Strip the temp prefixes so the patch is reproducible / canonical.
        out = proc.stdout
        out = out.replace(str(base_norm) + "/", f"a/{base}/")
        out = out.replace(str(tgt_norm) + "/", f"b/{target}/")
        out = out.replace(str(base_norm), f"a/{base}")
        out = out.replace(str(tgt_norm), f"b/{target}")
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="base experiment name, e.g. exp_040_selfdbg_fence")
    ap.add_argument("target", help="target experiment name")
    ap.add_argument("-o", "--out", help="write diff here instead of stdout")
    args = ap.parse_args()

    diff_text = compute_diff(args.base, args.target)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(diff_text, encoding="utf-8")
        n_lines = diff_text.count("\n")
        print(f"wrote {n_lines} lines to {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(diff_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
