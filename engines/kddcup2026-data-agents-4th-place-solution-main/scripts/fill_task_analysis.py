"""Fill TODO blocks in docs/PUBLIC_TASK_ANALYSIS.md from a per-task dict.

Each entry: task_id → {steps, failure, memo}.
The script replaces:
    **正解までの reasoning steps**: <!-- TODO: fill per-task -->
    **失敗モード / 過去 trace の傾向**: <!-- TODO: fill per-task -->
    **メモ**: <!-- TODO: fill per-task -->

with the corresponding multi-line content.

Per-task analyses live in docs/_task_fills.py to keep them out of the script.
Re-run after editing _task_fills.py.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MD_PATH = REPO / "docs" / "PUBLIC_TASK_ANALYSIS.md"
FILLS_PATH = REPO / "scripts" / "_task_fills.py"


def load_fills() -> dict:
    spec = importlib.util.spec_from_file_location("_task_fills", FILLS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FILLS


def main():
    if not FILLS_PATH.is_file():
        print(f"Missing {FILLS_PATH}; create it with FILLS dict")
        sys.exit(1)
    fills = load_fills()
    text = MD_PATH.read_text()

    # Split into per-task blocks (### task_NN ... ---)
    pattern = re.compile(r"^### (task_\d+)\b", re.MULTILINE)
    sections = []
    last = 0
    last_tid = None
    for m in pattern.finditer(text):
        if last_tid is not None:
            sections.append((last_tid, last, m.start()))
        last_tid = m.group(1)
        last = m.start()
    if last_tid:
        sections.append((last_tid, last, len(text)))

    n_filled = 0
    new_text = text
    # Process from bottom to top to preserve indices
    for tid, start, end in reversed(sections):
        if tid not in fills:
            continue
        block = new_text[start:end]
        f = fills[tid]
        # Replace reasoning steps
        if f.get("steps"):
            new_block = re.sub(
                r"\*\*正解までの reasoning steps\*\*: <!-- TODO: fill per-task -->",
                f"**正解までの reasoning steps**:\n{f['steps']}",
                block, count=1,
            )
        else:
            new_block = block
        # Replace failure mode
        if f.get("failure"):
            new_block = re.sub(
                r"\*\*失敗モード / 過去 trace の傾向\*\*: <!-- TODO: fill per-task -->",
                f"**失敗モード / 過去 trace の傾向**:\n{f['failure']}",
                new_block, count=1,
            )
        # Replace memo
        if f.get("memo"):
            new_block = re.sub(
                r"\*\*メモ\*\*: <!-- TODO: fill per-task -->",
                f"**メモ**:\n{f['memo']}",
                new_block, count=1,
            )
        if new_block != block:
            new_text = new_text[:start] + new_block + new_text[end:]
            n_filled += 1

    MD_PATH.write_text(new_text)
    print(f"filled {n_filled} task sections in {MD_PATH.name}")


if __name__ == "__main__":
    main()
