"""Pre-compute "Knowledge Brief" per task — STRUCTURE-ONLY transform.

Generic algorithm with NO question-specific selection. The preamble
already includes the full knowledge.md and a glossary section. The
brief only adds ONE structural change: surface `## Ambiguity Resolution`
(= section 6) at the very top, with a universal instruction to check
the listed terms against the question's nouns BEFORE computing.

Rationale (DOC_TEMPLATE_ANALYSIS.md):
- Section 6 is templated (= 100% of public knowledge.md have it)
- Section 6 contains synonym→column mappings critical for question
  interpretation (e.g. "cost" → check `amount` column)
- When section 6 sits at the bottom, agents skim past it and pick the
  wrong column

This is universal — same algorithm applies to any task. No leakage of
task-specific solutions.

Output: artifacts/qr_brief_cache/<task_id>.md

Usage:
    uv run python scripts/build_qr_brief.py [--task task_25]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kobushi_core.benchmark.dataset import DABenchPublicDataset

CACHE = REPO / "artifacts" / "qr_brief_cache"
CACHE.mkdir(parents=True, exist_ok=True)


SECTION6_RE = re.compile(
    r'^##\s*\d*\.?\s*Ambiguity Resolution.+?(?=^##(?!\#)|\Z)',
    re.MULTILINE | re.DOTALL,
)


def extract_section6(text: str) -> str:
    """Return the Ambiguity Resolution section verbatim (or empty)."""
    m = SECTION6_RE.search(text)
    return m.group(0).strip() if m else ""


def build_brief(knowledge_md_text: str) -> str:
    """Compose a STRUCTURE-ONLY brief.

    The same algorithm applies to any knowledge.md regardless of question.
    Only adds section 6 (Ambiguity Resolution) verbatim at the top with
    a universal instruction. The full knowledge.md remains in context
    via the preamble — this brief just re-emphasizes a section that
    agents tend to skim past when it sits at the bottom.
    """
    section6 = extract_section6(knowledge_md_text)
    if not section6:
        return ""  # No ambiguity section → nothing to surface

    lines = ["# Knowledge Brief (= section 6 surfaced)"]
    lines.append("")
    lines.append(
        "The full `knowledge.md` is included below. This brief surfaces "
        "**section 6 (Ambiguity Resolution)** at the top because the "
        "interpretive choices it lists determine which schema columns "
        "the question's nouns map to. Read this BEFORE the rest of the "
        "knowledge.md."
    )
    lines.append("")
    lines.append("## ⚠️ READ FIRST: Ambiguity Resolution")
    lines.append("")
    lines.append(
        "Before computing, scan the list below. The terms it flags may "
        "be synonyms of the nouns in your question (e.g. \"cost\" ↔ "
        "`amount`, \"name\" ↔ `event_name` / `member_name`, \"date\" ↔ "
        "`event_date` / `expense_date`). If your question uses such a "
        "noun, state your column choice and reasoning explicitly in the "
        "first thought."
    )
    lines.append("")
    # Strip the leading "## ... Ambiguity Resolution" header to avoid
    # duplication; keep only the body.
    body = re.sub(
        r'^##\s*\d*\.?\s*Ambiguity Resolution\s*$',
        '',
        section6,
        count=1,
        flags=re.MULTILINE,
    ).lstrip()
    lines.append(body)
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="single task_id (default: all)")
    args = ap.parse_args()

    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task_ids = [args.task] if args.task else ds.list_task_ids()

    written = 0
    skipped = 0
    for tid in task_ids:
        task = ds.get_task(tid)
        km = task.context_dir / "knowledge.md"
        if not km.exists():
            print(f"  {tid}: knowledge.md not found, skip")
            skipped += 1
            continue
        brief = build_brief(km.read_text())
        out = CACHE / f"{tid}.md"
        if not brief:
            # No ambiguity section → no brief; clean stale file if present.
            if out.exists():
                out.unlink()
            print(f"  {tid}: no section 6, skip")
            skipped += 1
            continue
        out.write_text(brief)
        written += 1
        print(f"  {tid}: brief {brief.count(chr(10))} lines / {len(brief)} chars")

    print(f"\nWrote {written} briefs ({skipped} skipped) to {CACHE}")


if __name__ == "__main__":
    main()
