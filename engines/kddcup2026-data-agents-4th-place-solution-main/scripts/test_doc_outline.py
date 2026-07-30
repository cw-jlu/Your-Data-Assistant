"""POC: extract section outline + line ranges from .md docs.

For each `### / ####` header, output:
  section_path  | line_start - line_end  | first_entity_id (= ID anchor hint)

The idea: inject this outline into preamble so agent navigates with
`read_doc(path, offset=line_N)` instead of blind grep scanning.
"""
from __future__ import annotations
import re
import json
from pathlib import Path

ROOT = Path("/home/kekshibata-admin/projects/kddcup2026-kobushi")


def parse_doc_outline(md_path: Path) -> list[dict]:
    """Return list of {level, title, line_start, line_end, first_id, sample_keys}."""
    text = md_path.read_text()
    lines = text.split("\n")
    n = len(lines)
    headers = []
    for i, line in enumerate(lines, start=1):
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            headers.append({"level": len(m.group(1)), "title": m.group(2).strip(), "line_start": i})

    # Compute line_end for each (= next same-or-higher level header line - 1, or EOF)
    for idx, h in enumerate(headers):
        end_line = n
        for nxt in headers[idx+1:]:
            if nxt["level"] <= h["level"]:
                end_line = nxt["line_start"] - 1
                break
        h["line_end"] = end_line

    # For each LEAF section (= one with no children at lower level), extract:
    #   - first ID anchor in that section's paragraphs
    #   - sample of numeric measurements / key terms
    for idx, h in enumerate(headers):
        # Is this a leaf? (= no immediate child header within its range)
        has_child = any(
            nxt["level"] > h["level"] and h["line_start"] < nxt["line_start"] <= h["line_end"]
            for nxt in headers[idx+1:]
        )
        h["is_leaf"] = not has_child
        # Extract first ID anchor and sample numeric measurements
        section_lines = lines[h["line_start"]:h["line_end"]]
        section_text = "\n".join(section_lines)
        # ID anchor patterns: "patient 3182521", "identifier 169", "code recXXX", "ID 43003", "rec[A-Za-z0-9]+"
        id_m = re.search(
            r"\b(patient|identifier|code|number|reference|ID|registry)\s+[a-z]*\s*([0-9]{2,12}|rec[A-Za-z0-9]{6,16})\b",
            section_text, re.IGNORECASE)
        h["first_id_anchor"] = id_m.group(0) if id_m else None
        # Sample of measurement terms (= "Xxx of N unit" or "Xxx is N")
        meas = re.findall(r"\b([A-Z]{2,5})\s+(?:of|was|is|reported as|level was)\s+([0-9.]+)\s*([A-Za-z/]+)\b", section_text)
        h["sample_measurements"] = list({m[0] for m in meas[:5]})
    return headers


def format_outline(headers: list[dict], path: Path) -> str:
    out = [f"=== {path.name} ({headers[0]['line_end']} lines) ==="]
    for h in headers:
        indent = "  " * (h["level"] - 1)
        title = h["title"][:90]
        line_info = f"L{h['line_start']:>4}-{h['line_end']:>4}"
        leaf = " [LEAF]" if h.get("is_leaf") else ""
        id_hint = f" | id-ex: {h['first_id_anchor']}" if h.get("first_id_anchor") else ""
        meas_hint = f" | meas: {','.join(h.get('sample_measurements', []))}" if h.get("sample_measurements") else ""
        out.append(f"{indent}{'#'*h['level']} {title:<70s} {line_info}{leaf}{id_hint}{meas_hint}")
    return "\n".join(out)


def main():
    # Try the 3 failing doc-heavy tasks
    failure_tasks = ["task_396", "task_418", "task_344", "task_173"]
    # Plus a few successful ones for contrast
    success_tasks = ["task_352", "task_408", "task_330"]
    for tid in failure_tasks + success_tasks:
        ctx = ROOT / f"data/public/input/{tid}/context"
        if not ctx.exists():
            continue
        mds = list(ctx.glob("doc/*.md"))
        if not mds:
            print(f"\n=== {tid}: no doc/ files ===\n")
            continue
        q = json.load(open(ROOT / f"data/public/input/{tid}/task.json"))["question"]
        print(f"\n{'='*90}")
        print(f"{tid}")
        print(f"Q: {q}")
        for md in mds:
            outline = parse_doc_outline(md)
            print(format_outline(outline, md))


if __name__ == "__main__":
    main()
