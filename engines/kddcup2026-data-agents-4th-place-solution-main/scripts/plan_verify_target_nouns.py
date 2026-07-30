"""Plan verifier POC: extract target nouns (= LLM sub-agent), then regex-match
against schema columns across ALL tables. Flag if PLAN didn't use a literal
match that exists.

Pipeline:
  1. LLM sub-agent: extract target nouns from question (= small focused task)
  2. Rule-based: regex search every schema column for literal match
  3. Compare to PLAN's chosen columns (= string contains check)
  4. Flag issues for re-PLAN

Usage:
    uv run python scripts/plan_verify_target_nouns.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


OUT_DIR = REPO / "artifacts" / "plan_verify_nouns"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAIL_TASKS = ["task_25", "task_163", "task_169"]  # known failures with real plans
OK_TASKS = ["task_11", "task_67", "task_74", "task_75", "task_19"]  # known passing
TARGET_TASKS = FAIL_TASKS + OK_TASKS


# =========================================================================
# Step 1: target-noun extractor sub-agent
# =========================================================================

_NOUN_EXTRACTOR_SYS = """You extract NOUNS from a question that the SQL answer must touch.
Focus on:
  - the SUBJECT (= what entity/attribute to return)
  - the OBJECT (= what entity/attribute to filter on)
  - any AGGREGATE term (= "total", "average", "count" - keep as a separate noun)

Rules:
  - Lowercase, singular (e.g., "expenses" → "expense").
  - Drop articles (a, the).
  - Keep multi-word concepts as a single string if they're a unit (e.g., "first name", "monthly consumption").
  - Output ONLY a JSON array of nouns, no other text.

Examples:
  Q "Which event has the lowest cost?" → ["event", "cost"]
  Q "Identify the type of expenses and their total value approved." → ["type", "expense", "total value", "approved"]
  Q "What is the average weight of female superheroes?" → ["average weight", "female", "superhero"]
"""


def extract_target_nouns(question: str, model: OpenAIModelAdapter) -> list[str]:
    response = model.complete(
        [
            ModelMessage(role="system", content=_NOUN_EXTRACTOR_SYS),
            ModelMessage(role="user", content=f"Q: {question}\n\nOutput the JSON array:"),
        ],
        enable_thinking=False,
        max_tokens=256,
    )
    # Extract JSON array
    m = re.search(r"\[.*?\]", response, re.DOTALL)
    if not m:
        return []
    try:
        result = json.loads(m.group(0))
        if isinstance(result, list):
            return [str(x).lower().strip() for x in result if x]
    except Exception:
        pass
    return []


# =========================================================================
# Step 2: schema column lookup (= rule-based)
# =========================================================================

def load_schema(task) -> dict[str, list[str]]:
    """Return {table_name: [column_names]} for the task's context."""
    try:
        res = execute_sql(task.context_dir, "SHOW TABLES")
        tables = [r[0] for r in res["rows"]]
    except Exception:
        return {}
    schema = {}
    for tbl in tables:
        try:
            cols = execute_sql(task.context_dir, f"DESCRIBE {tbl}")
            schema[tbl] = [r[0] for r in cols["rows"]]
        except Exception:
            schema[tbl] = []
    return schema


def find_literal_matches(noun: str, schema: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """Rule-based regex match. Returns list of (table, column, match_type).

    match_type:
      - "exact": column name == noun (case insensitive)
      - "plural_singular": singular/plural variation matches
      - "contains": noun is a substring of column name (or vice versa)
    """
    noun_lower = noun.lower().strip()
    if not noun_lower:
        return []
    # Multi-word noun: try its key word (= last meaningful word)
    words = noun_lower.split()
    candidates = [noun_lower]
    if len(words) > 1:
        candidates.extend(words)  # also try individual words

    matches: list[tuple[str, str, str]] = []
    for table, cols in schema.items():
        for col in cols:
            col_norm = col.lower().replace("_", "").replace(" ", "")
            for cand in candidates:
                cand_norm = cand.lower().replace("_", "").replace(" ", "")
                if not cand_norm:
                    continue
                if col_norm == cand_norm:
                    matches.append((table, col, "exact"))
                    break
                # Singular/plural
                if col_norm == cand_norm + "s" or col_norm + "s" == cand_norm:
                    matches.append((table, col, "plural_singular"))
                    break
                # Contains (= longer relaxation)
                if len(cand_norm) >= 4 and cand_norm in col_norm:
                    matches.append((table, col, "contains"))
                    break
                if len(col_norm) >= 4 and col_norm in cand_norm:
                    matches.append((table, col, "contains"))
                    break
    # Dedupe
    seen = set()
    out = []
    for m in matches:
        key = (m[0], m[1])
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


# =========================================================================
# Step 3: PLAN verifier
# =========================================================================

_TABLE_DOT_COL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def extract_chosen_refs(plan_text: str, schema: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Extract every (table, column) reference the plan mentions.
    Validates against schema so noise tokens don't get treated as refs."""
    valid_pairs = {(t.lower(), c.lower()) for t, cs in schema.items() for c in cs}
    found = set()
    for m in _TABLE_DOT_COL_RE.findall(plan_text):
        t, c = m[0].lower(), m[1].lower()
        if (t, c) in valid_pairs:
            found.add((t, c))
    return found


def check_plan_uses(
    plan_text: str, matches: list[tuple[str, str, str]], schema: dict[str, list[str]]
) -> list[bool]:
    """For each literal match (table, col, type), check if plan references it."""
    chosen = extract_chosen_refs(plan_text, schema)
    return [(table.lower(), col.lower()) in chosen for (table, col, _) in matches]


def verify(task_id: str, plan_text: str, model: OpenAIModelAdapter) -> dict:
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    task = ds.get_task(task_id)
    schema = load_schema(task)
    table_names_lower = {t.lower() for t in schema}
    nouns = extract_target_nouns(task.question, model)
    findings = []
    for noun in nouns:
        noun_norm = noun.lower().replace(" ", "")
        # Skip noun that IS a table name (= table already implicitly chosen)
        if noun_norm in table_names_lower or noun_norm.rstrip("s") in table_names_lower:
            findings.append({"noun": noun, "matches": [], "issue": None, "skipped": "table_name"})
            continue
        matches = find_literal_matches(noun, schema)
        # Only consider EXACT or PLURAL/SINGULAR matches as issue triggers
        strong = [m for m in matches if m[2] in ("exact", "plural_singular")]
        if not strong:
            findings.append({"noun": noun, "matches": matches, "issue": None})
            continue
        used = check_plan_uses(plan_text, strong, schema)
        used_matches = [m for m, u in zip(strong, used) if u]
        unused = [m for m, u in zip(strong, used) if not u]
        issue = None
        if not used_matches and unused:
            issue = f"unused exact match for '{noun}': {unused}"
        findings.append({
            "noun": noun,
            "matches": matches,
            "used": used_matches,
            "issue": issue,
        })
    return {
        "task_id": task_id,
        "question": task.question,
        "nouns": nouns,
        "findings": findings,
        "any_issue": any(f.get("issue") for f in findings),
    }


# =========================================================================
# Driver
# =========================================================================

REAL_PLAN_DIR = REPO / "artifacts" / "plan_explore_compare"


def load_real_plan_text(task_id: str) -> str | None:
    """Load the first available exp_122 plan-step thought for a task."""
    task_dir = REAL_PLAN_DIR / task_id
    if not task_dir.exists():
        return None
    for f in sorted(task_dir.glob("exp_122_*.md")):
        text = f.read_text()
        thoughts = re.findall(r"\*\*thought\*\*:\s*```(.*?)```", text, re.DOTALL)
        if thoughts:
            return "\n".join(thoughts)
    return None


_ONESHOT_PLAN_SYS = """You are running the PLAN phase of a SQL agent. Given a question and schema preview, output a PLAN that:
  - states column_count and per_column for the answer
  - INTERPRET each noun: list candidate columns, then CHOOSING <table>.<column> with reason
  - INTERPRET semantics (filters, joins) similarly
Be terse and explicit. Use the format from earlier examples."""


def generate_plan_text_oneshot(task, model: OpenAIModelAdapter) -> str:
    """Generate a synthetic PLAN text via 1-shot LLM call (= no tools)."""
    from experiments.exp_122_column_auditor.preamble import build_preamble
    preamble = build_preamble(task)
    user = f"{preamble.text}\n\n# Question\n{task.question}\n\nWrite the PLAN now."
    try:
        return model.complete(
            [
                ModelMessage(role="system", content=_ONESHOT_PLAN_SYS),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=2048,
        )
    except Exception as exc:
        return f"<plan generation failed: {exc}>"


def get_plan_text(task_id: str, model: OpenAIModelAdapter) -> tuple[str, str]:
    """Return (source, plan_text). Prefer real, fall back to 1-shot."""
    real = load_real_plan_text(task_id)
    if real:
        return ("real", real)
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    return ("oneshot", generate_plan_text_oneshot(ds.get_task(task_id), model))


def make_model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.3,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def verify_with_source(tid: str, model: OpenAIModelAdapter) -> dict:
    src, plan_text = get_plan_text(tid, model)
    r = verify(tid, plan_text, model)
    r["plan_source"] = src
    r["plan_text_preview"] = plan_text[:500]
    return r


def main():
    model = make_model()
    print(f"=== Plan target-noun verifier POC: {len(TARGET_TASKS)} tasks ===", flush=True)
    print(f"  FAIL_TASKS (expect flag): {FAIL_TASKS}", flush=True)
    print(f"  OK_TASKS (expect no flag): {OK_TASKS}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(verify_with_source, tid, model): tid for tid in TARGET_TASKS}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                cat = "FAIL" if tid in FAIL_TASKS else "ok  "
                print(f"  [{cat}] {tid} src={r['plan_source']:7s} flag={r['any_issue']}", flush=True)
                for f in r["findings"]:
                    if f.get("issue"):
                        print(f"    FLAG '{f['noun']}': {f['issue']}", flush=True)
            except Exception as exc:
                print(f"  {tid}: ERROR {exc}", flush=True)

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # Confusion matrix
    tp = sum(1 for r in results if r["task_id"] in FAIL_TASKS and r["any_issue"])
    fn = sum(1 for r in results if r["task_id"] in FAIL_TASKS and not r["any_issue"])
    fp = sum(1 for r in results if r["task_id"] in OK_TASKS and r["any_issue"])
    tn = sum(1 for r in results if r["task_id"] in OK_TASKS and not r["any_issue"])
    lines = [
        "# Plan target-noun verifier — Summary\n",
        f"## Confusion (flag = fail)",
        f"- TP (fail flagged): {tp}/{len(FAIL_TASKS)}",
        f"- FN (fail missed): {fn}/{len(FAIL_TASKS)}",
        f"- FP (ok flagged):  {fp}/{len(OK_TASKS)}",
        f"- TN (ok cleared):  {tn}/{len(OK_TASKS)}",
        "",
        "| task | category | source | flag | flagged_nouns |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: int(x["task_id"].split("_")[1])):
        cat = "FAIL" if r["task_id"] in FAIL_TASKS else "ok"
        flagged = [f["noun"] for f in r["findings"] if f.get("issue")]
        lines.append(
            f"| {r['task_id']} | {cat} | {r['plan_source']} | "
            f"{'X' if r['any_issue'] else '-'} | {flagged} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nTP={tp}/{len(FAIL_TASKS)} FP={fp}/{len(OK_TASKS)} → summary: {OUT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
