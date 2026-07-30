#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
except Exception:
    pass

from experiments.exp_156_column_advisor_poc.column_advisor import (  # noqa: E402
    advise_columns,
    advise_columns_with_profile,
    prompt_text,
)
from experiments.exp_156_column_advisor_poc.context_profile import (  # noqa: E402
    build_context_profile,
)
from kobushi_core.model import ModelMessage, OpenAIModelAdapter  # noqa: E402

try:
    from experiments.exp_155_phase_tool_visibility.prefix_cache import (  # noqa: E402
        with_prefix_cache_header,
    )
except Exception:  # pragma: no cover
    def with_prefix_cache_header(headers: dict[str, str], task_id: str | None = None) -> dict[str, str]:
        del task_id
        return headers


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"
DEFAULT_TASKS = ["task_1", "task_7", "task_23", "task_25", "task_42", "task_60"]


def _task_sort_key(task_id: str) -> int:
    try:
        return int(task_id.split("_", 1)[1])
    except Exception:
        return 10**9


def _load_question(task_id: str) -> str:
    path = INPUT_ROOT / task_id / "task.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["question"])


def _load_gold_table(task_id: str) -> tuple[list[str], list[list[str]]]:
    path = GOLD_ROOT / task_id / "gold.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _make_judge_model(task_id: str | None = None) -> OpenAIModelAdapter:
    api_base = os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL") or ""
    api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY") or ""
    model_name = (
        os.environ.get("COLUMN_ADVISOR_JUDGE_MODEL")
        or os.environ.get("COLUMN_ADVISOR_MODEL")
        or os.environ.get("AGENT_MODEL")
        or os.environ.get("MODEL_NAME")
        or "qwen3.5-35b-a3b"
    )
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.0,
        extra_headers=headers,
    )


_JUDGE_SYSTEM = """You judge a COLUMN ADVISOR output.

The advisor saw only the question and returned comma-separated final CSV column
semantics. Compare it with the gold final answer columns.

Ignore header aliases, language, and schema-specific names. Judge semantics only.
PASS only if the advisor columns are exactly the needed final output semantics:
- no missing gold output column
- no extra sort/filter/ranking/threshold/date/id/evidence column unless the
  question explicitly asks to output it
- same number of semantic output columns

Gold column names may be technical. Use the question and gold preview values to
infer their semantics.
When a question asks "which/name/list <entity>", an advisor column like
"fund manager", "company", or "fund" is acceptable for the entity's displayed
name/descriptor column. Do not require the advisor to literally say "name" or
match a gold header such as Name/姓名.

Output exactly one line:
PASS: <short reason>
or
FAIL: <short reason>
""".strip()


def _judge_content(
    *,
    task_id: str,
    question: str,
    advisor_columns: list[str],
    gold_columns: list[str],
    gold_preview_rows: list[list[str]],
) -> tuple[bool | None, str, str]:
    preview = gold_preview_rows[:3]
    user = (
        f"Question:\n{question}\n\n"
        f"Advisor columns:\n{', '.join(advisor_columns)}\n\n"
        f"Gold columns:\n{', '.join(gold_columns)}\n\n"
        f"Gold preview rows:\n{json.dumps(preview, ensure_ascii=False)}\n\n"
        "Verdict:"
    )
    try:
        raw = _make_judge_model(task_id).complete(
            [
                ModelMessage(role="system", content=_JUDGE_SYSTEM),
                ModelMessage(role="user", content=user),
            ],
            enable_thinking=False,
            max_tokens=160,
        ).strip()
    except Exception as exc:
        return None, f"judge_error={exc!r}", ""
    upper = raw.upper()
    if upper.startswith("PASS"):
        return True, raw, raw
    if upper.startswith("FAIL"):
        return False, raw, raw
    return None, f"unparseable_judge={raw}", raw


def _parse_tasks(raw: str | None, *, all_tasks: bool) -> list[str]:
    if all_tasks:
        return sorted(
            [p.name for p in INPUT_ROOT.glob("task_*") if p.is_dir()],
            key=_task_sort_key,
        )
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return DEFAULT_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 question-only comma-output column advisor PoC."
    )
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--all", action="store_true", help="run all 60 demo tasks")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="use deterministic heuristic fallback instead of the model",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the model-based semantic content judge",
    )
    parser.add_argument(
        "--with-profile",
        action="store_true",
        help="include a compact non-gold context data profile in the advisor input",
    )
    parser.add_argument("--profile-max-chars", type=int, default=18000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts" / "column_advisor_poc",
    )
    parser.add_argument(
        "--print-prompts",
        action="store_true",
        help="print the English and Chinese advisor prompts and exit",
    )
    args = parser.parse_args()

    if args.print_prompts:
        print("=== English prompt ===")
        print(prompt_text("en"))
        print("\n=== Chinese prompt ===")
        print(prompt_text("zh"))
        return

    task_ids = _parse_tasks(args.tasks, all_tasks=args.all)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        question = _load_question(task_id)
        gold_columns, gold_rows = _load_gold_table(task_id)
        profile = ""
        if args.with_profile:
            profile = build_context_profile(
                INPUT_ROOT / task_id / "context",
                max_chars=args.profile_max_chars,
            )
            result = advise_columns_with_profile(
                question,
                profile,
                task_id=task_id,
                use_model=not args.no_model,
            )
        else:
            result = advise_columns(
                question,
                task_id=task_id,
                use_model=not args.no_model,
            )
        content_match: bool | None = None
        content_reason = ""
        judge_raw = ""
        count_match = result.column_count == len(gold_columns)
        if not count_match:
            content_match = False
            content_reason = (
                f"FAIL: column count mismatch "
                f"(advisor={result.column_count}, gold={len(gold_columns)})"
            )
        elif not args.no_judge:
            content_match, content_reason, judge_raw = _judge_content(
                task_id=task_id,
                question=question,
                advisor_columns=result.columns,
                gold_columns=gold_columns,
                gold_preview_rows=gold_rows,
            )
        row = {
            "task_id": task_id,
            "prompt_lang": result.prompt_lang,
            "question": question,
            "advisor_columns": result.line,
            "advisor_count": result.column_count,
            "gold_columns": ", ".join(gold_columns),
            "gold_count": len(gold_columns),
            "count_match": count_match,
            "content_match": content_match,
            "content_reason": content_reason,
            "profile_chars": len(profile),
            "fallback_used": result.fallback_used,
            "error": result.error or "",
            "raw": result.raw,
            "judge_raw": judge_raw,
        }
        rows.append(row)
        mark = "OK" if row["count_match"] else "MISS"
        if content_match is True:
            content_mark = "PASS"
        elif content_match is False:
            content_mark = "FAIL"
        else:
            content_mark = "SKIP"
        fallback = " fallback" if result.fallback_used else ""
        print(
            f"{mark:<4} {content_mark:<4} {task_id:<8} advisor=[{result.line}] "
            f"gold_count={len(gold_columns)}{fallback}",
            flush=True,
        )

    csv_path = args.out_dir / "phase2_column_advisor_poc.csv"
    json_path = args.out_dir / "phase2_column_advisor_poc.json"
    fieldnames = [
        "task_id",
        "prompt_lang",
        "question",
        "advisor_columns",
        "advisor_count",
        "gold_columns",
        "gold_count",
        "count_match",
        "content_match",
        "content_reason",
        "profile_chars",
        "fallback_used",
        "error",
        "raw",
        "judge_raw",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n = len(rows)
    matches = sum(1 for row in rows if row["count_match"])
    judged = [row for row in rows if row["content_match"] is not None]
    content_matches = sum(1 for row in judged if row["content_match"] is True)
    print(f"\ncount_match={matches}/{n} ({matches / n:.1%})")
    if judged:
        print(
            f"content_match={content_matches}/{len(judged)} "
            f"({content_matches / len(judged):.1%})"
        )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
