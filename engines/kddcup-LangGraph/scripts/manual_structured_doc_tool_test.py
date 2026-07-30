from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import create_chat_model
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.context_preprocessor import prepare_task_context
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRuntimeContext, create_default_tool_registry


DEFAULT_FIELDS = ["personalcode", "totalfundnv"]


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_ok(tool_name: str, result: Any) -> dict[str, Any]:
    if not result.ok:
        raise RuntimeError(f"{tool_name} failed: {json.dumps(result.content, ensure_ascii=False)}")
    return result.content


def _select_blocks(blocks: list[dict[str, Any]], fields: list[str]) -> list[str]:
    selected: list[str] = []
    wanted = set(fields)
    for block in blocks:
        candidate_fields = set(block.get("candidate_fields") or [])
        if candidate_fields & wanted:
            selected.append(str(block["block_id"]))
    return selected


def _parse_line_ranges(value: str | None) -> list[list[int]] | None:
    if not value:
        return None
    ranges: list[list[int]] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid line range {part!r}; expected START-END.")
        raw_start, raw_end = part.split("-", 1)
        start = int(raw_start.strip())
        end = int(raw_end.strip())
        if start <= 0 or end < start:
            raise ValueError(f"Invalid line range {part!r}; expected positive START<=END.")
        ranges.append([start, end])
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Manually exercise inspect_doc_structure + extract_structured_doc "
            "with the same ToolRegistry path used by the main agent."
        )
    )
    parser.add_argument("--config", default="configs/easy.yaml")
    parser.add_argument("--task-id", default="task_3")
    parser.add_argument("--doc", default="doc/mf_fmscaleanalysisn.md")
    parser.add_argument("--target-table", default="mf_fmscaleanalysisn")
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    parser.add_argument(
        "--output-dir",
        default="artifacts/runs/manual_structured_doc_tool_test/task_3",
        help="Trace/log/artifact directory for this manual run.",
    )
    parser.add_argument("--inspect-calls", type=int, default=3)
    parser.add_argument("--extract-calls", type=int, default=20)
    parser.add_argument(
        "--block-ids",
        nargs="*",
        help="Use explicit block ids instead of auto-selecting blocks from inspect_doc_structure.",
    )
    parser.add_argument(
        "--line-ranges",
        help="Use explicit 1-based inclusive line ranges such as '2-52,54-104'.",
    )
    parser.add_argument(
        "--skip-inspect",
        action="store_true",
        help="Skip inspect_doc_structure; requires --line-ranges or --block-ids.",
    )
    args = parser.parse_args()
    explicit_line_ranges = _parse_line_ranges(args.line_ranges)

    config = load_app_config(Path(args.config))
    dataset = DABenchPublicDataset(config.dataset.root_path)
    source_task = dataset.get_task(args.task_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_task_context(source_task, output_dir, video_config=config.video_preprocessing)
    task = prepared.task
    model = create_chat_model(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        api_key_env=config.agent.api_key_env,
        temperature=config.agent.temperature,
        timeout_seconds=config.agent.model_request_timeout_seconds,
        max_tokens=config.agent.max_tokens,
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        budget=config.data_inspector.sample_budget,
        semantic_view_config=config.data_inspector.semantic_views,
        model=model,
        trace_dir=output_dir,
    )

    inspect_args: dict[str, Any] | None = None
    blocks: list[dict[str, Any]] = []
    if args.skip_inspect:
        if not args.block_ids and not explicit_line_ranges:
            raise ValueError("--skip-inspect requires --block-ids or --line-ranges.")
        selected_block_ids = list(args.block_ids or [])
        print("[1/3] inspect_doc_structure skipped")
    else:
        inspect_args = {
            "path": args.doc,
            "knowledge_path": "knowledge.md",
            "target_table": args.target_table,
            "fields": args.fields,
            "max_model_calls": args.inspect_calls,
        }
        print(f"[1/3] inspect_doc_structure args={json.dumps(inspect_args, ensure_ascii=False)}")
        inspect_content = _require_ok(
            "inspect_doc_structure",
            registry.execute(runtime_context, "inspect_doc_structure", inspect_args),
        )
        blocks = inspect_content["blocks"]
        selected_block_ids = list(args.block_ids or _select_blocks(blocks, args.fields))
        print(f"      blocks={len(blocks)} selected={selected_block_ids}")

    extract_args = {
        "path": args.doc,
        "knowledge_path": "knowledge.md",
        "target_table": args.target_table,
        "fields": args.fields,
        "max_model_calls": args.extract_calls,
    }
    if selected_block_ids:
        extract_args["block_ids"] = selected_block_ids
    if explicit_line_ranges:
        extract_args["line_ranges"] = explicit_line_ranges
    print(f"[2/3] extract_structured_doc args={json.dumps(extract_args, ensure_ascii=False)}")
    extract_content = _require_ok(
        "extract_structured_doc",
        registry.execute(runtime_context, "extract_structured_doc", extract_args),
    )
    extraction = extract_content["extraction"]
    registered_table = extraction["registered_table"]
    print(
        "      rows={rows} facts={facts} table={table} non_null={non_null}".format(
            rows=extraction.get("row_count"),
            facts=extraction.get("fact_count"),
            table=registered_table,
            non_null=extraction.get("column_non_null_counts"),
        )
    )

    select_fields = [field for field in args.fields if field in extract_content["columns"]]
    if not select_fields:
        select_fields = list(extract_content["columns"])
    numeric_scale_expr = (
        "TRY_CAST(regexp_extract(CAST(totalfundnv AS VARCHAR), "
        "'([0-9]+(?:\\\\.[0-9]+)?)', 1) AS DOUBLE)"
    )
    query_where = f"WHERE {numeric_scale_expr} > 100" if "totalfundnv" in select_fields else ""
    query_order = "ORDER BY personalcode" if "personalcode" in select_fields else ""
    query_args = {
        "queries": [
            f"""
            SELECT {", ".join(select_fields)}
            FROM {registered_table}
            {query_where}
            {query_order}
            """
        ]
    }
    print(f"[3/3] execute_probe_query args={json.dumps(query_args, ensure_ascii=False)}")
    query_result = registry.execute(runtime_context, "execute_probe_query", query_args)
    query_content = query_result.content
    query_rows = []
    if query_result.ok:
        query_rows = query_content["results"][0].get("rows", [])
        print(f"      query_rows={len(query_rows)}")
    else:
        print(f"      query_failed={json.dumps(query_content, ensure_ascii=False)}")

    summary = {
        "task_id": args.task_id,
        "doc": args.doc,
        "fields": args.fields,
        "inspect_args": inspect_args,
        "extract_args": extract_args,
        "query_args": query_args,
        "blocks": blocks,
        "selected_block_ids": selected_block_ids,
        "columns": extract_content["columns"],
        "row_count": len(extract_content["rows"]),
        "rows_preview": extract_content["rows"][:20],
        "extraction": extraction,
        "query_ok": query_result.ok,
        "query_result": query_content["results"][0] if query_content.get("results") else query_content,
        "output_dir": str(output_dir),
    }
    _dump_json(output_dir / "manual_tool_test_summary.json", summary)
    print(f"[done] summary={output_dir / 'manual_tool_test_summary.json'}")


if __name__ == "__main__":
    main()
