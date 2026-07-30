from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import create_chat_model
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import StructuredDocToolConfig, ToolConfig, load_app_config
from data_agent_baseline.run.context_preprocessor import prepare_task_context
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools.registry import ToolRuntimeContext, create_default_tool_registry


FIELDS = ["personalcode", "totalfundnv"]
DOC_PATH = "doc/mf_fmscaleanalysisn.md"
TARGET_TABLE = "mf_fmscaleanalysisn"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _archive_id(text: str) -> str | None:
    match = re.search(r"档案\s*(\d+)", text)
    return match.group(1) if match else None


def _personal_code(text: str) -> str | None:
    matches = re.findall(r"(?:PersonalCode|个人识别码(?:为)?|识别编码被确认为)\s*[:：]?\s*(\d{6,})", text, flags=re.I)
    return matches[-1] if matches else None


def _total_fund_nv(text: str) -> float | None:
    if "总资产净值" not in text and "总资产规模" not in text and "总规模" not in text:
        return None
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*亿元", text)
    return float(matches[-1]) if matches else None


def _baseline_from_doc(doc_path: Path) -> dict[str, float]:
    by_archive: dict[str, dict[str, Any]] = {}
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        archive = _archive_id(line)
        if not archive:
            continue
        entry = by_archive.setdefault(archive, {})
        code = _personal_code(line)
        scale = _total_fund_nv(line)
        if code is not None:
            entry["personalcode"] = code
        if scale is not None:
            entry["totalfundnv"] = scale
    return {
        str(values["personalcode"]): float(values["totalfundnv"])
        for values in by_archive.values()
        if values.get("personalcode") and float(values.get("totalfundnv", 0.0)) > 100.0
    }


def _extract_numeric(value: Any) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else None


def _result_codes(rows: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        code = row.get("personalcode")
        scale = _extract_numeric(row.get("totalfundnv"))
        if code is not None and scale is not None and scale > 100.0:
            output[str(code)] = scale
    return output


def _latest_log_event(log_path: Path, event_name: str) -> dict[str, Any] | None:
    for event in reversed(_load_jsonl(log_path)):
        if event.get("event") == event_name:
            return event
    return None


def _run_one(
    *,
    config_path: Path,
    task_id: str,
    output_dir: Path,
    min_chunk_lines: int,
    max_chunk_lines: int,
    baseline_codes: dict[str, float],
) -> dict[str, Any]:
    config = load_app_config(config_path)
    structured_config = StructuredDocToolConfig(
        min_chunk_lines=min_chunk_lines,
        max_chunk_lines=max_chunk_lines,
        max_selected_lines_for_llm_extraction=config.tool.structured_doc.max_selected_lines_for_llm_extraction,
        default_max_model_calls=config.tool.structured_doc.default_max_model_calls,
        hard_max_model_calls=config.tool.structured_doc.hard_max_model_calls,
        inspect_doc_structure_max_model_calls=config.tool.structured_doc.inspect_doc_structure_max_model_calls,
    )
    tool_config = ToolConfig(
        max_output_tokens=config.tool.max_output_tokens,
        max_list_items=config.tool.max_list_items,
        structured_doc=structured_config,
    )
    dataset = DABenchPublicDataset(config.dataset.root_path)
    prepared = prepare_task_context(dataset.get_task(task_id), output_dir, video_config=config.video_preprocessing)
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
    registry = create_default_tool_registry(tool_config=tool_config)
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(
            source_root=task.context_dir,
            context_view=task.assets.context_view,
        ),
        budget=config.data_inspector.sample_budget,
        semantic_view_config=config.data_inspector.semantic_views,
        model=model,
        trace_dir=output_dir,
    )
    started = time.perf_counter()
    inspect_result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": DOC_PATH,
            "knowledge_path": "knowledge.md",
            "target_table": TARGET_TABLE,
            "fields": FIELDS,
        },
    )
    if not inspect_result.ok:
        return {
            "min_chunk_lines": min_chunk_lines,
            "max_chunk_lines": max_chunk_lines,
            "ok": False,
            "stage": "inspect",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": inspect_result.content,
        }
    extract_result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": DOC_PATH,
            "knowledge_path": "knowledge.md",
            "target_table": TARGET_TABLE,
            "fields": FIELDS,
        },
    )
    elapsed = round(time.perf_counter() - started, 3)
    if not extract_result.ok:
        return {
            "min_chunk_lines": min_chunk_lines,
            "max_chunk_lines": max_chunk_lines,
            "ok": False,
            "stage": "extract",
            "elapsed_seconds": elapsed,
            "error": extract_result.content,
        }
    rows = extract_result.content["rows"]
    codes = _result_codes(rows)
    missing = sorted(set(baseline_codes) - set(codes))
    extra = sorted(set(codes) - set(baseline_codes))
    value_mismatch = sorted(
        code
        for code, expected in baseline_codes.items()
        if code in codes and abs(codes[code] - expected) > 1e-6
    )
    extraction = extract_result.content["extraction"]
    log_path = output_dir / str(extraction.get("log_file") or "")
    chunk_plan = _latest_log_event(log_path, "chunk_plan")
    schema_done = _latest_log_event(log_path, "schema_done")
    merge_done = _latest_log_event(log_path, "merge_done")
    return {
        "min_chunk_lines": min_chunk_lines,
        "max_chunk_lines": max_chunk_lines,
        "ok": not missing and not extra and not value_mismatch,
        "stage": "done",
        "elapsed_seconds": elapsed,
        "extract_elapsed_seconds": (extraction.get("log_summary") or {}).get("elapsed_seconds"),
        "model_call_count": extraction.get("model_call_count"),
        "chunk_count": extraction.get("chunk_count"),
        "chunk_sizes": ((chunk_plan or {}).get("details") or {}).get("chunk_sizes"),
        "chunking_reason": ((chunk_plan or {}).get("details") or {}).get("chunking_reason"),
        "entity_key_fields": extraction.get("entity_key_fields"),
        "schema_entity_key_fields": ((schema_done or {}).get("details") or {}).get("entity_key_fields"),
        "row_count": extraction.get("row_count"),
        "column_non_null_counts": extraction.get("column_non_null_counts"),
        "field_conflict_count": extraction.get("field_conflict_count"),
        "merge_details": (merge_done or {}).get("details"),
        "result_count_gt100": len(codes),
        "baseline_count_gt100": len(baseline_codes),
        "missing_codes": missing,
        "extra_codes": extra,
        "value_mismatch_codes": value_mismatch,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selected.yaml")
    parser.add_argument("--task-id", default="task_3")
    parser.add_argument("--output-root", default="artifacts/runs/structured_doc_chunk_scan_task3")
    parser.add_argument("--mins", default="10,15,20,25,30,35,40,45,50,60,70,80")
    parser.add_argument("--maxes", default="30,40,50,60,70,80,90,100,120,150")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_dir = output_root / "_baseline"
    config = load_app_config(config_path)
    prepared = prepare_task_context(
        DABenchPublicDataset(config.dataset.root_path).get_task(args.task_id),
        baseline_dir,
        video_config=config.video_preprocessing,
    )
    baseline_codes = _baseline_from_doc(prepared.generated_context_dir / DOC_PATH)
    (output_root / "baseline_gt100.json").write_text(
        json.dumps(baseline_codes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    min_values = [int(value) for value in args.mins.split(",") if value.strip()]
    max_values = [int(value) for value in args.maxes.split(",") if value.strip()]
    for max_chunk_lines in max_values:
        for min_chunk_lines in min_values:
            if min_chunk_lines > max_chunk_lines:
                continue
            run_dir = output_root / f"min{min_chunk_lines}_max{max_chunk_lines}"
            print(f"[scan] min={min_chunk_lines} max={max_chunk_lines}", flush=True)
            result = _run_one(
                config_path=config_path,
                task_id=args.task_id,
                output_dir=run_dir,
                min_chunk_lines=min_chunk_lines,
                max_chunk_lines=max_chunk_lines,
                baseline_codes=baseline_codes,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
            (output_root / "scan_results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
