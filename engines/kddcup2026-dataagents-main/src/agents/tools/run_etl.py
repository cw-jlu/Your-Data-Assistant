"""run_etl tool: on-demand prose-to-CSV extraction for selected documents."""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.etl._constants import PROSE_EXTS
from agents.etl.extractor import run_etl_for_task
from agents.tools.context import resolve_context_path
from agents.tools.registry import FunctionTool, ToolExecutionResult
from agents.tools.schema_normalize import normalize_for_vllm

if TYPE_CHECKING:
    from agents.llm import ModelAdapter


class RunETLInput(BaseModel):
    """Main-agent ETL tool input."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(
        min_length=1,
        description=(
            "Relative paths to the Markdown/text/PDF documents selected for ETL. "
            "Use paths returned by explore.etl_sources, for example "
            "['doc/fund_notes.pdf', 'doc/company_profiles.md']. Do not include "
            "knowledge.md; it is read automatically as ETL schema guidance."
        ),
    )


RUN_ETL_SCHEMA = normalize_for_vllm(RunETLInput.model_json_schema())

_ModelProvider = Callable[[], "ModelAdapter"]


def _missing_model_provider() -> ModelAdapter:
    raise RuntimeError("run_etl is not configured with a model adapter.")


def _resolve_requested_prose_files(task: PublicTask, paths: list[str]) -> list[Path]:
    selected: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        relative_path = raw_path.strip()
        if not relative_path:
            raise ValueError("run_etl: paths must not contain empty strings")
        resolved = resolve_context_path(task, relative_path)
        suffix = resolved.suffix.lower()
        if resolved.name.lower() == "knowledge.md":
            raise ValueError(
                "run_etl: knowledge.md is schema guidance and must not be ETL-converted"
            )
        if suffix not in PROSE_EXTS:
            allowed = ", ".join(sorted(PROSE_EXTS))
            raise ValueError(
                f"run_etl: {relative_path!r} is not an ETL prose document "
                f"(expected one of: {allowed})"
            )
        canonical = resolved.resolve()
        if canonical in seen:
            continue
        seen.add(canonical)
        selected.append(resolved)
    return selected


def _link_or_copy(source: Path, dest: Path) -> None:
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copy2(source, dest)


def _expose_csv_in_context(task: PublicTask, csv_path: Path) -> str:
    """Expose one ETL CSV under the current context and return a relative path."""

    source = csv_path.resolve()
    csv_dir = task.context_dir / "csv"
    dest = csv_dir / source.name

    if dest.exists() or dest.is_symlink():
        with contextlib.suppress(OSError):
            if dest.resolve() == source:
                return dest.relative_to(task.context_dir).as_posix()
        dest = csv_dir / "_etl" / source.name

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        with contextlib.suppress(OSError):
            if dest.resolve() == source:
                return dest.relative_to(task.context_dir).as_posix()
        if dest.is_dir():
            raise ValueError(f"run_etl: cannot expose CSV over directory {dest}")
        dest.unlink()

    _link_or_copy(source, dest)
    return dest.relative_to(task.context_dir).as_posix()


def _etl_result_payload(task: PublicTask, result: Any) -> dict[str, Any]:
    csv_path = Path(cast(str, result.csv_path))
    return {
        "source_file": cast(str, result.source_file),
        "csv_path": _expose_csv_in_context(task, csv_path),
        "columns": list(cast(list[str], result.columns)),
        "row_count": int(cast(int, result.row_count)),
    }


def create_run_etl_tool_definition(
    model_provider: _ModelProvider | None = None,
) -> FunctionTool:
    """Build the on-demand ETL tool definition.

    ``model_provider`` is intentionally deferred because the default tool
    registry is needed to construct the native-tools model adapter, while this
    tool later needs that adapter to build the no-tools ETL text adapter.
    """

    provider = model_provider or _missing_model_provider

    def handler(task: PublicTask, args: Any) -> ToolExecutionResult:
        request = args if isinstance(args, RunETLInput) else RunETLInput.model_validate(args)
        selected = _resolve_requested_prose_files(task, request.paths)
        results = run_etl_for_task(task, provider(), selected_prose_files=selected)
        payloads = [_etl_result_payload(task, item) for item in results or []]
        produced = {item["source_file"] for item in payloads}
        skipped = [path.name for path in selected if path.name not in produced]
        status = "ok" if payloads else "no_output"
        return ToolExecutionResult(
            ok=True,
            content={
                "status": status,
                "requested_paths": request.paths,
                "csv_files": payloads,
                "skipped_sources": skipped,
                "message": (
                    "Read each csv_path via preview_file or execute_python."
                    if payloads
                    else "ETL produced no valid CSV; fall back to the raw documents."
                ),
            },
        )

    return FunctionTool(
        name="run_etl",
        description=(
            "Run on-demand ETL for selected Markdown/text/PDF documents after "
            "explore identifies prose sources needed for the question. Processes "
            "the requested documents in parallel, writes valid CSV artifacts, and "
            "returns csv_files with relative csv_path values that can be read via "
            "preview_file or execute_python. Pass only necessary document paths; "
            "do not call this for every document unless explore says they are all "
            "required. Example: run_etl({'paths': ['doc/source.pdf']})"
        ),
        json_schema=RUN_ETL_SCHEMA,
        handler=handler,
        input_model=RunETLInput,
    )


__all__ = ["RunETLInput", "create_run_etl_tool_definition"]
