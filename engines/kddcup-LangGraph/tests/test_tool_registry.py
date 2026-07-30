from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import StructuredDocToolConfig, ToolConfig
from data_agent_baseline.config import DataInspectorSampleBudget
from data_agent_baseline.config import DataInspectorSemanticViewConfig
from data_agent_baseline.inspectors.semantic_catalog import (
    build_lightweight_catalog,
    build_semantic_catalog,
)
from data_agent_baseline.tools.doc_structure import (
    _candidate_blocks,
    _split_non_empty_lines,
    load_doc_structure,
)
from data_agent_baseline.tools.python_exec import TaskContextWorkspace
from data_agent_baseline.tools import registry as registry_module
from data_agent_baseline.tools.registry import (
    ToolExecutionResult,
    ToolRegistry,
    ToolRuntimeContext,
    create_default_tool_registry,
)
from data_agent_baseline.tools.structured_doc_extractor import (
    _build_extraction_plan,
    _bind_extraction_model,
    _chunking_config_for_cache,
    _chunk_lines,
    _merge_facts,
    _validate_facts,
)


def _field_value_specs(fields: list[str]) -> dict[str, dict[str, str | None]]:
    string_fields = {"personalcode", "innercode", "secuabbr"}
    return {
        field: {
            "value_type": "string" if field in string_fields else "number",
            "unit_source": "document_dominant" if field == "dailybenchgr" else "none",
            "normalization_rule": (
                "bare percentage points; 1.5% -> 1.5"
                if field == "dailybenchgr"
                else "bare JSON value without a unit suffix"
            ),
            "expected_source_units": ["%"] if field == "dailybenchgr" else None,
        }
        for field in fields
    }


class StructuredDocModel:
    def __init__(self) -> None:
        self.invoke_count = 0
        self.schema_request_count = 0

    def invoke(self, messages):  # noqa: ANN001
        self.invoke_count += 1
        payload = json.loads(messages[-1].content)
        if "candidate_blocks" in payload:
            blocks = [{"block_id": b["block_id"], "scope_id": "m", "scope_name": "m", "candidate_fields": ["personalcode", "totalfundnv", "qdiinv"], "confidence": 0.9, "evidence": "t"} for b in payload["candidate_blocks"]]
            return AIMessage(content=json.dumps({"blocks": blocks}, ensure_ascii=False))
        if "lines" not in payload:
            self.schema_request_count += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "target_fields": [
                            {"name": "personalcode", "description": "Fund manager identifier"},
                            {"name": "totalfundnv", "description": "Total fund net asset value"},
                            {"name": "qdiinv", "description": "QDII management scale"},
                        ],
                        "entity_key_fields": ["archive_id"],
                        "fallback_entity_key": "line_id",
                        "merge_grain": "one row per archive/fund manager entity",
                        "field_hints": {
                            "personalcode": "final confirmed PersonalCode",
                            "totalfundnv": "total fund net asset value",
                            "qdiinv": "QDII management scale",
                        },
                        "field_value_specs": _field_value_specs(payload["target_fields"]),
                    },
                    ensure_ascii=False,
                )
            )
        facts = []
        for line in payload["lines"]:
            text = line["text"]
            archive_match = re.search(r"档案\s*(\d+)", text)
            archive_id = None if archive_match is None else archive_match.group(1)
            code_matches = re.findall(r"\d{9}", text)
            final_code = code_matches[-1] if code_matches else None
            scale_match = re.search(
                r"(?:管理规模|总资产净值|资产总规模|资产总净值|totalfundnv)[^\d]*(\d+(?:\.\d+)?)",
                text,
                re.I,
            )
            qdii_match = re.search(r"QDII[^\d]*(\d+(?:\.\d+)?)", text, re.I)
            values = {}
            if final_code is not None:
                values["personalcode"] = final_code
            if scale_match is not None:
                values["totalfundnv"] = float(scale_match.group(1))
            if qdii_match is not None:
                values["qdiinv"] = float(qdii_match.group(1))
            if archive_id is not None or values:
                facts.append(
                    {
                        "line_id": line["line_id"],
                        "is_fact": True,
                        "entity_key": (
                            {"archive_id": archive_id}
                            if archive_id is not None else {"line_id": str(line["line_id"])}
                        ),
                        "values": values,
                        "evidence_fields": list(values),
                    }
                )
        return AIMessage(content=json.dumps({"facts": facts}, ensure_ascii=False))


class LineFallbackStructuredDocModel(StructuredDocModel):
    def invoke(self, messages):  # noqa: ANN001
        self.invoke_count += 1
        payload = json.loads(messages[-1].content)
        if "lines" not in payload:
            self.schema_request_count += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "target_fields": [
                            {"name": "personalcode", "description": "Fund manager identifier"},
                            {"name": "totalfundnv", "description": "Total fund net asset value"},
                        ],
                        "entity_key_fields": [],
                        "fallback_entity_key": "line_id",
                        "merge_grain": "one row per source line",
                        "field_hints": {},
                        "field_value_specs": _field_value_specs(payload["target_fields"]),
                    },
                    ensure_ascii=False,
                )
            )
        facts = []
        for line in payload["lines"]:
            text = line["text"]
            code_match = re.search(r"PersonalCode\s*(?:为)?\s*(\d{9})", text)
            scale_match = re.search(r"规模\s*(\d+(?:\.\d+)?)", text)
            values = {}
            if code_match is not None:
                values["personalcode"] = code_match.group(1)
            if scale_match is not None:
                values["totalfundnv"] = float(scale_match.group(1))
            if values:
                facts.append(
                    {
                        "line_id": line["line_id"],
                        "is_fact": True,
                        "entity_key": {},
                        "values": values,
                        "evidence_fields": list(values),
                    }
                )
        return AIMessage(content=json.dumps({"facts": facts}, ensure_ascii=False))


class EmptyFactsStructuredDocModel(StructuredDocModel):
    def invoke(self, messages):  # noqa: ANN001
        self.invoke_count += 1
        payload = json.loads(messages[-1].content)
        if "lines" not in payload:
            self.schema_request_count += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "target_fields": [
                            {"name": "personalcode", "description": "Fund manager identifier"},
                            {"name": "totalfundnv", "description": "Total fund net asset value"},
                        ],
                        "entity_key_fields": ["archive_id"],
                        "fallback_entity_key": "line_id",
                        "merge_grain": "one row per archive",
                        "field_hints": {},
                        "field_value_specs": _field_value_specs(payload["target_fields"]),
                    },
                    ensure_ascii=False,
                )
            )
        return AIMessage(content=json.dumps({"facts": []}, ensure_ascii=False))


class DistributedStructuredDocModel(StructuredDocModel):
    def invoke(self, messages):  # noqa: ANN001
        self.invoke_count += 1
        payload = json.loads(messages[-1].content)
        if "lines" not in payload:
            self.schema_request_count += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "target_fields": [
                            {"name": "personalcode", "description": "Fund manager identifier"},
                            {"name": "totalfundnv", "description": "Total fund net asset value"},
                            {"name": "qdiinv", "description": "QDII management scale"},
                        ],
                        "entity_key_fields": ["archive_id"],
                        "fallback_entity_key": "line_id",
                        "merge_grain": "one row per archive/fund manager entity",
                        "field_hints": {},
                        "field_value_specs": _field_value_specs(payload["target_fields"]),
                    },
                    ensure_ascii=False,
                )
            )
        facts = []
        for line in payload["lines"]:
            text = line["text"]
            archive_match = re.search(r"档案\s*(\d+)", text)
            if archive_match is None:
                continue
            values = {}
            code_matches = re.findall(r"\d{9}", text)
            if "PersonalCode" in text and code_matches:
                values["personalcode"] = code_matches[-1]
            qdii_match = re.search(r"QDII.*?总资产净值(?:为|高达)?\s*(\d+(?:\.\d+)?)", text)
            if qdii_match is not None:
                values["qdiinv"] = float(qdii_match.group(1))
            else:
                scale_match = re.search(r"总资产净值(?:约为|为|高达)?\s*(\d+(?:\.\d+)?)", text)
                if scale_match is not None:
                    values["totalfundnv"] = float(scale_match.group(1))
            if values:
                facts.append(
                    {
                        "line_id": line["line_id"],
                        "is_fact": True,
                        "entity_key": {"archive_id": archive_match.group(1)},
                        "values": values,
                        "evidence_fields": list(values),
                    }
                )
        return AIMessage(content=json.dumps({"facts": facts}, ensure_ascii=False))


class ConflictingSecuabbrStructuredDocModel(StructuredDocModel):
    def invoke(self, messages):  # noqa: ANN001
        self.invoke_count += 1
        payload = json.loads(messages[-1].content)
        if "candidate_blocks" in payload:
            blocks = [
                {
                    "block_id": block["block_id"],
                    "scope_id": "fund",
                    "scope_name": "fund",
                    "candidate_fields": ["innercode", "secuabbr", "dailybenchgr"],
                    "confidence": 0.9,
                    "evidence": "fund facts",
                }
                for block in payload["candidate_blocks"]
            ]
            return AIMessage(content=json.dumps({"blocks": blocks}, ensure_ascii=False))
        if "lines" not in payload:
            self.schema_request_count += 1
            return AIMessage(
                content=json.dumps(
                    {
                        "target_fields": [
                            {"name": "innercode", "description": "Internal fund identifier"},
                            {"name": "secuabbr", "description": "Security abbreviation"},
                            {"name": "dailybenchgr", "description": "Daily benchmark growth rate"},
                        ],
                        "entity_key_fields": ["record_id"],
                        "fallback_entity_key": "line_id",
                        "merge_grain": "one row per fund",
                        "field_hints": {},
                        "field_value_specs": _field_value_specs(payload["target_fields"]),
                    },
                    ensure_ascii=False,
                )
            )
        facts = []
        for line in payload["lines"]:
            text = line["text"]
            record_match = re.search(r"记录\s*(\d+)", text)
            if record_match is None:
                continue
            values = {}
            innercode_match = re.search(r"内部识别码为\s*(\d+)", text)
            if innercode_match is not None:
                values["innercode"] = innercode_match.group(1)
            abbr_match = re.search(r"(?:证券简称|官方简称)为“([^”]+)”", text)
            if abbr_match is not None:
                values["secuabbr"] = abbr_match.group(1)
            daily_match = re.search(r"当日.*?(\d+(?:\.\d+)?)%", text)
            if daily_match is not None:
                values["dailybenchgr"] = float(daily_match.group(1))
            if values:
                facts.append(
                    {
                        "line_id": line["line_id"],
                        "is_fact": True,
                        "entity_key": {"record_id": record_match.group(1)},
                        "values": values,
                        "evidence_fields": list(values),
                    }
                )
        return AIMessage(content=json.dumps({"facts": facts}, ensure_ascii=False))


class StructureAwareStructuredDocModel(DistributedStructuredDocModel):
    def __init__(self) -> None:
        super().__init__()
        self.structure_request_count = 0

    def invoke(self, messages):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        if "candidate_blocks" in payload:
            self.invoke_count += 1
            self.structure_request_count += 1
            blocks = []
            for block in payload["candidate_blocks"]:
                text = " ".join(
                    [str(block.get("boundary_text", ""))]
                    + [str(line.get("text", "")) for line in block.get("sample_lines", [])]
                )
                candidate_fields = []
                scope_id = "context"
                scope_name = "Context"
                if "PersonalCode" in text:
                    scope_id = "identity_baseline"
                    scope_name = "基金经理身份识别"
                    candidate_fields = ["personalcode"]
                elif "QDII" in text:
                    scope_id = "qdii_scale"
                    scope_name = "QDII基金管理规模"
                    candidate_fields = ["qdiinv"]
                elif "权益" in text:
                    scope_id = "equity_scale"
                    scope_name = "权益类基金管理规模"
                    candidate_fields = []
                elif "总规模" in text or "管理规模" in text:
                    scope_id = "overall_total_scale"
                    scope_name = "基金经理总体管理规模"
                    candidate_fields = ["totalfundnv"]
                blocks.append(
                    {
                        "block_id": block["block_id"],
                        "scope_id": scope_id,
                        "scope_name": scope_name,
                        "candidate_fields": candidate_fields,
                        "confidence": 0.9,
                        "evidence": text[:80],
                    }
                )
            return AIMessage(
                content=json.dumps(
                    {
                        "blocks": blocks,
                        "primary_key_field": "personalcode",
                        "primary_key_evidence": "PersonalCode is the stable identifier.",
                    },
                    ensure_ascii=False,
                )
            )
        return super().invoke(messages)


class RepairingDocStructureModel(StructureAwareStructuredDocModel):
    def __init__(self, *, failures_before_success: int = 1) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success

    def invoke(self, messages):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        if "candidate_blocks" in payload or "repair_instruction" in payload:
            if self.structure_request_count < self.failures_before_success:
                self.invoke_count += 1
                self.structure_request_count += 1
                return AIMessage(content=json.dumps({"bad": []}))
        return super().invoke(messages)


class PrimaryKeyCoverageRepairModel:
    def __init__(
        self,
        *,
        invalid_attempts: int = 0,
        primary_key_field: str | None = "personalcode",
    ) -> None:
        self.invalid_attempts = invalid_attempts
        self.primary_key_field = primary_key_field
        self.structure_request_count = 0
        self.payloads: list[dict[str, object]] = []

    def invoke(self, messages):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        assert "candidate_blocks" in payload
        self.payloads.append(payload)
        self.structure_request_count += 1
        invalid = self.structure_request_count <= self.invalid_attempts
        blocks = []
        for index, block in enumerate(payload["candidate_blocks"]):
            candidate_fields = ["totalfundnv"]
            if not invalid and index == 0 and self.primary_key_field is not None:
                candidate_fields.append("`PERSONALCODE`")
            blocks.append(
                {
                    "block_id": block["block_id"],
                    "scope_id": "structured",
                    "scope_name": "Structured facts",
                    "candidate_fields": candidate_fields,
                    "confidence": 0.9,
                    "evidence": "direct source evidence",
                }
            )
        return AIMessage(
            content=json.dumps(
                {
                    "blocks": blocks,
                    "primary_key_field": self.primary_key_field,
                    "primary_key_evidence": "Primary key is directly stated.",
                },
                ensure_ascii=False,
            )
        )


class RepairingStructuredDocModel(StructuredDocModel):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    def invoke(self, messages):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        if "lines" in payload and "repair_error" not in payload and not self.failed_once:
            self.failed_once = True
            self.invoke_count += 1
            return AIMessage(content=json.dumps({"bad": []}))
        return super().invoke(messages)


class RetryingChunkStructuredDocModel(StructuredDocModel):
    def __init__(self, *, failures_before_success: int) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success
        self.chunk_request_count = 0

    def invoke(self, messages):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        if "lines" in payload:
            self.invoke_count += 1
            self.chunk_request_count += 1
            if self.chunk_request_count <= self.failures_before_success:
                return AIMessage(content=json.dumps({"bad": []}))
        return super().invoke(messages)


class FailingSchemaModel:
    def invoke(self, messages):  # noqa: ANN001
        return AIMessage(content=json.dumps({"bad": []}))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _create_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_demo"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "users.csv").write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    (context_dir / "events.json").write_text(
        json.dumps({"records": [{"Id": 10, "UserId": 1}, {"Id": 11, "UserId": 2}]}),
        encoding="utf-8",
    )
    (context_dir / "notes.md").write_text("# Notes\nhello\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id="task_demo", difficulty="easy", question="Inspect schema."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_structured_doc_task(tmp_path: Path, *, conflict: bool = False) -> PublicTask:
    task_dir = tmp_path / ("task_structured_doc_conflict" if conflict else "task_structured_doc")
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True, exist_ok=True)
    (context_dir / "knowledge.md").write_text(
        "\n".join(
            [
                "# Knowledge",
                "### Fund Manager Scale Analysis (`mf_fmscaleanalysisn`)",
                "| Column | Semantic Definition |",
                "|--------|-------------------|",
                "| `personalcode` | Fund manager identifier |",
                "| `totalfundnv` | Total fund net asset value |",
                "| `qdiinv` | QDII management scale |",
            ]
        ),
        encoding="utf-8",
    )
    (context_dir / "doc" / "mf_fmscaleanalysisn.md").write_text(
        "\n".join(
            [
                "# Report",
                "档案 1 初始误录为 101000550，最终确认 PersonalCode 为 101000558，管理规模 120.5，QDII 30。",
                "档案 2 的 PersonalCode 101000559，管理规模 80。",
            ]
        ),
        encoding="utf-8",
    )
    if conflict:
        (context_dir / "mf_fmscaleanalysisn.csv").write_text(
            "personalcode,totalfundnv,qdiinv\nold,1,2\n",
            encoding="utf-8",
        )
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="Extract."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_large_structured_doc_task(tmp_path: Path, *, line_count: int) -> PublicTask:
    task_dir = tmp_path / f"task_structured_doc_large_{line_count}"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True, exist_ok=True)
    (context_dir / "knowledge.md").write_text(
        "\n".join(
            [
                "# Knowledge",
                "### Fund Manager Scale Analysis (`mf_fmscaleanalysisn`)",
                "| Column | Semantic Definition |",
                "|--------|-------------------|",
                "| `personalcode` | Fund manager identifier |",
                "| `totalfundnv` | Total fund net asset value |",
            ]
        ),
        encoding="utf-8",
    )
    rows = [
        f"档案 {index} 的 PersonalCode 101{index:06d}，管理规模 {100 + index}.0。"
        for index in range(1, line_count + 1)
    ]
    (context_dir / "doc" / "mf_fmscaleanalysisn.md").write_text(
        "\n".join(["# Report", *rows]),
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="Extract."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_distributed_structured_doc_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_structured_doc_distributed"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True, exist_ok=True)
    (context_dir / "knowledge.md").write_text(
        "\n".join(
            [
                "# Knowledge",
                "### Fund Manager Scale Analysis (`mf_fmscaleanalysisn`)",
                "| Column | Semantic Definition |",
                "|--------|-------------------|",
                "| `personalcode` | Fund manager identifier |",
                "| `totalfundnv` | Total fund net asset value |",
                "| `qdiinv` | QDII management scale |",
            ]
        ),
        encoding="utf-8",
    )
    (context_dir / "doc" / "mf_fmscaleanalysisn.md").write_text(
        "\n".join(
            [
                "# Report",
                "关于档案 36 的审查，初步记录为 101000550，最终确认 PersonalCode 101000558。",
                "档案 44 的记录显示，所涉基金经理内部识别编码被确认为 PersonalCode 101000559。",
                "在完成身份识别后，继续评估管理规模。",
                "关于档案 36 的韩海平，其管理的总资产净值为 182.488480 亿元。",
                "档案 44 的柳军，在QDII基金领域有所涉猎，总资产净值为 32.399156 亿元。",
                "档案 44 的柳军，其管理的总资产净值为 883.586211 亿元。",
            ]
        ),
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="Extract."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_conflicting_secuabbr_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_structured_doc_conflicting_secuabbr"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True, exist_ok=True)
    (context_dir / "knowledge.md").write_text(
        "\n".join(
            [
                "# Knowledge",
                "### Fund Benchmark Growth Rate (`mf_benchmarkgrowthrate`)",
                "| Column | Semantic Definition |",
                "|--------|-------------------|",
                "| `innercode` | Internal fund identifier |",
                "| `secuabbr` | Fund abbreviated name |",
                "| `dailybenchgr` | Daily benchmark growth rate |",
            ]
        ),
        encoding="utf-8",
    )
    (context_dir / "doc" / "mf_benchmarkgrowthrate.md").write_text(
        "\n".join(
            [
                "# Report",
                "关于记录 672 的投资组合，其内部识别码为 341520。其证券简称为“招商国证食品ETF”。",
                "关于记录 672 的投资组合，其官方简称为“招商国证食品饮料行业ETF”。",
                "关于记录 672 的投资组合，其业绩基准在当日录得了 5.0% 的增长。",
            ]
        ),
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="Extract."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_sectioned_structured_doc_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_structured_doc_sectioned"
    context_dir = task_dir / "context"
    (context_dir / "doc").mkdir(parents=True, exist_ok=True)
    (context_dir / "knowledge.md").write_text(
        "\n".join(
            [
                "# Knowledge",
                "### Fund Manager Scale Analysis (`mf_fmscaleanalysisn`)",
                "| Column | Semantic Definition |",
                "|--------|-------------------|",
                "| `personalcode` | Fund manager identifier |",
                "| `totalfundnv` | Total fund net asset value |",
                "| `qdiinv` | QDII management scale |",
            ]
        ),
        encoding="utf-8",
    )
    (context_dir / "doc" / "mf_fmscaleanalysisn.md").write_text(
        "\n".join(
            [
                "# Report",
                "关于档案 36 的审查，最终确认 PersonalCode 101000558。",
                "档案 44 的记录显示，PersonalCode 101000559。",
                "在确立身份标识后，接下来的分析将深入评估其各自管理的资产总规模。",
                "关于档案 36 的韩海平，其管理的总资产净值为 182.488480 亿元。",
                "档案 44 的柳军，其管理的总资产净值为 883.586211 亿元。",
                "在对基金经理的总体管理规模进行宏观评估后，本报告将进一步剖析权益类基金。",
                "档案 36 的韩海平，权益类基金总资产净值为 999.000000 亿元。",
                "在完成国内资产类别评估后，本报告将考察QDII基金领域的管理规模。",
                "档案 44 的柳军，在QDII基金领域有所涉猎，总资产净值为 32.399156 亿元。",
            ]
        ),
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(task_id=task_dir.name, difficulty="easy", question="Extract."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_extract_structured_doc_passes_priority_metadata_to_tool_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _create_structured_doc_task(tmp_path)
    registry = create_default_tool_registry()
    acquires: list[tuple[str, dict[str, object]]] = []
    releases: list[str] = []

    class FakeGate:
        def acquire(self, tool_name: str, **metadata: object) -> None:
            acquires.append((tool_name, dict(metadata)))

        def release(self, tool_name: str) -> None:
            releases.append(tool_name)

    def fake_estimate(**kwargs: object) -> SimpleNamespace:
        assert kwargs["path"] == "doc/mf_fmscaleanalysisn.md"
        assert kwargs["fields"] == ["personalcode", "totalfundnv"]
        return SimpleNamespace(
            priority_chunk_count=3,
            selected_line_count=87,
            priority_source="estimated_chunk_count",
            error=None,
        )

    def fake_extract(**kwargs: object) -> SimpleNamespace:
        assert kwargs["path"] == "doc/mf_fmscaleanalysisn.md"
        return SimpleNamespace(
            columns=["personalcode"],
            rows=[["101000558"]],
            metadata={"row_count": 1},
        )

    monkeypatch.setattr(
        registry_module,
        "estimate_structured_doc_chunk_count",
        fake_estimate,
    )
    monkeypatch.setattr(registry_module, "extract_structured_doc", fake_extract)

    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        tool_gate=FakeGate(),
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    assert acquires == [
        (
            "extract_structured_doc",
            {
                "priority_chunk_count": 3,
                "selected_line_count": 87,
                "priority_source": "estimated_chunk_count",
                "priority_error": None,
            },
        )
    ]
    assert releases == ["extract_structured_doc"]


def test_doc_structure_candidate_boundaries_use_generic_numeric_rules() -> None:
    text = "\n".join(
        [
            "# Report",
            "接下来，档案 44 的记录显示，PersonalCode 101001204 已确认。",
            "本报告在整体方法说明和后续审查节奏安排中，将于第2部分继续从宏观视角介绍数据组织方式。",
            "继续审查档案 275，其管理规模为 106.942808。",
            "没有任何数字的自然语言过渡段应该成为边界。",
            "最后，档案 268 的资产总规模为 165.004490。",
        ]
    )

    blocks = _candidate_blocks(_split_non_empty_lines(text))
    boundary_texts = [block["boundary_text"] for block in blocks]
    boundary_reasons = [block["boundary_reason"] for block in blocks]

    assert boundary_texts == [
        "# Report",
        "本报告在整体方法说明和后续审查节奏安排中，将于第2部分继续从宏观视角介绍数据组织方式。",
        "没有任何数字的自然语言过渡段应该成为边界。",
    ]
    assert boundary_reasons == [
        "markdown_heading",
        "sparse_numeric_transition",
        "no_digit_text",
    ]


def test_format_result_truncates_string_content() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_tokens=2, max_list_items=200),
    )

    payload = registry.format_result(
        "execute_python",
        ToolExecutionResult(ok=True, content={"output": "x" * 20}),
    )

    assert payload["ok"] is True
    output_str = str(payload["content"]["output"])
    assert output_str.startswith("x")
    assert output_str != "x" * 20  # truncated, not the full 20
    assert "内容已被截断" in output_str


def test_format_result_truncates_list_content() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_tokens=2000, max_list_items=2),
    )

    payload = registry.format_result(
        "execute_python",
        ToolExecutionResult(ok=True, content={"rows": [[1], [2], [3]]}),
    )

    assert payload["content"]["rows"][:2] == [[1], [2]]
    assert "内容已被截断" in str(payload["content"]["rows"][2])


def test_format_result_truncates_final_answer_payload_and_keeps_submission_content() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_tokens=2, max_list_items=1),
    )
    result = ToolExecutionResult(
        ok=True,
        content={"status": "submitted", "detail": "x" * 20},
        answer=AnswerTable(columns=["value"], rows=[["x" * 20]]),
    )

    payload = registry.format_result(
        "submit_tool_result",
        result,
    )

    assert payload["content"]["detail"] == "x" * 20
    assert payload["answer"]["columns"] == ["value"]
    assert payload["answer"]["rows"][0][0] != "x" * 20
    assert "内容已被截断" in payload["answer"]["rows"][0][0]
    assert result.answer is not None
    assert result.answer.rows == [["x" * 20]]


def test_format_result_truncates_submit_tool_result_answer_payload() -> None:
    registry = ToolRegistry(
        specs={},
        handlers={},
        tool_config=ToolConfig(max_output_tokens=2000, max_list_items=2),
    )

    payload = registry.format_result(
        "submit_tool_result",
        ToolExecutionResult(
            ok=True,
            content={"status": "submitted", "source_tool": "execute_probe_query", "row_count": 3},
            answer=AnswerTable(columns=["value"], rows=[[1], [2], [3]]),
        ),
    )

    assert payload["content"] == {
        "status": "submitted",
        "source_tool": "execute_probe_query",
        "row_count": 3,
    }
    assert payload["answer"]["columns"] == ["value"]
    assert payload["answer"]["rows"][:2] == [[1], [2]]
    assert "内容已被截断" in str(payload["answer"]["rows"][2])


def test_default_registry_exposes_probe_tools_and_hides_legacy_tools() -> None:
    registry = create_default_tool_registry()

    assert set(registry.handlers) == set(registry.specs)
    assert "execute_probe_query" in registry.specs
    assert "extract_structured_doc" in registry.specs
    assert "inspect_doc_structure" in registry.specs
    assert "get_column_distinct_values" in registry.specs
    assert "search_semantic_catalog" in registry.specs
    assert "get_table_profile" in registry.specs
    assert "get_field_profile" in registry.specs
    assert "get_table_relationships" in registry.specs
    assert "read_context_image" in registry.specs
    assert "execute_context_sql" not in registry.specs
    assert "inspect_all_schema" not in registry.specs
    assert "read_doc" in registry.specs
    assert "read_csv" not in registry.specs
    assert "read_json" not in registry.specs
    assert "answer" not in registry.specs
    assert "inspect_sqlite_schema" not in registry.specs
    assert "lookup_schema" not in registry.specs
    assert "execute_probe_query" in registry.handlers
    assert "inspect_doc_structure" in registry.handlers
    assert "get_column_distinct_values" in registry.handlers
    assert "search_semantic_catalog" in registry.handlers
    assert "get_table_profile" in registry.handlers
    assert "get_field_profile" in registry.handlers
    assert "get_table_relationships" in registry.handlers
    assert "read_context_image" in registry.handlers
    assert "inspect_all_schema" not in registry.handlers
    assert "inspect_sqlite_schema" not in registry.handlers
    assert "read_csv" not in registry.handlers
    assert "read_json" not in registry.handlers
    assert "answer" not in registry.handlers
    assert "lookup_schema" not in registry.handlers


def test_structured_doc_chunk_planner_uses_configured_line_bounds() -> None:
    config = StructuredDocToolConfig(
        min_chunk_lines=25,
        max_chunk_lines=40,
        max_selected_lines_for_llm_extraction=400,
        default_max_model_calls=20,
        hard_max_model_calls=20,
        inspect_doc_structure_max_model_calls=3,
    )

    def sizes_for(line_count: int, available_calls: int = 19) -> list[int]:
        lines = [{"line_id": index, "text": f"row {index}"} for index in range(1, line_count + 1)]
        plan = _chunk_lines(lines, available_calls, config=config)
        return [len(chunk) for chunk in plan.chunks]

    assert sizes_for(24) == [24]
    assert sizes_for(40) == [40]
    assert sizes_for(41) == [25, 16]
    assert sizes_for(50) == [25, 25]
    assert sizes_for(100) == [25, 25, 25, 25]
    assert sizes_for(243) == [27] * 9
    assert sizes_for(400) == [40] * 10


def test_structured_doc_binds_stable_sampling_profile_only_for_extraction() -> None:
    class BindableModel:
        def __init__(self) -> None:
            self.bound_kwargs: dict[str, object] | None = None

        def bind(self, **kwargs):  # noqa: ANN003
            self.bound_kwargs = kwargs
            return self

    model = BindableModel()
    bound_model = _bind_extraction_model(model, StructuredDocToolConfig())

    assert bound_model is model
    assert model.bound_kwargs == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "extra_body": {"repetition_penalty": 1.0},
    }


def test_structured_doc_cache_config_includes_sampling_profile() -> None:
    config = StructuredDocToolConfig()

    assert _chunking_config_for_cache(config)["llm"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }


def test_structured_doc_plan_requires_field_value_specs_for_every_target_field() -> None:
    class PlanningModel:
        def invoke(self, messages):  # noqa: ANN001
            payload = json.loads(messages[-1].content)
            assert "field_value_specs MUST contain exactly one object" in payload["instruction"]
            return AIMessage(
                content=json.dumps(
                    {
                        "entity_key_fields": ["archive_id"],
                        "fallback_entity_key": "line_id",
                        "merge_grain": "one row per archive",
                        "field_hints": {},
                        "field_value_specs": {
                            "personalcode": {
                                "value_type": "string",
                                "unit_source": "none",
                                "expected_source_units": [],
                                "normalization_rule": "exact identifier text",
                            },
                            "totalfundnv": {
                                "value_type": "number",
                                "unit_source": "knowledge",
                                "expected_source_units": ["亿元"],
                                "normalization_rule": "bare number in 亿元",
                            },
                            "dailybenchgr": {
                                "value_type": "number",
                                "unit_source": "document_dominant",
                                "expected_source_units": ["%"],
                                "normalization_rule": "bare percentage points; 1.5% -> 1.5",
                            },
                        },
                    },
                    ensure_ascii=False,
                )
            )

    plan, calls = _build_extraction_plan(
        PlanningModel(),
        target_table="fund_metrics",
        requested_fields=["personalcode", "totalfundnv", "dailybenchgr"],
        knowledge_text="totalfundnv is measured in 亿元",
        sample_blocks=[],
    )

    assert calls == 1
    assert plan.field_value_specs["personalcode"]["value_type"] == "string"
    assert plan.field_value_specs["totalfundnv"]["value_type"] == "number"
    assert plan.field_value_specs["totalfundnv"]["unit_source"] == "knowledge"
    assert plan.field_value_specs["totalfundnv"]["expected_source_units"] == ["亿元"]
    assert "亿元" in plan.field_value_specs["totalfundnv"]["normalization_rule"]
    assert plan.field_value_specs["dailybenchgr"]["normalization_rule"].endswith("1.5")


def test_structured_doc_value_contract_rejects_unit_suffixed_numeric_values() -> None:
    specs = {
        "personalcode": {
            "value_type": "string",
            "canonical_unit": None,
            "unit_source": "none",
            "normalization_rule": "exact identifier text",
        },
        "dailybenchgr": {
            "value_type": "number",
            "canonical_unit": "%",
            "unit_source": "document_dominant",
            "normalization_rule": "bare percentage points; 1.5% -> 1.5",
        },
    }
    valid = _validate_facts(
        {
            "facts": [
                {
                    "line_id": 1,
                    "entity_key": {"record_id": "1"},
                    "values": {"personalcode": "00123", "dailybenchgr": 1.5},
                }
            ]
        },
        fields=["personalcode", "dailybenchgr"],
        expected_line_ids={1},
        fallback_entity_key="line_id",
        entity_key_fields=["record_id"],
        field_value_specs=specs,
    )
    assert valid[0]["values"] == {"personalcode": "00123", "dailybenchgr": 1.5}

    with pytest.raises(ValueError, match="must be number in canonical unit '%'"):
        _validate_facts(
            {
                "facts": [
                    {
                        "line_id": 1,
                        "entity_key": {"record_id": "1"},
                        "values": {"dailybenchgr": "1.5%"},
                    }
                ]
            },
            fields=["personalcode", "dailybenchgr"],
            expected_line_ids={1},
            fallback_entity_key="line_id",
            entity_key_fields=["record_id"],
            field_value_specs=specs,
        )


def test_structured_doc_plan_rejects_missing_or_invalid_value_contract() -> None:
    class InvalidPlanningModel:
        def invoke(self, messages):  # noqa: ANN001
            return AIMessage(
                content=json.dumps(
                    {
                        "entity_key_fields": [],
                        "field_value_specs": {
                            "personalcode": {
                                "value_type": "number",
                                "canonical_unit": None,
                                "unit_source": "none",
                                "normalization_rule": "wrong semantic type",
                            }
                        },
                    }
                )
            )

    with pytest.raises(ValueError, match=r"missing=\['totalfundnv'\]"):
        _build_extraction_plan(
            InvalidPlanningModel(),
            target_table="fund_metrics",
            requested_fields=["personalcode", "totalfundnv"],
            knowledge_text="",
            sample_blocks=[],
        )


def test_structured_doc_chunk_planner_fails_when_budget_cannot_keep_max_size() -> None:
    config = StructuredDocToolConfig(max_chunk_lines=40)
    lines = [{"line_id": index, "text": f"row {index}"} for index in range(1, 82)]

    with pytest.raises(ValueError, match="required_chunks=3"):
        _chunk_lines(lines, 2, config=config)


def test_structured_doc_merge_keeps_first_conflicting_value() -> None:
    result = _merge_facts(
        [
            {
                "line_id": 1,
                "entity_key": {"record_id": "672"},
                "values": {"secuabbr": "招商国证食品ETF"},
            },
            {
                "line_id": 2,
                "entity_key": {"record_id": "672"},
                "values": {"secuabbr": "招商国证食品饮料行业ETF"},
            },
            {
                "line_id": 3,
                "entity_key": {"record_id": "672"},
                "values": {"dailybenchgr": 5.0},
            },
        ],
        columns=["secuabbr", "dailybenchgr"],
        entity_key_fields=["record_id"],
    )

    assert result.rows == [["招商国证食品ETF", 5.0]]
    assert result.column_non_null_counts == {"secuabbr": 1, "dailybenchgr": 1}
    assert result.field_conflicts == [
        {
            "entity_key": {"record_id": "672"},
            "field": "secuabbr",
            "old_value": "招商国证食品ETF",
            "new_value": "招商国证食品饮料行业ETF",
            "line_id": 2,
        }
    ]


def test_extract_structured_doc_adds_primary_key_from_doc_structure(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = StructureAwareStructuredDocModel()
    registry = create_default_tool_registry()
    trace_dir = tmp_path / "run_output" / "task_structured_doc_sectioned"
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
        trace_dir=trace_dir,
    )

    structure_result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["totalfundnv"],
        },
    )

    assert structure_result.ok is True
    structure = structure_result.content["structure"]
    assert structure["primary_key_field"] == "personalcode"

    extraction_result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["totalfundnv"],
            "max_model_calls": 4,
        },
    )

    assert extraction_result.ok is True
    assert extraction_result.content["columns"] == ["personalcode", "totalfundnv"]
    assert extraction_result.content["rows"] == [
        ["101000558", 182.488480],
        ["101000559", 883.586211],
    ]
    extraction = extraction_result.content["extraction"]
    assert extraction["requested_fields"] == ["totalfundnv"]
    assert extraction["effective_requested_fields"] == ["personalcode", "totalfundnv"]
    assert extraction["primary_key_field"] == "personalcode"
    assert extraction["field_value_specs"] == _field_value_specs(
        ["personalcode", "totalfundnv"]
    )
    log_events = _read_jsonl(
        trace_dir / "structured_doc" / "structured_doc_mf_fmscaleanalysisn.log.jsonl"
    )
    assert any(event["event"] == "primary_key_field_applied" for event in log_events)
    plan_event = next(event for event in log_events if event["event"] == "extraction_plan")
    assert plan_event["details"]["field_value_specs"] == extraction["field_value_specs"]
    workspace_root = runtime_context.python_workspace.path
    assert workspace_root is not None
    manifest = json.loads(
        (workspace_root / ".generated" / "structured_doc" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["tables"][0]["field_value_specs"] == extraction["field_value_specs"]

    cached_result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["totalfundnv"],
            "max_model_calls": 4,
        },
    )
    assert cached_result.ok is True
    assert cached_result.content["extraction"]["cache_hit"] is True
    assert cached_result.content["extraction"]["field_value_specs"] == extraction["field_value_specs"]


def test_extract_structured_doc_keeps_first_conflicting_field_value(tmp_path: Path) -> None:
    task = _create_conflicting_secuabbr_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=ConflictingSecuabbrStructuredDocModel(),
    )

    structure_result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_benchmarkgrowthrate.md",
            "target_table": "mf_benchmarkgrowthrate",
            "fields": ["innercode", "secuabbr", "dailybenchgr"],
        },
    )
    assert structure_result.ok is True

    extraction_result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_benchmarkgrowthrate.md",
            "target_table": "mf_benchmarkgrowthrate",
            "fields": ["innercode", "secuabbr", "dailybenchgr"],
            "max_model_calls": 4,
        },
    )

    assert extraction_result.ok is True
    assert extraction_result.content["columns"] == ["innercode", "secuabbr", "dailybenchgr"]
    assert extraction_result.content["rows"] == [["341520", "招商国证食品ETF", 5.0]]
    extraction = extraction_result.content["extraction"]
    assert extraction["field_conflict_count"] == 1


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_persists_and_registers_queryable_table(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    model = StructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    assert result.content["columns"] == ["personalcode", "totalfundnv", "qdiinv"]
    assert result.content["rows"] == [["101000558", 120.5, 30.0], ["101000559", 80.0, None]]
    extraction = result.content["extraction"]
    assert extraction["registered_table"] == "mf_fmscaleanalysisn"
    assert extraction["row_count"] == 2
    assert extraction["log_file"] is None
    log_summary = extraction["log_summary"]
    assert log_summary["events"]["schema_done"] == 1
    assert log_summary["events"]["extraction_plan"] == 1
    assert log_summary["events"]["input_size_check"] == 1
    assert log_summary["events"]["chunk_plan"] == 1
    assert log_summary["events"]["merge_done"] == 1
    assert log_summary["events"]["persist_done"] == 1
    assert log_summary["events"]["done"] == 1
    assert log_summary["schema_fields"] == ["personalcode", "totalfundnv", "qdiinv"]
    assert log_summary["merge_summary"]["column_non_null_counts"] == {
        "personalcode": 2,
        "totalfundnv": 2,
        "qdiinv": 1,
    }
    assert log_summary["log_file"] is None
    assert model.schema_request_count == 1
    assert model.invoke_count == 2

    workspace_root = runtime_context.python_workspace.path
    assert workspace_root is not None
    manifest = json.loads(
        (workspace_root / ".generated" / "structured_doc" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["tables"][0]["registered_table"] == "mf_fmscaleanalysisn"
    assert manifest["tables"][0]["log_file"] is None
    assert manifest["tables"][0]["plan_version"] == 3
    assert manifest["tables"][0]["chunking_version"] == 1
    assert manifest["tables"][0]["chunking_config"]["min_chunk_lines"] == 25
    assert manifest["tables"][0]["chunking_config"]["max_chunk_lines"] == 40
    assert manifest["tables"][0]["entity_key_fields"] == ["archive_id"]
    assert manifest["tables"][0]["fact_count"] == 2
    assert manifest["tables"][0]["merged_row_count"] == 2
    assert manifest["tables"][0]["column_non_null_counts"] == {
        "personalcode": 2,
        "totalfundnv": 2,
        "qdiinv": 1,
    }

    query_result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {
            "queries": [
                "SELECT personalcode FROM mf_fmscaleanalysisn "
                "WHERE totalfundnv > 100 ORDER BY personalcode"
            ]
        },
    )

    assert query_result.ok is True
    assert query_result.content["results"][0]["rows"] == [["101000558"]]


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_writes_log_to_trace_dir_when_available(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    trace_dir = tmp_path / "run_output" / "task_structured_doc"
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=StructuredDocModel(),
        trace_dir=trace_dir,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    log_file = "structured_doc/structured_doc_mf_fmscaleanalysisn.log.jsonl"
    assert result.content["extraction"]["log_file"] == log_file
    assert result.content["extraction"]["log_summary"]["log_file"] == log_file
    assert (trace_dir / log_file).exists()
    assert (trace_dir / "structured_doc" / "mf_fmscaleanalysisn.jsonl").exists()
    assert (trace_dir / "structured_doc" / "manifest.json").exists()
    log_events = _read_jsonl(trace_dir / log_file)
    assert [event["event"] for event in log_events] == [
        "start",
        "structure_selected",
        "cache_miss",
        "input_size_check",
        "schema_start",
        "schema_done",
        "extraction_plan",
        "chunk_plan",
        "chunk_start",
        "chunk_done",
        "merge_done",
        "persist_done",
        "done",
    ]
    chunk_done = next(event for event in log_events if event["event"] == "chunk_done")
    assert chunk_done["details"]["extracted_facts"] == [
        {
            "line_id": 2,
            "entity_key": {"archive_id": "1"},
            "values": {"personalcode": "101000558", "totalfundnv": 120.5, "qdiinv": 30.0},
            "evidence_fields": ["personalcode", "totalfundnv", "qdiinv"],
        },
        {
            "line_id": 3,
            "entity_key": {"archive_id": "2"},
            "values": {"personalcode": "101000559", "totalfundnv": 80.0},
            "evidence_fields": ["personalcode", "totalfundnv"],
        },
    ]
    merge_done = next(event for event in log_events if event["event"] == "merge_done")
    assert merge_done["details"]["merged_row_count"] == 2
    workspace_root = runtime_context.python_workspace.path
    assert workspace_root is not None
    manifest = json.loads(
        (workspace_root / ".generated" / "structured_doc" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["tables"][0]["log_file"] == log_file


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_merges_distributed_entity_facts(tmp_path: Path) -> None:
    task = _create_distributed_structured_doc_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=DistributedStructuredDocModel(),
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 7]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    assert result.content["columns"] == ["personalcode", "totalfundnv", "qdiinv"]
    assert result.content["rows"] == [
        ["101000558", 182.48848, None],
        ["101000559", 883.586211, 32.399156],
    ]
    extraction = result.content["extraction"]
    assert extraction["fact_count"] == 5
    assert extraction["merged_row_count"] == 2
    assert extraction["column_non_null_counts"] == {
        "personalcode": 2,
        "totalfundnv": 2,
        "qdiinv": 1,
    }

    query_result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {
            "queries": [
                "SELECT personalcode FROM mf_fmscaleanalysisn "
                "WHERE totalfundnv > 100 ORDER BY personalcode"
            ]
        },
    )

    assert query_result.ok is True
    assert query_result.content["results"][0]["rows"] == [["101000558"], ["101000559"]]


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_inspect_doc_structure_persists_blocks_and_extract_uses_block_ids(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = StructureAwareStructuredDocModel()
    registry = create_default_tool_registry()
    trace_dir = tmp_path / "run_output" / "task_structured_doc_sectioned"
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
        trace_dir=trace_dir,
    )

    structure_result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "qdiinv"],
        },
    )

    assert structure_result.ok is True
    blocks = structure_result.content["blocks"]
    assert [block["scope_id"] for block in blocks] == [
        "identity_baseline",
        "overall_total_scale",
        "equity_scale",
        "qdii_scale",
    ]
    assert [block["boundary_reason"] for block in blocks] == [
        "markdown_heading",
        "no_digit_text",
        "no_digit_text",
        "no_digit_text",
    ]
    assert "档案 44" not in {block["boundary_text"] for block in blocks}
    assert blocks[0]["data_start_line"] == 2
    assert blocks[0]["data_end_line"] == 3
    workspace_root = runtime_context.python_workspace.path
    assert workspace_root is not None
    assert (workspace_root / ".generated" / "doc_structure" / "mf_fmscaleanalysisn.json").exists()
    assert (trace_dir / "doc_structure" / "doc_structure_mf_fmscaleanalysisn.log.jsonl").exists()
    assert (trace_dir / "doc_structure" / "mf_fmscaleanalysisn.json").exists()

    extraction_result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "qdiinv"],
            "max_model_calls": 6,
        },
    )

    assert extraction_result.ok is True
    assert extraction_result.content["rows"] == [
        ["101000558", 182.48848, None],
        ["101000559", 883.586211, 32.399156],
    ]
    extraction = extraction_result.content["extraction"]
    assert extraction["selected_blocks"] == ["B001", "B002", "B004"]
    assert extraction["auto_selected_blocks"] is True
    assert extraction["auto_block_selection"]["field_to_blocks"] == {
        "personalcode": ["B001"],
        "totalfundnv": ["B002"],
        "qdiinv": ["B004"],
    }
    assert extraction["scope_filtered_fact_count"] == 0
    log_events = _read_jsonl(
        trace_dir / "structured_doc" / "structured_doc_mf_fmscaleanalysisn.log.jsonl"
    )
    assert any(event["event"] == "structure_selected" for event in log_events)
    assert any(event["event"] == "doc_structure_cache_loaded" for event in log_events)
    assert any(event["event"] == "auto_block_select_done" for event in log_events)
    assert not any(event["event"] == "scope_filtered_fact" for event in log_events)


def test_inspect_doc_structure_repairs_invalid_model_response(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = RepairingDocStructureModel(failures_before_success=1)
    registry = create_default_tool_registry()
    trace_dir = tmp_path / "run_output" / "task_structured_doc_sectioned"
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
        trace_dir=trace_dir,
    )

    result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "qdiinv"],
        },
    )

    assert result.ok is True
    assert len(result.content["blocks"]) == 4
    assert result.content["structure"]["model_call_count"] == 2
    assert model.structure_request_count == 2
    log_events = _read_jsonl(
        trace_dir / "doc_structure" / "doc_structure_mf_fmscaleanalysisn.log.jsonl"
    )
    assert [event["event"] for event in log_events if event["event"].startswith("classify")] == [
        "classify_start",
        "classify_attempt_start",
        "classify_attempt_failed",
        "classify_attempt_start",
        "classify_attempt_done",
        "classify_done",
    ]


def test_extract_structured_doc_retries_chunk_until_repair_succeeds(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = RetryingChunkStructuredDocModel(failures_before_success=2)
    registry = create_default_tool_registry()
    trace_dir = tmp_path / "run_output" / "task_chunk_repair_success"
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
        trace_dir=trace_dir,
    )

    assert registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {"path": "doc/mf_fmscaleanalysisn.md", "target_table": "mf_fmscaleanalysisn"},
    ).ok is True
    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "qdiinv"],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    summary = result.content["extraction"]["log_summary"]
    assert model.chunk_request_count == 3
    assert summary["model_call_count"] == 4
    assert summary["max_model_calls"] == 4
    assert summary["budget_exhausted"] is False
    assert summary["events"]["chunk_failed"] == 1
    assert summary["events"]["chunk_repair_start"] == 2
    assert summary["events"]["chunk_repair_failed"] == 1
    assert summary["events"]["chunk_repair_done"] == 1


def test_extract_structured_doc_exhausts_budget_on_current_failed_chunk(tmp_path: Path) -> None:
    task = _create_large_structured_doc_task(tmp_path, line_count=61)
    model = RetryingChunkStructuredDocModel(failures_before_success=99)
    registry = create_default_tool_registry()
    trace_dir = tmp_path / "run_output" / "task_chunk_repair_exhausted"
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
        trace_dir=trace_dir,
    )

    assert registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {"path": "doc/mf_fmscaleanalysisn.md", "target_table": "mf_fmscaleanalysisn"},
    ).ok is True
    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "max_model_calls": 4,
        },
    )

    assert result.ok is False
    assert "Model call budget exhausted while repairing chunk 1" in result.content["error"]
    summary = result.content["extraction"]["log_summary"]
    assert model.chunk_request_count == 3
    assert summary["model_call_count"] == 4
    assert summary["budget_exhausted"] is True
    assert summary["events"]["chunk_failed"] == 1
    assert summary["events"]["chunk_repair_failed"] == 2
    assert summary["events"]["failed"] == 1


def test_inspect_doc_structure_repairs_primary_key_without_candidate_coverage(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = PrimaryKeyCoverageRepairModel(invalid_attempts=1)
    registry = create_default_tool_registry()
    trace_dir = tmp_path / "run_output" / "task_primary_key_coverage"
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
        trace_dir=trace_dir,
    )

    result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
        },
    )

    assert result.ok is True
    assert result.content["structure"]["model_call_count"] == 2
    assert model.structure_request_count == 2
    assert "repair" in model.payloads[1]
    repair = model.payloads[1]["repair"]
    assert isinstance(repair, dict)
    assert "Primary key field 'personalcode'" in str(repair["previous_error"])
    assert "primary_key_field" in str(repair["repair_instruction"])
    assert any(
        field.casefold() == "personalcode"
        for block in result.content["blocks"]
        for field in block["candidate_fields"]
    )
    log_events = _read_jsonl(
        trace_dir / "doc_structure" / "doc_structure_mf_fmscaleanalysisn.log.jsonl"
    )
    assert any(
        event["event"] == "classify_attempt_failed"
        and "Primary key field 'personalcode'" in str(event["details"]["error"])
        for event in log_events
    )


def test_inspect_doc_structure_fails_after_primary_key_coverage_repair_budget(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = PrimaryKeyCoverageRepairModel(invalid_attempts=3)
    registry = create_default_tool_registry()
    workspace = TaskContextWorkspace(source_root=task.context_dir)
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=workspace,
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "max_model_calls": 3,
        },
    )

    assert result.ok is False
    assert "Primary key field 'personalcode'" in result.content["error"]
    assert model.structure_request_count == 3
    assert load_doc_structure(workspace.materialize(), "mf_fmscaleanalysisn") is None


def test_inspect_doc_structure_allows_null_primary_key_without_repair(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = PrimaryKeyCoverageRepairModel(primary_key_field=None)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["totalfundnv"],
        },
    )

    assert result.ok is True
    assert result.content["structure"]["primary_key_field"] is None
    assert model.structure_request_count == 1
    assert len(model.payloads) == 1


def test_inspect_doc_structure_stops_after_configured_repair_attempts(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = RepairingDocStructureModel(failures_before_success=3)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "qdiinv"],
            "max_model_calls": 2,
        },
    )

    assert result.ok is False
    assert "Document structure response must contain blocks list." in result.content["error"]
    assert model.structure_request_count == 2


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_requires_structure_cache_for_auto_block_selection(
    tmp_path: Path,
) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = StructureAwareStructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
        },
    )

    assert result.ok is False
    assert result.content["error_code"] == "missing_doc_structure"
    assert "Call inspect_doc_structure" in result.content["error"]
    assert result.content["extraction"]["log_summary"]["events"]["missing_doc_structure"] == 1
    assert model.invoke_count == 0
    assert model.schema_request_count == 0
    assert model.structure_request_count == 0


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_auto_block_selection_fails_for_missing_fields(
    tmp_path: Path,
) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = StructureAwareStructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    structure_result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "not_in_doc"],
        },
    )
    assert structure_result.ok is True

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "not_in_doc"],
        },
    )

    assert result.ok is False
    assert result.content["error_code"] == "missing_fields"
    assert result.content["missing_fields"] == ["not_in_doc"]
    assert result.content["field_to_blocks"]["personalcode"] == ["B001"]
    assert result.content["field_to_blocks"]["not_in_doc"] == []
    assert result.content["extraction"]["log_summary"]["events"]["doc_structure_cache_loaded"] == 1
    assert result.content["extraction"]["log_summary"]["events"]["auto_block_select_done"] == 1
    assert model.structure_request_count == 1
    assert model.schema_request_count == 0


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_explicit_block_ids_skip_auto_block_selection(
    tmp_path: Path,
) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    model = StructureAwareStructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    structure_result = registry.execute(
        runtime_context,
        "inspect_doc_structure",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv", "qdiinv"],
        },
    )
    assert structure_result.ok is True
    blocks = structure_result.content["blocks"]

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "block_ids": [blocks[0]["block_id"], blocks[1]["block_id"]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    extraction = result.content["extraction"]
    assert extraction["selected_blocks"] == ["B001", "B002"]
    assert extraction["auto_selected_blocks"] is False
    assert extraction["auto_block_selection"] is None


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_accepts_explicit_line_ranges(tmp_path: Path) -> None:
    task = _create_sectioned_structured_doc_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=StructureAwareStructuredDocModel(),
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "line_ranges": [[2, 6]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    assert result.content["rows"] == [
        ["101000558", 182.48848],
        ["101000559", 883.586211],
    ]
    assert result.content["extraction"]["selected_line_ranges"] == [(2, 6)]


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_line_id_fallback_still_handles_complete_rows(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=LineFallbackStructuredDocModel(),
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    assert result.content["columns"] == ["personalcode", "totalfundnv"]
    assert result.content["rows"] == [["101000558", 120.5], ["101000559", 80.0]]
    warnings = result.content["extraction"]["quality_warnings"]
    assert "No natural entity key was identified; line_id fallback was used." in warnings


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_uses_conflict_suffix_and_cache(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path, conflict=True)
    model = StructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )
    args = {
        "path": "doc/mf_fmscaleanalysisn.md",
        "target_table": "mf_fmscaleanalysisn",
        "line_ranges": [[2, 3]],
        "max_model_calls": 4,
    }

    first = registry.execute(runtime_context, "extract_structured_doc", args)
    second = registry.execute(runtime_context, "extract_structured_doc", args)

    assert first.ok is True
    assert second.ok is True
    assert first.content["extraction"]["registered_table"] == "mf_fmscaleanalysisn_extracted"
    assert second.content["extraction"]["cache_hit"] is True
    assert second.content["extraction"]["log_summary"]["events"]["cache_hit"] == 1
    assert model.schema_request_count == 1
    assert model.invoke_count == 2

    query_result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"queries": ["SELECT COUNT(*) FROM mf_fmscaleanalysisn_extracted"]},
    )

    assert query_result.ok is True
    assert query_result.content["results"][0]["rows"] == [[2]]


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_execute_python_query_and_direct_read_generated_structured_doc(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=StructuredDocModel(),
    )
    registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    result = registry.execute(
        runtime_context,
        "execute_python",
        {
            "code": (
                "import json\n"
                "from pathlib import Path\n"
                "queried = query('SELECT personalcode FROM mf_fmscaleanalysisn "
                "WHERE qdiinv IS NOT NULL')\n"
                "raw = [json.loads(x) for x in "
                "Path('.generated/structured_doc/mf_fmscaleanalysisn.jsonl')"
                ".read_text(encoding='utf-8').splitlines()]\n"
                "print(json.dumps({'columns': ['queried_count', 'raw_count'], "
                "'rows': [[len(queried['rows']), len(raw)]]}))\n"
            )
        },
    )

    assert result.ok is True
    payload = json.loads(result.content["output"])
    assert payload["rows"] == [[1, 2]]


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_logs_repair_success(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=RepairingStructuredDocModel(),
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is True
    log_summary = result.content["extraction"]["log_summary"]
    assert log_summary["events"]["chunk_failed"] == 1
    assert log_summary["events"]["chunk_repair_done"] == 1
    assert log_summary["repair_count"] == 1


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_failure_returns_log_summary(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    trace_dir = tmp_path / "run_output" / "task_structured_doc"
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=FailingSchemaModel(),
        trace_dir=trace_dir,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is False
    log_summary = result.content["extraction"]["log_summary"]
    assert log_summary["events"]["schema_failed"] == 1
    assert log_summary["events"]["failed"] == 1
    assert log_summary["log_file"] == "structured_doc/structured_doc_mf_fmscaleanalysisn.log.jsonl"
    assert (trace_dir / log_summary["log_file"]).exists()


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_empty_facts_fails_with_quality_log(tmp_path: Path) -> None:
    task = _create_structured_doc_task(tmp_path)
    trace_dir = tmp_path / "run_output" / "task_structured_doc"
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=EmptyFactsStructuredDocModel(),
        trace_dir=trace_dir,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "line_ranges": [[2, 3]],
            "max_model_calls": 4,
        },
    )

    assert result.ok is False
    log_summary = result.content["extraction"]["log_summary"]
    assert log_summary["events"]["merge_done"] == 1
    assert log_summary["events"]["quality_warning"] >= 1
    assert log_summary["events"]["failed"] == 1
    assert log_summary["merge_summary"]["merged_row_count"] == 0
    assert (trace_dir / log_summary["log_file"]).exists()


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_fails_before_model_when_selected_lines_too_large(
    tmp_path: Path,
) -> None:
    task = _create_large_structured_doc_task(tmp_path, line_count=401)
    model = StructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "line_ranges": [[2, 402]],
        },
    )

    assert result.ok is False
    assert "Selected document range is too large" in result.content["error"]
    log_summary = result.content["extraction"]["log_summary"]
    assert log_summary["events"]["input_size_check"] == 1
    assert log_summary["events"]["input_too_large"] == 1
    assert log_summary["events"]["failed"] == 1
    assert model.invoke_count == 0
    assert model.schema_request_count == 0


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_large_full_doc_can_use_small_line_range(
    tmp_path: Path,
) -> None:
    task = _create_large_structured_doc_task(tmp_path, line_count=401)
    model = StructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "line_ranges": [[2, 26]],
        },
    )

    assert result.ok is True
    assert result.content["extraction"]["input_line_count"] == 25
    assert result.content["extraction"]["chunk_count"] == 1
    assert result.content["extraction"]["row_count"] == 25


@pytest.mark.skip(reason="Requires update for new extract_structured_doc API")
def test_extract_structured_doc_fails_before_model_when_call_budget_too_small(
    tmp_path: Path,
) -> None:
    task = _create_large_structured_doc_task(tmp_path, line_count=82)
    model = StructuredDocModel()
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        model=model,
    )

    result = registry.execute(
        runtime_context,
        "extract_structured_doc",
        {
            "path": "doc/mf_fmscaleanalysisn.md",
            "target_table": "mf_fmscaleanalysisn",
            "fields": ["personalcode", "totalfundnv"],
            "line_ranges": [[2, 83]],
            "max_model_calls": 3,
        },
    )

    assert result.ok is False
    assert "cannot fit the configured chunk/model-call budget" in result.content["error"]
    assert model.invoke_count == 0
    assert model.schema_request_count == 0


# ---------------------------------------------------------------------------
# execute_probe_query / get_column_distinct_values
# ---------------------------------------------------------------------------


def test_default_registry_exposes_probe_tools() -> None:
    registry = create_default_tool_registry()

    assert "execute_probe_query" in registry.specs
    assert "execute_probe_query" in registry.handlers
    assert "get_column_distinct_values" in registry.specs
    assert "get_column_distinct_values" in registry.handlers


def test_semantic_catalog_tools_return_profiles_by_logical_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    table_profile = registry.execute(runtime_context, "get_table_profile", {"table": "users"})
    field_profile = registry.execute(
        runtime_context,
        "get_field_profile",
        {"table": "users", "column": "name"},
    )
    search_result = registry.execute(
        runtime_context,
        "search_semantic_catalog",
        {"query": "name", "scope": "fields", "limit": 5},
    )

    assert table_profile.ok is True
    assert table_profile.content["table"] == "users"
    assert any(field["name"] == "name" for field in table_profile.content["fields"])
    assert "asset_path" not in table_profile.content
    assert field_profile.ok is True
    assert field_profile.content["field"]["name"] == "name"
    assert field_profile.content["field"]["distinct_values"]
    assert search_result.ok is True
    assert any(
        match["table"] == "users" and match["column"] == "name"
        for match in search_result.content["matches"]
    )


def test_catalog_types_match_duckdb_runtime_schema(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_duckdb_types"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "lc_freefloat.csv").write_text(
        "id,SecuCode,ChangeDate,AFloats\n"
        "1,000021,2019-12-31 00:00:00,100.5\n"
        "2,600000,2020-01-01 00:00:00,200.5\n",
        encoding="utf-8",
    )
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (itemId INTEGER, code TEXT)")
        conn.execute("INSERT INTO items VALUES (1, 'A001')")

    task = PublicTask(
        record=TaskRecord(task_id="task_duckdb_types", difficulty="easy", question="Types."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(catalog_top_distinct_values=5),
    )
    csv_schema = next(schema for schema in catalog["schemas"] if schema["asset_path"] == "lc_freefloat.csv")
    csv_fields = {field["name"]: field for field in csv_schema["fields"]}
    assert csv_fields["ChangeDate"]["type"] == "TIMESTAMP"
    assert csv_fields["ChangeDate"]["source_inferred_type"] == "string"
    assert csv_fields["SecuCode"]["type"] == "VARCHAR"
    assert csv_fields["SecuCode"]["source_inferred_type"] == "integer"
    assert "min_value" not in csv_fields["SecuCode"]
    assert "max_value" not in csv_fields["SecuCode"]

    sqlite_schema = next(schema for schema in catalog["schemas"] if schema["asset_path"] == "sample.db")
    items_table = next(table for table in sqlite_schema["tables"] if table["name"] == "items")
    sqlite_fields = {field["name"]: field for field in items_table["fields"]}
    assert sqlite_fields["itemId"]["type"] == "BIGINT"
    assert sqlite_fields["itemId"]["source_declared_type"] == "INTEGER"
    assert sqlite_fields["code"]["type"] == "VARCHAR"
    assert sqlite_fields["code"]["source_declared_type"] == "TEXT"

    lightweight = build_lightweight_catalog(catalog)
    lightweight_table = next(
        surface for surface in lightweight["query_surfaces"] if surface["table"] == "lc_freefloat"
    )
    assert lightweight_table["kind"] == "original_table"
    assert "ChangeDate" in lightweight_table["key_columns"]
    assert "SecuCode" in lightweight_table["key_columns"]

    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )
    table_profile = registry.execute(
        runtime_context,
        "get_table_profile",
        {"table": "lc_freefloat"},
    )
    profile_types = {
        field["name"]: field["type"] for field in table_profile.content["fields"]
    }
    assert profile_types["ChangeDate"] == "TIMESTAMP"
    assert profile_types["SecuCode"] == "VARCHAR"

    runtime_types = registry.execute(
        runtime_context,
        "execute_probe_query",
        {
            "queries": [
                "SELECT typeof(ChangeDate) AS ChangeDate, typeof(SecuCode) AS SecuCode "
                "FROM lc_freefloat LIMIT 1"
            ],
            "limit": 1,
        },
    )
    assert runtime_types.ok is True
    assert runtime_types.content["results"][0]["rows"] == [["TIMESTAMP", "VARCHAR"]]


def test_semantic_view_tools_and_probe_query(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_semantic_view_tools"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "sales.csv").write_text(
        "CompanyCode,Amount\n1,10\n1,5\n2,20\n3,30\n",
        encoding="utf-8",
    )
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE lc_exgindustry (CompanyCode INTEGER, SecondIndustryName TEXT)"
        )
        conn.execute(
            "INSERT INTO lc_exgindustry VALUES "
            "(1, 'Industry A'), (2, 'Industry B'), (3, 'Industry C')"
        )
    task = PublicTask(
        record=TaskRecord(
            task_id="task_semantic_view_tools",
            difficulty="easy",
            question="sales by industry",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
        semantic_view_config=DataInspectorSemanticViewConfig(),
    )

    profile = registry.execute(
        runtime_context,
        "get_table_profile",
        {"table": "v_sales_enriched"},
    )
    search = registry.execute(
        runtime_context,
        "search_semantic_catalog",
        {"query": "SecondIndustryName", "scope": "fields", "limit": 10},
    )
    relationships = registry.execute(
        runtime_context,
        "get_table_relationships",
        {"table": "v_sales_enriched"},
    )
    query = registry.execute(
        runtime_context,
        "execute_probe_query",
        {
            "queries": [
                "SELECT SecondIndustryName, SUM(Amount) AS total_amount "
                "FROM v_sales_enriched GROUP BY SecondIndustryName ORDER BY SecondIndustryName"
            ],
            "limit": 10,
        },
    )

    assert profile.ok is True
    assert profile.content["kind"] == "derived_view"
    assert profile.content["is_original_table"] is False
    assert any(field["source_table"] == "lc_exgindustry" for field in profile.content["fields"])
    assert search.ok is True
    assert any(match["table"] == "v_sales_enriched" for match in search.content["matches"])
    assert relationships.ok is True
    assert relationships.content["embedded_joins"]
    assert query.ok is True
    assert query.content["results"][0]["rows"] == [
        ["Industry A", 15],
        ["Industry B", 20],
        ["Industry C", 30],
    ]


def test_get_table_profile_prefers_structured_table_over_same_stem_document(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "users.md").write_text(
        "# Users\nThis is a document, not a table.\n", encoding="utf-8"
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(runtime_context, "get_table_profile", {"table": "users"})

    assert result.ok is True
    assert result.content["table"] == "users"
    assert result.content["kind"] == "csv"
    assert any(field["name"] == "name" for field in result.content["fields"])


def test_table_profile_unknown_table_suggests_same_stem_document(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_doc_only"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "mf_investadvisoroutline.md").write_text("# Advisor Outline\n", encoding="utf-8")
    task = PublicTask(
        record=TaskRecord(
            task_id="task_doc_only", difficulty="easy", question="Which fund company?"
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context, "get_table_profile", {"table": "mf_investadvisoroutline"}
    )

    assert result.ok is False
    assert result.content["requested_table"] == "mf_investadvisoroutline"
    assert result.content["document_suggestions"][0]["path"] == "doc/mf_investadvisoroutline.md"
    assert result.content["document_suggestions"][0]["stem"] == "mf_investadvisoroutline"
    assert result.content["document_suggestions"][0]["recommended_tools"] == [
        "search_doc",
        "read_doc",
    ]
    assert "matched a document" in result.content["hint"]
    assert "ONLY authorized source" in result.content["hint"]
    assert "Do NOT query, profile, infer from, or substitute" in result.content["hint"]
    assert "next action MUST inspect that document" in result.content["hint"]
    assert "similarly named SQL table or view" in result.content["hint"]
    assert "inspect_doc_structure" in result.content["hint"]
    assert "extract_structured_doc" in result.content["hint"]
    assert "search_doc" in result.content["hint"]
    assert "read_doc" in result.content["hint"]


def test_field_and_relationship_tools_reuse_unknown_table_suggestions(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_doc_only_tools"
    context_dir = task_dir / "context"
    doc_dir = context_dir / "doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "mf_investadvisoroutline.md").write_text("# Advisor Outline\n", encoding="utf-8")
    task = PublicTask(
        record=TaskRecord(
            task_id="task_doc_only_tools", difficulty="easy", question="Which fund company?"
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    field_result = registry.execute(
        runtime_context,
        "get_field_profile",
        {"table": "mf_investadvisoroutline", "column": "EstablishmentDate"},
    )
    relationship_result = registry.execute(
        runtime_context,
        "get_table_relationships",
        {"table": "mf_investadvisoroutline"},
    )

    for result in (field_result, relationship_result):
        assert result.ok is False
    assert result.content["requested_table"] == "mf_investadvisoroutline"
    assert result.content["document_suggestions"][0]["path"] == "doc/mf_investadvisoroutline.md"
    assert "matched a document" in result.content["hint"]
    assert "ONLY authorized source" in result.content["hint"]


def test_read_context_image_attaches_model_only_image_part(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    image_path = task.context_dir / "frame.jpg"
    image_path.write_bytes(b"fake jpg bytes")
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(runtime_context, "read_context_image", {"path": "frame.jpg"})
    payload = registry.format_result("read_context_image", result)

    assert result.ok is True
    assert result.model_content_parts[0]["type"] == "text"
    assert result.model_content_parts[1]["type"] == "image_url"
    assert result.model_content_parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert payload["content"]["status"] == "image attached to next model request"
    assert "image_url" not in payload["content"]


def test_record_visual_evidence_requires_matching_inspected_image(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "frame.jpg").write_bytes(b"fake jpg bytes")
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    missing = registry.execute(
        runtime_context,
        "record_visual_evidence",
        {"path": "frame.jpg", "observations": "threshold 100"},
    )
    assert missing.ok is False
    assert "read_context_image" in missing.content["error"]

    image = registry.execute(runtime_context, "read_context_image", {"path": "frame.jpg"})
    assert image.ok is True
    recorded = registry.execute(
        runtime_context,
        "record_visual_evidence",
        {"path": "frame.jpg", "observations": "threshold: 100; year: 2019"},
    )
    assert recorded.ok is True
    assert recorded.content["path"] == "frame.jpg"
    assert recorded.content["observations"] == "threshold: 100; year: 2019"

    mismatched = registry.execute(
        runtime_context,
        "record_visual_evidence",
        {"path": "other.jpg", "observations": "not inspected"},
    )
    assert mismatched.ok is False


def test_execute_probe_query_csv_select(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM users", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["id", "name"]
    assert first["rows"] == [[1, "Alice"], [2, "Bob"]]
    assert first["row_count"] == 2
    assert first["truncated"] is False


def test_execute_probe_query_csv_where_filter(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_filter"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "events.csv").write_text(
        "id,type,amount\n1,A,100\n2,B,200\n3,A,300\n",
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_filter", difficulty="easy", question="Filter."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM events WHERE type = 'A'", "limit": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["row_count"] == 2
    assert [1, "A", 100] in first["rows"]
    assert [3, "A", 300] in first["rows"]


def test_execute_probe_query_json_records_select(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM events", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["Id", "UserId", "records"]
    assert [row[0] for row in first["rows"]] == [10, 11]
    assert [row[1] for row in first["rows"]] == [1, 2]
    assert first["row_count"] == 2


def test_execute_probe_query_json_records_using_asset_path(tmp_path: Path) -> None:
    """SQL can reference the asset path and it will be normalized."""
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT Id FROM events.json", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["Id"]
    assert [row[0] for row in first["rows"]] == [10, 11]


def test_execute_probe_query_with_group_by_and_aggregate(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_agg"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "orders.csv").write_text(
        "customer,amount\nAlice,100\nBob,200\nAlice,150\n",
        encoding="utf-8",
    )

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_agg", difficulty="easy", question="Aggregate."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["customer", "total"]
    rows = first["rows"]
    rows_as_tuples = sorted((row[0], float(row[1])) for row in rows)
    assert rows_as_tuples == [("Alice", 250.0), ("Bob", 200.0)]


def test_execute_probe_query_invalid_sql_rejected(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "INSERT INTO users VALUES (3, 'Eve')"},
    )

    assert result.ok is False
    assert "Only SELECT/WITH" in result.content["results"][0]["error"]


def test_execute_probe_query_ignores_leading_comments_and_empty_queries(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {
            "queries": [
                "-- Count user rows\nSELECT COUNT(*) AS user_count FROM users",
                "",
                "   -- Sample names\n   SELECT name FROM users ORDER BY id",
            ],
            "limit": 5,
        },
    )

    assert result.ok is True
    assert result.content["query_count"] == 2
    assert result.content["results"][0]["rows"] == [[2]]
    assert result.content["results"][1]["rows"] == [["Alice"], ["Bob"]]


def test_execute_probe_query_nonexistent_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM ghost"},
    )

    assert result.ok is False
    assert "error" in result.content["results"][0]


def test_execute_probe_query_limit_truncation(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_trunc"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{i},val{i}" for i in range(20))
    (context_dir / "big.csv").write_text(f"id,val\n{rows}\n", encoding="utf-8")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_trunc", difficulty="easy", question="Trunc."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM big", "limit": 3},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["row_count"] == 3
    assert first["truncated"] is True


def test_get_column_distinct_values_csv(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "get_column_distinct_values",
        {"table": "users", "column": "name", "top_n": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    assert result.content["table"] == "users"
    assert result.content["column"] == "name"
    values = result.content["values"]
    assert {"value": "Alice", "count": 1} in values
    assert {"value": "Bob", "count": 1} in values


def test_get_column_distinct_values_nonexistent_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "get_column_distinct_values",
        {"table": "ghost", "column": "x"},
    )

    assert result.ok is False
    assert "not found" in result.content.get("error", "")


def test_execute_probe_query_lazy_builds_catalog(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    assert runtime_context._catalog_cache is None
    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM users", "limit": 1},
    )
    assert result.ok is True
    assert runtime_context._catalog_cache is not None

    cache_after_first = runtime_context._catalog_cache
    result2 = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM events", "limit": 1},
    )
    assert result2.ok is True
    assert runtime_context._catalog_cache is cache_after_first


def test_execute_probe_query_sqlite(tmp_path: Path) -> None:
    import sqlite3

    task_dir = tmp_path / "task_probe_sqlite"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    db_path = context_dir / "data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
        conn.execute("INSERT INTO races VALUES (1, 'GP')")
        conn.execute("INSERT INTO races VALUES (2, 'WRC')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_probe_sqlite", difficulty="easy", question="Probe."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_probe_query",
        {"sql": "SELECT * FROM races", "limit": 5},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    first = result.content["results"][0]
    assert first["columns"] == ["raceId", "name"]
    assert first["rows"] == [[1, "GP"], [2, "WRC"]]


def test_execute_python_query_helper_reads_csv_logical_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_python",
        {
            "code": (
                "import json\n"
                "print(json.dumps(query('SELECT id, name FROM users ORDER BY id'), ensure_ascii=False))"
            ),
        },
    )

    assert result.ok is True
    payload = json.loads(result.content["output"])
    assert payload["columns"] == ["id", "name"]
    assert payload["rows"] == [[1, "Alice"], [2, "Bob"]]


def test_execute_python_query_helper_reads_json_records_logical_table(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_python",
        {
            "code": (
                "import json\n"
                "print(json.dumps(query('SELECT Id, UserId FROM events ORDER BY Id'), ensure_ascii=False))"
            ),
        },
    )

    assert result.ok is True
    payload = json.loads(result.content["output"])
    assert payload["columns"] == ["Id", "UserId"]
    assert payload["rows"] == [[10, 1], [11, 2]]


def test_execute_python_query_helper_reads_sqlite_logical_table(tmp_path: Path) -> None:
    import sqlite3

    task_dir = tmp_path / "task_python_sqlite"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    db_path = context_dir / "data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
        conn.execute("INSERT INTO races VALUES (1, 'GP')")
        conn.execute("INSERT INTO races VALUES (2, 'WRC')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_python_sqlite", difficulty="easy", question="Probe."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context,
        "execute_python",
        {
            "code": (
                "import json\n"
                "print(json.dumps(query('SELECT * FROM races ORDER BY raceId'), ensure_ascii=False))"
            ),
        },
    )

    assert result.ok is True
    payload = json.loads(result.content["output"])
    assert payload["columns"] == ["raceId", "name"]
    assert payload["rows"] == [[1, "GP"], [2, "WRC"]]


def test_get_column_distinct_values_sqlite(tmp_path: Path) -> None:
    import sqlite3

    task_dir = tmp_path / "task_dist_sqlite"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    db_path = context_dir / "data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER, category TEXT)")
        conn.execute("INSERT INTO items VALUES (1, 'A')")
        conn.execute("INSERT INTO items VALUES (2, 'A')")
        conn.execute("INSERT INTO items VALUES (3, 'B')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_dist_sqlite", difficulty="easy", question="Dist."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    result = registry.execute(
        runtime_context,
        "get_column_distinct_values",
        {"table": "items", "column": "category", "top_n": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    values = result.content["values"]
    assert len(values) == 2
    assert {"value": "A", "count": 2} in values
    assert {"value": "B", "count": 1} in values


# ---------------------------------------------------------------------------
# search_doc pagination
# ---------------------------------------------------------------------------


def _create_doc_task(tmp_path: Path, doc_name: str, lines: list[str]) -> PublicTask:
    task_dir = tmp_path / f"task_{doc_name}"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / f"{doc_name}.md").write_text("\n".join(lines), encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id=f"task_{doc_name}", difficulty="easy", question="Search."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_search_doc_default_pagination(tmp_path: Path) -> None:
    lines = ["line alpha"] * 5 + ["line beta"] * 3
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "search_doc",
        {"query": "alpha", "context_lines": 0},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 5
    assert content["page"] == 1
    assert content["page_size"] == 20
    assert content["total_pages"] == 1
    assert len(content["results"]) == 1
    assert len(content["results"][0]["matches"]) == 5


def test_search_doc_page2(tmp_path: Path) -> None:
    lines = []
    for i in range(25):
        lines.append(f"item_{i} alpha")
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "search_doc",
        {"query": "alpha", "context_lines": 0, "page": 2, "page_size": 10},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 25
    assert content["page"] == 2
    assert content["page_size"] == 10
    assert content["total_pages"] == 3
    assert len(content["results"]) == 1
    page_matches = content["results"][0]["matches"]
    assert len(page_matches) == 10
    assert page_matches[0]["line_number"] == 11
    assert page_matches[-1]["line_number"] == 20


def test_search_doc_page_out_of_range(tmp_path: Path) -> None:
    lines = ["alpha one", "beta two", "alpha three"]
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "search_doc",
        {"query": "alpha", "context_lines": 0, "page": 99, "page_size": 10},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 2
    assert content["page"] == 99
    assert content["page_size"] == 10
    assert content["total_pages"] == 1
    assert content["results"] == []


def test_search_doc_page_size_zero(tmp_path: Path) -> None:
    lines = ["alpha one", "alpha two", "alpha three", "delta four", "alpha five"]
    task = _create_doc_task(tmp_path, "doc", lines)
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=task.context_dir),
    )

    result = registry.execute(
        runtime_context,
        "search_doc",
        {"query": "alpha", "context_lines": 0, "page_size": 0},
    )

    assert result.ok is True
    content = result.content
    assert content["total_matches"] == 4
    assert content["page"] == 1
    assert content["total_pages"] == 1
    assert len(content["results"][0]["matches"]) == 4


def test_get_column_distinct_values_always_live_computation(tmp_path: Path) -> None:
    # 构造一个包含 10 个不同去重值的 CSV 任务
    task_dir = tmp_path / "task_always_live"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = ["name"] + [f"user_{i}" for i in range(10)]
    (context_dir / "users.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_always_live", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    # 手动触发并修改 catalog_cache，将其中缓存的 distinct_values 修改为仅有一项，模拟缓存数据量极少或缺失的场景
    from data_agent_baseline.inspectors.semantic_catalog import build_semantic_catalog

    runtime_context._catalog_cache = build_semantic_catalog(
        runtime_context.task,
        budget=runtime_context.budget,
    )
    for schema in runtime_context._catalog_cache.get("schemas", []):
        if schema.get("asset_path") == "users.csv":
            for field in schema.get("fields", []):
                if field.get("name") == "name":
                    field["distinct_values"] = [{"value": "mocked_val", "count": 1}]

    # 调用 get_column_distinct_values 并请求 10 个去重值
    result = registry.execute(
        runtime_context,
        "get_column_distinct_values",
        {"table": "users", "column": "name", "top_n": 10},
    )

    assert result.ok is True
    assert result.content["ok"] is True
    assert result.content["table"] == "users"
    assert result.content["column"] == "name"

    # 验证返回的是底层的实时计算结果（包含 user_0 到 user_9 且长度为 10），而非被篡改为 1 项的预计算缓存
    values = result.content["values"]
    assert len(values) == 10
    assert {"value": "user_0", "count": 1} in values
    assert {"value": "user_9", "count": 1} in values
    assert {"value": "mocked_val", "count": 1} not in values


def test_get_column_distinct_values_quoted_sqlite_table(tmp_path: Path) -> None:
    import sqlite3

    task_dir = tmp_path / "task_dist_quoted_sqlite"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    db_path = context_dir / "data.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE [items] (id INTEGER, category TEXT)")
        conn.execute("INSERT INTO [items] VALUES (1, 'A')")
        conn.execute("INSERT INTO [items] VALUES (2, 'A')")
        conn.execute("INSERT INTO [items] VALUES (3, 'B')")

    from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

    task = PublicTask(
        record=TaskRecord(task_id="task_dist_quoted_sqlite", difficulty="easy", question="Test."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    registry = create_default_tool_registry()
    runtime_context = ToolRuntimeContext(
        task=task,
        python_workspace=TaskContextWorkspace(source_root=context_dir),
    )

    # 1. 验证带双引号的表名 table='"items"'
    result = registry.execute(
        runtime_context,
        "get_column_distinct_values",
        {"table": '"items"', "column": "category", "top_n": 10},
    )
    assert result.ok is True
    assert result.content["ok"] is True
    values = result.content["values"]
    assert len(values) == 2
    assert {"value": "A", "count": 2} in values
    assert {"value": "B", "count": 1} in values

    # 2. 验证带前后空格的表名 table=' items '
    result_space = registry.execute(
        runtime_context,
        "get_column_distinct_values",
        {"table": " items ", "column": "category", "top_n": 10},
    )
    assert result_space.ok is True
    assert result_space.content["ok"] is True
    values_space = result_space.content["values"]
    assert len(values_space) == 2
    assert {"value": "A", "count": 2} in values_space
    assert {"value": "B", "count": 1} in values_space
