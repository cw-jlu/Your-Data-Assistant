"""Stub-adapter end-to-end tests for extract_prose_file.

合成 prose + 脚本化 LLM 回复 → extract_prose_file → 逐字节断言最终 CSV。
覆盖三个历史事故输入：verify 回复含竖线/换行/`\\1`、PK 非首列 schema、
`~` 值触发恒等式修复。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl import extractor
from agents.llm import ModelMessage, ModelResponse
from tests.helpers.scripted_adapters import ScriptedModelAdapter


class NoLLMAdapter:
    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        raise AssertionError("this scenario must not spend any LLM call")


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_e2e"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_e2e", difficulty="hard", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter: Any,
    compressed: str,
    entity_groups: dict[str, list[str]] | None,
    schema_columns: list[str],
    anchor_keys: list[str],
    schema_field_types: dict[str, str],
    prose_text: str,
) -> tuple[extractor.ETLResult | None, Path]:
    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")
    monkeypatch.setattr(
        extractor,
        "compress_prose",
        lambda *_args, **_kwargs: (compressed, entity_groups, None),
    )
    prose_path = task.context_dir / "records.md"
    prose_path.write_text(prose_text, encoding="utf-8")

    result = extractor.extract_prose_file(
        adapter,
        prose_path,
        task,
        prose_text,
        schema_columns=schema_columns,
        anchor_keys=anchor_keys,
        schema_field_types=schema_field_types,
    )
    csv_path = tmp_path / "etl" / task.task_id / "_etl" / "records.csv"
    return result, csv_path


def test_e2e_verify_reply_with_pipe_backref_and_newline_cannot_corrupt_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify 修正值含 `|`、`\\1` 与换行垃圾行——CSV 结构与相邻记录必须完好。

    同时覆盖 PK 非首列 schema（record_id 在每行第二个键位）。
    """
    # 单个 verify LLM 调用（只有实体 1 在 entity_groups 里）
    adapter = ScriptedModelAdapter(["name: has|pipe \\1\nscore: 999\njunk trailing line"])

    result, csv_path = _run(
        tmp_path,
        monkeypatch,
        adapter,
        compressed=(
            "name: Alice | record_id: 1 | score: 100\nname: Bob | record_id: 2 | score: 200"
        ),
        entity_groups={"1": ["Record 1 is named has|pipe \\1 and scores 999."]},
        schema_columns=["record_id", "name", "score"],
        anchor_keys=["record_id"],
        schema_field_types={"record_id": "scalar_id", "name": "string", "score": "number"},
        prose_text="Record 1 then record 2.",
    )

    assert result is not None
    assert csv_path.read_bytes() == (b"record_id,name,score\r\n1,has|pipe \\1,999\r\n2,Bob,200\r\n")
    assert not adapter._responses  # scripted verify reply was consumed


def test_e2e_approx_value_triggers_identity_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~` 模糊值经全管道后由挖掘出的加法恒等式重算，且 CSV 中无残留标记。"""
    exact_rows = [
        ("29", "2441732", "300000", "2141732"),
        ("30", "5469268", "785000", "4684268"),
        ("31", "3527568", "40000", "3487568"),
        ("48", "3487568", "20000", "3467568"),
        ("49", "2763268", "430000", "2333268"),
        ("51", "3013268", "100000", "2913268"),
    ]
    lines = [
        f"record_id: {rid} | sumbeforetran: {before} | "
        f"transfer_shares: {shares} | sumaftertran: {after}"
        for rid, before, shares, after in exact_rows
    ]
    # true before = 6723434 + 250000 = 6973434; 原文粗略表述为 ~6970000
    lines.append(
        "record_id: 1400 | sumbeforetran: ~6970000 | "
        "transfer_shares: 250000 | sumaftertran: 6723434"
    )

    result, csv_path = _run(
        tmp_path,
        monkeypatch,
        NoLLMAdapter(),
        compressed="\n".join(lines),
        entity_groups=None,  # 无 retry/verify —— 全程零 LLM 调用
        schema_columns=["record_id", "sumbeforetran", "transfer_shares", "sumaftertran"],
        anchor_keys=["record_id"],
        schema_field_types={
            "record_id": "integer_scalar local_record_id",
            "sumbeforetran": "number",
            "transfer_shares": "number",
            "sumaftertran": "number",
        },
        # 每个 pk 必须在原文中独立出现，否则 pre_merge 的 unanchored-pk 守卫会丢行
        prose_text="share transfer records: 29 30 31 48 49 51 1400",
    )

    assert result is not None
    content = csv_path.read_bytes().decode("utf-8")
    expected_rows = "\r\n".join(
        f"{rid},{before},{shares},{after}" for rid, before, shares, after in exact_rows
    )
    assert content == (
        "record_id,sumbeforetran,transfer_shares,sumaftertran\r\n"
        f"{expected_rows}\r\n"
        "1400,6973434,250000,6723434\r\n"
    )
    assert "~" not in content
