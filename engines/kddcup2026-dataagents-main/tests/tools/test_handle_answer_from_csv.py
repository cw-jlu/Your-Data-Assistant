"""`_answer_tool.handler` 的 from_csv artifact 路径单元测试。

锁住设计契约：
- 列级 dtype 推断与 inline 路径在 prediction.csv 字面值上对齐（int 不被悄悄变 float）
- 失败一律以 `ValueError` 上抛（dispatcher 翻译成 `ToolErrorEvent`），让 ReAct 循环 retry
- ack content 给 trace 留 `from_csv.dtypes` + `head_preview` 便于离线对账

XOR 校验、空 columns 报错由 `tests/test_dispatcher.py` 端到端校验；本文件只测 handler 内部行为。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import pytest

from agents.benchmark.schema import (
    AnswerTable,
    PublicTask,
    TaskAssets,
    TaskRecord,
)
from agents.tools.answer import AnswerInput, answer as _answer_tool

_answer_mod = importlib.import_module("agents.tools.answer")


@pytest.fixture
def task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    # Production now anchors _answer/ under `/tmp/dabench/<task_id>/_answer/` so
    # the §3.4 read-only `/input` mount doesn't break execute_python. Repointing
    # the scratch root at tmp_path keeps existing assertions (`task.task_dir /
    # "_answer"`) byte-equivalent — `_answer_dir_for(task)` collapses to
    # `tmp_path / "task_1" / "_answer"` which is exactly `task.task_dir / "_answer"`.
    monkeypatch.setattr(_answer_mod, "_ANSWER_SCRATCH_ROOT", tmp_path)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ----- #1 happy path：mixed dtypes -----


def test_from_csv_happy_path_returns_typed_answer(task: PublicTask, tmp_path: Path) -> None:
    """3 行 (int / str / float)：cell 类型按列推断而非全部落地为 str。"""
    csv_path = _write(
        tmp_path / "_answer" / "answer.csv",
        "id,name,ratio\n1,alpha,0.5\n2,beta,1.25\n3,gamma,2.0\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.is_terminal is True
    assert isinstance(result.answer, AnswerTable)
    assert result.answer.columns == ["id", "name", "ratio"]
    assert result.answer.rows == [
        [1, "alpha", 0.5],
        [2, "beta", 1.25],
        [3, "gamma", 2.0],
    ]
    # 类型必须严格保留（不是字符串等价）
    assert isinstance(result.answer.rows[0][0], int)
    assert isinstance(result.answer.rows[0][2], float)


# ----- #2-#5 column dtype inference -----


def _assert_rows_strict(actual: list[list[object]], expected: list[list[object]]) -> None:
    """值相等且 cell 类型精确一致（`type is`，天然防 bool 冒充 int / int 落地成 float）。"""
    assert actual == expected
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for actual_cell, expected_cell in zip(actual_row, expected_row, strict=True):
            assert type(actual_cell) is type(expected_cell), (actual_cell, expected_cell)


@pytest.mark.parametrize(
    ("csv_text", "expected_rows"),
    [
        # 整数主键 ID 必须以 int 落地（与 inline 路径一致），不被 float 抢占。
        # 回归参照 `test_dispatch_answer_preserves_integer_cells`：
        # avoid `163109.0` 写进 prediction.csv。
        pytest.param("ID\n163109\n2803470\n", [[163109], [2803470]], id="integer-ids"),
        pytest.param("ratio\n1.5\n0.25\n", [[1.5], [0.25]], id="float-values"),
        # 空字符串单元格落地为 None（而不是 `""`），保持与 inline `null` 语义对齐：
        # 第一列因为含空值 -> _classify_dtype 看 non_empty=[2] 返回 'int'，
        # cell "" 被强转为 None；非空 cell 被强转为 int
        pytest.param(
            "id,sex,diag\n,F,SLE\n2,M,RA\n",
            [[None, "F", "SLE"], [2, "M", "RA"]],
            id="empty-cell-becomes-none",
        ),
        pytest.param("flag\nTrue\nFalse\nTrue\n", [[True], [False], [True]], id="bool-column"),
    ],
)
def test_from_csv_dtype_inference(
    task: PublicTask, tmp_path: Path, csv_text: str, expected_rows: list[list[object]]
) -> None:
    csv_path = _write(tmp_path / "_answer" / "data.csv", csv_text)
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.answer is not None
    _assert_rows_strict(result.answer.rows, expected_rows)


# ----- #6-#8, #10, #11, #14 error scenarios -----


@pytest.mark.parametrize(
    ("make_from_csv", "size_limit", "match", "path_in_message"),
    [
        # error message 包含具体路径：模型才能 retry execute_python 重写文件
        pytest.param(
            lambda tmp_path: str(tmp_path / "_answer" / "ghost.csv"),
            None,
            r"from_csv file not found",
            True,
            id="missing-file",
        ),
        pytest.param(
            lambda tmp_path: "_answer/answer.csv",
            None,
            r"from_csv must be an absolute path",
            False,
            id="relative-path",
        ),
        pytest.param(
            lambda tmp_path: str(_write(tmp_path / "_answer" / "empty.csv", "")),
            None,
            r"from_csv is empty",
            False,
            id="zero-byte",
        ),
        pytest.param(
            lambda tmp_path: str(
                _write(tmp_path / "_answer" / "ragged.csv", "id,name,extra\n1,alpha\n")
            ),
            None,
            r"from_csv rows must have the same number of cells as the header row",
            False,
            id="row-width-mismatch",
        ),
        # 5MB 上限：不真造 5MB 文件，把阈值临时调小到 64 字节再触发
        pytest.param(
            lambda tmp_path: str(
                _write(
                    tmp_path / "_answer" / "big.csv",
                    "id\n" + "\n".join(str(n) for n in range(200)) + "\n",
                )
            ),
            64,
            r"from_csv exceeds 5MB",
            False,
            id="exceeds-size-limit",
        ),
        # \xff\xfe 是 UTF-16 BOM；以 UTF-8 解码必然失败。
        # 错误消息要可让模型自己决定要不要换 encoding。
        pytest.param(
            lambda tmp_path: str(
                _write_bytes(tmp_path / "_answer" / "broken.csv", b"\xff\xfeID\n1\n")
            ),
            None,
            r"from_csv is not valid UTF-8",
            False,
            id="invalid-utf8",
        ),
    ],
)
def test_from_csv_error_scenarios(
    task: PublicTask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_from_csv: Callable[[Path], str],
    size_limit: int | None,
    match: str,
    path_in_message: bool,
) -> None:
    """失败一律以 ValueError 上抛（dispatcher 翻译成 ToolErrorEvent），让 ReAct 循环 retry。"""
    if size_limit is not None:
        monkeypatch.setattr(_answer_mod, "_FROM_CSV_SIZE_LIMIT_BYTES", size_limit)
    from_csv = make_from_csv(tmp_path)
    if size_limit is not None:
        assert Path(from_csv).stat().st_size > size_limit  # sanity
    args = AnswerInput.model_validate({"from_csv": from_csv})

    with pytest.raises(ValueError, match=match) as exc_info:
        _answer_tool.handler(task, args)

    if path_in_message:
        assert from_csv in str(exc_info.value)


# ----- #9 header only, no data -----


def test_from_csv_header_only_returns_empty_answer(task: PublicTask, tmp_path: Path) -> None:
    """仅 header 无数据行：合法 → AnswerTable(header, [])。

    "空 answer 是合法的"——dispatcher 没有 row count 下限，符合 inline 路径
    `answer({"columns": ["x"], "rows": []})` 同等语义。
    """
    csv_path = _write(tmp_path / "_answer" / "head.csv", "id,name\n")
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.is_terminal is True
    assert result.answer is not None
    assert result.answer.columns == ["id", "name"]
    assert result.answer.rows == []


# ----- #16 ack content shape -----


def test_from_csv_ack_content_includes_dtypes_and_preview(task: PublicTask, tmp_path: Path) -> None:
    """ack `content.from_csv.{dtypes, head_preview}`：trace 离线对账依赖该结构。"""
    csv_path = _write(
        tmp_path / "_answer" / "preview.csv",
        "id,name,ratio\n" + "\n".join(f"{n},name_{n},{n / 4}" for n in range(1, 11)) + "\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.content["status"] == "submitted"
    assert result.content["column_count"] == 3
    assert result.content["row_count"] == 10

    from_csv_meta = result.content["from_csv"]
    assert from_csv_meta["path"] == str(csv_path)
    assert from_csv_meta["dtypes"] == ["int", "str", "float"]
    # head_preview 永远 ≤ 5 行
    assert len(from_csv_meta["head_preview"]) == 5
    assert from_csv_meta["head_preview"][0] == [1, "name_1", 0.25]


# ----- 补充：成功提交后 artifact 自动清理（避免污染输入树） -----


def test_from_csv_unlinks_artifact_after_successful_submission(
    task: PublicTask, tmp_path: Path
) -> None:
    """模型提交成功后，from_csv 指向的 CSV 与整个 `_answer/` 目录都必须被删除。

    输入树污染回归：曾出现 `data/public/input/task_*/_answer/answer.csv` 残留，
    造成离线分析时把它误读成"项目自带参考答案"；草稿/日志同理。失败路径不在本
    测试覆盖范围，那里保留文件让 ReAct 循环 retry 时 to_csv 直接覆盖更稳。
    """
    answer_dir = task.task_dir / "_answer"
    csv_path = _write(answer_dir / "answer.csv", "id,name\n1,alpha\n2,beta\n")
    # 草稿文件：模型可能把中间表/日志也丢在 DABENCH_ANSWER_DIR 下
    draft_path = _write(answer_dir / "draft.csv", "tmp\n1\n")
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert not csv_path.exists()
    assert not draft_path.exists()
    assert not answer_dir.exists()


def test_from_csv_external_path_still_clears_answer_dir(task: PublicTask, tmp_path: Path) -> None:
    """from_csv 指向 `_answer/` 之外（如 `/tmp/...`）也要清掉 `_answer/` 目录。

    场景：模型把 artifact 写到非约定路径（绕开 DABENCH_ANSWER_DIR），但更早的
    execute_python 已经在 `_answer/` 留下文件；rmtree 兜底带走，避免下次离线
    分析读到错位的 stale answer。
    """
    answer_dir = task.task_dir / "_answer"
    stale = _write(answer_dir / "answer.csv", "id\n999\n")
    external = _write(tmp_path / "elsewhere" / "real.csv", "id\n1\n2\n")
    args = AnswerInput.model_validate({"from_csv": str(external)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert not external.exists()
    assert not stale.exists()
    assert not answer_dir.exists()


def test_inline_submission_with_existing_artifact_raises(task: PublicTask, tmp_path: Path) -> None:
    """残留 `_answer/answer.csv` + inline 提交：handler 拒绝并指回 from_csv。

    回归参照 run 20260501-010/task_8：execute_python 已经把 140 行写到
    `_answer/answer.csv`，但 `answer` tool_call 的 JSON 参数里只 inline 了 head(5)
    的 5 行（attention drift 把更早一轮 head() 预览当成了完整表）。inline
    硬闸（>10 行 / >50 cells）抓不住此类小体量截断；提交时间点检测到 artifact
    残留就直接拒绝，让 ReAct 循环把错误回灌给模型自行改用 from_csv。

    artifact 文件必须保留在原位——下一轮模型才能用同一路径走 from_csv 重交。
    """
    artifact = task.task_dir / "_answer" / "answer.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("id\n999\n", encoding="utf-8")

    args = AnswerInput.model_validate({"columns": ["id"], "rows": [[1], [2]]})

    with pytest.raises(ValueError, match=r"answer artifact already exists") as exc_info:
        _answer_tool.handler(task, args)

    # 错误消息要把绝对路径与 from_csv 修复方向都贴上，模型下一轮才能直接照搬
    msg = str(exc_info.value)
    assert str(artifact) in msg
    assert "from_csv" in msg
    # 关键：保留 artifact 与 `_answer/` 目录，模型 retry 时还能指它
    assert artifact.exists()
    assert artifact.parent.exists()


def test_artifact_exists_beats_inline_threshold_when_both_fire(
    task: PublicTask, tmp_path: Path
) -> None:
    """两个拒绝条件同时成立时，handler 必须先抛 artifact-exists（含具体路径），
    而不是把模型导回去重写一遍它早已写好的 CSV。

    回归参照 run 20260614-009/task_31：execute_python 已经把表写到
    `_answer/answer.csv`，模型却又发了 12 行 inline。旧顺序下 Pydantic 先按"阈值"
    拒了 inline，错误信息是"write the table to ... answer.csv from execute_python and
    resubmit with from_csv"——模型读完只会再算一遍同一份表，从未反应过来应改 from_csv。
    新顺序让 handler 的 artifact-exists 错误胜出，里头直接给出可照搬的
    `answer({"from_csv": "<artifact path>"})`。
    """
    artifact = task.task_dir / "_answer" / "answer.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("id\n" + "\n".join(str(n) for n in range(20)) + "\n", encoding="utf-8")

    # 12 行 inline → 阈值条件 (n_rows > 10) 也会成立。
    args = AnswerInput.model_validate({"columns": ["id"], "rows": [[n] for n in range(12)]})

    with pytest.raises(ValueError) as exc_info:
        _answer_tool.handler(task, args)

    msg = str(exc_info.value)
    assert "answer artifact already exists" in msg
    assert str(artifact) in msg
    # threshold 文案绝不能在 artifact-exists 路径上出现，否则模型会把它当首要指令
    assert "inline rows payload exceeds" not in msg
    # artifact 保留：下一轮模型才能照搬 from_csv 路径
    assert artifact.exists()


def test_inline_threshold_check_in_handler_when_no_artifact(
    task: PublicTask, tmp_path: Path
) -> None:
    """没有 artifact 时，handler 内的阈值检查仍然兜底（错误信息与历史 Pydantic 版一致）。

    阈值检查从 Pydantic 搬到 handler 后必须有 happy path 之外的兜底覆盖，
    防止后续重构把这条规则误删。
    """
    answer_dir = task.task_dir / "_answer"
    if answer_dir.exists():
        for child in answer_dir.iterdir():
            child.unlink()

    args = AnswerInput.model_validate({"columns": ["id"], "rows": [[n] for n in range(11)]})

    with pytest.raises(
        ValueError,
        match=(
            r"^inline rows payload exceeds artifact-handoff threshold "
            r"\(got 11 rows, 11 cells; inline is allowed only for <= 10 rows AND "
            r"<= 50 cells; if either limit is exceeded, use from_csv\)"
        ),
    ):
        _answer_tool.handler(task, args)


def test_fallback_builder_rejects_inline_over_threshold(task: PublicTask) -> None:
    """`build_answer_table_without_side_effects` 必须与 handler 同源拒掉超阈值 inline。

    旧设计里 Pydantic 把超阈值挡在最外层，所以 fallback 函数不用再判一次；
    阈值检查迁到 handler 之后，如果 fallback 不同步，verifier 拒答路径会把
    超阈值 inline 当合法答案晋级——破坏阈值规则的不变量。
    """
    from agents.tools.answer import build_answer_table_without_side_effects

    arguments = {"columns": ["id"], "rows": [[n] for n in range(11)]}
    assert build_answer_table_without_side_effects(task, arguments) is None


def test_fallback_builder_accepts_inline_at_cell_threshold(task: PublicTask) -> None:
    """fallback builder 与 handler 同源：正好 50 cells 仍是合法 inline。"""
    from agents.tools.answer import build_answer_table_without_side_effects

    arguments = {
        "columns": [f"c{i}" for i in range(10)],
        "rows": [[row] * 10 for row in range(5)],
    }

    answer = build_answer_table_without_side_effects(task, arguments)

    assert answer is not None
    assert len(answer.columns) == 10
    assert len(answer.rows) == 5


def test_from_csv_normalizes_fullwidth_comma_in_header(task: PublicTask, tmp_path: Path) -> None:
    """Header 含全角逗号 `，` → 归一化为半角后正常解析，列名保持原样。

    回归参照 run 20260614-011/task_39：模型 header 行用全角逗号、数据行用半角。
    """
    csv_path = _write(
        tmp_path / "_answer" / "fullwidth.csv",
        "fund_name，value\n中融煤炭,48.706\n富国煤炭A,48.706\n平安新能源汽车ETF,42.881\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["fund_name", "value"]
    assert result.answer.rows == [
        ["中融煤炭", 48.706],
        ["富国煤炭A", 48.706],
        ["平安新能源汽车ETF", 42.881],
    ]


def test_from_csv_preserves_unquoted_fullwidth_comma_cells(
    task: PublicTask, tmp_path: Path
) -> None:
    """未加 quote 的数据单元格可合法包含全角逗号，不能把它当分隔符改写。"""
    csv_path = _write(
        tmp_path / "_answer" / "fullwidth_data.csv",
        "text\n你好，世界\n再见，朋友\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["text"]
    assert result.answer.rows == [["你好，世界"], ["再见，朋友"]]


def test_from_csv_preserves_other_fullwidth_punctuation(task: PublicTask, tmp_path: Path) -> None:
    """本救援只处理 header 全角逗号，不碰数据里的其他全角标点。"""
    csv_path = _write(
        tmp_path / "_answer" / "fullwidth_punctuation.csv",
        "text\n你好；世界\nA｜B\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["text"]
    assert result.answer.rows == [["你好；世界"], ["A｜B"]]


def test_from_csv_preserves_fullwidth_inside_quoted_cells(task: PublicTask, tmp_path: Path) -> None:
    """Quoted cell 内的全角逗号保持原样，不被归一化。"""
    csv_path = _write(
        tmp_path / "_answer" / "quoted_fw.csv",
        'name,desc\nAlice,"你好，世界"\nBob,"再见，朋友"\n',
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["name", "desc"]
    assert result.answer.rows == [["Alice", "你好，世界"], ["Bob", "再见，朋友"]]


def test_from_csv_rejects_wide_rows_after_normalization(task: PublicTask, tmp_path: Path) -> None:
    """归一化后数据行仍比 header 宽 → raise shape error。"""
    csv_path = _write(
        tmp_path / "_answer" / "still_wide.csv",
        "label\nA,1,10\nB,2,20\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    with pytest.raises(
        ValueError, match=r"from_csv rows must have the same number of cells as the header row"
    ):
        _answer_tool.handler(task, args)


def test_from_csv_accepts_quoted_commas(task: PublicTask, tmp_path: Path) -> None:
    """合法 CSV quote 仍由 pandas 正确处理，逗号文本不会被误判为宽行。"""
    csv_path = _write(
        tmp_path / "_answer" / "quoted_commas.csv",
        'comment\n"alpha,beta"\n"gamma,delta"\n',
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.columns == ["comment"]
    assert result.answer.rows == [["alpha,beta"], ["gamma,delta"]]


def test_inline_submission_with_zero_byte_artifact_passes(task: PublicTask, tmp_path: Path) -> None:
    """zero-byte 残留不视作 artifact：inline 提交照常成功并清理零字节文件。

    `_FROM_CSV_SIZE_LIMIT_BYTES` 上限不是这里的拦截理由；空文件更像 mkdir/touch
    误产物，不应触发 from_csv 改写要求。
    """
    artifact = task.task_dir / "_answer" / "answer.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"")
    assert artifact.exists()

    args = AnswerInput.model_validate({"columns": ["id"], "rows": [[1], [2]]})
    result = _answer_tool.handler(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.rows == [[1], [2]]
    assert not artifact.exists()
    assert not artifact.parent.exists()
