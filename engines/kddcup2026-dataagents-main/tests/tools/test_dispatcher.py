"""工具分发层 `dispatcher.dispatch_tool_call` 测试。

验证 D3 关键不变式：终止权威来自注册表元数据 + handler 成功；handler 抛异常一律
原样上抛，由调用方记 `tool_error` 事件；非终止工具永远 `should_terminate=False`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.llm.types import ModelToolCall
from agents.tools.dispatcher import dispatch_tool_call
from agents.tools.registry import create_default_tool_registry


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "hello.txt").write_text("hello\n")
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _call(name: str, **arguments: object) -> ModelToolCall:
    return ModelToolCall(id=f"call_{name}", name=name, arguments=dict(arguments))


def test_dispatch_non_terminal_success_does_not_terminate(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("inspect_files")

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is False


def test_dispatch_terminal_success_terminates(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("answer", columns=["greeting"], rows=[["hi"]])

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    assert result.answer.columns == ["greeting"]


def test_dispatch_invalid_answer_raises_and_does_not_terminate(task: PublicTask) -> None:
    """answer payload 校验失败 → registry 边界翻译为结构化 ValueError，dispatch 原样上抛。

    这个用例锁住 D3：terminal flag 是元数据，但只在 handler 成功返回后才生效。
    """
    registry = create_default_tool_registry()
    # rows 长度与 columns 不匹配 → AnswerInput 的 model_validator 抛错
    call = _call("answer", columns=["a", "b"], rows=[["only one"]])

    with pytest.raises(ValueError, match=r"^answer: row 0 has 1 cells but columns has 2"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_unknown_tool_raises_key_error(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("not_a_real_tool")

    with pytest.raises(KeyError, match="not_a_real_tool"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_path_escape_raises_value_error(task: PublicTask) -> None:
    """sandboxing：`resolve_context_path` 阻止 `../` 越界访问。"""
    registry = create_default_tool_registry()
    call = _call("preview_file", path="../../etc/passwd")

    with pytest.raises(ValueError, match="escapes context dir"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_missing_file_raises_file_not_found(task: PublicTask) -> None:
    """合法路径但文件不存在 → FileNotFoundError 上抛。"""
    registry = create_default_tool_registry()
    call = _call("preview_file", path="missing.csv")

    with pytest.raises(FileNotFoundError, match="Missing context asset"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_python_code_failure_returns_ok_false(task: PublicTask) -> None:
    """`execute_python` handler 不抛异常，但用户代码失败时 `result.ok=False` 且不终止。"""
    registry = create_default_tool_registry()
    call = _call("execute_python", code="raise RuntimeError('boom')")

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is False
    assert should_terminate is False


# ----- Pydantic 入参校验：spec ADDED 场景 (^anchor 锁住 tool 名是首 token) -----


def test_dispatch_missing_required_field_reports_field_name(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("preview_file")  # `path` 缺失

    with pytest.raises(ValueError, match=r"^preview_file: field 'path' is required"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_wrong_type_for_required_field_reports_field(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("preview_file", path=42)  # path 是 int 而非 str

    with pytest.raises(ValueError, match=r"^preview_file: field 'path' must be a string, got int"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_unknown_field_is_rejected(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("preview_file", path="csv/foo.csv", ttl=5)  # ttl 未声明

    with pytest.raises(ValueError, match=r"^preview_file: extra field 'ttl' is not permitted"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_preview_file_max_rows_is_rejected(task: PublicTask) -> None:
    """preview_file 的容量是固定契约，不允许模型用 max_rows 旋钮绕过。"""
    registry = create_default_tool_registry()
    call = _call("preview_file", path="csv/foo.csv", max_rows=200)

    with pytest.raises(ValueError, match=r"^preview_file: extra field 'max_rows' is not permitted"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_answer_row_width_mismatch_reports_index(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("answer", columns=["a", "b"], rows=[["only one"]])

    with pytest.raises(ValueError, match=r"^answer: row 0 has 1 cells but columns has 2"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_answer_neither_payload_is_rejected(task: PublicTask) -> None:
    """既不传 from_csv 也不传 columns/rows → XOR 校验失败提示模型补一份 inline 表。"""
    registry = create_default_tool_registry()
    call = _call("answer", columns=[], rows=[])

    with pytest.raises(
        ValueError, match=r"^answer: columns must be non-empty when from_csv is not set"
    ):
        dispatch_tool_call(task, call, registry)


def test_dispatch_answer_xor_violation_is_rejected(task: PublicTask) -> None:
    """同时传 from_csv 和 inline columns/rows → XOR 校验失败防止两路并存歧义。"""
    registry = create_default_tool_registry()
    call = _call(
        "answer",
        columns=["a"],
        rows=[["x"]],
        from_csv="/tmp/answer.csv",
    )

    with pytest.raises(
        ValueError,
        match=r"^answer: provide either from_csv or columns\+rows, not both",
    ):
        dispatch_tool_call(task, call, registry)


def test_dispatch_answer_inline_row_threshold_is_rejected(task: PublicTask) -> None:
    """11 行 inline rows → 触发硬闸；模型必须改走 from_csv。

    回归 run 20260430-015 task_8：execute_python 已写出正确的 140 行 CSV，但模型
    在 answer 工具调用里把 rows 自抄了一遍，长结构化生成出现 attention drift，
    多吐 3 行幻觉 + 57 行 amount 串味。INLINE_ANSWER_ROW_LIMIT 切断这条退路。
    """
    registry = create_default_tool_registry()
    call = _call("answer", columns=["x"], rows=[[i] for i in range(11)])

    with pytest.raises(
        ValueError,
        match=(
            r"^inline rows payload exceeds artifact-handoff threshold "
            r"\(got 11 rows, 11 cells; inline is allowed only for <= 10 rows AND "
            r"<= 50 cells; if either limit is exceeded, use from_csv\)"
        ),
    ):
        dispatch_tool_call(task, call, registry)


def test_dispatch_answer_inline_cell_threshold_is_rejected(task: PublicTask) -> None:
    """9 行 × 6 列 = 54 cells → 即使行数没超过 10，单元格量超阈值同样硬闸。

    覆盖宽表场景（如 task_4 风格的 schema dump）：少行多列同样会触发幻觉。
    """
    registry = create_default_tool_registry()
    cols = [f"c{i}" for i in range(6)]
    rows = [[i] * 6 for i in range(9)]
    call = _call("answer", columns=cols, rows=rows)

    with pytest.raises(
        ValueError,
        match=(
            r"^inline rows payload exceeds artifact-handoff threshold "
            r"\(got 9 rows, 54 cells; inline is allowed only for <= 10 rows AND "
            r"<= 50 cells; if either limit is exceeded, use from_csv\)"
        ),
    ):
        dispatch_tool_call(task, call, registry)


def test_dispatch_answer_inline_at_row_threshold_succeeds(task: PublicTask) -> None:
    """10 行 × 1 列 = 10 cells → 正好行数阈值，仍允许 inline。"""
    registry = create_default_tool_registry()
    call = _call("answer", columns=["x"], rows=[[i] for i in range(10)])

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    assert len(result.answer.rows) == 10
    assert result.content["row_count"] == 10
    assert result.content["column_count"] == 1


def test_dispatch_answer_inline_at_cell_threshold_succeeds(task: PublicTask) -> None:
    """5 行 × 10 列 = 50 cells → 正好单元格阈值，仍允许 inline。"""
    registry = create_default_tool_registry()
    cols = [f"c{i}" for i in range(10)]
    rows = [[i] * 10 for i in range(5)]
    call = _call("answer", columns=cols, rows=rows)

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    assert len(result.answer.rows) == 5
    assert result.content["row_count"] == 5
    assert result.content["column_count"] == 10


def test_dispatch_answer_inline_just_below_threshold_succeeds(task: PublicTask) -> None:
    """9 行 × 5 列 = 45 cells → 双阈值都没碰，inline 仍然终止成功。

    锁住"硬闸只对超量场景生效"：常规小答案不受影响。
    """
    registry = create_default_tool_registry()
    cols = [f"c{i}" for i in range(5)]
    rows = [[i] * 5 for i in range(9)]
    call = _call("answer", columns=cols, rows=rows)

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    assert len(result.answer.rows) == 9
    assert result.content["row_count"] == 9
    assert result.content["column_count"] == 5


def test_dispatch_answer_inline_path_unchanged_after_from_csv_addition(
    task: PublicTask,
) -> None:
    """回归 #15：from_csv 字段加入后，inline 路径行为必须 byte-identical。

    覆盖原 `test_dispatch_terminal_success_terminates` 的契约：
    columns/rows 单独提交即终止，answer.columns 落地无误。
    """
    registry = create_default_tool_registry()
    call = _call("answer", columns=["greeting"], rows=[["hi"]])

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    assert result.answer.columns == ["greeting"]
    assert result.answer.rows == [["hi"]]
    # ack content 仍然只暴露 status + counts，不含 from_csv 元信息
    assert result.content == {
        "status": "submitted",
        "column_count": 1,
        "row_count": 1,
    }


def test_dispatch_answer_preserves_integer_cells(task: PublicTask) -> None:
    """JSON 整数 cell 必须以 int 落地，不被 union 中的 float 静默 coerce。

    回归 task_11：模型提交 [[163109, "F", "SLE"]]，落地 prediction.csv 应为
    `163109,F,SLE` 而非 `163109.0,F,SLE`——后者会与 gold 的字面 `163109` 失配。
    依赖 AnswerInput.rows 的 union 把 `int` 排在 `float` 之前。
    """
    registry = create_default_tool_registry()
    call = _call(
        "answer",
        columns=["ID", "SEX", "Diagnosis"],
        rows=[[163109, "F", "SLE"], [2803470, "F", "SLE"]],
    )

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    cell = result.answer.rows[0][0]
    assert cell == 163109
    assert isinstance(cell, int)
    assert not isinstance(cell, bool)  # bool 是 int 子类，单独防御
    cell2 = result.answer.rows[1][0]
    assert cell2 == 2803470
    assert isinstance(cell2, int)


def test_dispatch_answer_preserves_float_cells(task: PublicTask) -> None:
    """JSON 浮点 cell 必须以 float 落地（不被 int 抢走匹配）。"""
    registry = create_default_tool_registry()
    call = _call("answer", columns=["ratio"], rows=[[1.5], [0.25]])

    result, _ = dispatch_tool_call(task, call, registry)

    assert result.answer is not None
    assert isinstance(result.answer.rows[0][0], float)
    assert result.answer.rows[0][0] == 1.5
    assert isinstance(result.answer.rows[1][0], float)


# ----- 不可变契约：registry.execute 不修改 caller 的 action_input dict -----


def test_registry_execute_does_not_mutate_payload_on_success(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    payload: dict[str, object] = {"path": "hello.txt"}
    snapshot = dict(payload)

    registry.execute(task, "preview_file", payload)

    assert payload == snapshot
    assert set(payload.keys()) == set(snapshot.keys())


def test_registry_execute_does_not_mutate_payload_on_failure(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    payload: dict[str, object] = {}  # 缺 path → 校验失败
    snapshot = dict(payload)

    with pytest.raises(ValueError):
        registry.execute(task, "preview_file", payload)

    assert payload == snapshot
