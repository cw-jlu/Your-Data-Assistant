"""`tools/execute_python.py` 流截断契约的单元测试。

设计意图（详见 `_read_captured_stream` docstring）：
- ≤ 64KB 输出原封返回，不引入任何 marker
- > 64KB 输出截断为头 32KB + 尾 32KB，中间插 `[TRUNCATED: N bytes elided]` 行
- stdout 与 stderr 各自独立判定，64KB 预算互不挤占
- 通过 `execute_python_code` 端到端跑一次 `print('x' * 200_000)`，
  确认大型 print 不会让 observation 撑过 context 上限

回归案例：run 20260430-025 task_2，单次 `print(records)` 输出 5.5MB stdout
后续 step 全部 400 InvalidParameter 至 max_steps。
"""

from __future__ import annotations

from pathlib import Path

from agents.tools.execute_python import (
    _PYTHON_EXEC_STREAM_CAP_BYTES,
    _PYTHON_EXEC_STREAM_HEAD_BYTES,
    _PYTHON_EXEC_STREAM_TAIL_BYTES,
    _read_captured_stream,
    execute_python_code,
)


def test_capture_stream_unchanged_when_under_cap(tmp_path: Path) -> None:
    """≤ 64KB 输出原样返回，不能引入任何 truncation marker。"""
    path = tmp_path / "stdout.txt"
    payload = "hello world\n" * 1000  # ~12KB，远低于 64KB
    path.write_text(payload, encoding="utf-8")

    out = _read_captured_stream(path)

    assert out == payload
    assert "[TRUNCATED:" not in out


def test_capture_stream_truncates_past_cap(tmp_path: Path) -> None:
    """> 64KB 输出：保留头/尾各 32KB；中段被丢；marker 含 elided 字节数。

    用三段可识别 sentinel 验证截断位置：
    - `START_xxxxx`     在头部（应保留）
    - `MIDDLE_xxxxx`    在中段（必须被丢）
    - `END_xxxxx`       在尾部（应保留）

    每段填充 40K，确保 32K head/tail 切点落在填充区内、不会切到中段 sentinel。
    """
    path = tmp_path / "stdout.txt"
    head_payload = "START_" + ("a" * 40_000) + "_HEAD_END"
    middle_payload = "MIDDLE_" + ("b" * 100_000) + "_MIDDLE_END"
    tail_payload = "TAIL_START_" + ("c" * 40_000) + "_END"
    full = head_payload + middle_payload + tail_payload
    path.write_text(full, encoding="utf-8")
    actual_size = path.stat().st_size
    assert actual_size > _PYTHON_EXEC_STREAM_CAP_BYTES

    out = _read_captured_stream(path)

    assert out.startswith("START_")
    assert out.rstrip().endswith("_END")
    # 中段 sentinel 必然落在 [32KB, size - 32KB) 区间内 → 被截断
    assert "MIDDLE_" not in out
    assert "_MIDDLE_END" not in out
    assert "[TRUNCATED:" in out
    # marker 中的 elided 字节数等于实际丢弃量
    expected_elided = actual_size - (
        _PYTHON_EXEC_STREAM_HEAD_BYTES + _PYTHON_EXEC_STREAM_TAIL_BYTES
    )
    assert f"[TRUNCATED: {expected_elided} bytes elided]" in out
    # 总长 ≈ 64KB + marker（一行短文本），允许少量浮动
    assert len(out) <= _PYTHON_EXEC_STREAM_CAP_BYTES + 200


def test_capture_stream_handles_non_utf8_bytes_in_truncation(tmp_path: Path) -> None:
    """截断分支对非 UTF-8 字节用 errors=replace 兜底，不应崩溃。"""
    path = tmp_path / "stdout.txt"
    # 拼一段超过 64KB 的二进制流；非 UTF-8 字节散布在头尾
    head_bytes = b"\xff\xfeHEADER_OK\n" + (b"a" * 30_000)
    middle_bytes = b"x" * 100_000
    tail_bytes = (b"b" * 30_000) + b"\xff\xfeTAILER_OK\n"
    path.write_bytes(head_bytes + middle_bytes + tail_bytes)

    out = _read_captured_stream(path)

    assert "HEADER_OK" in out
    assert "TAILER_OK" in out
    assert "[TRUNCATED:" in out


def test_execute_python_truncates_huge_print(tmp_path: Path) -> None:
    """端到端：`print('x' * 200_000)` → output 长度 ≤ 64KB + marker、含 TRUNCATED 标记。"""
    answer_dir = tmp_path / "_answer"
    result = execute_python_code(
        tmp_path,
        "print('x' * 200_000)",
        answer_dir=answer_dir,
        timeout_seconds=15,
    )

    assert result["success"] is True
    output: str = result["output"]
    assert "[TRUNCATED:" in output
    assert len(output) <= _PYTHON_EXEC_STREAM_CAP_BYTES + 200
    # 头尾仍是 'x'，验证截断后仍能看见原内容头/尾两端
    assert output.startswith("x" * 100)
    assert output.rstrip().endswith("x" * 100)


def test_execute_python_short_print_unchanged(tmp_path: Path) -> None:
    """端到端：< 64KB 的 print 输出不应触发任何截断。"""
    answer_dir = tmp_path / "_answer"
    result = execute_python_code(
        tmp_path,
        "print('hello world')",
        answer_dir=answer_dir,
        timeout_seconds=15,
    )

    assert result["success"] is True
    assert result["output"].strip() == "hello world"
    assert "[TRUNCATED:" not in result["output"]
