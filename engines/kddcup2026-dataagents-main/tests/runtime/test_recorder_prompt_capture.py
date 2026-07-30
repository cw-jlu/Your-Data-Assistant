"""ReAct 循环把 prompt_messages / response_message 落到 StepRecord 的回归测试。

覆盖：
- 每条成功 step 的 `prompt_messages` 至少包含 system + user 两条消息
- 每条成功 step 的 `response_message` 含 content / tool_calls / usage
- Step 1 的 prompt 不含 assistant 回放；Step 2 应该有上一轮的 assistant + tool 消息
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.agent import ReActAgent, ReActAgentConfig
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.llm.types import ModelResponse, ModelToolCall, TokenUsage
from agents.tools.registry import create_default_tool_registry
from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter


def _make_task(tmp_path: Path) -> PublicTask:
    """构造一个最小可用的任务（仅一个空的 context 目录）。"""
    task_dir = tmp_path / "task_capture"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_capture", difficulty="easy", question="q?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _build_two_turn_adapter() -> ScriptedNativeModelAdapter:
    """两轮脚本：inspect_files → answer。供"两步任务"路径使用。"""
    list_call = ModelToolCall(id="c1", name="inspect_files", arguments={})
    list_raw = {
        "id": "c1",
        "type": "function",
        "function": {"name": "inspect_files", "arguments": json.dumps({})},
    }
    answer_call = ModelToolCall(
        id="c2",
        name="answer",
        arguments={"columns": ["v"], "rows": [["ok"]]},
    )
    answer_raw = {
        "id": "c2",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": json.dumps({"columns": ["v"], "rows": [["ok"]]}),
        },
    }
    return ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="step1",
                tool_calls=[list_call],
                raw_response="step1",
                raw_tool_calls=[list_raw],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="step2",
                tool_calls=[answer_call],
                raw_response="step2",
                raw_tool_calls=[answer_raw],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
            ),
        ]
    )


def test_step_records_carry_prompt_and_response(tmp_path: Path) -> None:
    """端到端：跑两步 ReAct，断言每步都带 prompt_messages + response_message。"""
    task = _make_task(tmp_path)
    agent = ReActAgent(
        model=_build_two_turn_adapter(),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 2

    # ---- Step 1: prompt 仅含 system + user；尚无 assistant 回放 ----
    step1 = result.steps[0]
    assert step1.prompt_messages is not None
    roles_step1 = [m["role"] for m in step1.prompt_messages]
    assert roles_step1 == ["system", "user"]
    # response_message 应包含 content / tool_calls / usage
    assert step1.response_message is not None
    assert step1.response_message["content"] == "step1"
    assert len(step1.response_message["tool_calls"]) == 1
    assert step1.response_message["tool_calls"][0]["name"] == "inspect_files"
    assert step1.response_message["usage"]["prompt_tokens"] == 10

    # ---- Step 2: prompt 应含 system + user + assistant(step1) + tool(step1 obs) ----
    step2 = result.steps[1]
    assert step2.prompt_messages is not None
    roles_step2 = [m["role"] for m in step2.prompt_messages]
    assert roles_step2 == ["system", "user", "assistant", "tool"]
    assert step2.response_message is not None
    assert step2.response_message["tool_calls"][0]["name"] == "answer"


def test_parallel_tool_calls_only_lead_step_carries_prompt_and_response(
    tmp_path: Path,
) -> None:
    """lead-only 不变式：单 turn 并行 N 个 tool_calls 时，只有首条 StepRecord
    携带 `prompt_messages` / `response_message`，其余应为 None。

    动机：完整 messages + response 落到每条 StepRecord 会让数据体积线性膨胀
    （N 倍），且回放时只需读首条即可还原。该约定与 `raw_tool_calls` / `usage` /
    `reasoning` 保持一致——见 `agent.py` 中 `is_lead` 判定。
    """
    task = _make_task(tmp_path)

    list_call_a = ModelToolCall(id="par_a", name="inspect_files", arguments={})
    list_call_b = ModelToolCall(id="par_b", name="inspect_files", arguments={})
    list_raw_a = {
        "id": "par_a",
        "type": "function",
        "function": {"name": "inspect_files", "arguments": json.dumps({})},
    }
    list_raw_b = {
        "id": "par_b",
        "type": "function",
        "function": {"name": "inspect_files", "arguments": json.dumps({})},
    }
    answer_call = ModelToolCall(
        id="done",
        name="answer",
        arguments={"columns": ["v"], "rows": [["ok"]]},
    )
    answer_raw = {
        "id": "done",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": json.dumps({"columns": ["v"], "rows": [["ok"]]}),
        },
    }
    adapter = ScriptedNativeModelAdapter(
        [
            # 第一轮：并行 2 个 inspect_files（产生 2 条 StepRecord，共享 turn_index）
            ModelResponse(
                content="parallel explore",
                tool_calls=[list_call_a, list_call_b],
                raw_response="parallel explore",
                raw_tool_calls=[list_raw_a, list_raw_b],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            # 第二轮：answer 终止
            ModelResponse(
                content="finalize",
                tool_calls=[answer_call],
                raw_response="finalize",
                raw_tool_calls=[answer_raw],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
            ),
        ]
    )

    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 3  # 2 (parallel) + 1 (answer)

    lead_parallel = result.steps[0]
    follower_parallel = result.steps[1]
    answer_step = result.steps[2]

    # ---- lead 必须承载 prompt_messages + response_message ----
    assert lead_parallel.turn_index == follower_parallel.turn_index, (
        "并行 tool_calls 应共享 turn_index，否则 lead-only 切片逻辑会失效"
    )
    assert lead_parallel.prompt_messages is not None
    assert lead_parallel.response_message is not None
    # raw_tool_calls / usage / reasoning 同步检查（与 prompt/response 同约定）
    assert lead_parallel.raw_tool_calls is not None
    assert lead_parallel.usage.prompt_tokens == 10

    # ---- follower 必须为 None（即避免双计 + 节省体积）----
    assert follower_parallel.prompt_messages is None
    assert follower_parallel.response_message is None
    assert follower_parallel.raw_tool_calls is None
    assert follower_parallel.usage == TokenUsage()  # 默认全 0

    # ---- 下一轮（answer）作为新 turn 的 lead，重新承载 ----
    assert answer_step.turn_index != lead_parallel.turn_index
    assert answer_step.prompt_messages is not None
    assert answer_step.response_message is not None
