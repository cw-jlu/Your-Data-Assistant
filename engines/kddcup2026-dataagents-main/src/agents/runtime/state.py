"""Agent 运行时状态与结果对象。

区分三个层次：
- `StepRecord`:       单步产物（frozen，表示 "这步已经完成"）
- `AgentRuntimeState`: 循环过程中的可变状态容器（`slots=True` 但非 frozen）
- `AgentRunResult`:    整次运行的对外快照（frozen，供 runner 序列化）

保持 frozen 与 mutable 的区分，是为了避免 ReAct 循环中误改已经落定的历史步骤。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agents.benchmark.schema import AnswerTable
from agents.llm.types import TokenUsage


@dataclass(frozen=True, slots=True)
class StepRecord:
    """一次 ReAct 循环的落定步骤。

    - step_index:       从 1 开始的步号（human-friendly），跨 turn 在同一任务内唯一递增
    - turn_index:       同一 assistant 回合共享一个 turn_index；native 协议下一回合可以
                        产生多条 StepRecord（并行 tool_calls）
    - thought:          模型输出的自然语言思考（JSON payload 的 `thought`，native 下复用 content）
    - action:           被调用的工具名；错误分支使用哨兵值 `__error__`
    - action_input:     工具入参（来自模型输出的 `action_input`，native 下是 tool_call.arguments）
    - raw_response:     模型原始输出字符串，同一 turn 的多条记录共享
    - observation:      工具执行后回填给下一轮的观察结果
    - ok:               本步是否成功（失败时 observation 里带 error 字段）
    - tool_call_id:     上游 tool_call_id（重放 tool 消息时必需）；无 tool_call 时为 None
    - raw_tool_calls:   仅在同 turn 的**首条** StepRecord 上填写，用于 native 协议下重放
                        assistant 消息的 tool_calls 字段；其他记录留 None 以节省空间
    - reasoning:        vLLM `--reasoning-parser qwen3` / 百炼 Qwen3 返回的思维链；
                        与 `raw_tool_calls` 同约定，仅在同 turn 的首条 StepRecord 上填写
    - usage:            对应本 turn 那次 chat.completions 调用的 token 用量；
                        同样仅在首条 StepRecord 承载，后续并行 tool_call 留 default（全 0）
    - prompt_messages:  本 turn 发给 LLM 的完整 messages（含 system / user / assistant / tool）；
                        同 turn 的并行 tool_calls 中只有"首条"承载，便于回放与离线调试
    - response_message: 模型本 turn 的原始响应（content + tool_calls + reasoning + usage 等）；
                        与 prompt_messages 同约定，仅 lead 承载
    """

    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool
    # 默认值保持向后兼容：老的测试/调用只传 step_index...ok 也能构造成功
    turn_index: int = 0
    tool_call_id: str | None = None
    raw_tool_calls: list[dict[str, Any]] | None = None
    reasoning: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    prompt_messages: list[dict[str, Any]] | None = None
    response_message: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """asdict 把 dataclass 递归转成普通 dict（用于 JSON 序列化）。"""
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    """ReAct 循环的可变状态。

    - steps:           已完成的步骤列表（按完成顺序）
    - answer:          Agent 调用 `answer` 工具后填入；有值即视为成功终止
    - failure_reason:  异常终止或 max_steps 耗尽时写入
    非 frozen 是刻意为之：循环内需要不断 append/写入字段。
    """

    steps: list[StepRecord] = field(default_factory=lambda: [])  # noqa: PIE807
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """单个任务整轮 ReAct 运行的最终产物。

    由 `ReActAgent.run` 构造；runner 使用 `.to_dict()` 序列化结果。
    """

    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    @property
    def succeeded(self) -> bool:
        """成功的充要条件：`answer` 已提交且没有记录失败原因。

        注意这里的 "成功" 仅指 Agent 正常走到终止工具，不代表答案正确。
        """
        return self.answer is not None and self.failure_reason is None

    def total_usage(self) -> TokenUsage:
        """把各 StepRecord 的 usage 按字段求和。

        同 turn 只有首条 step 承载真实 usage，后续并行 step 默认全 0，直接求和无双计。
        """
        return TokenUsage(
            prompt_tokens=sum(step.usage.prompt_tokens for step in self.steps),
            completion_tokens=sum(step.usage.completion_tokens for step in self.steps),
            total_tokens=sum(step.usage.total_tokens for step in self.steps),
            cached_tokens=sum(step.usage.cached_tokens for step in self.steps),
            reasoning_tokens=sum(step.usage.reasoning_tokens for step in self.steps),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为 runner 可直接落盘的 dict。"""
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
            "total_usage": self.total_usage().to_dict(),
        }
