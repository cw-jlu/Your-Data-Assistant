"""测试用 Scripted 模型适配器。

把 `ScriptedModelAdapter` / `ScriptedNativeModelAdapter` 从生产包搬到这里：
- 真实 OpenAI 调用在 CI 上不可靠（网络 + 额度），单元测试需要可注入的确定性响应源；
- 这些类没有任何生产调用方，留在 `src/` 下会让 pyright strict 把测试 fixture
  的 kwargs 当作 Unknowns，污染严格类型门禁。

测试代码用 `from tests.helpers.scripted_adapters import ...` 引入。
"""

from __future__ import annotations

from typing import Any

from agents.llm.types import ModelMessage, ModelResponse


class ScriptedModelAdapter:
    """测试/调试专用：按预置**文本**列表依次返回响应。

    真正的 OpenAI 调用在 CI 上不可靠（网络 + 额度），单元测试用它注入确定性的
    输出。签名保持与 `ModelAdapter` 协议一致（返回 `ModelResponse`）。
    """

    def __init__(self, responses: list[str]) -> None:
        # 拷贝一份：避免外部修改列表影响播放顺序
        self._responses = list(responses)
        # 每次 complete 收到的 per-call tools（None 或 registry 对象），供行为断言
        self.tools_seen: list[object | None] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        # messages 对预置响应无用，显式 del 提示读者（也能在启用严格 linter 时消除未用警告）
        del messages, kwargs
        self.tools_seen.append(tools)
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        text = self._responses.pop(0)
        return ModelResponse(
            content=text,
            tool_calls=[],
            raw_response=text,
            raw_tool_calls=[],
        )


class ScriptedNativeModelAdapter:
    """测试/调试专用：按预置 `ModelResponse` 列表依次返回，覆盖 native 协议路径。

    与 `ScriptedModelAdapter` 的差异：测试可以直接塞入已解析好的 `tool_calls`
    与 `raw_tool_calls`，无需经过 JSON 字符串往返。用于断言 native 循环的
    分支行为（并行 tool_calls / answer 短路 / 空 tool_calls 兜底等）。
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        # 每次 complete 收到的 per-call tools（None 或 registry 对象），供行为断言
        self.tools_seen: list[object | None] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del messages, kwargs
        self.tools_seen.append(tools)
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
