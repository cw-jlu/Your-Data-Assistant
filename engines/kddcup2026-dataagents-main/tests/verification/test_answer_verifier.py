from __future__ import annotations

from typing import Any

from agents.llm.types import ModelMessage, ModelResponse
from agents.verification.answer import verify_answer


class _ScriptedVerifier:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        return ModelResponse(content=self.response, raw_response=self.response)


def test_verifier_rejects_global_superlative_multiple_rows() -> None:
    adapter = _ScriptedVerifier(
        '{"verdict": "reject", "check": "row_count", '
        '"reason": "Global superlative answers should have one row."}'
    )

    rejection = verify_answer(
        adapter,
        "Who is the fastest driver?",
        ["driver"],
        [["Alice"], ["Bob"]],
    )

    assert rejection is not None
    assert "row_count" in rejection


def test_verifier_allows_grouped_superlative_multiple_rows() -> None:
    adapter = _ScriptedVerifier('{"verdict": "ok"}')

    rejection = verify_answer(
        adapter,
        "Which driver is fastest in each team?",
        ["team", "driver"],
        [["red", "Alice"], ["blue", "Bob"]],
    )

    assert rejection is None


class _CapturingVerifier:
    def __init__(self, response: str = '{"verdict": "ok"}') -> None:
        self.response = response
        self.seen_content: object = None

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del tools
        self.seen_content = messages[-1].content
        return ModelResponse(content=self.response, raw_response=self.response)


def test_verifier_does_not_send_producing_code() -> None:
    adapter = _CapturingVerifier()
    verify_answer(
        adapter,
        "查一下总资产",
        ["TotalAssets"],
        [[1.0]],
        producing_code="SELECT TotalAssets FROM t WHERE TotalAssets IS NOT NULL",
    )
    assert adapter.seen_content is not None
    assert isinstance(adapter.seen_content, str)
    assert "PRODUCING CODE" not in adapter.seen_content


def test_verifier_rejects_null_filtered_listing() -> None:
    adapter = _ScriptedVerifier(
        '{"verdict": "reject", "check": "null_filter", '
        '"reason": "Final query drops NULL TotalAssets rows; rebuild without the filter."}'
    )

    rejection = verify_answer(
        adapter,
        "查一下总资产的金额大小",
        ["TotalAssets"],
        [[40622900.0]],
        producing_code="SELECT TotalAssets FROM t WHERE TotalAssets IS NOT NULL",
    )

    assert rejection is not None
    assert "null_filter" in rejection
