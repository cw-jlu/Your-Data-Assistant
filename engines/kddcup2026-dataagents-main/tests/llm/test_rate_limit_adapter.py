"""`RateLimitedAdapter` 的分支行为测试。

策略：
- 用 `_FakeBucket` 代替真实 `FileTokenBucket`，记录 acquire/refund 调用并按脚本返回
- 用 `_ScriptedInnerAdapter` 注入成功 / 429 / 5xx / 非可重试错误序列
- 用 monkeypatch 把 `time.sleep` 和 `random.uniform` 钉为可预测值，测试跑完立刻退出

覆盖：
- 成功路径：pre-deduct + reconcile（actual<reserved 退款 / actual>reserved 补扣）
- 429 带 Retry-After header → 按 header 休眠重试
- 429 无 header → 指数退避
- 5xx → 同重试路径
- 非可重试错误（400） → 原样上抛 RuntimeError，并退回预扣的 TPM
- 重试耗尽 → 抛 RateLimitExhausted，退回预扣的 TPM
- estimate_prompt_tokens 覆盖 role/content/tool_call_id/tool_calls
- rate_limit_metrics 在各行为测试内一并断言（成功挂 response，异常挂 _last_metrics）
"""

from __future__ import annotations

import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from openai import APIError

from agents.config import RateLimitConfig
from agents.llm.rate_limit import (
    RateLimitedAdapter,
    RateLimitExhausted,
    _is_retriable,
    _parse_retry_after,
    estimate_prompt_tokens,
)
from agents.llm.types import ModelMessage, ModelResponse, TokenUsage

# ------- 小工具 -------------------------------------------------------------------


def _fake_api_error(
    status_code: int, *, headers: dict[str, str] | None = None, message: str = "fake"
) -> APIError:
    """绕过 APIError 真实构造函数，直接塞属性——避免在测试里伪造 httpx.Response。

    `_is_retriable` / `_parse_retry_after` 只读 status_code 和 response.headers，够用。
    """
    err = APIError.__new__(APIError)
    err.message = message  # type: ignore[attr-defined]
    err.status_code = status_code  # type: ignore[attr-defined]
    err.response = types.SimpleNamespace(headers=headers or {})  # type: ignore[attr-defined]
    err.args = (message,)
    return err


def _wrap_as_runtime(exc: APIError) -> RuntimeError:
    """重现 OpenAIModelAdapter 的包装：`raise RuntimeError(...) from exc`。"""
    try:
        raise exc
    except APIError as caught:
        try:
            raise RuntimeError(f"Model request failed: {caught}") from caught
        except RuntimeError as wrapped:
            return wrapped


class _FakeBucket:
    """轻量 bucket 替身；记录所有调用并按脚本返回 wait 秒数。"""

    def __init__(self) -> None:
        self.acquire_calls: list[tuple[int, int]] = []
        self.refund_calls: list[tuple[int, int]] = []
        self._acquire_returns: list[float] = []

    def set_acquire_returns(self, returns: list[float]) -> None:
        """按顺序返回的 wait 值；用完后一律返 0.0（成功）。"""
        self._acquire_returns = list(returns)

    def acquire(self, *, need_rpm: int, need_tpm: int) -> float:
        self.acquire_calls.append((need_rpm, need_tpm))
        if self._acquire_returns:
            return self._acquire_returns.pop(0)
        return 0.0

    def refund(self, *, rpm: int, tpm: int) -> None:
        self.refund_calls.append((rpm, tpm))


class _ScriptedInnerAdapter:
    """内层 adapter 替身：按注入顺序返回 ModelResponse / 抛异常。"""

    def __init__(self, script: list[ModelResponse | BaseException]) -> None:
        self._script = list(script)
        self.calls = 0

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del messages, tools, kwargs
        self.calls += 1
        if not self._script:
            raise RuntimeError("Scripted inner adapter exhausted.")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _EstimatingInnerAdapter(_ScriptedInnerAdapter):
    """Inner adapter with a custom prompt-token estimator."""

    def __init__(self, script: list[ModelResponse | BaseException], prompt_tokens: int) -> None:
        super().__init__(script)
        self.prompt_tokens = prompt_tokens
        self.estimate_calls = 0

    def estimate_prompt_tokens(
        self, messages: list[ModelMessage], *, tools: object | None = None
    ) -> int:
        del messages, tools
        self.estimate_calls += 1
        return self.prompt_tokens


@pytest.fixture
def rate_limit_cfg(tmp_path: Path) -> RateLimitConfig:
    """通用 rate_limit 配置；state_dir 指向 tmp_path，backoff 小以便测试跑得快。"""
    return RateLimitConfig(
        rpm_per_key=600,
        tpm_per_key=1_000_000,
        safety_factor=1.0,
        reserve_output_tokens=1000,
        max_retries=3,
        backoff_initial_seconds=1.0,
        backoff_max_seconds=8.0,
        state_dir=tmp_path,
    )


@pytest.fixture
def fast_sleep_and_jitter(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """把 time.sleep 钉为记录调用但不真睡；随机抖动钉为 0。"""
    slept: list[float] = []
    monkeypatch.setattr(
        "agents.llm.rate_limit.time.sleep",
        lambda s: slept.append(s),
    )
    monkeypatch.setattr(
        "agents.llm.rate_limit.random.uniform",
        lambda a, b: 0.0,
    )
    return slept


def _make_adapter(
    inner: _ScriptedInnerAdapter, bucket: _FakeBucket, cfg: RateLimitConfig
) -> RateLimitedAdapter:
    """构造 RateLimitedAdapter 后把内部 bucket 替换成 FakeBucket。"""
    adapter = RateLimitedAdapter(inner=inner, api_key="test-key", rate_limit=cfg)
    adapter.bucket = bucket  # type: ignore[assignment]
    return adapter


def _make_response(total_tokens: int) -> ModelResponse:
    return ModelResponse(
        content="ok",
        usage=TokenUsage(total_tokens=total_tokens),
    )


def _messages() -> list[ModelMessage]:
    return [
        ModelMessage(role="system", content="you are helpful"),
        ModelMessage(role="user", content="hello"),
    ]


# ------- estimate_prompt_tokens ---------------------------------------------------


def test_estimate_counts_role_and_content(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenized: list[str] = []

    def fake_count_qwen_tokens(text: str) -> int:
        tokenized.append(text)
        return len(text)

    monkeypatch.setattr(
        "agents.llm.rate_limit.count_qwen_tokens",
        fake_count_qwen_tokens,
    )
    messages = [
        ModelMessage(role="user", content="a" * 250),  # 250 bytes
    ]
    est = estimate_prompt_tokens(messages)
    assert est == 4 + len("user") + 250
    assert tokenized == ["user", "a" * 250]


def test_estimate_counts_tool_call_fields() -> None:
    tc_payload = [
        {"id": "abc", "type": "function", "function": {"name": "x", "arguments": '{"k":1}'}}
    ]
    messages = [
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=tc_payload,
        ),
        ModelMessage(role="tool", content="observation", tool_call_id="abc"),
    ]
    est = estimate_prompt_tokens(messages)
    assert est >= 1  # 非零即可，证明进入 tool_calls 分支没 crash


def test_estimate_always_at_least_one() -> None:
    assert estimate_prompt_tokens([]) >= 1


def test_estimate_video_block_constant() -> None:
    messages = [
        ModelMessage(
            role="user",
            content=[
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        ),
    ]
    est = estimate_prompt_tokens(messages)
    assert est >= 16_384


# ------- _is_retriable / _parse_retry_after --------------------------------------


@pytest.mark.parametrize(
    ("make_error", "expected"),
    [
        pytest.param(lambda: _wrap_as_runtime(_fake_api_error(429)), True, id="429"),
        pytest.param(lambda: _wrap_as_runtime(_fake_api_error(503)), True, id="5xx"),
        pytest.param(lambda: _wrap_as_runtime(_fake_api_error(400)), False, id="400"),
        # 没挂 APIError，但消息里有 "rate limit"
        pytest.param(
            lambda: RuntimeError(
                "Model request failed: 429 Too Many Requests / rate limit exceeded"
            ),
            True,
            id="message-fallback",
        ),
    ],
)
def test_is_retriable(make_error: Callable[[], BaseException], expected: bool) -> None:
    assert _is_retriable(make_error()) is expected


def test_parse_retry_after_reads_header() -> None:
    wrapped = _wrap_as_runtime(_fake_api_error(429, headers={"Retry-After": "7"}))
    assert _parse_retry_after(wrapped) == 7.0


def test_parse_retry_after_missing_returns_none() -> None:
    wrapped = _wrap_as_runtime(_fake_api_error(429, headers={}))
    assert _parse_retry_after(wrapped) is None


# ------- RateLimitedAdapter.complete ---------------------------------------------


def test_happy_path_reconciles_unused_tpm(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    del fast_sleep_and_jitter  # 不需要校验 sleep，happy 路径不睡
    bucket = _FakeBucket()
    inner = _ScriptedInnerAdapter([_make_response(total_tokens=300)])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    response = adapter.complete(_messages())
    assert response.content == "ok"
    # 首次 acquire 应预扣 (1, estimate+reserve=...+1000)
    assert bucket.acquire_calls[0][0] == 1
    # 预扣 TPM 远大于 actual（300），应触发退款
    assert any(call[0] == 0 and call[1] > 0 for call in bucket.refund_calls)
    # 同一场景的 response 附带 metrics：无重试、无等待，tpm_delta = actual - reserved
    m = response.rate_limit_metrics
    assert m is not None
    assert m["retry_count"] == 0
    assert m["actual_tpm"] == 300
    assert m["tpm_delta"] == 300 - m["reserved_tpm"]
    assert m["acquire_wait_ms"] == 0.0
    assert "exhausted" not in m
    assert "error" not in m


def test_over_reserved_triggers_extra_debit(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    del fast_sleep_and_jitter
    bucket = _FakeBucket()
    # 预扣约 ≈ estimate(messages)+1000；传 total_tokens 远大于预扣
    inner = _ScriptedInnerAdapter([_make_response(total_tokens=100_000)])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    adapter.complete(_messages())
    # 至少两次 acquire：预扣 + 补扣
    assert len(bucket.acquire_calls) >= 2
    # 补扣的 RPM 应为 0（只补 TPM），对应 calls[1] 形如 (0, delta)
    assert bucket.acquire_calls[1][0] == 0
    assert bucket.acquire_calls[1][1] > 0


def test_429_with_retry_after_honored(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    err = _wrap_as_runtime(_fake_api_error(429, headers={"Retry-After": "3"}))
    bucket = _FakeBucket()
    inner = _ScriptedInnerAdapter([err, _make_response(total_tokens=100)])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    response = adapter.complete(_messages())
    assert response.content == "ok"
    # 第一次 sleep 应为 Retry-After 指定的 3s（+ jitter=0）
    assert 3.0 in fast_sleep_and_jitter
    assert inner.calls == 2
    assert response.rate_limit_metrics is not None
    assert response.rate_limit_metrics["retry_count"] == 1


def test_429_without_header_uses_exponential_backoff(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    err1 = _wrap_as_runtime(_fake_api_error(429, headers={}))
    err2 = _wrap_as_runtime(_fake_api_error(429, headers={}))
    bucket = _FakeBucket()
    inner = _ScriptedInnerAdapter([err1, err2, _make_response(total_tokens=100)])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    response = adapter.complete(_messages())
    assert response.content == "ok"
    # 第一次退避应 = backoff_initial_seconds (1.0)；第二次应翻倍 (2.0)
    assert 1.0 in fast_sleep_and_jitter
    assert 2.0 in fast_sleep_and_jitter


def test_5xx_triggers_retry(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    del fast_sleep_and_jitter
    err = _wrap_as_runtime(_fake_api_error(503))
    bucket = _FakeBucket()
    inner = _ScriptedInnerAdapter([err, _make_response(total_tokens=100)])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    response = adapter.complete(_messages())
    assert response.content == "ok"
    assert inner.calls == 2


def test_non_retriable_error_raises_immediately_and_refunds(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    del fast_sleep_and_jitter
    err = _wrap_as_runtime(_fake_api_error(400))
    bucket = _FakeBucket()
    inner = _ScriptedInnerAdapter([err])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    with pytest.raises(RuntimeError) as exc_info:
        adapter.complete(_messages())
    # 不应是 RateLimitExhausted；应是原 RuntimeError
    assert not isinstance(exc_info.value, RateLimitExhausted)
    # 退款：rpm=0, tpm>0（把预扣的 TPM 退回）
    assert bucket.refund_calls
    assert bucket.refund_calls[-1][0] == 0
    assert bucket.refund_calls[-1][1] > 0
    m = adapter._last_metrics
    assert m is not None
    assert m["error"] == "RuntimeError"
    assert m["actual_tpm"] == 0


def test_retries_exhausted_raises_rate_limit_exhausted(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    del fast_sleep_and_jitter
    # max_retries=3 → 1 初次 + 3 重试 = 4 次；全抛 429
    script: list[ModelResponse | BaseException] = [
        _wrap_as_runtime(_fake_api_error(429)) for _ in range(4)
    ]
    bucket = _FakeBucket()
    inner = _ScriptedInnerAdapter(script)
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    with pytest.raises(RateLimitExhausted):
        adapter.complete(_messages())
    assert inner.calls == 4
    # 退款 TPM：exhausted 路径应退预扣
    assert any(call[0] == 0 and call[1] > 0 for call in bucket.refund_calls)
    # 异常路径没有 response 可挂，metrics 落在 adapter._last_metrics
    m = adapter._last_metrics
    assert m is not None
    assert m["exhausted"] is True
    assert m["retry_count"] == 4


def test_acquire_waits_when_bucket_says_to(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    bucket = _FakeBucket()
    # 第一次 acquire 返 2s 等待，第二次返 0（成功）
    bucket.set_acquire_returns([2.0, 0.0])
    inner = _ScriptedInnerAdapter([_make_response(total_tokens=100)])
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    response = adapter.complete(_messages())
    # 应看到一次 2.0s 的 sleep（+ jitter=0），且等待时长记入 metrics
    assert 2.0 in fast_sleep_and_jitter
    assert response.rate_limit_metrics is not None
    assert response.rate_limit_metrics["acquire_wait_ms"] == 2000.0


def test_complete_uses_inner_prompt_token_estimator(
    rate_limit_cfg: RateLimitConfig,
    fast_sleep_and_jitter: list[float],
) -> None:
    del fast_sleep_and_jitter
    bucket = _FakeBucket()
    inner = _EstimatingInnerAdapter([_make_response(total_tokens=100)], prompt_tokens=12_345)
    adapter = _make_adapter(inner, bucket, rate_limit_cfg)

    adapter.complete(_messages())

    assert inner.estimate_calls == 1
    assert bucket.acquire_calls[0] == (1, 12_345 + rate_limit_cfg.reserve_output_tokens)


def test_constructor_rejects_none_state_dir() -> None:
    inner = _ScriptedInnerAdapter([])
    cfg = RateLimitConfig(state_dir=None)
    with pytest.raises(ValueError, match="state_dir"):
        RateLimitedAdapter(inner=inner, api_key="k", rate_limit=cfg)


def test_constructor_rejects_empty_api_key(tmp_path: Path) -> None:
    inner = _ScriptedInnerAdapter([])
    cfg = RateLimitConfig(state_dir=tmp_path)
    with pytest.raises(ValueError, match="api_key"):
        RateLimitedAdapter(inner=inner, api_key="", rate_limit=cfg)
