"""`_make_text_adapter` 的行为测试：ETL 专属的输出上限与限流预扣。

策略：
- 用裸 `OpenAIModelAdapter` 作 source，断言 ETL adapter 收紧了 `max_tokens`
- 用 `RateLimitedAdapter` 包一层作 source，断言 ETL 派生出更小的 `reserve_output_tokens`
  且 `dataclasses.replace` 保留了其余限流字段（state_dir / rpm / tpm / safety / retries）

只锁行为关系（"ETL adapter 用 ETL 专属常量、且预扣比全局更小"），不锁具体数字，
常量调整后测试仍成立。
"""

from __future__ import annotations

from pathlib import Path

from agents.config import RateLimitConfig
from agents.etl._constants import ETL_MAX_OUTPUT_TOKENS, ETL_RESERVE_OUTPUT_TOKENS
from agents.etl.extractor import _make_text_adapter
from agents.llm.openai import OpenAIModelAdapter
from agents.llm.rate_limit import RateLimitedAdapter


def _bare_openai(max_tokens: int = 0) -> OpenAIModelAdapter:
    """最内层 source：不实际发请求，仅供 _make_text_adapter 抽取凭证。"""
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        max_tokens=max_tokens,
    )


def test_caps_output_tokens_on_plain_adapter() -> None:
    # 裸 adapter 默认 max_tokens=0（无上限）→ ETL 应收紧到有限上限
    result = _make_text_adapter(_bare_openai())
    assert isinstance(result, OpenAIModelAdapter)
    assert result.max_tokens == ETL_MAX_OUTPUT_TOKENS
    assert result.max_tokens > 0


def test_shrinks_reserve_when_rate_limited(tmp_path: Path) -> None:
    cfg = RateLimitConfig(state_dir=tmp_path)  # reserve_output_tokens 默认值（全局）
    source = RateLimitedAdapter(inner=_bare_openai(), api_key="EMPTY", rate_limit=cfg)

    result = _make_text_adapter(source)

    assert isinstance(result, RateLimitedAdapter)
    # ETL 预扣按更小的量级计费，且严格小于全局 reserve
    assert result.rate_limit_config.reserve_output_tokens == ETL_RESERVE_OUTPUT_TOKENS
    assert result.rate_limit_config.reserve_output_tokens < cfg.reserve_output_tokens
    # 内层输出上限同样被收紧
    assert isinstance(result.inner, OpenAIModelAdapter)
    assert result.inner.max_tokens == ETL_MAX_OUTPUT_TOKENS


def test_preserves_other_rate_limit_fields(tmp_path: Path) -> None:
    # 评测端 vLLM 量级的配置：replace 只能动 reserve，其余必须原样保留
    cfg = RateLimitConfig(
        state_dir=tmp_path,
        rpm_per_key=2000,
        tpm_per_key=2_000_000,
        safety_factor=0.95,
        max_retries=6,
    )
    source = RateLimitedAdapter(inner=_bare_openai(), api_key="EMPTY", rate_limit=cfg)

    result = _make_text_adapter(source)
    assert isinstance(result, RateLimitedAdapter)
    rc = result.rate_limit_config

    assert rc.state_dir == tmp_path
    assert rc.rpm_per_key == 2000
    assert rc.tpm_per_key == 2_000_000
    assert rc.safety_factor == 0.95
    assert rc.max_retries == 6
    # 唯一被改写的字段
    assert rc.reserve_output_tokens == ETL_RESERVE_OUTPUT_TOKENS
