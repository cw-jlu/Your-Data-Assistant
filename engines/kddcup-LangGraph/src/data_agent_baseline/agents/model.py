from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


class TraceableChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that preserves provider reasoning text for trace output."""

    def _create_chat_result(self, response: dict[str, Any] | Any, generation_info: dict[str, Any] | None = None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()

        for generation, choice in zip(chat_result.generations, response_dict.get("choices") or []):
            message_payload = choice.get("message") or {}
            reasoning_content = message_payload.get("reasoning_content") or message_payload.get("reasoning")
            if reasoning_content is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning_content

        return chat_result


def create_chat_model(
    *,
    model: str,
    api_base: str,
    api_key: str,
    api_key_env: str | None = None,
    temperature: float,
    timeout_seconds: int | None = 1800,
    max_tokens: int | None = 8192,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    enable_thinking: bool | None = None,
) -> BaseChatModel:
    if not api_key:
        if api_key_env:
            raise RuntimeError(
                "Missing model API key. Checked the process environment and project .env "
                f"for config.agent.api_key_env={api_key_env!r}."
            )
        raise RuntimeError(
            "Missing model API key in config.agent.api_key or the project .env file. "
            "Set config.agent.api_key directly or point config.agent.api_key_env at a key in .env."
        )

    extra_body: dict[str, object] = {
        "repetition_penalty": repetition_penalty,
    }
    if enable_thinking is not None:
        extra_body["enable_thinking"] = enable_thinking

    request_kwargs: dict[str, object] = {
        "model": model,
        "base_url": api_base.rstrip("/"),
        "api_key": api_key,
        "temperature": temperature,
        "top_p": top_p,
        "extra_body": extra_body,
        "max_retries": 3,
    }
    if timeout_seconds is not None and timeout_seconds > 0:
        request_kwargs["timeout"] = timeout_seconds
    if max_tokens is not None and max_tokens > 0:
        request_kwargs["max_tokens"] = max_tokens


    return TraceableChatOpenAI(**request_kwargs)
