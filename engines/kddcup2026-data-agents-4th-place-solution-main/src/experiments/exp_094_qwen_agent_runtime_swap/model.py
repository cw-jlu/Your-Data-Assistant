"""Qwen-Agent native runtime adapter for exp_094.

Drop-in replacement for kobushi_core.model.OpenAIModelAdapter that routes
chat completions through qwen_agent.llm.get_chat_model() instead of the
raw openai SDK. The Qwen-Agent layer is maintained by the Qwen team and
applies Qwen3.5-native tool description / chat template / multi-turn
history handling, which (per PRIORITIES.md URGENT D) is hypothesized to
fit the qwen3.5-35b-a3b training distribution better than our raw
OpenAI-compat client.

Interface contract:
- Implements `complete(messages, *, enable_thinking=None) -> str`
- Accepts the same constructor kwargs as OpenAIModelAdapter so that
  exp_094.runner.build_model_adapter() can swap with no caller changes.
- Defends in depth with strip_thinking() (mirrors OpenAIModelAdapter).
- Does NOT implement complete_with_tools(): exp_086's agent.py uses
  ReAct JSON only and never calls it.

Wire-level diff to log during smoke (PRIORITIES.md 158-176, 8 items):
  (1) API call diff: messages, tools (none), extra_body
  (2) Tool schema (n/a here, ReAct path)
  (3) Chat history thinking-strip (Jinja2 vs strip_thinking double/missing)
  (4) Response parse (content vs reasoning_content)
  (5) Sampling params preservation (max_tokens=32K, top_p=0.95,
      presence_penalty=1.0, enable_thinking=True, top_k=20, min_p=0)
  (6) Authentication: CF-Access-Client-Id / CF-Access-Client-Secret
      passed via generate_cfg extra_headers → _chat_complete_create **kwargs
  (7) Model name (qwen3.5-35b-a3b — Qwen-Agent registry alias check)
  (8) Tool execute return format (n/a; ReAct path returns str)
"""
from __future__ import annotations

from typing import Any

from qwen_agent.llm import get_chat_model

from kobushi_core.model import ModelMessage, strip_thinking


class QwenAgentModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int = 32768,
        enable_thinking: bool = True,
        top_p: float | None = 0.95,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        presence_penalty: float | None = 0.0,
        request_timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.extra_headers = dict(extra_headers or {})
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.request_timeout = request_timeout

    def _build_generate_cfg(self, *, enable_thinking: bool | None = None) -> dict[str, Any]:
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        cfg: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": extra_body,
            # Qwen-Agent OAI backend renames this to `timeout` before passing
            # to openai SDK, mirroring OpenAIModelAdapter(timeout=180).
            # Without this, Qwen-Agent has no timeout → HTTP 524 on busy vLLM.
            "request_timeout": self.request_timeout,
            # Disable Qwen-Agent base.py auto-injection of `random.randint(...)`
            # seed. OpenAIModelAdapter doesn't pass seed (= vLLM uses its own
            # session-RNG), so to match its behavior we explicitly set None.
            # Empirical: random-per-call seed in qwen-agent caused the 3 union
            # attempts to over-diverge (= more dup-cols, lower λ0.5 by ~0.04).
            # The `'seed' not in generate_cfg` check in qwen_agent/llm/base.py
            # treats None as "present" so it won't overwrite. openai client
            # then omits None-valued kwargs from the wire request body,
            # producing the same behavior as the OpenAIModelAdapter path.
            "seed": None,
        }
        if self.top_p is not None:
            cfg["top_p"] = self.top_p
        if self.presence_penalty is not None:
            cfg["presence_penalty"] = self.presence_penalty
        # CF Access headers: passed via generate_cfg extra_headers →
        # flows through Qwen-Agent's _chat_complete_create(**generate_cfg)
        # to openai client.chat.completions.create(extra_headers=...).
        # Smoke item (6): verify in vLLM access log that headers arrive.
        if self.extra_headers:
            cfg["extra_headers"] = dict(self.extra_headers)
        return cfg

    def _llm(self, *, enable_thinking: bool | None = None):
        # Lazy construction per call so that __init__ stays pickleable
        # (multiprocessing.spawn safe for 3-attempt subprocess).
        return get_chat_model({
            "model": self.model,
            "model_server": self.api_base,
            "api_key": self.api_key,
            "generate_cfg": self._build_generate_cfg(enable_thinking=enable_thinking),
        })

    def complete(self, messages: list[ModelMessage], *, enable_thinking: bool | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        qa_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        try:
            llm = self._llm(enable_thinking=enable_thinking)
            # chat(stream=False) returns List[Message|dict] — NOT an iterator.
            result = llm.chat(messages=qa_messages, stream=False)
        except Exception as exc:
            raise RuntimeError(f"Qwen-Agent model request failed: {exc}") from exc

        content = ""
        for msg in result:
            # Messages can be Qwen-Agent Message objects or plain dicts
            # depending on whether input was dict or Message type.
            if isinstance(msg, dict):
                role = msg.get("role", "")
                body = msg.get("content")
            else:
                role = getattr(msg, "role", "")
                body = getattr(msg, "content", None)
            if role != "assistant":
                continue
            if isinstance(body, str):
                content = body
            elif isinstance(body, list):
                # Some Qwen-Agent versions return content as list of
                # {"text": "..."} segments; concatenate text fields.
                content = "".join(
                    seg.get("text", "")
                    for seg in body
                    if isinstance(seg, dict)
                )

        if not content:
            raise RuntimeError("Qwen-Agent response missing assistant text content.")
        # Defense in depth: strip <think>...</think> if reasoning-parser
        # left any tokens (mirrors OpenAIModelAdapter behavior).
        return strip_thinking(content)
