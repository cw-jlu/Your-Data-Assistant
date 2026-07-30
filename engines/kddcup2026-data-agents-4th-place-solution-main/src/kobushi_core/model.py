from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import APIError, OpenAI


# CF gateway returns transient 413 under burst load — body content is identical
# to requests that succeed seconds later, so retry-with-backoff recovers cleanly.
# OpenAI SDK's max_retries doesn't cover 413 (= treated as non-retriable client
# error), so we wrap calls with our own retry loop for 413 specifically.
_GATEWAY_413_RETRIES = 4
_GATEWAY_413_BASE_DELAY_S = 8.0


def _is_gateway_413(exc: BaseException) -> bool:
    """Detect transient 413 Payload Too Large from CF/upstream gateway.

    The CF response is HTML (= not the OpenAI JSON error envelope), so the
    SDK surfaces it as an APIError whose string contains '413 Payload Too Large'.
    """
    s = str(exc)
    return "413 Payload Too Large" in s or "413" in s and "cloudflare" in s.lower()


def _retry_on_413(fn):
    """Wrap a model call: retry up to N times with exponential backoff on 413."""
    last_exc = None
    for attempt in range(_GATEWAY_413_RETRIES + 1):
        try:
            return fn()
        except (APIError, RuntimeError) as exc:
            if not _is_gateway_413(exc):
                raise
            last_exc = exc
            if attempt == _GATEWAY_413_RETRIES:
                break
            delay = _GATEWAY_413_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 2)
            print(
                f"[model] 413 gateway error (attempt {attempt+1}/{_GATEWAY_413_RETRIES+1}), "
                f"retry in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# Transient failure patterns we retry on (= vLLM/CF gateway hiccups, not real errors).
# Distinct from _retry_on_413: 413 is body-too-large, this is empty/None response.
_TRANSIENT_RETRIES = 3
_TRANSIENT_BASE_DELAY_S = 5.0


def _retry_on_transient(fn):
    """Retry up to N times when the call returns None / empty choices / empty content.

    These manifest as `response.choices` being None or `content=''`, which we
    saw in bench v3 as 5 simultaneous task fails (= vLLM/CF gateway brief
    outage). Retrying with backoff recovers the request.
    """
    last_exc = None
    for attempt in range(_TRANSIENT_RETRIES + 1):
        try:
            response = fn()
        except (APIError, RuntimeError) as exc:
            last_exc = exc
            if attempt == _TRANSIENT_RETRIES:
                raise
            delay = _TRANSIENT_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1)
            print(
                f"[model] api error (attempt {attempt+1}): {str(exc)[:100]}, retry in {delay:.1f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay)
            continue

        # Validate response shape — retry on None or empty.
        if response is None:
            if attempt == _TRANSIENT_RETRIES:
                raise RuntimeError("Model response is None after retries.")
            delay = _TRANSIENT_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1)
            print(
                f"[model] response is None (attempt {attempt+1}), retry in {delay:.1f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay)
            continue
        choices = getattr(response, "choices", None) or []
        if not choices:
            if attempt == _TRANSIENT_RETRIES:
                raise RuntimeError("Model response missing choices after retries.")
            delay = _TRANSIENT_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1)
            print(
                f"[model] empty choices (attempt {attempt+1}), retry in {delay:.1f}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(delay)
            continue
        return response
    if last_exc:
        raise last_exc
    raise RuntimeError("Transient retry exhausted without success.")


# Qwen3.5 thinking content lives between <think>...</think> markers.
# vLLM's --reasoning-parser qwen3 normally strips this server-side and only
# returns the final answer in `msg.content`, but we add a defensive client-side
# strip in case of partial parses or fallback configurations. Per Qwen3.5
# official guidance, multi-turn conversation history should NOT carry forward
# thinking content from previous turns (degrades subsequent reasoning).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_DANGLING_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)


def strip_thinking(text: str | None) -> str:
    """Remove <think>...</think> blocks from a model response (defense in depth).

    Three patterns are handled:
    1. Balanced `<think>...</think>` blocks (paired, non-greedy across newlines)
    2. Dangling `<think>...` with no closing tag (max_tokens cut mid-thought)
    3. Orphan `</think>` with no opening tag — everything before it is treated
       as leaked thinking and dropped. This happens when vLLM's reasoning-parser
       consumes the opening `<think>` into reasoning_content but leaves the
       closing tag in content, or when an internal thinking_budget force-closes
       the thought block. Empirically common with Qwen3.5 + vLLM.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _DANGLING_THINK_RE.sub("", cleaned)
    # Orphan close tag: drop everything up to and including the last </think>.
    # Use rsplit so a single embedded </think> in legitimate later text isn't
    # confused with a stray opener — the *last* one is the actual boundary.
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1]
    return cleaned.lstrip()


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


@dataclass(frozen=True, slots=True)
class ToolCallInvocation:
    """A single tool call extracted from a function-calling response."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCallResponse:
    """Native function-calling response. Either text or tool_calls (or both)."""
    text: str | None  # assistant content (often the model's "thought")
    tool_calls: list[ToolCallInvocation] = field(default_factory=list)
    finish_reason: str = ""
    # Raw assistant message dict (for re-injecting into history with tool_calls)
    raw_assistant_message: dict[str, Any] | None = None


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        extra_headers: dict[str, str] | None = None,
        # Qwen3.5-35B-A3B official model card recommendations (2026-02-24):
        #   thinking mode (general):  T=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5
        #   thinking mode (precise):  T=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.0
        # Output length: standard queries 32K, complex benchmarks 81K.
        # DABench is complex (= multi-step ReAct + thinking budget), so default
        # to 81K to avoid "content=None" failures when thinking exhausts the
        # token budget. Adds ~20-30% per-step latency vs 32K but eliminates
        # ~10% fail rate observed in benches.
        # T=0 is not explicitly banned for Qwen3.5 (unlike Qwen3) but all
        # recommended sampling configs use T >= 0.6 — caller should override
        # via temperature= when running data-analysis "precise" tasks.
        max_tokens: int = 81920,
        enable_thinking: bool = True,
        top_p: float | None = 0.95,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        # presence_penalty: 0.0 matches Qwen3.5 official "thinking precise
        # coding/tool use" preset (DABench is closest to this category — exact
        # column names, exact SQL, etc., where penalizing repeated tokens hurts).
        # Use 1.5 only for "thinking general" tasks if loops become a problem.
        presence_penalty: float | None = 0.0,
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

    def _client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            default_headers=self.extra_headers,
            # max_retries=8 + the SDK's exponential backoff (default ~2s base, doubling)
            # tolerates upstream 502/503/504/429 from CF/vLLM during load spikes.
            max_retries=8,
            timeout=180.0,
        )

    def _sampling_kwargs(
        self,
        *,
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build the sampling kwargs (Qwen3.5 hybrid thinking model).

        max_tokens generously covers both thinking (<think>...</think>) content
        and the final assistant answer. Without it, the model can use up the
        small vLLM default budget on thinking and emit content=null.

        extra_body.chat_template_kwargs.enable_thinking explicitly enables the
        thinking mode (matches Qwen3.5 model card "extra_body" mechanism).
        Per-call override (enable_thinking arg, default None = use instance
        default) lets a caller flip thinking off for cheap tool-execution
        steps while keeping it on for plan/reasoning steps. This is the
        ONLY supported per-call thinking control on Qwen3.5: the model card
        explicitly notes that user-message `/think` and `/no_think` soft
        switches are NOT supported (Qwen3.5 differs from Qwen3 here).
        """
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": effective_max_tokens,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        # vLLM-specific sampling controls live under extra_body.
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        kwargs["extra_body"] = extra_body
        return kwargs

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        def _call():
            try:
                resp = self._client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    **self._sampling_kwargs(
                        enable_thinking=enable_thinking, max_tokens=max_tokens
                    ),
                )
            except APIError as exc:
                raise RuntimeError(f"Model request failed: {exc}") from exc
            # Validate content INSIDE _call so content=None (= thinking
            # exhausted token budget) becomes a retryable RuntimeError
            # caught by _retry_on_transient.
            choices = getattr(resp, "choices", None) or []
            if choices:
                c = choices[0].message.content
                if not isinstance(c, str):
                    raise RuntimeError(
                        "Model response missing text content (transient)."
                    )
            return resp

        # Two-layer retry: transient (= empty/None/missing-content) outermost,
        # 413 inner.
        response = _retry_on_transient(lambda: _retry_on_413(_call))

        choices = response.choices
        msg = choices[0].message
        content = msg.content
        # Capture the thinking trace. vLLM with the reasoning-parser routes
        # <think>...</think> to either `reasoning_content` (= OpenAI conv) or
        # `reasoning` (= gpuhost / older vLLM). Neither is exposed as a typed
        # SDK attribute, so read from Pydantic v2 model_extra.
        try:
            extra = getattr(msg, "model_extra", None) or {}
            self.last_reasoning_content = (
                extra.get("reasoning_content")
                or extra.get("reasoning")
                or getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
        except Exception:
            self.last_reasoning_content = ""
        # Defense in depth: strip <think>...</think> in case the vLLM
        # reasoning-parser left any (it normally does the strip server-side).
        return strip_thinking(content)

    def complete_with_logprobs(
        self,
        messages: list[ModelMessage],
        *,
        top_logprobs: int = 5,
        enable_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, float]]]:
        """Same as complete() but also returns per-output-token top-K logprobs.

        Used by AIMO3-faithful entropy-weighted voting (see exp_122 PoC):
        each output token's top_logprobs dict feeds mean-entropy estimation,
        which weights the candidate's vote (= weight = 1/entropy).

        Returns (text, list_of_top_logprobs_dicts). One dict per output token,
        keyed by token string -> log probability. Empty list when vLLM doesn't
        return logprobs (= some legacy server modes).
        """
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        def _call():
            try:
                resp = self._client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    **self._sampling_kwargs(
                        enable_thinking=enable_thinking, max_tokens=max_tokens
                    ),
                )
            except APIError as exc:
                raise RuntimeError(f"Model request failed: {exc}") from exc
            choices = getattr(resp, "choices", None) or []
            if choices:
                c = choices[0].message.content
                if not isinstance(c, str):
                    raise RuntimeError(
                        "Model response missing text content (transient)."
                    )
            return resp

        response = _retry_on_transient(lambda: _retry_on_413(_call))
        choices = response.choices
        content = choices[0].message.content

        # Parse OpenAI logprobs structure: choices[0].logprobs.content is a
        # list of TokenLogprobsEntry. Each entry has .top_logprobs (list of
        # {token, logprob}). We flatten to a list of {token: logprob} dicts.
        top_buf: list[dict[str, float]] = []
        lp = getattr(choices[0], "logprobs", None)
        if lp is not None:
            content_lp = getattr(lp, "content", None) or []
            for entry in content_lp:
                top_list = getattr(entry, "top_logprobs", None) or []
                top_dict: dict[str, float] = {}
                for cand in top_list:
                    tok = getattr(cand, "token", None)
                    val = getattr(cand, "logprob", None)
                    if tok is not None and val is not None:
                        top_dict[tok] = float(val)
                if top_dict:
                    top_buf.append(top_dict)

        return strip_thinking(content), top_buf

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "auto",
        enable_thinking: bool | None = None,
    ) -> ToolCallResponse:
        """Native function-calling completion.

        ``messages``: raw OpenAI-format messages (dicts with role/content/tool_calls/
            tool_call_id as appropriate). The caller is responsible for assembling
            them in the function-calling protocol — we don't transform here because
            the tool-result message shape (role="tool", tool_call_id=...) matters.

        ``tools``: list of OpenAI function definitions, e.g.
            [{"type": "function", "function": {"name": ..., "parameters": ...}}]

        Returns a ToolCallResponse that contains either text content (when the model
        chose to talk instead of calling a tool) or one-or-more parsed tool calls.
        Tool-call arguments are parsed from JSON; if parsing fails, we still return
        the tool name with arguments={} so the agent can decide what to do.
        """
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        def _call():
            try:
                return self._client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **self._sampling_kwargs(enable_thinking=enable_thinking),
                )
            except APIError as exc:
                raise RuntimeError(f"Model request failed: {exc}") from exc

        response = _retry_on_413(_call)

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        msg = choices[0].message
        finish_reason = choices[0].finish_reason or ""

        invocations: list[ToolCallInvocation] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                if not isinstance(args, dict):
                    args = {"_raw": args}
            except json.JSONDecodeError:
                args = {"_raw_arguments": tc.function.arguments}
            invocations.append(
                ToolCallInvocation(id=tc.id, name=tc.function.name, arguments=args)
            )

        # Reconstruct a serializable assistant message dict for history rebuild.
        # Qwen3.5 official: "historical model output should only include the
        # final output part and does not need to include the thinking content."
        # vLLM reasoning-parser does the strip server-side, but we apply
        # client-side strip too as defense in depth. Crucial for multi-turn
        # tool-call loops where each turn's history is fed back to the model.
        clean_content = strip_thinking(msg.content) if msg.content else None
        raw = {
            "role": "assistant",
            "content": clean_content,
        }
        if msg.tool_calls:
            raw["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]

        return ToolCallResponse(
            text=clean_content,
            tool_calls=invocations,
            finish_reason=finish_reason,
            raw_assistant_message=raw,
        )


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
