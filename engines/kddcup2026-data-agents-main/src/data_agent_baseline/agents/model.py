from __future__ import annotations

import json as _json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI


# Retry policy for transient endpoint glitches (network blip,
# 429/5xx). v4 50-task showed 1/50 outright loss to "Connection error"
# at step 0 — the organizer endpoint is likely more stable than our
# self-hosted DGX vLLM, but expanding retries cheaply hardens us
# against either side flickering. Backoff list = total ≤ 7s under the
# 10-minute task timeout.
_TRANSIENT_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0)


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


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    """OpenAI-compatible chat adapter with optional JSON-mode probe.

    On the first complete() call we probe the endpoint with a trivial
    `response_format={"type":"json_object"}` request. If the endpoint
    accepts it AND returns a parseable JSON object, every subsequent
    call passes response_format. If the probe fails for any reason we
    silently fall back to fenced-block parsing — the existing
    parse_model_step path already tolerates either shape.

    Set enable_json_mode=False to skip the probe entirely (useful for
    endpoints known to misbehave or for fast smoke tests).
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        enable_json_mode: bool = True,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.enable_json_mode = enable_json_mode
        # v5 S-1: Per-task deterministic seed. When set, every
        # chat.completions request includes this seed so the endpoint
        # selects the same sampling path across runs. vLLM ≥ 0.6 honors
        # it for nucleus / top-k sampling. None = legacy behavior (no
        # seed passed; endpoint picks its own).
        self.seed = seed
        # None = not yet probed; True/False = probe verdict.
        self._json_mode_supported: bool | None = None
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            # v_next: 240 → 200. v6 ship added explicit timeout (was httpx
            # default 60s), but max_workers also rose 4 → 6 in v_next.
            # 200s keeps a comfortable headroom over typical LLM round-trip
            # (~5-20s) but stays BELOW the smallest per-task subprocess
            # timeout (easy=120s — actually 200>120; we accept the LLM
            # timeout firing first on easy because easy LLM calls rarely
            # take >60s anyway). The per-task subprocess timeout remains
            # the hard ceiling that recovers a stuck task.
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=200.0,
            )
        return self._client

    def _probe_json_mode(self) -> bool:
        """Send a tiny request with response_format and verify the answer parses."""
        client = self._get_client()
        probe_messages = [
            {"role": "system", "content": "Respond only with a JSON object. No prose."},
            {"role": "user", "content": 'Respond with {"ok": true} and nothing else.'},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=probe_messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=32,
            )
        except Exception:  # noqa: BLE001 — any error means probe fails
            return False

        choices = response.choices or []
        if not choices:
            return False
        content = choices[0].message.content
        if not isinstance(content, str):
            return False
        try:
            payload = _json.loads(content)
        except (ValueError, TypeError):
            return False
        return isinstance(payload, dict)

    def _ensure_json_mode_decided(self) -> None:
        if self._json_mode_supported is not None:
            return
        if not self.enable_json_mode or not self.api_key:
            self._json_mode_supported = False
            return
        self._json_mode_supported = self._probe_json_mode()

    def complete(self, messages: list[ModelMessage]) -> str:
        # Single-model policy: refuse to run unless every piece of the
        # qwen contract is supplied via env (eval) or YAML (local). We
        # do NOT default to an external provider so that a missing env
        # var fails loudly instead of silently calling OpenAI.
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")
        if not self.api_base:
            raise RuntimeError("Missing model API base URL in config.agent.api_base.")
        if not self.model:
            raise RuntimeError("Missing model name in config.agent.model.")

        self._ensure_json_mode_decided()
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
        }
        if self._json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            kwargs["seed"] = self.seed

        response = self._call_with_retry(client, kwargs)

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content

    # ---- internals -------------------------------------------------------

    def _call_with_retry(self, client: OpenAI, kwargs: dict[str, Any]) -> Any:
        """One JSON-mode demote on hard 4xx, plus N transient-error retries.

        Order of fallbacks:
          1. Try as-is. If it succeeds, done.
          2. If APIError mentions response_format and JSON-mode is on, demote
             once and retry without response_format (legacy behavior).
          3. If the failure is transient (APIConnectionError / APITimeoutError /
             5xx / 429), back off and retry up to len(_TRANSIENT_RETRY_BACKOFF_SECONDS) times.
          4. If all retries exhausted, raise RuntimeError with the last error.
        """
        try:
            return client.chat.completions.create(**kwargs)
        except APIError as exc:
            if "response_format" in kwargs and self._json_mode_supported:
                self._json_mode_supported = False
                kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                try:
                    return client.chat.completions.create(**kwargs)
                except APIError as exc2:
                    return self._retry_loop(client, kwargs, exc2)
            return self._retry_loop(client, kwargs, exc)

    def _retry_loop(
        self,
        client: OpenAI,
        kwargs: dict[str, Any],
        last_exc: BaseException,
    ) -> Any:
        if not self._is_transient(last_exc):
            raise RuntimeError(f"Model request failed: {last_exc}") from last_exc
        for delay in _TRANSIENT_RETRY_BACKOFF_SECONDS:
            time.sleep(delay)
            try:
                return client.chat.completions.create(**kwargs)
            except APIError as exc:
                last_exc = exc
                if not self._is_transient(exc):
                    raise RuntimeError(f"Model request failed: {exc}") from exc
        raise RuntimeError(f"Model request failed: {last_exc}") from last_exc

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        """Heuristic: connection / timeout errors and 429/5xx are retriable."""
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True
        # OpenAI SDK sometimes wraps connection errors as plain APIError —
        # fall back to message inspection for that path.
        message = str(exc).lower()
        return "connection" in message or "timeout" in message or "timed out" in message


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
