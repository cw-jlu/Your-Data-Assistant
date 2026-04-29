"""
此模块定义了模型适配器接口，目前支持 OpenAI 兼容的 API 调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    reflection: str
    data_sufficient: bool
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_tokens: int | None = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._local_embedder = None

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        try:
            kwargs = {}
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
                
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature,
                **kwargs
            )
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._local_embedder is None:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            from sentence_transformers import SentenceTransformer
            self._local_embedder = SentenceTransformer(self.embedding_model)

        try:
            safe_texts = [t.strip()[:2000] for t in texts]
            if not safe_texts:
                return []
            embeddings = self._local_embedder.encode(safe_texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as exc:
            raise RuntimeError(f"Local embedding failed: {exc}") from exc


class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 128 for _ in texts]
