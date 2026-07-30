from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

AZURE_API_VERSION = "2025-10-15"
AZURE_ENDPOINT_ENV = "AZURE_SPEECH_ENDPOINT"
AZURE_KEY_ENV = "AZURE_SPEECH_KEY"
AZURE_LOCALES = ("zh-CN", "en-US")


class _HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class _HttpClient(Protocol):
    def post(self, *args: Any, **kwargs: Any) -> _HttpResponse: ...


class AzureFastTranscriptionError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _dotenv_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value.strip() or None
    return None


def resolve_azure_credentials(project_root: Path) -> tuple[str, str]:
    endpoint = os.environ.get(AZURE_ENDPOINT_ENV, "").strip() or _dotenv_value(
        project_root / ".env", AZURE_ENDPOINT_ENV
    )
    key = os.environ.get(AZURE_KEY_ENV, "").strip() or _dotenv_value(
        project_root / ".env", AZURE_KEY_ENV
    )
    missing = [
        name
        for name, value in ((AZURE_ENDPOINT_ENV, endpoint), (AZURE_KEY_ENV, key))
        if not value
    ]
    if missing:
        raise AzureFastTranscriptionError(
            "Missing Azure Speech credentials: "
            + ", ".join(missing)
            + ". Set them in the environment or project .env before generating gold."
        )
    parsed = urlparse(str(endpoint))
    if parsed.scheme != "https" or not parsed.netloc:
        raise AzureFastTranscriptionError(
            f"{AZURE_ENDPOINT_ENV} must be a full HTTPS endpoint, for example "
            "https://<region>.api.cognitive.microsoft.com."
        )
    return str(endpoint).rstrip("/"), str(key)


def endpoint_host(endpoint: str) -> str:
    return urlparse(endpoint).netloc


class AzureFastTranscriptionClient:
    def __init__(
        self,
        *,
        endpoint: str,
        key: str,
        api_version: str = AZURE_API_VERSION,
        client: _HttpClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.api_version = api_version
        self._client = client
        self._sleep = sleep
        self.max_retries = max_retries

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        client = self._client
        owns_client = client is None
        if client is None:
            try:
                import httpx
            except ImportError as exc:
                raise RuntimeError(
                    "httpx is required for Azure Fast Transcription. "
                    "Install the project dependencies first."
                ) from exc
            client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))

        try:
            return self._transcribe_with_client(client, audio_path)
        finally:
            if owns_client:
                close = getattr(client, "close", None)
                if callable(close):
                    close()

    def _transcribe_with_client(
        self,
        client: _HttpClient,
        audio_path: Path,
    ) -> dict[str, Any]:
        url = (
            f"{self.endpoint}/speechtotext/transcriptions:transcribe"
            f"?api-version={self.api_version}"
        )
        definition = {
            "locales": list(AZURE_LOCALES),
            "channels": [0],
            "profanityFilterMode": "Masked",
        }

        for attempt in range(self.max_retries + 1):
            with audio_path.open("rb") as audio_handle:
                response = client.post(
                    url,
                    headers={"Ocp-Apim-Subscription-Key": self.key},
                    data={"definition": json.dumps(definition, ensure_ascii=False)},
                    files={"audio": (audio_path.name, audio_handle, "audio/wav")},
                )
            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise AzureFastTranscriptionError(
                        "Azure Fast Transcription returned a non-object JSON response."
                    )
                return payload

            retryable = response.status_code == 429 or response.status_code >= 500
            message = _safe_error_message(response)
            if retryable and attempt < self.max_retries:
                self._sleep(float(2**attempt))
                continue
            raise AzureFastTranscriptionError(
                f"Azure Fast Transcription failed with HTTP {response.status_code}: {message}",
                status_code=response.status_code,
                retryable=retryable,
            )

        raise AssertionError("Azure retry loop exited unexpectedly.")


def _safe_error_message(response: _HttpResponse) -> str:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:500]
        if payload.get("message"):
            return str(payload["message"])[:500]
    text = str(getattr(response, "text", "")).strip()
    return text[:500] or "no response message"


def extract_azure_reference(payload: dict[str, Any]) -> tuple[str, str]:
    combined = payload.get("combinedPhrases")
    combined_texts = [
        str(item.get("text", "")).strip()
        for item in combined or []
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]
    phrases = [item for item in payload.get("phrases") or [] if isinstance(item, dict)]
    if combined_texts:
        text = " ".join(combined_texts)
    else:
        text = " ".join(
            str(item.get("text", "")).strip()
            for item in phrases
            if str(item.get("text", "")).strip()
        )
    if not text:
        raise AzureFastTranscriptionError("Azure response contains no transcript text.")

    locale_durations: dict[str, float] = defaultdict(float)
    for phrase in phrases:
        locale = str(phrase.get("locale", "")).strip()
        if not locale:
            continue
        duration = phrase.get("durationMilliseconds", 0)
        try:
            weight = max(float(duration), 1.0)
        except (TypeError, ValueError):
            weight = 1.0
        locale_durations[locale] += weight
    if not locale_durations:
        raise AzureFastTranscriptionError("Azure response contains no phrase locale.")
    locale = max(locale_durations.items(), key=lambda item: (item[1], item[0]))[0]
    if not (locale.lower().startswith("zh") or locale.lower().startswith("en")):
        raise AzureFastTranscriptionError(f"Unsupported Azure reference locale: {locale}")
    return text, locale
