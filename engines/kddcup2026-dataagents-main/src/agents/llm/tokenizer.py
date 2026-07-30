"""Lazy-load Qwen3 tokenizer singleton for accurate token counting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_LOCAL_PATH = Path(__file__).parent / "tokenizer.json"
_HF_REPO = "Qwen/Qwen3.5-35B-A3B"
_HF_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
_HF_FILENAME = "tokenizer.json"

_loaded: bool = False
_tokenizer: Any = None


def _resolve_tokenizer_path() -> str:
    """Return path to tokenizer.json: local file first, then HF download."""
    if _LOCAL_PATH.exists():
        return str(_LOCAL_PATH)
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]

    return hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME, revision=_HF_REVISION)


def get_tokenizer() -> Any:
    """Return a ``tokenizers.Tokenizer`` instance.

    Loads from a bundled ``tokenizer.json`` if present (Docker image), otherwise
    downloads from HuggingFace on first call.  Raises on failure.
    """
    global _loaded, _tokenizer
    if _loaded:
        return _tokenizer

    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    path = _resolve_tokenizer_path()
    _tokenizer = Tokenizer.from_file(path)
    _loaded = True
    return _tokenizer


def count_qwen_tokens(text: str) -> int:
    """Count text tokens with the configured Qwen3 tokenizer."""
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)
