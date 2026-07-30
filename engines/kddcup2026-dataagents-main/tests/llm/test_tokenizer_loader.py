"""Tests for the Qwen3 tokenizer lazy-loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def test_get_tokenizer_returns_instance() -> None:
    from agents.llm.tokenizer import get_tokenizer

    tok = get_tokenizer()
    assert tok is not None
    ids = tok.encode("hello world", add_special_tokens=False).ids
    assert len(ids) > 0


def test_get_tokenizer_raises_when_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    import agents.llm.tokenizer as mod

    monkeypatch.setattr(mod, "_LOCAL_PATH", Path("/nonexistent/tokenizer.json"))
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: (_ for _ in ()).throw(OSError("fake"))
    )

    with pytest.raises(OSError, match="fake"):
        mod.get_tokenizer()


def test_get_tokenizer_caches() -> None:
    from agents.llm.tokenizer import get_tokenizer

    a = get_tokenizer()
    b = get_tokenizer()
    assert a is not None
    assert a is b


def test_count_qwen_tokens() -> None:
    from agents.llm.tokenizer import count_qwen_tokens

    assert count_qwen_tokens("hello world") > 0


def test_get_tokenizer_raises_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    import agents.llm.tokenizer as mod

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "tokenizers":
            raise ImportError("fake")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="fake"):
        mod.get_tokenizer()
