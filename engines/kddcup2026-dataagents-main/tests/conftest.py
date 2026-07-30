from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_tokenizer_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset tokenizer singleton and keep unit tests independent of HuggingFace."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
    from tokenizers.models import WordLevel  # type: ignore[import-untyped]
    from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]

    import agents.llm.tokenizer as mod

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_path))

    mod._loaded = False
    mod._tokenizer = None
    monkeypatch.setattr(mod, "_LOCAL_PATH", tokenizer_path)
