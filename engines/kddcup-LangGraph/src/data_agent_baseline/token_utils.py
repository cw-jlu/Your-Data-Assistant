from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_TOKENIZER_DIR = _PROJECT_ROOT / "assets" / "huggingface" / "Qwen3.5-35B-A3B"

# 在任何 transformers import 之前设置，抑制无关警告
if _LOCAL_TOKENIZER_DIR.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_TOKENIZER_MODEL = str(_LOCAL_TOKENIZER_DIR) if _LOCAL_TOKENIZER_DIR.exists() else "Qwen/Qwen3.5-35B-A3B"
_TOKENIZER: PreTrainedTokenizerFast | None = None
_TOKENIZER_LOCK = threading.Lock()


def _get_tokenizer() -> PreTrainedTokenizerFast:
    global _TOKENIZER
    if _TOKENIZER is None:
        with _TOKENIZER_LOCK:
            if _TOKENIZER is None:
                from transformers import AutoTokenizer
                from transformers.utils import logging as transformers_logging

                transformers_logging.set_verbosity_error()
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

                _TOKENIZER = AutoTokenizer.from_pretrained(
                    _TOKENIZER_MODEL,
                    trust_remote_code=True,
                )
    return _TOKENIZER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokenizer = _get_tokenizer()
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) <= max_tokens:
        return text
    return tokenizer.decode(encoded[:max_tokens], skip_special_tokens=True)
