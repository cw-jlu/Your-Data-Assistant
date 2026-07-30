"""Shared validation helpers for ETL stages."""

from __future__ import annotations

import re

from agents.etl._constants import (
    APPROX_PREFIXES,
    BOOL_FALSE,
    BOOL_TRUE,
    DIGIT_RUN,
    PLACEHOLDER_VALS,
)


def is_embedded_digit_run(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is a digit fragment of a larger token.

    Record IDs are standalone digit tokens.  Captures inside comma/dot
    formatted numbers ("8,408,793" → "408", "671,408.57" → "408"), longer
    digit runs ("2003" → "200"), or digit+letter words ("30th" → "30") are
    false IDs: they spawn spurious entity groups and hijack paragraphs away
    from the entity they belong to.
    """
    before = text[start - 1] if start > 0 else ""
    before2 = text[start - 2] if start > 1 else ""
    after = text[end] if end < len(text) else ""
    after2 = text[end + 1] if end + 1 < len(text) else ""
    if before.isdigit() or after.isdigit():
        return True
    if before in ",." and before2.isdigit():
        return True
    if after in ",." and after2.isdigit():
        return True
    # ASCII letters only: a CJK char right after the digits ("档案286的…")
    # is ordinary prose, not a fragment.
    return after.isascii() and after.isalpha()


def has_standalone_id(text: str, rid: str) -> bool:
    """``rid`` occurs in ``text`` as a standalone digit token (anchor verification)."""
    for m in re.finditer(re.escape(rid), text):
        if not is_embedded_digit_run(text, m.start(), m.end()):
            return True
    return False


def first_standalone_match(regex: re.Pattern[str], text: str) -> re.Match[str] | None:
    """First match whose captured digits form a standalone token."""
    group = 1 if regex.groups else 0
    for m in regex.finditer(text):
        if m.group(group) is None:
            continue
        if not is_embedded_digit_run(text, m.start(group), m.end(group)):
            return m
    return None


def contains_standalone_number(text: str) -> bool:
    return any(
        not is_embedded_digit_run(text, m.start(), m.end()) for m in DIGIT_RUN.finditer(text)
    )


def split_approx_tag(val: str) -> tuple[str, bool]:
    """Split a leading approximate-value marker off ``val``.

    Compression tags values stated approximately in the source ("约六百九十七万"
    → ``~6970000``) so the deterministic identity-repair pass can recompute
    them; everything downstream of that pass must never see the tag.
    """
    stripped = val.strip()
    for prefix in APPROX_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip(), True
    return stripped, False


def is_placeholder(val: str) -> bool:
    return val.lower().strip() in PLACEHOLDER_VALS or "(placeholder)" in val.lower()


def normalize_boolean(val: str) -> str | None:
    """Normalize a boolean-ish string to 'true'/'false', or None if unrecognizable."""
    low = val.strip().lower()
    if low in BOOL_TRUE:
        return "true"
    if low in BOOL_FALSE:
        return "false"
    return None
