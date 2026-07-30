"""Shared document-local record-anchor discovery."""

from __future__ import annotations

import re
from collections.abc import Sequence

from agents.etl._constants import RECORD_ID_INLINE
from agents.etl._types import is_embedded_digit_run

RecordIdPattern = tuple[str, re.Pattern[str]]

AUTO_ANCHOR_LABEL_ID = re.compile(
    r"\b(?P<label>(?:[A-Za-z][A-Za-z-]*\s+){0,5}[A-Za-z][A-Za-z-]*)"
    r"\s*(?:\(?(?:ID|id|Id)[：:]\s*)?[：:#\-]?\s*(?P<id>\d{1,8})",
    re.IGNORECASE,
)
AUTO_ANCHOR_CJK_LABEL_ID = re.compile(r"(?P<label>[\u4e00-\u9fff]{1,8})\s*(?P<id>\d{1,8})")
AUTO_ANCHOR_CJK_ID_LABEL = re.compile(
    r"(?:第\s*)?(?P<id>\d{1,8})\s*号\s*(?P<label>[\u4e00-\u9fff]{1,8})"
)
AUTO_ANCHOR_MAX_PATTERNS = 8
AUTO_ANCHOR_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "id",
        "of",
        "on",
        "the",
        "this",
        "to",
        "under",
        "with",
    }
)
AUTO_ANCHOR_FIELD_LABELS = frozenset(
    {
        "age",
        "amount",
        "code",
        "count",
        "date",
        "height",
        "hospital id",
        "internal code",
        "identifier",
        "management code",
        "rank",
        "ratio",
        "ref",
        "report",
        "stay",
        "ticker",
        "total",
        "trading code",
        "value",
        "weight",
        "year",
        "year end",
        "year-end",
    }
)
AUTO_ANCHOR_ENTITY_NOUNS = frozenset(
    {
        "asset",
        "code",
        "docket",
        "entry",
        "entity",
        "instrument",
        "item",
        "ledger",
        "ref",
        "reference",
        "registry",
        "stay",
        "unit",
    }
)
AUTO_ANCHOR_TRAILING_FIELD_WORDS = frozenset(
    {
        "closed",
        "confirmed",
        "ended",
        "estimated",
        "filed",
        "reached",
        "recorded",
        "reported",
        "stood",
        "valued",
    }
)
AUTO_ANCHOR_GENERIC_SINGLE_WORD_LABELS = frozenset({"ref", "stay", "unit"})
AUTO_ANCHOR_CJK_FIELD_LABELS = frozenset(
    {
        "码",
        "编码",
        "代码",
        "证券代码",
        "交易代码",
        "挂牌代码",
        "内部编码",
        "日期",
        "时间",
        "年份",
        "排名",
        "数量",
        "规模",
        "净值",
        "比例",
        "金额",
        "年龄",
        "高度",
        "体重",
        "亿元",
        "万元",
    }
)
AUTO_ANCHOR_CJK_FIELD_CONTEXTS = (
    "内部",
    "管理",
    "完整",
    "证券",
    "交易",
    "挂牌",
    "市场",
    "识别",
)
AUTO_ANCHOR_CJK_ENTITY_SUFFIXES = (
    "档案编号",
    "备案号",
    "编号",
    "索引",
    "标的",
    "单元",
    "资产项",
    "资产",
    "工具",
    "样本",
    "条目",
    "案号",
)
AUTO_ANCHOR_CJK_TRAILING_PARTICLES = ("为", "是")


def record_ids_for_pattern(regex: re.Pattern[str], text: str) -> list[str]:
    """Return standalone record ids captured by one anchor regex."""
    record_ids: list[str] = []
    for match in regex.finditer(text):
        group_name = next(
            (
                name
                for name, value in match.groupdict().items()
                if name.startswith("id") and value is not None
            ),
            "",
        )
        record_id = match.group(group_name) if group_name else match.group(1)
        start = match.start(group_name) if group_name else match.start(1)
        end = match.end(group_name) if group_name else match.end(1)
        if is_embedded_digit_run(text, start, end):
            continue
        if record_id not in record_ids:
            record_ids.append(record_id)
    return record_ids


def _normalized_auto_anchor_labels(label: str) -> list[str]:
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z-]*", label)
        if word.lower() not in AUTO_ANCHOR_STOPWORDS
    ]
    if words and words[-1] in AUTO_ANCHOR_TRAILING_FIELD_WORDS:
        return []
    labels: list[str] = []
    for width in range(1, min(4, len(words)) + 1):
        candidate = " ".join(words[-width:])
        if candidate in AUTO_ANCHOR_FIELD_LABELS:
            continue
        if candidate.split()[-1] not in AUTO_ANCHOR_ENTITY_NOUNS:
            continue
        if candidate not in labels:
            labels.append(candidate)
    return labels


def _normalized_cjk_auto_anchor_labels(label: str) -> list[str]:
    normalized = label.strip()
    while normalized.endswith(AUTO_ANCHOR_CJK_TRAILING_PARTICLES):
        normalized = normalized[:-1].strip()
    if normalized.endswith(("编号", "档案编号", "代码", "编码")) and any(
        context in normalized for context in AUTO_ANCHOR_CJK_FIELD_CONTEXTS
    ):
        return []

    labels: list[str] = []
    for suffix in AUTO_ANCHOR_CJK_ENTITY_SUFFIXES:
        if normalized.endswith(suffix) and suffix not in AUTO_ANCHOR_CJK_FIELD_LABELS:
            labels.append(suffix)
    return labels


def _auto_anchor_score(label: str, counts: dict[str, int], leading: int, script: str) -> float:
    total = sum(counts.values())
    repeated = sum(1 for count in counts.values() if count > 1)
    script_bonus = 0.5 if script == "cjk" else 0.0
    length_penalty = (
        len(label.split()) * 0.05 if script == "latin" else max(len(label) - 2, 0) * 0.1
    )
    return (repeated * 4.0) + total + (leading * 2.0) + script_bonus - length_penalty


def _auto_anchor_is_supported(counts: dict[str, int], leading: int) -> bool:
    distinct = len(counts)
    total = sum(counts.values())
    repeated = sum(1 for count in counts.values() if count > 1)
    leading_ratio = leading / total if total else 0.0
    one_row_per_entity_table = distinct >= 10 and leading_ratio >= 0.45
    repeated_entity_sections = repeated >= 3 and total >= distinct + 3
    return distinct >= 3 and (repeated_entity_sections or one_row_per_entity_table)


def _auto_anchor_is_suffix(shorter: str, longer: str, script: str) -> bool:
    if shorter == longer:
        return False
    if script == "latin":
        return longer.endswith(f" {shorter}")
    return longer.endswith(shorter)


def _auto_anchor_support_covers(
    broader_counts: dict[str, int], narrower_counts: dict[str, int]
) -> bool:
    return set(narrower_counts) <= set(broader_counts) and sum(broader_counts.values()) >= sum(
        narrower_counts.values()
    )


def _prune_auto_anchor_labels(
    labels: list[str],
    label_counts: dict[str, dict[str, int]],
    label_scripts: dict[str, str],
) -> list[str]:
    dropped: set[str] = set()
    for label in labels:
        if label in dropped:
            continue
        script = label_scripts[label]
        for other in labels:
            if label == other or other in dropped or label_scripts[other] != script:
                continue
            if not _auto_anchor_is_suffix(label, other, script):
                continue

            label_counts_for_ids = label_counts[label]
            other_counts_for_ids = label_counts[other]
            if script == "latin":
                if (
                    len(label.split()) == 1
                    and label in AUTO_ANCHOR_GENERIC_SINGLE_WORD_LABELS
                    and _auto_anchor_support_covers(other_counts_for_ids, label_counts_for_ids)
                ):
                    dropped.add(label)
                    break
                if _auto_anchor_support_covers(label_counts_for_ids, other_counts_for_ids):
                    dropped.add(other)
            elif _auto_anchor_support_covers(label_counts_for_ids, other_counts_for_ids):
                dropped.add(other)
    return [label for label in labels if label not in dropped]


def _compile_auto_anchor_pattern(label: str, script: str) -> re.Pattern[str]:
    if script == "cjk":
        return re.compile(
            rf"(?:{re.escape(label)}\s*(?:为|是)?\s*"
            rf"(?:\(?(?:ID|id|Id)[：:]\s*)?[：:#\-]?\s*(?P<id_before>\d{{1,8}})"
            rf"|(?:第\s*)?(?P<id_after>\d{{1,8}})\s*号\s*[\u4e00-\u9fff]{{0,6}}"
            rf"{re.escape(label)})"
        )
    return re.compile(
        rf"\b{re.escape(label)}\s*"
        rf"(?:\(?(?:ID|id|Id)[：:]\s*)?[：:#\-]?\s*(?P<id>\d{{1,8}})",
        re.IGNORECASE,
    )


def discover_record_id_patterns(paragraphs: Sequence[str]) -> list[RecordIdPattern]:
    """Discover known and auto-labeled record-id patterns from document paragraphs."""
    label_counts: dict[str, dict[str, int]] = {}
    label_leading_counts: dict[str, int] = {}
    label_scripts: dict[str, str] = {}

    def add_candidate(label: str, script: str, record_id: str, start: int) -> None:
        label_counts.setdefault(label, {})
        label_counts[label][record_id] = label_counts[label].get(record_id, 0) + 1
        label_scripts[label] = script
        if start <= 80:
            label_leading_counts[label] = label_leading_counts.get(label, 0) + 1

    for text in paragraphs:
        for match in AUTO_ANCHOR_LABEL_ID.finditer(text):
            record_id = match.group("id")
            if is_embedded_digit_run(text, match.start("id"), match.end("id")):
                continue
            for label in _normalized_auto_anchor_labels(match.group("label")):
                add_candidate(label, "latin", record_id, match.start())
        for match in AUTO_ANCHOR_CJK_LABEL_ID.finditer(text):
            record_id = match.group("id")
            if is_embedded_digit_run(text, match.start("id"), match.end("id")):
                continue
            for label in _normalized_cjk_auto_anchor_labels(match.group("label")):
                add_candidate(label, "cjk", record_id, match.start())
        for match in AUTO_ANCHOR_CJK_ID_LABEL.finditer(text):
            record_id = match.group("id")
            if is_embedded_digit_run(text, match.start("id"), match.end("id")):
                continue
            for label in _normalized_cjk_auto_anchor_labels(match.group("label")):
                add_candidate(label, "cjk", record_id, match.start())

    candidates: list[tuple[float, str]] = []
    for label, counts in label_counts.items():
        leading = label_leading_counts.get(label, 0)
        if not _auto_anchor_is_supported(counts, leading):
            continue
        score = _auto_anchor_score(label, counts, leading, label_scripts.get(label, "latin"))
        candidates.append((score, label))

    if not candidates:
        return [("known", RECORD_ID_INLINE)]

    ordered_labels = [label for _score, label in sorted(candidates, reverse=True)]
    selected_labels = _prune_auto_anchor_labels(ordered_labels, label_counts, label_scripts)[
        :AUTO_ANCHOR_MAX_PATTERNS
    ]
    patterns: list[RecordIdPattern] = [("known", RECORD_ID_INLINE)]
    patterns.extend(
        (
            f"auto:{label}",
            _compile_auto_anchor_pattern(label, label_scripts.get(label, "latin")),
        )
        for label in selected_labels
    )
    return patterns


def paragraph_record_ids(text: str, patterns: Sequence[RecordIdPattern]) -> list[str]:
    """Return unique record ids found in ``text`` by the shared anchor patterns."""
    record_ids: list[str] = []
    for _name, pattern in patterns:
        for record_id in record_ids_for_pattern(pattern, text):
            if record_id not in record_ids:
                record_ids.append(record_id)
    return record_ids
