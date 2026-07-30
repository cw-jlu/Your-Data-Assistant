"""Paragraph grouping by entity — LLM batching with deterministic verification."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from agents.etl._anchors import RecordIdPattern, discover_record_id_patterns, paragraph_record_ids
from agents.etl._constants import (
    ENTITY_CHUNK_TOKEN_BUDGET,
    ENTITY_SEPARATOR,
    GROUP_LINE,
    GROUPING_BATCH_SIZE,
    GROUPING_SENTENCE_END,
    GROUPING_SYSTEM_PROMPT,
    GROUPING_TRUNCATE_CHARS,
    GROUPING_USER_TEMPLATE,
    GROUPS_NONE,
    INDEX_TOKEN,
    LLM_CALL_TIMEOUT,
    MAX_WORKERS,
    NONE_LINE,
)
from agents.etl._threading import submit_in_context
from agents.etl._types import (
    contains_standalone_number,
    has_standalone_id as _has_standalone_id,
)

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter, ModelMessage

logger = logging.getLogger(__name__)


def grouping_messages(numbered_block: str) -> list[ModelMessage]:
    """Build the grouping chat messages (shared with the validation script)."""
    from agents.llm.types import ModelMessage

    return [
        ModelMessage(role="system", content=GROUPING_SYSTEM_PROMPT),
        ModelMessage(role="user", content=GROUPING_USER_TEMPLATE.format(paragraphs=numbered_block)),
    ]


def split_prose_paragraphs(text: str) -> list[str]:
    """Split prose into paragraphs — THE shared 1-based index space."""
    return [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 30]


def grouping_head(para: str) -> str:
    """Paragraph opening rendered for grouping prompts."""
    flat = " ".join(para.split())
    end = 0
    while True:
        m = GROUPING_SENTENCE_END.search(flat, end)
        if not m:
            return flat
        end = m.end()
        head = flat[:end]
        if contains_standalone_number(head):
            return head


def make_grouping_batches(
    paras: list[str],
    batch_size: int = GROUPING_BATCH_SIZE,
    truncate: int = GROUPING_TRUNCATE_CHARS,
) -> list[tuple[list[int], str]]:
    """Contiguous balanced batches of (global 1-based indices, rendered numbered block)."""
    if not paras:
        return []
    n_batches = -(-len(paras) // batch_size)
    base, extra = divmod(len(paras), n_batches)
    batches: list[tuple[list[int], str]] = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < extra else 0)
        chunk = paras[start : start + size]
        indices = list(range(start + 1, start + 1 + size))
        lines: list[str] = []
        for idx, para in zip(indices, chunk, strict=True):
            head = grouping_head(para)
            lines.append(f"{idx}) {head}")
        batches.append((indices, "\n".join(lines)))
        start += size
    return batches


def parse_grouping_response(raw: str) -> tuple[dict[str, list[int]], list[int], bool, int]:
    """Parse one grouping response.

    Returns (groups rid->indices, none_indices, groups_none, unparsed_lines).
    """
    raw = raw.replace("```", "")
    groups: dict[str, list[int]] = {}
    none_indices: list[int] = []
    unparsed = 0
    if GROUPS_NONE.search(raw):
        return {}, [], True, 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if m := NONE_LINE.match(line):
            none_indices.extend(int(t) for t in INDEX_TOKEN.findall(m.group(1)))
            continue
        if m := GROUP_LINE.match(line):
            idxs = [int(t) for t in INDEX_TOKEN.findall(m.group(2))]
            if idxs:
                rids = [r.strip() for r in re.split(r"[,、]", m.group(1)) if r.strip()]
                for rid in rids:
                    groups.setdefault(rid, []).extend(idxs)
            continue
        unparsed += 1
    return groups, none_indices, False, unparsed


def verify_paragraph_groups(
    groups: dict[str, list[int]],
    paras: list[str],
    anchor_patterns: list[RecordIdPattern] | None = None,
) -> dict[str, list[int]] | None:
    """Deterministically verify LLM-proposed paragraph groups."""
    if anchor_patterns is None:
        anchor_patterns = discover_record_id_patterns(paras)
    valid_indices = set(range(1, len(paras) + 1))

    def has_anchor(text: str, rid: str) -> bool:
        return rid in paragraph_record_ids(text, anchor_patterns) or _has_standalone_id(text, rid)

    claims: dict[int, list[str]] = {}
    out_of_range = 0
    for rid, idxs in groups.items():
        for idx in idxs:
            if idx not in valid_indices:
                out_of_range += 1
                continue
            rids = claims.setdefault(idx, [])
            if rid not in rids:
                rids.append(rid)

    invalidated: set[int] = set()
    multi_member: list[int] = []
    for idx, rids in claims.items():
        if len(rids) == 1:
            continue
        anchored = [rid for rid in rids if has_anchor(paras[idx - 1], rid)]
        if not anchored:
            invalidated.add(idx)
            continue
        claims[idx] = anchored
        if len(anchored) > 1:
            multi_member.append(idx)

    cleaned: dict[str, list[int]] = {}
    for rid, idxs in groups.items():
        kept = [
            idx
            for idx in dict.fromkeys(idxs)
            if idx in claims and idx not in invalidated and rid in claims[idx]
        ]
        if kept:
            cleaned[rid] = kept

    anchors: dict[str, list[int]] = {}
    verified: dict[str, list[int]] = {}
    dropped_groups: list[str] = []
    for rid, idxs in cleaned.items():
        anch = [i for i in idxs if has_anchor(paras[i - 1], rid)]
        if anch:
            anchors[rid] = anch
            verified[rid] = idxs
        else:
            dropped_groups.append(rid)

    # Auto-share: when an anchor paragraph mentions another verified entity's
    # ID, grant that entity membership too (covers "units A and B are both
    # classified as X" patterns where the LLM only grouped the paragraph
    # under one entity).
    auto_shared = 0
    for rid, _idxs in list(verified.items()):
        for idx in anchors.get(rid, []):
            for other_rid in verified:
                if other_rid == rid:
                    continue
                if has_anchor(paras[idx - 1], other_rid) and idx not in verified[other_rid]:
                    verified[other_rid].append(idx)
                    anchors.setdefault(other_rid, []).append(idx)
                    auto_shared += 1
    if auto_shared:
        logger.info(
            "ETL compress: grouping auto-shared %d paragraph memberships "
            "for cross-entity declarations",
            auto_shared,
        )

    hijack_rejected: list[tuple[int, str, str]] = []
    hijack_reassign: dict[str, list[int]] = {}
    for rid, idxs in verified.items():
        for idx in idxs:
            if idx in anchors[rid]:
                continue
            other = next(
                (g for g in verified if g != rid and has_anchor(paras[idx - 1], g)),
                None,
            )
            if other is not None:
                hijack_rejected.append((idx, rid, other))
                hijack_reassign.setdefault(other, []).append(idx)
    rejected_idx = {i for i, _, _ in hijack_rejected}
    final = {
        rid: sorted(i for i in idxs if i not in rejected_idx) for rid, idxs in verified.items()
    }
    for rid, idxs in hijack_reassign.items():
        existing = set(final.get(rid, []))
        for idx in idxs:
            if idx not in existing:
                final.setdefault(rid, []).append(idx)
        if rid in final:
            final[rid] = sorted(final[rid])
    final = {rid: idxs for rid, idxs in final.items() if idxs}

    if out_of_range:
        logger.warning(
            "ETL compress: grouping dropped %d out-of-range paragraph indices", out_of_range
        )
    if multi_member:
        logger.info(
            "ETL compress: %d group paragraphs granted multi-entity membership: %s",
            len(multi_member),
            sorted(multi_member)[:10],
        )
    if invalidated:
        logger.warning(
            "ETL compress: grouping invalidated indices claimed by multiple groups "
            "with no anchoring claimant: %s",
            sorted(invalidated)[:10],
        )
    if dropped_groups:
        logger.warning(
            "ETL compress: grouping dropped %d hallucinated groups (no anchor paragraph): %s",
            len(dropped_groups),
            sorted(dropped_groups)[:10],
        )
    if hijack_rejected:
        logger.warning(
            "ETL compress: grouping hijack guard rejected %d attached members: %s",
            len(hijack_rejected),
            hijack_rejected[:10],
        )

    distinct = len(final)
    ids = [int(r) for r in final if r.isdigit()]
    tiny_dense_run = (
        bool(ids)
        and min(ids) <= 2
        and max(ids) <= 12
        and len(set(ids)) > (max(ids) - min(ids)) * 0.8
    )
    if distinct < 3 or tiny_dense_run:
        logger.info(
            "ETL compress: grouping gate rejected result (%d distinct entities, tiny_dense_run=%s)",
            distinct,
            tiny_dense_run,
        )
        return None
    logger.info(
        "ETL compress: LLM grouping verified %d entities over %d paragraphs "
        "(%d anchor-verified members)",
        distinct,
        sum(len(v) for v in final.values()),
        sum(len(v) for v in anchors.values()),
    )
    return final


def group_paragraphs_by_llm(
    adapter: ModelAdapter,
    paras: list[str],
    anchor_patterns: list[RecordIdPattern] | None = None,
) -> tuple[dict[str, list[int]] | None, bool]:
    """Number paragraphs, batch them to the LLM, verify the claimed groups."""
    if anchor_patterns is None:
        anchor_patterns = discover_record_id_patterns(paras)
    if len(paras) < 3:
        return None, True

    batches = make_grouping_batches(paras)

    def call(batch: tuple[list[int], str]) -> str:
        return adapter.complete(grouping_messages(batch[1])).content

    raw_results: list[str | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {submit_in_context(pool, call, batch): idx for idx, batch in enumerate(batches)}
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(batches)):
                idx = futs[fut]
                try:
                    raw_results[idx] = fut.result(timeout=LLM_CALL_TIMEOUT)
                except TimeoutError:
                    logger.warning("ETL compress: grouping batch %d timed out", idx)
                except Exception as exc:
                    logger.warning("ETL compress: grouping batch %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("ETL compress: grouping pool timed out; using partial batches")

    if all(raw is None for raw in raw_results):
        return None, False

    merged: dict[str, list[int]] = {}
    unparsed_total = 0
    for (indices, _), raw in zip(batches, raw_results, strict=True):
        if raw is None:
            continue
        groups, _nones, groups_none, unparsed = parse_grouping_response(raw)
        unparsed_total += unparsed
        if groups_none:
            continue
        batch_indices = set(indices)
        for rid, idxs in groups.items():
            kept = [i for i in idxs if i in batch_indices]
            if kept:
                merged.setdefault(rid, []).extend(kept)
    if unparsed_total:
        logger.info("ETL compress: grouping responses had %d unparsed lines", unparsed_total)

    return verify_paragraph_groups(merged, paras, anchor_patterns), True


def group_paragraphs_by_entity(
    text: str,
    adapter: ModelAdapter | None = None,
) -> tuple[dict[str, list[str]], list[str]] | None:
    """Group paragraphs by record ID — LLM grouping verified deterministically."""
    paras = split_prose_paragraphs(text)
    if not paras:
        return None
    anchor_patterns = discover_record_id_patterns(paras)

    if adapter is not None:
        verified, llm_responded = group_paragraphs_by_llm(adapter, paras, anchor_patterns)
        if verified:
            recovered = 0
            for i, para in enumerate(paras, start=1):
                for record_id in paragraph_record_ids(para, anchor_patterns):
                    if record_id in verified and i not in verified[record_id]:
                        verified[record_id].append(i)
                        recovered += 1
            if recovered:
                logger.info(
                    "ETL compress: deterministic recovery added %d paragraphs "
                    "missed by LLM grouping",
                    recovered,
                )
            ordered = sorted(verified.items(), key=lambda kv: kv[1][0])
            grouped = {rid: [paras[i - 1] for i in idxs] for rid, idxs in ordered}
            assigned = {i for idxs in verified.values() for i in idxs}
            unassigned = [
                para
                for i, para in enumerate(paras, start=1)
                if i not in assigned and contains_standalone_number(para)
            ]
            return grouped, unassigned
        if llm_responded:
            return None
        logger.warning(
            "ETL compress: all grouping batches failed; falling back to record-ID patterns"
        )

    groups: dict[str, list[str]] = {}
    leftover: list[str] = []
    tagged = 0
    for para in paras:
        record_ids = paragraph_record_ids(para, anchor_patterns)
        if record_ids:
            for record_id in record_ids:
                groups.setdefault(record_id, []).append(para)
            tagged += 1
        elif contains_standalone_number(para):
            leftover.append(para)

    if tagged < 3 or not groups:
        return None
    return groups, leftover


def entity_chunks_by_token_budget(
    entity_groups: dict[str, list[str]],
    token_budget: int = ENTITY_CHUNK_TOKEN_BUDGET,
) -> list[tuple[str, list[str]]]:
    """Pack complete entity contexts into chunks that fit a token budget."""
    from agents.llm.tokenizer import count_qwen_tokens

    entity_texts: list[tuple[str, str, int]] = []
    for rid, paras in entity_groups.items():
        full = f"[RECORD_ID: {rid}]\n" + "\n\n".join(paras)
        toks = count_qwen_tokens(full)
        entity_texts.append((rid, full, toks))

    chunks: list[tuple[str, list[str]]] = []
    current_parts: list[str] = []
    current_rids: list[str] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current_parts, current_rids, current_tokens
        if current_parts:
            chunks.append((ENTITY_SEPARATOR.join(current_parts), current_rids))
            current_parts = []
            current_rids = []
            current_tokens = 0

    for rid, full, toks in entity_texts:
        if toks >= token_budget:
            _flush()
            chunks.append((full, [rid]))
            continue

        if current_tokens + toks > token_budget and current_parts:
            _flush()

        current_parts.append(full)
        current_rids.append(rid)
        current_tokens += toks

    _flush()
    return chunks
