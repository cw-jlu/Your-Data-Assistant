"""Validate LLM-driven paragraph grouping against real benchmark documents.

Drives the PRODUCTION grouping components from ``agents.etl._grouping``
(prompt template, batcher, response parser, deterministic verifier) against
real documents, keeping only the reporting layer local.  For each document
this script reports:

- baseline: how many paragraphs the builtin RECORD_ID_INLINE regex tags;
- LLM grouping quality: verified groups, anchor-verification rate;
- agreement with the regex baseline on paragraphs both methods tag
  (the regex is a high-precision pseudo-ground-truth where it matches);
- gap rescue: paragraphs the builtin regex misses (e.g. "registered as 60")
  that the LLM groups AND deterministic verification confirms;
- the distinctness gate verdict (anti time-series-collapse).

Layer-by-layer rejection diagnostics (hallucinated groups, duplicated
indices, hijack rejections, out-of-range claims) are logged by the
production verifier itself — the script enables INFO logging to surface
them.

Usage::

    uv run python scripts/validate_llm_grouping.py \
        --config configs/react.local.yaml \
        [doc.md ...] [--batch-size 40] [--truncate 300] [--dry-run] \
        [--expected 50] [--json out.json]

With no doc arguments, defaults to the three task_59 markdown prose docs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agents.config import load_app_config
from agents.etl._constants import RECORD_ID_INLINE
from agents.etl._grouping import (
    GROUPING_BATCH_SIZE,
    GROUPING_TRUNCATE_CHARS,
    grouping_messages,
    make_grouping_batches,
    parse_grouping_response,
    split_prose_paragraphs,
    verify_paragraph_groups,
)
from agents.etl._types import (
    first_standalone_match,
    has_standalone_id as _has_standalone_id,
)
from agents.llm.openai import OpenAIModelAdapter, resolve_backend_kind
from agents.llm.tokenizer import count_qwen_tokens

DEFAULT_DOCS = [
    "data/demo_samples_phase2/input/task_59/context/doc/ed_otherdepositorycorpbs.md",
    "data/demo_samples_phase2/input/task_59/context/doc/ed_productexportimport.md",
    "data/demo_samples_phase2/input/task_59/context/doc/in_meansofproductionpi.md",
]


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from agents.etl._pdf import pdf_to_markdown

        return pdf_to_markdown(path)
    return path.read_text(encoding="utf-8")


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def baseline_assignments(paras: list[str]) -> dict[int, str]:
    """1-based index -> rid for paragraphs the builtin regex tags."""
    out: dict[int, str] = {}
    for i, para in enumerate(paras, start=1):
        m = first_standalone_match(RECORD_ID_INLINE, para)
        if m:
            out[i] = m.group(1)
    return out


def build_adapter(config_path: Path) -> OpenAIModelAdapter:
    cfg = load_app_config(config_path)
    resolved = resolve_backend_kind(cfg.agent.backend_kind, cfg.agent.api_base)
    is_deepseek = cfg.agent.thinking_mode is not None or resolved == "deepseek"
    return OpenAIModelAdapter(
        model=cfg.agent.model,
        api_base=cfg.agent.api_base,
        api_key=cfg.agent.api_key,
        temperature=0.1,
        enable_thinking=None if is_deepseek else False,
        thinking_mode="disabled" if is_deepseek else None,
        backend_kind=cfg.agent.backend_kind,
        max_retries=2,
        timeout=300.0,
    )


def run_document(
    adapter: OpenAIModelAdapter | None,
    path: Path,
    batch_size: int,
    truncate: int,
    dry_run: bool,
    workers: int,
) -> dict:
    text = load_document(path)
    paras = split_prose_paragraphs(text)
    baseline = baseline_assignments(paras)
    batches = make_grouping_batches(paras, batch_size=batch_size, truncate=truncate)

    print(f"\n=== {path.name} ===")
    print(f"paragraphs: {len(paras)} | batches: {len(batches)}")
    print(
        f"baseline (builtin regex): tagged {len(baseline)} paras, "
        f"{len(set(baseline.values()))} entities"
    )
    if dry_run:
        print("\n--- first prompt preview ---")
        for msg in grouping_messages(batches[0][1]):
            print(f"[{msg.role}]")
            print(_message_content_text(msg.content)[:1500])
        return {"doc": path.name, "dry_run": True}

    assert adapter is not None

    def call(batch: tuple[list[int], str]) -> tuple[str, float, int, int]:
        messages = grouping_messages(batch[1])
        t0 = time.time()
        response = adapter.complete(messages)
        return (
            response.content,
            time.time() - t0,
            sum(count_qwen_tokens(_message_content_text(m.content)) for m in messages),
            count_qwen_tokens(response.content),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(call, batches))

    # Merge with the same per-batch range filter as group_paragraphs_by_llm.
    merged: dict[str, list[int]] = {}
    none_indices: list[int] = []
    groups_none_batches = 0
    unparsed_total = 0
    wall = 0.0
    tok_in = tok_out = 0
    for (indices, _), (raw, dt, ti, to) in zip(batches, results, strict=True):
        wall = max(wall, dt)  # batches ran in parallel
        tok_in += ti
        tok_out += to
        groups, nones, gnone, unparsed = parse_grouping_response(raw)
        unparsed_total += unparsed
        if gnone:
            groups_none_batches += 1
            continue
        none_indices.extend(nones)
        batch_indices = set(indices)
        for rid, idxs in groups.items():
            kept = [i for i in idxs if i in batch_indices]
            if kept:
                merged.setdefault(rid, []).extend(kept)

    verified = verify_paragraph_groups(merged, paras)
    gate_pass = verified is not None
    final: dict[str, list[int]] = verified or {}
    anchors = {
        rid: [i for i in idxs if _has_standalone_id(paras[i - 1], rid)]
        for rid, idxs in final.items()
    }
    anchor_members = sum(len(v) for v in anchors.values())
    attached_ok = [(i, rid) for rid, idxs in final.items() for i in idxs if i not in anchors[rid]]

    valid = set(range(1, len(paras) + 1))
    assigned = {i: rid for rid, idxs in final.items() for i in idxs}
    accounted = set(assigned) | set(none_indices)
    unaccounted = sorted(valid - accounted)

    overlap = [i for i in baseline if i in assigned]
    agree = [i for i in overlap if baseline[i] == assigned[i]]
    disagree = [(i, baseline[i], assigned[i]) for i in overlap if baseline[i] != assigned[i]]
    baseline_lost = [i for i in baseline if i not in assigned]

    gap = [i for i in valid if i not in baseline]
    gap_anchored = [i for i in gap if i in assigned and i in anchors.get(assigned[i], [])]
    gap_attached = [i for i in gap if i in assigned and i not in gap_anchored]

    report = {
        "doc": path.name,
        "paragraphs": len(paras),
        "batches": len(batches),
        "baseline_tagged": len(baseline),
        "baseline_entities": len(set(baseline.values())),
        "llm_entities_verified": len(final),
        "llm_assigned_paras": len(assigned),
        "anchor_members": anchor_members,
        "attached_ok": len(attached_ok),
        "agreement_overlap": len(overlap),
        "agreement_same": len(agree),
        "agreement_diff": disagree[:10],
        "baseline_lost_by_llm": baseline_lost[:10],
        "baseline_lost_count": len(baseline_lost),
        "gap_paras": len(gap),
        "gap_rescued_anchor_verified": len(gap_anchored),
        "gap_rescued_attached": len(gap_attached),
        "unaccounted_count": len(unaccounted),
        "unparsed_lines": unparsed_total,
        "groups_none_batches": groups_none_batches,
        "distinctness_gate_pass": gate_pass,
        "entity_ids_sample": sorted(final, key=lambda r: int(r) if r.isdigit() else 0)[:60],
        "none_paras": len(set(none_indices)),
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "max_batch_seconds": round(wall, 1),
    }

    print(
        f"LLM: {report['llm_entities_verified']} verified entities, "
        f"{report['llm_assigned_paras']}/{len(paras)} paras assigned "
        f"({anchor_members} anchor-verified, {len(attached_ok)} attached)"
    )
    print(
        f"agreement vs regex: {len(agree)}/{len(overlap)} same"
        + (f", diff samples: {disagree[:5]}" if disagree else "")
    )
    print(
        f"regex-tagged paras LLM failed to group: {len(baseline_lost)}"
        + (f" {baseline_lost[:8]}" if baseline_lost else "")
    )
    print(
        f"gap rescue (regex missed): {len(gap_anchored)} anchor-verified "
        f"+ {len(gap_attached)} attached / {len(gap)} gap paras"
    )
    print(
        f"safety: unaccounted={report['unaccounted_count']}, "
        f"unparsed lines={unparsed_total} "
        "(rejection details in the ETL compress log lines above)"
    )
    print(
        f"distinctness gate: {'PASS' if gate_pass else 'REJECT'} "
        f"| GROUPS:NONE batches: {groups_none_batches}/{len(batches)}"
    )
    print(
        f"cost: ~{tok_in} tok in / {tok_out} tok out (qwen-count), "
        f"slowest batch {report['max_batch_seconds']}s"
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docs", nargs="*", default=DEFAULT_DOCS)
    ap.add_argument("--config", default="configs/react.local.yaml")
    ap.add_argument("--batch-size", type=int, default=GROUPING_BATCH_SIZE)
    ap.add_argument("--truncate", type=int, default=GROUPING_TRUNCATE_CHARS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--expected", type=int, default=None, help="expected entity count per doc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    # Surface the production verifier's rejection diagnostics.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    adapter = None if args.dry_run else build_adapter(Path(args.config))
    reports = []
    for doc in args.docs:
        path = Path(doc)
        if not path.exists():
            print(f"skip (not found): {doc}", file=sys.stderr)
            continue
        report = run_document(
            adapter, path, args.batch_size, args.truncate, args.dry_run, args.workers
        )
        if args.expected is not None and not args.dry_run:
            got = report["llm_entities_verified"]
            print(
                f"expected entities: {args.expected} | got: {got} "
                f"({'OK' if got == args.expected else 'MISMATCH'})"
            )
            report["expected"] = args.expected
        reports.append(report)

    if args.json_out and not args.dry_run:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
