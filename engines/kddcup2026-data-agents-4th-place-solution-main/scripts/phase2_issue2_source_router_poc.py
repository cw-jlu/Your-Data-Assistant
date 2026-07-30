#!/usr/bin/env python3
"""PoC for issue #2: does deleting L52 fix source_router's avoid misclassification?

Tests the source-router classifier in isolation on the REAL task_1 payload (and
task_60 as a don't-break check) with the OLD _SYSTEM prompt vs a NEW one that
deletes L52 ("If video/keyframes show the final displayed value/list/ranking,
you MUST set sql_role=avoid."). Runs N times each and reports the avoid vs
support_only split.

task_1 ground truth: video defines ONLY criteria (threshold 100亿, year 2019,
group-by 二级行业, dedup) and says results are pending export -> support_only is
correct; avoid causes the deadlock. task_60: video shows a displayed dashboard
value -> avoid is often correct (don't regress it to support_only blindly).

Run: set AGENT_API_BASE/KEY (gpuhost), then
  python3 scripts/phase2_issue2_source_router_poc.py [--n 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "src")
from kobushi_core.model import ModelMessage, OpenAIModelAdapter  # noqa: E402
import experiments.exp_166_domain_rules.source_router as SR  # noqa: E402

L52 = "  * If video/keyframes show the final displayed value/list/ranking, you MUST set sql_role=avoid.\n"
SYSTEM_OLD = SR._SYSTEM
assert L52 in SYSTEM_OLD, "L52 text not found verbatim — update the marker"
SYSTEM_NEW = SYSTEM_OLD.replace(L52, "")


def _model() -> OpenAIModelAdapter:
    headers = {}
    cid = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    csec = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    if cid:
        headers["CF-Access-Client-Id"] = cid
    if csec:
        headers["CF-Access-Client-Secret"] = csec
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.0,
        extra_headers=headers,
    )


def _payload(task_id: str) -> dict:
    base = "data/phase2_demo/demo_samples_phase2/input"
    run = ("artifacts/runs/exp_166_answer_shape_prose_video_keyframe_note_pdf_"
           "preprocess_source_router_anti_agg_anti_agg_sql_guard_domain_router_"
           "math_other_only_003")
    q = json.load(open(f"{base}/{task_id}/task.json"))["question"]
    note_path = f"{run}/{task_id}/video_keyframe_note.md"
    note = open(note_path, encoding="utf-8").read() if os.path.exists(note_path) else None
    # the broad aggregating query the agent wanted to run (the correct answer for task_1)
    sql = {
        "task_1": ("SELECT e.SecondIndustryName, COUNT(DISTINCT f.CompanyCode) AS company_count "
                   "FROM lc_freefloat f JOIN lc_exgindustry e ON f.CompanyCode = e.CompanyCode "
                   "WHERE f.AFloats > 100000000000 AND strftime('%Y', f.ChangeDate) = '2019' "
                   "GROUP BY e.SecondIndustryName"),
        "task_60": ("SELECT OperatingForm AS Type FROM mf_fundtype "
                    "ORDER BY avg_daily_return DESC LIMIT 1"),
    }[task_id]
    p = {
        "question": q,
        "proposed_sql": sql,
        "tool_counts": {"read_doc": 5, "grep": 1, "describe_data": 2},
        "saw_video_signal": True,
        "saw_prose_redirect": False,
        "evidence": [],
    }
    if note:
        p["video_keyframe_note"] = SR._clip(note, 2200)
    return p


def run_variant(model, system: str, payload: dict, n: int) -> Counter:
    out = Counter()
    for _ in range(n):
        try:
            resp = model.complete(
                [ModelMessage(role="system", content=system),
                 ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False))],
                enable_thinking=False, max_tokens=256,
            )
            routed = SR._parse_route(resp)
            out[routed["sql_role"] if routed else "PARSE_FAIL"] += 1
        except Exception as exc:
            out[f"ERR:{type(exc).__name__}"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tasks", default="task_1,task_60")
    args = ap.parse_args()
    if not os.environ.get("AGENT_API_BASE"):
        raise SystemExit("set AGENT_API_BASE / AGENT_API_KEY (source .env)")
    model = _model()
    print(f"endpoint: {os.environ['AGENT_API_BASE']}  n={args.n} per variant\n")
    for tid in args.tasks.split(","):
        p = _payload(tid)
        old = run_variant(model, SYSTEM_OLD, p, args.n)
        new = run_variant(model, SYSTEM_NEW, p, args.n)
        want = "support_only" if tid == "task_1" else "avoid(or support_only ok)"
        print(f"### {tid}  (ground-truth role: {want})")
        print(f"  OLD prompt      : {dict(old)}")
        print(f"  NEW (L52 deleted): {dict(new)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
