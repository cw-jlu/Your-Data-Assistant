"""Aggregate exp_122 trace.log files to find bottlenecks.

Sources: artifacts/bench_phased_vote3_full50_exp126_exp123/

Analyses:
  1. Action histogram (= what tools are called) per phase
  2. Per-phase step count (= median, p90)
  3. Tool error rate per action
  4. Per-task: pre-vote score (= best single attempt) vs voted score
  5. Inter-attempt agreement (= do attempts converge or diverge?)
  6. Failure clustering: what action sequence comes before fail?

Output: artifacts/v5_v6_compare/bottleneck_report.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "artifacts" / "bench_phased_vote3_full50_exp126_exp123"
GT = json.load(open(BENCH / "summary.json"))["scores"]
OUT = REPO / "artifacts" / "v5_v6_compare" / "bottleneck_report.md"
OUT.parent.mkdir(parents=True, exist_ok=True)


def parse_trace(text: str) -> list[dict]:
    """Extract list of {phase, step, action, ok, content} from trace.log."""
    steps = []
    for line in text.splitlines():
        # phase=X step N action=ACT
        m = re.match(r"\[(\w+)\] phase=(\w+) step (\d+) action=(\w+)", line)
        if m:
            steps.append({
                "task_id": m.group(1), "phase": m.group(2),
                "step": int(m.group(3)), "action": m.group(4),
                "ok": None, "result_content": "",
            })
            continue
        # phase=X step N result ok=B terminal=Y content=...
        m = re.match(r"\[(\w+)\] phase=(\w+) step (\d+) result ok=(\w+) terminal=(\w+) content=(.*)", line)
        if m and steps:
            steps[-1]["ok"] = (m.group(4) == "True")
            steps[-1]["result_content"] = m.group(6)[:200]
            continue
    return steps


def main():
    # Walk all attempts
    by_phase_action = defaultdict(Counter)  # {phase: Counter(action)}
    by_action_errs = defaultdict(lambda: [0, 0])  # {action: [n_total, n_err]}
    phase_step_counts = defaultdict(list)  # {phase: [steps_in_phase, ...]} per attempt
    attempt_meta = []  # per-attempt records
    attempt_action_seq = defaultdict(list)  # task_id → [(attempt, action_seq, succeeded)]

    for tdir in sorted(BENCH.glob("task_*"), key=lambda p: int(p.name.split("_")[1])):
        tid = tdir.name
        gt = GT.get(tid, 0)
        for attempt_dir in sorted(tdir.glob("attempt_*")):
            tlog = attempt_dir / "trace.log"
            meta_p = attempt_dir / "meta.json"
            if not (tlog.exists() and meta_p.exists()):
                continue
            meta = json.load(open(meta_p))
            steps = parse_trace(tlog.read_text())
            # Aggregate
            phase_counts = Counter(s["phase"] for s in steps)
            for ph, n in phase_counts.items():
                phase_step_counts[ph].append(n)
            for s in steps:
                by_phase_action[s["phase"]][s["action"]] += 1
                by_action_errs[s["action"]][0] += 1
                if s["ok"] is False:
                    by_action_errs[s["action"]][1] += 1
            attempt_meta.append({
                "task_id": tid, "attempt": attempt_dir.name,
                "succeeded": meta["succeeded"], "n_steps": meta["n_steps"],
                "took": meta.get("took"),
                "gt_score": gt,
                "phase_counts": dict(phase_counts),
            })
            action_seq = tuple(s["action"] for s in steps)
            attempt_action_seq[tid].append((attempt_dir.name, action_seq, meta["succeeded"]))

    # Voted-score comparison: voted score is GT
    voted_score = {tid: GT.get(tid, 0) for tid in (p.name for p in BENCH.glob("task_*"))}

    # ---- Output ----
    lines = ["# exp_122 trace bottleneck analysis (= v5 bench, n=3 attempts × 50 tasks)\n"]

    # 1) Action histogram per phase
    lines.append("## 1. Action histogram per phase (= total calls across all attempts)")
    for phase in ("plan", "explore", "answer", "verify"):
        if phase not in by_phase_action: continue
        actions = by_phase_action[phase].most_common()
        total = sum(c for _, c in actions)
        lines.append(f"\n### Phase {phase.upper()} (total {total} calls)")
        for act, c in actions:
            pct = c/total*100
            lines.append(f"  - `{act}`: {c} ({pct:.1f}%)")

    # 2) Per-phase step count (median, p90, max)
    lines.append("\n## 2. Steps per attempt by phase (= median, p90, max)")
    lines.append("| phase | median | p90 | max | mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for phase in ("plan", "explore", "answer", "verify"):
        if phase not in phase_step_counts: continue
        xs = sorted(phase_step_counts[phase])
        med = xs[len(xs)//2] if xs else 0
        p90 = xs[int(len(xs)*0.9)] if xs else 0
        lines.append(f"| {phase} | {med} | {p90} | {max(xs)} | {mean(xs):.1f} |")

    # 3) Tool error rate
    lines.append("\n## 3. Tool error rate per action (= ok=False / total)")
    lines.append("| action | total | err | err_rate |")
    lines.append("|---|---:|---:|---:|")
    for act, (tot, err) in sorted(by_action_errs.items(), key=lambda kv: -kv[1][0]):
        if tot < 10: continue
        rate = err/tot if tot else 0
        flag = "❗" if rate > 0.2 else ""
        lines.append(f"| {act} | {tot} | {err} | {rate*100:.1f}% {flag}|")

    # 4) Per-attempt success/fail × n_steps
    succ = [a for a in attempt_meta if a["succeeded"]]
    fail = [a for a in attempt_meta if not a["succeeded"]]
    lines.append(f"\n## 4. Attempt-level summary")
    lines.append(f"- Total attempts: {len(attempt_meta)}")
    lines.append(f"- Succeeded: {len(succ)} ({len(succ)/len(attempt_meta)*100:.1f}%)")
    lines.append(f"- Failed: {len(fail)} ({len(fail)/len(attempt_meta)*100:.1f}%)")
    if succ:
        lines.append(f"- succ avg n_steps: {mean(a['n_steps'] for a in succ):.1f}")
    if fail:
        lines.append(f"- fail avg n_steps: {mean(a['n_steps'] for a in fail):.1f}")

    # 5) Vote-rescue analysis: tasks where >=1 attempt succeeded but voted answer is wrong
    lines.append(f"\n## 5. Vote-rescue and vote-fail patterns")
    rescue = []  # at least one good attempt + final voted=fail
    fail_consensus = []  # 0 good attempts + voted=fail (= no chance)
    success_consensus = []  # 3 good + voted=ok
    for tid in (p.name for p in BENCH.glob("task_*")):
        atts = [a for a in attempt_meta if a["task_id"] == tid]
        if not atts: continue
        n_good = sum(1 for a in atts if a["succeeded"])
        v = voted_score.get(tid, 0)
        if n_good >= 1 and v < 1.0:
            rescue.append((tid, n_good, len(atts), v))
        if n_good == 0:
            fail_consensus.append((tid, len(atts), v))
        if n_good == len(atts) and v >= 1.0:
            success_consensus.append((tid, len(atts), v))
    lines.append(f"- Vote-fail-despite-some-success: {len(rescue)} tasks → vote dropped good attempts")
    for tid, ng, nt, v in rescue[:10]:
        lines.append(f"  - {tid}: {ng}/{nt} attempts succeeded but voted score={v}")
    lines.append(f"- All-attempts-fail: {len(fail_consensus)} tasks → no attempt got it (= deep blindspot)")
    for tid, nt, v in fail_consensus[:10]:
        lines.append(f"  - {tid}: all {nt} attempts failed, voted={v}")
    lines.append(f"- All-attempts-success: {len(success_consensus)} tasks (= solid)")

    # 6) Inter-attempt agreement: do attempts converge to same answer?
    lines.append(f"\n## 6. Inter-attempt action-sequence diversity (= top 5 most-diverse tasks)")
    diverse = []
    for tid, seqs in attempt_action_seq.items():
        if len(seqs) < 2: continue
        # Distinct (action, step_index) profiles
        distinct = len({s[1] for s in seqs})
        diverse.append((tid, distinct, [len(s[1]) for s in seqs]))
    diverse.sort(key=lambda x: -x[1])
    for tid, dist, lengths in diverse[:8]:
        lines.append(f"  - {tid}: {dist} distinct action seqs across attempts, lengths={lengths}, GT={GT.get(tid,'?')}")

    OUT.write_text("\n".join(lines))
    for line in lines:
        print(line)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
