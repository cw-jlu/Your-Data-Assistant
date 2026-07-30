"""Single-task single-attempt smoke test of NEW advisor prompt + agent.
Heavy logging at each step to diagnose where the previous 19min run hung.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("import kobushi_core.eval...")
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions

log("import kobushi_core.model...")
from kobushi_core.model import OpenAIModelAdapter, ModelMessage

log("import benchmark...")
from kobushi_core.benchmark import DABenchPublicDataset

log("import exp_137 components...")
from experiments.exp_137_math_advisor.preamble import build_preamble
from experiments.exp_137_math_advisor.phased_agent import (
    PhasedReActAgent,
    PhasedAgentConfig,
)
from experiments.exp_137_math_advisor.tools.registry import create_default_tool_registry

log("loading NEW_SYS from test script...")
import re
src = (ROOT / "scripts/test_new_advisor_agent.py").read_text()
NEW_SYS = re.search(r'NEW_SYS = """(.*?)"""', src, re.DOTALL).group(1)

TID = "task_418"


def make_model(temp):
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=temp,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


log(f"loading task: {TID}")
ds = DABenchPublicDataset(root_dir=ROOT / "data" / "public" / "input")
task = ds.get_task(TID)
log(f"  Q: {task.question[:120]}")

log("calling advisor (thinking OFF)...")
t0 = time.time()
m_adv = make_model(0.0)
formula = m_adv.complete(
    [ModelMessage(role="system", content=NEW_SYS),
     ModelMessage(role="user", content=f"Q: {task.question}\n\nA:")],
    enable_thinking=False, max_tokens=400,
).strip()[:500]
log(f"  formula ({time.time()-t0:.1f}s): {formula}")

skip = formula.upper().startswith("NO_CALC")
log(f"  skip_advisor: {skip}")

log("building preamble + agent...")
m = make_model(0.6)
preamble = build_preamble(task)
hint = (
    "# MATH HINT\n"
    f"{formula}\n\n"
    "WHERE (uppercase) = SQL filter-back computation — follow literally.\n"
    "`|` (such that) = concept predicate — verify column names and value\n"
    "formats in EXPLORE before turning into SQL.\n\n"
)
injected = preamble.text if skip else hint + preamble.text
log(f"  preamble length: {len(injected)} chars")

tools = create_default_tool_registry(
    auditor_model=m,
    question_provider=lambda: task.question,
    context_dir=task.context_dir,
)
agent = PhasedReActAgent(
    model=m, tools=tools,
    config=PhasedAgentConfig(max_steps=64, min_explore_queries=3),
    preamble=injected,
)

log("agent.run() starting (thinking ON, max_steps=64)...")
t1 = time.time()
r = agent.run(task)
log(f"agent.run done in {time.time()-t1:.1f}s, succeeded={r.succeeded}, n_steps={len(r.steps)}")

# Dump every step's action + phase to see where it got stuck
log("=== AGENT STEP TRACE ===")
for j, s in enumerate(r.steps):
    phase = getattr(s, "phase", "?")
    action = s.action
    ai = str(s.action_input)[:120] if s.action_input else ""
    obs_ok = s.observation.get("ok") if isinstance(s.observation, dict) else None
    obs_preview = str(s.observation.get("content") if isinstance(s.observation, dict) else s.observation)[:150]
    log(f"  step {j:2d} [{phase:8s}] {action:18s} input={ai} ok={obs_ok}")
    if not obs_ok and obs_preview:
        log(f"            obs: {obs_preview}")
    elif obs_preview:
        log(f"            obs: {obs_preview[:80]}")
if r.answer:
    log(f"  answer cols: {list(r.answer.columns)}")
    log(f"  answer rows ({len(r.answer.rows)}): {[list(row) for row in r.answer.rows[:5]]}")
else:
    log("  NO ANSWER")

if r.answer and r.answer.rows:
    out_dir = ROOT / "artifacts" / "test_new_advisor_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "prediction.csv"
    import csv as _csv
    with pred_path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(r.answer.columns)
        for row in r.answer.rows:
            w.writerow(row)
    gold = ROOT / "data" / "public" / "output" / TID / "gold.csv"
    log(f"  gold: {gold.read_text().strip()}")
    e = _evaluate_task(task_id=TID, prediction_path=pred_path, gold_path=gold,
                       options=EvaluationOptions())
    log(f"OFFICIAL SCORE λ=0.5: {e.official_score_lambda_0_5}")
    log(f"  matched_cols={e.matched_cols} gold_cols={e.gold_cols} pred_cols={e.pred_cols}")
    log(f"  recall={e.recall} extras_ratio={e.extras_ratio}")
log(f"TOTAL: {time.time()-t0:.1f}s")
