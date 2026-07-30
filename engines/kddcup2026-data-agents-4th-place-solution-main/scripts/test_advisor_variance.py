"""Sample advisor formulas at temp=0.6 + thinking ON to see variance."""
from __future__ import annotations
import os, sys, time, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kobushi_core.model import OpenAIModelAdapter, ModelMessage

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

src = (ROOT / "scripts/test_new_advisor_agent.py").read_text()
NEW_SYS = re.search(r'NEW_SYS = """(.*?)"""', src, re.DOTALL).group(1)


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


def call(m, q, thinking, max_tokens=400):
    return m.complete(
        [ModelMessage(role="system", content=NEW_SYS),
         ModelMessage(role="user", content=f"Q: {q}\n\nA:")],
        enable_thinking=thinking, max_tokens=max_tokens,
    ).strip()[:500]


tids = ["task_418", "task_180", "task_352", "task_25"]
N_SAMPLES = 3

log("Loading tasks...")
qs = {t: json.load(open(ROOT / f"data/public/input/{t}/task.json"))["question"] for t in tids}

m_off = make_model(0.0)
m_on = make_model(0.6)

for tid, q in qs.items():
    print(f"\n=== {tid} ===\nQ: {q}\n")
    log(f"  baseline (T=0, thinking=OFF):")
    f0 = call(m_off, q, False)
    log(f"    {f0}")
    log(f"  T=0.6 + thinking=ON × {N_SAMPLES}:")
    for i in range(N_SAMPLES):
        t0 = time.time()
        f = call(m_on, q, True, max_tokens=6000)
        log(f"    [{i+1}] ({time.time()-t0:.1f}s) {f}")
