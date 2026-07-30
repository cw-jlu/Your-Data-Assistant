"""exp_138 focused test: math advisor + LIMIT 1 guard.

Test on:
  - target (= LIMIT 1 likely issues): task_80 × 3 (=reproducibility), task_75
  - sanity (= passing today): task_24, task_19, task_67, task_283, task_22

Reuses math_advisor_full50.py's logic but on a subset.
"""
from __future__ import annotations

import csv, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

# Reuse advisor + guard from existing exp_137 script + exp_122 registry (= contains LIMIT guard)
from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage
from experiments.exp_122_column_auditor.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_122_column_auditor.preamble import build_preamble
from experiments.exp_122_column_auditor.tools.registry import create_default_tool_registry
from experiments.exp_122_column_auditor.adaptive_vote import adaptive_vote

# Same math expert prompt as exp_137
_EXPERT_SYS = open(REPO / "scripts" / "math_advisor_full50.py").read().split('_EXPERT_SYS = """')[1].split('"""')[0]

OUT = REPO / "artifacts" / "exp_138_focused"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [("task_80", "rep0"), ("task_80", "rep1"), ("task_80", "rep2"), ("task_75", "rep0")]
SANITY = [(t, "fp") for t in ["task_24","task_19","task_67","task_283","task_22"]]
ALL = TARGETS + SANITY

N_ATTEMPTS = 3
MAX_WORKERS = 4


def make_model(temp=0.0):
    return OpenAIModelAdapter(model="qwen3.5-35b-a3b", api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"], temperature=temp,
        extra_headers={"CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID",""),
                       "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET","")})


def get_formula(question, model):
    r = model.complete([ModelMessage(role="system", content=_EXPERT_SYS),
                        ModelMessage(role="user", content=f"Q: {question}\n\nA:")],
                       enable_thinking=False, max_tokens=300)
    return r.strip()[:400]


def norm(rows):
    out = set()
    for r in rows:
        nr = tuple(re.sub(r"\.0+$", "", str(x).strip().lower()) for x in r)
        nr = tuple(f"{float(v):.2f}" if re.match(r"^-?\d+\.\d+$", v) else v for v in nr)
        out.add(nr)
    return out


def score_csv(pred, gold):
    p = list(csv.reader(open(pred))); g = list(csv.reader(open(gold)))
    if len(p)<2 or len(g)<2: return 0.0
    P, G = norm(p[1:]), norm(g[1:])
    if not G: return 0.0
    return max(0.0, len(P&G)/len(G) - 0.5*(len(P-G)/max(1,len(P))))


def run_one(tid, label):
    ds = DABenchPublicDataset(root_dir=REPO/"data"/"public"/"input")
    task = ds.get_task(tid)
    model = make_model()
    formula = get_formula(task.question, model)
    skip = formula.strip().upper().startswith("NO_CALC")
    task_dir = OUT / f"{tid}_{label}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "formula.txt").write_text(formula)
    (task_dir / "advisor_used.txt").write_text("False" if skip else "True")

    def attempt(i):
        m = make_model(temp=0.6)
        preamble = build_preamble(task)
        if skip:
            injected = preamble.text
        else:
            injected = (
                "# MATH FORMULA HINT (= expert calculation guide)\n"
                "Use this pseudo-math formula as the structural template for your SQL:\n"
                f"  {formula}\n\n"
                "Follow this aggregation/division/filter structure EXACTLY. If the formula\n"
                "says AVG, use AVG (not SUM/N). If it says *100, include *100. If it says\n"
                "filter-back, use (SELECT MIN/MAX) subquery (not LIMIT 1).\n\n"
            ) + preamble.text
        tools = create_default_tool_registry(
            auditor_model=m, question_provider=lambda: task.question, context_dir=task.context_dir,
        )
        agent = PhasedReActAgent(
            model=m, tools=tools,
            config=PhasedAgentConfig(max_steps=32, min_explore_queries=3),
            preamble=injected,
        )
        try:
            r = agent.run(task)
            tr = {
                "attempt": i,
                "answer": {"columns": list(r.answer.columns), "rows": [list(row) for row in r.answer.rows]} if r.answer else None,
                "steps": [{"i": j, "action": s.action, "action_input": s.action_input,
                           "thought": (s.thought or "")[:1500],
                           "observation_content": str(s.observation.get("content") if isinstance(s.observation,dict) else s.observation)[:500],
                          } for j, s in enumerate(r.steps)],
            }
            ans = r.answer if (r.answer and r.answer.rows) else None
            return (ans, tr)
        except Exception as e:
            return (None, {"attempt": i, "error": str(e)[:300]})

    from concurrent.futures import ThreadPoolExecutor as _IE
    answers, traces = [], []
    with _IE(max_workers=N_ATTEMPTS) as iex:
        for ans, tr in iex.map(attempt, range(N_ATTEMPTS)):
            if ans: answers.append(ans)
            traces.append(tr)
    with open(task_dir/"trace.json","w") as f:
        json.dump(traces, f, indent=2, default=str)
    if not answers:
        return {"tid": tid, "label": label, "score": 0.0, "formula": formula}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "label": label, "score": 0.0, "formula": formula}
    pred = task_dir/"prediction.csv"
    with open(pred,"w",newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    gold = REPO/"data"/"public"/"output"/tid/"gold.csv"
    s = score_csv(pred, gold)
    return {"tid": tid, "label": label, "score": s, "formula": formula,
            "cols": list(voted.columns), "n_rows": len(voted.rows)}


def main():
    print(f"=== exp_138 focused: {len(ALL)} runs ===\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, t, l): (t, l) for t, l in ALL}
        results = []
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            mark = "TGT" if any(r["tid"] == t and r["label"] == l for t, l in TARGETS) else "san"
            print(f"  [{mark}] {r['tid']}_{r['label']}: score={r['score']:.2f} cols={r.get('cols','?')} n_rows={r.get('n_rows','?')}", flush=True)
    t_scores = [r["score"] for r in results if r["tid"] in {t for t,_ in TARGETS}]
    s_scores = [r["score"] for r in results if r["tid"] in {t for t,_ in SANITY}]
    print(f"\nTARGET mean (task_80 + task_75): {sum(t_scores)/len(t_scores):.3f}")
    print(f"SANITY mean: {sum(s_scores)/len(s_scores):.3f}")
    with open(OUT/"results.json","w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
