"""Smoke test: 1 task via exp_137_math_advisor.runner._run_single_task_with_timeout."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main():
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")

    from experiments.exp_137_math_advisor.config import load_app_config
    from experiments.exp_137_math_advisor.runner import _run_single_task_with_timeout

    cfg = load_app_config(REPO / "src/experiments/exp_137_math_advisor/config.yaml")
    print(f"config: max_steps={cfg.agent.max_steps}, timeout={cfg.run.task_timeout_seconds}", flush=True)
    print(f"  model={cfg.agent.model}, api_base={cfg.agent.api_base[:40]}...", flush=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] starting task_25 (= 'lowest cost' simple)...", flush=True)
    result = _run_single_task_with_timeout(task_id="task_25", config=cfg)
    print(f"[{time.strftime('%H:%M:%S')}] elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  succeeded: {result.get('succeeded')}")
    print(f"  failure_reason: {result.get('failure_reason')}")
    print(f"  n_attempts_succeeded: {result.get('n_attempts_succeeded')}")
    print(f"  vote_strategy: {result.get('vote_strategy')}")
    ans = result.get('answer')
    if ans:
        print(f"  answer cols: {ans.get('columns')}")
        print(f"  answer rows: {ans.get('rows', [])[:3]}")


if __name__ == "__main__":
    main()
