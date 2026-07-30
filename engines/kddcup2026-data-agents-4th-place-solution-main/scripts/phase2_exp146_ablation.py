"""Phase 2 exp_146 per-lever ABLATION bench.

Each lever is toggled by an env var (see experiments.exp_146_modality.flags):
  EXP146_ANSWER_SHAPE=1  → ① DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP146_PROSE=1         → ② pypdf read + prose_docs inventory + 404→doc redirect + protocol
  EXP146_VIDEO=1         → ③ pre-extract video summary into preamble (no raw video in loop)
  EXP146_KEYFRAMES=1     → attach extracted unique video keyframes as images (no raw video)
  EXP146_WATCH_VIDEO=1   → expose raw-video inspection as an EXPLORE tool

ALL flags off ⇒ identical pipeline to exp_143_dynamic_doc (sanity baseline).

Run dir is named by the active levers, e.g.:
  EXP146_ANSWER_SHAPE=1 python scripts/phase2_exp146_ablation.py   → exp_146_answer_shape_001
  EXP146_PROSE=1        python scripts/phase2_exp146_ablation.py   → exp_146_prose_001
  EXP146_KEYFRAMES=1    python scripts/phase2_exp146_ablation.py   → exp_146_keyframes_001
  EXP146_WATCH_VIDEO=1  python scripts/phase2_exp146_ablation.py   → exp_146_watch_video_001
  (no flags)            python scripts/phase2_exp146_ablation.py   → exp_146_baseline_001

att=1, workers=12 by default (exploration class). Scored via kobushi_core.eval.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_146_modality import flags
from experiments.exp_146_modality.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_146_modality.preamble import build_preamble
from experiments.exp_146_modality.tools.registry import create_default_tool_registry
from experiments.exp_146_modality.adaptive_vote import adaptive_vote
from experiments.exp_146_modality.math_advisor import generate_formula
from experiments.exp_146_modality.video_summarizer import summarize_video
from experiments.exp_146_modality.prefix_cache import (
    prefix_cache_enabled,
    with_prefix_cache_header,
)

DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"

N_ATTEMPTS = 1
MAX_STEPS = 64


def make_model(temp: float = 0.6, *, task_id: str | None = None) -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=temp,
        extra_headers=headers,
    )


def _official_score(pred_csv: Path, gold_csv: Path) -> float:
    from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions
    if not gold_csv.is_file():
        return 0.0
    try:
        ev = _evaluate_task(
            task_id=str(pred_csv.parent.name),
            prediction_path=pred_csv, gold_path=gold_csv,
            options=EvaluationOptions(),
        )
        return float(ev.official_score_lambda_0_5)
    except Exception as e:
        print(f"  [score err] {e}", flush=True)
        return 0.0


def run_one(tid: str, out_dir: Path) -> dict:
    t0 = time.time()
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    try:
        task = ds.get_task(tid)
    except Exception as e:
        return {"tid": tid, "error": f"load: {e}", "score": 0.0, "elapsed": time.time() - t0}

    task_dir = out_dir / tid
    task_dir.mkdir(parents=True, exist_ok=True)

    if flags.pdf_preprocess_on():
        try:
            from experiments.exp_146_modality.pdf_text_cache import ensure_task_pdf_cache
            pdf_cache_results = ensure_task_pdf_cache(task)
            (task_dir / "pdf_preprocess.json").write_text(
                json.dumps(
                    [
                        {
                            "source_rel": r.source_rel,
                            "text_path": str(r.text_path),
                            "meta_path": str(r.meta_path),
                            "page_count": r.page_count,
                            "text_chars": r.text_chars,
                            "source_sha1": r.source_sha1,
                        }
                        for r in pdf_cache_results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except Exception as e:
            (task_dir / "pdf_preprocess_error.txt").write_text(str(e))

    if flags.keyframes_on():
        try:
            from experiments.exp_146_modality.keyframe_cache import ensure_task_keyframes
            keyframe_result = ensure_task_keyframes(task)
            if keyframe_result is not None:
                (task_dir / "keyframes.json").write_text(
                    json.dumps(
                        {
                            "source_rel": keyframe_result.source_rel,
                            "frames_dir": str(keyframe_result.frames_dir),
                            "meta_path": str(keyframe_result.meta_path),
                            "contact_sheet": str(keyframe_result.contact_sheet),
                            "duration": keyframe_result.duration,
                            "sampled": keyframe_result.sampled,
                            "kept": keyframe_result.kept,
                            "selected": keyframe_result.selected,
                            "source_sha1": keyframe_result.source_sha1,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        except Exception as e:
            (task_dir / "keyframes_error.txt").write_text(str(e))

    formula = "NO_CALC"
    try:
        formula = generate_formula(task.question, task_id=tid)
    except Exception as e:
        formula = f"NO_CALC ({e})"
    (task_dir / "formula.txt").write_text(formula)
    skip_advisor = formula.strip().upper().startswith("NO_CALC")
    (task_dir / "advisor_used.txt").write_text("False" if skip_advisor else "True")

    # ③ video lever: pre-extract a summary ONCE (per task) and inject into preamble.
    video_summary = None
    if flags.video_on():
        try:
            video_summary = summarize_video(task, make_model(temp=0.6, task_id=tid))
        except Exception as e:
            video_summary = f"[video summary error: {e!r}]"
        if video_summary:
            (task_dir / "video_summary.txt").write_text(video_summary)

    answers = []
    attempt_traces = []
    for i in range(N_ATTEMPTS):
        try:
            m = make_model(temp=0.6, task_id=tid)
            preamble = build_preamble(task, video_summary=video_summary)
            if skip_advisor:
                injected = preamble.text
            else:
                injected = (
                    "# MATH FORMULA HINT (= expert calculation guide)\n"
                    f"  {formula}\n\n"
                    "Follow this aggregation/division/filter structure EXACTLY.\n\n"
                ) + preamble.text
            tools = create_default_tool_registry(
                auditor_model=m, question_provider=lambda: task.question, context_dir=task.context_dir,
            )
            agent = PhasedReActAgent(
                model=m, tools=tools,
                config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
                preamble=injected,
            )
            agent.trace_log_path = str(task_dir / f"attempt_{i}.steps.log")
            r = agent.run(task)
            if r.answer and r.answer.rows:
                answers.append(r.answer)
            attempt_traces.append({
                "attempt": i,
                "answer": {"columns": list(r.answer.columns), "rows": [list(row) for row in r.answer.rows]} if r.answer else None,
                "failure_reason": getattr(r, "failure_reason", None),
                "steps": [
                    {
                        "i": j, "action": s.action, "action_input": s.action_input,
                        "thought": (s.thought or "")[:2000],
                        "observation_ok": s.observation.get("ok") if isinstance(s.observation, dict) else None,
                        "observation_content_preview": str(s.observation.get("content") if isinstance(s.observation, dict) else s.observation)[:400],
                    }
                    for j, s in enumerate(r.steps)
                ],
            })
        except Exception as e:
            attempt_traces.append({"attempt": i, "error": str(e)[:300]})

    (task_dir / "trace.json").write_text(json.dumps(attempt_traces, ensure_ascii=False, indent=2, default=str))

    if not answers:
        return {"tid": tid, "formula": formula, "score": 0.0, "n_ok": 0, "elapsed": time.time() - t0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "formula": formula, "score": 0.0, "n_ok": len(answers), "elapsed": time.time() - t0}
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    score = _official_score(pred, GOLD_ROOT / tid / "gold.csv")
    return {"tid": tid, "formula": formula, "score": score, "n_ok": len(answers),
            "cols": list(voted.columns), "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-attempts", type=int, default=1, help="attempts per task (adaptive_vote union)")
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task_ids (default: all 60)")
    a = ap.parse_args()

    global N_ATTEMPTS
    N_ATTEMPTS = a.n_attempts

    levers = flags.active_levers()
    tag = "_".join(levers) if levers else "baseline"
    if N_ATTEMPTS != 1:
        tag = f"{tag}_{N_ATTEMPTS}att"

    runs = REPO / "artifacts" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    i = 1
    while (runs / f"exp_146_{tag}_{i:03d}").exists():
        i += 1
    out_dir = runs / f"exp_146_{tag}_{i:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if flags.pdf_preprocess_on() and not os.environ.get("EXP146_PDF_TEXT_CACHE_ROOT"):
        os.environ["EXP146_PDF_TEXT_CACHE_ROOT"] = str(
            Path("/tmp") / "kobushi_exp146_pdf_text_cache" / out_dir.name
        )
    if flags.keyframes_on() and not os.environ.get("EXP146_KEYFRAME_ROOT"):
        os.environ["EXP146_KEYFRAME_ROOT"] = str(
            Path("/tmp") / "kobushi_exp146_keyframe_cache" / out_dir.name
        )

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    if a.tasks:
        task_ids = [t.strip() for t in a.tasks.split(",") if t.strip()]
    else:
        task_ids = sorted([t.task_id for t in ds.iter_tasks()], key=lambda x: int(x.split("_")[1]))

    print(f"=== exp_146 ABLATION levers={levers or ['baseline']} → {out_dir.name} ===")
    print(f"  workers={a.workers}, att={N_ATTEMPTS}, max_steps={MAX_STEPS}, T=0.6, "
          f"{len(task_ids)} tasks, prefix_cache={prefix_cache_enabled()}", flush=True)
    if flags.pdf_preprocess_on():
        print(f"  pdf_text_cache_root={os.environ.get('EXP146_PDF_TEXT_CACHE_ROOT')}", flush=True)
    if flags.keyframes_on():
        print(f"  keyframe_root={os.environ.get('EXP146_KEYFRAME_ROOT')}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, tid, out_dir): tid for tid in task_ids}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # defensive: never let one task kill the run
                r = {"tid": futs[fut], "error": f"worker crash: {e}"[:120], "score": 0.0}
            results.append(r)
            mark = "✓" if r.get("score", 0) >= 0.999 else "✗"
            running = sum(rr["score"] for rr in results) / len(results)
            print(f"[{k:3d}/{len(task_ids)}] {mark} {r['tid']:<10} score={r.get('score',0):.2f} "
                  f"n_ok={r.get('n_ok',0)} t={r.get('elapsed',0):.0f}s mean={running:.4f} "
                  f"{(r.get('error') or '')[:40]}", flush=True)

    elapsed = time.time() - t_start
    mean = sum(r["score"] for r in results) / len(results) if results else 0
    n_perfect = sum(1 for r in results if r["score"] >= 0.999)
    n_zero = sum(1 for r in results if r["score"] < 0.01)
    (out_dir / "summary.json").write_text(json.dumps({
        "levers": levers, "tag": tag,
        "n_tasks": len(results), "mean_score": mean,
        "n_perfect": n_perfect, "n_zero": n_zero,
        "elapsed_minutes": elapsed / 60,
        "config": {"n_attempts": N_ATTEMPTS, "max_workers": a.workers,
                   "max_steps": MAX_STEPS, "model": "qwen3.5-35b-a3b", "temperature": 0.6,
                   "prefix_cache": prefix_cache_enabled(),
                   "prefix_cache_namespace": os.environ.get("EXP146_PREFIX_CACHE_NAMESPACE", "kobushi-phase2-exp146"),
                   "source_router": flags.source_router_on(),
                   "watch_video": flags.watch_video_on(),
                   "source_router_model": os.environ.get("EXP146_SOURCE_ROUTER_MODEL", "1"),
                   "pdf_text_cache_root": os.environ.get("EXP146_PDF_TEXT_CACHE_ROOT"),
                   "keyframe_root": os.environ.get("EXP146_KEYFRAME_ROOT"),
                   "keyframe_max_images": os.environ.get("EXP146_KEYFRAME_MAX_IMAGES", "18")},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n=== DONE levers={levers or ['baseline']} mean={mean:.4f} "
          f"perfect={n_perfect}/{len(results)} zero={n_zero} elapsed={elapsed/60:.1f}min ===")
    print(f"saved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
