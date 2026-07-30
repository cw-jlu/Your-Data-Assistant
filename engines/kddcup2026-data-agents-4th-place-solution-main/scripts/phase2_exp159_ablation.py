"""Phase 2 exp_159 v4 ABLATION bench.

Each lever is toggled by an env var (see experiments.exp_159_finance_domain_prompt.flags):
  EXP159_ANSWER_SHAPE=1  → ① DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP159_PROSE=1         → ② pypdf read + prose_docs inventory + 404→doc redirect + protocol
  EXP159_VIDEO=1         → ③ pre-extract video summary into preamble (no raw video in loop)
  EXP159_KEYFRAMES=1     → attach extracted unique video keyframes as images (no raw video)
  EXP159_VIDEO_KEYFRAME_NOTE=1 → extract visible keyframe information into preamble/router
  EXP159_WATCH_VIDEO=1   → expose raw-video inspection as an EXPLORE tool
  EXP159_FINAL_SQL_GUARD=1 → classify and rewrite unnecessary DISTINCT / NULL filters
  EXP159_AUDIO_ASR=1     → transcribe task video audio as a cautious preamble/router hint
  EXP159_FINANCE_DOMAIN_PROMPT=1 → inject a compact finance note for fund/stock/macro

exp159 = exp155/v4 base + task-domain router + compact finance note. The v2
projection pruner and final SQL guard are deliberately off by default.

Run dir is named by the active levers, e.g.:
  EXP159_ANSWER_SHAPE=1 python scripts/phase2_exp159_ablation.py   → exp_159_answer_shape_001
  EXP159_PROSE=1        python scripts/phase2_exp159_ablation.py   → exp_159_prose_001
  EXP159_KEYFRAMES=1    python scripts/phase2_exp159_ablation.py   → exp_159_keyframes_001
  EXP159_WATCH_VIDEO=1  python scripts/phase2_exp159_ablation.py   → exp_159_watch_video_001
  (no flags)            python scripts/phase2_exp159_ablation.py   → exp_159_baseline_001

att=1, workers=8 by default. Scored via kobushi_core.eval.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_159_finance_domain_prompt import flags
from experiments.exp_159_finance_domain_prompt.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_159_finance_domain_prompt.preamble import build_preamble
from experiments.exp_159_finance_domain_prompt.tools.registry import create_default_tool_registry
from experiments.exp_159_finance_domain_prompt.adaptive_vote import adaptive_vote
from experiments.exp_159_finance_domain_prompt.anti_aggregation import no_agg_note, review_aggregation
from experiments.exp_159_finance_domain_prompt.math_advisor import generate_formula
from experiments.exp_159_finance_domain_prompt.video_keyframe_note import (
    extract_keyframe_note,
    preamble_block as video_keyframe_preamble_block,
)
from experiments.exp_159_finance_domain_prompt.video_summarizer import summarize_video
from experiments.exp_159_finance_domain_prompt.prefix_cache import (
    prefix_cache_enabled,
    with_prefix_cache_header,
)
from experiments.exp_159_finance_domain_prompt.audio_asr import (
    audio_preamble_block,
    find_video,
    load_existing_asr,
    prepare_task_audio,
)
from experiments.exp_159_finance_domain_prompt.domain_prompt import finance_domain_note
from experiments.exp_159_finance_domain_prompt.domain_router import classify_domain

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


def _run_advisors(question: str, tid: str) -> tuple[str, object]:
    """Run independent planning sub-agents for one task."""
    if not flags.anti_agg_on():
        formula = generate_formula(question, task_id=tid)
        anti = SimpleNamespace(
            raw="AGG: disabled", label="AGG", prompt_lang="en", no_agg=False
        )
        return formula, anti

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_formula = ex.submit(generate_formula, question, task_id=tid)
        f_anti = ex.submit(review_aggregation, question, task_id=tid)
        formula = f_formula.result()
        anti = f_anti.result()
    return formula, anti


def run_one(
    tid: str,
    out_dir: Path,
    existing_asr: dict[str, dict],
    whisper_model,
    whisper_lock,
    args,
) -> dict:
    t0 = time.time()
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    try:
        task = ds.get_task(tid)
    except Exception as e:
        return {"tid": tid, "error": f"load: {e}", "score": 0.0, "elapsed": time.time() - t0}

    task_dir = out_dir / tid
    task_dir.mkdir(parents=True, exist_ok=True)

    domain_route = None
    domain_note = ""
    if flags.finance_domain_prompt_on():
        domain_route = classify_domain(task, make_model(temp=0.0, task_id=f"{tid}:domain"))
        domain_note = finance_domain_note(domain_route.domain)
        (task_dir / "domain_route.json").write_text(
            json.dumps(domain_route.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if domain_note:
            (task_dir / "domain_prompt.md").write_text(domain_note, encoding="utf-8")

    if flags.pdf_preprocess_on():
        try:
            from experiments.exp_159_finance_domain_prompt.pdf_text_cache import ensure_task_pdf_cache
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
            from experiments.exp_159_finance_domain_prompt.keyframe_cache import ensure_task_keyframes
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

    video_keyframe_note = None
    if flags.video_keyframe_note_on():
        try:
            video_keyframe_note = extract_keyframe_note(
                task, make_model(temp=0.2, task_id=tid)
            )
            if video_keyframe_note is not None:
                (task_dir / "video_keyframe_note.md").write_text(video_keyframe_note.text)
                (task_dir / "video_keyframe_note_meta.json").write_text(
                    json.dumps(
                        {
                            "prompt_lang": video_keyframe_note.prompt_lang,
                            "n_keyframes": video_keyframe_note.n_keyframes,
                            "frames": video_keyframe_note.frames,
                            "keyframes": video_keyframe_note.meta,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        except Exception as e:
            (task_dir / "video_keyframe_note_error.txt").write_text(str(e))

    asr_text = None
    asr_meta = {"status": "disabled"}
    if flags.audio_asr_on():
        try:
            asr_text, asr_meta = prepare_task_audio(
                task_id=tid,
                task=task,
                task_dir=task_dir,
                whisper_model=whisper_model,
                whisper_lock=whisper_lock,
                language=args.asr_language,
                quality_filter=args.asr_quality_filter,
                existing_asr=existing_asr,
                existing_asr_path=args.asr_existing_json,
                reuse_cache=args.reuse_asr_cache,
            )
        except Exception as e:
            asr_meta = {"status": "error", "error": repr(e)}
            (task_dir / "asr_error.txt").write_text(str(e), encoding="utf-8")

    formula = "NO_CALC"
    try:
        formula, anti_agg = _run_advisors(task.question, tid)
    except Exception as e:
        formula = f"NO_CALC ({e})"
        anti_agg = SimpleNamespace(
            raw=f"AGG: advisor_error={e!r}", label="AGG", prompt_lang="en", no_agg=False
        )
    (task_dir / "formula.txt").write_text(formula)
    (task_dir / "anti_aggregation.txt").write_text(
        f"label={anti_agg.label}\nlang={anti_agg.prompt_lang}\nraw={anti_agg.raw}\n"
    )
    skip_advisor = formula.strip().upper().startswith("NO_CALC") or anti_agg.no_agg
    advisor_used = not skip_advisor
    (task_dir / "advisor_used.txt").write_text("True" if advisor_used else "False")

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
            preamble_text = preamble.text
            router_video_note = video_keyframe_note.text if video_keyframe_note else None
            if asr_text:
                preamble_text = audio_preamble_block(
                    task.question,
                    asr_text,
                    max_chars=args.asr_max_chars,
                ) + preamble_text
                if args.audio_to_router:
                    router_video_note = (
                        (router_video_note or "")
                        + "\n\n# VIDEO AUDIO TRANSCRIPT (ASR; may contain errors)\n"
                        + asr_text[: args.router_asr_max_chars]
                    ).strip()
            if video_keyframe_note is not None:
                preamble_text = video_keyframe_preamble_block(video_keyframe_note) + preamble_text
            if domain_note:
                preamble_text = domain_note + preamble_text
            if anti_agg.no_agg:
                injected = no_agg_note(anti_agg.prompt_lang) + preamble_text
            elif skip_advisor:
                injected = preamble_text
            else:
                injected = (
                    "# MATH FORMULA HINT (= expert calculation guide)\n"
                    f"  {formula}\n\n"
                    "Follow this aggregation/division/filter structure EXACTLY.\n\n"
                ) + preamble_text
            tools = create_default_tool_registry(
                auditor_model=m,
                question_provider=lambda: task.question,
                context_dir=task.context_dir,
                anti_agg_label_provider=lambda: anti_agg.label,
            )
            agent = PhasedReActAgent(
                model=m, tools=tools,
                config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
                preamble=injected,
                video_keyframe_note=router_video_note,
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
        return {"tid": tid, "domain": domain_route.domain if domain_route else None,
                "domain_prompt_included": bool(domain_note),
                "formula": formula, "anti_aggregation": anti_agg.raw,
                "anti_agg_label": anti_agg.label, "advisor_used": advisor_used,
                "asr_included": bool(asr_text), "asr_status": asr_meta.get("status"),
                "score": 0.0, "n_ok": 0, "elapsed": time.time() - t0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "domain": domain_route.domain if domain_route else None,
                "domain_prompt_included": bool(domain_note),
                "formula": formula, "anti_aggregation": anti_agg.raw,
                "anti_agg_label": anti_agg.label, "advisor_used": advisor_used,
                "asr_included": bool(asr_text), "asr_status": asr_meta.get("status"),
                "score": 0.0, "n_ok": len(answers), "elapsed": time.time() - t0}
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    score = _official_score(pred, GOLD_ROOT / tid / "gold.csv")
    return {"tid": tid, "domain": domain_route.domain if domain_route else None,
            "domain_prompt_included": bool(domain_note),
            "formula": formula, "anti_aggregation": anti_agg.raw,
            "anti_agg_label": anti_agg.label, "advisor_used": advisor_used,
            "asr_included": bool(asr_text), "asr_status": asr_meta.get("status"),
            "score": score, "n_ok": len(answers),
            "cols": list(voted.columns), "elapsed": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-attempts", type=int, default=1, help="attempts per task (adaptive_vote union)")
    ap.add_argument("--tasks", type=str, default=None, help="comma-separated task_ids (default: all 60)")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--asr-cpu-threads", type=int, default=4)
    ap.add_argument("--asr-language", default="auto", choices=["auto", "zh", "en"])
    ap.add_argument("--asr-existing-json", type=Path, default=None)
    ap.add_argument("--reuse-asr-cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--asr-quality-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--asr-max-chars", type=int, default=2600)
    ap.add_argument("--audio-to-router", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--router-asr-max-chars", type=int, default=1600)
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
    while (runs / f"exp_159_{tag}_{i:03d}").exists():
        i += 1
    out_dir = runs / f"exp_159_{tag}_{i:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if flags.pdf_preprocess_on() and not os.environ.get("EXP159_PDF_TEXT_CACHE_ROOT"):
        os.environ["EXP159_PDF_TEXT_CACHE_ROOT"] = str(
            Path("/tmp") / "kobushi_exp159_pdf_text_cache" / out_dir.name
        )
    if (flags.keyframes_on() or flags.video_keyframe_note_on()) and not os.environ.get("EXP159_KEYFRAME_ROOT"):
        os.environ["EXP159_KEYFRAME_ROOT"] = str(
            Path("/tmp") / "kobushi_exp159_keyframe_cache" / out_dir.name
        )
    if flags.audio_asr_on() and not os.environ.get("EXP159_AUDIO_ASR_CACHE_ROOT"):
        os.environ["EXP159_AUDIO_ASR_CACHE_ROOT"] = str(
            Path("/tmp") / "kobushi_exp159_audio_asr_cache" / out_dir.name
        )

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    if a.tasks:
        task_ids = [t.strip() for t in a.tasks.split(",") if t.strip()]
    else:
        task_ids = sorted([t.task_id for t in ds.iter_tasks()], key=lambda x: int(x.split("_")[1]))

    existing_asr = load_existing_asr(a.asr_existing_json) if flags.audio_asr_on() else {}
    whisper_model = None
    whisper_lock = threading.Lock()
    if flags.audio_asr_on():
        needs_whisper = False
        for tid in task_ids:
            if tid in existing_asr:
                continue
            try:
                task = ds.get_task(tid)
            except Exception:
                continue
            if find_video(task.context_dir) is not None:
                needs_whisper = True
                break
        if needs_whisper:
            from faster_whisper import WhisperModel

            print(f"  loading ASR model={a.asr_model} on CPU int8", flush=True)
            whisper_model = WhisperModel(
                a.asr_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=a.asr_cpu_threads,
            )

    print(f"=== exp_159 ABLATION levers={levers or ['baseline']} → {out_dir.name} ===")
    print(f"  workers={a.workers}, att={N_ATTEMPTS}, max_steps={MAX_STEPS}, T=0.6, "
          f"{len(task_ids)} tasks, prefix_cache={prefix_cache_enabled()}", flush=True)
    if flags.pdf_preprocess_on():
        print(f"  pdf_text_cache_root={os.environ.get('EXP159_PDF_TEXT_CACHE_ROOT')}", flush=True)
    if flags.keyframes_on() or flags.video_keyframe_note_on():
        print(f"  keyframe_root={os.environ.get('EXP159_KEYFRAME_ROOT')}", flush=True)
    if flags.audio_asr_on():
        print(f"  audio_asr_cache_root={os.environ.get('EXP159_AUDIO_ASR_CACHE_ROOT')}", flush=True)

    results = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {
            ex.submit(run_one, tid, out_dir, existing_asr, whisper_model, whisper_lock, a): tid
            for tid in task_ids
        }
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
                   "prefix_cache_namespace": os.environ.get("EXP159_PREFIX_CACHE_NAMESPACE", "kobushi-phase2-exp159"),
                   "source_router": flags.source_router_on(),
                   "anti_agg": flags.anti_agg_on(),
                   "anti_agg_sql_guard": flags.anti_agg_sql_guard_on(),
                   "final_sql_guard": flags.final_sql_guard_on(),
                   "finance_domain_prompt": flags.finance_domain_prompt_on(),
                   "video_keyframe_note": flags.video_keyframe_note_on(),
                   "video_keyframe_note_max_chars": os.environ.get("EXP159_VIDEO_KEYFRAME_NOTE_MAX_CHARS", "1800"),
                   "watch_video": flags.watch_video_on(),
                   "source_router_model": os.environ.get("EXP159_SOURCE_ROUTER_MODEL", "1"),
                   "pdf_text_cache_root": os.environ.get("EXP159_PDF_TEXT_CACHE_ROOT"),
                   "keyframe_root": os.environ.get("EXP159_KEYFRAME_ROOT"),
                   "keyframe_max_images": os.environ.get("EXP159_KEYFRAME_MAX_IMAGES", "18"),
                   "audio_asr": flags.audio_asr_on(),
                   "audio_asr_cache_root": os.environ.get("EXP159_AUDIO_ASR_CACHE_ROOT"),
                   "asr_model": a.asr_model,
                   "asr_language": a.asr_language,
                   "asr_existing_json": str(a.asr_existing_json) if a.asr_existing_json else None,
                   "asr_quality_filter": a.asr_quality_filter,
                   "asr_max_chars": a.asr_max_chars,
                   "audio_to_router": a.audio_to_router,
                   "router_asr_max_chars": a.router_asr_max_chars},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n=== DONE levers={levers or ['baseline']} mean={mean:.4f} "
          f"perfect={n_perfect}/{len(results)} zero={n_zero} elapsed={elapsed/60:.1f}min ===")
    print(f"saved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
