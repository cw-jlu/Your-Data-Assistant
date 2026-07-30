"""Phase 2 exp_167 per-lever ABLATION bench.

Each lever is toggled by an env var (see experiments.exp_172_ehr_distinct.flags):
  EXP172_ANSWER_SHAPE=1  → ① DISTINCT→JOIN-fanout-only + NULL/BLANK preserve
  EXP172_PROSE=1         → ② pypdf read + prose_docs inventory + 404→doc redirect + protocol
  EXP172_VIDEO=1         → ③ pre-extract video summary into preamble (no raw video in loop)
  EXP172_KEYFRAMES=1     → attach plateau-selected keyframes as images (no raw video)
  EXP172_VIDEO_KEYFRAME_NOTE=1 → extract visible keyframe information into preamble/router
  EXP172_VISUAL_ATTACH=0 → keep keyframe text note but attach no images/video
  EXP172_WATCH_VIDEO=1   → expose raw-video inspection as an EXPLORE tool

ALL flags off ⇒ identical pipeline to exp_143_dynamic_doc (sanity baseline).

Run dir is named by the active levers, e.g.:
  EXP172_ANSWER_SHAPE=1 python scripts/phase2_exp166_ablation.py   → exp_167_answer_shape_001
  EXP172_PROSE=1        python scripts/phase2_exp166_ablation.py   → exp_167_prose_001
  EXP172_KEYFRAMES=1    python scripts/phase2_exp166_ablation.py   → exp_167_keyframes_001
  EXP172_WATCH_VIDEO=1  python scripts/phase2_exp166_ablation.py   → exp_167_watch_video_001
  (no flags)            python scripts/phase2_exp166_ablation.py   → exp_167_baseline_001

att=1, workers=12 by default (exploration class). Scored via kobushi_core.eval.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_172_ehr_distinct import flags
from experiments.exp_172_ehr_distinct.phased_agent import PhasedReActAgent, PhasedAgentConfig
from experiments.exp_172_ehr_distinct.preamble import build_preamble
from experiments.exp_172_ehr_distinct.tools.registry import create_default_tool_registry
from experiments.exp_172_ehr_distinct.domain_router import classify_domain
from experiments.exp_172_ehr_distinct.domain_prompt import domain_note, modality_note
from experiments.exp_172_ehr_distinct.adaptive_vote import adaptive_vote
from experiments.exp_172_ehr_distinct.anti_aggregation import no_agg_note, review_aggregation
from experiments.exp_172_ehr_distinct.math_advisor import generate_formula
from experiments.exp_172_ehr_distinct.video_keyframe_note import (
    extract_keyframe_note,
    preamble_block as video_keyframe_preamble_block,
)
from experiments.exp_172_ehr_distinct.video_summarizer import summarize_video
from experiments.exp_172_ehr_distinct.prefix_cache import (
    prefix_cache_enabled,
    with_prefix_cache_header,
)

# DATA_ROOT defaults to the Phase 2 demo set; override via EXP172_DATA_ROOT
# (e.g. data/public for a Phase 1 regression check). Expects <root>/input + <root>/output.
DATA_ROOT = Path(os.environ.get("EXP172_DATA_ROOT") or (REPO / "data" / "phase2_demo" / "demo_samples_phase2"))
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


def _run_advisors(question: str, tid: str, run_formula: bool = True) -> tuple[str, object]:
    """Run independent planning sub-agents for one task.

    `run_formula=False` skips the math-formula advisor (no LLM call), matching the
    runner: with EXP172_MATH_OTHER_ONLY the formula fires only when domain='other'.
    """
    disabled_anti = SimpleNamespace(
        raw="AGG: disabled", label="AGG", prompt_lang="en", no_agg=False
    )
    if not run_formula:
        anti = review_aggregation(question, task_id=tid) if flags.anti_agg_on() else disabled_anti
        return "NO_CALC (domain!=other)", anti

    if not flags.anti_agg_on():
        return generate_formula(question, task_id=tid), disabled_anti

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_formula = ex.submit(generate_formula, question, task_id=tid)
        f_anti = ex.submit(review_aggregation, question, task_id=tid)
        formula = f_formula.result()
        anti = f_anti.result()
    return formula, anti


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
            from experiments.exp_172_ehr_distinct.pdf_text_cache import ensure_task_pdf_cache
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
            from experiments.exp_172_ehr_distinct.keyframe_cache import ensure_task_keyframes
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

    # Domain router (deterministic) — computed once and reused for: the math
    # gate, the domain/modality preamble notes, and the column-auditor gate.
    # This matches the runner/submission path so the bench measures what ships.
    domain_for_audit = "other"
    domain_note_text = ""
    modality_note_text = ""
    if flags.domain_router_on():
        try:
            dr = classify_domain(task)
            domain_for_audit = dr.domain
            domain_note_text = domain_note(dr.domain)
            modality_note_text = modality_note(task.context_dir)
        except Exception:
            domain_for_audit = "other"

    run_formula = (not flags.math_other_only_on()) or domain_for_audit == "other"
    formula = "NO_CALC"
    try:
        formula, anti_agg = _run_advisors(task.question, tid, run_formula=run_formula)
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
            # exp_170: prose-extraction startup pass (once per task; idempotent).
            # Materialises prose__<doc> views under a runtime dir (not context_dir),
            # BEFORE build_preamble so the catalog + profile pick them up.
            if flags.prose_extract_on():
                import os as _os2
                from experiments.exp_172_ehr_distinct.prose_extractor import extract_prose_tables, select_prose_docs, prose_gate
                _root = _os2.environ.setdefault("EXP172_EXTRACTED_ROOT", str(out_dir / "_prose_extracted"))
                if not (Path(_root) / tid / "_manifest.json").is_file():
                    _sm = make_model(temp=0.0, task_id=tid)
                    _df = _os2.environ.get("EXP172_PROSE_DOC_FILTER")  # optional manual override
                    # design ③: deterministic gate (要/不要) BEFORE select+extract.
                    # gate off == exp_170 always-extract. 不要 -> skip (no prose view, no decoy).
                    _do_extract = True
                    if flags.prose_gate_on() and not _df:
                        _needed, _gdocs = prose_gate(task, _sm)
                        _do_extract = _needed
                        print(f"  [prose_gate] task={tid} needed={_needed} docs={_gdocs}", flush=True)
                    if _do_extract:
                        _sel = None if _df else select_prose_docs(task, _sm)  # LLM doc-router
                        extract_prose_tables(task, Path(_root), _sm, doc_filter=_df, only_docs=_sel)
            # domain_for_audit / domain_note_text / modality_note_text were
            # computed once above (before the math gate); reuse them here.
            preamble = build_preamble(task, video_summary=video_summary)
            preamble_text = preamble.text
            if video_keyframe_note is not None:
                preamble_text = video_keyframe_preamble_block(video_keyframe_note) + preamble_text
            if modality_note_text or domain_note_text:
                preamble_text = modality_note_text + domain_note_text + preamble_text
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
                domain_provider=lambda d=domain_for_audit: d,
            )
            agent = PhasedReActAgent(
                model=m, tools=tools,
                config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
                preamble=injected,
                video_keyframe_note=video_keyframe_note.text if video_keyframe_note else None,
                domain=domain_for_audit,
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
        return {"tid": tid, "formula": formula, "anti_aggregation": anti_agg.raw,
                "anti_agg_label": anti_agg.label, "advisor_used": advisor_used,
                "score": 0.0, "n_ok": 0, "elapsed": time.time() - t0}
    voted = adaptive_vote(answers)
    if voted is None or not voted.rows:
        return {"tid": tid, "formula": formula, "anti_aggregation": anti_agg.raw,
                "anti_agg_label": anti_agg.label, "advisor_used": advisor_used,
                "score": 0.0, "n_ok": len(answers), "elapsed": time.time() - t0}
    pred = task_dir / "prediction.csv"
    with open(pred, "w", newline="") as f:
        w = csv.writer(f); w.writerow(voted.columns); [w.writerow(r) for r in voted.rows]
    score = _official_score(pred, GOLD_ROOT / tid / "gold.csv")
    return {"tid": tid, "formula": formula, "anti_aggregation": anti_agg.raw,
            "anti_agg_label": anti_agg.label, "advisor_used": advisor_used,
            "score": score, "n_ok": len(answers),
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
    while (runs / f"exp_172_{tag}_{i:03d}").exists():
        i += 1
    out_dir = runs / f"exp_172_{tag}_{i:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("EXP172_DATA_ROOT"):
        try:
            (out_dir / "_task_root").write_text(os.environ["EXP172_DATA_ROOT"].strip(), encoding="utf-8")
        except Exception:
            pass

    if flags.pdf_preprocess_on() and not os.environ.get("EXP172_PDF_TEXT_CACHE_ROOT"):
        os.environ["EXP172_PDF_TEXT_CACHE_ROOT"] = str(
            Path("/tmp") / "kobushi_exp166_pdf_text_cache" / out_dir.name
        )
    if (flags.keyframes_on() or flags.video_keyframe_note_on()) and not os.environ.get("EXP172_KEYFRAME_ROOT"):
        os.environ["EXP172_KEYFRAME_ROOT"] = str(
            Path("/tmp") / "kobushi_exp166_keyframe_cache" / out_dir.name
        )

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    if a.tasks:
        task_ids = [t.strip() for t in a.tasks.split(",") if t.strip()]
    else:
        task_ids = sorted([t.task_id for t in ds.iter_tasks()], key=lambda x: int(x.split("_")[1]))

    print(f"=== exp_171 ABLATION levers={levers or ['baseline']} → {out_dir.name} ===")
    print(f"  workers={a.workers}, att={N_ATTEMPTS}, max_steps={MAX_STEPS}, T=0.6, "
          f"{len(task_ids)} tasks, prefix_cache={prefix_cache_enabled()}", flush=True)
    if flags.pdf_preprocess_on():
        print(f"  pdf_text_cache_root={os.environ.get('EXP172_PDF_TEXT_CACHE_ROOT')}", flush=True)
    if flags.keyframes_on() or flags.video_keyframe_note_on():
        print(f"  keyframe_root={os.environ.get('EXP172_KEYFRAME_ROOT')}", flush=True)

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
                   "prefix_cache_namespace": os.environ.get("EXP172_PREFIX_CACHE_NAMESPACE", "kobushi-phase2-exp166"),
                   "source_router": flags.source_router_on(),
                   "anti_agg": flags.anti_agg_on(),
                   "anti_agg_sql_guard": flags.anti_agg_sql_guard_on(),
                   "keyframes": flags.keyframes_on(),
                   "domain_router": flags.domain_router_on(),
                   "math_other_only": flags.math_other_only_on(),
                   "video_keyframe_note": flags.video_keyframe_note_on(),
                   "video_keyframe_note_max_chars": os.environ.get("EXP172_VIDEO_KEYFRAME_NOTE_MAX_CHARS", "1800"),
                   "watch_video": flags.watch_video_on(),
                   "source_router_model": os.environ.get("EXP172_SOURCE_ROUTER_MODEL", "1"),
                   "pdf_text_cache_root": os.environ.get("EXP172_PDF_TEXT_CACHE_ROOT"),
                   "keyframe_root": os.environ.get("EXP172_KEYFRAME_ROOT"),
                   "keyframe_max_images": os.environ.get("EXP172_KEYFRAME_MAX_IMAGES", "50")},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n=== DONE levers={levers or ['baseline']} mean={mean:.4f} "
          f"perfect={n_perfect}/{len(results)} zero={n_zero} elapsed={elapsed/60:.1f}min ===")
    print(f"saved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
