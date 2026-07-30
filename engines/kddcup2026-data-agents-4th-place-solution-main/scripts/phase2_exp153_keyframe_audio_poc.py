"""Phase 2 PoC: exp152-style keyframes + audio ASR, with raw video avoided.

This script intentionally does not modify exp152. It reuses exp152 components,
turns on keyframe image attachment, injects a CPU-ASR transcript into the
preamble, and optionally exposes the same combined video note to source_router.

Default PoC tasks are the video tasks where audio looked most likely to matter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

# Match exp152 production defaults, plus keyframe images. With keyframes present,
# the main agent receives image_url keyframes instead of raw video_url.
os.environ.setdefault("EXP152_KEYFRAMES", "1")
os.environ.setdefault("EXP152_VIDEO_KEYFRAME_NOTE", "1")
os.environ.setdefault("EXP152_VIDEO", "0")
os.environ.setdefault("EXP152_WATCH_VIDEO", "0")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task
from kobushi_core.model import OpenAIModelAdapter

from experiments.exp_152_final_sql_guard import flags
from experiments.exp_152_final_sql_guard.adaptive_vote import adaptive_vote
from experiments.exp_152_final_sql_guard.anti_aggregation import (
    no_agg_note,
    review_aggregation,
)
from experiments.exp_152_final_sql_guard.math_advisor import generate_formula
from experiments.exp_152_final_sql_guard.pdf_text_cache import ensure_task_pdf_cache
from experiments.exp_152_final_sql_guard.phased_agent import (
    PhasedAgentConfig,
    PhasedReActAgent,
)
from experiments.exp_152_final_sql_guard.preamble import build_preamble
from experiments.exp_152_final_sql_guard.prefix_cache import (
    prefix_cache_enabled,
    with_prefix_cache_header,
)
from experiments.exp_152_final_sql_guard.tools.registry import create_default_tool_registry
from experiments.exp_152_final_sql_guard.video_keyframe_note import (
    extract_keyframe_note,
    preamble_block as video_keyframe_preamble_block,
)


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
GOLD_ROOT = DATA_ROOT / "output"
DEFAULT_EXISTING_ASR = REPO / "artifacts" / "audio_transcript_poc" / "whisper_small_transcripts.json"
DEFAULT_TASKS = "task_6,task_24,task_34,task_39,task_48,task_54,task_57,task_60"
MAX_STEPS = 64


ASR_NOTE_EN = """# VIDEO AUDIO TRANSCRIPT (ASR)
The text below is automatic speech recognition from the briefing video and may contain errors.

Use it only to understand scope, selected tabs, metrics, filters, exclusions, ranking direction, and boundary notes.
Do not use ASR as final evidence for names, numbers, dates, IDs, or codes unless verified by tables, documents, or visible keyframes.
If ASR conflicts with verified table/document/keyframe evidence, prefer the verified evidence.
"""

ASR_NOTE_ZH = """# 视频音频转写（ASR）
以下文本由自动语音识别生成，可能包含错误。

只用它理解任务口径、选中页签、指标、筛选条件、排除项、排序方向和边界说明。
不要把 ASR 中的名称、数字、日期、ID 或代码直接作为最终答案；必须用表格、文档或关键帧可见信息核对。
如果 ASR 与可核对的表格、文档或关键帧信息冲突，以可核对信息为准。
"""

ZH_ASR_PROMPT = (
    "以下是金融、医疗或数据看板语音。常见术语：流通A股股本、流通股、股本、筛选、"
    "准入线、统计年份、二级行业、行业分类、字段、批次、导出、队列、加载、"
    "视图、口径、公司代码、净资产、监控、配置、边界样本、基金、患者。"
)

EN_ASR_PROMPT = (
    "This is an analytics dashboard briefing. Common terms include configuration, "
    "criteria, threshold, date range, scope, boundary sample, export, net foreign "
    "assets, M2, patient, procedure, fund, industry, and company code."
)


def _is_zh(text: str) -> bool:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff") >= 4


def _find_video(context_dir: Path) -> Path | None:
    for ext in ("*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv"):
        hits = sorted(context_dir.glob(f"**/{ext}"))
        if hits:
            return hits[0]
    return None


def _make_model(temp: float = 0.6, *, task_id: str | None = None) -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model=os.environ.get("EXP153_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=temp,
        extra_headers=headers,
    )


def _official_score(pred_csv: Path, gold_csv: Path) -> float:
    if not gold_csv.is_file():
        return 0.0
    try:
        ev = _evaluate_task(
            task_id=pred_csv.parent.name,
            prediction_path=pred_csv,
            gold_path=gold_csv,
            options=EvaluationOptions(),
        )
        return float(ev.official_score_lambda_0_5)
    except Exception as exc:
        print(f"  [score err] {pred_csv.parent.name}: {exc}", flush=True)
        return 0.0


def _run_advisors(question: str, task_id: str) -> tuple[str, Any]:
    if not flags.anti_agg_on():
        formula = generate_formula(question, task_id=task_id)
        anti = SimpleNamespace(
            raw="AGG: disabled", label="AGG", prompt_lang="en", no_agg=False
        )
        return formula, anti

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_formula = ex.submit(generate_formula, question, task_id=task_id)
        f_anti = ex.submit(review_aggregation, question, task_id=task_id)
        formula = f_formula.result()
        anti = f_anti.result()
    return formula, anti


def _load_existing_asr(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _format_existing_transcript(entry: dict[str, Any]) -> str:
    lines = [
        "source=existing_whisper_cache "
        f"model_language={entry.get('language', '')} "
        f"language_probability={float(entry.get('language_probability') or 0):.3f} "
        f"duration={float(entry.get('duration') or 0):.1f}s "
        f"elapsed={float(entry.get('elapsed') or 0):.1f}s"
    ]
    segments = entry.get("segments")
    if isinstance(segments, list) and segments:
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start") or 0)
            end = float(seg.get("end") or 0)
            lines.append(f"[{start:05.1f}-{end:05.1f}] {text}")
    else:
        text = str(entry.get("text", "")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _transcribe_with_whisper(
    *,
    whisper_model,
    video_path: Path,
    question: str,
    language: str,
) -> tuple[str, dict[str, Any]]:
    lang = None if language == "auto" else language
    if lang is None:
        lang = "zh" if _is_zh(question) else None
    prompt = ZH_ASR_PROMPT if lang == "zh" else EN_ASR_PROMPT
    t0 = time.time()
    segments, info = whisper_model.transcribe(
        str(video_path),
        language=lang,
        beam_size=5,
        vad_filter=True,
        initial_prompt=prompt,
    )
    lines = [
        "source=faster_whisper "
        f"model_language={info.language} "
        f"language_probability={info.language_probability:.3f} "
        f"duration={info.duration:.1f}s "
        f"elapsed={time.time() - t0:.1f}s"
    ]
    seg_payload = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(f"[{seg.start:05.1f}-{seg.end:05.1f}] {text}")
        seg_payload.append({"start": seg.start, "end": seg.end, "text": text})
    meta = {
        "source": "faster_whisper",
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": seg_payload,
    }
    return "\n".join(lines), meta


def _asr_quality(text: str) -> dict[str, Any]:
    content = "\n".join(line for line in text.splitlines() if not line.startswith("source="))
    chars = len(content.strip())
    cjk = "".join(ch for ch in content if "\u4e00" <= ch <= "\u9fff")
    bigrams = [cjk[i : i + 2] for i in range(len(cjk) - 1)]
    top_bigram = ""
    top_share = 0.0
    if bigrams:
        top_bigram, top_count = Counter(bigrams).most_common(1)[0]
        top_share = top_count / len(bigrams)
    suppress_reasons = []
    if chars < 80:
        suppress_reasons.append("too_short")
    if top_share >= 0.12 and len(cjk) >= 80:
        suppress_reasons.append("repetitive_cjk")
    return {
        "chars": chars,
        "cjk_chars": len(cjk),
        "top_bigram": top_bigram,
        "top_bigram_share": top_share,
        "suppressed": bool(suppress_reasons),
        "suppress_reasons": suppress_reasons,
    }


def _audio_preamble_block(question: str, transcript: str, *, max_chars: int) -> str:
    header = ASR_NOTE_ZH if _is_zh(question) else ASR_NOTE_EN
    text = transcript.strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return f"{header}\n{text}\n\n"


def _prepare_audio(
    *,
    task_id: str,
    task,
    task_dir: Path,
    existing_asr: dict[str, dict[str, Any]],
    whisper_model,
    args,
) -> tuple[str | None, dict[str, Any]]:
    video_path = _find_video(task.context_dir)
    if video_path is None:
        return None, {"status": "no_video"}

    transcript = None
    meta: dict[str, Any] = {"video": str(video_path)}
    asr_file = task_dir / "asr_transcript.txt"
    if args.reuse_existing_asr and task_id in existing_asr:
        transcript = _format_existing_transcript(existing_asr[task_id])
        meta.update({"status": "existing_asr", "existing_asr_path": str(args.existing_asr)})
    elif args.reuse_asr and asr_file.exists():
        transcript = asr_file.read_text(encoding="utf-8", errors="replace")
        meta.update({"status": "task_cache"})
    else:
        if whisper_model is None:
            raise RuntimeError("whisper_model is required when no reusable ASR transcript exists")
        transcript, whisper_meta = _transcribe_with_whisper(
            whisper_model=whisper_model,
            video_path=video_path,
            question=task.question,
            language=args.language,
        )
        meta.update(whisper_meta)
        meta["status"] = "transcribed"

    asr_file.write_text(transcript, encoding="utf-8")
    quality = _asr_quality(transcript)
    meta["quality"] = quality
    include = not (args.quality_filter and quality["suppressed"])
    meta["included_in_preamble"] = include
    (task_dir / "asr_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not include:
        (task_dir / "asr_suppressed.txt").write_text(
            ",".join(quality["suppress_reasons"]) + "\n",
            encoding="utf-8",
        )
        return None, meta
    return transcript, meta


def run_one(task_id: str, out_dir: Path, existing_asr: dict[str, dict[str, Any]], whisper_model, args) -> dict[str, Any]:
    t0 = time.time()
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    task = ds.get_task(task_id)
    task_dir = out_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    if flags.pdf_preprocess_on():
        try:
            pdf_results = ensure_task_pdf_cache(task)
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
                        for r in pdf_results
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            (task_dir / "pdf_preprocess_error.txt").write_text(str(exc), encoding="utf-8")

    video_keyframe_note = None
    try:
        video_keyframe_note = extract_keyframe_note(
            task, _make_model(temp=0.2, task_id=task_id)
        )
        if video_keyframe_note is not None:
            (task_dir / "video_keyframe_note.md").write_text(
                video_keyframe_note.text,
                encoding="utf-8",
            )
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
                + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        (task_dir / "video_keyframe_note_error.txt").write_text(str(exc), encoding="utf-8")

    asr_text, asr_meta = _prepare_audio(
        task_id=task_id,
        task=task,
        task_dir=task_dir,
        existing_asr=existing_asr,
        whisper_model=whisper_model,
        args=args,
    )

    formula = "NO_CALC"
    anti_agg = SimpleNamespace(
        raw="AGG: advisor_error", label="AGG", prompt_lang="en", no_agg=False
    )
    try:
        formula, anti_agg = _run_advisors(task.question, task_id)
    except Exception as exc:
        formula = f"NO_CALC ({exc})"
        anti_agg = SimpleNamespace(
            raw=f"AGG: advisor_error={exc!r}", label="AGG", prompt_lang="en", no_agg=False
        )
    (task_dir / "formula.txt").write_text(formula, encoding="utf-8")
    (task_dir / "anti_aggregation.txt").write_text(
        f"label={anti_agg.label}\nlang={anti_agg.prompt_lang}\nraw={anti_agg.raw}\n",
        encoding="utf-8",
    )

    preamble = build_preamble(task)
    preamble_text = preamble.text
    router_video_note = video_keyframe_note.text if video_keyframe_note else None

    if asr_text:
        audio_block = _audio_preamble_block(
            task.question,
            asr_text,
            max_chars=args.asr_max_chars,
        )
        preamble_text = audio_block + preamble_text
        if args.audio_to_router:
            router_video_note = (
                (router_video_note or "")
                + "\n\n# VIDEO AUDIO TRANSCRIPT (ASR; may contain errors)\n"
                + asr_text[: args.router_asr_max_chars]
            ).strip()

    if video_keyframe_note is not None:
        preamble_text = video_keyframe_preamble_block(video_keyframe_note) + preamble_text

    skip_advisor = formula.strip().upper().startswith("NO_CALC") or anti_agg.no_agg
    advisor_used = not skip_advisor
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

    (task_dir / "injected_preamble.txt").write_text(injected, encoding="utf-8")
    (task_dir / "advisor_used.txt").write_text(
        "True" if advisor_used else "False", encoding="utf-8"
    )

    m = _make_model(temp=args.temperature, task_id=task_id)
    tools = create_default_tool_registry(
        auditor_model=m,
        guard_model=_make_model(temp=0.0, task_id=task_id) if flags.final_sql_guard_on() else None,
        question_provider=lambda: task.question,
        context_dir=task.context_dir,
        anti_agg_label_provider=lambda: anti_agg.label,
    )
    agent = PhasedReActAgent(
        model=m,
        tools=tools,
        config=PhasedAgentConfig(max_steps=MAX_STEPS, min_explore_queries=3),
        preamble=injected,
        video_keyframe_note=router_video_note,
    )
    agent.trace_log_path = str(task_dir / "attempt_0.steps.log")
    result = agent.run(task)
    trace = result.to_dict()
    trace["poc_metadata"] = {
        "active_levers": flags.active_levers(),
        "asr": asr_meta,
        "audio_to_router": args.audio_to_router,
        "advisor_used": advisor_used,
        "video_keyframe_note": (
            {
                "prompt_lang": video_keyframe_note.prompt_lang,
                "n_keyframes": video_keyframe_note.n_keyframes,
                "frames": video_keyframe_note.frames,
            }
            if video_keyframe_note is not None
            else None
        ),
    }
    (task_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    if not result.answer or not result.answer.rows:
        return {
            "tid": task_id,
            "score": 0.0,
            "n_ok": 0,
            "elapsed": time.time() - t0,
            "asr_included": bool(asr_text),
            "failure_reason": result.failure_reason,
        }

    voted = adaptive_vote([result.answer])
    pred = task_dir / "prediction.csv"
    with pred.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(voted.columns)
        writer.writerows(voted.rows)
    score = _official_score(pred, GOLD_ROOT / task_id / "gold.csv")
    return {
        "tid": task_id,
        "score": score,
        "n_ok": 1,
        "cols": list(voted.columns),
        "elapsed": time.time() - t0,
        "asr_included": bool(asr_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--language", default="auto", choices=["auto", "zh", "en"])
    parser.add_argument("--reuse-asr", action="store_true")
    parser.add_argument("--reuse-existing-asr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--existing-asr", type=Path, default=DEFAULT_EXISTING_ASR)
    parser.add_argument("--quality-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--asr-max-chars", type=int, default=2600)
    parser.add_argument("--audio-to-router", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-asr-max-chars", type=int, default=1600)
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    runs = REPO / "artifacts" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    i = 1
    while (runs / f"exp_153_keyframe_audio_poc_{i:03d}").exists():
        i += 1
    out_dir = runs / f"exp_153_keyframe_audio_poc_{i:03d}"
    out_dir.mkdir(parents=True)

    os.environ.setdefault(
        "EXP152_KEYFRAME_ROOT",
        str(Path("/tmp") / "kobushi_exp153_keyframe_cache" / out_dir.name),
    )
    os.environ.setdefault(
        "EXP152_PDF_TEXT_CACHE_ROOT",
        str(Path("/tmp") / "kobushi_exp153_pdf_text_cache" / out_dir.name),
    )

    existing_asr = _load_existing_asr(args.existing_asr) if args.reuse_existing_asr else {}
    needs_whisper = any(tid not in existing_asr for tid in task_ids)
    whisper_model = None
    if needs_whisper:
        from faster_whisper import WhisperModel

        print(f"loading ASR model {args.asr_model} on CPU...", flush=True)
        whisper_model = WhisperModel(
            args.asr_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=args.cpu_threads,
        )

    print(f"=== exp153 keyframe+audio POC -> {out_dir.name} ===", flush=True)
    print(
        f"tasks={task_ids}, workers={args.workers}, prefix_cache={prefix_cache_enabled()}, "
        f"levers={flags.active_levers()}, existing_asr={len(existing_asr)}",
        flush=True,
    )
    print(f"keyframe_root={os.environ.get('EXP152_KEYFRAME_ROOT')}", flush=True)

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_one, tid, out_dir, existing_asr, whisper_model, args): tid
            for tid in task_ids
        }
        for k, fut in enumerate(as_completed(futures), 1):
            tid = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "tid": tid,
                    "score": 0.0,
                    "n_ok": 0,
                    "error": str(exc)[:400],
                    "elapsed": 0.0,
                }
            results.append(result)
            mark = "✓" if float(result.get("score", 0.0)) >= 0.999 else "✗"
            running = sum(float(r.get("score", 0.0)) for r in results) / len(results)
            print(
                f"[{k:2d}/{len(task_ids)}] {mark} {tid:<8} "
                f"score={float(result.get('score', 0.0)):.2f} "
                f"asr={result.get('asr_included')} t={float(result.get('elapsed', 0.0)):.0f}s "
                f"mean={running:.4f} {(result.get('error') or result.get('failure_reason') or '')[:80]}",
                flush=True,
            )

    elapsed = time.time() - t_start
    mean = sum(float(r.get("score", 0.0)) for r in results) / len(results) if results else 0.0
    summary = {
        "tag": "keyframe_audio_poc",
        "n_tasks": len(results),
        "mean_score": mean,
        "n_perfect": sum(float(r.get("score", 0.0)) >= 0.999 for r in results),
        "n_zero": sum(float(r.get("score", 0.0)) < 0.01 for r in results),
        "elapsed_minutes": elapsed / 60,
        "config": {
            "tasks": task_ids,
            "workers": args.workers,
            "temperature": args.temperature,
            "max_steps": MAX_STEPS,
            "asr_model": args.asr_model,
            "reuse_existing_asr": args.reuse_existing_asr,
            "existing_asr": str(args.existing_asr),
            "quality_filter": args.quality_filter,
            "asr_max_chars": args.asr_max_chars,
            "audio_to_router": args.audio_to_router,
            "router_asr_max_chars": args.router_asr_max_chars,
            "active_levers": flags.active_levers(),
            "keyframe_root": os.environ.get("EXP152_KEYFRAME_ROOT"),
            "pdf_text_cache_root": os.environ.get("EXP152_PDF_TEXT_CACHE_ROOT"),
            "prefix_cache": prefix_cache_enabled(),
        },
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"=== DONE mean={mean:.4f} perfect={summary['n_perfect']}/{len(results)} "
        f"zero={summary['n_zero']} elapsed={elapsed/60:.1f}min ===",
        flush=True,
    )
    print(f"saved {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
