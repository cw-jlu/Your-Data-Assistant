"""Task-scoped video-audio ASR support for exp_158.

ASR is an auxiliary hint only. It is never treated as final evidence without
verification against tables, documents, or visible video/keyframe content.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


ASR_NOTE_EN = """# VIDEO AUDIO TRANSCRIPT (ASR)
The text below is automatic speech recognition from the briefing video and may contain errors.

Use it only to understand scope, selected tabs, metrics, filters, exclusions, ranking direction, and boundary notes.
Do not use ASR as final evidence for names, numbers, dates, IDs, or codes unless verified by tables, documents, or visible video/keyframe evidence.
If ASR conflicts with verified table/document/video evidence, prefer the verified evidence.
"""

ASR_NOTE_ZH = """# 视频音频转写（ASR）
以下文本由自动语音识别生成，可能包含错误。

只用它理解任务口径、选中页签、指标、筛选条件、排除项、排序方向和边界说明。
不要把 ASR 中的名称、数字、日期、ID 或代码直接作为最终答案；必须用表格、文档或视频/关键帧可见信息核对。
如果 ASR 与可核对的表格、文档或视频信息冲突，以可核对信息为准。
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


def audio_asr_cache_root() -> Path:
    root = Path(os.environ.get("EXP158_AUDIO_ASR_CACHE_ROOT", "/tmp/kobushi_exp158_audio_asr_cache"))
    return root if root.is_absolute() else Path.cwd() / root


def is_zh(text: str) -> bool:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff") >= 4


def find_video(context_dir: Path) -> Path | None:
    for ext in ("*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv"):
        hits = sorted(context_dir.glob(f"**/{ext}"))
        if hits:
            return hits[0]
    return None


def _source_fingerprint(path: Path) -> dict[str, Any]:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    stat = path.stat()
    return {
        "source_sha1": h.hexdigest(),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


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


def load_existing_asr(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _transcribe_with_whisper(
    *,
    whisper_model,
    video_path: Path,
    question: str,
    language: str,
) -> tuple[str, dict[str, Any]]:
    lang = None if language == "auto" else language
    if lang is None:
        lang = "zh" if is_zh(question) else None
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
        "elapsed": time.time() - t0,
        "segments": seg_payload,
    }
    return "\n".join(lines), meta


def asr_quality(text: str) -> dict[str, Any]:
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


def audio_preamble_block(question: str, transcript: str, *, max_chars: int) -> str:
    header = ASR_NOTE_ZH if is_zh(question) else ASR_NOTE_EN
    text = transcript.strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return f"{header}\n{text}\n\n"


def prepare_task_audio(
    *,
    task_id: str,
    task,
    task_dir: Path,
    whisper_model=None,
    whisper_lock=None,
    language: str = "auto",
    quality_filter: bool = True,
    existing_asr: dict[str, dict[str, Any]] | None = None,
    existing_asr_path: Path | None = None,
    reuse_cache: bool = True,
) -> tuple[str | None, dict[str, Any]]:
    video_path = find_video(task.context_dir)
    if video_path is None:
        return None, {"status": "no_video"}

    fp = _source_fingerprint(video_path)
    root = audio_asr_cache_root()
    cache_dir = root / task_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_text = cache_dir / "asr_transcript.txt"
    cache_meta = cache_dir / "asr_meta.json"
    transcript = None
    meta: dict[str, Any] = {
        "video": str(video_path),
        "cache_dir": str(cache_dir),
        **fp,
    }

    if existing_asr and task_id in existing_asr:
        transcript = _format_existing_transcript(existing_asr[task_id])
        meta.update({"status": "existing_asr", "existing_asr_path": str(existing_asr_path or "")})
    elif reuse_cache and cache_text.exists() and cache_meta.exists():
        try:
            cached_meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        except Exception:
            cached_meta = {}
        if cached_meta.get("source_sha1") == fp["source_sha1"]:
            transcript = cache_text.read_text(encoding="utf-8", errors="replace")
            meta.update(cached_meta)
            meta["status"] = "cache"
    if transcript is None:
        if whisper_model is None:
            raise RuntimeError("whisper_model is required when ASR cache is missing")
        if whisper_lock is None:
            transcript, whisper_meta = _transcribe_with_whisper(
                whisper_model=whisper_model,
                video_path=video_path,
                question=task.question,
                language=language,
            )
        else:
            with whisper_lock:
                transcript, whisper_meta = _transcribe_with_whisper(
                    whisper_model=whisper_model,
                    video_path=video_path,
                    question=task.question,
                    language=language,
                )
        meta.update(whisper_meta)
        meta["status"] = "transcribed"

    cache_text.write_text(transcript, encoding="utf-8")
    quality = asr_quality(transcript)
    meta["quality"] = quality
    include = not (quality_filter and quality["suppressed"])
    meta["included_in_preamble"] = include
    cache_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (task_dir / "asr_transcript.txt").write_text(transcript, encoding="utf-8")
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
