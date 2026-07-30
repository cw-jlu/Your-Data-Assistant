from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_167_domain_pruner.keyframe_cache import ensure_task_keyframes


_SYSTEM_EN = """You extract visible information from keyframes for a data question.
Do not classify, judge source priority, decide whether SQL is needed, or decide
whether the visible rows are complete. Do not give advice.

Task: from the attached keyframes, write a concise extraction of ALL visible
information that is relevant to the question.

Rules:
- Use only visible information in the keyframes.
- Prefer short exact quotes and numbers as they appear on screen.
- Include filters, thresholds, dates, selected options, field names, visible rows,
  table values, dashboard labels, notes, warnings, and instructions if relevant.
- If a visible row/table/note may matter, include it exactly; do not interpret it
  as final, partial, example, or complete.
- Do not add conclusion bullets, hints, or "based on the visible information"
  reasoning. Do not infer that a row qualifies unless that exact status is shown.
- If a count/value appears in a side note or background panel, copy it as a
  side note/background value only; do not connect it to the final question.
- If text is blurry or uncertain, mark it as [unclear] instead of guessing.
- Do not mention gold labels, previous runs, traces, scores, or SQL.

Output plain text only, with short bullets. No JSON."""


_SYSTEM_ZH = """你负责从关键帧中抽取可见信息，用于回答数据问题。
不要分类，不要判断 source priority，不要判断是否需要 SQL，也不要判断可见行是否完整。
不要给建议。

任务：根据所附关键帧，简洁列出所有与问题相关的屏幕可见信息。

规则：
- 只能使用关键帧中可见的信息。
- 尽量按屏幕原文摘录短语、数字和标签。
- 包括相关的筛选条件、阈值、日期、选中项、字段名、可见行、表格取值、看板标签、说明、警告和操作提示。
- 如果某个可见行/表格/说明可能相关，就按可见内容列出；不要解释成最终、部分、示例或完整。
- 不要添加结论、提示或“根据可见信息”的推理。除非屏幕原文明确显示通过/符合，
  否则不要推断某一行符合条件。
- 如果数量或数值出现在 side note 或 background 面板中，只按背景/旁注摘录；
  不要把它连接到最终问题。
- 如果文字模糊或不确定，写 [unclear]，不要猜。
- 不要提到 gold、历史 run、trace、分数或 SQL。

只输出普通文本短项目符号。不要 JSON。"""


@dataclass(frozen=True, slots=True)
class KeyframeNoteResult:
    text: str
    prompt_lang: str
    n_keyframes: int
    frames: list[str]
    meta: dict[str, object]


def _is_zh(text: str) -> bool:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff") >= 4


def _max_images() -> int:
    try:
        return max(1, int(os.environ.get("EXP167_KEYFRAME_MAX_IMAGES", "18")))
    except ValueError:
        return 18


def _select_frames(frames_dir: Path, max_images: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) <= max_images:
        return frames
    if max_images <= 1:
        return frames[:1]
    idxs = [round(i * (len(frames) - 1) / (max_images - 1)) for i in range(max_images)]
    return [frames[i] for i in sorted(set(idxs))]


def extract_keyframe_note(task: PublicTask, model: ModelAdapter) -> KeyframeNoteResult | None:
    """Extract a plain text note from task keyframes, or None if no video exists."""
    keyframes = ensure_task_keyframes(task)
    if keyframes is None:
        return None

    frames = _select_frames(keyframes.frames_dir, _max_images())
    if not frames:
        return None

    lang = "zh" if _is_zh(task.question) else "en"
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"Task: {task.task_id}\n"
                f"Question: {task.question}\n\n"
                "Extract only visible task-relevant information from the attached keyframes."
            ),
        }
    ]
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        content.append({"type": "text", "text": f"Keyframe: {frame.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    raw = model.complete(
        [
            ModelMessage(role="system", content=_SYSTEM_ZH if lang == "zh" else _SYSTEM_EN),
            ModelMessage(role="user", content=content),
        ],
        enable_thinking=False,
        max_tokens=1400,
    )
    text = (raw or "").strip()
    if not text:
        return None

    return KeyframeNoteResult(
        text=text,
        prompt_lang=lang,
        n_keyframes=len(frames),
        frames=[str(p) for p in frames],
        meta={
            "source_rel": keyframes.source_rel,
            "frames_dir": str(keyframes.frames_dir),
            "meta_path": str(keyframes.meta_path),
            "contact_sheet": str(keyframes.contact_sheet),
            "duration": keyframes.duration,
            "sampled": keyframes.sampled,
            "kept": keyframes.kept,
            "selected": keyframes.selected,
            "source_sha1": keyframes.source_sha1,
        },
    )


def preamble_block(note: KeyframeNoteResult, *, max_chars: int | None = None) -> str:
    if max_chars is None:
        try:
            max_chars = int(os.environ.get("EXP167_VIDEO_KEYFRAME_NOTE_MAX_CHARS", "1800"))
        except ValueError:
            max_chars = 1800
    text = note.text.strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return (
        "# VIDEO KEYFRAME EXTRACTION\n"
        "Plain extraction of visible task-relevant information from video keyframes only.\n"
        "This is not a final-answer judgment and does not decide whether SQL is needed.\n\n"
        f"{text}\n\n"
    )
