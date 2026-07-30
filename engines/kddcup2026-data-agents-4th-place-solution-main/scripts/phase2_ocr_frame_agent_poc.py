"""PoC: keyframe content extraction with mechanical OCR hints.

This does not change the benchmark path. It reads already-extracted keyframes
and RapidOCR output, then asks the VLM to transcribe each frame faithfully with
the OCR text as a hint.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import ModelMessage, OpenAIModelAdapter

from experiments.exp_150_projection_pruner.prefix_cache import with_prefix_cache_header


DATA_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2"
INPUT_ROOT = DATA_ROOT / "input"
DEFAULT_KEYFRAME_ROOT = Path(
    "/tmp/kobushi_exp150_keyframe_cache/"
    "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_007"
)
DEFAULT_RAPIDOCR_ROOT = REPO / "artifacts" / "runs" / "exp_150_keyframe_rapidocr_poc_001"

SYSTEM_PROMPT = """You are an OCR assistant for video keyframes.

You receive keyframe images and raw OCR text from a mechanical OCR engine.
Use the raw OCR text only as a hint to read the images more accurately.

Write down the visible content of each frame as faithfully as possible.
Preserve names, numbers, codes, labels, table text, and on-screen UI text.
If a frame contains a table or list, keep its visible structure in plain text.

Do not answer the user's question.
Do not infer content that is not visible in the frame.
Do not combine information across frames.
Do not decide whether the frame contains a final answer.

Output plain text only, grouped by frame name."""


def make_model(*, task_id: str) -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        f"ocr-frame-agent-{task_id}",
    )
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.2,
        extra_headers=headers,
    )


def image_part(path: Path) -> dict[str, object]:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def load_rapidocr_by_frame(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for frame in data:
        name = Path(str(frame.get("frame", ""))).name
        lines = []
        for item in frame.get("items", []) or []:
            text = str(item.get("text", "")).strip()
            if text:
                lines.append(text)
        out[name] = lines
    return out


def select_frames(frames_dir: Path, max_images: int) -> list[Path]:
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) <= max_images:
        return frames
    if max_images <= 1:
        return frames[:1]
    idxs = [round(i * (len(frames) - 1) / (max_images - 1)) for i in range(max_images)]
    return [frames[i] for i in sorted(set(idxs))]


def run_one(task_id: str, args: argparse.Namespace, ds: DABenchPublicDataset, out_root: Path) -> dict[str, object]:
    t0 = time.time()
    task = ds.get_task(task_id)
    task_dir = out_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = args.keyframe_root / task_id / "keyframes"
    frames = select_frames(frames_dir, args.max_images)
    if not frames:
        raise FileNotFoundError(f"No keyframes found: {frames_dir}")

    ocr_path = args.rapidocr_root / task_id / "rapidocr.json"
    ocr_by_frame = load_rapidocr_by_frame(ocr_path)

    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"Task: {task_id}\n"
                f"Question for context only: {task.question}\n\n"
                "For each attached keyframe, write the visible frame content. "
                "Use the mechanical OCR text only as a reading hint.\n"
            ),
        }
    ]
    for frame in frames:
        ocr_lines = ocr_by_frame.get(frame.name, [])
        hint = "\n".join(f"- {line}" for line in ocr_lines) if ocr_lines else "(none)"
        content.append({"type": "text", "text": f"\nFrame: {frame.name}\nMechanical OCR hint:\n{hint}"})
        content.append(image_part(frame))

    model = make_model(task_id=task_id)
    raw = model.complete(
        [
            ModelMessage(role="system", content=SYSTEM_PROMPT),
            ModelMessage(role="user", content=content),
        ],
        enable_thinking=False,
        max_tokens=args.max_tokens,
    )
    text = raw.strip()
    (task_dir / "ocr_frame_agent_note.md").write_text(text + "\n", encoding="utf-8")
    (task_dir / "prompt_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (task_dir / "frames_used.txt").write_text(
        "\n".join(str(p) for p in frames) + "\n", encoding="utf-8"
    )
    return {
        "task_id": task_id,
        "frames": len(frames),
        "chars": len(text),
        "elapsed": round(time.time() - t0, 3),
        "out": str(task_dir / "ocr_frame_agent_note.md"),
    }


def next_out_dir() -> Path:
    runs = REPO / "artifacts" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    i = 1
    while (runs / f"exp_150_ocr_frame_agent_poc_{i:03d}").exists():
        i += 1
    out = runs / f"exp_150_ocr_frame_agent_poc_{i:03d}"
    out.mkdir()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="task_7,task_26,task_39")
    parser.add_argument("--keyframe-root", type=Path, default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--rapidocr-root", type=Path, default=DEFAULT_RAPIDOCR_ROOT)
    parser.add_argument("--max-images", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=5000)
    args = parser.parse_args()

    out_root = next_out_dir()
    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]

    print(f"=== OCR frame agent PoC -> {out_root} ===", flush=True)
    results = []
    for task_id in tasks:
        print(f"[start] {task_id}", flush=True)
        result = run_one(task_id, args, ds, out_root)
        results.append(result)
        print(
            f"[done] {task_id} frames={result['frames']} chars={result['chars']} "
            f"elapsed={result['elapsed']}s",
            flush=True,
        )

    (out_root / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary: {out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
