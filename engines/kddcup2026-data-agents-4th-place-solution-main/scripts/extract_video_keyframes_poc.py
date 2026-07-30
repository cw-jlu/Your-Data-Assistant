"""Extract unique dashboard keyframes from Phase 2 briefing videos.

This is a local-only PoC: no model/API calls. It samples frames at a fixed FPS,
keeps frames that are visually different from the previously kept frame, writes
timestamped JPEGs, and creates a contact sheet for quick inspection.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = ROOT / "data" / "phase2_demo" / "demo_samples_phase2" / "input"


def ahash(frame: np.ndarray, size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small > small.mean()


def hash_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def resized_gray(frame: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def overlay_timestamp(frame: np.ndarray, timestamp: float) -> np.ndarray:
    out = frame.copy()
    label = f"t={timestamp:05.1f}s"
    cv2.rectangle(out, (10, 10), (190, 50), (0, 0, 0), -1)
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_contact_sheet(frames: list[tuple[float, Path]], out_path: Path, thumb_w: int = 360) -> None:
    images = []
    for ts, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = int(h * thumb_w / w)
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        images.append(thumb)
    if not images:
        return

    cols = 3
    rows = math.ceil(len(images) / cols)
    cell_h = max(img.shape[0] for img in images)
    sheet = np.full((rows * cell_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y = r * cell_h
        x = c * thumb_w
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
    cv2.imwrite(str(out_path), sheet)


def extract(task_id: str, out_root: Path, sample_fps: float, hash_thresh: int, diff_thresh: float, max_frames: int) -> dict:
    video = INPUT_ROOT / task_id / "context" / "video" / "briefing.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    out_dir = out_root / task_id / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / src_fps if src_fps else 0.0
    stride = max(1, int(round(src_fps / sample_fps)))

    kept: list[tuple[float, Path]] = []
    last_hash = None
    last_gray = None
    sampled = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        timestamp = frame_idx / src_fps
        sampled += 1
        cur_hash = ahash(frame)
        cur_gray = resized_gray(frame)
        keep = False
        if last_hash is None:
            keep = True
        else:
            hd = hash_distance(cur_hash, last_hash)
            md = mean_abs_diff(cur_gray, last_gray)
            keep = hd >= hash_thresh or md >= diff_thresh

        if keep:
            out_img = overlay_timestamp(frame, timestamp)
            path = out_dir / f"frame_{len(kept):02d}_t{timestamp:05.1f}.jpg"
            cv2.imwrite(str(path), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            kept.append((timestamp, path))
            last_hash = cur_hash
            last_gray = cur_gray
        frame_idx += 1

    cap.release()

    if len(kept) > max_frames:
        # Keep uniform coverage across the selected unique frames.
        idxs = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        selected = [kept[int(i)] for i in sorted(set(idxs))]
    else:
        selected = kept

    sheet = out_root / task_id / "contact_sheet.jpg"
    make_contact_sheet(selected, sheet)

    return {
        "task_id": task_id,
        "video": str(video),
        "src_fps": src_fps,
        "duration": duration,
        "sampled": sampled,
        "kept": len(kept),
        "selected": len(selected),
        "contact_sheet": str(sheet),
        "frames": [{"timestamp": ts, "path": str(path)} for ts, path in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="comma-separated task ids")
    parser.add_argument("--out", default="artifacts/video_keyframes_poc")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--hash-thresh", type=int, default=80)
    parser.add_argument("--diff-thresh", type=float, default=7.0)
    parser.add_argument("--max-frames", type=int, default=18)
    args = parser.parse_args()

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for task_id in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        r = extract(task_id, out_root, args.sample_fps, args.hash_thresh, args.diff_thresh, args.max_frames)
        results.append(r)
        print(
            f"{task_id}: duration={r['duration']:.1f}s sampled={r['sampled']} "
            f"unique={r['kept']} selected={r['selected']} sheet={r['contact_sheet']}"
        )

    import json

    (out_root / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
