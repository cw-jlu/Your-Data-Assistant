"""Extract and save frames that are unique to the new config vs baseline."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_agent_baseline.run.video_preprocessor import extract_stable_frames

# Configs
BASELINE = (0.01, 25, 1.0, "baseline")
NEW      = (0.01, 20, 0.0, "new")

# Output directory for extra frames
OUT_DIR = Path("artifacts/extra_frames_new_config")
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

VIDEO_DIR = Path("data/input")
videos = sorted(VIDEO_DIR.glob("task_*/context/video/briefing.mp4"))
print(f"Processing {len(videos)} videos...\n")

for video_path in videos:
    task_id = video_path.parent.parent.parent.name

    # Run both configs
    tmp0 = Path(tempfile.mkdtemp(prefix=f"baseline_{task_id}_"))
    tmp1 = Path(tempfile.mkdtemp(prefix=f"new_{task_id}_"))

    m0 = extract_stable_frames(
        video_path=video_path, output_dir=tmp0,
        sample_fps=2.0, diff_threshold=0.01, pixel_delta=25,
        min_stable_duration=1.0, resize_width=320, jpg_quality=95,
    )
    m1 = extract_stable_frames(
        video_path=video_path, output_dir=tmp1,
        sample_fps=2.0, diff_threshold=0.01, pixel_delta=20,
        min_stable_duration=0.0, resize_width=320, jpg_quality=95,
    )

    c0, c1 = m0["saved_image_count"], m1["saved_image_count"]

    if c1 > c0:
        # Find frames unique to new config
        # Compare by representative time (rounded to 3 decimals)
        times0 = {s["representative_sec"] for s in m0["segments"]}
        extra_segments = [s for s in m1["segments"] if s["representative_sec"] not in times0]

        print(f"  {task_id}: {c0} → {c1}  (+{c1 - c0}),  unique frames:")
        task_dir = OUT_DIR / task_id
        task_dir.mkdir(exist_ok=True)
        for seg in extra_segments:
            src = tmp1 / seg["filename"]
            dst = task_dir / f"t{seg['representative_sec']:07.2f}s_{seg['filename']}"
            shutil.copy2(src, dst)
            print(f"    t={seg['representative_sec']:.2f}s  dur={seg['duration_sec']:.2f}s  →  {dst.name}")

        # Also copy baseline frames for comparison
        bl_dir = task_dir / "_baseline"
        bl_dir.mkdir(exist_ok=True)
        for seg in m0["segments"]:
            src = tmp0 / seg["filename"]
            if src.exists():
                shutil.copy2(src, bl_dir / f"t{seg['representative_sec']:07.2f}s_{seg['filename']}")

    elif c1 < c0:
        print(f"  {task_id}: {c0} → {c1}  (-{c0 - c1})  [lost frames]")
    else:
        print(f"  {task_id}: {c0} → {c1}  (same)")

    shutil.rmtree(tmp0, ignore_errors=True)
    shutil.rmtree(tmp1, ignore_errors=True)

print(f"\nDone. Extra frames saved to: {OUT_DIR}")
