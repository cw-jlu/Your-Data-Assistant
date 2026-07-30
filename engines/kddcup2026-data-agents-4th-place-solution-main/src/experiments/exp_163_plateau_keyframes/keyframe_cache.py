"""Task-scoped video keyframe cache for exp_163.

Keyframes are derived inputs, so the default cache lives under /tmp and is
created at task runtime, mirroring the PDF text cache design.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from kobushi_core.benchmark.schema import PublicTask


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def keyframe_cache_root() -> Path:
    root = Path(os.environ.get("EXP163_KEYFRAME_ROOT", "/tmp/kobushi_exp163_keyframe_cache"))
    if not root.is_absolute():
        root = repo_root() / root
    return root


def _find_video(task: PublicTask) -> Path | None:
    for ext in ("*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv"):
        hits = sorted(task.context_dir.glob(f"**/{ext}"))
        if hits:
            return hits[0]
    return None


def _source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
    }


def _ahash(frame, size: int = 32):
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small > small.mean()


def _hash_distance(a, b) -> int:
    import numpy as np

    return int(np.count_nonzero(a != b))


def _resized_gray(frame, size: tuple[int, int] = (160, 90)):
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _mean_abs_diff(a, b) -> float:
    import numpy as np

    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _overlay_timestamp(frame, timestamp: float):
    import cv2

    out = frame.copy()
    label = f"t={timestamp:05.1f}s"
    cv2.rectangle(out, (10, 10), (190, 50), (0, 0, 0), -1)
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _make_contact_sheet(frames: list[tuple[float, Path]], out_path: Path, thumb_w: int = 360) -> None:
    import cv2
    import numpy as np

    images = []
    for _, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = int(h * thumb_w / w)
        images.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
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


@dataclass(frozen=True, slots=True)
class KeyframeCacheResult:
    task_id: str
    source_rel: str
    frames_dir: Path
    meta_path: Path
    contact_sheet: Path
    duration: float
    sampled: int
    kept: int
    selected: int
    source_sha1: str


def _settings() -> dict[str, object]:
    def _float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except ValueError:
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    return {
        "sample_fps": _float("EXP163_KEYFRAME_SAMPLE_FPS", 1.0),
        "hash_thresh": _int("EXP163_KEYFRAME_HASH_THRESH", 80),
        "diff_thresh": _float("EXP163_KEYFRAME_DIFF_THRESH", 7.0),
        "max_frames": _int("EXP163_KEYFRAME_MAX_IMAGES", 50),
        # exp163 plateau-detection: probe densely, keep one SETTLED frame per
        # stable slide (a held panel), skipping transition/animation frames.
        "probe_fps": _float("EXP163_PLATEAU_PROBE_FPS", 5.0),
        "stable_eps": _float("EXP163_PLATEAU_STABLE_EPS", 2.0),
        "min_stable_s": _float("EXP163_PLATEAU_MIN_STABLE_S", 1.2),
        "settle_back_s": _float("EXP163_PLATEAU_SETTLE_BACK_S", 0.3),
    }


def ensure_task_keyframes(task: PublicTask, *, force: bool = False) -> KeyframeCacheResult | None:
    video = _find_video(task)
    if video is None:
        return None

    root = keyframe_cache_root()
    frames_dir = root / task.task_id / "keyframes"
    meta_path = root / task.task_id / "keyframes.meta.json"
    contact_sheet = root / task.task_id / "contact_sheet.jpg"
    source_rel = video.resolve().relative_to(task.context_dir.resolve()).as_posix()
    fp = _source_fingerprint(video)
    settings = _settings()

    if not force and meta_path.exists() and list(frames_dir.glob("*.jpg")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("source_sha1") == fp["source_sha1"]
                and meta.get("settings") == settings
            ):
                return KeyframeCacheResult(
                    task_id=task.task_id,
                    source_rel=source_rel,
                    frames_dir=frames_dir,
                    meta_path=meta_path,
                    contact_sheet=contact_sheet,
                    duration=float(meta.get("duration", 0.0)),
                    sampled=int(meta.get("sampled", 0)),
                    kept=int(meta.get("kept", 0)),
                    selected=int(meta.get("selected", 0)),
                    source_sha1=str(fp["source_sha1"]),
                )
        except Exception:
            pass

    import cv2
    import numpy as np

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / src_fps if src_fps else 0.0

    # exp163 plateau detection: probe densely, find runs where the frame is
    # held still (a slide is on screen), and emit ONE settled frame per slide.
    # Transition/animation frames (high motion between plateaus) are skipped,
    # and the settled frame is taken near the END of each plateau so count-up /
    # fade-in animations have finished and displayed values are final.
    probe_stride = max(1, int(round(src_fps / float(settings["probe_fps"]))))
    stable_eps = float(settings["stable_eps"])
    min_stable_s = float(settings["min_stable_s"])
    settle_back_s = float(settings["settle_back_s"])

    probes: list[tuple[float, int, object]] = []  # (timestamp, frame_idx, small_gray)
    sampled = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % probe_stride == 0:
            sampled += 1
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54),
                interpolation=cv2.INTER_AREA,
            ).astype("int16")
            probes.append((frame_idx / src_fps, frame_idx, small))
        frame_idx += 1
    cap.release()

    # Segment probes into stable plateaus separated by motion (diff >= eps).
    plateaus: list[tuple[float, float]] = []  # (start_t, end_t) of held slides
    if probes:
        run_start = 0
        for k in range(1, len(probes)):
            moving = float(np.mean(np.abs(probes[k][2] - probes[k - 1][2]))) >= stable_eps
            if moving:
                if probes[k - 1][0] - probes[run_start][0] >= min_stable_s:
                    plateaus.append((probes[run_start][0], probes[k - 1][0]))
                run_start = k
        if probes[-1][0] - probes[run_start][0] >= min_stable_s:
            plateaus.append((probes[run_start][0], probes[-1][0]))

    # Choose the settled timestamp per plateau (near the end, past animations).
    targets = [max(a, b - settle_back_s) for a, b in plateaus]
    # If no plateau detected (e.g. continuous motion), fall back to uniform probes.
    if not targets and probes:
        n = min(max(1, int(settings["max_frames"])), len(probes))
        targets = [probes[int(i)][0] for i in np.linspace(0, len(probes) - 1, n).round().astype(int)]

    # Cap to max_frames, preferring the LONGEST-held plateaus (most important
    # panels are shown longest), then re-sort chronologically.
    max_frames = max(1, int(settings["max_frames"]))
    if len(targets) > max_frames and plateaus:
        order = sorted(range(len(plateaus)), key=lambda i: plateaus[i][1] - plateaus[i][0], reverse=True)
        keep_idx = sorted(order[:max_frames])
        targets = [targets[i] for i in keep_idx]
    elif len(targets) > max_frames:
        idxs = np.linspace(0, len(targets) - 1, max_frames).round().astype(int)
        targets = [targets[int(i)] for i in sorted(set(idxs))]

    # Grab the actual full-res frame at each settled timestamp.
    cap = cv2.VideoCapture(str(video))
    selected_frames: list[tuple[float, object]] = []
    for t in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * src_fps)))
        ok, frame = cap.read()
        if ok:
            selected_frames.append((t, frame.copy()))
    cap.release()
    kept_frames = selected_frames

    written: list[tuple[float, Path]] = []
    for i, (timestamp, frame) in enumerate(selected_frames):
        path = frames_dir / f"frame_{i:02d}_t{timestamp:05.1f}.jpg"
        cv2.imwrite(str(path), _overlay_timestamp(frame, timestamp), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written.append((timestamp, path))

    _make_contact_sheet(written, contact_sheet)
    meta = {
        "task_id": task.task_id,
        "source_rel": source_rel,
        "frames_dir": str(frames_dir),
        "contact_sheet": str(contact_sheet),
        "duration": duration,
        "sampled": sampled,
        "kept": len(kept_frames),
        "selected": len(written),
        "frames": [{"timestamp": ts, "path": str(path)} for ts, path in written],
        "settings": settings,
        **fp,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return KeyframeCacheResult(
        task_id=task.task_id,
        source_rel=source_rel,
        frames_dir=frames_dir,
        meta_path=meta_path,
        contact_sheet=contact_sheet,
        duration=duration,
        sampled=sampled,
        kept=len(kept_frames),
        selected=len(written),
        source_sha1=str(fp["source_sha1"]),
    )
