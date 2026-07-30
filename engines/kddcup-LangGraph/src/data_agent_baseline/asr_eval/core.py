from __future__ import annotations

import hashlib
import json
import os
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class AsrSample:
    task_id: str
    video_path: Path
    source_relative_path: str
    source_sha256: str


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_asr_samples(
    dataset_root: Path,
    *,
    task_ids: list[str] | tuple[str, ...] | None = None,
) -> list[AsrSample]:
    root = dataset_root.resolve()
    input_root = root / "input" if (root / "input").is_dir() else root
    if not input_root.is_dir():
        raise FileNotFoundError(f"ASR dataset root does not exist: {input_root}")

    requested = tuple(dict.fromkeys(str(item).strip() for item in (task_ids or ()) if str(item).strip()))
    task_dirs = (
        [input_root / task_id for task_id in requested]
        if requested
        else sorted(
            (path for path in input_root.glob("task_*") if path.is_dir()),
            key=lambda path: _task_sort_key(path.name),
        )
    )

    samples: list[AsrSample] = []
    missing_tasks: list[str] = []
    for task_dir in task_dirs:
        if not task_dir.is_dir():
            missing_tasks.append(task_dir.name)
            continue
        videos = sorted((task_dir / "context" / "video").glob("briefing.mp4"))
        if len(videos) > 1:
            raise ValueError(f"Task {task_dir.name} has more than one briefing.mp4.")
        if not videos:
            continue
        video_path = videos[0].resolve()
        samples.append(
            AsrSample(
                task_id=task_dir.name,
                video_path=video_path,
                source_relative_path=video_path.relative_to(input_root).as_posix(),
                source_sha256=sha256_file(video_path),
            )
        )

    if missing_tasks:
        raise FileNotFoundError(f"Requested task directories do not exist: {', '.join(missing_tasks)}")
    if not samples:
        scope = ", ".join(requested) if requested else str(input_root)
        raise FileNotFoundError(f"No briefing.mp4 samples found for: {scope}")
    return samples


def _task_sort_key(task_id: str) -> tuple[int, int | str]:
    suffix = task_id.removeprefix("task_")
    return (0, int(suffix)) if suffix.isdigit() else (1, task_id)


def classify_asr_run(*, run_id: str, sample_count: int, stage: str | None) -> str:
    normalized = run_id.casefold()
    if "speed" in normalized or "threads" in normalized:
        return "speed-tests"
    if "smoke" in normalized or "task" in normalized or sample_count < 30:
        return "smoke-tests"
    if stage in {
        "medium-baseline",
        "tiny-route",
        "dynamic-prompt",
        "dynamic-plus-ui",
    }:
        return "prompt-ablation"
    return "model-reproduction"


def find_asr_run_dir(output_root: Path, run_id: str) -> Path | None:
    runs_root = output_root / "runs"
    legacy = runs_root / run_id
    if legacy.is_dir():
        return legacy
    matches = sorted(
        {
            manifest.parent.resolve()
            for manifest in runs_root.glob(f"*/{run_id}/manifest.json")
            if manifest.parent.name == run_id
        }
    )
    if len(matches) > 1:
        raise ValueError(
            f"ASR run_id {run_id!r} exists in multiple categories: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decode_to_pcm_wav(source_path: Path, destination_path: Path) -> None:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except ImportError as exc:
        raise RuntimeError(
            "PyAV is required to create canonical ASR audio. Install the project dependencies first."
        ) from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    try:
        with av.open(str(source_path)) as container:
            if not container.streams.audio:
                raise RuntimeError(f"Video has no audio stream: {source_path}")
            stream = container.streams.audio[0]
            resampler = AudioResampler(format="s16", layout="mono", rate=16000)
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                for frame in container.decode(stream):
                    for converted in resampler.resample(frame):
                        output.writeframes(converted.to_ndarray().tobytes())
                for converted in resampler.resample(None):
                    output.writeframes(converted.to_ndarray().tobytes())
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_canonical_audio(
    sample: AsrSample,
    audio_cache_root: Path,
    *,
    decoder: Callable[[Path, Path], None] | None = None,
) -> tuple[Path, str, float]:
    audio_path = audio_cache_root / f"{sample.task_id}.wav"
    metadata_path = audio_cache_root / f"{sample.task_id}.json"
    existing_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_metadata = {}

    if (
        audio_path.is_file()
        and existing_metadata.get("source_sha256") == sample.source_sha256
        and existing_metadata.get("audio_sha256") == sha256_file(audio_path)
    ):
        return (
            audio_path,
            str(existing_metadata["audio_sha256"]),
            float(existing_metadata["duration_seconds"]),
        )

    (decoder or _decode_to_pcm_wav)(sample.video_path, audio_path)
    _validate_canonical_wav(audio_path)
    audio_sha256 = sha256_file(audio_path)
    duration_seconds = canonical_audio_duration(audio_path)
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": sample.task_id,
            "source_relative_path": sample.source_relative_path,
            "source_sha256": sample.source_sha256,
            "audio_sha256": audio_sha256,
            "duration_seconds": duration_seconds,
            "format": {
                "container": "wav",
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
            "generated_at": utc_now_iso(),
        },
    )
    return audio_path, audio_sha256, duration_seconds


def _validate_canonical_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as handle:
        actual = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
    expected = (1, 2, 16000)
    if actual != expected:
        raise ValueError(
            "Canonical audio decoder produced an unexpected WAV format: "
            f"expected channels/sample_width/rate={expected}, got {actual}."
        )


def canonical_audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        return handle.getnframes() / frame_rate if frame_rate else 0.0
