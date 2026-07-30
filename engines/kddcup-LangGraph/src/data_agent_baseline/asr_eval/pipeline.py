from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from data_agent_baseline.asr_runtime import (
    PROMPT_TEMPLATE_VERSION,
    AsrPipelineRuntime,
    AsrRuntimeSettings,
    sha256_text,
    validate_stage,
)
from data_agent_baseline.asr_eval.azure import (
    AZURE_LOCALES,
    AzureFastTranscriptionClient,
    endpoint_host,
    extract_azure_reference,
    resolve_azure_credentials,
)
from data_agent_baseline.asr_eval.core import (
    SCHEMA_VERSION,
    AsrSample,
    _atomic_write_json,
    classify_asr_run,
    discover_asr_samples,
    ensure_canonical_audio,
    find_asr_run_dir,
    utc_now_iso,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _AzureClient(Protocol):
    endpoint: str
    api_version: str

    def transcribe(self, audio_path: Path) -> dict[str, Any]: ...


def _validated_id(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not _ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen."
        )
    return normalized


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _sample_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["task_id"]): item
        for item in manifest.get("samples") or []
        if isinstance(item, dict) and item.get("task_id")
    }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def generate_azure_gold(
    *,
    project_root: Path,
    dataset_root: Path,
    gold_id: str,
    task_ids: list[str] | tuple[str, ...] | None = None,
    azure_client: _AzureClient | None = None,
    audio_decoder: Callable[[Path, Path], None] | None = None,
    evaluation_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    effective_gold_id = _validated_id(gold_id, field_name="gold_id")
    root = evaluation_root or (project_root / "evaluation" / "asr")
    gold_dir = root / "gold" / effective_gold_id
    manifest_path = gold_dir / "manifest.json"
    samples = discover_asr_samples(dataset_root, task_ids=task_ids)

    if azure_client is None:
        endpoint, key = resolve_azure_credentials(project_root)
        azure_client = AzureFastTranscriptionClient(endpoint=endpoint, key=key)

    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        provider = manifest.get("provider") or {}
        expected_provider = {
            "name": "azure-fast-transcription",
            "api_version": azure_client.api_version,
            "endpoint_host": endpoint_host(azure_client.endpoint),
            "candidate_locales": list(AZURE_LOCALES),
        }
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Existing gold manifest uses an incompatible schema version.")
        if manifest.get("gold_id") != effective_gold_id or provider != expected_provider:
            raise ValueError(
                "Existing gold_id belongs to a different Azure configuration. "
                "Use a new gold_id instead of overwriting it."
            )
    else:
        if gold_dir.exists() and any(gold_dir.iterdir()):
            raise ValueError(
                f"Gold directory is non-empty but has no manifest: {gold_dir}. "
                "Use a new gold_id."
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "gold_id": effective_gold_id,
            "reference_policy": "Azure output is frozen verbatim and is not human-corrected.",
            "provider": {
                "name": "azure-fast-transcription",
                "api_version": azure_client.api_version,
                "endpoint_host": endpoint_host(azure_client.endpoint),
                "candidate_locales": list(AZURE_LOCALES),
            },
            "dataset_root": str(dataset_root.resolve()),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "completed": False,
            "samples": [],
        }

    gold_dir.mkdir(parents=True, exist_ok=True)
    audio_root = root / "audio"
    entries = _sample_lookup(manifest)
    requested_ids = set(manifest.get("requested_task_ids") or ())
    requested_ids.update(sample.task_id for sample in samples)

    for sample in samples:
        audio_path, audio_sha256, duration_seconds = ensure_canonical_audio(
            sample,
            audio_root,
            decoder=audio_decoder,
        )
        existing = entries.get(sample.task_id)
        if existing is not None:
            if (
                existing.get("source_sha256") != sample.source_sha256
                or existing.get("audio_sha256") != audio_sha256
            ):
                raise ValueError(
                    f"Input hash changed for {sample.task_id}. "
                    "Create a new gold_id; frozen gold cannot be overwritten."
                )
            if existing.get("status") == "ok":
                continue

        entry: dict[str, Any] = {
            "task_id": sample.task_id,
            "source_relative_path": sample.source_relative_path,
            "source_sha256": sample.source_sha256,
            "audio_path": _relative_or_absolute(audio_path, project_root),
            "audio_sha256": audio_sha256,
            "duration_seconds": duration_seconds,
            "status": "failed",
            "generated_at": utc_now_iso(),
        }
        try:
            raw_payload = azure_client.transcribe(audio_path)
            reference_text, locale = extract_azure_reference(raw_payload)
            raw_path = gold_dir / "raw" / f"{sample.task_id}.json"
            _atomic_write_json(raw_path, raw_payload)
            entry.update(
                {
                    "status": "ok",
                    "locale": locale,
                    "text": reference_text,
                    "raw_response_path": raw_path.relative_to(gold_dir).as_posix(),
                    "error": None,
                }
            )
        # Provider failures must be persisted per sample so a long gold run can resume.
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entries[sample.task_id] = entry
        manifest["samples"] = sorted(entries.values(), key=lambda item: _task_sort_key(item["task_id"]))
        manifest["updated_at"] = utc_now_iso()
        manifest["completed"] = all(
            entries.get(task_id, {}).get("status") == "ok" for task_id in requested_ids
        )
        _atomic_write_json(manifest_path, manifest)

    manifest["samples"] = sorted(entries.values(), key=lambda item: _task_sort_key(item["task_id"]))
    manifest["updated_at"] = utc_now_iso()
    manifest["completed"] = all(entries[task_id].get("status") == "ok" for task_id in requested_ids)
    manifest["requested_task_ids"] = sorted(requested_ids, key=_task_sort_key)
    _atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest


def _task_sort_key(task_id: str) -> tuple[int, int | str]:
    suffix = str(task_id).removeprefix("task_")
    return (0, int(suffix)) if suffix.isdigit() else (1, str(task_id))


def _default_whisper_factory(**kwargs: Any) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required to run an ASR candidate. "
            "Install the project dependencies first."
        ) from exc
    return WhisperModel(**kwargs)


def run_candidate_asr(
    *,
    project_root: Path,
    dataset_root: Path,
    run_id: str,
    model_name: str,
    device: str,
    compute_type: str,
    stage: str = "medium-baseline",
    detector_model_name: str = "tiny",
    cpu_threads: int = 4,
    num_workers: int = 1,
    max_workers: int = 1,
    language_threshold: float = 0.5,
    prompt_min_terms: int = 8,
    prompt_max_terms: int = 15,
    ui_terms: tuple[str, ...] | list[str] = (),
    prompt_model: Any | None = None,
    prompt_model_name: str | None = None,
    prompt_model_base: str | None = None,
    task_ids: list[str] | tuple[str, ...] | None = None,
    model_factory: Callable[..., Any] | None = None,
    audio_loader: Callable[[Path], Any] | None = None,
    audio_decoder: Callable[[Path, Path], None] | None = None,
    evaluation_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    effective_run_id = _validated_id(run_id, field_name="run_id")
    effective_stage = validate_stage(stage)
    if cpu_threads < 1 or num_workers < 1 or max_workers < 1:
        raise ValueError("ASR cpu_threads, num_workers, and max_workers must be positive.")
    eval_root = evaluation_root or (project_root / "evaluation" / "asr")
    output_root = artifacts_root or (project_root / "artifacts" / "asr")
    samples = discover_asr_samples(dataset_root, task_ids=task_ids)
    existing_run_dir = find_asr_run_dir(output_root, effective_run_id)
    run_category = classify_asr_run(
        run_id=effective_run_id,
        sample_count=len(samples),
        stage=effective_stage,
    )
    run_dir = existing_run_dir or (
        output_root / "runs" / run_category / effective_run_id
    )
    manifest_path = run_dir / "manifest.json"
    candidate_config = {
        "engine": "faster-whisper",
        "model": model_name,
        "stage": effective_stage,
        "detector_model": detector_model_name,
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "num_workers": num_workers,
        "max_workers": max_workers,
        "language_threshold": language_threshold,
        "prompt_min_terms": prompt_min_terms,
        "prompt_max_terms": prompt_max_terms,
        "ui_terms": list(ui_terms),
        "prompt_model": prompt_model_name,
        "prompt_model_base": prompt_model_base.rstrip("/") if prompt_model_base else None,
        "prompt_template": PROMPT_TEMPLATE_VERSION,
        "transcribe_options": {
            "language_routing": effective_stage != "medium-baseline",
            "dynamic_prompt": effective_stage
            in {"dynamic-prompt", "dynamic-plus-ui"},
            "fixed_ui_terms": effective_stage == "dynamic-plus-ui",
            "vad_filter": False,
        },
        "canonical_audio": "pcm_s16le/16000Hz/mono",
    }

    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Existing ASR run uses an incompatible schema version.")
        if manifest.get("run_id") != effective_run_id or manifest.get("candidate") != candidate_config:
            raise ValueError(
                "Existing run_id belongs to a different candidate configuration. "
                "Use a new run_id."
            )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"ASR run directory is non-empty but has no manifest: {run_dir}")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": effective_run_id,
            "candidate": candidate_config,
            "dataset_root": str(dataset_root.resolve()),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "completed": False,
            "samples": [],
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    entries = _sample_lookup(manifest)
    prepared: list[tuple[AsrSample, Path, str, float, str, str]] = []
    input_root = (
        dataset_root / "input"
        if (dataset_root / "input").is_dir()
        else dataset_root
    )
    for sample in samples:
        audio_path, audio_sha256, duration_seconds = ensure_canonical_audio(
            sample,
            eval_root / "audio",
            decoder=audio_decoder,
        )
        task_dir = input_root / sample.task_id
        task_payload_path = task_dir / "task.json"
        task_payload = (
            _load_json(task_payload_path) if task_payload_path.is_file() else {}
        )
        question = str(task_payload.get("question", ""))
        knowledge_path = task_dir / "context" / "knowledge.md"
        knowledge_text = (
            knowledge_path.read_text(encoding="utf-8", errors="replace")
            if knowledge_path.is_file()
            else ""
        )
        question_sha256 = sha256_text(question)
        knowledge_sha256 = sha256_text(knowledge_text)
        existing = entries.get(sample.task_id)
        if existing is not None:
            if (
                existing.get("source_sha256") != sample.source_sha256
                or existing.get("audio_sha256") != audio_sha256
                or existing.get("question_sha256") != question_sha256
                or existing.get("knowledge_sha256") != knowledge_sha256
            ):
                raise ValueError(
                    f"Input hash changed for {sample.task_id}. Use a new run_id."
                )
            if existing.get("status") == "ok":
                continue
        prepared.append(
            (
                sample,
                audio_path,
                audio_sha256,
                duration_seconds,
                question,
                knowledge_text,
            )
        )

    runtime = AsrPipelineRuntime(
        AsrRuntimeSettings(
            model_name=model_name,
            detector_model_name=detector_model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            language_threshold=language_threshold,
            prompt_min_terms=prompt_min_terms,
            prompt_max_terms=prompt_max_terms,
            ui_terms=tuple(ui_terms) or AsrRuntimeSettings().ui_terms,
        ),
        model_factory=model_factory or _default_whisper_factory,
        audio_loader=audio_loader,
    )

    def process_sample(
        item: tuple[AsrSample, Path, str, float, str, str],
    ) -> dict[str, Any]:
        sample, audio_path, audio_sha256, duration_seconds, question, knowledge_text = item
        entry: dict[str, Any] = {
            "task_id": sample.task_id,
            "source_relative_path": sample.source_relative_path,
            "source_sha256": sample.source_sha256,
            "audio_path": _relative_or_absolute(audio_path, project_root),
            "audio_sha256": audio_sha256,
            "question_sha256": sha256_text(question),
            "knowledge_sha256": sha256_text(knowledge_text),
            "duration_seconds": duration_seconds,
            "status": "failed",
            "generated_at": utc_now_iso(),
        }
        try:
            transcript = runtime.transcribe(
                audio_path,
                stage=effective_stage,
                question=question,
                knowledge_text=knowledge_text,
                prompt_model=prompt_model,
                prompt_model_name=prompt_model_name,
            )
            elapsed = float(transcript["timings"]["total_seconds"])
            entry.update(
                {
                    "status": "ok",
                    "text": transcript["text"],
                    "segments": transcript["segments"],
                    "detected_language": transcript["language"],
                    "language_probability": transcript["language_probability"],
                    "language_detection": transcript["language_detection"],
                    "prompt": transcript["prompt"],
                    "timings": transcript["timings"],
                    "elapsed_seconds": elapsed,
                    "real_time_factor": elapsed / duration_seconds if duration_seconds > 0 else None,
                    "error": None,
                }
            )
        # Model/runtime failures are data for this benchmark and count as failed samples.
        except Exception as exc:  # noqa: BLE001
            entry.update(
                {
                    "elapsed_seconds": None,
                    "real_time_factor": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        return entry

    completed_entries: list[dict[str, Any]] = []
    if max_workers == 1:
        completed_entries = [process_sample(item) for item in prepared]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_sample, item): item[0].task_id for item in prepared}
            for future in as_completed(futures):
                completed_entries.append(future.result())

    for entry in completed_entries:
        sample_task_id = str(entry["task_id"])
        entries[sample_task_id] = entry
        manifest["samples"] = sorted(entries.values(), key=lambda item: _task_sort_key(item["task_id"]))
        manifest["updated_at"] = utc_now_iso()
        _atomic_write_json(manifest_path, manifest)

    requested_ids = set(manifest.get("requested_task_ids") or ())
    requested_ids.update(sample.task_id for sample in samples)
    manifest["samples"] = sorted(entries.values(), key=lambda item: _task_sort_key(item["task_id"]))
    manifest["requested_task_ids"] = sorted(requested_ids, key=_task_sort_key)
    manifest["completed"] = all(entries[task_id].get("status") == "ok" for task_id in requested_ids)
    manifest["updated_at"] = utc_now_iso()
    _atomic_write_json(manifest_path, manifest)
    return manifest_path, manifest
