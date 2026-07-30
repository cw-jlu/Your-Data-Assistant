"""Reproducible ASR evaluation utilities."""

from data_agent_baseline.asr_eval.azure import (
    AZURE_API_VERSION,
    AzureFastTranscriptionClient,
    AzureFastTranscriptionError,
    extract_azure_reference,
    resolve_azure_credentials,
)
from data_agent_baseline.asr_eval.core import (
    SCHEMA_VERSION,
    AsrSample,
    canonical_audio_duration,
    discover_asr_samples,
    ensure_canonical_audio,
    sha256_file,
)
from data_agent_baseline.asr_eval.pipeline import (
    generate_azure_gold,
    run_candidate_asr,
)
from data_agent_baseline.asr_eval.scoring import (
    EditCounts,
    edit_counts,
    normalize_chinese,
    normalize_english,
    score_asr_run,
)

__all__ = [
    "AZURE_API_VERSION",
    "SCHEMA_VERSION",
    "AsrSample",
    "AzureFastTranscriptionClient",
    "AzureFastTranscriptionError",
    "EditCounts",
    "canonical_audio_duration",
    "discover_asr_samples",
    "edit_counts",
    "ensure_canonical_audio",
    "extract_azure_reference",
    "generate_azure_gold",
    "normalize_chinese",
    "normalize_english",
    "resolve_azure_credentials",
    "run_candidate_asr",
    "score_asr_run",
    "sha256_file",
]
