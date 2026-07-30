"""Shared constants for tool schemas and handlers."""

from __future__ import annotations

from typing import Final

# Keep this tools-local until prompts/ is allowed to consume the same source.
PATH_CONVENTION_NOTE: Final = (
    "Paths are relative to the task context directory; DO NOT prepend 'context/'. "
    "For example use 'knowledge.md' or 'csv/foo.csv', not 'context/knowledge.md'."
)

# Agent arbitrary-code hard timeout. Small values choke legitimate pandas work;
# large values make run-benchmark failures expensive.
EXECUTE_PYTHON_TIMEOUT_SECONDS: Final = 30

# `answer` inline path hard limits.  Values are the max allowed counts;
# exceeding either forces `from_csv`.
INLINE_ANSWER_ROW_LIMIT: Final = 10
INLINE_ANSWER_CELL_LIMIT: Final = 50

ANSWER_ARTIFACT_ENV_VAR: Final = "DABENCH_ANSWER_DIR"
ANSWER_ARTIFACT_FILENAME: Final = "answer.csv"

NON_SQLITE_PATH_WARNING: Final = "do NOT call on `.csv` / `.json` / `.md` / `.txt` paths."

SQLITE_QUERY_TIMEOUT_SECONDS: Final = 30


def inline_answer_size_phrase() -> str:
    """Human-readable size contract for inline answer payloads."""
    return f"≤{INLINE_ANSWER_ROW_LIMIT} rows AND ≤{INLINE_ANSWER_CELL_LIMIT} cells"


def artifact_answer_size_phrase() -> str:
    """Human-readable size contract for CSV artifact answer payloads."""
    return f"answers exceeding {INLINE_ANSWER_ROW_LIMIT} rows OR {INLINE_ANSWER_CELL_LIMIT} cells"


def answer_artifact_expression() -> str:
    """Canonical Python expression for the answer CSV artifact path."""
    return f"os.path.join(os.environ['{ANSWER_ARTIFACT_ENV_VAR}'], '{ANSWER_ARTIFACT_FILENAME}')"
