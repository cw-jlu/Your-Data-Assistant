from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_baseline.tools.python_exec import TaskContextWorkspace


class _FlakyTemporaryDirectory:
    name = "locked-workspace"

    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def cleanup(self) -> None:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise PermissionError(32, "file is in use")


def test_workspace_cleanup_retries_transient_file_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = TaskContextWorkspace(source_root=Path("."))
    temporary_dir = _FlakyTemporaryDirectory(failures_before_success=2)
    workspace._temporary_dir = temporary_dir  # type: ignore[assignment]
    workspace._workspace_root = Path("locked-workspace/context")
    monkeypatch.setattr("data_agent_baseline.tools.python_exec.time.sleep", lambda _: None)

    workspace.cleanup()

    assert temporary_dir.calls == 3
    assert workspace.path is None


def test_workspace_cleanup_warns_without_failing_on_persistent_file_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = TaskContextWorkspace(source_root=Path("."))
    temporary_dir = _FlakyTemporaryDirectory(failures_before_success=3)
    workspace._temporary_dir = temporary_dir  # type: ignore[assignment]
    monkeypatch.setattr("data_agent_baseline.tools.python_exec.time.sleep", lambda _: None)

    with pytest.warns(ResourceWarning, match="Could not remove temporary task workspace"):
        workspace.cleanup()

    assert temporary_dir.calls == 3
    assert workspace.path is None
