import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.workspaces import WorkspaceManager


class WorkspaceManagerTests(unittest.TestCase):
    def test_capabilities_report_per_engine_support(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with patch("app.workspaces.WORKSPACES_ROOT", Path(temp) / "workspaces"):
                capabilities = WorkspaceManager().capabilities()

        support = capabilities["engine_support"]
        self.assertIn(".png", support["langgraph"]["extensions"])
        self.assertNotIn(".png", support["mamba"]["extensions"])
        self.assertIn(".xlsx", support["memory"]["extensions"])
        self.assertNotIn(".mp3", capabilities["accept"])

    def test_upload_and_finalize_builds_standard_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace_root = Path(temp) / "workspaces"
            with patch("app.workspaces.WORKSPACES_ROOT", workspace_root):
                manager = WorkspaceManager()
                workspace = manager.create()
                workspace_id = str(workspace["id"])
                uploaded = manager.add_file(
                    workspace_id,
                    "sample.csv",
                    io.BytesIO(b"name,value\nA,1\n"),
                    len(b"name,value\nA,1\n"),
                )
                dataset_root = manager.finalize(workspace_id, "What is the value?")

                task = json.loads(
                    (dataset_root / "task_1" / "task.json").read_text(encoding="utf-8")
                )
                context_file = dataset_root / "task_1" / "context" / "sample.csv"
                context_exists = context_file.is_file()

        self.assertEqual("sample.csv", uploaded["name"])
        self.assertEqual("What is the value?", task["question"])
        self.assertTrue(context_exists)

    def test_rejects_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with patch("app.workspaces.WORKSPACES_ROOT", Path(temp) / "workspaces"):
                manager = WorkspaceManager()
                workspace_id = str(manager.create()["id"])
                with self.assertRaisesRegex(ValueError, "暂不支持"):
                    manager.add_file(workspace_id, "script.exe", io.BytesIO(b"x"), 1)


if __name__ == "__main__":
    unittest.main()
