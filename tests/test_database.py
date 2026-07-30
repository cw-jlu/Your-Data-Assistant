import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.database import AppDatabase
from app.manager import RunManager, RunRecord


class DatabasePersistenceTests(unittest.TestCase):
    def test_run_and_logs_survive_manager_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = AppDatabase(Path(temp) / "app.db")
            manager = RunManager(database)
            record = RunRecord(
                id="run-persisted",
                engine_id="mamba",
                engine_name="Mamba Agent",
                mode="batch",
                task_id="",
                status="running",
                output_dir=str(Path(temp) / "outputs"),
            )
            manager._runs[record.id] = record  # noqa: SLF001 - persistence unit test
            manager._persist(record)  # noqa: SLF001 - persistence unit test
            manager._append_log(record, "visible trace line")  # noqa: SLF001

            restored = RunManager(database).get(record.id, include_log=True)

        self.assertEqual("interrupted", restored["status"])
        self.assertEqual(["visible trace line"], restored["log"])

    def test_schema_contains_application_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = AppDatabase(Path(temp) / "app.db")
            with closing(database._connect()) as connection:  # noqa: SLF001
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

        self.assertTrue(
            {"runs", "run_logs", "workspaces", "workspace_files", "settings"}.issubset(tables)
        )


if __name__ == "__main__":
    unittest.main()
