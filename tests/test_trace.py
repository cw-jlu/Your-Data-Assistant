import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.manager import RunManager, RunRecord


class TraceAggregationTests(unittest.TestCase):
    def test_collects_process_json_and_sqlite_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp)
            output = runtime / "outputs" / "run-1"
            output.mkdir(parents=True)
            (output / "trace.json").write_text(
                json.dumps({"steps": [{"tool": "inspect_files"}]}),
                encoding="utf-8",
            )
            trace_dir = runtime / "traces"
            trace_dir.mkdir()
            database = trace_dir / "run-1.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE spans (id TEXT, name TEXT)")
            connection.execute("INSERT INTO spans VALUES ('1', 'agent.run')")
            connection.commit()
            connection.close()

            with patch("app.manager.RUNTIME_ROOT", runtime):
                manager = RunManager()
                manager._runs["run-1"] = RunRecord(  # noqa: SLF001 - focused unit test
                    id="run-1",
                    engine_id="mamba",
                    engine_name="Mamba Agent",
                    mode="batch",
                    task_id="",
                    status="succeeded",
                    output_dir=str(output),
                    log_lines=["started", "finished"],
                )
                trace = manager.trace("run-1")

        self.assertEqual(["started", "finished"], trace["process_log"])
        self.assertEqual({"json", "sqlite"}, {item["kind"] for item in trace["artifacts"]})
        sqlite_artifact = next(item for item in trace["artifacts"] if item["kind"] == "sqlite")
        self.assertEqual("agent.run", sqlite_artifact["content"]["spans"][0]["name"])


if __name__ == "__main__":
    unittest.main()

