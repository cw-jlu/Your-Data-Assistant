import json
import tempfile
import unittest
from pathlib import Path

from app.engines import ENGINE_BY_ID, list_tasks


class EngineRegistryTests(unittest.TestCase):
    def test_expected_engines_are_registered(self) -> None:
        self.assertEqual({"langgraph", "kobushi", "memory", "mamba"}, set(ENGINE_BY_ID))
        self.assertFalse(ENGINE_BY_ID["kobushi"].supports_single)

    def test_list_tasks_reads_standard_layout_and_sorts_numerically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for task_id in ("task_10", "task_2"):
                task_dir = root / task_id
                (task_dir / "context").mkdir(parents=True)
                (task_dir / "task.json").write_text(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "difficulty": "easy",
                            "question": f"Question for {task_id}",
                        }
                    ),
                    encoding="utf-8",
                )
                (task_dir / "context" / "data.csv").write_text("a\n1\n", encoding="utf-8")

            tasks = list_tasks(str(root))

        self.assertEqual(["task_2", "task_10"], [task["id"] for task in tasks])
        self.assertEqual(1, tasks[0]["file_count"])


if __name__ == "__main__":
    unittest.main()

