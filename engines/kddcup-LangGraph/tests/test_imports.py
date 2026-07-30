from __future__ import annotations

import subprocess
import sys


def test_data_understanding_agent_can_be_imported_directly() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent; "
            "print(DataUnderstandingAgent.__name__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "DataUnderstandingAgent"
