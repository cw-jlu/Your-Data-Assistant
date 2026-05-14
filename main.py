import os
import sys
import traceback
from pathlib import Path

# Add src to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.run.runner import run_benchmark
from data_agent_baseline.config import load_app_config, DatasetConfig, RunConfig, AppConfig

class DualLogger:
    """A logger that writes to both terminal and a file."""
    def __init__(self, filepath, stream):
        self.terminal = stream
        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        # Flush to ensure logs are written immediately, crucial for debugging crashes
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_persistent_logging(log_dir: Path):
    """Redirect stdout and stderr to /logs/runtime.log as per spec 3.7"""
    if log_dir.exists():
        log_file = log_dir / "runtime.log"
        print(f"Setting up persistent logging to {log_file}")
        sys.stdout = DualLogger(log_file, sys.stdout)
        sys.stderr = DualLogger(log_file, sys.stderr)

        # Hook unhandled exceptions to ensure they are logged
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            print("Uncaught exception:", file=sys.stderr)
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

        sys.excepthook = handle_exception

def main():
    # Strict compliance with Official Technical Specifications
    EVAL_INPUT = Path("/input")
    EVAL_OUTPUT = Path("/output")
    EVAL_LOGS = Path("/logs")

    # 1. Setup mandatory persistent logging to /logs/runtime.log (Spec 3.7)
    setup_persistent_logging(EVAL_LOGS)
    
    print("=== Official Evaluation Startup ===")
    print(f"Input path: {EVAL_INPUT}")
    print(f"Output path: {EVAL_OUTPUT}")

    # 2. Load application configuration
    # Note: Sensitive configs (API Key, URL, Model Name) are read from env vars in config.py (Spec 3.5 & 5.2)
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / "configs" / "react_baseline.local.yaml"
    app_config = load_app_config(config_path)

    # 3. Override configs with official evaluation paths
    dataset_config = DatasetConfig(root_path=EVAL_INPUT)
    run_config = RunConfig(
        output_dir=EVAL_OUTPUT,
        run_id="evaluation",
        max_workers=app_config.run.max_workers,
        task_timeout_seconds=app_config.run.task_timeout_seconds
    )
    app_config = AppConfig(
        dataset=dataset_config,
        agent=app_config.agent,
        run=run_config
    )

    # 4. Run the benchmark (Spec 3.4 & 3.6)
    # use_flat_output=True ensures /output/task_id/prediction.csv structure (Spec 2.1 & 2.4)
    print("Starting benchmark evaluation loop...")
    run_output_dir, artifacts = run_benchmark(
        config=app_config, 
        use_flat_output=True
    )
    
    print(f"Benchmark finished.")
    print(f"Tasks attempted: {len(artifacts)}")
    print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")



if __name__ == "__main__":
    main()
