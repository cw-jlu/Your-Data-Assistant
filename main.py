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
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / "configs" / "react_baseline.local.yaml"
    
    # Load base config
    app_config = load_app_config(config_path)

    # Detect if we are in Docker eval environment
    eval_input = Path("/input")
    eval_output = Path("/output")
    eval_logs = Path("/logs")
    
    if eval_input.exists():
        # Competition Spec 3.7: Setup persistent logging
        setup_persistent_logging(eval_logs)
        
        print(f"Detected evaluation environment: /input exists. Overriding paths.")
        dataset_config = DatasetConfig(root_path=eval_input)
        run_config = RunConfig(
            output_dir=eval_output,
            run_id=app_config.run.run_id,
            max_workers=app_config.run.max_workers,
            task_timeout_seconds=app_config.run.task_timeout_seconds
        )
        app_config = AppConfig(
            dataset=dataset_config,
            agent=app_config.agent,
            run=run_config
        )
    else:
        print("Running in local development environment.")
        # Make sure data/public/input exists, otherwise we fall back to user's kdd path
        if not app_config.dataset.root_path.exists():
            alternate_path = Path(__file__).resolve().parents[1] / "public" / "input"
            print(f"Dataset not found at {app_config.dataset.root_path}, trying {alternate_path}")
            if alternate_path.exists():
                dataset_config = DatasetConfig(root_path=alternate_path)
                app_config = AppConfig(
                    dataset=dataset_config,
                    agent=app_config.agent,
                    run=app_config.run
                )
        
    print(f"Using input directory: {app_config.dataset.root_path}")
    print(f"Using output directory: {app_config.run.output_dir}")

    # Allow limiting tasks locally for quick test
    limit_str = os.environ.get("BENCHMARK_LIMIT")
    limit = int(limit_str) if limit_str else None

    # Run the benchmark
    print("Starting benchmark evaluation loop...")
    run_output_dir, artifacts = run_benchmark(config=app_config, limit=limit)
    
    print(f"Benchmark finished. Run output: {run_output_dir}")
    print(f"Tasks attempted: {len(artifacts)}")
    print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


if __name__ == "__main__":
    main()
