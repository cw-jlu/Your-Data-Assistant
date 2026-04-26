import os
import sys
from pathlib import Path

# Add src to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.run.runner import run_benchmark
from data_agent_baseline.config import load_app_config, DatasetConfig, RunConfig, AppConfig

def main():
    root_dir = Path(__file__).resolve().parent
    config_path = root_dir / "configs" / "react_baseline.example.yaml"
    
    # Load base config
    app_config = load_app_config(config_path)

    # Detect if we are in Docker eval environment
    eval_input = Path("/input")
    eval_output = Path("/output")
    
    if eval_input.exists():
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
