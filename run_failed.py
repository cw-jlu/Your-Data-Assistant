import sys
import io

# Fix Windows encoding issues
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path

# Add src to the path
sys.path.insert(0, str(Path(r"d:\code\python\kdd\starter-kit\src")))

from data_agent_baseline.config import load_app_config, DatasetConfig, RunConfig, AppConfig
from data_agent_baseline.run.runner import run_single_task, create_run_output_dir

def run_specific_tasks():
    root_dir = Path(r"d:\code\python\kdd\starter-kit")
    config_path = root_dir / "configs" / "react_baseline.local.yaml"
    
    app_config = load_app_config(config_path)
    alternate_path = root_dir.parent / "public" / "input"
    if alternate_path.exists():
        app_config = AppConfig(
            dataset=DatasetConfig(root_path=alternate_path),
            agent=app_config.agent,
            run=app_config.run
        )
    
    # Override timeout to 120 seconds for quick debugging
    app_config = AppConfig(
        dataset=app_config.dataset,
        agent=app_config.agent,
        run=RunConfig(
            output_dir=app_config.run.output_dir,
            run_id=app_config.run.run_id,
            max_workers=app_config.run.max_workers,
            task_timeout_seconds=120
        )
    )
    
    effective_run_id, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id="debug_failed_tasks")
    
    tasks_to_run = ["task_11", "task_19", "task_24", "task_25", "task_26", "task_27", "task_38", "task_214"]
    
    print(f"Running {len(tasks_to_run)} tasks with a 120s timeout...")
    print(f"Output dir: {run_output_dir}")
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_task, task_id=t, config=app_config, run_output_dir=run_output_dir): t for t in tasks_to_run}
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                print(f"Task {t} finished. Succeeded: {result.succeeded}")
            except Exception as e:
                print(f"Task {t} failed with exception: {e}")

if __name__ == "__main__":
    run_specific_tasks()
