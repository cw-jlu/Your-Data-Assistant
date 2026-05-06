import sys
import os
import json
from pathlib import Path

# Add src to the path
sys.path.insert(0, str(Path(r"D:\code\python\kdd\starter-kit\src")))

from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.runner import run_benchmark
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
import multiprocessing

def run_subset():
    config = load_app_config(Path(r"D:\code\python\kdd\starter-kit\configs\react_baseline.local.yaml"))
    
    tasks_to_run = [
        "task_25", "task_80", "task_89", "task_163", "task_169", 
        "task_173", "task_180", "task_196", "task_199", "task_200", 
        "task_243", "task_249", "task_257", "task_259", "task_283", 
        "task_344", "task_379", "task_396", "task_408", "task_415", 
        "task_418"
    ]
    
    print(f"Running subset: {tasks_to_run}")
    
    output_root = Path(r"D:\code\python\kdd\starter-kit\artifacts\runs")
    from data_agent_baseline.run.runner import create_run_output_dir
    effective_run_id, run_output_dir = create_run_output_dir(output_root, run_id=None)
    print(f"Output dir: {run_output_dir}")

    # Use sequential execution to easily view logs if needed
    for t in tasks_to_run:
        print(f"\n--- Starting task_{t} ---")
        try:
            from data_agent_baseline.run.runner import run_single_task
            result = run_single_task(task_id=t, config=config, run_output_dir=run_output_dir)
            print(f"Task {t} finished. Succeeded: {result.succeeded}")
        except Exception as e:
            print(f"Task {t} failed with exception: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_subset()