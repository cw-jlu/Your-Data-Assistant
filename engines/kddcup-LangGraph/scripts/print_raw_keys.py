import json
from pathlib import Path

def print_raw_keys():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260602T125853Z\task_6\trace.json")
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    steps = trace_data if isinstance(trace_data, list) else trace_data.get("steps", [])
    
    print(f"Total steps: {len(steps)}")
    
    # 打印第一步和第二步的全部 JSON（或者只打印 keys 和简短值）
    for idx in range(min(5, len(steps))):
        print(f"\n--- STEP {idx+1} ---")
        step = steps[idx]
        for k, v in step.items():
            val_str = str(v)
            if len(val_str) > 300:
                val_str = val_str[:300] + " ... [TRUNCATED]"
            print(f"  {k}: {val_str}")
            
if __name__ == "__main__":
    print_raw_keys()
