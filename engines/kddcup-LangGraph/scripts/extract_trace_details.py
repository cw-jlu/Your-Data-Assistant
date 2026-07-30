import json
from pathlib import Path

def extract_details():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260602T125853Z\task_6\trace.json")
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    steps = trace_data if isinstance(trace_data, list) else trace_data.get("steps", [])
    
    # 提取 Step 3 (read_doc video/briefing_timeline.md) 的结果
    step3 = steps[2] # 索引从0开始，Step 3 是 index 2
    step3_result = step3.get("tool_results", [])
    
    # 提取 Step 7 (execute_probe_query 2020%) 的结果
    step7 = steps[6] # Index 6
    step7_result = step7.get("tool_results", [])
    
    # 提取最后一步 Step 9 的 tool_calls (answer)
    step9 = steps[8]
    
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\trace_investigation.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("=== STEP 3 TOOL RESULT (briefing_timeline.md) ===\n")
        out.write(json.dumps(step3_result, ensure_ascii=False, indent=2) + "\n\n")
        
        out.write("=== STEP 7 TOOL RESULT (execute_probe_query) ===\n")
        out.write(json.dumps(step7_result, ensure_ascii=False, indent=2) + "\n\n")
        
        out.write("=== STEP 9 ANSWER ===\n")
        out.write(json.dumps(step9.get("tool_calls"), ensure_ascii=False, indent=2) + "\n\n")
        
    print("Done extracting trace investigation.")

if __name__ == "__main__":
    extract_details()
