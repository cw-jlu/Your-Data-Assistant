import json
from pathlib import Path

def parse_new_trace():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260603T114710Z\task_6\trace.json")
    if not trace_path.exists():
        print(f"File not found: {trace_path}")
        return
        
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    steps = trace_data if isinstance(trace_data, list) else trace_data.get("steps", [])
    print("New Trace Steps Count:", len(steps))
    
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\new_parsed_trace.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, step in enumerate(steps):
            out.write(f"\n================ STEP {idx+1} ================\n")
            out.write(f"Node: {step.get('node')} | Ok: {step.get('ok')}\n")
            
            # 思想（还原由于编码问题在python输出时容易乱码的问题，我们直接把其 repr 或者是字符本身写到文件）
            msg = step.get("assistant_message", "")
            if msg:
                out.write(f"--- Assistant Message ---\n{msg}\n")
                
            # 工具调用
            tcs = step.get("tool_calls", [])
            if tcs:
                out.write("--- Tool Calls ---\n")
                for tc in tcs:
                    out.write(f"  Tool: {tc.get('name') or tc.get('function', {}).get('name')}\n")
                    args = tc.get("args") or tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                    out.write(f"  Args: {json.dumps(args, ensure_ascii=False)}\n")
                    
            # 工具输出
            trs = step.get("tool_results", [])
            if trs:
                out.write("--- Tool Results ---\n")
                for tr in trs:
                    tr_str = str(tr)
                    if len(tr_str) > 2000:
                        tr_str = tr_str[:2000] + " ... [TRUNCATED]"
                    out.write(f"  {tr_str}\n")
                    
    print("New parsed trace written successfully.")

if __name__ == "__main__":
    parse_new_trace()
