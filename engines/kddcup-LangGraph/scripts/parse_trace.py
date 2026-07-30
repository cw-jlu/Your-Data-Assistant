import json
from pathlib import Path

def parse_trace():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260602T125853Z\task_6\trace.json")
    if not trace_path.exists():
        print(f"File not found: {trace_path}")
        return
        
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    print("Trace Keys:", trace_data.keys() if isinstance(trace_data, dict) else "List of items")
    
    # 如果 trace 是个列表，或者包含 steps 列表
    steps = []
    if isinstance(trace_data, dict):
        steps = trace_data.get("steps", [])
    elif isinstance(trace_data, list):
        steps = trace_data
        
    print(f"Total steps: {len(steps)}")
    
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\parsed_trace.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, step in enumerate(steps):
            out.write(f"\n================ STEP {idx+1} ================\n")
            # 提取思考内容和工具调用
            source = step.get("source", "UNKNOWN")
            step_type = step.get("type", "UNKNOWN")
            out.write(f"Source: {source} | Type: {step_type}\n")
            
            content = step.get("content", "")
            if content:
                out.write(f"--- Thought ---\n{content}\n")
                
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                out.write("--- Tool Calls ---\n")
                for tc in tool_calls:
                    out.write(f"  Tool: {tc.get('name') or tc.get('function', {}).get('name')}\n")
                    # 避免打印过多冗长的 arguments，仅提取简短的 args
                    args = tc.get("args") or tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                    out.write(f"  Args: {json.dumps(args, ensure_ascii=False)[:300]}\n")
                    
            # 也可以记录工具调用的输出
            # 看看 step 结构里是否有结果
            result = step.get("result") or step.get("output")
            if result:
                out.write(f"--- Output ---\n{str(result)[:500]}\n")
                
    print("Parsed trace written to parsed_trace.txt.")

if __name__ == "__main__":
    parse_trace()
