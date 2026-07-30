import json
from pathlib import Path

def inspect_trace_details():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260602T125853Z\task_6\trace.json")
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    steps = trace_data if isinstance(trace_data, list) else trace_data.get("steps", [])
    
    # 打印每一步的详细思考、工具调用及工具返回，写入文件
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\trace_details.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, step in enumerate(steps):
            out.write(f"\n================ STEP {idx+1} ================\n")
            out.write(f"Source: {step.get('source')} | Type: {step.get('type')}\n")
            
            # 思考内容
            content = step.get("content", "")
            if content:
                out.write(f"--- Thought content ---\n{content}\n")
                
            # 工具调用
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                out.write("--- Tool Calls ---\n")
                for tc in tool_calls:
                    out.write(f"  Tool: {tc.get('name') or tc.get('function', {}).get('name')}\n")
                    args = tc.get("args") or tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                    out.write(f"  Args: {json.dumps(args, ensure_ascii=False)}\n")
            
            # 工具输出结果
            # 在某些 trace 结构中，工具输出保存在下一条 SYSTEM 消息中，或者在同一个 step 的 'observation' 或 'output' / 'result' 里
            # 我们检查这个 step 里面所有的 keys
            for k in ["observation", "output", "result", "response"]:
                if k in step and step[k]:
                    out.write(f"--- Step {k} ---\n{str(step[k])[:1000]}\n")
                    
    print("Done writing trace details to trace_details.txt.")

if __name__ == "__main__":
    inspect_trace_details()
