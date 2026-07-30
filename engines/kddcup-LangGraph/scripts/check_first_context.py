import json
from pathlib import Path

def check_first_context():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260603T114710Z\task_6\trace.json")
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    steps = trace_data if isinstance(trace_data, list) else trace_data.get("steps", [])
    
    # 寻找第一次大模型请求 (通常是 step_index=2 或者 node='model' 的第一次)
    first_model_step = None
    for step in steps:
        if step.get("node") == "model" and step.get("model_request"):
            first_model_step = step
            break
            
    if not first_model_step:
        print("First model step not found.")
        return
        
    model_req = first_model_step.get("model_request")
    print("Model Request keys:", model_req.keys())
    
    # 将完整的 last_message 或者是 messages 列表写入文件，方便查看
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\first_context.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("=== FIRST MODEL STEP REQUEST ===\n")
        out.write(f"Message Count: {model_req.get('message_count')}\n\n")
        
        # 很多时候 last_message 或者 messages 里保存了传递给模型的原始 payload
        # 让我们把整个 model_request 用 pretty print 写入文件
        out.write(json.dumps(model_req, ensure_ascii=False, indent=2))
        
    print("First context dumped successfully.")

if __name__ == "__main__":
    check_first_context()
