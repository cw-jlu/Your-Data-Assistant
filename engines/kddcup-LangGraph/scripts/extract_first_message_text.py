import json
from pathlib import Path

def extract_first_message_text():
    trace_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\artifacts\runs\20260603T114710Z\task_6\trace.json")
    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
        
    steps = trace_data if isinstance(trace_data, list) else trace_data.get("steps", [])
    
    first_model_step = None
    for step in steps:
        if step.get("node") == "model" and step.get("model_request"):
            first_model_step = step
            break
            
    if not first_model_step:
        print("First model step not found.")
        return
        
    model_req = first_model_step.get("model_request")
    last_msg = model_req.get("last_message", {})
    
    # 在有些系统中，last_message 可能是以 dict 存在，并且 'content' 字段是一个 JSON 字符串，代表 parts 列表
    # 我们看一下 last_msg 里面有没有 content 或者是 content 列表
    # 根据 content_preview 的格式: "[{\"type\": \"text\", \"text\": ..."
    # 物理 trace.json 里很可能有一个 'content' 字段
    # 让我们来看看 trace.json 中，最后这个 step 的物理表示
    # 我们直接写个脚本把 last_msg 的 'content' (如果它存在并且很大) 的前 10000 字符和后 5000 字符写入文件
    content_raw = last_msg.get("content")
    
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\first_message_text.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        if not content_raw:
            # 也许 'content' 没有被保存在 last_message 里，或者是在 'messages' 里
            # 让我们查找 trace_data 结构中有没有大段文本
            out.write("No 'content' found in last_message. Dumping the whole last_message keys:\n")
            out.write(str(last_msg.keys()))
            # 看看里面是不是有别的大内容
            for k, v in last_msg.items():
                if k != 'content':
                    out.write(f"\nKey: {k}, type: {type(v)}, value_preview: {str(v)[:200]}\n")
            return
            
        # 如果 content_raw 是个 string (比如是一个 JSON string)
        if isinstance(content_raw, str):
            try:
                parts = json.loads(content_raw)
            except Exception as e:
                out.write(f"Failed to parse content as JSON: {e}\n")
                out.write(content_raw[:10000])
                return
        else:
            parts = content_raw
            
        out.write(f"Parsed content parts: {len(parts) if isinstance(parts, list) else 'Not a list'}\n\n")
        if isinstance(parts, list):
            for i, part in enumerate(parts):
                p_type = part.get("type")
                out.write(f"--- Part {i+1} (type: {p_type}) ---\n")
                if p_type == "text":
                    text_content = part.get("text", "")
                    out.write(f"Text Length: {len(text_content)}\n")
                    # 写入前 50000 字符，看看注入了什么数据！
                    out.write(text_content[:50000])
                    out.write("\n... [TRUNCATED MID] ...\n")
                    out.write(text_content[-10000:])
                else:
                    out.write(str(part)[:500] + "\n")
                    
    print("Dumped first message text.")

if __name__ == "__main__":
    extract_first_message_text()
