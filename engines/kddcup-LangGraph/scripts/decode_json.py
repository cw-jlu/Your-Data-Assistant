import json

def try_decode():
    json_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\json\lc_ashareseasonednewissue.json"
    
    # 1. 尝试直接以原始 binary 读取，然后用不同编码 decode
    with open(json_path, "rb") as f:
        content_bytes = f.read()
        
    encodings = ["utf-8", "gbk", "gb18030", "utf-16", "latin1"]
    for enc in encodings:
        try:
            text = content_bytes.decode(enc)
            # 看看里面是不是有我们认识的中文汉字，而不是乱码
            # 中文字符范围一般在 \u4e00-\u9fff
            # 我们可以打印前几个含有非 ASCII 字符的片段
            print(f"\n--- Decoded with {enc} ---")
            data = json.loads(text)
            records = data.get("records", [])
            if records:
                sample = records[0]
                print("ChiName:", sample.get("ChiName"))
                print("ChiNameAbbr:", sample.get("ChiNameAbbr"))
                
                # 如果是 Mojibake，可能是用 utf-8 decode 了 gbk 编码的数据
                # 我们也可以尝试用 latin1 编码回 bytes 然后用 gbk 解码
                # 比如：sample.get("ChiName").encode('latin1').decode('gbk')
                # 或者是 encode('utf-8').decode('gbk')
                try:
                    # 尝试修复
                    # 比如: »ñװ²Ĺɷ޹˾ 可能是 mojibake
                    pass
                except Exception as ex:
                    pass
        except Exception as e:
            print(f"Error decoding with {enc}: {e}")

if __name__ == "__main__":
    try_decode()
