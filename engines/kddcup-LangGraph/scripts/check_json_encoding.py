import json

def check_json():
    json_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\json\lc_ashareseasonednewissue.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = data.get("records", [])
    if records:
        r = records[0]
        name = r.get("ChiName")
        abbr = r.get("ChiNameAbbr")
        print("ChiName repr:", repr(name))
        print("ChiName UTF-8 bytes:", name.encode('utf-8') if name else None)
        print("ChiName decode/encode test:")
        try:
            # 试一下如果是 Mojibake 修复
            # 比如 name 是以 utf-8 解码了 gbk 字节，那么还原：
            # 先用 latin1 或 cp1252 编码回 bytes，再用 gbk 或 utf-8 解密
            # 或者是用 utf-8 编码，再用 gbk 解密？
            b = name.encode('utf-8')
            # 看看能不能用 gbk 或 gb18030 解码它？
            print("Decoded as gbk:", b.decode('gbk', errors='ignore'))
            print("Decoded as gb18030:", b.decode('gb18030', errors='ignore'))
        except Exception as e:
            print("Error in decoding test:", e)
            
if __name__ == "__main__":
    check_json();
