import json

def inspect():
    json_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\json\lc_ashareseasonednewissue.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = data.get("records", [])
    print("Number of records:", len(records))
    if records:
        print("First record:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
        
        # 看看有什么样的字段
        print("Keys in records[0]:", list(records[0].keys()))
        
        # 统计有哪些年份，或者是 advance_date，或者其他的批次信息
        years = set()
        for r in records:
            # 查找可能与年份或日期相关的字段
            # 在 records 中字段名可能是小写或者驼峰式，让我们找找
            for k, v in r.items():
                if "date" in k.lower() and v:
                    try:
                        years.add(v[:4])
                    except:
                        pass
        print("Years in records based on date fields:", sorted(list(years)))

if __name__ == "__main__":
    inspect()
