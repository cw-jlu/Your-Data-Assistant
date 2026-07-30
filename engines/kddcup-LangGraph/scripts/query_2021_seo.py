import json
import sqlite3

def query():
    json_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\json\lc_ashareseasonednewissue.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    records = data.get("records", [])
    
    # 过滤出 2021 年的记录
    records_2021 = []
    for r in records:
        date_str = r.get("AdvanceDate")
        if date_str and date_str.startswith("2021"):
            records_2021.append(r)
            
    print(f"Total 2021 records: {len(records_2021)}")
    
    # 我们从数据库读取 SecuCode 到真实中文名称的映射，以防止乱码。
    db_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\db\sub_db.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 我们可以通过多个表如 lc_coconcept, lc_dividend, lc_buyback 建立 SecuCode -> (ChiName, ChiNameAbbr) 的映射
    code_map = {}
    
    tables_to_check = [
        ("lc_coconcept", "SecuCode", "ChiName", "SecuAbbr"), # 在这个表里简称为 SecuAbbr 
        ("lc_dividend", "SecuCode", "ChiName", "ChiNameAbbr"),
        ("lc_buyback", "SecuCode", "ChiName", "ChiNameAbbr"),
        ("lc_ashareipobid", "SecuCode", "ChiName", "ChiNameAbbr"),
        ("lc_ashareplacement", "SecuCode", "ChiName", "ChiNameAbbr")
    ]
    
    for table, code_col, name_col, abbr_col in tables_to_check:
        try:
            cursor.execute(f"SELECT DISTINCT {code_col}, {name_col}, {abbr_col} FROM {table}")
            for row in cursor.fetchall():
                code, name, abbr = row
                if code:
                    # 如果有未乱码的名字，我们保存起来
                    # 我们知道 utf-8 字节是正常的，所以我们在 python 内部得到的已经是正常的中文字符串
                    if code not in code_map:
                        code_map[code] = {"ChiName": name, "ChiNameAbbr": abbr}
                    else:
                        if name and not code_map[code]["ChiName"]:
                            code_map[code]["ChiName"] = name
                        if abbr and not code_map[code]["ChiNameAbbr"]:
                            code_map[code]["ChiNameAbbr"] = abbr
        except Exception as e:
            # 忽略没有该表或该列的错误
            pass
            
    conn.close()
    
    # 输出结果到文件
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\query_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("2021 Seasoned Equity Offering Records:\n")
        f.write(f"{'Code':<10} | {'Name (from JSON)':<25} | {'Name (DB mapping)':<25} | {'Date':<20} | {'Purpose'}\n")
        f.write("-" * 120 + "\n")
        for r in records_2021:
            code = r.get("SecuCode")
            json_name = r.get("ChiNameAbbr")
            db_name = code_map.get(code, {}).get("ChiNameAbbr", "Unknown")
            db_full_name = code_map.get(code, {}).get("ChiName", "Unknown")
            date = r.get("AdvanceDate")
            purpose = r.get("IssuePurpose")
            f.write(f"{code:<10} | {json_name:<25} | {db_name:<25} ({db_full_name}) | {date:<20} | {purpose}\n")
            
    print("Query results written to query_results.txt.")

if __name__ == "__main__":
    query()
