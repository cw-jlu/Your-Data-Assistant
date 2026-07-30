import sqlite3
import json

def check():
    db_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\db\sub_db.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查找是否有不包含乱码的行
    cursor.execute("SELECT ChiName, ChiNameAbbr, SecuCode FROM lc_buyback LIMIT 5")
    rows = cursor.fetchall()
    print("lc_buyback sample (repr):")
    for r in rows:
        print(f"ChiName: {repr(r[0])}, bytes: {r[0].encode('utf-8') if r[0] else None}")
        
    # 我们也可以查找其他的 csv 文件
    csv_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\csv\lc_actualcontroller.csv"
    print("\nReading CSV with different encodings:")
    with open(csv_path, "rb") as f:
        head = f.read(500)
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            print(f"--- CSV with {enc} ---")
            print(head.decode(enc)[:200])
        except Exception as e:
            print(e)
            
    conn.close()

if __name__ == "__main__":
    check()
