import sqlite3
import pandas as pd

def explore():
    db_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\db\sub_db.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print("Tables in database:", tables)
    
    for table in tables:
        print(f"\n--- Table: {table} ---")
        # 打印列信息
        cursor.execute(f"PRAGMA table_info({table});")
        info = cursor.fetchall()
        cols = [col[1] for col in info]
        print("Columns:", cols)
        
        # 打印前 5 行
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            print("First 5 rows:")
            print(df)
        except Exception as e:
            print("Error reading table:", e)
            
    conn.close()

if __name__ == "__main__":
    explore()
