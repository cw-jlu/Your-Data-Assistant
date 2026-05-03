import sqlite3
import pandas as pd
conn = sqlite3.connect('D:/code/python/kdd/public/input/task_180/context/db/transactions_1k.db')
df = pd.read_sql_query("SELECT * FROM transactions_1k WHERE ProductID = 5 AND Price > 29.00", conn)
print(f"Total rows: {len(df)}")
print(df.head(20))
conn.close()
