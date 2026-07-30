# -*- coding: utf-8 -*-
import os
import sqlite3
import pandas as pd
import json

def get_db_schema(db_path):
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [r[1] for r in cursor.fetchall()]
        schema[table] = cols
    conn.close()
    return schema

def get_csv_schema(csv_dir):
    if not os.path.exists(csv_dir):
        return {}
    schema = {}
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            name = f[:-4]
            path = os.path.join(csv_dir, f)
            try:
                # read only the header
                df = pd.read_csv(path, nrows=2)
                schema[name] = list(df.columns)
            except Exception as e:
                schema[name] = f"Error: {str(e)}"
    return schema

def get_json_schema(json_dir):
    if not os.path.exists(json_dir):
        return {}
    schema = {}
    for f in os.listdir(json_dir):
        if f.endswith('.json'):
            name = f[:-5]
            path = os.path.join(json_dir, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        schema[name] = list(data[0].keys())
                    else:
                        schema[name] = ["list of unknown or non-dict elements"]
                elif isinstance(data, dict):
                    # check if keys map to dicts
                    first_val = next(iter(data.values()))
                    if isinstance(first_val, dict):
                        schema[name] = ["Key (Dict Key)"] + list(first_val.keys())
                    else:
                        schema[name] = list(data.keys())
                else:
                    schema[name] = [f"Unsupported JSON type: {type(data)}"]
            except Exception as e:
                schema[name] = f"Error: {str(e)}"
    return schema

def main():
    base_dir = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit"
    for i in range(1, 6):
        task_name = f"task_{i}"
        print(f"\n==================== {task_name} ====================")
        task_dir = os.path.join(base_dir, "data", "input", task_name, "context")
        
        db_path = os.path.join(task_dir, "db", "sub_db.sqlite")
        db_schema = get_db_schema(db_path)
        print("SQLite tables & columns:")
        for t, cols in db_schema.items():
            print(f"  {t}: {cols}")
            
        csv_dir = os.path.join(task_dir, "csv")
        csv_schema = get_csv_schema(csv_dir)
        print("CSV virtual tables & columns:")
        for t, cols in csv_schema.items():
            print(f"  {t}: {cols}")
            
        json_dir = os.path.join(task_dir, "json")
        json_schema = get_json_schema(json_dir)
        print("JSON virtual tables & columns:")
        for t, cols in json_schema.items():
            print(f"  {t}: {cols}")

if __name__ == "__main__":
    main()
