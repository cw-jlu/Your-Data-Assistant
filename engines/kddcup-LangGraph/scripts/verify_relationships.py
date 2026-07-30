# -*- coding: utf-8 -*-
"""
该脚本用于自动化验证 relationships.json 的有效性。
它检查 data/join/task_1 到 data/join/task_15 下的 relationships.json 是否存在，
验证其 JSON 结构是否满足要求，并与 SQLite 数据库中的表和字段进行对比，确认其实际存在。
所有注释和文档均使用中文编写。
"""
import os
import json
import sqlite3
import sys

def verify_relationships():
    # 获取项目根目录 (即 scripts 目录的上一级目录)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # 记录是否有任何错误
    has_errors = False
    
    # 遍历任务 1 到 15
    for i in range(1, 16):
        task_id = f"task_{i}"
        
        # 1. 检查 relationships.json 是否存在
        json_path = os.path.join(base_dir, "data", "join", task_id, "relationships.json")
        if not os.path.exists(json_path):
            print(f"错误: 任务 {task_id} 的关系定义文件不存在: {json_path}", file=sys.stderr)
            has_errors = True
            continue
            
        # 2. 读取并解析 JSON 文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"错误: 任务 {task_id} 的 relationships.json 不是有效的 JSON 格式: {e}", file=sys.stderr)
            has_errors = True
            continue
            
        # 3. 验证外层 Schema
        # 必须包含 task_id (且为字符串)
        if "task_id" not in data:
            print(f"错误: {json_path} 缺少 'task_id' 字段。", file=sys.stderr)
            has_errors = True
            continue
        if not isinstance(data["task_id"], str):
            print(f"错误: {json_path} 中的 'task_id' 字段必须为字符串类型。", file=sys.stderr)
            has_errors = True
            continue
        if data["task_id"] != task_id:
            print(f"警告: {json_path} 中的 'task_id' ({data['task_id']}) 与文件夹名称 ({task_id}) 不匹配。", file=sys.stderr)
            
        # 必须包含 relationships 列表
        if "relationships" not in data:
            print(f"错误: {json_path} 缺少 'relationships' 列表。", file=sys.stderr)
            has_errors = True
            continue
        if not isinstance(data["relationships"], list):
            print(f"错误: {json_path} 中的 'relationships' 必须是列表(list)类型。", file=sys.stderr)
            has_errors = True
            continue
            
        # 4. 验证 relationships 列表中的每个对象
        task_db_path = os.path.join(base_dir, "data", "input", task_id, "context", "db", "sub_db.sqlite")
        if not os.path.exists(task_db_path):
            print(f"错误: 找不到任务 {task_id} 的数据库文件: {task_db_path}", file=sys.stderr)
            has_errors = True
            continue
            
        try:
            # 连接数据库获取真实的表和字段结构进行比对
            conn = sqlite3.connect(task_db_path)
            cursor = conn.cursor()
            
            # 获取数据库中所有的表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            db_tables = {row[0] for row in cursor.fetchall()}
            
            # 缓存表字段结构以减少查询次数
            table_columns_cache = {}
            
            for idx, rel in enumerate(data["relationships"]):
                if not isinstance(rel, dict):
                    print(f"错误: {json_path} 中 relationships 列表的第 {idx} 个元素不是对象类型。", file=sys.stderr)
                    has_errors = True
                    continue
                    
                # 检查必需字段的存在性与非空字符串要求
                required_fields = ["foreign_table", "foreign_key", "primary_table", "primary_key"]
                field_missing_or_empty = False
                for field in required_fields:
                    if field not in rel:
                        print(f"错误: {json_path} 的 relationships 列表中第 {idx} 个对象缺少字段 '{field}'。", file=sys.stderr)
                        field_missing_or_empty = True
                    elif not isinstance(rel[field], str) or rel[field].strip() == "":
                        print(f"错误: {json_path} 的 relationships 列表中第 {idx} 个对象的字段 '{field}' 必须为非空字符串。", file=sys.stderr)
                        field_missing_or_empty = True
                        
                if field_missing_or_empty:
                    has_errors = True
                    continue
                    
                # 提取字段值
                f_table = rel["foreign_table"]
                f_key = rel["foreign_key"]
                p_table = rel["primary_table"]
                p_key = rel["primary_key"]
                
                # 验证 foreign_table 是否存在
                if f_table not in db_tables:
                    print(f"错误: {json_path} 中的外键表 '{f_table}' 不在数据库 {task_id} 中。", file=sys.stderr)
                    has_errors = True
                else:
                    # 获取并缓存字段列表
                    if f_table not in table_columns_cache:
                        cursor.execute(f"PRAGMA table_info({f_table});")
                        table_columns_cache[f_table] = {row[1] for row in cursor.fetchall()}
                    # 验证 foreign_key 是否存在
                    if f_key not in table_columns_cache[f_table]:
                        print(f"错误: {json_path} 中的外键列 '{f_table}.{f_key}' 在数据库中不存在。", file=sys.stderr)
                        has_errors = True
                        
                # 验证 primary_table 是否存在
                if p_table not in db_tables:
                    print(f"错误: {json_path} 中的主键表 '{p_table}' 不在数据库 {task_id} 中。", file=sys.stderr)
                    has_errors = True
                else:
                    # 获取并缓存字段列表
                    if p_table not in table_columns_cache:
                        cursor.execute(f"PRAGMA table_info({p_table});")
                        table_columns_cache[p_table] = {row[1] for row in cursor.fetchall()}
                    # 验证 primary_key 是否存在
                    if p_key not in table_columns_cache[p_table]:
                        print(f"错误: {json_path} 中的主键列 '{p_table}.{p_key}' 在数据库中不存在。", file=sys.stderr)
                        has_errors = True
            
            conn.close()
        except sqlite3.Error as e:
            print(f"错误: 连接或查询数据库 {task_db_path} 时发生 SQLite 错误: {e}", file=sys.stderr)
            has_errors = True
            
    if has_errors:
        print("\n验证失败：检测到上述关系配置错误或缺失文件。", file=sys.stderr)
        return False
    else:
        print("Verification successful.")
        return True

if __name__ == "__main__":
    success = verify_relationships()
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
