# -*- coding: utf-8 -*-
"""
该脚本遍历 task_1 到 task_15 的数据库文件，并打印每个任务中包含的表名、字段名以及物理外键。
同时，结合 context/knowledge.md 提取逻辑外键，提供初步的外键关系推导原型。
所有的注释和文档均使用中文编写。
"""
import os
import re
import sqlite3

def parse_knowledge(knowledge_path, db_tables):
    """
    解析 knowledge.md，提取表和字段的描述信息。
    """
    if not os.path.exists(knowledge_path):
        return {}, {}
    
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    table_desc = {}
    column_desc = {}
    
    # 匹配三级标题，可能包含表名，形如：
    # ### 2.1 Main Operating Income — lc_mainoperincome
    # ### 2.1 Monetary Authority Balance Sheet (ed_moneyauthoritybs)
    # ### 2.1 Monetary Authority Balance Sheet (`ed_moneyauthoritybs`)
    # 我们用正则提取可能的表名
    headers = re.findall(r'^###\s+(.*?)$', content, re.MULTILINE)
    
    # 找出每个表所在的块，用来提取其中的表格/字段定义
    # 我们可以通过定位 ### 标题的起始位置来切分
    sections = re.split(r'^###\s+', content, flags=re.MULTILINE)
    
    for section in sections:
        # 第一行一般是标题
        lines = section.strip().split('\n')
        if not lines:
            continue
        header = lines[0]
        
        # 尝试匹配 db_tables 中的表名
        matched_table = None
        for table in db_tables:
            if table in header or f"`{table}`" in header:
                matched_table = table
                break
        
        if not matched_table:
            # 如果标题里没有，可能是部分匹配或者没有，用正则搜索
            # 搜索是否有 `table_name` 或 (table_name) 或 — table_name
            for table in db_tables:
                if re.search(r'\b' + re.escape(table) + r'\b', header):
                    matched_table = table
                    break
        
        if matched_table:
            table_desc[matched_table] = header
            # 解析这个 section 中的列定义表格
            # 表格一般是 | Column | Semantic Definition | ... | 形式
            col_definitions = {}
            for line in lines[1:]:
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    # 过滤掉表头和分隔线
                    if len(parts) >= 3:
                        col_name = parts[1].replace('`', '').strip()
                        semantic = parts[2].strip()
                        if col_name and col_name.lower() != 'column' and not col_name.startswith('---'):
                            # 过滤掉非字母数字下划线的噪音
                            if re.match(r'^[a-zA-Z0-9_]+$', col_name):
                                col_definitions[col_name] = semantic
            if col_definitions:
                column_desc[matched_table] = col_definitions
                
    return table_desc, column_desc

def analyze_tasks(start_task=1, end_task=15):
    base_dir = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input"
    output_lines = []
    
    for i in range(start_task, end_task + 1):
        task_name = f"task_{i}"
        task_dir = os.path.join(base_dir, task_name)
        db_path = os.path.join(task_dir, "context", "db", "sub_db.sqlite")
        knowledge_path = os.path.join(task_dir, "context", "knowledge.md")
        
        output_lines.append(f"\n=========================================")
        output_lines.append(f" 任务: {task_name}")
        output_lines.append(f"=========================================")
        
        if not os.path.exists(db_path):
            output_lines.append(f"警告: 数据库文件不存在 {db_path}")
            continue
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 1. 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cursor.fetchall() if t[0] != 'sqlite_sequence']
            output_lines.append(f"数据库中的表: {', '.join(tables)}")
            
            # 2. 解析 knowledge.md
            table_desc, column_desc = parse_knowledge(knowledge_path, tables)
            
            table_columns = {} # 存储每个表的列，便于后续推导逻辑外键
            table_pks = {}     # 存储主键
            
            for table in tables:
                output_lines.append(f"\n  [表名]: {table}")
                if table in table_desc:
                    output_lines.append(f"  [说明 (knowledge.md)]: {table_desc[table]}")
                
                # 获取列信息
                cursor.execute(f"PRAGMA table_info({table});")
                cols_info = cursor.fetchall()
                # cols_info 格式: (cid, name, type, notnull, dflt_value, pk)
                
                table_columns[table] = [c[1] for c in cols_info]
                table_pks[table] = [c[1] for c in cols_info if c[5] > 0]
                
                output_lines.append("  [字段列表]:")
                col_definitions = column_desc.get(table, {})
                for col in cols_info:
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = " (主键)" if col[5] > 0 else ""
                    semantic = col_definitions.get(col_name, "无")
                    output_lines.append(f"    - {col_name} ({col_type}){is_pk} : 语义: {semantic}")
                
                # 物理外键
                cursor.execute(f"PRAGMA foreign_key_list({table});")
                fks = cursor.fetchall()
                # fks 格式: (id, seq, table, from, to, on_update, on_delete, match)
                if fks:
                    output_lines.append("  [物理外键 (PRAGMA foreign_key_list)]:")
                    for fk in fks:
                        output_lines.append(f"    - {fk[3]} -> {fk[2]}({fk[4]})")
                else:
                    output_lines.append("  [物理外键]: 无")
            
            # 3. 逻辑外键推导原型
            # 我们根据字段名称的重合来推导逻辑外键
            # 规则：若两个不同的表拥有相同的字段名，且该字段不是通用的日期或简单标识(如 enddate, changedate, id 等，但如果具有明确业务含义如 companycode, secucode 则极为可能是逻辑外键)
            # 我们列出所有同名字段（排除 enddate, changedate, id，除非它们在其中一个表里是主键）
            output_lines.append("\n  [逻辑外键推导原型]:")
            found_logical_fks = False
            
            # 记录已经被检测过的表对，防止重复输出 A-B 和 B-A
            checked_pairs = set()
            
            for t1 in tables:
                for t2 in tables:
                    if t1 == t2:
                        continue
                    pair_key = tuple(sorted([t1, t2]))
                    if pair_key in checked_pairs:
                        continue
                    
                    # 找出同名字段
                    common_cols = set(table_columns[t1]).intersection(set(table_columns[t2]))
                    for col in common_cols:
                        # 过滤掉过于通用的字段，如只表示时间的字段，除非它是关联键
                        # 比如有些表可能通过 (companycode, enddate) 联合关联，保留 enddate 作为辅助也是可以的，
                        # 但核心关联键是 companycode
                        is_weak = col.lower() in ('id', 'enddate', 'changedate', 'date', 'time', 'year', 'month')
                        
                        # 如果是强关联键，或者即使是弱关联键但它是其中一个表的主键
                        is_t1_pk = col in table_pks[t1]
                        is_t2_pk = col in table_pks[t2]
                        
                        if not is_weak or is_t1_pk or is_t2_pk:
                            # 判定谁是 primary_table, 谁是 foreign_table
                            # 通常包含主键的表是 primary_table
                            if is_t2_pk and not is_t1_pk:
                                p_table, p_key = t2, col
                                f_table, f_key = t1, col
                            elif is_t1_pk and not is_t2_pk:
                                p_table, p_key = t1, col
                                f_table, f_key = t2, col
                            else:
                                # 如果两个都是主键，或者都不是，按照表名字典序决定，或者直接列出
                                p_table, p_key = t1, col
                                f_table, f_key = t2, col
                                
                            output_lines.append(f"    - 候选关系: {f_table}.{f_key} == {p_table}.{p_key} (同名列: {col})")
                            found_logical_fks = True
                            checked_pairs.add(pair_key)
            
            if not found_logical_fks:
                output_lines.append("    - 未推导到明显的逻辑外键")
                
            conn.close()
        except Exception as e:
            output_lines.append(f"处理任务 {task_name} 时发生异常: {e}")
            
    # 输出到屏幕
    final_output = "\n".join(output_lines)
    print(final_output)
    
    # 同时也保存到文件，以便查阅
    report_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\schema_analysis_result.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_output)
    print(f"\n[完成] 完整分析结果已保存至 {report_path}")

if __name__ == "__main__":
    analyze_tasks()
