# -*- coding: utf-8 -*-
"""
该脚本连接 task_1 到 task_15 的 SQLite 数据库，并提取其 Schema 信息，
包括表名、字段名、字段类型、是否为主键以及物理外键信息。
提取的信息会以结构化的中文格式写入到 data/schema_dump.txt 中。
"""

import os
import sqlite3


def dump_schemas():
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 输入输出路径
    output_file_path = os.path.join(project_root, "data", "schema_dump.txt")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    output_lines = []

    for i in range(1, 16):
        task_name = f"task_{i}"
        db_relative_path = os.path.join("data", "input", task_name, "context", "db", "sub_db.sqlite")
        db_path = os.path.join(project_root, db_relative_path)

        output_lines.append("=" * 60)
        output_lines.append(f"任务: {task_name}")
        output_lines.append(f"数据库路径: {db_relative_path}")
        output_lines.append("=" * 60)

        if not os.path.exists(db_path):
            output_lines.append(f"【错误】: 数据库文件不存在: {db_relative_path}")
            output_lines.append("\n")
            continue

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 1. 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            # 过滤掉 sqlite_sequence 等内部表
            tables = [t for t in tables if t != 'sqlite_sequence']

            if not tables:
                output_lines.append("数据库中没有表。")
            else:
                for table in tables:
                    output_lines.append(f"\n表名: {table}")

                    # 2. 获取表的所有字段信息
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = cursor.fetchall()
                    # columns 格式: (cid, name, type, notnull, dflt_value, pk)

                    output_lines.append("  字段列表:")
                    for col in columns:
                        col_name = col[1]
                        col_type = col[2] or "TEXT"  # 有时类型可能为空
                        is_pk = col[5]
                        pk_str = " [主键]" if is_pk > 0 else ""
                        notnull_str = " [非空]" if col[3] > 0 else ""
                        output_lines.append(f"    - {col_name} ({col_type}){pk_str}{notnull_str}")

                    # 3. 获取物理外键信息
                    cursor.execute(f"PRAGMA foreign_key_list({table});")
                    fks = cursor.fetchall()
                    # fks 格式: (id, seq, table, from, to, on_update, on_delete, match)

                    output_lines.append("  物理外键:")
                    if not fks:
                        output_lines.append("    - 无")
                    else:
                        for fk in fks:
                            from_col = fk[3]
                            to_table = fk[2]
                            to_col = fk[4]
                            to_col_str = f"({to_col})" if to_col else ""
                            output_lines.append(f"    - {from_col} -> {to_table}{to_col_str}")

            conn.close()
        except Exception as e:
            output_lines.append(f"【异常】: 处理任务 {task_name} 时发生异常: {str(e)}")

        output_lines.append("\n")

    # 将提取的 Schema 写入到文件中
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"成功将数据库 Schema 信息写入到: {output_file_path}")


if __name__ == "__main__":
    dump_schemas()
