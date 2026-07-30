"""分析 data/input 下所有非 knowledge.md 的 md 文档中，首数字出现位置与行类型的关系。

目标：找到一个 N 值，使得规则"前 N 个字符内无数字 → 边界行"能够
正确区分数据行和叙述/标题行。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/input")


def first_digit_position(text: str) -> int | None:
    """返回第一个阿拉伯数字 (0-9) 在文本中的位置，-1 表示不存在。"""
    for i, ch in enumerate(text):
        if ch.isdigit():
            return i
    return None


def line_kind(text: str) -> str:
    """将行分为几类。"""
    stripped = text.strip()
    if not stripped:
        return "empty"

    # Markdown 标题
    if re.match(r"^#{1,6}\s+", stripped):
        return "heading"

    # Markdown 表格行
    if stripped.startswith("|") or (stripped.endswith("|") and "|" in stripped[:-1]):
        return "table_row"

    # 列表项
    if re.match(r"^(?:[-*+]\s+|\d+[.)、]\s+)", stripped):
        return "list_item"

    # 键值对风格 (key: value 或 key： value)
    if re.match(r"^[^\d:：]+[:：]\s*\S", stripped):
        return "key_value"

    # 纯数字+单位结尾的行（可能的数据行或标题）
    # 包含大量逗号 → CSV 风格
    if stripped.count(",") >= 2 and any(ch.isdigit() for ch in stripped):
        return "csv_data"

    # 包含数字 → 候选数据行
    if any(ch.isdigit() for ch in stripped):
        return "has_digits"

    # 无数字 → 纯叙述
    return "no_digits"


def analyze_file(path: Path) -> dict:
    """分析单个文件。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]

    heading_positions = []
    narr_positions = []       # no_digits: 无数字的纯叙述行
    has_digit_positions = []  # has_digits: 含数字但不是明确结构化数据的行
    table_positions = []
    list_positions = []
    kv_positions = []
    csv_positions = []
    sparse_positions = []     # 含数字但第一个数字位置 > N 的行

    for line in lines:
        kind = line_kind(line)
        pos = first_digit_position(line)

        if kind == "heading":
            if pos is not None:
                heading_positions.append(pos)
        elif kind == "table_row":
            if pos is not None:
                table_positions.append(pos)
        elif kind == "list_item":
            if pos is not None:
                list_positions.append(pos)
        elif kind == "key_value":
            if pos is not None:
                kv_positions.append(pos)
        elif kind == "csv_data":
            if pos is not None:
                csv_positions.append(pos)
        elif kind == "has_digits":
            if pos is not None:
                has_digit_positions.append(pos)
        elif kind == "no_digits":
            narr_positions.append(-1)  # 哨兵

    return {
        "file": str(path.relative_to(DATA_DIR)),
        "total_lines": len(lines),
        "heading_count": len([l for l in lines if line_kind(l) == "heading"]),
        "heading_digit_positions": heading_positions,
        "narr_count": len(narr_positions),
        "has_digit_count": len(has_digit_positions),
        "has_digit_positions": has_digit_positions,
        "table_count": len(table_positions),
        "table_digit_positions": table_positions,
        "list_count": len(list_positions),
        "list_digit_positions": list_positions,
        "kv_count": len(kv_positions),
        "kv_digit_positions": kv_positions,
        "csv_count": len(csv_positions),
        "csv_digit_positions": csv_positions,
    }


def main():
    all_heading_pos = []
    all_narr_count = 0
    all_has_digit_pos = []   # has_digits 行的首数字位置
    all_table_pos = []
    all_list_pos = []
    all_kv_pos = []
    all_csv_pos = []

    files = sorted(DATA_DIR.rglob("*.md"))
    files = [f for f in files if f.name != "knowledge.md"]

    for f in files:
        result = analyze_file(f)
        all_heading_pos.extend(result["heading_digit_positions"])
        all_narr_count += result["narr_count"]
        all_has_digit_pos.extend(result["has_digit_positions"])
        all_table_pos.extend(result["table_digit_positions"])
        all_list_pos.extend(result["list_digit_positions"])
        all_kv_pos.extend(result["kv_digit_positions"])
        all_csv_pos.extend(result["csv_digit_positions"])

    print(f"分析文件数: {len(files)}")
    print(f"总行数统计:")
    print(f"  标题行:     {len(all_heading_pos):>6} (其中有数字的: {len([p for p in all_heading_pos if p >= 0])})")
    print(f"  纯叙述行:   {all_narr_count:>6} (无数字)")
    print(f"  has_digits: {len(all_has_digit_pos):>6}")
    print(f"  表格行:     {len(all_table_pos):>6}")
    print(f"  列表行:     {len(all_list_pos):>6}")
    print(f"  KV行:       {len(all_kv_pos):>6}")
    print(f"  CSV数据行:  {len(all_csv_pos):>6}")
    print()

    # has_digits 行（候选边界行）的首数字位置分布
    # 这些是"含数字但不够结构化"的行，是我们新规则要重点区分的
    print("=" * 70)
    print("has_digits 行的首数字位置分布（这些是最需要新规则判断的）:")
    print("=" * 70)
    if all_has_digit_pos:
        sorted_pos = sorted(all_has_digit_pos)
        print(f"  样本数: {len(sorted_pos)}")
        print(f"  最小值: {min(sorted_pos)}")
        print(f"  最大值: {max(sorted_pos)}")
        print(f"  中位数: {sorted_pos[len(sorted_pos)//2]}")
        print(f"  平均值: {sum(sorted_pos)/len(sorted_pos):.1f}")

        # 分位数
        for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
            idx = int(len(sorted_pos) * pct / 100)
            print(f"  P{pct}: {sorted_pos[min(idx, len(sorted_pos)-1)]}")

    # 关键分析：对于不同的 N，看看"前 N 字符内无数字"这条规则
    # 会把多少 has_digits 行判定为边界，把多少表格/数据行判定为边界
    print()
    print("=" * 70)
    print("对于不同 N 值，规则「前 N 字符内无数字 → 边界行」的影响:")
    print("=" * 70)

    # 把需要保护的数据行类型汇总
    data_positions = all_table_pos + all_list_pos + all_csv_pos

    for n in [0, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
        # has_digits 中，前 N 字符无数字的会被判为边界
        has_digit_boundary = sum(1 for p in all_has_digit_pos if p < 0 or p > n)
        has_digit_data = sum(1 for p in all_has_digit_pos if 0 <= p <= n)

        # 数据行（表格/列表/CSV）中，前 N 字符无数字的会被误判为边界
        data_boundary = sum(1 for p in data_positions if p < 0 or p > n)
        data_safe = sum(1 for p in data_positions if 0 <= p <= n)

        # 标题行中，前 N 字符有数字的（意味着规则无法覆盖）
        heading_covered = sum(1 for p in all_heading_pos if p < 0 or p > n)
        heading_missed = sum(1 for p in all_heading_pos if 0 <= p <= n)

        total_boundary = has_digit_boundary + data_boundary + len(all_heading_pos) + all_narr_count
        total_data = has_digit_data + data_safe

        print(f"\n  N={n:>3}:")
        print(f"    has_digits → 边界: {has_digit_boundary:>6}  数据: {has_digit_data:>6}")
        print(f"    结构化数据 → 边界: {data_boundary:>6}  数据: {data_safe:>6}  (误判边界=漏提取)")
        print(f"    标题行覆盖: {heading_covered:>6}  未覆盖: {heading_missed:>6}")

    # 打印一些 has_digits 行中首数字位置较大的样本
    print()
    print("=" * 70)
    print("has_digits 行中「首数字位置较大」的样本（新规则会判为边界的）:")
    print("=" * 70)
    samples = []
    for f in files[:10]:
        text = f.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            kind = line_kind(line)
            if kind == "has_digits":
                pos = first_digit_position(line)
                if pos is not None and pos > 10:
                    samples.append((pos, line.strip()[:80]))
    samples.sort(key=lambda x: x[0], reverse=True)
    for pos, text in samples[:20]:
        print(f"  pos={pos:>3}: {text}")

    # 打印 has_digits 行中首数字位置很小的样本（新规则会判为数据的）
    print()
    print("=" * 70)
    print("has_digits 行中「首数字位置很小」的样本（新规则会判为数据的）:")
    print("=" * 70)
    samples2 = []
    for f in files[:10]:
        text = f.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            kind = line_kind(line)
            if kind == "has_digits":
                pos = first_digit_position(line)
                if pos is not None and pos <= 5:
                    samples2.append((pos, line.strip()[:80]))
    samples2.sort(key=lambda x: x[0])
    for pos, text in samples2[:20]:
        print(f"  pos={pos:>3}: {text}")


if __name__ == "__main__":
    main()
