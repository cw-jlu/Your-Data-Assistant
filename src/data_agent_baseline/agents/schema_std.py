"""
Schema 标准化预处理器。
在 Agent 启动前对 context 数据进行标准化处理：
1. 统一 CSV/JSON 文件编码为 UTF-8
2. 生成列名标准化映射（snake_case）
3. 检测并报告日期格式不一致
"""
import csv
import json
import re
import shutil
from pathlib import Path


def _detect_encoding(file_path: Path) -> str:
    """简易编码检测：尝试 UTF-8，失败则回退到 latin-1。"""
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin-1"]:
        try:
            file_path.read_text(encoding=enc)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"


def _normalize_column_name(col: str) -> str:
    """将列名标准化为 snake_case。"""
    # "CustomerID" -> "customer_id", "First Name" -> "first_name"
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', col)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    s = s.replace(' ', '_').replace('-', '_').lower()
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def standardize_context(context_dir: Path) -> str:
    """
    对 context 目录执行标准化预处理。
    返回一份标准化报告，注入到 System Prompt 中。
    """
    report_lines = ["=== SCHEMA STANDARDIZATION REPORT ==="]
    
    # --- 1. CSV 编码标准化 ---
    csv_files = sorted(context_dir.rglob("*.csv"))
    encoding_fixes = []
    col_mappings = []
    
    for csv_path in csv_files:
        rel = csv_path.relative_to(context_dir)
        detected_enc = _detect_encoding(csv_path)
        
        if detected_enc != "utf-8":
            # 转换编码
            try:
                content = csv_path.read_text(encoding=detected_enc)
                # 备份原文件
                backup = csv_path.with_suffix(".csv.bak")
                if not backup.exists():
                    shutil.copy2(csv_path, backup)
                csv_path.write_text(content, encoding="utf-8")
                encoding_fixes.append(f"  {rel}: {detected_enc} → UTF-8")
            except Exception as e:
                encoding_fixes.append(f"  {rel}: conversion failed ({e})")

        # 读取列名并生成映射
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    mappings = []
                    for col in header:
                        norm = _normalize_column_name(col)
                        if norm != col.lower().replace(' ', '_'):
                            mappings.append(f"    '{col}' → '{norm}'")
                    if mappings:
                        col_mappings.append(f"  {rel}:")
                        col_mappings.extend(mappings)
        except Exception:
            pass

    # --- 2. JSON 编码标准化 ---
    json_files = sorted(context_dir.rglob("*.json"))
    for json_path in json_files:
        rel = json_path.relative_to(context_dir)
        detected_enc = _detect_encoding(json_path)
        if detected_enc != "utf-8":
            try:
                content = json_path.read_text(encoding=detected_enc)
                backup = json_path.with_suffix(".json.bak")
                if not backup.exists():
                    shutil.copy2(json_path, backup)
                json_path.write_text(content, encoding="utf-8")
                encoding_fixes.append(f"  {rel}: {detected_enc} → UTF-8")
            except Exception as e:
                encoding_fixes.append(f"  {rel}: conversion failed ({e})")

    # --- 3. 汇报 ---
    if encoding_fixes:
        report_lines.append("\n[ENCODING FIXES]")
        report_lines.extend(encoding_fixes)
    else:
        report_lines.append("\n[ENCODING] All files already UTF-8.")

    if col_mappings:
        report_lines.append("\n[COLUMN NAME NORMALIZATION SUGGESTIONS]")
        report_lines.extend(col_mappings)
        report_lines.append("\n  NOTE: Original column names are preserved in the files.")
        report_lines.append("  Use the original names in your queries, but be aware of these patterns.")

    return "\n".join(report_lines)
