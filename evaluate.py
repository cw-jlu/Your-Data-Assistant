#!/usr/bin/env python3
"""
评分脚本：按照官方 Leaderboard Scoring Rule 对预测结果与金标准进行比较。

评分规则（6.3 Scoring Metric Calculation）：
1. 列级内容一致性匹配：基于列数据内容（忽略列名和行顺序）
2. 对每个任务独立评分：
   - Recall = Matched Columns / Gold Columns
   - Score = Recall - λ·(Extra Columns / Predicted Columns)
   - Score的下界为0
3. 总分 = 所有任务的平均Score
4. 排序：按总分降序，同分按提交时间升序

支持功能：
- 支持重复列（相同数据内容的列）
- 基于列内容签名匹配，忽略列名
- 重视完全覆盖（Recall优先），同时控制冗余输出
"""
import sys
from pathlib import Path
from typing import Optional
import csv
import json
from collections import defaultdict

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / "public"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "runs"

# 全局评分参数
LAMBDA_PARAM = 0.5  # 惩罚项权重，用于平衡覆盖率和冗余预测

console = Console()


def read_csv_as_dict_of_lists(csv_path: Path) -> dict[str, list[str]]:
    """
    读取 CSV 文件，返回 {列名: [值列表]} 的字典。
    """
    data = defaultdict(list)
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}
            for row in reader:
                for col_name, value in row.items():
                    if col_name:
                        data[col_name].append(str(value) if value is not None else "")
        return dict(data)
    except Exception as e:
        console.print(f"[red]Error reading {csv_path}: {e}[/red]")
        return {}


def normalize_value(val: str) -> str:
    val = val.strip()
    if val.lower() in ["", "null", "nan", "none", "na", "<na>"]:
        return ""
    try:
        f_val = float(val)
        # Handle cases where .00 is not needed vs needed. Formatting to .2f.
        return f"{f_val:.2f}"
    except ValueError:
        return val

def normalize_column_vector(col_values: list[str]) -> list[str]:
    """
    将列向量规范化为带频次的有序列表（支持无序且保留重复项频次的比较）。
    
    官方规则：基于列数据内容匹配，忽略列名和行顺序，且支持重复列（频率需一致）
    """
    return sorted(normalize_value(v) for v in col_values)


def compare_column_vectors(pred_vector: list[str], gold_vector: list[str]) -> bool:
    """
    比较两个列向量是否相同（忽略行数序，但元素频次必须完全一致）。
    """
    return pred_vector == gold_vector


def evaluate_task(task_id: str, pred_csv_path: Path, gold_csv_path: Path) -> dict:
    """
    评估单个任务，按官方规则进行列级匹配。
    
    返回结果字典：
    {
        task_id: str,
        score: int (0 or 1, for backward compatibility),
        gold_columns: list,        # gold CSV 的所有列名
        pred_columns: list,        # prediction CSV 的所有列名
        matched_columns: list,     # 成功匹配的列名
        missing_columns: list,     # 缺失的列名（在gold中但不在pred中）
        extra_columns: list,       # 额外的列名（在pred中但不在gold中）
        mismatched_columns: dict,  # 不匹配的列：{col_name: {gold: [], pred: []}}
        error: str or None,        # 错误信息
        _score: float,             # 计算出的最终分数 (0-1)
        _recall: float,            # Recall值
        _penalty: float,           # 惩罚项值
    }
    """
    result = {
        "task_id": task_id,
        "score": 0,
        "gold_columns": [],
        "pred_columns": [],
        "matched_columns": [],
        "missing_columns": [],
        "extra_columns": [],
        "mismatched_columns": {},  # {col_name: (gold_set, pred_set)}
        "error": None,
    }

    # 检查文件是否存在
    if not gold_csv_path.exists():
        result["error"] = f"Gold CSV not found: {gold_csv_path}"
        return result
    
    if not pred_csv_path.exists():
        result["error"] = f"Prediction CSV not found: {pred_csv_path}"
        return result

    # 读取 CSV
    gold_data = read_csv_as_dict_of_lists(gold_csv_path)
    pred_data = read_csv_as_dict_of_lists(pred_csv_path)

    if not gold_data:
        result["error"] = f"Gold CSV is empty or cannot be read: {gold_csv_path}"
        return result

    if not pred_data:
        result["error"] = f"Prediction CSV is empty or cannot be read: {pred_csv_path}"
        return result

    gold_columns = set(gold_data.keys())
    pred_columns = set(pred_data.keys())

    result["gold_columns"] = sorted(gold_columns)
    result["pred_columns"] = sorted(pred_columns)

    # 检查每个 gold 列是否被正确预测
    import itertools
    gold_vectors = {c: normalize_column_vector(gold_data[c]) for c in gold_columns}
    pred_vectors = {c: normalize_column_vector(pred_data[c]) for c in pred_columns}

    matched_gold_cols = set()
    matched_pred_cols = set()

    for g_col, g_vec in gold_vectors.items():
        for p_col, p_vec in pred_vectors.items():
            if p_col in matched_pred_cols:
                continue
            if compare_column_vectors(p_vec, g_vec):
                result["matched_columns"].append(g_col)
                matched_gold_cols.add(g_col)
                matched_pred_cols.add(p_col)
                break

    unmatched_gold = set(gold_columns) - matched_gold_cols
    unmatched_pred = set(pred_columns) - matched_pred_cols

    # 尝试合并 gold 列 (例如 first_name, last_name -> full_name)
    # 限制：仅当列名包含 "name" 时才进行合并尝试，防止非预期的拼接命中
    if len(unmatched_gold) >= 2 and len(unmatched_pred) >= 1:
        for g1, g2 in list(itertools.permutations(unmatched_gold, 2)):
            if g1 not in unmatched_gold or g2 not in unmatched_gold: continue
            if "name" not in g1.lower() and "name" not in g2.lower(): continue
            merged_gold_vec = sorted(normalize_value(f"{v1} {v2}") for v1, v2 in zip(gold_data[g1], gold_data[g2]))
            for p_col in list(unmatched_pred):
                if merged_gold_vec == pred_vectors[p_col]:
                    result["matched_columns"].extend([g1, g2])
                    matched_gold_cols.update([g1, g2])
                    matched_pred_cols.add(p_col)
                    unmatched_gold.difference_update([g1, g2])
                    unmatched_pred.remove(p_col)
                    break

    # 尝试合并 pred 列 (例如 full_name -> first_name, last_name)
    if len(unmatched_pred) >= 2 and len(unmatched_gold) >= 1:
        for p1, p2 in list(itertools.permutations(unmatched_pred, 2)):
            if p1 not in unmatched_pred or p2 not in unmatched_pred: continue
            if "name" not in p1.lower() and "name" not in p2.lower(): continue
            merged_pred_vec = sorted(normalize_value(f"{v1} {v2}") for v1, v2 in zip(pred_data[p1], pred_data[p2]))
            for g_col in list(unmatched_gold):
                if merged_pred_vec == gold_vectors[g_col]:
                    result["matched_columns"].append(g_col)
                    matched_gold_cols.add(g_col)
                    matched_pred_cols.update([p1, p2])
                    unmatched_gold.remove(g_col)
                    unmatched_pred.difference_update([p1, p2])
                    break

    matched = True
    for g_col in sorted(gold_columns):
        if g_col not in matched_gold_cols:
            result["missing_columns"].append(g_col)
            result["mismatched_columns"][g_col] = {
                "gold": sorted(gold_vectors[g_col]),
                "pred": [],
            }
            matched = False

    # 检查额外列
    extra_cols = set(pred_columns) - matched_pred_cols
    result["extra_columns"] = sorted(extra_cols)

    # 计算分数 (backward compatibility: score 0 or 1)
    result["score"] = 1 if matched else 0

    return result


def evaluate_run(run_id: Optional[str] = None) -> list[dict]:
    """
    评估整个运行。如果 run_id 为 None，使用最新的运行。
    """
    if run_id is None:
        # 找到最新的运行目录
        run_dirs = sorted(ARTIFACTS_ROOT.glob("*"), key=lambda p: p.name, reverse=True)
        if not run_dirs:
            console.print("[red]No run directories found in artifacts/runs/[/red]")
            return []
        run_id = run_dirs[0].name
    
    run_output_dir = ARTIFACTS_ROOT / run_id
    if not run_output_dir.exists():
        console.print(f"[red]Run directory not found: {run_output_dir}[/red]")
        return []

    results = []
    
    # 遍历所有任务目录
    task_dirs = sorted(
        [d for d in run_output_dir.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda d: int(d.name.split("_")[1])
    )

    for task_dir in task_dirs:
        task_id = task_dir.name
        pred_csv = task_dir / "prediction.csv"
        gold_csv = DATA_ROOT / "output" / task_id / "gold.csv"
        
        result = evaluate_task(task_id, pred_csv, gold_csv)
        results.append(result)

    return results


def calculate_scores(results: list[dict]) -> dict:
    """
    按照官方评分规则计算评分指标。
    使用全局 LAMBDA_PARAM 作为惩罚项权重。
    
    对每个任务：
    - Recall = Matched Columns / Gold Columns
    - Score = Recall - LAMBDA_PARAM·(Extra Columns / Predicted Columns)
    - Score的下界为0
    
    整体指标：
    - 执行成功率 = 成功运行的任务 / 总任务数
    - 正确率 = 所有有效任务的平均Score
    - 完全正确率 = Score = 1.0 的任务 / 有效任务数
    """
    total_tasks = len(results)
    if total_tasks == 0:
        return {}
    
    non_error_tasks = 0
    perfect_score_count = 0
    total_score_sum = 0.0
    
    for result in results:
        if result["error"]:
            continue
        
        non_error_tasks += 1
        
        gold_count = len(result["gold_columns"])
        matched_count = len(result["matched_columns"])
        extra_count = len(result["extra_columns"])
        pred_count = len(result["pred_columns"])
        
        # 计算 Recall
        recall = matched_count / gold_count if gold_count > 0 else 0.0
        
        # 计算惩罚项
        penalty = LAMBDA_PARAM * (extra_count / pred_count) if pred_count > 0 else 0.0
        
        # 计算最终分数（下界为0）
        score = max(0.0, recall - penalty)
        
        total_score_sum += score
        
        # 统计完全正确的任务
        if abs(score - 1.0) < 1e-6:
            perfect_score_count += 1
        
        # 存储计算结果到原result中
        result["_score"] = score
        result["_recall"] = recall
        result["_penalty"] = penalty
    
    execution_rate = (non_error_tasks / total_tasks * 100) if total_tasks > 0 else 0
    avg_score = (total_score_sum / non_error_tasks) if non_error_tasks > 0 else 0
    perfect_rate = (perfect_score_count / non_error_tasks * 100) if non_error_tasks > 0 else 0
    
    return {
        "total_tasks": total_tasks,
        "non_error_tasks": non_error_tasks,
        "lambda_param": LAMBDA_PARAM,
        "execution_rate": execution_rate,
        "accuracy": avg_score * 100,
        "perfect_rate": perfect_rate,
        "average_score": avg_score,
        "perfect_count": perfect_score_count,
    }


def print_summary(results: list[dict]) -> dict:
    """
    打印评分汇总表。
    使用全局 LAMBDA_PARAM 进行评分。
    """
    table = Table(title="Evaluation Summary")
    table.add_column("Task ID", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Status", style="bold")
    table.add_column("Recall/Penalty")

    for result in results:
        task_id = result["task_id"]
        score = result.get("_score", result["score"])

        if result["error"]:
            table.add_row(task_id, "ERROR", "[red]ERROR[/red]", result["error"][:40])
        else:
            recall = result.get("_recall", 0)
            penalty = result.get("_penalty", 0)
            score_str = f"{score:.2f}"
            
            if abs(score - 1.0) < 1e-6:  # 完全正确
                table.add_row(task_id, f"[green]{score_str}[/green]", "[green]PASS[/green]", f"Recall={recall:.2f}, Penalty={penalty:.2f}")
            elif score >= 0.5:
                table.add_row(task_id, f"[yellow]{score_str}[/yellow]", "[yellow]PASS[/yellow]", f"Recall={recall:.2f}, Penalty={penalty:.2f}")
            else:
                table.add_row(task_id, f"[red]{score_str}[/red]", "[red]FAIL[/red]", f"Recall={recall:.2f}, Penalty={penalty:.2f}")

    console.print(table)
    
    # 计算评分
    scores = calculate_scores(results)
    
    # 汇总统计 - 三个关键指标
    summary_text = f"\n[bold]总任务数：{scores['total_tasks']}[/bold]\n"
    summary_text += f"[bold]有效任务数（无错误）：{scores['non_error_tasks']}[/bold]\n"
    summary_text += f"[dim]Lambda参数：{scores['lambda_param']}[/dim]\n\n"
    summary_text += f"[yellow]1️⃣  执行成功率：{scores['non_error_tasks']}/{scores['total_tasks']} = {scores['execution_rate']:.2f}%[/yellow]\n"
    summary_text += f"[cyan]2️⃣  正确率（平均Score）：{scores['average_score']:.4f} = {scores['accuracy']:.2f}%[/cyan]\n"
    summary_text += f"[green]3️⃣  完全正确率（Score=1.0）：{scores['perfect_count']}/{scores['non_error_tasks']} = {scores['perfect_rate']:.2f}%[/green]"
    
    console.print(Panel(summary_text, title="Overall Statistics"))
    
    return scores


def print_detailed_errors(results: list[dict]) -> None:
    """打印详细的错误对比。"""
    failed_tasks = [r for r in results if r["score"] == 0 and not r["error"]]
    
    if not failed_tasks:
        console.print("[green]✓ All tasks passed![/green]")
        return

    console.print(f"\n[bold red]Failed Tasks: {len(failed_tasks)}[/bold red]\n")

    for result in failed_tasks:
        task_id = result["task_id"]
        console.print(f"[bold underline]{task_id}[/bold underline]")

        # 打印缺失列
        if result["missing_columns"]:
            console.print("[yellow]❌ Missing Columns:[/yellow]")
            for col in result["missing_columns"]:
                console.print(f"   - {col}")

        # 打印不匹配列
        if result["mismatched_columns"]:
            console.print("[yellow]❌ Mismatched Columns:[/yellow]")
            for col_name, vectors in result["mismatched_columns"].items():
                gold_set = vectors["gold"]
                pred_set = vectors["pred"]
                
                console.print(f"\n   Column: [cyan]{col_name}[/cyan]")
                console.print(f"   [green]Gold (Expected):[/green] {gold_set}")
                console.print(f"   [red]Pred (Got):[/red]      {pred_set}")
                
                # 显示差异
                missing_vals = set(gold_set) - set(pred_set)
                extra_vals = set(pred_set) - set(gold_set)
                
                if missing_vals:
                    console.print(f"   [red]   Missing values: {missing_vals}[/red]")
                if extra_vals:
                    console.print(f"   [yellow]   Extra values: {extra_vals}[/yellow]")

        # 打印额外列（只提示，不影响评分）
        if result["extra_columns"]:
            console.print(f"[blue]ℹ Extra Columns (不影响评分):[/blue]")
            for col in result["extra_columns"]:
                console.print(f"   - {col}")

        console.print()


def generate_markdown_report(results: list[dict], run_id: str, scores: dict, agent_cfg: dict = None) -> str:
    """
    生成markdown格式的评估报告。
    """
    md = []
    md.append("# 评估报告\n")
    md.append(f"**运行ID：** {run_id}\n")
    md.append(f"**评分公式：** Score = Recall - λ·(Extra/Predicted), λ = {scores['lambda_param']}\n\n")
    
    if agent_cfg:
        md.append("## ⚙️ 运行参数\n")
        md.append(f"- **Model**: {agent_cfg.get('model', 'N/A')}\n")
        md.append(f"- **Max Steps**: {agent_cfg.get('max_steps', 'N/A')}\n")
        md.append(f"- **Temperature**: {agent_cfg.get('temperature', 'N/A')}\n")
        md.append(f"- **Max Tokens**: {agent_cfg.get('max_tokens', 'N/A')}\n\n")
    
    # 统计信息
    md.append("## 📊 评分统计\n")
    md.append("| 指标 | 数值 |\n")
    md.append("|-----|-----|\n")
    md.append(f"| 总任务数 | {scores['total_tasks']} |\n")
    md.append(f"| 有效任务数 | {scores['non_error_tasks']} |\n")
    md.append(f"| **执行成功率** | {scores['non_error_tasks']}/{scores['total_tasks']} = {scores['execution_rate']:.2f}% |\n")
    md.append(f"| **正确率（平均Score）** | {scores['average_score']:.4f} = {scores['accuracy']:.2f}% |\n")
    md.append(f"| **完全正确率（Score=1.0）** | {scores['perfect_count']}/{scores['non_error_tasks']} = {scores['perfect_rate']:.2f}% |\n\n")
    
    # 详细结果
    md.append("## 任务结果详情\n\n")
    
    # 按状态分类
    passed_tasks = [r for r in results if r.get("_score", 0) >= 1.0 and not r["error"]]
    partial_tasks = [r for r in results if 0 < r.get("_score", 0) < 1.0 and not r["error"]]
    failed_tasks = [r for r in results if r.get("_score", 0) == 0 and not r["error"]]
    error_tasks = [r for r in results if r["error"]]
    
    # 完全正确的任务
    if passed_tasks:
        md.append(f"### ✅ 完全正确的任务 ({len(passed_tasks)})\n\n")
        for result in passed_tasks:
            recall = result.get("_recall", 1.0)
            penalty = result.get("_penalty", 0)
            score = result.get("_score", 1.0)
            md.append(f"- **{result['task_id']}** (Score={score:.4f}): {len(result['gold_columns'])}列完全匹配, Recall={recall:.2f}, Penalty={penalty:.4f}\n")
        md.append("\n")
    
    # 部分正确的任务
    if partial_tasks:
        md.append(f"### ⚠️ 部分正确的任务 ({len(partial_tasks)})\n\n")
        partial_tasks_sorted = sorted(partial_tasks, key=lambda r: r.get("_score", 0), reverse=True)
        for result in partial_tasks_sorted:
            task_id = result["task_id"]
            score = result.get("_score", 0)
            recall = result.get("_recall", 0)
            penalty = result.get("_penalty", 0)
            
            md.append(f"#### {task_id}\n\n")
            md.append(f"**得分:** {score:.4f} (Recall={recall:.2f}, Penalty={penalty:.4f})\n\n")
            md.append(f"| 指标 | 数值 |\n")
            md.append(f"|-----|-----|\n")
            md.append(f"| 预期列数 | {len(result['gold_columns'])} |\n")
            md.append(f"| 预测列数 | {len(result['pred_columns'])} |\n")
            md.append(f"| 匹配列数 | {len(result['matched_columns'])} |\n")
            md.append(f"| 缺失列数 | {len(result['missing_columns'])} |\n")
            md.append(f"| 不匹配列数 | {len(result['mismatched_columns'])} |\n")
            md.append(f"| 额外列数 | {len(result['extra_columns'])} |\n\n")
            
            if result["missing_columns"]:
                md.append(f"**❌ 缺失列 ({len(result['missing_columns'])}):**\n")
                for col in result["missing_columns"]:
                    md.append(f"- `{col}`\n")
                md.append("\n")
            
            if result["mismatched_columns"]:
                md.append(f"**❌ 不匹配的列 ({len(result['mismatched_columns'])}):**\n")
                for col_name, vectors in result["mismatched_columns"].items():
                    gold_set = vectors["gold"]
                    pred_set = vectors["pred"]
                    missing_vals = set(gold_set) - set(pred_set)
                    extra_vals = set(pred_set) - set(gold_set)
                    
                    md.append(f"- **`{col_name}`**: 预期{len(gold_set)}个值, 实际{len(pred_set)}个值\n")
                    if missing_vals and len(missing_vals) <= 5:
                        md.append(f"  - 缺失: {missing_vals}\n")
                    if extra_vals and len(extra_vals) <= 5:
                        md.append(f"  - 多余: {extra_vals}\n")
                
                md.append("\n")
            
            if result["extra_columns"]:
                md.append(f"**ℹ️ 额外列 ({len(result['extra_columns'])}):**\n")
                for col in result["extra_columns"][:10]:  # 最多显示10个
                    md.append(f"- `{col}`\n")
                if len(result["extra_columns"]) > 10:
                    md.append(f"- ... 以及{len(result['extra_columns']) - 10}个其他列\n")
                md.append("\n")
        
        md.append("\n")
    
    # 完全错误的任务
    if failed_tasks:
        md.append(f"### ❌ 完全错误的任务 ({len(failed_tasks)})\n\n")
        for result in failed_tasks:
            task_id = result["task_id"]
            md.append(f"- **{task_id}**\n")
        md.append("\n")
    
    # 执行失败的任务
    if error_tasks:
        md.append(f"### 🚫 执行失败的任务 ({len(error_tasks)})\n\n")
        for result in error_tasks:
            md.append(f"- **{result['task_id']}**: {result['error']}\n")
        md.append("\n")
    
    return "".join(md)


def save_markdown_report(results: list[dict], run_id: str, scores: dict, output_path: Optional[Path] = None, agent_cfg: dict = None) -> Path:
    """
    保存markdown报告到文件。
    默认保存到 artifacts/runs/{run_id}/report.md
    """
    if output_path is None:
        output_path = ARTIFACTS_ROOT / run_id / "report.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_content = generate_markdown_report(results, run_id, scores, agent_cfg=agent_cfg)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    console.print(f"[green]✓ Markdown report saved to: {output_path}[/green]")
    return output_path


def main():
    """主函数。"""
    import argparse
    global LAMBDA_PARAM

    parser = argparse.ArgumentParser(
        description="Evaluate predictions against gold standard using official Leaderboard Scoring Rule"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (default: latest run)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Show only error details",
    )
    parser.add_argument(
        "--save-md",
        action="store_true",
        help="Save evaluation report to markdown file",
    )
    parser.add_argument(
        "--md-path",
        type=str,
        default=None,
        help="Custom path to save markdown report",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=LAMBDA_PARAM,
        dest="lambda_param",
        help=f"Lambda parameter for penalty term (default: {LAMBDA_PARAM})",
    )

    args = parser.parse_args()

    # 如果命令行指定了lambda，更新全局变量
    if args.lambda_param != LAMBDA_PARAM:
        LAMBDA_PARAM = args.lambda_param

    # 确定 run_id
    if args.run_id is None:
        run_dirs = sorted(ARTIFACTS_ROOT.glob("*"), key=lambda p: p.name, reverse=True)
        if not run_dirs:
            console.print("[red]No run directories found in artifacts/runs/[/red]")
            return
        args.run_id = run_dirs[0].name

    console.print(f"[bold blue]Evaluating run: {args.run_id}[/bold blue]")
    console.print(f"[dim]Global LAMBDA_PARAM: {LAMBDA_PARAM}[/dim]\n")

    agent_cfg = None
    summary_path = ARTIFACTS_ROOT / args.run_id / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            if "config" in summary_data:
                agent_cfg = summary_data["config"].get("agent", {})
                params_text = (
                    f"Model: {agent_cfg.get('model', 'N/A')}\n"
                    f"Max Steps: {agent_cfg.get('max_steps', 'N/A')}\n"
                    f"Temperature: {agent_cfg.get('temperature', 'N/A')}\n"
                    f"Max Tokens: {agent_cfg.get('max_tokens', 'N/A')}"
                )
                console.print(Panel(params_text, title="Run Parameters", border_style="cyan"))
        except Exception as e:
            console.print(f"[dim]Could not load run parameters from summary.json: {e}[/dim]")

    results = evaluate_run(args.run_id)

    if not results:
        console.print("[red]No results to evaluate.[/red]")
        return

    if not args.errors_only:
        scores = print_summary(results)
    else:
        scores = calculate_scores(results)

    print_detailed_errors(results)

    # 保存markdown报告
    if args.save_md or args.md_path:
        md_path = Path(args.md_path) if args.md_path else None
        save_markdown_report(results, args.run_id, scores, md_path, agent_cfg=agent_cfg)


if __name__ == "__main__":
    main()
