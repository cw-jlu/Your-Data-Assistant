import subprocess
import os
import json
from pathlib import Path
import time

# 配置
BRANCHES = [
    "main",
    "feat/db-schema-only",
    "exp/pagerag",
    "exp/graphrag",
    "exp/schema-std",
    "exp/kg+pagerag",
    "exp/kg+graphrag"
]

ARTIFACTS_ROOT = Path("artifacts/runs")

def run_command(cmd, cwd=None):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=cwd
    )
    
    # 实时打印输出，解决长耗时任务无反馈的问题
    output = []
    for line in process.stdout:
        print(line, end="")
        output.append(line)
    
    process.wait()
    return "".join(output), process.returncode

def get_latest_run_id():
    """找到最新的运行目录名"""
    run_dirs = sorted(ARTIFACTS_ROOT.glob("*"), key=lambda p: p.name, reverse=True)
    if not run_dirs:
        return None
    return run_dirs[0].name

def main():
    results_summary = []
    
    print("=== KDD Cup 2026 Experiment Runner (V2) ===")
    
    for branch in BRANCHES:
        print(f"\n\n>>> Testing Branch: {branch}")
        
        # 1. 切换分支
        _, code = run_command(f"git checkout {branch}")
        if code != 0:
            print(f"Failed to checkout {branch}, skipping...")
            continue
            
        # 2. 同步依赖 (确保 V2 模块可用)
        run_command("uv sync")
        
        # 3. 运行测试 (冒烟测试：8个难题)
        # 如果你想跑全量，请改为 uv run dabench run-benchmark --config configs/react_baseline.local.yaml
        print(f"Running Benchmark on {branch}...")
        run_command("uv run python run_failed.py")
        
        # 4. 获取 Run ID 并评估
        run_id = get_latest_run_id()
        if not run_id:
            print("No run artifacts found.")
            continue
            
        print(f"Evaluating Run: {run_id}...")
        eval_cmd = f"uv run python evaluate.py --run-id {run_id} --branch {branch} --save-md"
        output, _ = run_command(eval_cmd)
        
        # 5. 解析评分 (从 evaluate.py 的输出中提取)
        # 简单粗暴点：读取生成的 report.md 或 summary.json
        summary_path = ARTIFACTS_ROOT / run_id / "summary.json"
        score = 0.0
        success_rate = "0/0"
        
        # 再次运行 evaluate 以获取精准分数值（这里通过重新解析 summary.json 更稳妥）
        # 但 evaluate.py 已经把得分存入 summary.json 了吗？不一定。
        # 我们从 evaluate.py 的标准输出中捕获 Overall Statistics
        try:
            if "正确率（平均Score）：" in output:
                score = output.split("正确率（平均Score）：")[1].split("=")[0].strip()
            if "执行成功率：" in output:
                success_rate = output.split("执行成功率：")[1].split("=")[0].strip()
        except:
            pass
            
        results_summary.append({
            "branch": branch,
            "run_id": run_id,
            "score": score,
            "success_rate": success_rate
        })

    # 6. 生成对比汇总表
    print("\n\n" + "="*50)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*50)
    print(f"| {'Branch':<20} | {'Run ID':<20} | {'Score':<10} | {'Success':<10} |")
    print(f"| {'-'*20} | {'-'*20} | {'-'*10} | {'-'*10} |")
    
    report_md = ["# 实验对比汇总报告\n", "| 分支 | Run ID | 平均得分 | 成功数 |\n", "| :--- | :--- | :--- | :--- |\n"]
    
    for res in results_summary:
        print(f"| {res['branch']:<20} | {res['run_id']:<20} | {res['score']:<10} | {res['success_rate']:<10} |")
        report_md.append(f"| **{res['branch']}** | `{res['run_id']}` | {res['score']} | {res['success_rate']} |\n")
    
    # 保存汇总文件
    with open("experiment_summary.md", "w", encoding="utf-8") as f:
        f.writelines(report_md)
        
    print(f"\nFull report saved to: experiment_summary.md")
    
    # 7. 切回 main
    run_command("git checkout main")

if __name__ == "__main__":
    # 确保在根目录执行
    main()
