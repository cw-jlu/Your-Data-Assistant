"""
此模块提供 Python 代码执行工具，允许 Agent 在任务上下文中运行自定义数据处理逻辑。
改用 subprocess 并通过临时文件传递代码，彻底解决转义和稳定性问题。
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


def execute_python_code(context_root: Path, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    """
    通过 subprocess 在独立进程中执行 Python 代码。
    代码通过独立的 .py 文件传递，避免 repr() 或 json 带来的转义困扰。
    """
    resolved_context_root = context_root.resolve()
    scratch_root = Path.cwd() / "scratch" / "execute_python"
    scratch_root.mkdir(parents=True, exist_ok=True)
    
    run_id = uuid.uuid4().hex
    
    # 1. 代理代码文件：存放 Agent 编写的原始代码
    agent_code_path = scratch_root / f"{run_id}_agent.py"
    # 2. 包装脚本文件：负责环境准备、目录切换和执行代理代码
    wrapper_path = scratch_root / f"{run_id}_wrapper.py"
    
    wrapper_code = f"""
import os
import sys
import traceback
from pathlib import Path

context_root = Path({repr(resolved_context_root.as_posix())})
agent_code_file = Path({repr(agent_code_path.as_posix())})

# 切换到任务上下文目录
os.chdir(context_root)

# 准备执行命名空间
namespace = {{
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "context_root": context_root,
    "Path": Path,
}}

try:
    import pandas as pd
    namespace["pd"] = pd
except ImportError:
    pass

try:
    # 直接读取并执行原始代码文件，不经过任何中间转义
    with open(agent_code_file, "r", encoding="utf-8") as f:
        exec(f.read(), namespace, namespace)
except BaseException:
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
"""

    try:
        # 写入代码文件
        agent_code_path.write_text(code, encoding="utf-8")
        wrapper_path.write_text(wrapper_code, encoding="utf-8")
        
        # 执行子进程
        result = subprocess.run(
            [sys.executable, str(wrapper_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "stderr": result.stderr,
            }
        else:
            return {
                "success": False,
                "output": result.stdout,
                "stderr": result.stderr,
                "error": "Python execution failed. See stderr for traceback.",
            }

    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "output": exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else ""),
            "stderr": exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else ""),
            "error": f"Python execution timed out after {timeout_seconds} seconds.",
        }
    except Exception as exc:
        return {
            "success": False,
            "output": "",
            "stderr": "",
            "error": f"Unexpected error during execution: {str(exc)}",
            "traceback": traceback.format_exc(),
        }
    finally:
        # 清理临时文件
        for p in [agent_code_path, wrapper_path]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
