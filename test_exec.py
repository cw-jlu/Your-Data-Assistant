import sys
import os
from pathlib import Path

# Add src to sys.path first
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_agent_baseline.tools.python_exec import execute_python_code

def test():
    root = Path('D:/code/python/kdd/public/input/task_11/context')
    code = "print('hello world')"
    print("Executing python code...")
    result = execute_python_code(root, code, timeout_seconds=10)
    print("Result:", result)

if __name__ == "__main__":
    test()
