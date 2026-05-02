# 待办事宜 (TODO)

## 待办实验与调优 (Next Steps)
1. **效果回归测试**：
   - [DONE] 在 `exp/kg+pagerag` 分支通过 JSON 化 Roadmap 显著提升了 baseline 表现。
   - [TODO] 验证 V2.5 优化在 `task_145` 到 `task_420` 这一批长任务中的稳定性。
2. **关系提取准确度检查**：
   - [TODO] 随机抽样 10 个任务，检查 `implicit_logic_joins` 是否准确覆盖了 `knowledge.md` 中的所有关键 Join 路径。

## 官方提交前必须修改的内容 (已完成)

1. **[DONE] 修改 `config.py` 中的环境变量加载优先级**：
   - 已完成。`AgentConfig` 现已优先读取 `MODEL_NAME`、`MODEL_API_URL` 等环境变量，完美适配评测系统。
