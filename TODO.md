# 待办事宜 (TODO)

## 待办实验与调优 (Next Steps)
1. **效果回归测试**：
   - [DONE] 在 `exp/kg+pagerag` 分支通过 JSON 化 Roadmap 显著提升了 baseline 表现。
   - [DONE] 验证 V3.1 (Subprocess + Branching) 在 Easy 任务中的稳定性，成功找回丢失的超时分数。
2. **输出质量调优**：
   - [TODO] 强化 `answer` 工具对多余列的自动裁剪逻辑，减少 λ 惩罚导致的失分。
3. **大数据量优化**：
   - [TODO] 针对 `task_38` 等百万行 CSV 任务，在 Prompt 中引导模型使用 `pd.read_csv(usecols=...)` 提高效率。

## 官方提交前必须修改的内容 (已完成)

1. **[DONE] 修改 `config.py` 中的环境变量加载优先级**：
   - 已完成。`AgentConfig` 现已优先读取 `MODEL_NAME`、`MODEL_API_URL` 等环境变量，完美适配评测系统。
