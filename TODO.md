# 待办事宜 (TODO)

## 待办实验与调优 (Next Steps)
1. **模块化参数调优**：
   - 针对 `exp/kg+pagerag` 分支，对比 `rag_top_k` 为 3, 5, 7 时的 Score 变化。
   - 调整 `db_navigator` 中的样本数据量，观察对 Context Window 的影响。
2. **效果对比实验**：
   - 依次在所有分支运行 `run_failed.py`，记录成功率和平均得分，更新实验记录表。
3. **GraphRAG 置信度阈值**：
   - 测试将 `min_confidence` 从 0.5 提升至 0.7 对减少无效推理的帮助。

## 官方提交前必须修改的内容

1. **修改 `config.py` 中的环境变量加载优先级**：
   - 当前在 `AgentConfig` 生成时，我们使用的仍是直接从 `yaml` 文件读取的硬编码策略，方便本地快速测试。
   - **提交评测系统前必须改回以下逻辑：**
     官方评测时会通过注入环境变量的方式来接管所有的模型调用配置，若不修改会导致线上拉取不到模型环境并评测失败 0 分。
     
     请在构建 Docker 镜像前，将 `starter-kit/src/data_agent_baseline/config.py` 中的 `AgentConfig` 构造逻辑替换如下：
     
     ```python
     import os
     
     agent_config = AgentConfig(
         model=os.environ.get("MODEL_NAME", str(agent_payload.get("model", agent_defaults.model))),
         api_base=os.environ.get("MODEL_API_URL", str(agent_payload.get("api_base", agent_defaults.api_base))),
         api_key=os.environ.get("MODEL_API_KEY", str(agent_payload.get("api_key", agent_defaults.api_key))),
         max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
         temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
     )
     ```
