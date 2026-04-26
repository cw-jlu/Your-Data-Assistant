# 待办事宜 (TODO)
## 需要完善的地方
1. **完善 starter-kit\src\data_agent_baseline\tools\filesystem.py**：
   - csv，json等预览函数需要改：实现读取文件的前几行，并返回一个包含数据预览的字典。
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
