# Implementation Plan — 可插拔语义导航架构 (Plug-and-Play Semantic Navigation)

## 1. 目标
构建一个模块化的导航层，解决 17 个失败任务中暴露的语义盲区、大文件懒政和转录错误问题。系统设计必须支持在不同复杂度的方案（GraphRAG, KG, Prompt Map）之间快速切换。

## 2. 方案对比与决策

| 方案模式 | 解决的问题 | 复杂度 | 灵活性 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt Map (Minimal)** | 消除文件盲区，提供基本表头。 | ★☆☆ | 高 | 适合结构化数据简单的任务。 |
| **Knowledge Graph (KG)** | 明确跨表 Join 路径，对齐业务术语。 | ★★☆ | 中 | 适合多表关联及有 `knowledge.md` 的任务。 |
| **GraphRAG (Advanced)** | 叙事文档解析（如致癌分子），实体属性检索。 | ★★★ | 高 | 适合长文档、非结构化知识点任务。 |

**策略**：先实现 **Navigator 抽象接口**，然后默认提供 **GraphRAG** 实现，若效果不佳可快速降级。

## 3. 最小化修改路线图

### 第一阶段：基础设施与接口定义 [NEW]
- **[NEW] `src/data_agent_baseline/agents/navigator/base.py`**:
    - 定义 `get_initial_context()`: 返回注入到 System Prompt 的初始知识。
    - 定义 `query(text: str)`: 响应 Agent 的运行时语义查询。
- **[NEW] `src/data_agent_baseline/agents/navigator/factory.py`**:
    - 根据配置文件开关实例化具体的 Navigator 模式。

### 第二阶段：具体模块实现 [MODULAR]
- **[NEW] `src/data_agent_baseline/agents/navigator/graph_rag.py`**:
    - **逻辑**：对 MD/TXT 进行 Entity-Relation 抽取（Triples），对 DB/CSV 进行 Schema 扫描，融合成内存图。
- **[NEW] `src/data_agent_baseline/agents/navigator/schema_graph.py`**:
    - **逻辑**：仅处理确定性的 Schema 关联，生成 JSON 格式的节点/边映射。
- **[NEW] `src/data_agent_baseline/agents/navigator/prompt_map.py`**:
    - **逻辑**：简单的文本拼接地图（即之前讨论的 SchemaNavigator）。

### 第三阶段：Agent 主循环集成 [MODIFY]
- **[MODIFY] `src/data_agent_baseline/agents/react.py`**:
    - 在 `run()` 方法启动循环前，初始化 Navigator 并调用其获取初始上下文。
    - 注入到 System Prompt 消息列表。
- **[MODIFY] `src/data_agent_baseline/agents/prompt.py`**:
    - 调整 `build_system_prompt` 以支持 Roadmap 内容的注入。

### 第四阶段：配置与规则补丁 [CONFIG]
- **[MODIFY] `configs/react_baseline.local.yaml`**:
    - 增加 `agent.navigator_type: "graph_rag" # options: "graph_rag", "schema_graph", "prompt_map"`。
- **[MODIFY] `react_system_prompt.txt`**:
    - 追加针对 17 类死穴的 7 条红线指令。

## 4. 验证计划 (Verification)
1. **模块替换测试**：在 YAML 中切换三个模式，确认 `agent.log` 中注入的内容随之变化。
2. **专项测试**：针对 `task_396` (GraphRAG 场景) 和 `task_173` (Schema 场景) 跑单项测试。
3. **全量对比**：记录三种模式下的 Benchmark 评分。

## 5. CHANGES.md 规范
- 所有接口定义、新文件创建和配置变更需实时同步更新至 `starter-kit\CHANGES.md`。
