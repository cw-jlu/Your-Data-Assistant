# LangGraph → HEAD 升级对比报告

> 统计口径：`LangGraph` 分支为基准，`HEAD` 为当前最新提交。
> 共计 **109 个 commits**领先，**86 个文件**变更，**+1,006,059 行 / -13,349 行**。

---

## 一、整体架构演进

LangGraph 分支的核心架构是一个以 **Data Understanding Agent**（数据理解智能体）+ **Probe Engine**（探针引擎）为主体的单智能体系统，通过 inspectors（检查器）模块进行数据分析。

HEAD 分支在此基础上完成了 **从单智能体到多智能体协作** 的架构升级。新增了 2 个专职智能体节点，形成"主智能体推理 → 过程校验 → 答案校验"的流水线。

```
LangGraph:  [Data Understanding Agent] → 答案
HEAD:       [Data Understanding Agent] → [Process Validator] → [Answer Validator] → 答案
```

---

## 二、新增智能体模块

### 2.1 Process Validator（过程校验 Agent）

**相关 commits:** `3208455`, `285cdd3`, `1773b13`, `a754f99`

**核心变化:**

- 新增专门的"过程验证专员"，在答案生成前校验推理过程
- 添加详细的验证规则和说明
- 支持最值多答案并列规则（如并列第一的情况）
- 收紧提示词约束以减少误判

**新增文件:**

- `src/data_agent_baseline/agents/process_validator.py`
- `tests/test_process_validator.py`

### 2.2 Answer Validator（答案校验器）

**相关 commits:** `83798a6`, `6a598db`, `a99e8ef`, `eb6183f`, `ef4915f`

**核心变化:**

- 新增答案校验节点，对最终输出进行格式和内容检查
- 增强列范围检查，避免多余列输出
- 支持验证历史记录和缓存机制
- 修复百分比格式检查和答案缺失 bug

**新增文件:**

- `src/data_agent_baseline/agents/answer_validator.py`
- `tests/test_answer_validator.py`

> **注：** Ambiguity Analyzer（歧义分析器，`ambiguity_analyzer.py`）代码已提交至仓库，但通过配置开关 `enable_ambiguity_analysis` 控制，**当前默认为关闭状态**（`False`），不参与实际运行流程。

---

---

## 三、Catalog 与 Schema 管理重构

### 3.1 轻量化 Catalog 注入

**相关 commits:** `12857a4`, `0ea8760`, `38bc592`

- 从 LangGraph 的"全量 catalog 注入"改为"轻量化注入 + 按需查询"
- 使用本地 tokenizer 精确计算 token 数，替代字符估算
- 单个值过长时自动截断并标注 `[被截断]`

### 3.2 Schema 查询工具升级

**相关 commits:** `ebaf743`, `17f3162`, `0cfe05d`, `42d3767`

- `lookup_schema` 改为支持批量字段查询
- 新增 `prob` 表工具，支持批量 SQL 探查
- 移除 `lookup_schema` 工具，改为直接注入完整 catalog
- 新增 `search_doc` 工具，支持正则/关键词搜索文档

### 3.3 Catalog Semantic Enrichment 演进

**相关 commits:** `45fd1f1`（新增）→ `282c8b5`（移除）

- 曾新增 Catalog Semantic Enrichment 节点，通过 LLM 为字段生成业务描述
- 最终决定删除该功能及相关配置（`enable_semantic_enrichment`），简化流水线

---

## 四、工具基础设施升级

### 4.1 探针引擎迁移

**变更:** `src/data_agent_baseline/inspectors/probe_engine.py` → `src/data_agent_baseline/tools/probe_engine.py`

探针引擎从 inspectors 模块迁移到 tools 模块，语义上更准确地反映了它作为工具而非检查器的角色。

### 4.2 新增工具

| 工具                   | 功能                          | 相关 commit                           |
| ---------------------- | ----------------------------- | ------------------------------------- |
| `search_doc`         | 正则/关键词搜索文档，支持分页 | `0c078b3`, `d531660`, `42c6abe` |
| `read_doc`           | 结构化读取 Markdown 文档      | `80ba647`, `5678a3d`              |
| `lookup_doc_outline` | 查看文档目录结构              | `5678a3d`                           |
| `prob_table`         | 批量 SQL 探查表数据           | `a90b42f`                           |

### 4.3 工具输出上限提升

**相关 commits:** `c164c04`, `387ccb4`

- 工具最大输出字符数从默认值提升至 **40000 字符**
- 新增 Bash 和 Read 命令支持

### 4.4 工具注册重构

**新增文件:** `tests/test_tool_registry.py`（568 行测试）

工具注册机制进行了集中化重构，增强了可测试性。

---

## 五、移除的模块与功能

### 5.1 废弃的 Inspectors 模块

以下文件从代码库中完全移除：

| 文件                             | 原始功能        | 移除原因                     |
| -------------------------------- | --------------- | ---------------------------- |
| `inspectors/perception.py`     | 实体/过滤词感知 | 架构简化，职责由主智能体承接 |
| `inspectors/exchange.py`       | 数据交换        | 不再需要                     |
| `inspectors/handoff.py`        | 任务交接        | 架构简化                     |
| `inspectors/semantic_index.py` | 语义索引        | 被轻量化 catalog 替代        |
| `inspectors/semantic_query.py` | 语义查询        | 被 search_doc 工具替代       |
| `inspectors/prompts.py`        | 检查器提示词    | 职责分散到各子模块           |

### 5.2 废弃的配置开关

| 配置项                            | 相关 commit |
| --------------------------------- | ----------- |
| `enable_semantic_enrichment`    | `282c8b5` |
| `enable_problem_grounding`      | `1b0ee7b` |
| `enable_global_exploration_llm` | `1b0ee7b` |
| `catalog_sample_rows`           | `bd801ce` |
| `max_join_hops`                 | `1631582` |
| `rules/hybrid` 双模式分支       | `0a2e201` |

### 5.3 废弃的处理节点

- `build_document_context_node` — 架构简化，不再需要前置文档上下文节点（`a468ea2`）

---

## 六、性能与可靠性改进

### 6.1 模型调用优化

- **429 避退重试**：遇到 API 限流自动退避重试（`484c48e`）
- **请求超时策略调整**：先移除超时（`0c078b3`），后增加超时重试提示（`b282b1b`）
- **重试机制修复**：修复 retry 逻辑 bug（`ea65a4b`）
- **模型温度与样本预算调优**（`bef4859`）

### 6.2 Token 管理

**新增文件:** `src/data_agent_baseline/token_utils.py`

- **内置本地 Qwen3.5-35B-A3B tokenizer 缓存**，避免每次调用远程 API 计 token
- 用于精确控制 catalog 注入量

### 6.3 可观测性

- **节点耗时追踪**：每个节点的执行时间记录到 trace（`e59fbb5`, `81c2f25`）
- **工具调用统计**：评分阶段统计 trace 工具调用次数（`f13643f`）
- **全局总时长统计**：在评测报告中增加运行总时长（`69dfee4`）

### 6.4 截断与工具修复

- 修复 LangGraph 工具截断功能（原来未生效）（`247f998`）
- 修复探针单查询数据重复返回和标题静默截断（`627cd3a`）
- 清理伪工具调用块（`4cf3040`）

---

## 七、基础设施与工程化

### 7.1 Docker 化

**新增文件:** `Dockerfile`, `.dockerignore`

项目已支持容器化部署。

### 7.2 UV 包管理器

项目从传统 pip 迁移到 `uv` 包管理器，锁文件从旧格式更新为 `uv.lock`（4227 行变更）。

### 7.3 CLI 工具

**新增文件:** `src/data_agent_baseline/cli.py`（138 行）

新增命令行接口，支持：

- 批量运行单个题目并计算正确率（`1fb15e5`）
- 跳过已完成任务（`821e1fe`）
- 任务中断处理（`f677b2f`）

### 7.4 本地 Tokenizer 资产

新增 Qwen3.5-35B-A3B 完整 tokenizer 资源文件，用于本地精确 token 计数：

- `tokenizer.json`（~496K 行）
- `vocab.json`（~248K 行）
- `merges.txt`（~248K 行）

### 7.5 测试体系扩充

| 新增测试文件                        | 行数 |
| ----------------------------------- | ---- |
| `tests/test_tool_registry.py`     | 568  |
| `tests/test_model_retry.py`       | 98   |
| `tests/test_process_validator.py` | 85   |
| `tests/test_answer_validator.py`  | 36   |

---

## 八、文档变更

### 新增文档

- `docs/agent-flow-easy.md` — Easy 题目智能体执行流程
- `docs/config-guide.md` — 配置指南
- `docs/handoff_optimization_proposal.md` — Handoff 优化提案
- `docs/relationship-inference-design.md` — 关系推断设计
- `docs/two-stage-data-understanding-architecture.md` — 两阶段数据理解架构

### 移除的过期文档

- `data_agent_software_design_doc.md`
- `docs/agent_error_analysis_20260412.md`
- `docs/eval_protocol.md`
- `docs/inspectors_runtime_flow.md`
- `docs/task_11_baseline_execution_flow.md`
- 及若干过期分析报告

---

## 九、总结

### 升级关键词

| 维度         | LangGraph     | HEAD                                |
| ------------ | ------------- | ----------------------------------- |
| 智能体数量   | 1（主智能体） | 3（主智能体 + 过程校验 + 答案校验） |
| Catalog 策略 | 全量注入      | 轻量化注入 + 按需查询               |

| Token 管理 | 字符估算 | 本地精确 tokenizer |
| 工具数量 | ~6 | ~10（新增 search_doc, read_doc 等） |
| 校验机制 | 无 | 过程校验 + 答案校验双重保障 |
| 可观测性 | 基础 | 节点级耗时追踪 + 工具调用统计 |
| 部署方式 | 本地运行 | 本地运行 + Docker |
| 包管理 | pip | uv |

### 核心设计思想变化

1. **从"一个模型解决所有"到"分工协作"** — 将过程校验、答案校验等职责拆分为独立的专用节点，每个节点聚焦单一任务
2. **从"全量注入"到"按需查询"** — Catalog 不再全量塞入 prompt，而是精简注入 + 工具按需查询，减少 token 浪费
3. **从"只管生成"到"校验闭环"** — 新增过程和答案两级校验，减少幻觉和格式错误
4. **从"实验代码"到"工程化代码"** — 移除大量实验性分支、过期文档和无效配置，保留经过验证的路径
