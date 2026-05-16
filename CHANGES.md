# KDD Cup 2026 Data Agents - 技术改进与坑点总结

本项目已基于官方 Starter Kit 完成了针对评测环境的适配。以下是核心改动说明及开发过程中的经验总结。

---

## 1. 配置文件 (YAML) 参数说明
**⚠️ 结论：YAML 的参数结构（Schema）未发生任何变动。**

所有底层逻辑改进均在 Python 代码内完成，以保持与原始代码包的 100% 配置兼容。
- **参数名称与数量**：未增加、未减少、未重命名任何 YAML 配置项（即 `dataset`、`agent`、`run` 及其子项完全保留原始定义）。
- **兼容性**：您可以直接复用参赛包提供的 `react_baseline.example.yaml`。
- **配置建议**：本地开发时，建议在 YAML 中**注释掉** `run_id` 这一行，利用系统自动生成的带时间戳 ID 来避免目录冲突。

---

## 2. 核心逻辑改进对比 (Modification Comparison)

| 模块 | 原始逻辑 | 改进后逻辑 | 改进原因 |
| :--- | :--- | :--- | :--- |
| **进程间通信 (IPC)** | 通过 `multiprocessing.Queue` (系统管道) 传输任务结果。 | 改用 **临时文件 (Temporary File)** 模式进行跨进程数据中转。 | **根本消除死锁**：当日志（Trace）超过 OS 管道缓冲区（通常 64KB）时，父子进程会发生永久性相互等待。 |
| **错误捕获** | `error: str(exc)` 仅记录异常的字符串描述。 | `error: traceback.format_exc()` 记录完整的错误堆栈。 | **零描述异常排查**：解决大量 Python 内置异常由于描述为空（Empty String）导致无法定位报错位置的问题。 |
| **路径自动适配** | 需手动修改 YAML 中的 `root_path` 切换本地与评测路径。 | 新增 `main.py` 入口，自动探测 `Path("/input")` 路径是否存在。 | **评测自动化**：确保代码上传到官方 Docker 容器后能自动识别 `/input` 挂载点，无需手动干预。 |

---

## 3. 常见坑点汇总 (Pitfalls Summary)

| 坑点现象 | 根本原因 | 解决方案 |
| :--- | :--- | :--- |
| **任务 0.7s 即“成功”结束，但 `failure_reason` 为空** | **管道缓冲区死锁**。Agent 产生的推理 Trace 数据太大，导致子进程被堵塞在写管道这一步无法退出。 | 已修改 `runner.py`，改用全磁盘模式中转结果，不再受缓冲区容量限制。 |
| **`run-task` 命令直接退出，终端无任何输出信息** | **目录冲突**。YAML 中配置了固定 `run_id` 且对应目录已存在。`FileExistsError` 被 Cli 库静默拦截。 | 在 YAML 中注释掉固定 `run_id`，或手动删除旧的 artifacts 目录。 |
| **将 `max_workers` 改为 1 依然无法解决上述崩溃** | `max_workers` 控制的是外层多线程，而**死锁发生在任务内部的子进程通信层**。 | 必须修复 IPC 通信机制（方案 2），调节并发数无法触及根因。 |
| **`[SUBPROCESS] Task completed` 出现后系统永久卡死** | 死锁发生的具体位置：子进程完成核心推理正要向主进程返回数据包。 | 现在的逻辑已将“写回数据”与“进程 join”的操作完全解耦，彻底修复。 |

---

## 4. 后续开发建议
- **并发性能**：将 `max_workers` 设为 `4` 或 `8` 可充分利用 CPU 核心。新的临时文件方案是并发安全的。
- **本地调试**：可以直接运行 `uv run py main.py` 进行全量测试。
- **提交准备**：在导出镜像前，务必检查 `TODO.md`，将 `config.py` 中的模型配置切回环境变量优先模式，确保能读取到评测系统的 Key。

---

## 5. 模块化导航架构 V2 (Modular Navigation V2)

本项目在 V1 基础上进行了重构，解耦了导航逻辑，支持多种检索策略的“排列组合”。

| 模块 | V2 核心改进 | 解决的痛点 |
| :--- | :--- | :--- |
| **Schema KG** | 引入 `PRAGMA FK` 提取真实外键；增加行数统计与样本数据预览；解析 `knowledge.md` 业务定义。 | 解决 Agent 无法识别表关联和数据特征的问题。 |
| **PageRAG** | 升级为 **两层级联检索 v2.1**；采用 **标题/摘要向量化 (Title-only)**；支持 `doc/` 目录精准激活。 | **极致信噪比与效率**：解决正文向量化导致的索引膨胀和干扰问题。 |
| **GraphRAG** | 标准化 12 种关系类型；引入**置信度过滤**与**实体去重**；支持 LLM 异步预提取。 | 解决三元组冗余、语义模糊及无效信息干扰问题。 |
| **配置系统** | 新增全局参数 `rag_top_k`。 | 解决检索深度难以统一调节的问题。 |
| **环境兼容** | `run_failed.py` 强制开启 UTF-8 模式。 | 彻底解决 Windows 环境下的 `GBK` 编码崩溃。 |

---

## 6. LLM Wiki 知识库系统 (LLM Wiki Knowledge Base)

基于 Andrej Karpathy (2026-04) 提出的 LLM Wiki 设计模式实现。核心思想：Agent 不再对每个任务从零开始推理，而是**增量构建并维护一个持久化的、相互链接的 Markdown 知识维基**。

### 架构设计

| 层 | 说明 |
| :--- | :--- |
| **原始数据 (Raw Sources)** | 任务 context/ 目录中的 CSV、SQLite、JSON、文档。不可变。 |
| **Wiki (Markdown Pages)** | Agent 自动维护的结构化知识页面。包含实体页、概念页、任务摘要页、综合分析页。通过 `[[wikilinks]]` 互相链接。 |
| **Schema (Index + Log)** | `index.md` 按类别编目所有页面；`log.md` 按时间顺序记录所有操作。 |

### 新增模块

| 文件 | 功能 |
| :--- | :--- |
| `src/data_agent_baseline/wiki/engine.py` | Wiki 核心引擎：页面 CRUD、索引管理、wikilink 解析 |
| `src/data_agent_baseline/wiki/retrieval.py` | TF-IDF 余弦相似度检索：为新任务找到相关已有知识 |
| `src/data_agent_baseline/wiki/ingest.py` | 知识提取器：从已完成任务中提取 schema、SQL 模式、领域概念 |
| `src/data_agent_baseline/wiki/registry.py` | Wiki 工具注册：wiki_search、wiki_get_page、wiki_list_index |

### 工作流

```
任务开始 → Wiki Retrieval (TF-IDF 搜索相关知识)
         → 注入 wiki_context 到系统提示词
         → Agent 利用已有知识 + 工具解决任务
         → 任务完成 → Wiki Ingest (提取新知识)
         → 创建/更新实体页、概念页、任务摘要页
         → 更新 index.md 和 log.md
```

### 配置

```yaml
wiki:
  enabled: true          # 是否启用 LLM Wiki
  wiki_root: wiki        # Wiki 目录路径
  retrieval_top_k: 5     # 检索返回的最大页面数
  auto_ingest: true      # 任务完成后自动提取知识
```

### 合规性

- 不使用额外 LLM：知识提取基于规则（schema 解析、SQL 模式匹配）
- 不访问外部网络：纯本地 TF-IDF 检索，无 embedding API 调用
- 符合评测规则：wiki 工具作为辅助工具注册，不替代主推理模型
