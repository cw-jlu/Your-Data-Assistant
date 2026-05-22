# KDD Cup 2026 Data Agents — Phase 1 完整报告

**队伍 ID**：team1121  
**报告日期**：2026-05-22  
**最终 A-Board 得分**：0.5114（v3）  
**B-Board 提交版本**：team1121:final（基于 v3 重新标记）

---

## 一、比赛概览

### 1.1 赛题简介

KDD Cup 2026 Data Agents 挑战赛要求参赛者构建一个全自动的数据分析 Agent，能够：
- 遍历 `/input` 下所有任务目录，读取 `task.json` 中的自然语言问题
- 自主探索 `context/` 下的多模态数据源（CSV、SQLite、JSON、Markdown 文档）
- 通过代码生成与执行（Python / SQL）完成数据分析
- 将最终结果写入 `/output/task_<id>/prediction.csv`

### 1.2 评测环境约束

| 约束项 | 规格 |
|:---|:---|
| 模型 | Qwen3.5-35B-A3B（统一部署，禁止自带 LLM） |
| CPU | 16 核 x86-64 |
| 内存 | 64 GB（超限 OOM Kill） |
| GPU | 无 |
| 网络 | 完全断网，仅可访问内部模型服务 |
| A-Board 时限 | 2 小时 / ~60 tasks |
| B-Board 时限 | 12 小时 / ~320 tasks |

### 1.3 评分机制

```
Score = Recall - λ · (Extra Columns / Predicted Columns)

Recall = Matched Columns / Gold Columns
λ = 0.5（官方默认）
```

- 基于**列级内容一致性匹配**，忽略列名和行顺序
- 数值归一化至 2 位小数，字符串大小写敏感
- 总分 = 所有任务得分的平均值

---

## 二、技术方案演进

### 2.1 架构总览

所有方案均基于 **ReAct（Reasoning + Acting）** 框架，核心循环为：

```
Thought → Action → Observation → ... → Answer
```

Agent 在每一步输出一个 JSON 对象，包含 `thought`（推理）、`action`（工具调用）、`action_input`（参数），系统执行工具后返回 `Observation`，循环直至调用 `answer` 工具提交结果。

### 2.2 分支策略

采用 **Git 分支隔离** 的控制变量实验法，每个分支代表一种独立的技术方案或优化策略：

```mermaid
graph LR
    main["main<br/>官方基准微调<br/>A-Board: 0.3298"]
    v3["v3<br/>KG + PageIndex<br/>A-Board: 0.5114 ⭐"]
    v4["v4<br/>v3 同源<br/>A-Board: 0.2982"]
    v5["v5<br/>v3 + Prompt优化<br/>未提交A-Board"]
    LLMWIKI["LLMWIKI<br/>LLM Wiki 模式<br/>A-Board: 0.1658"]
    
    main --> v3
    v3 --> v4
    v3 --> v5
    main --> LLMWIKI
    
    style v3 fill:#2d6a4f,stroke:#1b4332,color:#fff
    style main fill:#264653,stroke:#2a9d8f,color:#fff
    style v4 fill:#6c757d,stroke:#495057,color:#fff
    style v5 fill:#457b9d,stroke:#1d3557,color:#fff
    style LLMWIKI fill:#e76f51,stroke:#e63946,color:#fff
```

---

## 三、各版本详细对比

### 3.1 成绩总览

| 版本 | 分支 | 公开测试集得分 | A-Board 得分 | 核心技术 | 提交状态 |
|:---|:---|:---|:---|:---|:---|
| **v3** | `v3` | 64.00% | **0.5114** ⭐ | Schema KG + PageIndex + Answer Interceptor | ✅ 已提交，最高分 |
| **v1** | `main` | 72.38% | **0.3298** | 官方 Starter Kit 微调 | ✅ 已提交 |
| **v5** | `v5` | 64.00% | **0.3184** | v3 + Prompt 边界检查 + 并发优化 | ✅ 已提交 |
| **v4** | `v4` | — | **0.2982** | v3 同源代码 | ✅ 已提交 |
| **v6** | `LLMWIKI` | **最高** | **0.1658** | LLM Wiki（Karpathy 模式）| ✅ 已提交 |
| exp/* | 实验分支 | 较低 | — | 向量检索 / GraphRAG / Embedding | ❌ 未提交 |

### 3.2 关键发现

> [!IMPORTANT]
> **公开测试集得分与 A-Board 隐藏集得分严重不相关。**
> 
> - v6（LLMWIKI）在公开测试集上得分最高，但 A-Board 仅 0.1658，排名最低
> - v3 在公开测试集上仅 64%，但 A-Board 高达 0.5114
> - main（v1）公开集 72.38%，A-Board 0.3298
> 
> 这表明公开测试集存在过拟合风险，隐藏集的任务分布和难度与公开集差异显著。

### 3.3 各版本技术方案详解

#### v1 / main — 官方基准微调（A-Board: 0.3298）

**核心思路**：在官方 Starter Kit 基础上进行轻量级微调。

- **数据导航**：简单的文件列表扫描，无 Schema 分析
- **工具集**：基础工具（`read_csv`、`execute_python`、`execute_context_sql`）
- **Prompt**：基础 ReAct 指令 + 简单输出格式约束
- **优势**：代码简洁，运行稳定，无额外依赖
- **劣势**：缺乏跨表关联感知，Agent 对数据结构理解有限

#### v3 — Schema KG + PageIndex（A-Board: 0.5114 ⭐ 最高分）

**核心思路**：通过预计算的 Schema 知识图谱和文档索引树，让 Agent 在推理前就掌握数据全貌。

**关键技术栈**：

1. **Schema Knowledge Graph（db_navigator.py）**
   - 自动扫描所有 DB/CSV/JSON 数据源，提取完整的表结构、列名、外键
   - 跨文件同名字段关联（Shared Fields / JOIN Hints）：自动发现 `raceId` 同时出现在 `races.csv` 和 `results.db` 中
   - 注入 `row_count` 让 Agent 感知数据规模
   - 主动扫描并优先列出 `knowledge.md` / `doc/` 目录

2. **PageIndex 文档索引（pageindex_lite.py）**
   - 对 `doc/` 目录下的 Markdown 文档生成带行号的树状结构索引
   - Agent 可通过 `read_doc_lines` 精准读取特定行范围，避免全量加载
   - 适配 `hard` / `extreme` 难度的超长文档场景

3. **Answer Interceptor（answer 工具拦截器）**
   - Agent 首次调用 `answer` 时被拦截，返回 Final Review Checklist
   - 强制二次核查：格式、聚合逻辑、时间参考、知识规则
   - 通过后再次调用 `answer` 才真正提交

4. **历史压缩（History Compression）**
   - 连续失败的历史步骤被折叠，仅保留最后一条
   - 节省 Context Window，避免 Token 溢出

5. **JSON 鲁棒性**
   - Greedy Brace Fix：从截断的 JSON 响应中自动恢复
   - 字面换行符转义：处理 Qwen 输出中的非标准格式

**为什么 v3 得分最高**：
- Schema KG 让 Agent 从第一步就知道"数据在哪、怎么关联"，大幅减少盲目探索
- PageIndex 精准定位文档内容，避免大文件全量读取导致的 Token 浪费和超时
- Answer Interceptor 强制二次审核，减少粗心错误
- 整体方案**增强信息而不增加复杂度**，不引入额外的推理链路

#### v4 — v3 同源（A-Board: 0.2982）

**核心思路**：与 v3 代码完全相同（同一个 commit `485bf80`），但 A-Board 得分显著低于 v3。

**得分差异原因分析**：
- v3 和 v4 指向同一个 Git commit，代码逻辑完全一致
- A-Board 得分差异（0.5114 vs 0.2982）可能源于：
  1. **模型推理的非确定性**：Qwen3.5-35B-A3B 的 `temperature=0` 在 vLLM 的 8 路张量并行下仍存在浮点精度差异
  2. **任务执行顺序效应**：不同提交时的评测队列状态可能影响模型服务的响应
  3. **超时边界效应**：2 小时 A-Board 时限下，个别任务的执行时间波动可能导致部分任务未完成

> [!WARNING]
> v4 的得分表明，即使代码完全相同，不同次评测的得分也可能有 **0.2+** 的波动。这在依赖 LLM 推理的系统中是需要重视的稳定性问题。

#### v5 — v3 + Prompt 优化（A-Board: 0.3184）

**核心思路**：在 v3 基础上，根据失败任务分析进一步优化 Prompt。

**新增改进**（相比 v3）：
1. **Boundary Inequality Checking**：在系统提示词中新增严格的"临界边界与不等式检查"规则
   - 例：遇到"less than 70"必须用 `< 70`，禁止用 `<= 70`
2. **环境自动检测**：`main.py` 自动区分官方评测环境和私有服务器测试
3. **PageIndex 并发控制**：添加 `asyncio.Semaphore` 限制摘要并发数（后因不稳定被 revert）

**得分反而下降的原因**：
- Prompt 变更虽然修复了部分边界案例，但可能**过度约束**了 Agent 的灵活性
- "改进后得分反降"现象在 LLM Agent 中常见：**局部优化可能破坏全局平衡**
- 公开测试集上 v5 和 v3 得分相同（64%），说明改进主要影响边缘案例

#### v6 / LLMWIKI — LLM Wiki 模式（A-Board: 0.1658）

**核心思路**：受 Andrej Karpathy 的 LLM Wiki 思想启发，让 Agent 在执行前先通过 LLM 生成数据的 Wiki 式描述。

**关键技术**：
- **LLM Wiki 预处理**：在任务开始前，调用 LLM 对每个数据源生成结构化的 Wiki 描述
- 基于 `main` 分支构建，补充了 `use_flat_output` 和 `DualLogger` 合规性补丁

**在公开测试集上得分最高的原因**：
- Wiki 描述为 Agent 提供了丰富的语义上下文
- 公开测试集任务相对简单，额外的上下文信息正好帮助 Agent 理解数据

**在 A-Board 上得分最低的原因**：
- LLM Wiki 预处理**消耗大量 Token 和时间**
- A-Board 仅 2 小时时限 + ~60 任务，Wiki 生成的时间开销导致大量任务超时
- Wiki 描述引入了 LLM 的幻觉风险，在复杂任务中可能误导 Agent
- 整体架构复杂度增加，**收益不抵开销**

#### 实验分支（exp/*）— 未提交

| 分支 | 技术 | 未提交原因 |
|:---|:---|:---|
| `exp/pagerag` | 基础 PageRAG（RRF + BM25） | 效果不稳定，精度不如纯 KG |
| `exp/kg+pagerag` | KG + PageRAG v2.2 | 引入 Embedding 后镜像体积暴增，且精度未提升 |
| `exp/graphrag` | LLM 驱动的三元组抽取 | 预处理耗时过长，三元组质量不稳定 |
| `exp/kg+graphrag` | KG + GraphRAG | 融合后复杂度过高，得分反降 |
| `exp/schema-std` | Schema 预处理标准化 | 改进幅度有限，独立贡献不显著 |
| `feat/db-schema-only` | 纯 Schema KG | 已被 v3 完整吸收 |

> [!NOTE]
> **向量检索类方案（PageRAG、GraphRAG）在本赛题中效果均不理想。**
> 
> 核心原因：
> 1. 赛题的数据源以结构化数据（CSV/DB）为主，非结构化文档（doc/）占比较小
> 2. Embedding 模型需要 `sentence-transformers`，引入 PyTorch 和 CUDA 依赖，镜像体积从 ~260MB 暴增到 ~3GB
> 3. 向量检索的精度在小规模文档上不如基于行号的精准定位（PageIndex）
> 4. 额外的检索链路增加了系统复杂度和延迟，在严格时限下弊大于利

---

## 四、核心架构详解（v3 最终方案）

### 4.1 系统 Prompt 结构

```
[A] react_system_prompt.txt      ← 核心行为准则（6 条操作原则 + 工具规则 + 输出格式）
[B] Data Roadmap                 ← 动态数据地图（Schema KG + PageIndex，按难度生成）
[C] Tool Descriptions            ← 工具说明（registry.py 自动生成）
[D] response_examples.txt        ← 3 个输出格式示例（SQL / Python / 提交）
[E] 输出格式强制声明              ← 硬编码在 build_system_prompt() 末尾
```

### 4.2 Data Roadmap 生成策略

| 难度 | 策略 | 内容 |
|:---|:---|:---|
| `easy` | 精简模式 | 仅文件列表（`list_context_tree` depth=2），避免 Token 过载 |
| `medium` / `hard` / `extreme` | 全量模式 | Schema KG + PageIndex 文档树 |

### 4.3 工具集

| 工具 | 用途 |
|:---|:---|
| `list_context` | 列出 context 目录结构 |
| `read_csv` | 读取 CSV 预览 |
| `read_json` | 读取 JSON 预览 |
| `read_doc` | 按页读取文本文档 |
| `read_doc_lines` | 精准行读取，配合 PageIndex 行号 |
| `get_doc_structure` | 获取 Markdown 文档的树状结构 |
| `inspect_sqlite_schema` | 查看 DB 文件表结构 |
| `execute_context_sql` | 对 DB 执行只读 SQL |
| `execute_python` | 执行 Python 代码（60s 超时） |
| `answer` | 提交最终答案（带拦截器二次审核） |

### 4.4 防护机制

| 机制 | 描述 |
|:---|:---|
| Answer Interceptor | 首次提交被拦截，强制二次审核 |
| 死循环检测 | 连续 3 次相同 Action 触发警告 |
| 熔断保护 | 连续 6 次错误终止任务 |
| 反思提醒 | 连续 3 次错误注入 `[SYSTEM WARNING]` |
| 历史压缩 | 折叠连续失败步骤，节省 Token |
| JSON 自动修复 | Greedy Brace Fix + 换行符转义 |
| Python 超时 | 单脚本 60 秒超时保护 |
| 任务超时 | 单任务 1200 秒（20 分钟）超时 |

---

## 五、关键教训与反思

### 5.1 ✅ 成功经验

1. **"信息前置"优于"按需检索"**
   - Schema KG 在推理前一次性提供数据全貌，比让 Agent 自行探索高效得多
   - PageIndex 的带行号树状结构比全量 Embedding 检索更精准、更轻量

2. **"简单稳定"优于"复杂精巧"**
   - v3（KG + PageIndex）代码简洁、运行稳定，远胜复杂的 RAG 方案
   - 向量检索、图知识抽取等复杂组件在本赛题中 ROI 为负

3. **Answer Interceptor 显著提升精度**
   - 强制二次审核有效减少了格式错误和粗心计算错误

4. **镜像体积控制**
   - 移除 `sentence-transformers` 后，镜像从 ~3GB 降至 ~260MB
   - 构建时间从 20 分钟降至 1 分钟

### 5.2 ❌ 失败教训

1. **不要过度依赖公开测试集**
   - 公开集与隐藏集的分布差异极大，v6 的"公开最高、A-Board 最低"是惨痛教训

2. **Prompt 优化有双刃剑效应**
   - v5 的 Prompt 改进在公开集不变但 A-Board 下降，说明过度约束反而限制 Agent 灵活性

3. **LLM 推理的非确定性不可忽视**
   - v3 和 v4 代码相同但得分差 0.2+，提醒我们 LLM-based 系统的方差很大

4. **向量检索在结构化数据场景效果有限**
   - 本赛题以 CSV/DB 为主，Embedding 检索的优势场景（大规模非结构化文档）出现频率低

### 5.3 🔮 如果重来会怎么做

1. **更早放弃向量检索方向**，集中精力优化 Schema KG 和工具链
2. **在 A-Board 上多次验证同一版本**，评估得分方差
3. **增加对 A-Board 时限的敏感性测试**（2 小时 / 60 tasks ≈ 2 分钟/任务）
4. **对 Prompt 变更做 A/B 测试**，而非仅依赖公开集验证

---

## 六、B-Board 提交

### 6.1 最终决策

基于 A-Board 得分对比，选择 **v3** 作为 B-Board 最终提交版本。

**操作流程**：
1. 从现有 `team1121_v3.tar.gz` 加载 Docker 镜像
2. `docker tag team1121:v3 team1121:final` 重新标记
3. `docker save` + `gzip` 导出为 `team1121_final.tar.gz`（267.51 MB）
4. 镜像 SHA256 验证：`team1121:final` 与 `team1121:v3` 完全一致

### 6.2 验证结果

| 检查项 | 状态 |
|:---|:---|
| 镜像 ID 一致性 | ✅ SHA256 完全相同 |
| 容器正常启动 | ✅ exit code = 0 |
| 单任务测试（task_25） | ✅ 1/1 succeeded |
| 输出格式 | ✅ `/output/task_25/prediction.csv` 标准 CSV |
| 日志写入 | ✅ `/logs/runtime.log` 正常 |
| 文件大小 | ✅ 267.51 MB < 10 GB |

### 6.3 提交信息

- **镜像名**：`team1121:final`
- **压缩包**：`team1121_final.tar.gz`
- **邮件主题**：`[KDDCup2026 Data Agents] B-board Final Submission - team1121`
- **截止时间**：2026-05-23 19:59 北京时间

---

## 七、提交历史

| 序号 | 版本 | 分支 | 提交时间 | A-Board 得分 | 备注 |
|:---|:---|:---|:---|:---|:---|
| 1 | v1 | main | Phase 1 早期 | 0.3298 | 官方基准微调 |
| 2 | v3 | v3 | Phase 1 中期 | **0.5114** | Schema KG + PageIndex |
| 3 | v4 | v4 | Phase 1 中期 | 0.2982 | v3 同源，得分波动 |
| 4 | v5 | v5 | Phase 1 后期 | 0.3184 | Prompt 优化 |
| 5 | v6 | LLMWIKI | Phase 1 后期 | 0.1658 | LLM Wiki 模式 |
| **Final** | **final** | **v3** | **2026-05-22** | — | **B-Board 提交** |

---

*KDD Cup 2026 Data Agents Challenge — Team 1121 Phase 1 Report*
