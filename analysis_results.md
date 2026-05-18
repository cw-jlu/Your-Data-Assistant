# 🌟 KDD Cup 数据智能体：18个失败任务像素级深度诊断与优化讨论报告

我们对最新评测运行（运行ID: `20260518T123626Z`，最终得分：**`64.00%`**）中所有失败（Score=0.0）的任务进行了地毯式的排查和全链路追踪。以下是针对这 18 个失败任务的详细诊断报告，深度分析了它们的失败原因（Root Causes）并提出了明确的改进与规避方案。

---

## 📊 18 个失败任务像素级对齐详情

| 任务ID | 任务提问 (Question) | 标准答案 (Gold Shape & Sample) | 智能体实际预测 (Pred Shape & Sample) | 诊断结论 (Root Cause) |
| :--- | :--- | :--- | :--- | :--- |
| **`task_11`** | 查询严重血栓（severe degree of thrombosis）患者的 ID、性别和诊断疾病。 | **3行**<br>· `ID`: `['163109', '2803470', '4395720']`<br>· `Diagnosis`: `['SLE', 'SLE', 'SLE']` | **3行**<br>· `ID`: `['14872', '1567380', '3192610']`<br>· `Diagnosis`: `['MCTD', 'SLE', 'APS']` | 临界值/字段判定逻辑出现偏差。智能体未完全与 `knowledge.md` 中特定表对严重程度的定义对齐。 |
| **`task_25`** | 找出成本最低的活动（lowest cost event）。 | **1行**<br>· `event_name`: `['November Speaker']` | **3行**<br>· `event_name`: `['Officers meeting - November', ...]` | 排序/聚合深度逻辑偏差。智能体计算的是“总预算最低的活动”，而 Gold 期待的是“单笔开销最低的活动（LIMIT 1）”。 |
| **`task_75`** | 查询第19场比赛第二阶段（Q2）最佳单圈车手的姓氏。 | **1行**<br>· `surname`: `['Fisichella']` | **1行**<br>· `surname`: `['Räikkönen']` | 字符串格式排序陷阱。Q2时间（如 `"1:12.345"`）在 SQLite 中作为 raw 字符串时未进行秒数转换或过滤 `\N` 导致排序错乱。 |
| **`task_80`** | 查询在排位赛No.903的Q3中跑出 `0:01:54` 的车手号码。 | **2行**<br>· `number`: `['3', '5']` | **1行**<br>· `number`: `['44']` | 时间正则/格式匹配遗漏。未能将 `0:01:54` 模糊泛化到不同的秒数/毫秒表示导致数据少漏。 |
| **`task_89`** | 2008年中国大奖赛获得第二名的车手的完赛时间。 | **1行**<br>· `time`: `['+16.445']` | **1行**<br>· `time`: `['+14.925']` | 排序/排名判定逻辑错误。未能准确抓取到正确的第二名车手。 |
| **`task_145`** | 超过10个会员参加的活动中，有多少个是会议？ | **4行 (特殊结构)**<br>· `COUNT(...)`: `['1', '1', '1', '1']` | **1行**<br>· `count`: `['4']` | **Gold 标准答案设计缺陷（聚合漏加）**。自然答案是聚合整数 `4`，但 Gold 标准 SQL 中漏写了聚合，带着多余的 `GROUP BY` 导出了 4 个 `1`。 |
| **`task_163`** | 识别 October Meeting 被批准的费用类型（expense type）和总额。 | **1行 (特殊结构)**<br>· `type`: `['Meeting']`<br>· `SUM(cost)`: `['175.39']` | **3行**<br>· `expense_type`: `['Posters', 'Pizza', ...]`<br>· `total_value`: `['54.25', ...]` | **Gold 标准答案概念混淆**。问题问的是“费用类型”，智能体输出了 Posters/Pizza 等；而 Gold 的 SQL 选的却是“活动类型（event_type）”，从而合并计算。 |
| **`task_169`** | 中小企业（SME）客户在2013年的平均月消费是多少？ | **1行**<br>· `AVG/12`: `['459.96']` | **1行**<br>· `average_monthly_consumption`: `['82027220.30']` | 数据归一化/除数错误。智能体求了整体大聚合，未能按客户或月份正确分解月均，算成了天文数字。 |
| **`task_173`** | 列出 2013 年 6 月发生交易的加油站所属国家。 | **2行**<br>· `Country`: `['CZE', 'SVK']` | **0行**<br>· `Country`: `[]` | 日期格式过滤失效。June, 2013 字符串在 SQLite 中可能以 `13.06.2013` 等非标准形式存储，智能体直接用标准 `LIKE '2013-06%'` 过滤导致空结果。 |
| **`task_180`** | 购买产品5单价超29元的所有用户，在2012年8月的消费状态。 | **10行**<br>· `Consumption`: `['1903.2', '88265.39', ...]` | **9行**<br>· `CustomerID`: `['5443', ...]`<br>· `Consumption`: `['88265.39', ...]` | 临界值精度/时区遗漏。在单价“大于29”或时间跨度上漏掉了恰好落在边界上的 1 个用户。 |
| **`task_199`** | 在 Riverside 地区且数学 SAT 均分大于 400 的学校及类型。 | **6行**<br>· `sname`: `['Arlington High', ...]` | **60行**<br>· `School Name`: `['River Springs...', ...]` | 空间过滤条件缺失。智能体漏掉了“Riverside 地区”的严格行政区划过滤，导出了全县所有合格学校。 |
| **`task_200`** | 计算含有磷（P）或溴（Br）的三键分子的原子总数。 | **1行**<br>· `COUNT`: `['1']` | **1行**<br>· `total_atoms`: `['4']` | 分子化学逻辑理解偏差。智能体在图结构解析中把分子数 and 原子数搞混，算大了结果。 |
| **`task_344`** | 白细胞正常但纤维蛋白原异常的男性患者人数。 | **1行**<br>· `COUNT`: `['4']` | **1行**<br>· `count`: `['2']` | 临界值边界不一致。正常/异常区间白细胞范围在 `knowledge.md` 里的上下界使用有偏差。 |
| **`task_352`** | Yearly Kickoff 广告预算是 October Meeting 的几倍？ | **1行**<br>· `Ratio`: `['2.73']` | **0行**<br>· `count`: `[]` | 数据流截断/除零错误。智能体未能在 context 中正确解析出 October Meeting 的预算，导致分母为 0 或空集。 |
| **`task_379`** | 计算致癌分子中第 4 个原子的毒理学元素分布。 | **7行**<br>· `element`: `['c', 'br', 'cl', 's', 'o', ...]` | **6行**<br>· `element`: `['c', 'o', 'cl', 'br', 'n', ...]` | 索引越界与偏移量（Off-by-one）误差。智能体对“第 4 个原子”的提取在 Python 0-based 索引与 1-based 索引中偏离，误选了 `n` 漏掉了 `s`。 |
| **`task_396`** | 身高 150-180 之间的英雄中漫威漫画出版的比例。 | **1行**<br>· `Ratio`: `['54.84']` | **1行**<br>· `percentage`: `['50.00']` | 文本解析不全。智能体读取 `superhero.md` 长文本时漏掉了开头 Part IV 处的英雄，手动硬编码导致分母算错。 |
| **`task_408`** | 2008年澳大利亚站，冠军车手比最后一名快了百分之多少？ | **1行**<br>· `Ratio`: `['0.32']` | **1行**<br>· `percentage_faster`: `['0.52']` | 完赛状态筛选遗漏。智能体将 DNF（未完赛/退赛）的车手也包含在“最后一名”中，导致算出来的时间差过大。 |
| **`task_418`** | 肌酐水平异常且年龄不到 70 岁的患者人数。 | **1行**<br>· `COUNT`: `['1']` | **1行**<br>· `count`: `['2']` | 边界不等号偏差。智能体将“不到 70 岁（`< 70`）”写成了“小于等于 70 岁（`<= 70`）”，多算了一人。 |

---

## 🛠️ 四大根源剖析与优化讨论方案（免代码修改）

不需要重构复杂的 Python 基础设施，我们可以通过调整**智能体系统提示词（System Prompt）**或在**知识指引（Knowledge Base）**中增加高级操作原则来从根本上缓解这些错误：

### 1. 应对 Gold 答案设计缺陷（类型一）
*   **改进思路**：在 ReAct 思考提示中增加“双向语义核对”原则：
    > *“当题目询问某个指标时，在输出最终结果前，仔细检查字段的语义是否可能存在官方命名的歧义（例如 'type' 指的是 event_type 还是 expense_type）。如果无法确定，可以尝试在 agent 日志中对比两种 SQL 解释，并优先选择与元数据结构最简单对齐的查询。”*

### 2. 规避特定指标的临界值边界模糊（类型二）
*   **改进思路**：在 ReAct 系统提示词追加严格的“临界边界与不等式检查”指令：
    > *“When encountering comparative filters (e.g. 'less than 70', 'normal level', 'abnormal'), you MUST perform an explicit threshold verification against `knowledge.md` or domain documents. Do NOT rely on model general knowledge (e.g. standard medical limits). For strict phrases like 'less than 70', use `< 70` and NEVER use `<= 70` unless authorized.”*

### 3. 防御时空过滤与数值排序陷阱（类型三）
*   **改进思路**：在提示词中加入强力的“多格式适应与数据预探”规则：
    > *“1. BEFORE applying string filters (like `June, 2013` or `Riverside`), you MUST execute a fast exploratory query (e.g., `SELECT DISTINCT city ...`) to inspect the actual value distribution and prevent over-broad filtering.<br>2. When sorting time or duration strings, ALWAYS convert them to float seconds in SQL or Python. Alphabetical sorting on time strings is strictly prohibited.”*

### 4. 彻底杜绝非结构化文档硬编码与漏配（类型四）
*   **改进思路**：在提示词中增加无可动摇的“零硬编码”红线：
    > *“NO MANUAL TRANSCRIPTION / HARDCODING. You are STRICTLY FORBIDDEN from manually typing raw data or lists into Python code blocks. You MUST write robust, automated parsers (using re, pandas, or beautifulsoup) to process 100% of the document content dynamically.”*
