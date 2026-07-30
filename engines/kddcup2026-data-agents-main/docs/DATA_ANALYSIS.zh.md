# DataAgent-Bench — 数据分析报告

> 🌐 **Language**: [English](DATA_ANALYSIS.md) · [한국어](DATA_ANALYSIS.ko.md) · **中文**

KDD Cup 2026 DataAgent-Bench 挑战公开 50 任务集的定量分析,以及截至 v2 (2026-04-29) 的 holdout 实测结果。是决定 v3+ 改哪里的依据。

> 本文档范围:**数据集本身 + 我们的智能体在其上如何运作**。
> 系统内部见 [`ARCHITECTURE.zh.md`](ARCHITECTURE.zh.md);提交历史与分数趋势见 [`SUBMISSION_LOG.zh.md`](SUBMISSION_LOG.zh.md);运维手册见 [`../CLAUDE.zh.md`](../CLAUDE.zh.md)。

---

## 1. 一目了然

| 项目 | 值 |
|---|---|
| 公开任务数 | 50 |
| 拆分 | train 40 / holdout 10 / smoke 5 (`scoring/holdout.py` 确定性拆分) |
| 难度分布 | easy 15 / medium 23 / hard 11 / extreme 1 |
| 上下文文件类型 | 仅 `.csv`, `.json`, `.db` (SQLite), `.md` — 无 PDF/Excel/Parquet/图像 |
| 始终存在的资源 | `context/knowledge.md` (50/50 — 领域字典、表说明) |
| 最大上下文 | task_257 = **441 MB** (medium) |
| 答案典型形态 | 1–3 列, 中位数 1 行, 最多 140 |

核心启示:
- 每个公开任务都带 `knowledge.md` — 不自动注入会丢失领域上下文 (v2 已处理)。
- 多数标准答案是 1×1 单值 (lookup 风格) — 一处归一化错位即可让该任务变 0 分。
- Phase 1 数据无 PDF/Excel/图像,但主办方 hidden set 明示多模态 → v3 之后加 reader 候选。

---

## 2. 难度 × 上下文形态 (50 任务)

### 2.1 难度分布

| Difficulty | n | 占比 | 规则定义 |
|---|---|---|---|
| easy | 15 | 30% | 结构化文件 + 知识文档 |
| medium | 23 | 46% | 结构化 + DB + 文档 |
| hard | 11 | 22% | 多源 + 非结构化 (~10K~128K tokens) |
| extreme | 1 | 2% | 超长输入 (>128K tokens) |

extreme 仅 1 个,但 hidden set 比重可能上升 — runner 的 `max_steps_by_difficulty` 映射 (extreme=28) 已为该场景准备。

### 2.2 上下文子树出现频率 (50 中)

| 子树 | 频次 | 用途 |
|---|---|---|
| `csv/` | 36 | 表格 |
| `json/` | 30 | 结构化数据 (常含嵌套对象) |
| `db/` (SQLite `.db`) | 27 | 规范化表 — JOIN/聚合 |
| `doc/` | 12 | 辅助文档 (`.md`) |
| **`knowledge.md`** | **50** | 领域字典 — 100% 存在 |

组合模式 (50 中频次靠前):
- CSV+JSON (12) — 通过外键 lookup
- CSV+DB (10) — DB schema → CSV 增加事实
- JSON+DB (6) — DB 规范化, JSON 嵌套
- CSV+JSON+DB (4) — 三源组合 (主要 medium+)

### 2.3 文件后缀汇总

| 后缀 | 数量 | 备注 |
|---|---|---|
| `.md` | 64 | knowledge.md 50 + doc/* 14 |
| `.csv` | 40 | |
| `.json` | 37 | |
| `.db` | 27 | SQLite |
| 其它 | **0** | 无 PDF/Excel/Parquet/图像 |

---

## 3. 上下文大小分布 — 内存负载指标

| Difficulty | n | min | median | max |
|---|---|---|---|---|
| easy | 15 | 17 KB | 287 KB | **58 MB** |
| medium | 23 | 38 KB | 1.4 MB | **441 MB** |
| hard | 11 | 41 KB | 256 KB | **267 MB** |
| extreme | 1 | 376 KB | 376 KB | 376 KB |

**8 个大上下文 (≥50 MB):**

| task_id | difficulty | 大小 |
|---|---|---|
| task_257 | medium | 441 MB |
| task_250 | medium | 384 MB |
| task_330 | hard | 267 MB |
| task_259 | medium | 182 MB |
| task_249 | medium | 166 MB |
| task_243 | medium | 137 MB |
| task_420 | hard | 59 MB |
| task_38 | easy | 58 MB |

**启示:**
- 32K 上下文窗口 (我们的 vLLM) 无法装下原始数据。→ `dataframe_describe`/`dataframe_head` 压缩 prepass 必需 (v2 引入)。
- 每次调用 reload 在 30s 限制下 IO 受卡。→ persistent IPython kernel (v2 引入) 在 task_249 / task_250 这类大场景上有直接 ROI。

---

## 4. 答案 (`gold.csv`) 形态分布

| 形态 | 频次 |
|---|---|
| 1 列 | 40 / 50 (80%) |
| 2 列 | 7 / 50 |
| 3 列 | 3 / 50 |

行数: min 1, **median 1**, p90 7, max 140.

**启示:**
- 答案为单值 (1×1) 的情况最常见 → easy 多为 lookup 风格问题。
- column-ablation (Phase 3 计划) 仅在列 6+ 时触发,公开集几乎不会触发 → 给 hidden set 的宽表场景留代码。
- 也有 140 行长列表 (task_180) — 长列表答案的归一化成本与行数线性相关。

---

## 5. 问题 (自然语言) 长度

| 统计 | 值 |
|---|---|
| min | 30 字符 |
| median | 90 字符 |
| max | 144 字符 |
| mean | 88 字符 |

问题很短 — 模型需要在很大的语义空白中推断意图。系统 prompt 必须明示答案格式 (归一化、列签名) 才能最小化分数损失 (v2 引入)。

---

## 6. v2 holdout 实测 (2026-04-29)

### 6.1 分数概览 (`λ ∈ {0.05, 0.10, 0.20}`)

| λ | mean Score | mean Recall |
|---|---|---|
| 0.05 | 0.6300 | 0.6333 |
| **0.10** | **0.6267** | **0.6333** |
| 0.20 | 0.6200 | 0.6333 |

三个 λ 上 Score 几乎平移 — 额外列罚分影响很小 (我们不会过度产出列)。

### 6.2 按难度 (容器, λ=0.10)

| difficulty | n | score | recall |
|---|---|---|---|
| easy | 3 | 0.6667 | 0.6667 |
| medium | 5 | **0.8000** | 0.8000 |
| hard | 2 | **0.1333** | 0.1667 |
| extreme | 0 | — | — |

**解读:**
- **medium 0.80** — knowledge.md 注入 + dataframe prepass + persistent kernel 协同效应。该桶在 v3 难再榨 ROI (接近平台期)。
- **easy 0.67** — 1×1 lookup 但 1/3 仍 0 分。怀疑归一化或过度产出。
- **hard 0.13** — 最大泄漏。是 v3 的主要目标。

### 6.3 holdout 10 任务逐行测量

各任务上下文形态 + 答案匹配模式:

| task_id | diff | ctx 子树 | size | steps | gold (c×r) | pred (c×r) | 形态匹配 | 结论 |
|---|---|---|---|---|---|---|---|---|
| task_11 | easy | json | 0.5 MB | 5 | 3 × 3 | 3 × 6 | **rows ↑** | 列签名错乱 (over-emission) |
| task_27 | easy | json | 24 KB | 5 | 3 × 1 | 3 × 1 | OK | 估计匹配 |
| task_75 | easy | csv+json | 0.5 MB | 8 | 1 × 1 | 1 × 1 | OK | 估计匹配 |
| task_169 | medium | csv+db | 9 MB | 7 | 1 × 1 | 1 × 1 | OK | 匹配 |
| task_249 | medium | db+json | 174 MB | 7 | 2 × 1 | 2 × 1 | OK | 匹配 (大上下文 OK) |
| task_250 | medium | csv+db+json | **403 MB** | 6 | 1 × 1 | 1 × 1 | OK | 匹配 (kernel/dataframe 收益) |
| task_261 | medium | csv+db+json | 0.13 MB | 6 | 1 × 1 | 1 × 1 | OK | 匹配 |
| task_303 | medium | db+json | 0.24 MB | 9 | 1 × 1 | 1 × 1 | OK | 匹配 |
| task_355 | hard | csv+doc | 41 KB | 6 | **3 × 1** | 2 × 1 | **cols ↓** | 列缺失 (under-emission) |
| task_408 | hard | db+doc | 1.1 MB | 8 | 1 × 1 | 1 × 1 | OK | 值不匹配 (归一化/答案错) 估计 |

**所有 10 个任务都到达 `_answer`** (succeeded=True) — 即从 0/10 fail 恢复到 10/10 答题。剩余分数泄漏来自 "到达但答错"。

---

## 7. 失败模式分类

v2 holdout 上 0 分(或部分)的任务有三种清晰可区分的模式。

### 7.1 Over-emission (行 ↑) — task_11

`gold` 3×3, 我们 `pred` 3×6。同样 3 列但 6 行。评分器在归一化后用排序值 multiset 构建签名,所以多余的行使 multiset 不匹配 → 0 列匹配。即 **3 行正确答案 + 6 行多余数据合并提交**的模式。这一个任务把 easy 平均拉低。

**应对候选:**
- 系统 prompt 添加: "答案只包含问题要求的精确行数,无多余"。
- `answer_validator` (Phase 3) 在不知道 gold 行数时也警告 "返回原始查询结果未做 dedup/筛选" 模式。

### 7.2 Under-emission (列 ↓) — task_355

`gold` 3×1 vs `pred` 2×1。漏一列。评分器最多匹配 2 → recall ≤ 2/3, λ-罚 0。某 hard 任务在 0~0.33 之间。

**应对候选:**
- "When in doubt about including a column, INCLUDE" 已在 v2 prompt 中。但 hard 任务的多源 join 中模型仍漏一列。
- `answer_validator` 启发式: 比较问题中的名词短语数 (例如 "list their A, B, and C") 与答案列数。

### 7.3 值不匹配 (归一化/计算) — task_408

`gold` 1×1, `pred` 1×1, 形态匹配但分数为 0 (或部分)。最难的模式: 形态对了但值不同。可能原因:
- SQL/Python 计算结果错误
- 归一化 — numeric ROUND_HALF_UP vs HALF_EVEN, 日期格式, trimming 缺失
- doc/ 中非结构化线索被模型误解

**应对候选:**
- 直接分析 trace.json 看哪一步推断错了 — task_408 trace 8 步 marathon 风格推断,可能存在错误的领域假设。
- `dataframe_describe` 输出加 numeric 列的 ROUND_HALF_UP 归一化预览。

---

## 8. 优势·弱点单页总结

### 优势 (v2 时点)

1. **Medium 0.80** — knowledge.md 自动注入 + persistent kernel + dataframe_describe 协同发力。
2. **大上下文 (≥100 MB) 全通过** — task_249 (174 MB), task_250 (403 MB) 都匹配。验证 persistent kernel 引入的合理性。
3. **Easy + medium 的 1×1 lookup** — JSON-mode probe + plan-then-execute prompt 在单值提取上稳定。
4. **所有任务都到达 `_answer`** — 10/10。`_coerce_action_input` 热修后无步骤浪费。

### 弱点 (v3+ 目标)

| 优先级 | 领域 | 证据 | 候选 |
|---|---|---|---|
| 🔴 High | hard 0.13 | task_355 列缺失, task_408 值不匹配 | answer_validator + question parsing 启发式 |
| 🟠 Mid | easy 1/3 为 0 | task_11 over-emission | "exact row count" prompt 强调 |
| 🟡 Low | hidden set 的 PDF/Excel | 公开集 0 但规则提及多模态 | 按 Phase 1.5 计划添加 reader |
| 🟡 Low | extreme (>128K) | 公开集 1 个,376 KB 较小 | YaRN/RoPE 扩展归 vLLM 责任 |

---

## 9. 下一轮 (v3) 候选 — ROI 排序

1. **answer_validator + conditional terminal** (Phase 3) — 在 `_answer` 之前: (a) 检查 numeric 列 dtype 一致性, (b) 比较问题名词短语数 vs 答案列数, (c) 对可疑单元格 warning。warning 作为 observation 返回 → agent 修正后再提交。
2. **hard tier trace 调试** — 除 task_355, task_408 外,分析 train 40 中 11 个 hard trace。找出共性失败模式,在系统 prompt 中加 1-2 行。
3. **column-signature self-consistency** (Phase 4 gate) — k=3, temperature 0→0.5, multiset 投票。用多数表决吸收 hard 任务的随机错答。仅在 12h 预算模拟通过时上线。
4. **额外行 guard** — `_coerce_action_input` 旁加 "rows 异常长警告" 启发式 (gold 未知时的 per-task 保守 cap)。

每次改动后用 holdout 10 个回归测量 → 只有 mean Score ≥ v2 + 3 分时才提交 v3。

---

## 10. 数据更新流程

§2~§5 是 **公开 50 任务静态分析** — 数据集不变就 stable。§6~§8 必须 **每次提交后更新**:

```bash
# 1. 新 run
uv run dabench run-benchmark --config configs/local.yaml \
  --task-set data/public/holdout_ids.txt

# 2. mock_scorer
uv run python -m data_agent_baseline.scoring.mock_scorer \
  --predictions artifacts/runs/<run_id> \
  --gold data/public/output --input data/public/input \
  --lambda-values 0.05 0.10 0.20

# 3. per-task shape 提取 (§6.3 表更新)
.venv/bin/python -c "
import json, pathlib
run_dir = pathlib.Path('artifacts/runs/<run_id>')
for trace in sorted(run_dir.glob('task_*/trace.json')):
    d = json.loads(trace.read_text())
    print(d['task_id'], d.get('succeeded'), len(d.get('steps', [])))
"
```

仓库 `.gitignore` 把本文档白名单 (`!docs/DATA_ANALYSIS.md`) 用于 git tracking — 团队共享。
