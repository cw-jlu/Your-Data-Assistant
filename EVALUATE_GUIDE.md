# 评分脚本使用指南

## 功能说明

评分脚本 `evaluate.py` 按照 **官方 Leaderboard Scoring Rule** 自动对预测结果进行评分，并生成详细的错误对比报告。

### 评分规则（官方规则 6.3）

本脚本基于官方规则的列级内容一致性匹配方法：

1. **Recall计算**：`Recall = 匹配的列数 / Gold列总数`
2. **惩罚项**：`Penalty = λ · (额外列数 / 预测列总数)`
3. **最终分数**：`Score = Recall - Penalty`，下界为0

其中：
- `λ` (lambda) 是惩罚权重，平衡覆盖率和冗余预测（默认0.5）
- 匹配标准：列数据内容相同（无序比较），忽略列名和行顺序
- 额外列：预测中存在但Gold中不存在的列

### 三个关键指标

| 指标 | 定义 |
|-----|------|
| **成功率** | 成功运行的任务 / 总任务数 |
| **Score（平均得分）** | 所有任务的平均得分（运行报错任务记为 0 分） |
| **准确率** | 执行成功中完全正确（Score = 1.0）的任务比例（完全正确任务数 / 有效任务数） |

### 内容归一化规则（官方规则 6.5）

在构建列签名并比较前，脚本会对单元格内容自动进行归一化：
1. **空值**：统一转换为空字符串（支持忽略大小写的 `null`, `none`, `nan`, `nat`, `<na>`）
2. **数值**：解析为小数并四舍五入到小数点后 2 位（如 `42000000` -> `4200000.00`）
3. **日期/时间**：转化为 ISO 8601 标准，有时间及带时区的将被转换为 UTC 格式（以 `Z` 结尾），纯日期格式为 `YYYY-MM-DD`
4. **字符串**：去除前后空白及 `\r\n`，大小写敏感
5. **名称字段**：脚本内建列合并机制，支持将名/姓组合匹配单列全名

## 快速使用

### 方式一：使用 uv（推荐）

#### 1. 评估最新运行

```powershell
uv run python evaluate.py
```

#### 2. 评估指定运行

```powershell
uv run python evaluate.py --run-id 20260424T123624Z
```

#### 3. 自定义Lambda参数

```powershell
uv run python evaluate.py --lambda 0.3
```

#### 4. 保存Markdown报告

```powershell
uv run python evaluate.py --save-md
```

#### 5. 自定义报告路径

```powershell
uv run python evaluate.py --save-md --md-path ./my_report.md
```

#### 6. 仅显示错误详情

```powershell
uv run python evaluate.py --errors-only
```

### 方式二：直接运行（需要环境配置）

如果已配置好Python环境，可直接运行：

```bash
python evaluate.py                          # 评估最新运行
python evaluate.py --run-id <run_id>       # 评估指定运行
python evaluate.py --lambda 0.3             # 自定义Lambda
python evaluate.py --save-md                # 保存MD报告
python evaluate.py --errors-only            # 仅显示错误
```

**环境要求**：需要安装依赖包
```bash
pip install pandas rich
```

### 报告保存位置

使用 `--save-md` 生成的Markdown报告默认保存到：
```
artifacts/runs/<run_id>/report.md
```

自定义保存位置：
```bash
python evaluate.py --save-md --md-path ./my_report.md
```

## 输出说明

### 汇总表 (Evaluation Summary)

| 列名 | 含义 |
| --- | --- |
| Task ID | 任务 ID |
| Score | 0.00 - 1.00，反映Recall-Penalty的结果 |
| Status | PASS / FAIL / ERROR |
| Recall/Penalty | Recall和Penalty的具体值 |

**颜色说明**：
- 🟢 绿色：Score = 1.0（完全正确）
- 🟡 黄色：0.5 ≤ Score < 1.0（部分正确）
- 🔴 红色：Score < 0.5（失败）

### 整体统计 (Overall Statistics)

- **总任务数**：参与评分的任务总数
- **有效任务数**：成功运行的任务数（无错误）
- **1️⃣ 成功率**：有效任务数 / 总任务数
- **2️⃣ Score（平均得分）**：所有任务的Score平均值（包括报错任务的0分）
- **3️⃣ 准确率**：Score=1.0的任务数 / 有效任务数

### 详细错误 (Detailed Errors)

对每个失败任务输出：

1. **❌ Missing Columns**：任务中 gold 标准要求但预测缺失的列
2. **❌ Mismatched Columns**：预测包含 gold 列但值不匹配，显示：
   - `预期值`：标准答案的值
   - `实际值`：预测的值
   - `缺失值`：应有但未有的值
   - `多余值`：有但不应有的值
3. **ℹ️ 额外列**：预测中的额外列（会产生Penalty）

### ERROR 状态

- `Prediction CSV not found`：任务未生成预测文件
- `Gold CSV not found`：标准答案不存在（数据问题）

## 示例解读

### 示例 1：完全正确（Score = 1.0）

```
task_10 │ 1.00 │ PASS │ Recall=1.00, Penalty=0.00
```

所有Gold列都完全匹配，无额外列。

### 示例 2：缺失列（Score < 0.5）

```
task_19 │ 0.33 │ FAIL │ Recall=0.67, Penalty=0.34
❌ Missing Columns:
   - first_name
   - last_name
```

缺失2个Gold列（预期3列，只匹配1列），Recall=1/3≈0.33。

### 示例 3：有额外列（Score < 1.0但Recall = 1.0）

```
task_11 │ 0.83 │ PASS │ Recall=1.00, Penalty=0.17
ℹ️ 额外列 (1):
   - extra_col_1
```

所有Gold列匹配，但有额外列。假设预测3列，Gold2列：
- Recall = 2/2 = 1.0
- Penalty = 0.5 × (1/3) ≈ 0.17
- Score = 1.0 - 0.17 = 0.83

### 示例 4：值不匹配（Score = 0.5）

```
task_25 │ 0.50 │ FAIL │ Recall=0.50, Penalty=0.00
❌ 不匹配的列 (1):
   - event_name: 预期1个值, 实际1个值
```

有2个Gold列，只1列匹配，1列不匹配：
- Recall = 1/2 = 0.5
- Score = 0.5 - 0 = 0.5

## 结果路径

- **汇总报告**：命令行输出（自动刷新）
- **Markdown报告**：`artifacts/runs/<run_id>/report.md`（使用--save-md生成）
- **详细追踪**：每个任务的 trace.json 在 `artifacts/runs/<run_id>/<task_id>/`
- **预测结果**：每个任务的预测 CSV 在 `artifacts/runs/<run_id>/<task_id>/prediction.csv`
- **金标准**：公开集的标准答案在 `../public/output/task_<id>/gold.csv`

## Lambda参数调优

根据不同场景调整λ值：

| λ值 | 倾向 | 适用场景 |
|-----|------|---------|
| 0.0 | 仅看Recall（完全忽略额外列） | 只关心覆盖率 |
| 0.3 | 轻微惩罚冗余 | 允许一些额外列 |
| **0.5** | **默认均衡** | **一般情况** |
| 0.7 | 强烈惩罚冗余 | 严格控制输出 |
| 1.0 | 严厉惩罚额外列 | 极其严格的场景 |

修改全局参数：编辑 `evaluate.py` 第37行的 `LAMBDA_PARAM = 0.5`

## 故障排除

| 问题 | 原因 | 解决方案 |
| --- | --- | --- |
| `No run directories found` | 没有运行结果 | 先执行 `uv run dabench run-benchmark` |
| `Prediction CSV not found` | 任务未成功生成预测 | 查看 trace.json 中的 failure_reason |
| `Gold CSV not found` | 数据文件丢失 | 检查 public/output 目录结构 |
| 所有任务Score都很低 | 模型输出的列数与预期不符 | 检查prompt是否正确指导生成预期列名 |
| Penalty值很高 | 额外列过多 | 调整模型提示或增加λ值 |
| `ModuleNotFoundError: No module named 'pandas'` | 缺少依赖包 | 运行 `pip install pandas rich` 或使用 `uv run` |

## 进阶：自定义评分

### 修改全局Lambda参数

编辑 `evaluate.py` 第37行：
```python
LAMBDA_PARAM = 0.5  # 改为其他值
```

### 自定义列匹配逻辑

如需修改列的比较规则，编辑以下函数：

- `normalize_column_vector()`：列值预处理（当前：转为集合，支持未来的column signature）
- `compare_column_vectors()`：向量比较逻辑
- `evaluate_task()`：任务评分流程
