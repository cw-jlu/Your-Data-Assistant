# 评估报告
**运行ID：** 20260427T131119Z
**评分公式：** Score = Recall - λ·(Extra/Predicted), λ = 0.5

## ⚙️ 运行参数
- **Model**: qwen3.5-35b-a3b
- **Max Steps**: 35
- **Temperature**: 0.0
- **Max Tokens**: 8192

## 📊 评分统计
| 指标 | 数值 |
|-----|-----|
| 总任务数 | 50 |
| 有效任务数 | 45 |
| **执行成功率** | 45/50 = 90.00% |
| **正确率（平均Score）** | 0.7238 = 72.38% |
| **完全正确率（Score=1.0）** | 32/45 = 71.11% |

## 任务结果详情

### ✅ 完全正确的任务 (32)

- **task_11** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_19** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_22** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_26** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_64** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_67** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_74** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_75** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_145** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_194** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_196** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_214** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_218** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_243** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_249** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_250** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_257** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_261** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_269** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_283** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_287** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_292** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_303** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_305** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_330** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_349** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_350** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_352** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_355** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_408** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_415** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_420** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000

### ⚠️ 部分正确的任务 (1)

#### task_38

**得分:** 0.5714 (Recall=1.00, Penalty=0.4286)

| 指标 | 数值 |
|-----|-----|
| 预期列数 | 1 |
| 预测列数 | 7 |
| 匹配列数 | 1 |
| 缺失列数 | 0 |
| 不匹配列数 | 0 |
| 额外列数 | 6 |

**ℹ️ 额外列 (6):**
- `account_id`
- `amount`
- `balance`
- `date`
- `operation`
- `type`


### ❌ 完全错误的任务 (12)

- **task_25**
- **task_80**
- **task_86**
- **task_89**
- **task_169**
- **task_173**
- **task_180**
- **task_199**
- **task_200**
- **task_344**
- **task_379**
- **task_418**

### 🚫 执行失败的任务 (5)

- **task_24**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T131119Z/task_24/prediction.csv
- **task_27**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T131119Z/task_27/prediction.csv
- **task_163**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T131119Z/task_163/prediction.csv
- **task_259**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T131119Z/task_259/prediction.csv
- **task_396**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T131119Z/task_396/prediction.csv

