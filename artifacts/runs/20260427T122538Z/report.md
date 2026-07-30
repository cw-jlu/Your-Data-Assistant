# 评估报告
**运行ID：** 20260427T122538Z
**评分公式：** Score = Recall - λ·(Extra/Predicted), λ = 0.5

## ⚙️ 运行参数
- **Model**: qwen3.5-35b-a3b
- **Max Steps**: 50
- **Temperature**: 0.0
- **Max Tokens**: 8192

## 📊 评分统计
| 指标 | 数值 |
|-----|-----|
| 总任务数 | 50 |
| 有效任务数 | 41 |
| **执行成功率** | 41/50 = 82.00% |
| **正确率（平均Score）** | 0.5915 = 59.15% |
| **完全正确率（Score=1.0）** | 24/41 = 58.54% |

## 任务结果详情

### ✅ 完全正确的任务 (24)

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

#### task_257

**得分:** 0.2500 (Recall=0.50, Penalty=0.2500)

| 指标 | 数值 |
|-----|-----|
| 预期列数 | 2 |
| 预测列数 | 2 |
| 匹配列数 | 1 |
| 缺失列数 | 1 |
| 不匹配列数 | 1 |
| 额外列数 | 1 |

**❌ 缺失列 (1):**
- `DisplayName`

**❌ 不匹配的列 (1):**
- **`DisplayName`**: 预期1个值, 实际0个值
  - 缺失: {'mbq'}

**ℹ️ 额外列 (1):**
- `user_name`


### ❌ 完全错误的任务 (16)

- **task_25**
- **task_67**
- **task_80**
- **task_86**
- **task_89**
- **task_163**
- **task_169**
- **task_173**
- **task_180**
- **task_199**
- **task_259**
- **task_261**
- **task_344**
- **task_379**
- **task_396**
- **task_418**

### 🚫 执行失败的任务 (9)

- **task_11**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_11/prediction.csv
- **task_19**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_19/prediction.csv
- **task_22**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_22/prediction.csv
- **task_24**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_24/prediction.csv
- **task_26**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_26/prediction.csv
- **task_27**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_27/prediction.csv
- **task_38**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_38/prediction.csv
- **task_64**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_64/prediction.csv
- **task_200**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T122538Z/task_200/prediction.csv

