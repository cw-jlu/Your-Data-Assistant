# 评估报告
**运行ID：** 20260427T080845Z
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
| 有效任务数 | 40 |
| **执行成功率** | 40/50 = 80.00% |
| **正确率（平均Score）** | 0.6719 = 67.19% |
| **完全正确率（Score=1.0）** | 26/40 = 65.00% |

## 任务结果详情

### ✅ 完全正确的任务 (26)

- **task_64** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_67** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_74** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_75** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_145** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_194** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_196** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_200** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_218** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_243** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_249** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_250** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
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
- **task_355** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_408** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_415** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_420** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000

### ⚠️ 部分正确的任务 (2)

#### task_259

**得分:** 0.6250 (Recall=1.00, Penalty=0.3750)

| 指标 | 数值 |
|-----|-----|
| 预期列数 | 1 |
| 预测列数 | 4 |
| 匹配列数 | 1 |
| 缺失列数 | 0 |
| 不匹配列数 | 0 |
| 额外列数 | 3 |

**ℹ️ 额外列 (3):**
- `comment_id`
- `post_id`
- `score`

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
- `last_poster`


### ❌ 完全错误的任务 (12)

- **task_22**
- **task_80**
- **task_86**
- **task_89**
- **task_163**
- **task_169**
- **task_180**
- **task_199**
- **task_344**
- **task_379**
- **task_396**
- **task_418**

### 🚫 执行失败的任务 (10)

- **task_11**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_11/prediction.csv
- **task_19**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_19/prediction.csv
- **task_24**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_24/prediction.csv
- **task_25**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_25/prediction.csv
- **task_26**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_26/prediction.csv
- **task_27**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_27/prediction.csv
- **task_38**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_38/prediction.csv
- **task_173**: Prediction CSV is empty or cannot be read: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_173/prediction.csv
- **task_214**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_214/prediction.csv
- **task_352**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T080845Z/task_352/prediction.csv

