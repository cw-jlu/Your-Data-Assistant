# 评估报告
**运行ID：** 20260427T111355Z
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
| **正确率（平均Score）** | 0.7028 = 70.28% |
| **完全正确率（Score=1.0）** | 31/45 = 68.89% |

## 任务结果详情

### ✅ 完全正确的任务 (31)

- **task_11** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_19** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_26** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_27** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
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
- **task_418** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_420** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000

### ⚠️ 部分正确的任务 (1)

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


### ❌ 完全错误的任务 (13)

- **task_25**
- **task_80**
- **task_86**
- **task_89**
- **task_163**
- **task_169**
- **task_173**
- **task_180**
- **task_199**
- **task_200**
- **task_283**
- **task_344**
- **task_379**

### 🚫 执行失败的任务 (5)

- **task_22**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T111355Z/task_22/prediction.csv
- **task_24**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T111355Z/task_24/prediction.csv
- **task_38**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T111355Z/task_38/prediction.csv
- **task_64**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T111355Z/task_64/prediction.csv
- **task_396**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T111355Z/task_396/prediction.csv

