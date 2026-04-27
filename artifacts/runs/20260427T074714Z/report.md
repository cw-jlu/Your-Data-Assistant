# 评估报告
**运行ID：** 20260427T074714Z
**评分公式：** Score = Recall - λ·(Extra/Predicted), λ = 0.5

## 📊 评分统计
| 指标 | 数值 |
|-----|-----|
| 总任务数 | 50 |
| 有效任务数 | 37 |
| **执行成功率** | 37/50 = 74.00% |
| **正确率（平均Score）** | 0.7196 = 71.96% |
| **完全正确率（Score=1.0）** | 26/37 = 70.27% |

## 任务结果详情

### ✅ 完全正确的任务 (26)

- **task_11** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_19** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_27** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_74** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_75** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
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
- **task_355** (Score=1.0000): 3列完全匹配, Recall=1.00, Penalty=0.0000
- **task_408** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_415** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000
- **task_420** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000

### ⚠️ 部分正确的任务 (1)

#### task_38

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
- `amount`
- `balance`
- `date`


### ❌ 完全错误的任务 (10)

- **task_25**
- **task_80**
- **task_163**
- **task_169**
- **task_199**
- **task_259**
- **task_344**
- **task_379**
- **task_396**
- **task_418**

### 🚫 执行失败的任务 (13)

- **task_22**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_22/prediction.csv
- **task_24**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_24/prediction.csv
- **task_26**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_26/prediction.csv
- **task_64**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_64/prediction.csv
- **task_67**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_67/prediction.csv
- **task_86**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_86/prediction.csv
- **task_89**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_89/prediction.csv
- **task_145**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_145/prediction.csv
- **task_173**: Prediction CSV is empty or cannot be read: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_173/prediction.csv
- **task_180**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_180/prediction.csv
- **task_200**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_200/prediction.csv
- **task_350**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_350/prediction.csv
- **task_352**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260427T074714Z/task_352/prediction.csv

