# 评估报告
**运行ID：** 20260425T033931Z
**评分公式：** Score = Recall - λ·(Extra/Predicted), λ = 0.5

## 📊 评分统计
| 指标 | 数值 |
|-----|-----|
| 总任务数 | 50 |
| 有效任务数 | 23 |
| **执行成功率** | 23/50 = 46.00% |
| **正确率（平均Score）** | 0.7205 = 72.05% |
| **完全正确率（Score=1.0）** | 16/23 = 69.57% |

## 任务结果详情

### ✅ 完全正确的任务 (16)

- **task_24** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_26** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_64** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_67** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_145** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_194** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_214** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_243** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_250** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_292** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_303** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_349** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_350** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_352** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_408** (Score=1.0000): 1列完全匹配, Recall=1.00, Penalty=0.0000
- **task_415** (Score=1.0000): 2列完全匹配, Recall=1.00, Penalty=0.0000

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


### ❌ 失败的任务 (6)

- **task_22**
- **task_25**
- **task_80**
- **task_89**
- **task_169**
- **task_180**

### 🚫 错误的任务 (27)

- **task_11**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_11/prediction.csv
- **task_19**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_19/prediction.csv
- **task_27**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_27/prediction.csv
- **task_74**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_74/prediction.csv
- **task_75**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_75/prediction.csv
- **task_86**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_86/prediction.csv
- **task_163**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_163/prediction.csv
- **task_173**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_173/prediction.csv
- **task_196**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_196/prediction.csv
- **task_199**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_199/prediction.csv
- **task_200**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_200/prediction.csv
- **task_218**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_218/prediction.csv
- **task_249**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_249/prediction.csv
- **task_257**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_257/prediction.csv
- **task_259**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_259/prediction.csv
- **task_261**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_261/prediction.csv
- **task_269**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_269/prediction.csv
- **task_283**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_283/prediction.csv
- **task_287**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_287/prediction.csv
- **task_305**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_305/prediction.csv
- **task_330**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_330/prediction.csv
- **task_344**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_344/prediction.csv
- **task_355**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_355/prediction.csv
- **task_379**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_379/prediction.csv
- **task_396**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_396/prediction.csv
- **task_418**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_418/prediction.csv
- **task_420**: Prediction CSV not found: /home/gubin/Oralagent/kdd/starter-kit/artifacts/runs/20260425T033931Z/task_420/prediction.csv

