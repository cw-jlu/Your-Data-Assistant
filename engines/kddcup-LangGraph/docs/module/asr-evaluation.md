# ASR 评测系统

> 文档状态：Current  
> 评测对象：`briefing.mp4` 中的旁白转写  
> 参考口径：Azure Fast Transcription 原始输出  
> 当前基线：`faster-whisper base / CPU / int8`

## 评测边界

本系统用于回答：

- 当前 faster-whisper 基线与冻结的 Azure 转写结果有多接近；
- 中文 CER、英文 WER 分别是多少；
- 错误来自替换、删除还是插入；
- 语言检测、运行成功率和耗时是否稳定。

Azure 输出不是人工逐字标注，因此 CER/WER 只能表述为“相对于 Azure Fast
Transcription 的错误率”，不能解释为绝对真实准确率。

首版不包含 medium、Qwen 术语生成、UI 词 Prompt、末尾幻觉裁剪或多模型比较。

## 三阶段流程

```text
briefing.mp4
  └─ 16 kHz / mono / PCM WAV
       ├─ Azure Fast Transcription → 冻结 gold
       └─ faster-whisper base      → candidate run
                                      └─ 离线 CER/WER
```

Azure 与 faster-whisper 使用同一份标准 WAV，避免输入容器和解码方式影响比较。
生产视频预处理流程仍然可以直接读取 MP4，不受评测代码影响。

### 生成 Azure gold

```powershell
uv run dabench asr-generate-gold `
  --config configs/full.yaml `
  --gold-id azure-2025-10-15-initial
```

只处理部分任务：

```powershell
uv run dabench asr-generate-gold `
  --config configs/full.yaml `
  --gold-id azure-smoke `
  --task-id task_6
```

同一个 `gold-id` 可以安全恢复未完成请求。已经成功且音频哈希一致的样本会被跳过；
源视频或标准 WAV 哈希发生变化时，命令会拒绝复用旧 gold。Azure 模型或生成日期变化
时应使用新的 `gold-id`，不能覆盖既有基准。

### 运行当前 ASR 基线

```powershell
uv run dabench asr-run `
  --config configs/full.yaml `
  --run-id base-cpu-int8
```

模型、设备和计算类型来自配置文件的 `video_preprocessing`：

```yaml
video_preprocessing:
  asr_model: base
  asr_device: cpu
  asr_compute_type: int8
```

模型在一次 run 中只加载一次。成功结果可以断点复用；配置或音频哈希不一致时必须使用
新的 `run-id`。失败样本在再次执行同一 run 时会重试。

### 离线评分

```powershell
uv run dabench asr-score `
  --gold-id azure-2025-10-15-initial `
  --run-id base-cpu-int8
```

评分不访问 Azure，也不会再次运行 Whisper。候选转写缺失或失败时，该样本按全部删除
计入错误率，并同时进入失败率统计。

## Azure Speech 配置

当前使用 GA API `2025-10-15`：

```text
POST {endpoint}/speechtotext/transcriptions:transcribe?api-version=2025-10-15
```

接口字段、文件限制和响应结构以
[Azure Fast Transcription REST API](https://learn.microsoft.com/en-us/rest/api/speechtotext/transcriptions/transcribe?view=rest-speechtotext-2025-10-15)
为准；标准 WAV 属于 Azure
[明确支持的输入格式](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription-audio-data)。

需要先在 Azure Portal 中创建支持 Speech 的 Azure AI Services/Speech 资源，并取得
资源 endpoint 与 key。项目不会自动创建资源或产生云端费用。

将凭据放入系统环境变量，或写入不会提交的项目根目录 `.env`：

```dotenv
AZURE_SPEECH_ENDPOINT=https://<region>.api.cognitive.microsoft.com
AZURE_SPEECH_KEY=<subscription-key>
```

凭据只用于请求头，不会写入 manifest、原始响应或终端日志。manifest 仅保存 endpoint
主机名。

生成完整 gold 前建议先对一条视频做 smoke test：

```powershell
uv run dabench asr-generate-gold `
  --config configs/full.yaml `
  --gold-id azure-smoke `
  --task-id task_6
```

确认 locale、文本和时间戳正常后，再为全部 30 条视频创建新的正式 `gold-id`。

## 归一化与指标

参考文本和候选文本同时执行：

1. Unicode NFKC；
2. 英文字母小写化；
3. 繁体转简体；
4. 去除标点和无关字符。

中文保留汉字、ASCII 英文字母和数字，以字符为单位计算 CER。英文保留 ASCII
字母和数字，以词为单位计算 WER：

```text
Error Rate = (Substitutions + Deletions + Insertions) / Reference Length
```

报告同时给出：

- corpus error rate：先累加同语言全部 S/D/I 和参考长度，再计算主指标；
- macro error rate：逐条错误率的简单平均，用于发现长短音频差异；
- 每条视频的 S/D/I、语言检测结果、耗时和实时因子；
- 候选成功率与语言检测准确率。

中英文不合并成单个错误率。

## 产物

本地数据和实验产物沿用仓库的忽略规则，不提交到 Git：

```text
evaluation/asr/
  audio/
    task_6.wav
    task_6.json
  gold/<gold-id>/
    manifest.json
    raw/task_6.json

artifacts/asr/runs/<run-id>/
  manifest.json
  scores/<gold-id>/
    score.json
    per_sample.csv
    score_report.md
```

`manifest.json` 使用带版本号的结构，并绑定：

- task ID 和源视频相对路径；
- 源视频与标准 WAV 的 SHA-256；
- Azure API/locale 或 faster-whisper 模型配置；
- 生成时间、成功状态和失败原因。

原始 Azure JSON 会完整保存，但订阅密钥不会进入任何产物。

## 离线开发验收

在尚未开通 Azure Speech 资源时，可以运行固定响应 fixture 的测试：

```powershell
uv run pytest tests/test_asr_evaluation.py
```

测试覆盖归一化、编辑距离、Azure 响应解析与重试、模型单次加载、冻结 gold、断点续跑、
失败样本计分、三类报告和离线 CLI。测试不会调用真实 Azure，也不会下载 Whisper 模型。
