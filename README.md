# Data Agent Unified Client

## Windows 桌面版

构建便携版客户端：

```powershell
.\build-windows.ps1
```

产物位于：

- `dist/DataAgent/DataAgent.exe`：直接运行的 Windows 桌面客户端；
- `dist/DataAgent-Windows-x64.zip`：包含 EXE、四套引擎源码和 `uv.exe` 的完整分发包。

客户端使用系统 WebView2 渲染原生窗口，不会打开外部浏览器。本地 HTTP 服务仅绑定随机
loopback 端口，并随窗口关闭。应用数据库会在首次启动时自动创建：

```text
%LOCALAPPDATA%\DataAgent\data-agent.db
```

SQLite 保存任务状态、用户可见 Trace 日志、工作区和上传文件元数据。上传原文件及运行输出
保存在同目录下的文件夹中，不作为 BLOB 写进数据库，以保持数据库轻量。首次执行某个引擎时，
内置的 `uv.exe` 会根据该引擎的锁文件安装运行依赖。

这是四套 KDD Cup 2026 DataAgent-Bench 方案的本地统一客户端。它不会把上游仓库直接揉成一个难以升级的包，而是先建立稳定的任务、运行和结果协议：

- 同一个页面选择 LangGraph、Kobushi、Memory ReAct、Mamba Agent；
- 点击上传按钮查看支持格式并上传本地资料；
- 输入自然语言 Query，客户端自动生成内部标准任务；
- 统一注入 OpenAI-compatible 模型参数；
- 单次 Query 可交给一个或多个引擎；
- 查看输出、停止任务，并展开完整进程与 artifact Trace；
- 每套引擎仍在自己的源码目录和依赖环境中运行。

## 启动

在 PowerShell 中运行：

```powershell
cd D:\code\python\kdd\unified-client
.\start.ps1
```

或：

```powershell
python -m app.server --open
```

默认地址是 [http://127.0.0.1:8765](http://127.0.0.1:8765)。客户端自身只使用 Python 标准库；首次启动某个引擎时，`uv` 会按该引擎自己的 `pyproject.toml` 同步依赖。

## 使用方式

1. 在首页选择一个或多个引擎，Mamba 默认选中。
2. 点击 Query 输入框下方的 `＋`，先查看可用格式，再选择文件。
3. 输入针对这些资料的问题。
4. 如有需要，点击 `⌁` 设置 API 地址、模型、Key、并发和超时。
5. 点击 `↑` 提交。
6. 答案会直接出现在对话中；点击“完整 Trace”查看运行日志、`trace.json`、`summary.json` 和 Mamba `tracing.db`。

客户端在内部自动生成：

```text
.runtime/workspaces/<workspace-id>/input/task_1/
├── task.json       # Query
└── context/        # 上传文件
```

因此所有上游引擎仍然接收标准 DABench 任务，Kobushi 也可以直接用于单次上传式 Query。

## 文件兼容性

上传弹窗按当前所选引擎取扩展名交集，不再把“客户端允许上传”当成“所有 Agent 都能处理”。

| 引擎 | 主要原生支持 |
| --- | --- |
| LangGraph | CSV、JSON、SQLite、MD/TXT/PDF、JPG/PNG/WebP、常见视频 |
| Kobushi | CSV/TSV、JSON/JSONL、SQLite、MD/TXT/PDF/HTML/XML、常见视频 |
| Memory ReAct | CSV/TSV、JSON、SQLite、MD/TXT/PDF、XLSX/XLSM、Parquet |
| Mamba Agent | CSV、JSON、SQLite、MD/TXT/PDF、常见视频 |

独立 MP3/WAV、DOCX、旧式 XLS 等格式没有被这些 Agent 的实际工具链共同支持，因此已从上传列表移除。Mamba 视频单文件还有 100 MB 的上游硬限制。弹窗可以展开查看每个引擎的完整扩展名与限制说明。

## 四个适配器

| 客户端名称 | 上游入口 | 主要能力 | 模式 |
| --- | --- | --- | --- |
| LangGraph | `uv run dabench ...` | 图工作流、多模态预处理、多级验证 |
| Kobushi | `uv run python submission/main.py` | 分阶段 ReAct、实验包切换、ASR |
| Memory ReAct | `uv run dabench ...` | 自一致性、多轮投票、跨任务记忆 |
| Mamba Agent | `uv run dabench ...` | ETL 路由、原生工具、SQLite tracing |

Kobushi 的官方 submission 入口会扫描输入目录；客户端为每次 Query 创建只含 `task_1` 的独立目录，因此它现在和其他引擎一样可直接选择。

## 运行产物

所有客户端运行产物位于：

```text
.runtime/
├── outputs/<run-id>/
├── logs/<run-id>/
├── traces/<run-id>.db
└── workspaces/<workspace-id>/
```

运行期 YAML 配置位于 `.runtime/configs/`，进程结束后会立即删除。前三个适配器尽量通过环境变量注入 API Key；Mamba 上游配置加载器不支持 Key 的环境变量覆盖，所以客户端会短暂写入运行期配置，并在退出后清理。`.runtime/` 已加入 `.gitignore`。

## 当前边界

- 客户端已完成“适配器级整合”：统一上传、Query、调度、输出与 Trace。
- 不同引擎的答案目前并列展示，尚未自动判定哪一份更优。
- 进程内存状态会在客户端服务重启后清空，但磁盘输出仍保留。
- 首次 `uv run` 可能下载大量依赖；ASR 引擎还可能需要模型权重。

进一步的能力级融合方案见 [docs/融合架构.md](docs/融合架构.md)。
