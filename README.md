# Data Agent Desktop

一个面向本地文件分析的 Windows 桌面客户端

1. 选择一个或多个 Agent；
2. 点击上传按钮并选择本地文件；
3. 输入自然语言 Query；
4. 查看结果和完整运行 Trace。

客户端使用随程序打包的 Qt WebEngine 显示桌面窗口，不会打开外部浏览器。任务记录、
日志和上传元数据通过 SQLite 持久化，应用重启后仍可查看。

## 下载与运行

推荐从 GitHub Releases 下载：

```text
DataAgent-Windows-x64.zip
```

解压后双击：

```text
DataAgent.exe
```

> 不要只复制 `DataAgent.exe`。当前版本采用便携目录结构，EXE 还需要同目录中的
> `_internal/`、`engines/` 和 `uv.exe`。请完整解压 ZIP 后运行。

系统要求：

- Windows 10 或 Windows 11，64 位；
- 不需要预装 Python、Conda、.NET、Java 或 WebView2；
- 首次运行某个 Agent 时需要联网安装其 Python 依赖；
- 一个 OpenAI-compatible 模型服务及 API Key。

## 功能

- Codex 风格的单窗口任务界面；
- 上传按钮动态显示当前所选 Agent 真正支持的文件类型；
- 同一个 Query 可提交给一个或多个 Agent；
- Kobushi 可直接处理客户端生成的单任务工作区；
- 实时展示运行状态和进程日志；
- 结果、标准输出、JSON artifact 与 SQLite Trace 对用户可见；
- 可停止正在运行的任务；
- SQLite 持久化任务历史、Trace 日志和工作区元数据；
- 应用异常退出后，未完成任务会恢复为“已中断”状态；
- Qt WebEngine 与 Python 运行时包含在便携包中；
- 本地服务只监听随机的 `127.0.0.1` 回环端口，并随窗口关闭。

## 内置 Agent

| Agent | 主要能力 | 客户端入口 |
| --- | --- | --- |
| LangGraph | 状态图编排、多模态预处理、多级验证 | `uv run dabench` |
| Kobushi | 分阶段 ReAct、实验配置、视频 ASR | `uv run python submission/main.py` |
| Memory ReAct | Self-consistency、跨运行投票、任务记忆 | `uv run dabench` |
| Mamba Agent | ETL 路由、原生工具调用、SQLite tracing | `uv run dabench` |

每套 Agent 仍在自己的源码目录和依赖环境中运行。桌面客户端只负责统一任务协议、
上传、调度、结果展示和 Trace 聚合，不会强行合并四套上游实现。

## 文件支持

上传窗口会根据当前所选 Agent 计算扩展名交集，避免出现“客户端允许上传，但某个
Agent 实际无法处理”的情况。

| Agent | 主要支持格式 |
| --- | --- |
| LangGraph | CSV、JSON、SQLite、Markdown、TXT、PDF、JPG/PNG/WebP、常见视频 |
| Kobushi | CSV/TSV、JSON/JSONL、SQLite、Markdown、TXT、PDF、HTML/XML、常见视频 |
| Memory ReAct | CSV/TSV、JSON、SQLite、Markdown、TXT、PDF、XLSX/XLSM、Parquet |
| Mamba Agent | CSV、JSON、SQLite、Markdown、TXT、PDF、常见视频 |

独立 MP3/WAV、DOCX 和旧式 XLS 没有被四套工具链共同可靠支持，因此未作为通用
上传格式开放。Mamba 的视频工具还存在单文件 100 MB 的上游限制。

## 使用方法

1. 打开 `DataAgent.exe`。
2. 在左侧选择一个或多个 Agent；Mamba 默认选中。
3. 点击 Query 输入区旁的上传按钮。
4. 查看支持格式，选择一个或多个本地文件。
5. 打开设置，填写 API 地址、模型名称和 API Key。
6. 输入针对这些文件的问题并提交。
7. 在对话区域查看输出，点击“完整 Trace”查看全部过程记录。

客户端会把一次上传自动转换成标准 DABench 任务：

```text
workspaces/<workspace-id>/input/task_1/
├── task.json       # 自然语言 Query
└── context/        # 上传的原文件
```

因此四套 Agent 接收到的仍然是统一的标准任务目录。

## 数据与 SQLite

桌面版的用户数据默认位于：

```text
%LOCALAPPDATA%\DataAgent\
├── data-agent.db
├── outputs/
├── logs/
├── traces/
├── workspaces/
└── webview/
```

`data-agent.db` 使用 SQLite WAL 模式，保存：

- 任务及其开始、结束、退出状态；
- 用户可见的逐行 Trace 日志；
- 工作区和上传文件元数据；
- 数据库 Schema 版本；
- 后续可迁移的应用设置。

上传原文件和运行输出不会作为 BLOB 写入数据库，而是保存在对应目录中。这样可以
让 SQLite 保持轻量，也更方便直接检查和备份文件。

API Key 当前不会持久化到 SQLite。Mamba 的上游配置加载器无法通过环境变量覆盖
Key，因此客户端只会生成运行期临时配置，并在子进程结束后删除。

## 从源码运行

要求：

- Python 3.11 或更高版本；
- `uv`；
- 四套 Agent 源码位于 `engines/`。

启动浏览器开发模式：

```powershell
python -m app.server --open
```

默认地址：

```text
http://127.0.0.1:8765
```

启动桌面开发模式：

```powershell
python desktop.py
```

## 构建 Windows 客户端

安装 PyInstaller 和 PySide6 后执行：

```powershell
.\build-windows.ps1
```

构建脚本会：

1. 用 PyInstaller 生成自包含的 `onedir` 桌面程序；
2. 打包 Qt WebEngine 和完整 Python 运行时，目标电脑不需要 Python 环境；
3. 复制四套 Agent 源码；
4. 将当前 `uv.exe` 放入便携目录；
5. 生成可作为 GitHub Release 附件的 ZIP。

构建产物：

```text
dist/
├── DataAgent/
│   ├── DataAgent.exe
│   ├── _internal/
│   ├── engines/
│   └── uv.exe
└── DataAgent-Windows-x64.zip
```

当前自包含构建约为：

- EXE 启动器：2 MB；
- 完整便携目录：552 MB；
- ZIP：217 MB。

体积主要来自随包提供的 Qt WebEngine/Chromium、Python 运行时和四套 Agent 源码。
这些文件确保客户端不依赖目标电脑上的 Python、Conda、.NET 或 WebView2。

## 测试

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- 四个 Agent 注册和 Kobushi 模式；
- 标准任务目录识别；
- 文件类型能力和上传；
- JSON、SQLite 与进程日志 Trace 聚合；
- SQLite Schema、任务日志持久化和中断恢复。

## 项目结构

```text
app/
├── database.py     # SQLite 数据层
├── desktop.py      # Qt WebEngine 桌面窗口与服务生命周期
├── engines.py      # 四套 Agent 适配器
├── manager.py      # 任务调度、日志、结果和 Trace
├── paths.py        # 开发/打包环境路径解析
├── server.py       # 本地 HTTP API 与静态资源服务
└── workspaces.py   # 上传能力与标准任务工作区

static/             # 桌面界面
engines/            # 四套上游 Agent
tests/              # 客户端测试
build-windows.ps1   # Windows 打包脚本
desktop.py          # 桌面版入口
```

## 当前边界

- 多个 Agent 的结果目前并列展示，不自动判断哪一个更好；
- 首次执行 Agent 时，`uv` 可能需要下载较多依赖；
- 视频 ASR 可能需要额外模型权重；
- 便携包尚未进行 Windows 代码签名，首次运行可能触发 SmartScreen 提示；
- GitHub 仓库中的各上游 Agent 保留其各自许可证和说明。
