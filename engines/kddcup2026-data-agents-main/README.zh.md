<div align="center">

# kddcup2026-data-agents — team1438

> 🌐 **Language**: [English](README.md) · [한국어](README.ko.md) · **中文**

KDD Cup 2026 **DataAgent-Bench** Leaderboard 赛道挑战的 ReAct 智能体框架。

[![竞赛](https://img.shields.io/badge/KDD%20Cup%202026-DataAgent--Bench-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![赛道](https://img.shields.io/badge/Track-Leaderboard-f59e0b?style=for-the-badge&labelColor=0f172a)](https://dataagent.top/rules)
[![队伍](https://img.shields.io/badge/Team%20ID-team1438-be123c?style=for-the-badge&labelColor=0f172a)](#)

</div>

> 一个**单一 ReAct 智能体**,接收自然语言分析问题,审查上下文 (CSV / SQLite / JSON / Markdown / PDF · 图像 · Excel · Parquet — Phase 2 防御用) 后产出 `prediction.csv`。在主办方挂载的 Docker 容器中运行,LLM 仅使用主办方强制的 `qwen3.5-35b-a3b`。

---

## 0. 最终结果 — Phase 1 收官 (2026-07)

**team1438 的 Phase 1 已结束,本仓库作为竞赛存档公开。** [Phase 1 官方排行榜](https://dataagent.top/leaderboard):

| 指标 | team1438 |
|---|---|
| A-board (2 小时 wall-clock, 57 task) | **0.3886** |
| B-board (12 小时 wall-clock, 324 task) | **0.4349** |
| **Phase 1 最终分** (按 task 数加权 ≈ 0.15·A + 0.85·B) | **0.4279** |
| 最终排名 | 约 300 队中 **第 137 名** |
| Top-60 晋级线 (Phase 2 资格) | 0.5209 — **未晋级** |

分数轨迹讲述了这个阶段的真实故事:**v1** 评测直接失败 (arm64 manifest — 当时在 DGX 上原生构建),**v2** 修复 linux/amd64 交叉构建后拿到首个分数 (**0.3386**),**v4** 因 harness 相对 2 小时 A-board 预算 naive 需要 ~32 小时而被 SIGTERM 截断,*倒退*到 **0.2281**,**v6 → v8** 几乎纯靠 wall-clock 工程 (worker 并行、按难度的 task 超时、压力下折半预算的级联 governor) 收复到 **0.3509 → 0.3886**。唯一一次 B-board 提交 (同一 v8 镜像,完整 12 小时预算) 得分 **0.4349**。

留给未来队伍的经验:**在硬 wall-clock 基准上,调度胜过模型的聪明。** v4 之后追回的每一分都来自*完成更多 task*,而不是答得更好 — 单 task 答案质量 (public-set mock ≈ 0.69–0.73) 从来不是瓶颈,时间才是。

本节以下的所有内容按竞赛期间的工作记录原样保留 (含日期、预测值与"当前状态")。逐次提交的完整历史见 [docs/SUBMISSION_LOG.zh.md](docs/SUBMISSION_LOG.zh.md)。

---

## 1. 一句话总结

| 项目 | 值 |
|---|---|
| 竞赛 | [KDD Cup 2026 — DataAgent-Bench](https://dataagent.top) |
| 赛道 | Leaderboard (放弃 Creative 赛道) |
| 队伍 ID | `team1438` |
| 评测 LLM | `qwen3.5-35b-a3b` (主办方强制,环境变量注入) |
| **模型政策 (自定)** | **仅 qwen** — 不使用任何辅助 LLM/embedding/vision/web API。config 默认空字符串,缺失 env 时显式报错 (阻止 silent fallback)。 |
| 计算资源 | 16 vCPU · 64 GB RAM · 无 GPU · **12 小时合计** · linux/amd64 |
| 挂载 | `/input` RO · `/output` RW · `/logs` RW |
| 网络 | 仅 `MODEL_API_URL` 可达 (`--network=eval_net`) |
| 提交 | Docker tar.gz ≤ 10 GB → Google Drive 共享 → 邮件主办方。每天 1 次 / Phase 1 共 30 次 |
| Phase 1 截止 | 2026-05-23 (AoE) |

完整规则见 `.claude/skills/kddcup-rules-*` 6 个或 [官方规则页](https://dataagent.top/rules)。

---

## 2. 当前状态 (2026-05-11)

| 阶段 | 状态 |
|---|---|
| Phase 1.0 substrate (knowledge.md 注入 + persistent IPython kernel + dataframe prepass) | ✅ 已合并 |
| Phase 2.0 substrate (JSON-mode probe + 难度感知 max_steps + wall-clock governor + parse-retry + plan-then-execute) | ✅ 已合并 |
| Phase 3 substrate (answer_validator + conditional terminal + doc auto-injection + column_ablation + name-equivalence) | ✅ 已合并 |
| §3.1–§3.5 (format dispatcher / size-aware streaming / hierarchical inspect_file / domain sanity) | ✅ 已合并 |
| 合规强化 (UV_OFFLINE=1, 3 重 linux/amd64 守卫, 可观测工具 `dabench inspect-trace` / `summarize-traces`) | ✅ 已合并 |
| **v3 agent 轮次 (G-1 retry hint, G-2 SelfConsistencyAgent k=3, G-3 knowledge.md 5000 字符 + question-keyword reorder)** | ✅ 已合并 |
| **v3 runtime 轮次 (H-1 multi-pass orchestrator + cross_run_vote `repeat_max=3 pass_safety_margin=1.1`, K-1 更紧 margin, L-1 tier-aware retry skip)** | ✅ 已合并 |
| **v3 tools 轮次 (H-3 streaming JSON: streaming_json_keys / count / aggregate)** | ✅ 已合并 |
| **v3 记忆层 (M-1~M-5 TaskShape + ShapePolicy + learnings.json + recorder + `dabench update-learnings`)** | ✅ 已合并 |
| **v3 错误模式记忆 (N-1 error_patterns.json + recorder cross-task aggregation, N-2 in-loop repeat-error guard, N-3 pre-flight task brief)** | ✅ 已合并 |
| **v3 构建 + 49-task 测量 + ship** | ▶︎ `team1438_v3.tar.gz` (0.38 GB) |

### 主办方 leaderboard 历史

| 版本 | 评测结果 | 备注 |
|---|---|---|
| v1 | 评测失败 | arm64 manifest 问题 (当时 DGX 原生构建) — 之后引入 3 重 amd64 守卫 |
| **v2** | **0.3386** | linux/amd64 交叉构建后首次成功评测。当前 baseline floor。 |
| **v3** | **已构建 + 已测量, 待评测** | sha256 `1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09` — 合并轮次 (G-1~G-3, H-1, H-3, K-1, L-1, M-1~M-5, N-1~N-3);mock 0.7254 single-pass (44 scored, 49-task 子集 31 perfect)。 |

### v3 测量 (公开 50-task mock_scorer, λ=0.10)

49 task 子集 (排除 task_418 hang;OneDrive 挂载 D-state 问题 — eval 环境无关)。

| 测量 | Perfect | Mean (scored) | **50-task 推算** | Wall-clock |
|---|---|---|---|---|
| v2 baseline floor | — | — | 0.3386 leaderboard | n/a |
| v3 single-pass (记忆层前) | 31 | 0.6986 | 0.6566 | 4911 s |
| **v3 single-pass (含记忆层, current)** | **31** | **0.7254** | **0.6384** | 5841 s |
| v3 multi-pass smoke (in-runner 2-pass voting) | — | 0.7036 | 0.6473 | 12426 s |

production 中 eval 容器使用 `repeat_max=3` multi-pass + voting — 严格优于上面 single-pass 数字。3 次独立运行 ablation 收敛到 0.6457,远比单次运行稳定。

### Leaderboard 推算

```
v2 baseline (实测):       0.3386
mock → leaderboard 差距:   约 −0.15 ~ −0.20 (假设 hidden 分布相似)
v3 期望 leaderboard:      0.45 ~ 0.55 (production multi-pass + voting)
相对 v2 floor 改进:        +0.11 ~ +0.17
```

### v3 制品

```
镜像             : team1438:v3 (linux/amd64, 0.38 GB)
Tarball          : submissions/team1438_v3.tar.gz (0.38 GB)
sha256           : 1bb11bac62010faf21ef16a5163d8bbdb258f7344a55c47ea42a80747db64a09
LLM endpoint     : http://<VLLM_HOST>:8000/v1 (DGX, qwen3.5-35b-a3b, max_ctx 262 144)
Workers          : ThreadPool max_workers=4, difficulty timeout(300/600/900/1200), wall_clock 12h
                   + multi-pass orchestrator (repeat_max=3, pass_safety_margin=1.1)
G-2 SC           : SelfConsistencyAgent k=3 temp=0.5 在 hard/extreme tier
记忆层           : memory/learnings.json + memory/error_patterns.json 内嵌镜像;
                   recorder 在每次 public-set run 后摄取 trace.json 并产生 delta
合规守卫         : UV_OFFLINE=1, FROM --platform=linux/amd64, build-time uname guard, manifest inspect,
                   _fallback_copy_pass (即使 voter 失败也保留 pass 0 → 不会比 single-pass 更差)
```

### 评测后打分

```bash
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/eval_full_v3/output \
    --gold        data/public/output \
    --input       data/public/input \
    --lambda-values 0.05 0.10 0.20
```

---

## 3. 评测环境 ↔ 我们容器 1:1 对照

主办方接收我们的 tarball 后逐字执行的 §runtime 命令:

```bash
docker run --rm \
  --network=eval_net --cpus=16 --memory=64g \
  -v /input:/input:ro \
  -v /output:/output:rw \
  -v /logs:/logs:rw \
  -e MODEL_API_URL=...  -e MODEL_API_KEY=...  -e MODEL_NAME=qwen3.5-35b-a3b \
  team1438:v<N>
```

| 规则 (skill) | 规则规范 | 我们的行为 | 验证依据 |
|---|---|---|---|
| **runtime §1** ENTRYPOINT | `team1438:v<N>` 自身可执行 | `uv run dabench run-benchmark --config configs/eval.yaml` | `docker inspect` |
| **runtime §2** /input RO | 修改即违规 | `resolve_context_path` 强制 sandbox | `grep` 命中数=0 |
| **runtime §2** /output 格式 | `/output/task_<id>/prediction.csv` | `flat_output_dir: true` → 直接写入 task 目录 | `eval.yaml` |
| **runtime §3** env 注入 (3 个) | 禁硬编码,强制 env | eval.yaml 中 model/api_base/api_key 为空字符串 | `grep -rE "api_key\s*=" src/` 命中数=0 |
| **runtime §4** 网络隔离 | 仅允许 `MODEL_API_URL` | 0 次 `requests`/`urllib`/`httpx`/外部 LLM 调用 | grep 空 |
| **compute §1** 强制 linux/amd64 | 仅 x86-64 | 3 重守卫: ① `FROM --platform=linux/amd64` ② Dockerfile 构建时 `uname -m` 守卫 ③ 构建后 `docker image inspect` 架构验证 | `docker image inspect team1438:v3` → `amd64 linux` |
| **compute §2** 12h 合计 | 非 per-task | `wall_clock_budget_seconds: 43200` + 8h 触发 downgrade governor | `runner.py:_governor_should_engage` |
| **compute §5** SIGTERM 30s 宽限 | 30 秒内 partial flush | `_install_sigterm_trap` (优雅) | `runner.py:_install_sigterm_trap` |
| **model §1** 强制 qwen3.5-35b-a3b | 禁其他 LLM 主求解器 | 唯一 `OpenAIModelAdapter`, 0 个辅助 LLM | pyproject 依赖 |
| **model §3** 禁 CPU 上小型 LLM | 容器内禁其它 LLM 权重 | 镜像 0.38 GB (无模型权重) | `du -sh` |
| **submission §1** 命名 | `<team_id>:v<N>` + `<team_id>_v<N>.tar.gz` | `team1438:v3` + `submissions/team1438_v3.tar.gz` | `ls submissions/` |
| **submission §2** ≤ 10 GB | 压缩后限制 | 0.38 GB | `du -h submissions/*.tar.gz` |
| **submission §4** 1/天 · 30/Phase 1 | 频率限制 | 累计追踪 | [`docs/SUBMISSION_LOG.zh.md`](docs/SUBMISSION_LOG.zh.md) |
| **output §1-3** CSV 格式 | UTF-8 + 1 行表头 + 列序无关 | `_answer` handler → `normalize_answer_table` → flat path | `scoring/normalize.py` |
| **output §4-9** 归一化 | HALF_UP / ISO / strip | 显式使用 `decimal.Decimal` + `ROUND_HALF_UP` | `normalize.py:_normalize_numeric` |
| **output §10** name-equivalence | `FN+LN` ↔ `FN LN` | `mock_scorer.py` 3-phase matching | `mock_scorer.py:match_columns_with_name_equivalence` |
| **prohibitions §1** 外部互联网绕路 | 严禁 | 代码 grep 0 命中 + 假定 `--network=eval_net` | grep |
| **prohibitions §3** /input 修改 / env 篡改 | 严禁 | 0 次 `os.environ[MODEL_*] = …` | grep 空 |
| **prohibitions §5** 探测 infra | 禁对抗性提交 | 每次提交都是真实改进 | [`docs/SUBMISSION_LOG.zh.md`](docs/SUBMISSION_LOG.zh.md) |

### 残留差距 — 无法消除的部分

| 项目 | 评测环境 | 我们环境 | 影响 |
|---|---|---|---|
| 宿主机 | 主办方 Linux x86-64 native | macOS Apple Silicon → QEMU 模拟 | 速度更慢,结果一致 |
| `MODEL_API_URL` | 主办方内部 qwen endpoint | DGX 自托管 vLLM | 模型 ID 一致 |
| 并发请求限额 | 主办方 rate limit 未知 | DGX vLLM 限额 | 评测时若被限速会触发 OpenAIAdapter retry |
| 网络 | 仅 `eval_net` 可达 | macOS → DGX LAN | 我们代码 0 次外部调用,无关 |

**结论:** v3 容器 100% 满足规则规范。只有评测端点的 max_context / rate limit 需要从 leaderboard 响应中验证。

---

## 4. 快速开始 — macOS dev + DGX vLLM 分离工作流

team1438 推荐布局:
- **DGX (`<VLLM_HOST>`, aarch64)**:vLLM (`qwen3.5-35b-a3b`) **仅服务**。
- **macOS local (Apple Silicon arm64)**:构建 / 测试 / 提交打包。`linux/amd64` 交叉构建匹配评测环境。

```bash
# 0. (DGX, 一次) 启动 vLLM 服务
ssh <user>@<VLLM_HOST> 'cd <repo-dir> && bash scripts/serve_qwen_docker.sh'

# 1. (macOS) 依赖 + 环境变量
uv sync --extra dev
export MODEL_API_URL=http://<VLLM_HOST>:8000/v1
export MODEL_API_KEY=EMPTY
export MODEL_NAME=qwen3.5-35b-a3b

# 2. (macOS) 数据可见性
uv run dabench status        --config configs/local.yaml
uv run dabench inspect-task task_<id> --config configs/local.yaml

# 3. (macOS) host run — 单 / 全部 / holdout
uv run dabench run-task     task_<id> --config configs/local.yaml
uv run dabench run-benchmark          --config configs/local.yaml
uv run dabench run-benchmark          --config configs/local.yaml --task-set data/public/holdout_ids.txt

# 4. (macOS) 打分
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/runs/<run_id> --gold data/public/output \
    --input data/public/input --lambda-values 0.05 0.10 0.20 --ablate

# 5. (macOS) 提交打包 — linux/amd64 交叉构建
bash scripts/build_submission.sh v<N>
docker tag dabench:v<N> team1438:v<N> && \
  docker save team1438:v<N> | gzip --best > submissions/team1438_v<N>.tar.gz
shasum -a 256 submissions/team1438_v<N>.tar.gz

# 6. (可选) 容器复现验证
DABENCH_LAMBDAS="0.05 0.10 0.20" \
  MODEL_API_URL=http://<VLLM_HOST>:8000/v1 \
  bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
```

### 评测环境镜像 — 50 task 整体容器执行

```bash
mkdir -p artifacts/eval_full_v3/{output,logs}
docker run --rm -d --name team1438_v3_full \
    --platform linux/amd64 \
    -e MODEL_API_URL="http://<VLLM_HOST>:8000/v1" \
    -e MODEL_API_KEY="EMPTY" \
    -e MODEL_NAME="qwen3.5-35b-a3b" \
    -v "$(pwd)/data/public/input:/input:ro" \
    -v "$(pwd)/artifacts/eval_full_v3/output:/output" \
    -v "$(pwd)/artifacts/eval_full_v3/logs:/logs" \
    team1438:v3
```

此命令的 **挂载、env、platform 与 §runtime 完全一致**。

---

## 5. 7-Layer 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 7 — Submission   Docker tarball, Drive, email — team1438:v<N>│
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6 — Scoring      normalize + column-signature multiset        │
│                        + name-equivalence (rules §10)               │
│                        + cross_run_vote (column-multiset majority)  │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 5 — Tools        filesystem · sqlite · python_kernel          │
│                        dataframe_describe/head · _answer            │
│                        validator (conditional terminal)             │
│                        format dispatcher (PDF/Excel/Parquet/Img)    │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4 — Agent        ReAct loop · JSON contract · parse-retry     │
│                        difficulty-aware max_steps · plan-execute    │
│                        SelfConsistencyAgent (k=3, hard/extreme)     │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3 — Runtime      per-task subprocess · ThreadPool batches     │
│                        wall-clock governor · SIGTERM trap           │
│                        run_benchmark_with_passes (multi-pass H-1)   │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2 — Model        OpenAIModelAdapter + JSON-mode probe         │
│                        env-injected MODEL_API_URL/KEY/NAME          │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 1 — Infra        Docker · 16 vCPU / 64 GB / 12h · /input RO  │
└─────────────────────────────────────────────────────────────────────┘
```

层级职责一图概览见 [`docs/SYSTEM_ARCHITECTURE.zh.md`](docs/SYSTEM_ARCHITECTURE.zh.md)。数据/执行流 mermaid 8+ 见 [`docs/SYSTEM_FLOW.zh.md`](docs/SYSTEM_FLOW.zh.md)。组件依赖图 + 外部通信策略 + 构建管线 + 轮次变更日志见 [`docs/HARNESS_STRUCTURE.zh.md`](docs/HARNESS_STRUCTURE.zh.md) — **活文档**。

---

## 6. 目录结构

```
.
├── src/data_agent_baseline/
│   ├── cli.py                    # Typer 4 子命令
│   ├── config.py                 # env > YAML > default 叠加
│   ├── benchmark/                # DABenchPublicDataset + PublicTask schema
│   ├── agents/
│   │   ├── prompt.py             # system / task / observation prompt + knowledge.md (5000 字符, question keyword reorder) & doc/*.md auto-inject
│   │   ├── react.py              # ReActAgent.run + parse-retry + action_input coercion + difficulty max_steps
│   │   ├── self_consistency.py   # SelfConsistencyAgent — hard/extreme 上 k=3 voting (G-2)
│   │   ├── model.py              # OpenAIModelAdapter (JSON-mode probe + fallback)
│   │   └── runtime.py            # StepRecord / AgentRuntimeState / AgentRunResult
│   ├── tools/
│   │   ├── registry.py           # 工具目录 + 条件终结的 _answer (16 个工具)
│   │   ├── filesystem.py         # csv/json/doc/pdf/excel/parquet/image/archive + dataframe_* + inspect_file
│   │   ├── sqlite.py             # 只读 SQL + schema inspect
│   │   ├── python_exec.py        # 临时 subprocess fallback
│   │   └── python_kernel.py      # 持久化 IPython kernel (默认)
│   ├── scoring/
│   │   ├── normalize.py          # null / 2dp HALF_UP / ISO date / strip
│   │   ├── mock_scorer.py        # 3-phase matching 含规则 §10 name-equivalence
│   │   ├── answer_validator.py   # 启发式 ValidationReport (§3.5 domain sanity)
│   │   ├── column_ablation.py    # λ-agreement 诊断
│   │   ├── cross_run_vote.py     # N 个 benchmark root 之间的 column-multiset majority (H-1)
│   │   └── holdout.py            # blake2b 确定性 80/20 拆分
│   └── run/runner.py             # subprocess 隔离 + ThreadPool + governor + RuntimeLogger + SIGTERM trap
│                                 # + run_benchmark_with_passes (multi-pass orchestrator, H-1)
├── configs/
│   ├── eval.yaml                     # Docker submission (env-overridable, 空字符串默认, repeat_max=3)
│   └── local.yaml                    # 开发用 (假定自托管 vLLM)
├── docker/vllm-qwen35/                  # 自托管 vLLM (DGX)
├── scripts/
│   ├── build_submission.sh           # docker buildx (linux/amd64) + gzip + sha256 + 10GB 验证
│   ├── local_eval.sh                 # 容器复现 + mock_scorer + 渲染报告
│   ├── serve_qwen_docker.sh          # vLLM compose wrapper
│   ├── probe_qwen.py                 # endpoint 能力测量
│   └── render_score_report.py        # JSON → markdown 报告
├── docs/
│   ├── ARCHITECTURE.{md,ko.md,zh.md}            # 代码 · 运行时指南
│   ├── SYSTEM_ARCHITECTURE.{md,ko.md,zh.md}     # 一张图系统架构 (v3 轮次)
│   ├── SYSTEM_FLOW.{md,ko.md,zh.md}             # 8+ mermaid (数据/执行流)
│   ├── HARNESS_STRUCTURE.{md,ko.md,zh.md}       # 7-Layer + 依赖 + 变更日志
│   ├── DATA_ANALYSIS.{md,ko.md,zh.md}           # 50-task 统计 + 失败模式
│   ├── SUBMISSION_LOG.{md,ko.md,zh.md}          # 提交历史 + 预算追踪
│   └── qwen_endpoint_capabilities.{md,ko.md,zh.md} # 自托管 vLLM probe
├── .claude/skills/                   # 12 个 KDD Cup skills
├── Dockerfile                        # 评测容器 (强制 linux/amd64)
├── pyproject.toml                    # data-agent-baseline package
└── CLAUDE.{md,ko.md,zh.md}           # AI 助手运维手册
```

`tests/`、`data/`、`artifacts/`、`submissions/` 在 gitignore 中。`docs/*` 中只有英文/韩文/中文三语版本被白名单。

---

## 7. 提交流程 (摘要)

```
1. 构建            bash scripts/build_submission.sh v<N>
2. retag           docker tag dabench:v<N> team1438:v<N> &&
                   docker save team1438:v<N> | gzip --best > submissions/team1438_v<N>.tar.gz
3. 复现验证        DABENCH_LAMBDAS="0.05 0.10 0.20" \
                   bash scripts/local_eval.sh v<N> data/public/holdout_ids.txt
                   → 与 host 在 ±0.005 内一致
4. sha256 校验     shasum -a 256 submissions/team1438_v<N>.tar.gz
5. Drive 上传       "Anyone with the link can view" 权限
6. 主办方邮件      team_id + version + Drive URL + sha256
7. 更新日志        docs/SUBMISSION_LOG.zh.md budget tracker (Used N→N+1)
```

规则:**每天 1 次**, **Phase 1 总共 30 次**, 在上一份评测完成前不发新的提交。

---

## 8. 开发命令

```bash
uv run pytest                                   # 单元测试
uv run pytest tests/path/to/test_x.py::test_y   # 单测
uv run ruff check src tests                     # lint (line-length 100, py310)

# 重新生成 holdout split (数据更新时)
uv run python -m data_agent_baseline.scoring.holdout \
    --dataset-root data/public/input --output-dir data/public

# 打分诊断 (附 column ablation)
uv run python -m data_agent_baseline.scoring.mock_scorer \
    --predictions artifacts/eval_full_v3/output \
    --gold data/public/output --input data/public/input \
    --lambda-values 0.05 0.10 0.20 --ablate
```

完整开发流程见 [`CLAUDE.zh.md`](CLAUDE.zh.md)。

---

## 9. 依赖 / 运行时

- Python ≥ 3.10 (uv 安装 3.10/3.11)
- Docker 24+ + buildx (linux/amd64 交叉构建)
- (可选) NVIDIA GPU + Docker compose — 自托管 vLLM 时
- 评测容器内所有 deps 通过 `uv.lock` 确定性安装

依赖见 `pyproject.toml` `[project.dependencies]`。代表项: pandas, numpy, openai, polars, pyarrow, pypdf, openpyxl, Pillow, IPython。

---

## 10. 仓库来历

[HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit) 的 fork。starter kit 仅提供 Phase 0 baseline (分数 ≈ 0),我们几乎全面重写了 `src/data_agent_baseline/`:

- 归一化 + mock_scorer + holdout split (Phase 0 ship gate)
- knowledge.md/doc auto-injection, persistent kernel, dataframe prepass (Phase 1.0)
- JSON-mode probe, 难度 max_steps, wall-clock governor, parse-retry, plan-then-execute (Phase 2.0)
- answer_validator, conditional terminal, name-equivalence, column_ablation (Phase 3 substrate)
- §3.1 format dispatcher (PDF/Excel/Parquet/Image/Archive readers)
- §3.2 size-aware streaming + §3.3 hierarchical inspect_file + §3.5 domain sanity heuristics
- **Runtime + 合规强化:** column-count ratio>1.5× blocking, difficulty-aware task_timeout (300/600/900/1200), OpenAIAdapter retry 1→3 with backoff, plan-then-execute prompt 强化, hard/extreme max_steps 24/32, `UV_OFFLINE=1` (匹配 `--network=eval_net`)
- **可观测工具:** `dabench inspect-trace` (单一 trace 彩色 step view), `dabench summarize-traces` (50-task 分类 + diff)
- **法医分析种子的 prompt cue:** 0-row trap, source-schema lock, pre-answer self-verify, plural-cue under-emission validator
- **v3 agent + runtime 轮次 (G / H / K / L):**
  - G-1: 把 `Request timed out` 加入 first-step transient hint (回收 endpoint overload run 中 21 次 timeout)
  - G-2: `SelfConsistencyAgent` 在 hard/extreme tier 上 k=3 column-signature voting (temperature 0.5)
  - G-3: knowledge.md cap 3000 → 5000 字符 + question keyword H2/H3 reorder
  - **H-1: multi-pass orchestrator + cross_run_vote** — 容器把同一 task set 跑最多 `repeat_max=3` 次, 按 column-multiset majority voting。budget guard 保证不会比 single-pass 更差 (`pass_safety_margin=1.1`, `_fallback_copy_pass`)。
  - **H-3: streaming JSON 工具** — `streaming_json_keys / count / aggregate` 通过 `ijson` 流式处理大 JSON, 无 size cap。
  - K-1 / L-1: 更紧的 `pass_safety_margin`, hard/extreme subprocess timeout 不再 retry (tier-aware)。
- **v3 记忆层轮次 (M / N):**
  - M-1~M-5: `memory/` (TaskShape classifier, ShapePolicy resolver, bundled `learnings.json`, recorder, `dabench update-learnings` CLI)。
  - **N-1: 错误模式记忆** — bundled `error_patterns.json` 跨任务签名聚合;任务开始时把匹配的 advisory 注入 prompt。
  - N-2: in-loop 重复错误熔断 — 连续 2 次同签名错误时在下一 model turn 前追加一行 "do NOT retry the same approach" cue。
  - N-3: pre-flight task brief — 对 `context/` 做确定性只读扫描 (CSV peek, SQLite table, join key 候选), `Pre-flight task brief:` 块嵌入 user prompt。
- **法医纪律:** 轮次中检讨过的两个 prompt cue ("simulate the grader" / 严格 0-row override) 在多采样 ablation 后被驳回 — 单次运行比较误导了诊断, variance 才是真正原因。今后所有 patch 必须通过多采样 ablation。

详细变更见 `git log --oneline`。

---

## 11. 许可证 / 来源

本仓库始于 [HKUSTDial/kddcup2026-data-agents-starter-kit](https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit) 的 fork。上游 starter kit **没有明确的许可证文件**;原始 starter-kit 部分的权利归上游作者 (HKUST DIAL) 所有,本 fork 对这些部分遵循同样条款。`src/data_agent_baseline/` 树在 Phase 1 期间由 team1438 几乎全面重写。

本仓库在 Phase 1 结束 (最终评审 2026-07-14 截止) 后作为存档与复盘公开。竞赛期间全程私有,符合禁止跨队分享的规则。`.claude/skills/kddcup-rules-*` 包是对[官方规则](https://dataagent.top/rules)的摘要/转述,请以官方页面为准。
