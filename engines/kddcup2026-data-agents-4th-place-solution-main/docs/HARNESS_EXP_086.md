# Harness `exp_086_r1_official_params` 解説

> 旧 baseline (= 提出 v2 の元実験)。3-attempt union × Qwen3.5 公式 R1 sampling × classic preamble。
> n=2 mean λ0.5 = **0.7353** ± 0.0135 (= 50-task local)。LB 提出値は v2 として返答待ち中。

---

## 1. 設計の核

| 軸 | 値 |
|---|---|
| Model | qwen3.5-35b-a3b (vLLM、enable_thinking=true) |
| Attempts per task | **3** (= multiprocessing.spawn の並列 fork) |
| Attempt temperatures | **(0.6, 0.6, 0.7)** |
| Workers (outer) | 4 (= 4 タスク × 3 attempt = **12 vLLM streams**) |
| Per-task timeout | 900 秒 |
| Max steps | 32 |
| Min steps before answer | 4 (= "exploration first" guard) |
| Sampling parameters | T=0.6/0.7, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=**1.0**, max_tokens=32768 |

ベース論文: Qwen3.5 公式 model card "thinking precise" preset (= T=0.6/top_p=0.95/top_k=20/min_p=0/presence_penalty=0)。`presence_penalty` のみ 1.0 に上書き (= general preset の 1.5 と precise の 0 の中間、loop 抑制用)。

---

## 2. Preamble (= classic style)

agent が最初の user message として受け取るのは task の question + データ全部の "summary dump"。

### 構成 (`src/experiments/exp_086_r1_official_params/preamble.py`)

```
1. ## Note for agent (= データは全部含めた、ツール無しでも reasoning 可能)
2. # Task overview (task_id, difficulty, question)
3. # Workspace overview (= ファイル一覧 + サイズ)
4. # Glossary (= knowledge.md の `- **term**: def` 形式抽出)
5. ## [CSV] / [SQLite] / [JSON] / [DOC] 各セクション
6. # Knowledge (full) (= knowledge.md の本文)
```

### 各ファイル type の処理

| 型 | 小ファイル | 大ファイル |
|---|---|---|
| **CSV** | 全文 (50KB cap 内) | head 100 + tail 50 + random 50 行サンプリングで `to_string()` |
| **SQLite** | DDL + 各 table 先頭 100 行サンプル (`SELECT * LIMIT 100`) | 同 (= row 数で hard cap) |
| **JSON** | 全文 (20KB cap 内) | 先頭 30 records (= JSON array) or 切り詰め |
| **DOC** | 全文 (30KB cap 内) | 30KB で切り詰め |

### Budget 管理

総 budget = 150,000 tokens × 4 chars/token = **600,000 chars**。超過時は priority order で **ファイル type 丸ごと drop**:
```
drop order: json → csv → doc → sqlite → knowledge_full
```
= JSON が最初に消える、knowledge.md は最後まで残す。

**弱点:** drop は per-type 一括削除 = 1 つだけでも巨大 JSON があると全 JSON が消える (rich preamble 系で改善)。

---

## 3. Tools (= 10 個)

`src/experiments/exp_086_r1_official_params/tools/registry.py` で定義。

### 終端 tools (= 提出系)

| tool | 用途 |
|---|---|
| `answer` | columns + rows を直接渡す。手書き値投入用 |
| `answer_from_python` | Python コード実行、`answer_df` (DataFrame) or `answer_table` (dict) を抽出 |
| `answer_from_sql` | SQLite 1 query 結果を直接答えに変換 |

### 探索 tools

| tool | 用途 |
|---|---|
| `list_context` | context dir の再帰 ls |
| `inspect_sqlite_schema` | DDL + table 列リスト |
| `execute_context_sql` | read-only SQL クエリ実行 |
| `execute_python` | 任意 Python (= context dir = working dir) |
| `read_csv` | CSV preview (max_rows) |
| `read_json` | JSON preview (max_chars) |
| `read_doc` | text/md preview (max_chars) |

`execute_python` はほぼ "なんでも箱" — pandas / sqlite3 / json / re 全部利用可能。

---

## 4. System prompt rules (= 17 ルール)

`src/experiments/exp_086_r1_official_params/prompt.py:REACT_SYSTEM_PROMPT`。要約:

| Rule | 内容 |
|---|---|
| 1 | 1st thought に `Answer plan: column_count=N, per_column=[...], row_count=M` を必須 |
| 2-3 | tools 使う、観察済情報のみで答える |
| 4-5 | answer ツールで終了、planned column_count を超えない |
| 6 | answer 前に plan を再唱 + column/row_count 一致確認 |
| 7-9 | JSON 出力 format (= ```json fenced + `thought, action, action_input` keys) |
| 10 | superlative (lowest/highest 等) は LIMIT 1 ではなく **filter-back** (= `WHERE col = (SELECT MAX(col) ...)`) で ties 取る |
| 11 | SQL 書く前に `inspect_sqlite_schema` で table/column 名確認 |
| 12 | SQL の `SELECT *` 禁止、explicit 列指定 |
| 13 | `__error__` 観察時、原因を 1 文で書いてから次 action |
| 14 | SQL の ROUND() は最外 SELECT のみ (= 中間 round 禁止) |
| 16 | 双方向 edge (A→B, B→A) のような重複 row 注意、sorted-pair で COUNT DISTINCT |
| 17 | answer 前に最低 4 探索 step 必須 (= min_steps guard) |

**Rule 1 の row_count 宣言** = 後の exp_109 で問題視されて撤廃される (= 質問段階では確定不能のため "row_count=1" guess が tied results を切り捨てる害)。

---

## 5. ReAct loop (`agent.py`)

```
state = AgentRuntimeState()
for step in range(1, 33):
    raw = model.complete([system, user(preamble+task), *history])
    parsed = parse_model_step(raw)  # JSON: {thought, action, action_input}
    if step <= 4 and action in {answer, answer_from_*}:
        observation = "Too early to answer"
    else:
        observation = tools.execute(action, action_input)
    state.steps.append(StepRecord(...))
    if action in ANSWER_TOOLS and observation.ok:
        break
```

= 各 step で 1 thought + 1 action + 1 observation。最大 32 step。

---

## 6. Multi-attempt union (`runner.py:_signature_majority_merge`)

各タスクで 3 attempt を **multiprocessing.spawn** で並列実行 (= 別 process)。各 attempt は独立した temperature。

### Merge ロジック (= union, k=1)

```python
for ans in [attempt1.answer, attempt2.answer, attempt3.answer]:
    for col in ans.columns:
        sig = tuple(sorted(_normalize_value(v) for v in col_values))
        if sig appears in >= 1 attempt:
            keep this column
# All kept columns are placed side by side. If row counts differ, pad with None.
```

= 各 attempt の各列について **値の sorted multiset** を signature として、1 attempt 以上で出現したら採用。

### ⚠️ Padding バグ (= exp_086 の弱点、後の exp_093 で fix)

異なる長さの列をそろえる際 `max_len` まで `None` で埋める。CSV に書き出すと `None` → `""` で signature が破壊され、本来正しい列も recall=0 に。実際 task_11 / task_25 / task_259 で発火し、全体で **-0.05〜-0.10 の隠れ regression** を起こしていた可能性あり (= rich preamble + 3-att 組み合わせで顕在化、= exp_103 で実証)。

---

## 7. データフロー (= 1 タスク終了まで)

```
task_id
  → load preamble (= ファイル全 dump サマリ)
  → 3 multiprocessing fork (T=0.6/0.6/0.7)
       → 各 fork: ReAct loop (max 32 step)
       → answer tool 呼び出しで stop
  → 3 attempt の answer を _signature_majority_merge
  → merged answer を prediction.csv に書き出し
  → eval (= gold.csv との column-by-column value-multiset match)
```

per-task elapsed: ~5.3 min avg (= 多並列の per-attempt latency が支配的、3 attempts × 平均 4 step × 5-10 sec/step + python 実行)。

---

## 8. 実績と限界

### 強み

- **Robust baseline**: 50 task 中 17 が 90%+ perfect (= always-solved)、15 が 50-90% (often-solved)
- **3-attempt union の +0.04 boost**: 単 attempt 比 +0.04 score gain (vs single-att 同 preamble)
- **Filter-back rule (Rule 10)** で superlative ties 救済
- **Min steps guard (Rule 17)** で early answer による失敗削減

### 弱み (exp_109 で fix される対象)

- **Padding バグ** (= 行数違う attempt を None pad → CSV 書き出しで signature 破壊)
- **Row_count の predeclaration** (Rule 1) が tied gold tasks (task_25 / task_180 等) を切り捨て
- **Per-file caps** (50KB CSV / 20KB JSON) で大ファイル task に schema 全体像が伝わらず → task_330 (279MB CSV)、task_249 (166MB JSON) で失敗多発
- **Rule 6 の column_count check** が "Plan satisfied" の自己申告で形骸化 → wext (extra columns) 削減効果薄

### スコア

n=2 mean λ0.5 = **0.7353** ± 0.0135 (clean、run_002=0.7449 / run_003=0.7258)。LB v2 提出値は返答待ち中。

---

## 9. 関連ファイル

```
src/experiments/exp_086_r1_official_params/
├── agent.py        # ReAct loop
├── config.py       # AppConfig (max_steps=32, min_steps=4, max_workers=4, task_timeout=900)
├── config.yaml     # YAML 上書き
├── preamble.py     # classic preamble (= 600KB budget, 5-tier drop order)
├── prompt.py       # 17 ルール system prompt + 5 例示
├── runner.py       # 3-attempt fork + signature_majority_merge + run-benchmark CLI
├── runtime.py      # AgentRuntimeState
├── run.py          # CLI entry (run-task / run-benchmark)
└── tools/
    ├── registry.py # 10 tools 定義
    ├── filesystem.py
    ├── python_exec.py
    └── sqlite.py
```
