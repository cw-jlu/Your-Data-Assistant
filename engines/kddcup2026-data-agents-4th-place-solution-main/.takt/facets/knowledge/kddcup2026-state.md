# KDD Cup 2026 DataAgent-Bench — Project State (stable invariants)

## 競技概要
- 公式: https://dataagent.top / https://dataagent.top/rules
- 締切: **2026-05-25** EoD AoE (Phase 1 final)
- 提出: Docker image を Google Drive 経由でメール送付 (1/day, 計 30 回まで)
- Hidden test: A-board (~60 tasks, 2h wall-clock) + B-board (~320 tasks, 12h total)

## メトリクス (CRITICAL — 全 persona が読むこと)

### 公式スコア式
```
Score_per_task = max(0, Recall − λ · (Extra Columns / Predicted Columns))
Final Score    = mean over tasks
```

- **Recall** = `(gold列のうち、pred列の値signatureが一致する列の数) / (gold列の数)`
- **Extra Columns** = `(pred列の数) - (gold列にマッチした列の数)`
- λ ∈ {0.0, 0.1, 0.3, 0.5, 1.0} を全部出している (`kobushi_core/eval/csv_compare.py`)、本番 λ は非開示

### 列マッチング = 値 signature 一致 (列名は完全無視)

公式 eval (`_column_signature` in `kobushi_core/eval/csv_compare.py`):
```python
def _column_signature(vector: list[str]) -> tuple[str, ...]:
    return tuple(sorted(vector))   # = sorted multiset of normalized cell values
```

各セルは正規化:
- **数値**: `Decimal.quantize(0.01, ROUND_HALF_UP)` → `"1.234" → "1.23"`、`"1.235" → "1.24"`
- **null variants**: `"", "null", "none", "nan", "nat", "<na>"` (case-insensitive) → `""`
- **日時**: ISO 形式に正規化
- **その他文字列**: strip のみ (改行除去)、**lowercase はしない** ← 注意、case sensitive

→ 2 つの列のセル値ベクトルを正規化して `tuple(sorted(...))` した結果が **完全一致**したら match。

**具体例**:
```
gold col:  ["Tokyo", "Osaka", "Tokyo"]      → signature ("Osaka", "Tokyo", "Tokyo")
pred col1: ["Tokyo", "Tokyo", "Osaka"]      → signature ("Osaka", "Tokyo", "Tokyo") ✅ MATCH (順序不問)
pred col2: ["tokyo", "osaka", "tokyo"]      → signature ("osaka", "tokyo", "tokyo") ❌ case mismatch
pred col3: ["Tokyo", "Osaka"]               → signature ("Osaka", "Tokyo") ❌ multiset 違う (重複も区別)
```

**重要な含意**:
1. **列名は完全無視** → "event_name" でも "name" でも値が一致すれば match。リネーム提案は無価値
2. **行順は無視 (sorted)** → ORDER BY 不要 (= deterministic 化のためなら使う価値はあるが、score には不要)
3. **重複個数は match 対象** → multiset。`["A", "A"]` と `["A"]` は別物
4. **数値は 2dp** → `1.234` も `1.235` も `1.23` 扱い (ROUND_HALF_UP)
5. **大文字小文字は厳格** → 文字列値は元の case 維持で出すこと
6. **null 周辺は緩い** → `None` / `"null"` / `"NaN"` 全部 `""` に揃う

- 分割名 ⇄ 結合名のマッチング (e.g. first+last ↔ full name) も公式で許容 (`_match_split_name_columns` の特殊処理)

### 戦略的含意 (実験設計時の判断軸)

1. **Recall = 0 なら何しても 0**: ペナルティをいかに抑えても recall ゼロ救済なし。
   → **どの軸も最優先は「正答列を出すこと」**。「extra を恐れて推測しない」は誤戦略
2. **Recall=1.0 なら下限 0.5 (λ=0.5想定)**: 余分列があってもスコア > 0.5 確保。
   → **多めに出して当てるほうが、少なく出して外すよりマシ**
3. **Extra ペナルティの上限**: extras_ratio ∈ [0,1] なので λ=0.5 で最大 −0.5。
   → 1 タスクの最低スコア ≠ 大爆発しない
4. **列名ガン無視**: `event_name` だろうと `name` だろうと値ベクトル一致なら同等。
   → 列名の正確性に労力を割くのは無駄
5. **目的関数のヒエラルキー** (高 → 低):
   1. Recall を上げる (ゼロをノンゼロに、partial を full に)
   2. Missing を減らす (timeout / 空回答救済)
   3. Extras を減らす (与えるなら最後に)

### ノイズ vs 信号の判定 — 計算済み統計値を使う (CRITICAL)

**重要**: ノイズ床は固定値ではない。実験ごとに API 条件・workers 設定・per-task variance で σ が変わる。
ただし **3 runs 以上 replicate していれば mean / std / 95% CI は計算済み** (推定ではなく決定論的計算)。

#### データソース (全部既に書かれてる)

1. **per-exp 統計**: `artifacts/replications/<exp>/summary_<ts>.json`
   - `lambda_0_5.mean`, `.median`, `.std`, `.ci_95`, `.ci_95_half_width`
   - `runs[]` 配列に per-run の生データ
   - `replicate_bench.sh` が aggregate して書き込む

2. **全候補ランキング**: `python3 scripts/aggregate_replications.py`
   - 全 replicated exp を mean 降順で表示
   - exp_040 mean を baseline として Δmean ベース verdict を自動付与
   - 1 行で全比較が完結する

→ deep-analyzer / code-proposer は **このスクリプトを実行して結果を読むだけ**。
   自分で statistics 計算する必要はない、既に computed。

#### 判定ルール (計算済みの値で相対比較)

- **真の win**: `Δmean > 2 × max(std_self, std_base)` AND 95% CI 非重複
- **likely win**: `Δmean > 1 × std` AND CI が baseline mean を含まない
- **noise**: `|Δmean| < 1 × std` → 信号として扱わない
- **regression**: `Δmean < -2 × std`

`aggregate_replications.py` の出力末尾に既に verdict (confirmed_win / likely_win / noise / regression)
が表示される。これをそのまま使う。

#### 1 run スコアの扱い

- replication が無い exp は **判定不能**。1 run の数字だけで "効いた / 効かない" を断定するのは禁止
- 過去事例: exp_068 1-run 0.7367 は元々 "≥2σ outlier" と誤判定したが、n=4 で計算すると
  mean=0.6923, std=0.0319 で **+1.4σ の通常範囲**だった
  → **1 run のスコア 1 つから断定しない**。replicate しろ

#### 観測されている std レンジ (参考)
- exp_068 n=4: std = 0.0319
- exp_067 n=3: std = 0.0271
- exp_061 n=3: std = 0.0278
- exp_040 n=3: std = 0.0702 ⚠ (= API 汚染で拡大、clean runs のみなら半減)
- 典型 clean range: 0.02-0.04、汚染あり: 0.05+

## 制約 (動かせない前提)
- モデル固定: **`qwen3.5-35b-a3b`** (Qwen3.5 MoE, 3B active params)
- 学習・蒸留・FT 不可
- 推論 API only: OpenAI 互換、CPU 実行のみ (16 vCPU / 64 GB / no GPU)
- Container は **モデル API のみ** ネットワーク到達可。外部完全遮断
- per-task 時間予算 ≒ A-board 120s / B-board 135s が現実的上限

## 🚫 厳守ルール: テストデータ・リーク禁止 (CRITICAL — 全 exp に永続適用)

**「リーク」の正しい定義 (狭義 → 広義の 3 段階)**:

### Tier 1: verbatim リーク (確実に NG)
公開 50 タスク (`data/public/input/task_*`) の question / gold / context データを system prompt や preamble や fewshot 例にそのままコピー。
- ✗ `task.json` の question を fewshot に貼る
- ✗ `gold.csv` の値を fewshot の "Gold output" に書く
- ✗ 公開タスクの SQL クエリをそのまま例示として system prompt に入れる

### Tier 2: 構造リーク (NG: 結果は信用できない)
fewshot 例が **公開タスクと類似スキーマ・類似パターン**で書かれている場合、たとえ DB 名や列名を架空にしても **公開 50 上での score 向上はリーク経由**の可能性が高い。
- ✗ 「合成例」と称して `event(event_id, event_name, type)` + `member(member_id, ...)` のような **公開 student_club DB に酷似したスキーマ**で書く
- ✗ 「filter-back SQL pattern」「multi-table JOIN with 1 filter」のような **公開タスクで頻出する構造**を fewshot で例示する → 公開 50 score は上がるが A/B-board では効かない

### Tier 3: 評価リーク (慎重に判断する)
**実験設計を public 50 の結果で iterate すること自体が overfit**。我々が exp_NNN を改善するたびに public 50 score を見て「効いた / 効かなかった」を判断しているため、設計の細部が public 50 の特性に最適化されている。
- 例: exp_023 の glossary 抽出 regex は task_163 の `**term**: definition` を救うために調整 → task_163 が hidden test に無ければ無意味
- 緩和策: **設計の justification は「該当タスクの failure mode 分析」ではなく「一般原則」で書く**。e.g. ❌「task_163 の type 取り違えを救うため」→ ✅「knowledge.md は LLM のコンテキスト窓で読み飛ばされやすいので、定義文だけ抽出して目立つ位置に置く」

### 隠しテストでの効果見積もり

公開 50 上で +Δ 取れたとしても、A/B-board での見込み は次のように割引いて考える:
- 純粋 prompt エンジニアリング (e.g. format rule, JSON 推奨など): 8 割伝達
- per-task 動的処理 (knowledge.md glossary 抽出 など、agent が runtime に読み込む): 8 割伝達
- ✅ public-task-agnostic な構造改善 (D1 tool tolerance): 8 割伝達 — 隠しテストでも効く
- ⚠ 公開 50 にしか出ない特定スキーマ・特定 DB 名に依存した最適化: 0 割
- ⚠ public タスクのパターンを fewshot で例示: 0〜2 割（パターンが偶然一致したときだけ）

### 実装上の確認 (新 exp 追加時に必ず行う)

1. system prompt / preamble に **公開 50 の question / gold / SQL の文字列が含まれていない**こと (`grep` で確認)
2. fewshot 例の DB スキーマが **公開 50 のどのデータセットとも被っていない**こと (e.g. student_club / formula1 / financial / molecule / cards など 既知ドメインを避ける)
3. 設計 justification に **特定 task_NNN の名前**を書かないこと (一般原則で書く)
4. 違反検出時は smoke FAIL → implement に戻る

### 既知のリーク済み実験 (新ベース選定時に rank しない)

- **exp_026_fewshot_pathfix** (λ0.5 = 0.7242): Tier 1 リーク確定。FIXED_EXAMPLES の Example A=task_145 / B=task_19 / C=task_173 を verbatim 引用。
- **exp_027_loop_break** (λ0.5 = 0.7067): exp_026 を base にしているため Tier 1 リーク継承。
- **exp_028_fewshot_synthetic** (λ0.5 = 0.6631): 合成例だが Tier 2 構造リーク疑い。実際の score も exp_023 を下回っているので採用しない。

### 真のベスト (リーク撤去後)

- ✅ **exp_023_glossary_preamble** (λ0.5 = 0.6837): 全 Tier 安全。preamble glossary は agent が runtime に読む knowledge.md からの抽出。submission/main.py の EXPERIMENT_NAME はこれを基準にする。

## ファイル配置
```
src/
├── kobushi_core/                # 全 exp 共有
│   ├── benchmark/dataset.py     # public dataset loader
│   ├── eval/csv_compare.py      # 公式スコア計算 (注意: lock-stepの真実)
│   ├── model.py                 # OpenAIModelAdapter (max_retries=8, timeout=120s)
│   └── ...
└── experiments/
    ├── exp_001_react_baseline/  # 公式 starter kit, max_steps=16
    ├── exp_002_max_steps_32/    # max_steps=32
    ├── exp_003_preamble/        # preamble (knowledge.md + dtype + サンプル)
    ├── exp_004_preamble_max_steps_32/  # preamble + max_steps=32
    ├── exp_005_column_plan/     # plan first prompt
    ├── exp_006_plan_no_preamble/  # plan + tools (no preamble)
    ├── exp_007_plan_preamble_tools/  # 全部入り (現状ベスト 0.6510)
    ├── exp_008_thinking_off/    # ❌ 失敗 (CoT 無効化で format error 連鎖)
    └── exp_009_best_of_n_vote/  # N=3 候補 + 列 signature 投票 (進行中/直近)

submission/
├── Dockerfile                   # python:3.10-slim + uv install
├── main.py                      # /input → /output エントリ
├── build.sh                     # build + save tar.gz
└── README.md

BACKLOG.md       # アイデア蓄積 (人間も追記可)
EXPERIMENTS.md   # 実験結果ログ (このワークフローが追記する)
PRIORITIES.md    # 人間が今すぐ試させたいことを書く即時フック (任意)
```

## 失敗カテゴリ語彙
- `perfect_recall_no_extras`: 公式スコア 1.0 (列・値完全一致)
- `perfect_recall_with_extras`: 全 gold 列を当てたが余分な列も出した
- `partial_recall`: 一部の gold 列のみ正解
- `zero_recall`: 答えたが値集合がどの gold 列とも一致しない
- `missing_prediction`: agent が `answer` を呼ばずに終わった (max_steps 切れ / API エラー / timeout)
- `api_502` / `api_530` / `context_overflow`: 環境起因。3 連発したらサーバ問題と切り分け

## 現状ベスト構成 (2026-05-01 時点)
- `exp_007_plan_preamble_tools`: λ0.5 = **0.6510**, perfect 32/50 (+1 with_extras)
- 構成: preamble + max_steps=32 + plan instruction + auto-answer tools (`answer_from_python` / `answer_from_sql`)
- 残課題: zero_recall 11 件 (答えたが値間違い) / missing 6 件 (32-step 切れ or API)

## ベンチマーク実行のお作法
```bash
# 単一タスク (smoke):
timeout 300 uv run python -m experiments.exp_NNN_<slug>.run run-task task_26

# 50 タスク + 評価:
uv run python -m experiments.exp_NNN_<slug>.run run-benchmark --evaluate \
  2>&1 | tee /tmp/exp_NNN.log

# 結果:
artifacts/runs/exp_NNN_<slug>_<seq>/
├── evaluation.json     # スコア + 内訳
├── evaluation.csv      # タスクごとの行
├── summary.json        # run-level メタデータ
└── task_<id>/
    ├── trace.json      # agent の全 step + 観測 + 答え
    └── prediction.csv  # 提出物
```

## 重要な設計原則
1. **既存実験は変更しない**。新実験は新ディレクトリ `exp_NNN_<slug>/` を切る。
2. **kobushi_core への変更は別 PR**。実験ディレクトリ内の差分で済ませる。
3. **smoke を必ず通してからベンチマークを起動**。32 step 全部 `__error__` のループに気付かず 50 タスク × 600s = 8h 浪費した実例あり (exp_008)。
4. **API ノイズが疑わしいラン (api_5xx > 20%) は再実行して比較**。並列ベンチ実行は禁止。
5. **失敗実験 (negative result) もディレクトリ + EXPERIMENTS.md に記録**。後の選定で重複回避するため。

## よくある落とし穴
- preamble に JSON プレビューが含まれると agent 出力の JSON フォーマットが乱れて `__error__` が増える
- `read_csv` / `read_json` が大きい結果を返すと context 32k を圧迫し observation 累積で overflow (server max-model-len は 262k だがクライアント側で制限している場合あり)
- `enable_thinking: False` を Qwen3.5 で使うとツール observation を読み返して self-correct する能力が落ちる (exp_008 の教訓)
- subprocess を fork ベースで実行すると pandas/sqlite の状態継承で deadlock (exp_003 で発覚、spawn に切替済み)
- Best-of-N で N 候補をシーケンシャル実行する場合、`task_timeout_seconds` を N × 単発タスク時間に拡げないと subprocess 殺される (exp_009 で発覚、1800s に増やした)
