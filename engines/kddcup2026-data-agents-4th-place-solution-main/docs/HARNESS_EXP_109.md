# Harness `exp_109_plan_first_strengthen` 解説

> 現行 best 候補 (= 提出 v3 候補)。exp_086 の 4 つの弱点を fix した拡張版。
> n=2 mean λ0.5 = **0.7337** ± 0.009 (= run_001=0.7400 / run_002=0.7273)、n=3 確定 ~13:00 JST。
> exp_086 比 ±0.01 で **統計的に互角**、ただし構造的に異なる勝因と robust 性。

---

## 1. exp_086 からの差分 (= 4 つの fix を統合)

| fix | 効果 | 根拠 |
|---|---|---|
| **Rich preamble** (= per-col profile + raw) | 大ファイル task で schema 全体像確保 | exp_101 で +0.045 確認 (single-att) |
| **Padding fix** (= mode-by-attempts length group) | union での None pad による signature 破壊解消 | exp_093 で実証 (task_11 / task_379 救済) |
| **row_count predeclaration 撤廃** | tied-gold tasks 取り捨て解消 | task_25 (+0.31)、task_379 (+0.38) で寄与確認 |
| **Plan-First column verify** (Rule 18) | extra column 提出を機械可読 verify で削減 | task_38/259/67 系で寄与 |

その他は exp_086 と完全互換 (= sampling, multiprocessing, tool 構成、step budget)。

---

## 2. 設計の核 (exp_086 と差分のみハイライト)

| 軸 | 値 | exp_086 比 |
|---|---|---|
| Model | qwen3.5-35b-a3b | = |
| Attempts per task | 3 | = |
| Attempt temperatures | (0.6, 0.6, 0.7) | = |
| Workers (outer) | 4 | = |
| Per-task timeout | 900 秒 | = |
| Max steps | 32 | = |
| Min steps | 4 | = |
| Sampling | T=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.0, max_tokens=32K | = |
| **Preamble** | **rich profile + raw** | **★差分: 後述** |
| **Union merge** | **mode-by-length group, no padding** | **★差分: 後述** |
| **System prompt** | **19 rules, row_count 撤廃 + verify rules** | **★差分: 後述** |

---

## 3. Rich preamble (= 最大の差分)

`src/experiments/exp_109_plan_first_strengthen/preamble.py` + `profile.py`。

### 構成 (= classic と同じ骨子だが各セクション内が profile 化)

```
1. ## Note for agent
2. # Task overview
3. # Workspace overview
4. # Glossary
5. ## [CSV] / [SQLite] / [JSON] / [DOC] 各セクション
   ├ [profile] = per-column 統計 + 5-row sample
   └ [raw content] = 小ファイルのみ全文同梱
6. # Knowledge (full)
```

### Per-column profile (= classic にない情報)

数値列:
```
- amount: int64, null=0, unique=9, min=10 q25=20 median=55 q75=150 max=350 mean=80.7
```

カテゴリ列:
```
- category: str, null=0, unique=5, mean_len=8.4, top5=['Food'(22), 'Advertisement'(15), ...]
```

→ agent が `df.describe()` を呼ばずに済む + 巨大ファイル (= sampling では届かない 100+ 列) の全列を確認可能。

### 大ファイル時の挙動

- **CSV**: 50KB 超過時、profile + first-5 行のみ (= sampled rows dump 廃止)
- **JSON**: 20KB 超過時、profile + 1 サンプルレコードのみ
- **SQLite**: 各 table の DDL + profile + 5-row sample
- **DOC**: 30KB 超過時、先頭 1000 字のみ

### 効果実証 (exp_101 = rich + single-att)

- 50 task 平均 preamble サイズ: **78K → 36K chars** (-54%)
- 最大: **400K → 74K chars** (-82%) → 巨大 task の attention dilution 解消
- single-att score: **0.6700 (classic) → 0.7147 (rich)** = **+0.045**

詳細は `src/experiments/exp_101_rich_preamble/profile.py:column_profile_lines` を参照。

---

## 4. Tools (= exp_086 と同一、10 個)

変更なし。`answer / answer_from_python / answer_from_sql / execute_context_sql / execute_python / inspect_sqlite_schema / list_context / read_csv / read_doc / read_json`。

(exp_108_tool_refactor で `read_csv/json/doc` 削減を試行 → -0.099 regression で廃棄、現状維持が optimal と確認)

---

## 5. System prompt rules (= 19 ルール、exp_086 の 17 + 新 2)

`src/experiments/exp_109_plan_first_strengthen/prompt.py:REACT_SYSTEM_PROMPT`。

### 改修されたルール

**Rule 1 改 (= row_count 撤廃):**
```
1. Your VERY FIRST action's `thought` must contain an explicit answer plan with two parts:
   - column_count: how many columns the final answer will have.
   - per_column: what each column represents.
   (Do NOT predeclare row_count — actual row count emerges from the data
   after filtering / aggregation; guessing it upfront leads to either
   truncating valid tied results or padding to a guessed length.)
```

**Rule 6 改 (= column_count のみ confirm):**
```
6. Before calling `answer`, restate your plan in the `thought` and confirm
   your output's column_count matches it.
```

### 新規ルール (= Rule 18, 19)

**Rule 18 (= column_count 機械可読 verify、必須):**
```
18. Before calling any answer tool, your `thought` MUST contain:
    "VERIFY: plan column_count=X, answer column_count=Y → {MATCH | VIOLATION}."
    If VIOLATION, drop extra columns BEFORE submitting.
    Common failure: Python script returned DataFrame with `id` or other
    metadata columns — drop them with `df[planned_cols]`.
```

**Rule 19 (= 曖昧名詞の interpretation 明示、step 1 で必須):**
```
19. AMBIGUOUS-NOUN INTERPRETATION (mandatory in step 1).
    If the question contains an ambiguous noun ("type", "amount", "level"…),
    the FIRST thought MUST contain:
        "INTERPRET: '<noun>' refers to <specific entity in schema>, not <alternative>."
```

### 削除されたルール

- Rule 6 の row_count 確認 (= 上記 Rule 1 改で撤廃)
- (元案 Rule 20: superlative tied-N 宣言 → Rule 10 filter-back に統合済のため削除)

### exp_086 と互換のルール (= 13 個)

Rule 2-5, 7-14, 16-17 は変更なし (= filter-back / SQL schema-first / explicit SELECT cols / ROUND 規範 / undirected edge dedup / min_steps guard 等)。

---

## 6. Multi-attempt union (= exp_093 padding fix 移植)

`src/experiments/exp_109_plan_first_strengthen/runner.py:_signature_majority_merge`。

### 旧 (exp_086) の問題

3 attempt が異なる行数を返すと `max_len` で None pad → CSV → eval 時 `None → ""` で multiset signature 破壊 → 正解列も recall 0。

### 新 (exp_093 fix 移植)

```python
# 1. Collect all column signatures (= sorted multiset of values) across 3 attempts
# 2. Group columns by intrinsic length (= number of values, NOT padded)
# 3. Score each length group: (# distinct attempts contributing, -length)
#    = mode by attempts, ties broken by smaller length
# 4. Pick winning length, drop columns from other length groups
# 5. Concatenate kept columns side-by-side (all same length, no padding needed)
```

### 効果

- 全 50 task 中 4-7 task で blow-up 解消 (= task_11 / task_379 / task_25 / task_259)
- exp_093 単体では classic preamble 上で +0.005 程度 (= bug 顕在化 task が少なかった)
- exp_109 = rich preamble × padding fix の combo で **rich preamble の divergence amplification を相殺**

シミュレーション: exp_103 (= rich preamble + 3-att, padding バグあり) の merged answer に fix を後付けで適用すると **+0.029 改善** (task_379: 0.00 → 0.75, task_11: 0.00 → 0.67、副作用ゼロ)。

### "Shortest tie-break" の妥当性

DABench gold の row 数分布:
- 1 行: 36 task (72%)
- 2-3 行: 6 task
- 5+ 行: 7 task

→ shortest 偏重デフォルトは **72% で構造的に正しい**。長 gold task では mode-by-attempts が tie-break より先に effect。

詳細: `docs/PUBLIC_TASK_ANALYSIS.md` Type H section 参照。

---

## 7. データフロー (= exp_086 とほぼ同じ、preamble + merge のみ差分)

```
task_id
  → load rich preamble (= per-col profile + raw, schema-only fallback)
  → 3 multiprocessing fork (T=0.6/0.6/0.7)
       → 各 fork: ReAct loop (max 32 step) with 19-rule system prompt
       → Rule 18 column_count VERIFY 必須
       → answer tool 呼び出しで stop
  → 3 attempt answer を _signature_majority_merge (mode-by-length group)
  → merged answer を prediction.csv に書き出し
  → eval
```

per-task elapsed: **avg 4.0 min, max 7.5 min** (= exp_086 比 -25%)。rich preamble の profile が早期収束を促進。

---

## 8. 実績と限界

### 強み (= exp_086 比)

- ✅ **rich preamble の +0.045 を 3-att 環境にも持ち込み**: 巨大 ファイル task (task_330 / task_259 / task_420) で schema 全体像
- ✅ **padding fix で hidden -0.05〜-0.10 regression 解消**: rich preamble × 3-att union での divergence amplification を吸収
- ✅ **row_count 撤廃で tied gold task 救済**: task_25 / task_379 で +0.3〜+0.4 寄与
- ✅ **Column_count 機械可読 verify**: task_38 / task_259 / task_67 系の wext 削減
- ✅ **Per-task latency -25%**: profile が agent の探索 step 浪費を削減

### 弱み (= 残存課題)

- ❌ **計算ロジック sanity check なし** (Type B): task_169 (82M vs gold 460) は依然救えない
- ❌ **Filter scope 妥当性チェックなし** (Type C): task_180 (153 行 vs gold 9 行) も救えない
- ❌ **外部知識 access なし** (Type G): task_344 / 418 (= 医学閾値) は構造的に解けない
- ❌ **Narrative doc parsing 強化なし** (Type F): task_344 / 418 / 396 で文書抽出が agent 任せ
- ❌ **多解釈生成なし** (Type E): task_163 (= "type of expenses" の解釈 bias) で gold disagreement

詳細失敗 type 分析: `docs/PUBLIC_TASK_ANALYSIS.md` 横断的失敗パターン section。

### スコア (= 現状)

n=2 mean λ0.5 = **0.7337** ± 0.009、n=3 確定 ~13:00 JST。**exp_086 (= 0.7353) と統計的に互角**。

---

## 9. 提出戦略への含意

- **構造的 robustness が向上**: padding バグなし、preamble truncation なし、column verify あり
- **score 上は exp_086 とほぼ同じ**: +0.005 程度の差は noise band 内
- **LB 提出 v3 候補**: exp_086 と互角 + 構造的 robust → **LB transfer 効率がより安定**期待 (= local→LB gap が exp_086 より小さい可能性)
- **天井までの距離**: Type B/C/E/F/G を Tier 2/3 軸で attack すれば **0.78〜0.82 圏内**狙えるが、それは exp_109 のさらに先

---

## 10. 関連ファイル

```
src/experiments/exp_109_plan_first_strengthen/
├── agent.py        # exp_086 と同一 (= 19-rule system prompt 経由)
├── config.py       # exp_086 と同一
├── config.yaml     # exp_086 と同一
├── preamble.py     # rich preamble (= profile + raw + 4-tier fallback)
├── profile.py      # ★新規: column_profile_lines, csv_profile_text, sqlite_table_profile
├── prompt.py       # 19-rule system prompt (= row_count 撤廃 + Rule 18/19 追加)
├── runner.py       # _signature_majority_merge: mode-by-length group fix 移植
├── runtime.py      # exp_086 と同一
├── run.py          # CLI entry
└── tools/          # exp_086 と完全同一の 10 ツール
```

## 11. 次の改善候補 (= exp_110 案)

1. **計算結果オーダー verify** (= Type B 救済): "result が 1e6 オーダーだが question scope は 100s なら警告"
2. **Filter-scope 妥当性 verify** (= Type C): "row count vs question word count" の rule-based 比較
3. **多解釈 union** (= Type E): step 1 で 2-3 通り plan 生成 → 並列実行 → 結果並記

これらは `docs/PUBLIC_TASK_ANALYSIS.md` の Tier 2/3 と整合。
