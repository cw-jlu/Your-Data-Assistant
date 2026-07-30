# Experiments log

Concise running log of experiments under `src/experiments/exp_NNN_*/`.
Public dataset: 50 tasks (15 easy / 23 medium / 11 hard / 1 extreme).
Metric: `official_score_lambda_0_5_mean` (公式スコア、λ=0.5 想定値).
Model: `qwen3.5-35b-a3b` (固定、MoE). Run: `max_workers=8`, `task_timeout_seconds=900` (lock-active).

## 🎯 Verified n=3 candidate ranking (2026-05-04, lock-active)

Statistical replication で確定した真スコア。1-run スコア (Summary 表) は全て stochastic
で ±0.05 揺れることが判明したので、以下 mean ベースの判定を **真の比較基準** とする。

| Rank | Experiment | n | mean λ0.5 | median | std | 95% CI | 1-run record |
|------|------------|---|-----------|--------|-----|--------|--------------|
| 🆕 **submitted v3 (2026-05-07)** | **`exp_109_plan_first_strengthen`** (rich preamble + padding fix + row_count predeclaration removed + Rule 18 column_count VERIFY mandatory) | **3 clean** | **0.7328** | 0.7311 | **0.0065** | [0.7254, 0.7402] | 0.7400 (run_001) — exp_086 と統計的に互角 (-0.0025)、stdev 半減で robustness ↑。LB 予測 ~0.49 |
| 🆕 **single-att rec (2026-05-07)** | **`exp_101_rich_preamble`** (per-col profile [profile] section: dtype/null/unique/min/q25/median/q75/max/mean for numeric, top5(counts) for categorical; small files raw-included; large files profile-only replacing sample dump) | **3 clean** | **0.7147** | 0.7164 | **0.0226** | [0.6904, 0.7390] | 0.7364 (run_003) — **single-attempt class 歴代 best**, +0.045 vs single-att floor 0.6700; -0.021 vs 3-attempt baseline (= 3-att union boost ぶん残) |
| 🥇 **1** (3-attempt) | **`exp_086_r1_official_params`** (T=0.6/0.6/0.7 union, min_steps=4, R1 official params) | 2 clean ⚠️ run_004 invalidated | **0.7353** | 0.7353 | **0.0135** | [0.7167, 0.7540] | 0.7449 (run_002; 1run last: 0.7258); run_004=0.5567 API汚染 |
| 2 | `exp_081_union_t0` (3-attempt union, T=0 only) | 3 | **0.6906** | 0.6892 | **0.0138** | [0.675, 0.706] | 0.7017 |
| 3 | `exp_068_kira_function_calling_v2` (min_steps=4 guard) | 3 | **0.6828** | 0.6717 | **0.0411** | [0.6362, 0.7293] | 0.7367 (run_001, +1.4σ outlier確定; 1run last: 0.6483) |
| 4 | `exp_061_r16_only` (Rule 16 deterministic fix) | 3 | 0.6733 | 0.6683 | 0.0278 | [0.642, 0.705] | 0.7083 |
| 5 | `exp_067_c1_col_minimize` | 4 | 0.6687 | 0.6600 | 0.0260 | [0.643, 0.694] | 0.7067 (run_006; 1run last: 0.6483) |
| 6 | `exp_040_selfdbg_fence` (canonical baseline) | 3 | **0.6633** ✅clean | 0.6533 | **0.0218** | [0.639, 0.688] | 0.7267 |

**注 (2026-05-04 確定)**: exp_040 clean re-replicate **完了** (runs 014/015/016)。旧 n=3 (008/009) は API 障害で miss=11 (22% タスク欠測) と確定→棄却。**clean baseline = 0.6633 ± 0.0218 (n=3) [0.639, 0.688]**。pooled clean (n=4, runs 010+014–016) = mean 0.6579, std 0.0208。

### 確定した知見
- **exp_086_r1_official_params が確定 win (n=2 clean, mean=0.7353 Δ+0.0447 vs exp_081)** — R1 公式 params (T=0.6/0.6/0.7, presence_penalty=1.0) が temperature diversification 効果で recall +0.12。CI [0.7167, 0.7540] が exp_081 CI [0.675, 0.706] と非重複。⚠️ run_004 は API汚染 (missing=13/50、後半連続帯で vLLM stall) で invalidate。clean n=2 は有効、**run_005 で clean n=3 確定推奨**。**新 submission ベース候補 (v3)**
- **exp_087_r2_column_vote は ❌ regression (n=2 mean=0.7098 Δ-0.0255 vs exp_086)** — k=1→k=2 の 1 行 ablation。extras 削減 (with_extras 5→2.5) は確認できたが **recall loss (-0.04) がスコア寄与で 4-10× 上回る**。k=2 は temperature 多様化 (T=0.6/0.6/0.7) と相性が悪い: diversity が高いほど majority 不成立で union-only 正解が脱落。**k と temperature diversity は逆相関設計パラメータ** — k=1+高diversity が SOTA、k=2+低diversity は alternative。**[aggr] R2 ⌈n/2⌉ majority vote 軸は廃棄確定**
- **exp_088_r3_mschema_plan_first は ❌ regression 確定 (n=2 clean, mean=0.6915 ± 0.0270, Δ-0.044 vs exp_086)** — R3 M-Schema × Plan-first + 3-attempt union。runs: run_002=0.7106 (missing=2), run_003=0.6724 (missing=1), run_004 廃棄 (vLLM 502 outage)。reproducibility: run 間 Δ=0.038 (中程度)。**run 間差異の大きいタスク**: task_11 (両 run zero — M-Schema × T-diversity × union 三項相互作用で 9cols×75rows 膨張 → recall=0)、task_25 (両 run zero — `runner.py:_signature_majority_merge` row-padding バグで multiset signature 破壊)。with_extras 増加 run_002:+5 → run_003:+9 (Plan-first が attempt 別 column commit を促し union 後列数増)。M-Schema は task_379/67/257 で +0.375/+0.125/+0.083 の純粋 benefit 確認 → **single-attempt 化なら有望**。**[context] R3 M-Schema + Plan-first + 3-attempt union 廃棄確定**。`_signature_majority_merge` row-padding バグは独立 fix 候補 (横断効果 +0.02〜+0.04 期待)
- **exp_090_r5_structured_error_hint は ⚠️ ノイズ範囲内 (n=1 run_002=0.7236, Δ=-0.012 vs exp_086 mean=0.7353)** — R5 structured error hint (10-class regex exception handler)。hint 発火 19/50 task だが実発火は KeyError (18 件) + FileNotFound (3 件) の 2 class のみ、残り 8 class は dead code。最大損失 task_259 (-0.594、wrong-column failure → R5 圏外)、最大利得 task_379 (+0.375)。観測 Δ=-0.012 は task_259 の -0.594/50 ≈ -0.012 でほぼ単独説明 → 軸効果 ±0.005 程度の微細。**n=1 未確定、隣接 exp_046/056 も error-feedback 系統はスコア寄与微小で再現**。replication 判断は pick_next に委ねる
- **exp_096_cursor_harness_v2 は ❌ regression vs exp_086 (n=1 run_002=0.6700, Δ=-0.065)、baseline exp_040 と同水準 (+0.005)** — [arch:harness] cursor 流 schema-only preamble + tool discovery。positive: with_extras=0 (vs exp_086 avg 5) 完全削減、task_89/86 format 認識で zero→perfect (+0.04)。negative: task_11/173/408 knowledge.md semantic skip (-0.06)、8 missing infra brittleness (timeout 4/no_answer 2)。**exp_082 cursor_harness (workers=20) の workers汚染は確認**: workers=20 → 0.6174 vs workers=10 → 0.67 (+0.05 改善)、ただし exp_086 R1 比でまだ -0.065 構造的下位。single-attempt class score ≈ 0.67 (4 例: exp_040/092/097/096) 再確認。schema-only 軸は extras 削減に有効だが recall 確保には fulldata preamble + 3-attempt が必要
- **exp_097_mschema_single_attempt は ❌ raw regression (n=1 run_002=0.6614, Δ=-0.074) だが真の M-Schema axis effect ≈ 0 と確定** — [context-single] M-Schema DDL+sample block (Plan-first 抜き) + single-attempt。Δ=-0.074 の主因は **timeout 7 task による missing** (M-Schema 4K preamble で per-call latency 増 → 900s timeout で single-attempt に fallback なし → λ=0 で score 引き下げ)。timeout 除外 43-task subset Δ=-0.003 = noise 圏。**M-Schema 純粋 axis effect の確定**: improvement task_67/196/200/330 (+0.125 each, extras 削減効果) vs loss task_379 (-0.375 sampling miss), task_25 (-0.312 row-padding バグ)。**exp_088 (-0.044) の害は M-Schema 単体ではなく 3-attempt union × T-diversity との attempt-divergence 増幅が主因** — M-Schema 単体を clean class で測定すると効果 ≈ 0 ± noise。single-attempt class の標準 score ≈ 0.66 改めて確認 (exp_040: 0.6633, exp_092: 0.6400, exp_097: 0.6614)。**M-Schema 系統の扱い**: 3-attempt union + row-padding fix との combo で再評価が必要
- **exp_094_qwen_agent_runtime_swap は ❌ regression 確定 (n=1 run_003=0.7119, Δ=-0.024 vs exp_086 mean=0.7353)** — [arch:runtime] Qwen-Agent native wire swap。wire-level audit (commit ad7b923) で主因確定: `qwen_agent/llm/base.py:177-178` が call 毎に `random.randint(0, 2^30)` を auto-inject → vLLM RNG re-seed → 3 attempts 完全独立 divergence → union extras 増 → λ0.5 低下。with_extras=9 (vs exp_086 avg 5)、λ1.0=0.659 (vs exp_086 0.701)。task_80 (+1.000 unexpected improvement、n=1 で stochastic or native chat template 効果か不明)。**vendor magic 仮説は存在しないことが実証**、framework swap 系は今後試さない
- **exp_095_column_candidate_hint は ⚠️ likely_regression (n=1 run_002=0.7086, Δ=-0.027 vs exp_086 mean=0.7353、replication 待ち)** — [preamble:hint] column candidate hint (non-LLM NLP heuristic top_n=7)。**column hint 軸自体の害はなし** (発火 41 task mean=0.7453、非発火 9 task mean=0.5417 — これは row-padding バグに弱い task 構造の反映)。Δ=-0.027 の主因は **row-padding バグの stochastic 表面化** (task_11/22/25 で合計 Δ≈-0.025)、hint 純粋効果 ≈0。task_379 (+0.0075) で hint の正方向寄与あり。exp_094 vLLM 占有でレプリケーション 2/3 待機中、n=1 で確定不能。
- **exp_093_runner_padding_fix は ⚠️ ノイズ範囲内 (n=1 partial 49/50 = 0.7328 imputed, Δ=-0.0025 vs exp_086 mean=0.7353)** — [bugfix-infra] `_signature_majority_merge` row-padding fix。期待 +0.02〜+0.04 は出ず。task_259 (+0.031) では正解 text 列を保持できたが、task_25 (依然 zero: 全 attempt が wrong filter に convergence → fix 対象外)、task_11 (-0.125: length-grouping 副作用で extras +1)。**bug fix は logically 正しいが axis win にならない**。replication 失敗 (run_002: 49/50 で wall-clock kill, run_003: stale lock abort) → n=1 確定不能。bug fix の "守りの価値" (future combo での padding 汚染排除) のみ確定。roq-padding バグの主因は "全 attempt content divergence" であり fix 単体では不十分。
- **exp_092_verify_rule_single は ❌ regression 確定 (n=1 run_004=0.6400、baseline 未満)** — [verify-rule] single-attempt + non-LLM regex verifier。失敗は 2 重構造: (1) **single-attempt 化 (-0.06)** — exp_086 の T-diversity × union の +0.045 を直接喪失 (task_11/25/257/259/352 で recall 消失)、(2) **verifier regex false positive (-0.04)** — `_SINGLE_VALUE_PATTERNS` が multi-ask question ("X and Y") を blind; task_249 で agent が正解 2-col を提出 → verifier "1 col implied" hint → agent が 1-col に collapse → recall=0。hits 5 tasks 中 4 が hurt。run_004 run_001-003 は smoke runs (1 task)。zero_recall=13/50 (vs exp_086 avg ~10)、missing=3。**[verify-rule] regex verifier は廃棄確定、非 LLM verifier の multi-ask blind 根本欠陥確認**
- **exp_081_union_t0 が旧ベスト mean (0.6906) + 最低 std (0.0138)** = exp_086 前の安定最高。submission v2 候補 (exp_086 に置換予定)
- **exp_068 の新 n=3 (runs 006/007/008): mean=0.6828, std=0.0411** — 旧 n=3 比 std が 2.8× 膨張。元 "0.7367" は pooled mean 0.6882 の +1.4σ 外れ値確定。"新 clean best" タグ撤回。⚠️ 旧 n=3 (003/004/005, std=0.0146) と現 n=3 (006/007/008, std=0.0411) で std が大きく乖離 — pooled n=7 (mean=0.6882, std=0.0332) で判断するのが適切
- **col-minimize (exp_067) と Rule 16 (exp_061) は exp_068 と mean で tie** (0.005 以内、CI 完全重複)
  → これらの軸は真の独立効果 ≈ 0、min_steps=4 の貢献が dominant
- **Rule 16 prompt の確実性は 4/7 run (57%)** — "deterministic fix" とは言えない。確実化には runner-level post-process が必要
- **真の score スケール**: 公開 50 で **0.65〜0.71 圏**。submission v1 (exp_068) 本番期待値 ~0.67; submission v2 候補は exp_081 (std 1/2.4)
- **exp_040 clean baseline 確定 (2026-05-04, runs 014/015/016)**: mean=**0.6633**, std=**0.0218**, CI=[0.639, 0.688]。旧 n=3 (008/009 miss=11) は API 障害汚染確定・棄却済み。clean baseline は **0.66 ± 0.02 圏**で固定。全候補の Δ vs clean baseline: exp_081 Δ**+0.0273** (+1.25σ)、exp_068 Δ**+0.0195** (+0.47σ)、exp_061 Δ**+0.0100** (+0.36σ)、exp_067 Δ**+0.0054** (+0.21σ, n=4 更新) — **全て ⚠ likely_win (CI 重複)、confirmed_win = 0。exp_067 は n=4 でノイズ確定**。
- **12 always-zero タスクの構造的失敗 (exp_040 n=3 cross-run 分析)**: Type A timeout 系 (task_80/180/344/418) — max_steps 32 or timeout 900s 到達で no-answer。Type B 3 run 同一 wrong answer (task_89/169/259/379/396) — agent の確信が間違い方向に固定。Type C 列スキーマ揺れ (task_25/163/199) — plan 段階で answer schema が run ごとに変わる。Type B は answer 直前の Python sanity-check で救済可能性あり (exp_value_audit 候補)。

### 3-attempt + signature merge 軸の検証 (exp_078/079/080/081)

| Variant | 1-run λ0.5 | n=3 mean | std | 比較 |
|---------|-----------|----------|-----|------|
| exp_078 (T=0/0.3/0.7, k=2 majority) | 0.6767 | — | — | ≈ exp_068 mean (1-run) |
| exp_080 (T=0 only, k=2 majority) | 0.6767 | — | — | ≈ exp_068 mean (1-run) |
| **exp_081 (T=0 only, k=1 union)** | 0.7017 | **0.6906** | **0.0138** | **n=3 で確認: exp_068 と統計 tie だが std が最小** |

**確認できた示唆**:
- **k=1 union が k=2 majority より優れる** (1-run 比較: exp_081 0.7017 > exp_078/080 0.6767)
  → union は 1 attempt でも正答出れば残す = recall 重視の merge が有効
- **temperature 多様化は不要** (T=0 only でも k=1 union が 0.6906 出る = 必要十分)
- **3-attempt union は std を 0.0138 まで下げる** (exp_068 n=3 の 0.0146 より低い、exp_068 n=4 の 0.0319 より遥かに低い)
  → **同じ mean なら std 低い方が submission 向け** (本番期待値の信頼区間が狭い)
- mean は exp_068 と CI 重なり = **mean では tie**、std で勝つ

**submission v2 候補**: exp_081 は std が最低なので、本番期待値 0.69 ± 0.01 と狭い CI で
保証できる。ただし **3-parallel attempt × per-task budget A-board 120s** だと
リスクあり (= 3 attempts 同時実行で local 60-90s/task が公式環境で 120s 内に収まるか未確認)。
v1 を維持するか v2 で exp_081 にするかは、提出前 Docker での local smoke で判断。

### 廃棄確定軸 (replication で noise と判定)
- col-minimize Rule 17 (exp_067 n=4, exp_072): exp_067 n=4 mean Δ+0.0054 vs baseline = noise確定; exp_072 min_steps+Rule18 も同結果 → **[output] C1 col-minimize Rule 17 単独軸廃棄確定**
- Rule 16 bidir dedup の **単独効果**: 確実な signal なし。ただし task_196 の deterministic fix
  自体は本物 (per-task で 0→1.0 確定)、構造は維持価値あり



Important spec corrections (per https://dataagent.top/rules):
- Phase 1 deadline: **2026-05-25** (旧 5/20 から修正、freeze 4/30–5/2 含む).
- Hidden test: **A-board ~60 tasks (2h wall-clock cap) + B-board ~320 tasks**.
- Server max-model-len: **262144 tokens**（per-call の 32k 上限は別の制約）.
- Submission: Docker image to Google Drive, 10 GB max, 1/day, 30 max in Phase 1.

Defensive infra adds:
- `OpenAIModelAdapter` に `max_retries=8`, `timeout=120s`（CF 502/503/504 を吸収）.

## Summary

| exp | preamble | max_steps | plan | auto-answer tools | λ0.5 score | perfect | answered | run dir |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---|
| exp_001_react_baseline | – | 16 | – | – | **0.2650** | 13/50 | 16/50 | exp_001_react_baseline_002 |
| exp_002_max_steps_32 | – | 32 | – | – | **0.4200** | 21/50 | 28/50 | exp_002_max_steps_32_002 |
| exp_003_preamble | ✓ | 16 | – | – | **0.4600** | 23/50 | 26/50 | exp_003_preamble_005 |
| exp_004_preamble_max_steps_32 | ✓ | 32 | – | – | _partial_ | 20/35 (stopped) | 20/35 | exp_004_preamble_max_steps_32_002 |
| exp_005_column_plan | ✓ | 32 | ✓ | – | _partial_ | 8/17 (stopped) | 8/17 | exp_005_column_plan_002 |
| exp_006_plan_no_preamble | – | 32 | ✓ | ✓ | **0.1600** ⚠ | 8/50 | 16/50 | exp_006_plan_no_preamble_003 |
| exp_007_plan_preamble_tools | ✓ | 32 | ✓ | ✓ | **0.3510** ⚠ | 17/50 (+1 wext) | 29/50 | exp_007_plan_preamble_tools_002 |
| exp_007 (再実行 solo + retry) | ✓ | 32 | ✓ | ✓ | **0.6510** | **32/50** (+1 wext) | 44/50 | exp_007_plan_preamble_tools_004 |
| exp_008_thinking_off | ✓ | 32 | ✓ | ✓ | **fail (smoke)** ❌ | – | – | exp_008_thinking_off_001 |
| exp_009_best_of_n_vote | ✓ | 32 | ✓ | ✓ + N=3 vote | _stopped_ | 部分集計のみ | – | exp_009_best_of_n_vote_004 |
| exp_010_verify_before_execute | ✓ | 32 | ✓+verify | ✓ | **0.4820** ❌ | **23/50** (+1 wext, +2 partial) | 38/50 | exp_010_verify_before_execute_002 |
| exp_011_spec_driven | ✓ | 32 | ✓+spec | ✓ | **0.5270** ❌ | 25/50 (+2 wext) | 34/50 | exp_011_spec_driven_002 |
| exp_012_tool_tolerance | ✓ | 32 | ✓ | ✓ | **0.6710** ✅ | **33/50** (+1 wext) | 46/50 | exp_012_tool_tolerance_002 |
| exp_013_sample_rich_preamble | ✓+value_sample | 32 | ✓ | ✓ | **0.6400** ❌ | 32/50 (+1 wext) | 45/50 | exp_013_sample_rich_preamble_002 |
| exp_015_cascade_json_fix | ✓ | – | – | DuckDB only | **0.3320** ❌ | 16/50 (+1 wext) | 30/50 | exp_015_cascade_json_fix_004 |
| exp_016_forbid_manual_answer | ✓ | 32 | ✓ | ✓ | **0.6510** ❌ | 32/50 (+1 wext) | 47/50 | exp_016_forbid_manual_answer_002 |
| exp_017_tie_aware_superlative | ✓ | 32 | ✓ | ✓ | **0.6752** ✅ | 31/50 (+4 wext) | 46/50 | exp_017_tie_aware_superlative_002 |
| exp_018_fix_select_star | ✓ | 32 | ✓ | ✓ | **0.6620** ❌ | 31/50 (+3 wext) | 46/50 | exp_018_fix_select_star_002 |
| exp_019_column_count_coercion | ✓ | 32 | ✓ | ✓ | **0.6087** ❌ | 29/50 (+2 wext) | 47/50 | exp_019_column_count_coercion_003 |
| exp_020_truncate_offload | ✓ | 32 | ✓ | ✓ | **0.6402** ❌ | 30/50 (+3 wext) | 47/50 | exp_020_truncate_offload_004 |
| exp_021_question_paraphrase | ✓ | 32 | ✓ | ✓ | **0.6617** ❌ | 31/50 (+3 wext) | 47/50 | exp_021_question_paraphrase_002 |
| exp_022_cascade_v3 | ✓ | 24+12 | cascade | DuckDB only | **0.3117** ❌ | 15/50 (+1 wext) | 45/50 | exp_022_cascade_v3_002 |
| exp_023_glossary_preamble | ✓+glossary | 32 | ✓ | ✓ | **0.6837** ✅ | 33/50 (+2 wext) | 44/50 | exp_023_glossary_preamble_002 |
| exp_024_glossary_v2 | ✓+glossary+doc_hint | 32 | ✓ | ✓ | **0.6233** ❌ | 29/50 (+3 wext) | 47/50 | exp_024_glossary_v2_002 |
| exp_025_fewshot_3ex | ✓+glossary+3-shot | 32 | ✓ | ✓ | **0.6517** ❌ | 32/50 (+2 wext) | 44/50 | exp_025_fewshot_3ex_002 |
| exp_026_fewshot_pathfix | ✓+glossary+3-shot(pathfix) | 32 | ✓ | ✓ | **0.7242** ⚠ LEAK | 35/50 (+2 wext) | 46/50 | exp_026_fewshot_pathfix_002 |
| exp_027_loop_break | ✓+glossary+3-shot(pathfix)+loop_break | 32 | ✓ | ✓ | **0.7067** ⚠ LEAK | 34/50 (+2 wext) | 49/50 | exp_027_loop_break_002 |
| exp_028_fewshot_synthetic | ✓+glossary+3-shot(synthetic) | 32 | ✓ | ✓ | **0.6631** ❌ | 32/50 (+2 wext) | 48/50 | exp_028_fewshot_synthetic_002 |
| exp_029_schema_first | ✓+glossary+Rule11 | 32 | ✓ | ✓ | **0.7067** ❌ | 34/50 | 48/50 | exp_029_schema_first_002 |
| exp_030_leak_free_baseline | ✓+glossary+Rule11 | 32 | ✓ | ✓ | **0.6431** ✅ clean | 31/50 | 48/50 | exp_030_leak_free_baseline_002 |
| exp_031_deterministic_cascade | ✓+fulldata(150k) | 32 | ✓ | ✓ | **0.6531** ✅ | 30/50 (+4 wext) | 48/50 | exp_031_deterministic_cascade_003 |
| exp_032_cascade_self_correct | ✓+fulldata(150k) | 48 | ✓+spec | ✓ | **0.6067** ❌ | 29/50 (+2 wext) | 43/50 | exp_032_cascade_self_correct_003 |
| exp_033_profile_table | ✓+fulldata(150k) | 48 | ✓ | ✓ | **0.5917** ❌ | 30/50 | 39/50 | exp_033_profile_table_003 |
| exp_034_no_manual_answer_validated | ✓+fulldata(150k) | 48 | ✓ | answer_from_* only | **0.5756** ❌ | 30/50 | 45/50 | exp_034_no_manual_answer_validated_003 |
| exp_035_fk_map_fence_fix | ✓+fulldata(150k) | 48 | ✓ | ✓+FK map | **0.5539** ❌ | 30/50 | 44/50 | exp_035_fk_map_fence_fix_003 |
| exp_036_explicit_select_cols | ✓+fulldata(150k) | 32 | ✓ | ✓ | **0.6833** ✅ | 32/50 (+3 wext) | 44/50 | exp_036_explicit_select_cols_003 |
| exp_037_lenient_json | ✓+fulldata(150k) | 32 | ✓ | ✓ | **0.5717** ❌ | 28/50 (+1 wext) | 40/50 | exp_037_lenient_json_003 |
| exp_038_function_calling | ✓+fulldata(150k) | 32 | ✓ | ✓ (OpenAI tools) | **0.5764** ❌ | 27/50 (+3 wext) | 39/50 | exp_038_function_calling_003 |
| exp_039_answer_validation | ✓+fulldata(150k) | 32 | ✓ | ✓ | **0.4117** ❌ | 20/50 (+1 wext) | 28/50 | exp_039_answer_validation_002 |
| exp_039_preamble_sqlite_cap | ✓+fulldata(150k)+SQLite cap | 32 | ✓ | ✓ | **0.5917** ❌ | 29/50 (+1 wext) | 41/50 | exp_039_preamble_sqlite_cap_002 |
| exp_040_selfdbg_fence | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13/14 | **0.7267** ✅ ~~新clean best~~ | 35/50 (+2 wext) | 49/50 | exp_040_selfdbg_fence_002 |
| exp_041_inline_spec | ✓+fulldata(150k) | 32 | ✓+inline spec | ✓ | **0.6275** ❌ | 29/50 (+3 wext) | 41/50 | exp_041_inline_spec_002 |
| exp_042_inline_spec_clean | ✓+fulldata(150k) | 32 | ✓+inline spec | ✓ (exp_036 base) | **0.6517** ❌ | 32/50 (+1 wext) | 48/50 | exp_042_inline_spec_clean_002 |
| exp_043_doc_search | ✓+fulldata(150k) | 32 | ✓ | ✓+search_doc+5K preamble | **0.6483** ❌ | 31/50 (+2 wext) | 46/50 | exp_043_doc_search_003 |
| exp_044_inline_spec_v2 | ✓+fulldata(150k) | 32 | ✓+inline spec | ✓+fence3+Rule13/14 | **0.7067** ❌ | 34/50 (+2 wext) | 45/50 | exp_044_inline_spec_v2_002 |
| exp_045_adaptive_doc_search | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13/14+adaptive_doc+search_doc | **0.6083** ❌ | 29/50 (+2 wext) | 48/50 | exp_045_adaptive_doc_search_003 |
| exp_046_tool_error_truncate | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13/14+error_hint | **0.6798** ❌ | 32/50 (+3 wext) | 49/50 | exp_046_tool_error_truncate_005 |
| exp_047_skill_split | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule1-9+read_skill | **0.6917** ❌ | 34/50 (+1 wext) | 46/50 | exp_047_skill_split_005 |
| exp_048_inspect_table | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13/14+inspect_table+Rule15 | **0.6083** ❌ | 29/50 (+2 wext) | 46/50 | exp_048_inspect_table_004 |
| exp_049_targeted_rules | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-16 (avg/0禁止+bidir dedup) | **0.6817** ❌ | 32/50 (+3 wext) | 49/50 | exp_049_targeted_rules_004 |
| exp_050_format_precommit | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13/14+Rule1 COMMIT+Rule6 drop | **0.6483** ❌ | 31/50 (+2 wext) | 48/50 | exp_050_format_precommit_004 |
| exp_051_pre_answer_verify | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-15+Rule17 alignment check | **0.6883** ❌ | 33/50 (+2 wext) | 46/50 | exp_051_pre_answer_verify_004 |
| exp_052_filetype_labels | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-15+filetype labels preamble | **0.6283** ❌ | 30/50 (+2 wext) | 47/50 | exp_052_filetype_labels_004 |
| exp_053_rule6_drop_extra | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-15+Rule6 drop extra cols | **0.5917** ⚠️API汚染 | 29/50 (+1 wext) | 43/50 | exp_053_rule6_drop_extra_006 |
| exp_054_safe_mean_helper | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-15+helpers.py safe_mean+sys.path fix | **0.6867** ❌ | 33/50 (+2 wext) | 45/50 | exp_054_safe_mean_helper_005 |
| exp_055_rule16_bidir | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-15+Rule16 bidir dedup | **0.7083** ❌ | 34/50 (+2 wext) | 46/50 | exp_055_rule16_bidir_005 |
| exp_056_error_explain | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule13-15+Rule18 error analysis format | **0.6883** ❌ | 33/50 (+2 wext) | 46/50 | exp_056_error_explain_005 |
| exp_057_combo_r6_r16 | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule6drop+Rule15+Rule16 combo | **0.6483** ❌⚠️API汚染 | 31/50 (+2 wext) | 43/50 | exp_057_combo_r6_r16_004 |
| exp_058_robust_sql_guard | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rule6+Rule15+Rule16+Rule19 sql guard | **0.6517** ❌ | 32/50 (+1 wext) | 46/50 | exp_058_robust_sql_guard_004 |
| exp_059_dual_eval | ✓+fulldata(150k) | 37 | ✓ | ✓+fence3+Rules1-14+dual eval 2nd LLM pass (agent.py) | **0.7198** ❌ | 34/50 (+3 wext) | 47/50 | exp_059_dual_eval_005 |
| exp_060_search_doc_additive | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14 (≡exp_040, import path only) | **0.7083** ❌ | 34/50 (+2 wext) | 46/50 | exp_060_search_doc_additive_002 |
| exp_061_r16_only | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16 bidir dedup only (+589 chars) | **0.7083** ❌raw / stable-41 **+0.0244** | 34/50 (+2 wext) | 49/50 | exp_061_r16_only_004 |
| exp_063_bon_n2 | ✓+fulldata(150k) | 32 | ✓ | BoN N=2 (T=0.0+T=1.0) + voting (exp_061 base) | **0.6317** ❌ | 31/50 (+1 wext) | 41/50 | exp_063_bon_n2_002 |
| exp_065_kira_double_confirm | ✓+fulldata(150k) | 32 | ✓ | double-confirm registry: 1st answer returns checklist, 2nd accepted (exp_061 base) | **0.7033** ❌ | 33/50 (+3 wext) | 47/50 | exp_065_kira_double_confirm_002 |
| exp_067_c1_col_minimize | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16 bidir dedup+Rule17 col-minimize (timeout=600,workers=4) | **0.6687 ± 0.026 (n=4) [0.643, 0.694]**   \|   1run last: 0.6483 | avg 31.5/50 | avg 45.8/50 | exp_067_c1_col_minimize runs 005–008 |
| exp_068_kira_function_calling_v2 | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4 prompt)+runner min_steps=4 guard (exp_061 base) | **0.7367** ~~新clean best~~ (+1.4σ outlier) | 34/50 (+4 wext) | 46/50 | exp_068_kira_function_calling_v2_001 |
| exp_068_kira_function_calling_v2 **(n=3 rep2)** | ✓+fulldata(150k) | 32 | ✓ | same; runs 006/007/008 | **0.6828 ± 0.0411 (n=3) [0.6362, 0.7293]**   \|   1run last: 0.6483 | avg 33/50 | avg 46/50 | rep2: runs 006–008 |
| exp_069_kira_proactive_summarize | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17+context_summarize_threshold=0.80 LLM 圧縮 (exp_061 base) | **0.6398** ❌ | 30/50 (+2 wext) | 43/50 | exp_069_kira_proactive_summarize_001 |
| exp_070_c4_dtype_canon | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17+Rule18 (dtype: 数値2dp/null→空/ISO日時) (exp_068 base) | **0.6367** ❌ | 29/50 (+4 wext) | 39/50 | exp_070_c4_dtype_canon_002 |
| exp_071_kira_obs_truncate | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17+obs_truncate 30KB cap (exp_068 base) | **0.6283** ❌ | 30/50 (+2 wext) | 47/50 | exp_071_kira_obs_truncate_002 |
| exp_072_c1_col_minimize_v2 | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17+Rule18 (col_minimize: answer 前に不要列削除) (exp_068 base) | **0.7083** ❌ | 34/50 (+2 wext) | 47/50 | exp_072_c1_col_minimize_v2_002 |
| exp_073_reason_min_steps_6 | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=6 prompt)+runner min_steps=6 guard (exp_068 base) | **0.6683** ❌ | 32/50 (+2 wext) | 45/50 | exp_073_reason_min_steps_6_002 |
| exp_074_infra_timeout_900 | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17+task_timeout_seconds=900 (exp_068 base) | **0.7083** ❌ | 34/50 (+2 wext) | 48/50 | exp_074_infra_timeout_900_002 |
| exp_075_kira_obs_truncate_10k | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17+obs_truncate 10KB cap + task_timeout_seconds=900 (exp_068 base) | **0.6683** ❌ | 32/50 (+2 wext) | 47/50 | exp_075_kira_obs_truncate_10k_003 |
| exp_082_cursor_harness | schema-only ~5K | 32 | ✓ | ✓+fence3+Rules1-15+Rule16+grep_file+stat_file (schema-only preamble, single-attempt, workers=20, exp_081 base) | **0.6174 ± 0.0281 (n=3) [0.5855, 0.6492]**   ❌   \|   1run last: 0.6434 | avg 29/50 (+2 wext) | avg 42/50 | exp_082_cursor_harness_002–004 |
| exp_083_additive_tools | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4 prompt)+runner min_steps=4 guard+grep_file+stat_file (3-attempt union T=0,0,0 k=1, **workers=6×3=18 streams ⚠️ config error**, exp_081 base) | **0.3331 ± 0.3402 (n=2, ⚠️ run_003 vLLM outage — 判定不能) [N/A]   \|   1run last: 0.0925 (run_003, 502 outage)** — 有効完了: run_002 40t→λ=0.7170 / run_003 6t→λ=0.7708 | avg 14.5/50 (+3 wext) | avg 23/50 | exp_083_additive_tools_002–003 |
| exp_086_r1_official_params | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard (3-attempt union T=0.6/0.6/0.7 k=1, R1 公式 params, workers=4, exp_081 base) | **0.7353 ± 0.0135 (n=2 clean) [0.7167, 0.7540]** ✅確定win   \|   1run last: 0.7258; run_004=0.5567 ❌API汚染 (missing=13) | avg 33/50 (clean); run_004: 25/50 | avg 49/50 (clean) | exp_086_r1_official_params_002–004 |
| exp_087_r2_column_vote | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard (3-attempt union T=0.6/0.6/0.7 **k=2 majority**, R2 [aggr], workers=4, exp_086 base, 1行変更 ablation) | **0.7098 ± 0.0182 (n=2) [0.6847, 0.7350]** ❌ regression vs exp_086   \|   1run last: 0.7227 | avg 33.5/50 | avg 48/50 | exp_087_r2_column_vote_004–005 |
| exp_088_r3_mschema_plan_first | ✓+M-Schema+fulldata(150k) | 32 | ✓+Rule12a(Plan-first) | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+Rule12a Plan-first (_build_mschema_section DDL+sample 最大4K chars) (3-attempt union T=0.6/0.6/0.7 k=1, R3 [context] M-Schema, workers=4, exp_086 base) | **0.6915 ± 0.0270 (n=2 clean) [0.6541, 0.7289]** ❌ regression   \|   1run last: 0.6724 | avg 29.5/50 (run_002: +5 wext; run_003: +9 wext) | avg 48.5/50 | exp_088_r3_mschema_plan_first_002–003 (run_004 vLLM outage 廃棄) |
| exp_090_r5_structured_error_hint | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+structured exception hint (10-class regex, R5 [tool], 3-attempt union T=0.6/0.6/0.7 k=1, workers=4, exp_086 base) | **0.7236 (n=1 のみ) ⚠️ ノイズ範囲内 (Δ=-0.012)   \|   1run: 0.7236** | 32/50 (+5 wext) | 48/50 | exp_090_r5_structured_error_hint_002 |
| exp_092_verify_rule_single | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+verifier hook (non-LLM regex, max 1 retry, [verify-rule], **single-attempt** T=0.6, workers=10, exp_086 base) | **0.6400 (n=1) ❌ regression (baseline 未満)   \|   1run: 0.6400** | 29/50 (+2 wext) | 47/50 | exp_092_verify_rule_single_004 |
| exp_093_runner_padding_fix | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard (3-attempt union T=0.6/0.6/0.7 k=1, [bugfix-infra] `_signature_majority_merge` row-padding fix, workers=4, exp_086 base) | **0.7328 (n=1 partial 49/50) ⚠️ ノイズ範囲内 (Δ=-0.0025)   \|   1run: 0.7328 (imputed)** | 33/49 | 49/49 | exp_093_runner_padding_fix_002 (partial, killed at 90 min) |
| exp_094_qwen_agent_runtime_swap | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard (3-attempt union T=0.6/0.6/0.7 k=1, [arch:runtime] QwenAgentModelAdapter wire swap, workers=4, exp_086 base) | **0.7119 (n=1) ❌ regression (Δ=-0.024)   \|   1run: 0.7119** | 29/50 (+9 wext) | 49/50 | exp_094_qwen_agent_runtime_swap_003 |
| exp_095_column_candidate_hint | ✓+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+column candidate hint (non-LLM NLP heuristic, top_n=7, advisory, [preamble:hint], 3-attempt union T=0.6/0.6/0.7 k=1, workers=4, exp_086 base) | **0.7086 (n=1) ⚠️ likely_regression (Δ=-0.027、replication 待ち)   \|   1run: 0.7086** | 30/50 (+7 wext) | 49/50 | exp_095_column_candidate_hint_002 |
| exp_096_cursor_harness_v2 | schema-only ~5K | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+schema-only preamble+grep_file/stat_file (cursor 流 tool discovery, [arch:harness], **single-attempt** T=0.6, workers=10, exp_081 base 再評価) | **0.6700 (n=1) ❌ regression vs exp_086 (Δ=-0.065)、+0.005 vs baseline exp_040   \|   1run: 0.6700** | 33/50 (wext=0 ✅) | 42/50 (missing=8) | exp_096_cursor_harness_v2_002 |
| exp_097_mschema_single_attempt | ✓+M-Schema+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+M-Schema DDL+sample 4K block (Plan-first 抜き, [context-single], **single-attempt** T=0.6, workers=10, exp_086 base) | **0.6614 (n=1) ❌ raw regression (Δ=-0.074)、真の axis effect ≈0 (timeout 7 task 除外 43-task Δ=-0.003)   \|   1run: 0.6614** | 32/50 (+1 wext) | 43/50 (missing=7 timeout) | exp_097_mschema_single_attempt_002 |
| exp_099_schema_graph_fallback | ✓+fulldata(150k)+schema-graph FK+orphan-column advisory (LLM-judged, pre-computed) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+schema-graph FK+orphan-column hint (LLM-judged precomputed advisory, [preamble:hint+schema-graph], **single-attempt** T=0.6, workers=10, exp_086 base) | **0.5961 (n=1) ❌ definitive regression (Δ=-0.067 vs single-attempt floor, −3.08σ)   \|   1run: 0.5961; run_001 dir deleted, runs 002–004 v4 partial aborts** | 28/50 (+2 wext) | 40/50 (missing=10) | exp_099_schema_graph_fallback_001 (dir deleted; runs 002–004 v4 partial) |
| exp_101_rich_preamble | ✓+profile(per-col)+fulldata(150k) | 32 | ✓ | ✓+fence3+Rules1-14+Rule16+Rule17(min_steps=4)+runner min_steps=4 guard+rich profile preamble (profile.py 171行, CSV/SQLite/JSON per-col dtype+quartile+top5, [preamble:rich], **single-attempt** T=0.6, workers=10, exp_086 base) | **0.7039 ± 0.018 (n=2) [0.6794, 0.7284]**   \|   1run last: 0.7164 | avg 33.5/50 | avg 47/50 (missing=3/run) | exp_101_rich_preamble_001–002 |

**⚠ exp_026/027 はリーク**: FIXED_EXAMPLES の 3 例 (Example A=task_145, B=task_19, C=task_173) が公開 50 タスクの question・gold を verbatim 引用。task_173 は exp_023 で zero_recall → exp_026 で perfect 化（リーク経由 +1）。**真のベスト = exp_023_glossary_preamble (0.6837)**。**A-board / B-board の隠しテストではリーク無効化されるので、submission の最終選定は exp_023 から行う**。

⚠ exp_006 と exp_007 (旧) は **API 502 大量混入**（並列実行で vLLM 上限が overload、`api_502` が exp_006=24 / exp_007=21 件）。  
✅ exp_007 を **単独 + retry 付き** で再実行した結果が **0.6510** (32 perfect)。~~これが現状ベスト。~~  
✅ exp_012_tool_tolerance が **0.6710** (33 perfect) で**旧ベスト**（+0.020 vs exp_007）。`action_input` 文字列→`{"code": str}` 自動変換で `__error__` −78%。  
✅ exp_017_tie_aware_superlative が **0.6752** (λ0.0=0.7000!) で**旧ベスト**（+0.0042 vs exp_012）。superlative filter-back 例示が task_352 を zero→perfect に救済。副作用として wext +3（task_259/330 退行）。  
❌ exp_008 は `enable_thinking: False` を kobushi_core/model.py に投入し workers=16 でスモーク → **思考が浅くなり action_input フォーマット違反を 31 連続繰り返す致命リグレッション**。即 revert。  
❌ exp_010 (verify-before-execute) は exp_007 から **−0.169 の大幅後退** (0.4820 vs 0.6510)。verify ブロックの 3 項目が thought を肥大化させて step 予算を圧迫し、missing が 6 → 12 に倍増、perfect も 32 → 23 に減少。**AIMO 3 (Nitarach) の "プロンプト工夫は天井" 観察が我々のデータでも実証された**。系列 A (reason 軸) への追加投資は ROI 低い。  
❌ exp_011 (spec-driven 3-phase + DuckDB) は exp_007 から **−0.124 の後退** (0.5270 vs 0.6510)。JSON 崩壊ループが主因 — DuckDB コードを `answer_from_python` に渡す際に改行・クォートが JSON シリアライズを壊し、`__error__` 総数が +89%（204→385 steps）。**根本対策は tool input tolerance（D1）**。  
❌ exp_013 (sample-rich preamble) は exp_012 から **−0.031 の後退** (0.6400 vs 0.6710)。preamble が JSON ファイルを SQLite テーブルと混同させ task_11/task_269 でクエリ失敗。ファイル種別ラベルなしで value_sample を追加すると逆効果。

✅ exp_023_glossary_preamble が **0.6837** (33 perfect) で**旧ベスト**（+0.0085 vs exp_017）。knowledge.md glossary 抽出・preamble 挿入で perfect +2。  
✅ exp_026_fewshot_pathfix が **0.7242** (λ0.0=0.7400!) で**現ベスト**（+0.0405 vs exp_023）。3-shot 例示パス修正で task_173/199/86 が新規 perfect。  

`auto-answer tools` = `answer_from_python` (DataFrame 自動展開) + `answer_from_sql` (SQL 結果を直接 answer 化).

**🥇 リーク無しベスト (確定 win): `exp_086_r1_official_params`（n=2 mean λ0.5=0.7353 ± 0.0135, CI [0.7167, 0.7540]）**（旧ベスト n=3 verified: `exp_081_union_t0` mean=0.6906, std=0.0138）（1-run record: `exp_040_selfdbg_fence` λ0.5=0.7267 は 1-run outlier、n=3 clean mean=0.6633 ⚠️）（リーク込み最高: `exp_026_fewshot_pathfix` λ0.5=0.7242 だが Tier 1 リーク確定・隠しテスト無効）





### exp_016_forbid_manual_answer (λ0.5 = 0.6510 ❌ regression vs exp_012, Δ −0.020)
- ベース: exp_012 をコピー → `prompt.py` に Rule 10「`answer` は 1×1 literal のみ」追加 + 旧推奨末尾段落削除、`tools/registry.py` の `_answer` に multi-row/col 時 soft warning（ok=True + content["warning"]）を追加。
- 結果: 0.6710 → 0.6510（Δ −0.020）、perfect 33 → 32（−1）、zero_recall 12 → 14（+2）。
- **仮説 D5 は不成立**: `answer` 呼び出し回数は exp_012 と同数（33 回 vs 33 回）。soft warning も prompt も無視された。
- **最大の退行 task_352**: exp_012 では `answer` 手書き（ratio=2.73）2 ステップで perfect → exp_016 では prompt 誘導で `answer_from_python` を 22 ステップ試みて誤答（0.6510）。
- **唯一の改善 task_418**: exp_012 missing → exp_016 perfect。
- **教訓**: 「`answer` を使うな」というプロンプト制約は 1×1 computation タスクを壊す（exp_010/011 と同パターン）。系列 D（tool) の直接プロンプト制約アプローチは ROI 低い。zero_recall 14 件の根本原因は wrong_value（計算誤り 7 件）と wrong_rows（行数誤り 5 件）で、`answer` 呼び出し起因ではなかった。

### exp_017_tie_aware_superlative (λ0.5 = 0.6752 ✅ 新ベスト vs exp_012, Δ +0.0042)
- ベース: exp_012 をコピー → `prompt.py` に Rule 10（superlative → filter-back パターン）追加 + RESPONSE_EXAMPLES に tie-safe SQL 例（`WHERE col = (SELECT MIN(col) FROM t)`）を追加。
- 結果: 0.6710 → 0.6752（Δ +0.0042）、λ0.0 = **0.7000**（70% マイルストーン達成）、zero_recall 12 → 11（−1）、wext 1 → 4（+3 副作用）。
- **成功**: task_352 が zero_recall → perfect（filter-back が直接効いた）。task_379 が zero_recall → wext（recall 達成、extra_col=1 で減点）。
- **未解決**: task_25/task_80 は改善なし（これらはタイ問題ではなく「解釈誤り」の可能性が高い）。
- **副作用 wext +3**: task_259（gold 1 列 Text のみ→pred 4 列フルレコード）、task_330（gold 2 列→pred 3 列 date 追加）が perfect → wext に退行。filter-back 例示が「確認列を含める」習慣を植え付けた副作用。
- **教訓**: 例示 SQL に `SELECT *` を使うと agent が余分列を含める可能性がある。次回は `SELECT <必要列のみ>` を例示するか、Rule 1/5 との組み合わせで「answer plan の列のみ SELECT」を強調する。wext 4 件が解消されれば λ0.5 ≒ 0.700 が届く。

### exp_018_fix_select_star (λ0.5 = 0.6620 ❌ regression vs exp_017, Δ −0.0132)
- ベース: exp_017 をコピー → Rule 10 インライン例示の `SELECT *` → `SELECT col1, col2 FROM t WHERE sort_col = (SELECT MIN(sort_col) FROM t)` に変更、「Never use SELECT *」注記追加。RESPONSE_EXAMPLES は変更なし。
- 結果: 0.6752 → 0.6620（Δ −0.0132）、wext 4 → 3（部分改善）、zero_recall 11 → 12（+1 悪化）。
- **wext は部分改善**: task_38(9→4 extra)、task_259(3→2 extra) は改善。task_330 は変化なし。task_379 は wext→zero に悪化（SELECT 絞り過ぎでフィルタ崩壊）。
- **task_194 新規 regression**: perfect → zero（3 列提出）。「Never SELECT *」が bond テーブルの全列を answer_plan に列挙する過剰反応を誘発。
- **教訓**: wext の根本原因は Rule 10 の例示文体ではなく、agent の column_count 計算ミス（答えに不要な列を SELECT する）。「SELECT * 禁止」という prompt 制約は task_194 のような「正しく 1 列を取り出すべきタスク」を壊す典型的な過規制。wext 解消はプロンプト制約ではなく、answer_plan の column_count 遵守を強化する別アプローチが必要。

### exp_019_column_count_coercion (λ0.5 = 0.6087 ❌ regression vs exp_017, Δ −0.0665)
- ベース: exp_018 をコピー → `agent.py` の `run()` で step_index==1 の thought から `column_count=N` を正規表現抽出し、`answer_from_python`/`answer_from_sql` 呼び出し時に `__max_columns__: N` を inject。`tools/registry.py` の 2 ハンドラで `len(columns) > N` なら先頭 N 列に truncate。
- 結果: 0.6752 → 0.6087（Δ −0.0665）、wext 4 → 2（部分改善）、zero_recall 11 → 16（+5 大幅悪化）、`__error__` 率 10.9% → 15.2%（逆行）。
- **wext 部分改善のみ**: task_38(9→4) は改善。task_259/330/379 は改善なし or 悪化。
- **新規 zero_recall +5 件**: task_194（3 列→截断→0 col）、task_259（3 列截断→zero）、task_379（行数誤り→zero）、task_11（step0 __error__ で coercion 不発→wrong_rows）、task_??? 等。
- **設計欠陥確定**: agent の thought 内 `column_count` は gold 正解列数を保証しない。agent の計画ミス（column_count を多く宣言）に対して truncate を重ねると召喚された zero が更に増える。「agent の宣言を信頼して truncate」というアプローチは agent 精度に上限される。
- **教訓**: D6 コード側 coercion は「agent の計画が正しい前提でのみ機能」する。wext の根本は「agent が答えに不要な列を計画に含める」認知問題であり、外部からの強制 truncate では解消できない。次は agent の計画精度そのものを改善する（B3 glossary 教示 / A4 question paraphrase）か、wext と無関係な missing 削減（D2）へ切り替えること。

### exp_020_truncate_offload (λ0.5 = 0.6402 ❌ regression vs exp_017, Δ −0.0350)
- ベース: exp_017 をコピー → `tools/filesystem.py` の `read_csv_preview` / `read_json_preview` / `read_doc_preview` の戻り値が 4KB を超えたら `/tmp/kobushi_offload_<task>_<step>_<file>.txt` に full content を書き出し、observation には `{"preview": "<最初 200char>...", "offload_path": "/tmp/...", "full_size": N}` を返す。新ツール `read_offloaded(path, lines, columns)` を追加して agent が部分再取得できるように。
- 結果: 0.6752 → 0.6402（Δ −0.0350）、missing 4 → 3（−1 = 救済成功）、zero_recall 11 → 14（+3 悪化）、perfect 31 → 30（−1）。
- **D2 設計の欠陥**: 4KB 閾値の preview が agent から見ると「データの一部」として中途半端。agent は preview を見て「もう十分」と判断したり、`read_offloaded` を使い切れずに recall を落とす。preview と full の二段階提示が agent の混乱を招く。
- **特に regression が大きいタスク**: 中規模 CSV（5〜20KB）で preview だけで answer に走り、データ完全性を損なう pattern。task_173/89/200 等の wrong_value で previously perfect だったタスクが zero に転落。
- **教訓**: context 圧迫対策はオプトイン（agent が明示的に巨大ファイルを宣言した時のみ）にすべき。デフォルトで preview 化するのは agent の習慣を破壊する。改修するなら「ファイルサイズが 50KB+ のときだけ preview」など、稀なケース限定にする。task_344/396/420 の missing 救済は別アプローチ（max_steps 増加 / 軽量 fallback agent）を検討。

### exp_021_question_paraphrase (λ0.5 = 0.6617 ❌ regression vs exp_017, Δ −0.0135)
- ベース: exp_017 をコピー → `prompt.py` の Rules 末尾に Rule 11 追加（Paraphrase A/B/Chosen 形式）と RESPONSE_EXAMPLES 例示更新。`agent.py / registry.py` には変更なし。
- 結果: 0.6752 → 0.6617（Δ −0.0135）、perfect 31（同数）、wext 4 → 3（−1）、zero_recall 11 → 13（+2）、missing 4 → 3（−1）、`__error__` 率は微減 11.9% → 11.4%。
- **wext −1 と missing −1 の救済はあるが zero_recall +2 で相殺**: paraphrase の強制二択が、ある 2 タスクで誤った解釈に倒した（agent が Paraphrase A vs B のうち間違った方を Chose）。
- **task_25/80（タイ問題）は引き続き未解消**: paraphrase はタイ問題には効かず、構造的な解釈ミス系（task_163, task_352）でも解決できなかった。
- **教訓**: A 系列 [reason] は exp_010 verify-before-execute と同じパターンで失敗する。「思考量を増やすと選択肢の幅も増えて誤選択リスクも増える」というトレードオフが prompt 制約系の根本的限界。**A 系列 [reason] axis は完全に凍結すべき**。次は構造的に違う axis（cascade Phase 2 / few-shot E1 / BoN I3）を試すこと。

### exp_017〜exp_021 の総括（5 連続実験での学び、2026-05-02）
- **wext 攻撃 (exp_017→018→019) は ROI ゼロ**: 3 世代で全て regression を生む。task_259/330/38 は agent の認知的限界による column_count 計画ミスであり、prompt/registry/coercion のどれでも解消困難。**exp_017 の wext=4 (λ0.5=0.6752) が現時点での最良トレードオフ**。
- **D2 truncate-and-offload (exp_020) も ROI ゼロ**: missing 1 件救済の代償に zero_recall +3 件発生。デフォルト挙動を変えると agent の習慣が破壊される。
- **次の有望軸**:
  - **B3 (knowledge.md term extraction)**: task_163 の chronic 解釈ミスを辞書ベースで矯正（preamble 拡張のみ、副作用最小）
  - **B1 (schema-first agent)**: wrong_rows 多すぎ系 (task_180/199/379) を schema 検査強制で抑制
  - **I3 (BoN with GenSelect)**: 既知の I 系列再挑戦（exp_009 から学ぶ）
- **避けるべき軸**: prompt 制約系（exp_010/016/018 で全滅）、cascade（exp_014/015 で大失敗）、column_count coercion（exp_019 で大失敗）。

### exp_022_cascade_v3 (λ0.5 = 0.3117 ❌ large regression vs exp_017, Δ −0.3635)
- ベース: exp_017 をコピー → cascade 2 段構成に置換（Phase 1: 探索エージェント max_steps=24、Phase 2: DuckDB 専用エージェント max_steps=12）。3 つの設計修正（Fix #1〜5）を全て適用。
- 結果: 0.6752 → 0.3117（Δ −0.3635）、perfect 31 → 15（−16）、zero_recall 11 → 29（+18）、missing 3 → 5（+2）、wext 4 → 1（−3）。
- **失敗原因 1 — EXTRA_COLS（17 件）**: shape_mismatch guard の attempt 2 無条件受容が「諦め提出」退路になった。`{table, records}` JSON ラッパーを SELECT * で取得した 2〜9 列がそのまま受容される。task_26/80/352 等が全て zero_recall に転落。
- **失敗原因 2 — WRONG_VALUE（10 件）**: Phase 2 は DuckDB 専用 1 ツールのみ。JSON unnesting 失敗時に `answer_from_python` / `answer` フォールバックが存在しないため脱出路なし。exp_017 系列が同ケースで多様なツールで逃げられていたのに対し完全にロック。
- **失敗原因 3 — MISSING_PREDICTION（5 件）**: DuckDB エラーループが 12 ステップ使い切り後も続く。`answer` ツール不在で手書きフォールバック不可。task_19（SQLite DB）、task_173/344/396/420（DuckDB エラーループ）。
- **新規 win 2 件**（task_200, task_330）は cascade の設計優位性ではなく偶然。Phase 1 の独立探索フェーズが偶然正しい join 計画を立案した副産物。
- **教訓**: cascade アーキテクチャは「Phase 2 の唯一ツールが DuckDB」という設計前提が致命的。DuckDB が失敗した場合の退路（answer_from_python / answer）がなければ、Phase 2 の多ステップは逆に EXTRA_COLS / MISSING を増やすだけ。**F 系列 cascade は設計欠陥確定・再試行不要**。シングルエージェント (exp_017 系列) に回帰し、B3 (knowledge.md glossary) / I3 (BoN GenSelect) / B1 (schema-first) を次の有望軸とする。

### exp_023_glossary_preamble (λ0.5 = 0.6837 ✅ 新ベスト vs exp_017, Δ +0.0085)
- ベース: exp_017 をコピー → `preamble.py` のみ変更。`_extract_glossary()` で `- **term**: def` パターン抽出（regex）、`_build_glossary_section()` で知識ファイル読み込み・800 chars truncate、`_assemble()` の Workspace overview ↔ File details 間に条件挿入。
- 結果: 0.6752 → **0.6837**（Δ +0.0085）、λ0.0=0.7000（同数）、perfect 31 → **33**（+2）、wext 4 → 2（−2）、zero_recall 11 → 9（−2）、missing 3 → **6**（+3 悪化）。
- **成功**: task_196（zero→perfect、金融用語の文書理解改善）、task_330（wext→perfect、wext 解消）。
- **退行**: task_352（perfect→missing、glossary 注入が doc/budget.md を JSON 構造化データと誤認させ 32 ステップ消尽）。task_379/418 が 0 ステップ（実行環境問題の可能性、exp_017 も missing）。
- **task_352 退行の詳細**: exp_017 では `read_doc` で budget.md を直読み → 4 ステップで完答。exp_023 では preamble の glossary に "budget"/"spent"/"amount" 等の金融用語が入り、agent が doc ファイルを JSON 解析しようとした。`Expecting ',' delimiter` を 23 回繰り返し全ステップ消尽。
- **残課題**: WRONG_VALUE 9 件（task_80/25/89/169 等）はすべて計算ロジック誤り。知識注入では解消困難。missing 6 件のうち task_352 は glossary 副作用で退行、task_420/180 は継続問題。
- **教訓**: B3 glossary injection は preamble 改善として有効（+0.0085）だが、doc/ ファイルを含むタスクでは「文書をデータ処理対象と誤認する」リスクがある。次の有望軸: I3 (BoN GenSelect) / B1 (schema-first) / missing 削減 (task_352 の退行修正)。

### exp_024_glossary_v2 (λ0.5 = 0.6233 ❌ regression vs exp_023, Δ −0.0604)
- ベース: exp_023 をコピー → `preamble.py` のみ変更。`_DOC_EXTENSIONS = frozenset({".md", ".txt"})` 定数追加、`_build_doc_hint_section()` で `.md`/`.txt` ファイルをリストして「use read_doc, NOT a JSON parser」と注記、`_assemble()` の Glossary 直後に条件挿入。
- 結果: 0.6837 → 0.6233（Δ −0.0604）、perfect 33 → 29（−4）、wext 2 → 3（+1）、zero_recall 9 → 15（+6 大幅悪化）、missing 6 → 3（−3 改善）。
- **部分成功**: task_352（missing→perfect、doc/budget.md の JSON 解析ループ防止）、task_379（missing→wext）。
- **退行 task_249/200/257/269/259**: doc_hint が全 50 タスクに適用される設計欠陥が致命的。全タスクが `context/knowledge.md` を持つため、「knowledge.md — use read_doc」という注記が全タスクのプリアンブルに追加された。これが SQLite + JSON のみで解けるタスクでも agent の attention を knowledge.md に引き寄せ、不必要な JSON parse error ループを誘発（task_249 で 30 回繰り返し）。
- **task_249 ↔ task_352 スワップ現象**: doc_hint は問題を「移動」させただけ。task_352 が復活した代わりに task_249 が missing に転落。net zero。
- **設計欠陥**: `_build_doc_hint_section` は knowledge.md を常にリストアップする。doc_hint を有効にするなら「`doc/` ディレクトリのファイルのみ」か「知識 MD 以外のテキストファイルのみ」に絞る必要があった。
- **教訓**: preamble への注記挿入は「全タスクに何かが追加される」という副作用を常に考慮すること。B3 系列（glossary 拡張）はここで凍結。knowledge.md 自体はプリアンブルの File details に既に含まれており、agent はそれを読んで理解できる。task_352 の問題は「glossary が金融用語を注入したこと」が原因であり、「doc ファイルの種別ヒント」ではなく「glossary の対象タスク限定」が正しい修正方向だった。

### exp_025_fewshot_3ex (λ0.5 = 0.6517 ❌ regression vs exp_023, Δ −0.0320)
- ベース: exp_023 をコピー → `prompt.py` のみ変更。`FIXED_EXAMPLES` 定数（task_145/19/173 の実 gold 例 3 件）を追加し、`build_system_prompt()` の `RESPONSE_EXAMPLES` 直後に挿入。
- 結果: 0.6837 → 0.6517（Δ −0.0320）、perfect 33 → 32（−1）、wext 2 → 2（±0）、zero_recall 9 → 9（±0）、missing 6 → 8（+2 悪化）。
- **部分成功**: task_86（zero→perfect、multi-row JOIN 例示が効果）、task_352（missing→perfect、exp_023 の退行が解消）。
- **主要退行原因 — パス模倣バグ**: `FIXED_EXAMPLES` 内の SQL パスを `"context/database.sqlite"` 形式で記述したため、モデルが全タスクで `context/xxx` プレフィックス付きパスを模倣。10 タスクで `Missing context asset: context/xxx` エラーが発生（exp_023 では 1 件のみ）。
- **0 ステップタイムアウト +2 件**: task_249（166MB context）、task_259（183MB context）が steps=0 で 600s タイムアウト。exp_023 の 2 件に加え 2 件増加。巨大 context でのプリアンブルスキャンコスト増大が疑われる。
- **fewshot コンセプト自体は有効**: 前述の 2 件の win はいずれも few-shot 例示の直接効果。パス表記修正（`context/` プレフィックスなし）で exp_026 を試す価値あり。
- **教訓**: few-shot 例示内のリソースパスは実際のエージェント呼び出しと同じ相対パス形式（`database.sqlite` または `db/events.sqlite`）にすること。`context/` プレフィックスを含めると全タスクに模倣される。巨大 context タスク（150MB+）は別途タイムアウト対策が必要。

### exp_026_fewshot_pathfix (λ0.5 = 0.7242 ✅ 新ベスト vs exp_023, Δ +0.0405)
- ベース: exp_025 をコピー → `prompt.py` の `FIXED_EXAMPLES` のみ変更。`context/database.sqlite` → 実タスク固有パス（`db/event.db` / `club.db` / `db/transactions_1k.db`）に修正、冒頭に「Do NOT copy these paths literally」注記追加。
- 結果: 0.6837 → **0.7242**（Δ +0.0405）、λ0.0=0.7400、perfect 33 → **35**（+2）、wext 2 → 2（±0）、zero_recall 9 → 9（±0）、missing 6 → **4**（−2）。
- **成功**: task_173（extra_cols→perfect）、task_199（extra_cols→perfect）、task_86（extra_cols→perfect）— いずれもパス模倣バグ除去により正しいファイルを参照できるようになった結果。
- **退行 task_257**: perfect → extra_cols（zero_recall=0）。`json/posts.json`（159MB）を直接 open してヒットせず 0〜3 ステップで誤答。SQLite（`postHistory.db` 263MB）を正しく辿れなかった。few-shot 例示に大規模 JSON 複数ファイル混在タスクのパターンが含まれていないため短絡。
- **0 ステップタイムアウト**: exp_025 の 4 件から 2 件に改善（task_249/259 が復活）。task_25/418 の 2 件が依然タイムアウト。
- **残課題**: WRONG_VALUE 9 件（task_80/89/163/169/344 等）は計算ロジック誤りで few-shot 改善では解消困難。task_257 は大規模 JSON + SQLite 混在タスクのカバレッジ不足。
- **教訓**: 実パスを例示することが模倣誘導を防ぐ鍵。「パスは模倣するな」注記は有効。次候補: task_257 退行の fix（大規模 JSON 混在例示追加）or 別軸（I3 BoN GenSelect）。

### exp_027_loop_break (λ0.5 = 0.7067 ❌ regression vs exp_026, Δ −0.0175)
- ベース: exp_026 をコピー → `agent.py` のみ変更。`run()` に `_prev_obs_hash` による同一 observation 連続検出を追加（`loop_detected: True` + `loop_hint` 付加）。`__error__` ブランチではハッシュ更新なし。
- 結果: 0.7242 → 0.7067（Δ −0.0175）、λ0.0=0.7200、perfect 35 → 34（−1）、wext 2 → 2（±0）、zero_recall 9 → 13（+4 悪化）、missing 4 → 1（−3 改善）。
- **loop_break は実質無効**: `loop_detected` が発火したのは task_420 の 1 件のみ（7 回）。task_420 は exp_026/027 両方で 32 ステップ missing のまま変化なし。実際のループ（task_330 型の execute_python サイズチェック）は observation が毎ステップ微妙に変化するため同一 observation 検出では捉えられない。
- **fewshot 例と公開テストの完全一致が致命的退行原因**: Example C = task_173 そのもの。モデルが同一の問いを認識し Example C のショートカット SQL（`transactions_1k.db` + `2013-06` filter）を直接実行 → 該当データが存在せず 0 行 → 誤答（zero_recall）。exp_026 では 9 ステップの探索で `csv/yearmonth.csv` から 2013-06 データを発見して正解していた（探索が必要なタスク）。
- **重大教訓**: fewshot 例に公開テストと完全一致するタスクを使うと、モデルがショートカット実行を試みて正解率が下がるリスクがある。Example A/B/C（task_145/19/173）はすべて公開データセットのタスク — 今後の fewshot 例は**テストセットに含まれない**典型例に変更する必要がある。
- **missing −3 改善**: exp_026 の task_25/418 が 0 ステップタイムアウトから復活（インフラ競合消滅と推定）、miss → zero_recall に変化。これは loop_break の効果ではなくインフラ要因。
- **H1 系列（同一 observation ループ検出）は設計欠陥確定**: 実際のループは observation が微変化するため効果なし。loop_break 軸は凍結。

### exp_028_fewshot_synthetic (λ0.5 = 0.6631 ❌ regression vs exp_026, Δ −0.0611)
- ベース: exp_026 をコピー → `prompt.py` の `FIXED_EXAMPLES` のみ変更。公開テストと完全一致する 3 例（task_145/19/173）を架空 DB（`bookstore.db` / `hr.db` / `hospital.db`）の合成例に全置換。
- 結果: 0.7242 → 0.6631（Δ −0.0611）、λ0.0=0.6800、perfect 35 → 32（−3）、wext 2 → 2（±0）、zero_recall 9 → 14（+5 悪化）、missing 4 → 2（−2 改善）。
- **task_173 ショートカット問題は解消**: 合成例に task_173 は含まれないため、モデルが同一の問いを認識してショートカット実行することはなくなった。
- **新たな致命的問題 — SQL 一辺倒バイアス**: 合成例 3 件が全て `answer_from_sql` のみを使用する純 SQL パターン。モデルがこのパターンを強く学習し、JSON/Python/doc 処理が必要なタスクでも SQL 経由を試みた。task_11（JSON JOIN → execute_context_sql で parse error ループ）、task_196（read_csv 比率計算 → SQL 失敗）、task_199（frpm.csv JOIN → 454 行爆発）、task_200（wrong filter）が新規退行（perfect → zero/wext）。
- **task_420 初 perfect**: 28 ステップ/496 秒、execute_python × 20 の長い探索で成功。ただし他タスクへのリソース圧迫が全体スコアを下げた可能性がある。
- **E1 fewshot 軸の根本的制約が明確化**: fewshot 例は単なる「形式ガイド」ではなく「ツール選択・アプローチパターンのバイアス注入」として機能する。公開例（exp_026）は偶然にも多様なツール使用が例示されていたが、合成例（exp_028）は SQL 一辺倒だった。
- **教訓**: fewshot 例を設計する際は（1）テストセットと重複しない、（2）ツール多様性を持つ（SQL / Python / doc 各 1 件以上）、の両方を満たす必要がある。現在の E1 軸は設計制約が厳しく、追加投資の ROI は低い。**I3 BoN GenSelect / B1 schema-first など別軸に切り替えを推奨**。

### exp_029_schema_first (λ0.5 = 0.7067 ❌ vs LEAK exp_026 Δ −0.0175, but ✅ vs leak-clean exp_023 Δ +0.0230)
- ベース: exp_023 をコピー → `prompt.py` のみ変更。Rule 11「SQL 書く前に `inspect_sqlite_schema` でスキーマ確認」を追加、`RESPONSE_EXAMPLES` に schema-first デモ例を追加。
- 結果: 0.6837 → 0.7067（Δ +0.0230 vs exp_023 leak-clean）、λ0.0=?、perfect 33 → 34（+1）、zero_recall 9 → 9（±0）。
- **Rule 11（schema-first）は部分的に有効**: task_257 で SQL スキーマ確認が奏功、テーブル名を正確に特定して SQL 成功（exp_026 比で win）。
- **FIXED_EXAMPLES 削除が退行の主因**: exp_026 比で task_86（off-by-one JOIN）、task_199（JOIN 爆発 57 行 vs gold=6 行）、task_200（count=4 vs gold=1）、task_259（3 列 vs gold=1 列）が退行。exp_023 ベースなため fewshot の恩恵がない。
- **Rule 10 は task_80 に不適用**: 「Q3 タイム」は最上位クエリではなく exact-match。Rule 10 の適用条件設計が不十分で、agent は row_count=1 と予測し誤答。Rule 10 の効果は限定的。
- **exp_028 比 +3 perfect**: task_11/196/352 が回復（SQL 一辺倒バイアスの解消）。
- **task_420 が 26 ステップ perfect**: `execute_python` × 26 の探索で成功。
- **教訓**: Rule 11 (schema-first) は task_257 への直接効果確認済み。ただし exp_023 ベースでは fewshot なしの制約がある。**次の推奨**: exp_026 ベース + Rule 11 のみ追加した exp_030 で fewshot × schema-first の組み合わせを試す（ただし LEAK 問題は残る → 真に評価するには合成 fewshot のツール多様性問題も解決する必要あり）。あるいは deterministic cascade（PRIORITIES.md URGENT）を優先。

### exp_030_leak_free_baseline (λ0.5 = 0.6431 ✅ リーク無し clean base)
- **目的**: exp_017〜exp_029 系列は全て Tier 2 構造リーク（RESPONSE_EXAMPLES の student_club domain priming）を含むため、リーク無しの真のベースラインを確立する。
- ベース: exp_029 から `FIXED_EXAMPLES`（student_club 系スキーマ priming）を完全削除し、library/weather の clean examples のみ残した版。Rule 11 (schema-first) は維持。
- 結果: 0.6431（exp_023 clean base 0.6837 の見かけ上の差は fewshot 削除分）、perfect 31/50、answered 48/50。
- **リーク無し真のベースライン確立**: submission/main.py の EXPERIMENT_NAME を exp_030 に更新済み。
- **教訓**: fewshot examples を削除しても exp_012 (0.6710) より若干下回る程度。Rule 11 (schema-first) の +0.023 効果は維持される。exp_031 以降はこれをベースに iterate する。

### exp_031_deterministic_cascade (λ0.5 = 0.6531 ✅ 新リーク無しベスト vs exp_030, Δ +0.010)
- ベース: exp_030 をコピー → `preamble.py` を全面書き換え。トークン上限 8k→150k、ファイル種別ごとに専用ヘルパー（`_csv_full`/`_sqlite_full`/`_json_full`/`_doc_full`）を新設、`_budget_cut()` で優先順に削減（json > csv > doc > sqlite > knowledge_full）。preamble 冒頭に `## Note for agent` 追加。
- 結果: 0.6431 → **0.6531**（Δ +0.010）、perfect 30/50 (+4 wext)、answered 48/50。
- **部分成功**: 150k preamble が full context を agent に提示。探索ステップが減り、答えに向かう直行率が上がった。
- **wext +4 が課題**: 大量のコンテキストが agent に余分な列を含ませるバイアスを生んだ可能性。exp_032 の structured spec (expected_row_count / answer_columns) で修正を狙う。
- **残存失敗**: 意味解釈ミス 7 件（正しいフィルタ/集計列の取り違え）、行数過多 5 件、タイムアウト 2 件（task_344/396）。
- **教訓**: deterministic preamble の情報注入効果は実証された（+0.010）。次段階の exp_032 では structured spec + 診断的 self-correction で wrong_value/wrong_rows 系を削減する。

### exp_032_cascade_self_correct (λ0.5 = 0.6067 ❌ large regression vs exp_031, Δ −0.046)
- ベース: exp_031 をコピー → `config.yaml` max_steps 32→48、`prompt.py` に構造化 spec (Rule 1: `## Spec` / answer_columns / expected_row_count) + Rule 12 (診断的 retry) 追加、`tools/answer_validator.py` 新規作成、`tools/registry.py` に validator 注入、`agent.py` に record_thought 1 行追加。
- 結果: 0.6531 → **0.6067**（Δ −0.046）、perfect 30→29（−1）、missing 2→7（+5）、`__error__` ステップ ~25→93（+68）。
- **根本原因 (JSON パースエラー)**: Qwen3.5-35b が `｝"｝`（末尾余分 `"｝` 付加）を多発。構造化 spec prompt が長い出力を誘発し悪化。task_80 では 40 ステップ中 38 回が `__error__`、推定 930 秒のタイムロスでタイムアウト急増。
- **self-correction が逆効果**: task_292 で `spec_wrong_likely` ヒントが正解クエリを誤修正させた。validation_hint 発火: spec_wrong_likely 10回 / row_count_mismatch 3回 → 有効改善 +3 tasks に対し regression −6 tasks。
- **唯一の収穫**: task_420 の DuckDB 空文字バグ修正で +1。task_396 は preamble が 262k tokens 超で 400 エラー（コンテキスト上限問題）。
- **教訓**: 複雑な structured spec prompt は JSON パースエラー率を大幅悪化させる。診断的 self-correction ループの設計は正しいが、モデルへの prompt 負荷削減が先決。exp_034 では手書き answer 撤廃 + simpler validation でリトライ。

### exp_033_profile_table (λ0.5 = 0.5917 ❌ large regression vs exp_031, Δ −0.061)
- ベース: exp_032 をコピー → `tools/filesystem.py` に `profile_table()` 追加（CSV/SQLite::TableName/SQLite-all-tables 対応、dtype/null率/cardinality/min-max/top5、1500chars 段階削減）、`tools/registry.py` に `_profile_table_handler` 登録、`prompt.py` に Rule 13 追加（profile_table 使用推奨）。spec parsing (answer_validator / record_thought) は exp_032 から継承。
- 結果: 0.6531 → **0.5917**（Δ −0.061）、λ0.0=0.6000、perfect 30/50、missing 2→**11**（+9）、`__error__` ステップ ~25→**121**（+96）。
- **新種バグ (closing fence なし)**: モデルが ` ```json {...}` を出力するが **閉じる ` ``` ` を生成しない**→ regex がマッチせず空文字 → `"Expecting value: line 1 column 1"` 71 回発生。task_420 では 48/48 全ステップがこのエラーで answer 未提出（missing）。
- **タイムアウト急増 (2→11)**: 10 tasks が steps=0 — 内 3 件 (task_243/249/259) は posts.db (137MB) サンプリング遅延。他はパースエラーループによる 8 workers のリソース飢餓（間接的 CPU 枯渇）。
- **profile_table はほぼ未使用**: 50 tasks 中 4 tasks のみ使用。探索効率向上効果はほぼゼロ。
- **唯一の収穫**: WON task_86/196/200（+3）。非 missing の zero_recall は 14→9（−5）に改善 — exp_032 バグが再顕在化しなければ推論品質自体は exp_031 超の可能性。
- **教訓**: closing fence なしエラーは新種。**`_strip_json_fence` の regex を「閉じ fence 必須」→「なくてもOK」に変更することで 71 件を救済可能**。exp_032 の構造化 spec prompt が依然として出力を不安定化させている（exp_032 ベースだったため引き継ぎ）。exp_034 では spec parsing ごと廃止して問題を構造的に解消。

### exp_034_no_manual_answer_validated (λ0.5 = 0.5756 ❌ large regression vs exp_031, Δ −0.078)
- ベース: exp_033 をコピー → `tools/answer_validator.py` を spec parsing から出力直接バリデーション（5 チェック）に全面書き換え、`tools/registry.py` から `answer` 削除・`answer_from_duckdb` 追加、`prompt.py` Rule 4/5/6 + RESPONSE_EXAMPLES を「手書き禁止」に更新。
- 結果: 0.6531 → **0.5756**（Δ −0.078）、perfect 30/50（±0）、missing 2→5（+3）、`__error__` ステップ ~25→87（+62）。
- **新種バグ (f-string JSON エスケープ失敗 70 件)**: モデルが `answer_from_python` のコード文字列中に f-string を書くと JSON `"` の escape が壊れ `"Expecting ',' delimiter"` エラー。task_303 が最多（17 ステップ中 10+ 回）。closing fence / trailing `}"` は解消、代わりに f-string エスケープが露出した。
- **4 tasks 新規クリア**: task_173/196/379/420 が `answer_from_python` / `answer_from_duckdb` 経由でクリア — `answer` 撤廃の恩恵。
- **task_26 で answer_from_python 形式バグ**: `pd.DataFrame(answer_table)` が `{"columns":...,"rows":...}` dict を そのまま DF 化 → prediction が `columns,rows / count,[1]` に（regression）。
- **Unknown tool エラー 2 件**: モデルが `execute_sql` を生成（`execute_context_sql` が正解）。ツール名の類似性による誤生成。
- **タイムアウト継続 (2→5)**: task_38/249/283/344/418。特に task_418 は 48 ステップ全消費（execute_python 42 回ループ）。max_steps=48 が悪化要因。
- **教訓**: JSON 内 Python コードのエスケープ失敗は `answer_from_python` が主力である限り継続する根本課題。exp_035 の fence fix で closing fence / trailing `}"` は解消済み。f-string エスケープは `_load_single_json_object` のコード部分への特別処理が必要。**max_steps を 32 に戻すことも優先度高**。

### exp_035_fk_map_fence_fix (λ0.5 = 0.5539 ❌ large regression vs exp_031, Δ −0.099)
- ベース: exp_034 をコピー → `agent.py` に Case 3（unclosed fence）を `_strip_json_fence` に追加、`preamble.py` に `_build_fk_map_section()`（`PRAGMA foreign_key_list`）を追加して `_sqlite_full()` に挿入。
- 結果: 0.6531 → **0.5539**（Δ −0.099）、perfect 30/50（±0）、missing 2→6（+4）、`__error__` ~25→69（+44）、total_extra_cols 31→41（+10）。
- **fence fix は部分成功**: trailing `}"` エラー ~25→2 ✅、unclosed fence 0→0 ✅。総 `__error__` は exp_034 の 87 から 69 に改善。ただし f-string エスケープ失敗が 60 件残存（変形していない）。
- **FK Map が extra_cols を悪化**: JOIN 関係を preamble に明示したことでモデルが関連列を過剰出力。total_extra_cols +10、λ=0.5 では exp_034 より悪化（0.5756→0.5539）。FK Map は **廃止確定**。task_259（extra+6）・task_38（extra+5）が代表例。
- **タイムアウト継続 (2→6)**: task_11/25/249/303/396/418 が steps=0。max_steps=48 継続 + f-string ループが worker スレッドを飢餓状態にする間接原因。
- **WON (3)**: task_200/379/420 — fence fix の恩恵（exp_034 でクリアしていた task_379/420 は継承）。
- **教訓**: fence fix（Case 3）は価値あり — exp_031 ベースに持ち込む価値がある。FK Map は廃止。**max_steps=32 回帰が急務**（4 連続で 48 → regression 連続）。f-string エスケープは system prompt での禁止 instruction（Rule 追加）が最もシンプルな対策。

### exp_036_explicit_select_cols (λ0.5 = 0.6833 ✅ 新リーク無しベスト vs exp_031, Δ +0.030)
- ベース: exp_031 をコピー → `prompt.py` に Rule 12「先行列計画 (column_count + per_column) + SELECT 列名明示 + SELECT * 禁止」を追加（Rule 11 直後に挿入）。`config.yaml` max_steps=32 を維持（exp_032〜035 の 48 から回帰）。
- 結果: 0.6531 → **0.6833**（Δ +0.030）、λ0.0=**0.700** (+0.020)、perfect 32/50 (+3 wext)、answered 44/50（missing 6）、total_extra_cols 31→**18**（−42%）。
- **extra_cols 大幅削減**: total_extra_cols 31→18（−13）、tasks_with_extra 18→12。Rule 12 の先行列計画が agent に SELECT 列数を自己制限させた直接効果。λ=0.5 で +0.030 の大半はこの要因。
- **zero_recall −5 改善**: 14→9。task_173/196/379 が新規 perfect（f-string ループ 60→3 への回帰と max_steps=32 復帰が主因）。
- **WON (3)**: task_173（f-string エラー解消）、task_196（列計画が正確な 1 列選択を誘導）、task_379（same）。
- **LOST (2)**: task_259（column_count=1 と計画したが正しくない 1 列を選択、exp_031 は全列返却で偶然 perfect_with_extras）、task_408（SQL 内 ROUND() を実行→精度喪失、Rule 12 の「値を確定してから answer に渡す」指示がモデルの早期丸めを誘発）。
- **unclosed fence 再発（trailing_brace +9）**: exp_031 の `_strip_json_fence` には Case 3 がない（exp_035 で追加したが exp_036 は exp_031 ベース）。exp_037 で Case 3 移植が必要。
- **missing 2→6**: task_180/199/344/396/418/420 の 6 件 — ただし 031 でもこれら全て recall=0 or タイムアウトであり pure regression はなし。JSON-only 6 タスク中 `file is not a database` エラーが新出（knowledge.md に "Database" が含まれ `inspect_sqlite_schema` 試行を誘発）。
- **教訓**: Rule 12（先行列計画 + SELECT 明示）は extra_cols 削減に高い ROI（−42%、λ0.5 +0.030）。ただし (1) task_408 型の早期丸め誘発リスク、(2) task_259 型の wrong-column 選択（列数は正しいが中身が違う）、(3) unclosed fence Case 3 欠落の 3 点が残課題。**exp_037 は exp_036 ベース + unclosed fence Case 3 移植で trailing_brace 9→0 を狙う**。

### exp_037_lenient_json (λ0.5 = 0.5717 ❌ large regression vs exp_036, Δ −0.112)
- ベース: exp_036 をコピー → `agent.py` の `_strip_json_fence` に Case 3（unclosed fence: ` ```json {...}` without closing ` ``` `）追加、trailing junk trimmer を英数字チェックに強化。
- 結果: 0.6833 → **0.5717**（Δ −0.112）、perfect 32→28（−4）、zero_recall 9→11（+2）、missing 6→10（+4）。
- **主因: fence Case 3 が貪欲マッチしすぎ** — `re.search(r"```(?:json)?\s*(.*)", text, flags=DOTALL)` が通常の JSON 出力の ` ``` ` prefix を誤認識し、正常 JSON からも不要部分を削ぎ落として破損。
- **教訓**: unclosed fence の修正は Case 3 が **DOTALL で全文 gobble しないよう** 非貪欲マッチが必要。exp_037 の実装は逆効果。fence 修正は exp_036 ベースで慎重に再実装すること。

### exp_038_function_calling (λ0.5 = 0.5764 ❌ large regression vs exp_036, Δ −0.107)
- ベース: exp_037 をコピー → OpenAI tool-calling プロトコル（function calling）に全面移行。ReAct テキスト JSON パースを廃止し、`tools=[...]` パラメータ経由でツール呼び出し。
- 結果: 0.6833 → **0.5764**（Δ −0.107）、perfect 27/50（+3 wext）、missing 11（+5 vs exp_036）。
- **主因**: OpenAI function calling では思考 (reasoning) を tool 引数に埋め込めず、モデルが列計画を事前に立てられない。Rule 12 の「先行列計画 thought」が機能しなくなり extra_cols が再増。
- **教訓**: qwen3.5-35b-a3b は ReAct テキスト形式（thought → JSON action）の方が適している。function calling は思考ブロックを分離しない限り列計画 Rules と相性が悪い。**function calling 軸は凍結**。

### exp_039_answer_validation (λ0.5 = 0.4117 ❌ catastrophic regression vs exp_036, Δ −0.272)
- ベース: exp_037 をコピー → Rule 12 の SELECT 明示を `answer_from_sql` への instruction に追加。
- 結果: 0.6833 → **0.4117**（Δ −0.272）、perfect 20/50、missing 22！ — 過去最悪。
- **主因: exp_037 の fence Case 3 バグを継承** + answer validation 追加 prompt がステップを消耗させ missing 急増。
- **教訓**: exp_037 の fence バグが連鎖している。exp_038 系列は exp_037 ベースを引き継いだ時点で既にアウト。**exp_039_answer_validation は廃止。**

### exp_039_preamble_sqlite_cap (λ0.5 = 0.5917 ❌ regression vs exp_036, Δ −0.092)
- ベース: exp_038_selfdbg_sql_precision をコピー → `preamble.py` に `_SQLITE_FILE_SIZE_LIMIT = 10MB` キャップ追加。10MB 超の SQLite は DDL のみ出力し「use execute_python to query」を注記。
- 結果: 0.5917（Δ −0.092 vs exp_036）、perfect 29/50（+1 wext）、missing 9。task_344/396/418 の大容量 SQLite タイムアウトは **改善できていない可能性あり**（ベース自体が exp_038_selfdbg 系の preamble を使っており、exp_036 直系でないため比較困難）。
- **教訓**: exp_039_preamble_sqlite_cap 自体の SQLite cap 効果は exp_036 直系で再検証が必要。cap コード自体は正しい実装であり、exp_036 ベースへの移植価値は残る。

### exp_041_inline_spec (λ0.5 = 0.6275 ❌ regression vs exp_036, Δ −0.056)
- ベース: exp_038_selfdbg_sql_precision をコピー → Rule 1 の answer plan を `## Spec` ブロック（data_sources / answer_columns / expected_row_count）に拡張。
- 結果: 0.6833 → **0.6275**（Δ −0.056）、perfect 29/50（+3 wext）、missing 9（+3 vs exp_036）。
- **WON (3)**: task_408（ROUND precision 問題を spec ブロックが誘導 — inline spec 仮説実証）、task_200（perfect）、task_259（wext に改善）。
- **LOST (7)**: task_11/196/379/420（ゼロリコール退行）、task_22/38/352（新規タイムアウト — exp_038 base preamble が遅く、SQLite size cap 未搭載の影響）。task_344/396/418 は全実験で一貫 timeout。
- **回帰の主因は exp_038 base 継承のタイムアウト** — inline spec 仮説自体は task_408 で実証。exp_036 直系ベースに Rule 1 inline spec のみ追加すれば WIN を保持しつつ task_22/38/352 の退行を解消できる可能性が高い。
- **教訓**: inline spec は計算精度タスクに有効。ベースライン選択が重要 — exp_038 ベース継承のコスト（preamble 遅延 + SQLite cap 欠落）が inline spec の利得を上回った。**次 = exp_036 直系 + inline spec のみ追加**。

### exp_040_selfdbg_fence (λ0.5 = 0.7267 ✅ 新リーク無しベスト vs exp_036, Δ +0.043)
- ベース: exp_038_selfdbg_sql_precision をコピー (≡ exp_036 + fence Case 3 + Rule 13 self-debug + Rule 14 ROUND outermost)。`config.yaml` max_workers=4。
- 結果: 0.6833 → **0.7267**（Δ +0.043）、λ0.0=**0.7400**、perfect 32→**35**（+3）、missing 6→**1**（−5）、answered 44→**49**/50。
- **WON (5)**: task_199（missing→perfect、timeout 解消）、task_259（zero→perfect）、task_408（zero→perfect、Rule 14 ROUND outermost 直効）、task_420（missing→perfect、fence Case 3 + Rule 13 self-debug の連携）、task_330（wext→perfect、extra_col 除去）。合計 Δ+4.17。
- **LOST (2)**: task_196（perfect→zero、connected.csv double-count 問題）、task_67（perfect→zero、weight=0 除外の誤フィルタ）。
- **fence Case 3 有効**: task_420 で malformed JSON __error__ を Rule 13 self-debug で 1 step 診断→正しい SQL 生成（32 step → 4 step 圧縮）。
- **Rule 14 有効**: task_408 の ROUND precision 喪失を直接解消。exp_036 では零→exp_040 で perfect。
- **missing 激減**: task_199/259/330/344/396 がすべて完走。max_workers=4 + fence fix の相乗効果。
- **残課題**: task_344/396 は完走したが doc truncate で zero_recall → `search_doc` ツール (exp_042) で解決期待。task_196/67 の regression は Rule 追加で対応可能。
- **教訓**: fence Case 3 + Rule 13 (self-debug) + Rule 14 (ROUND) は exp_036 直系への 3 合体として有効。リーク無しベスト大幅更新 (+0.043)。**exp_042 ベースは exp_040 とすること**。

### exp_040_selfdbg_fence (λ0.5 = 0.7267 ✅ 新リーク無しベスト vs exp_036, Δ +0.043)
- ベース: exp_038_selfdbg_sql_precision をコピー → モジュール名 rename のみ (実質ゼロ diff)。exp_038_selfdbg はすでに fence Case 3 + Rule 13 (self-debug) + Rule 14 (ROUND outermost) を実装済み。
- 結果: 0.6833 → **0.7267**（Δ +0.043）、λ0.0=**0.7400**、perfect 35/50（+2 wext）、answered 49/50（missing 1: task_418 のみ）、total_extra_cols=19。
- **WON (5)**: task_408（ROUND precision 直接効果）、task_420（fence Case 3 解消）、task_259（fence 修正で完走 → perfect）、task_199（same）、task_330（wext → perfect）。
- **LOST (2)**: task_196（exp_036 では正確な 1 列選択だが exp_040 では計算値ゼロ退行）、task_67（exp_036 では正解したが exp_040 で列選択ミス）。
- **missing 6→1**: task_199/259/330/344/396/420 が完走。task_418 のみ残存（大容量 SQLite + 複雑クエリ）。
- **教訓**: fence Case 3 + Rule 13 (self-debug) + Rule 14 (ROUND outermost) の 3 合体が高い ROI を示した。残 2 LOSS (task_196/67) は計算ロジック誤りであり fence/ROUND とは無関係 — 別軸での対策が必要。**新リーク無しベスト確定。submission を exp_040 に切り替える**。

### exp_042_inline_spec_clean (λ0.5 = 0.6517 ❌ exp_040 比 −0.075)
- ベース: exp_036_explicit_select_cols (クリーン直系)。変更: Rule 1 → `## Spec` inline block + Rule 13 ROUND outermost + fence Case 3 (非DOTALL) + preamble 10MB SQLite cap。
- 結果: 0.7267 → **0.6517**（Δ −0.075 vs exp_040 現ベスト）、perfect 35→**32**（−3）、zero_recall 12→**15**（+3）、missing 1→**2**（+1）、extra_cols 19→**27**（+8）。
- **WON (2)**: task_196（inline spec が double-count 構造を明確化 → perfect）、task_200（atom 集計 spec 明確化 → perfect）。合計 Δ+2.00。
- **LOST (6)**: task_420（Rule 13 self-debug 欠如 → malformed JSON error 後に迷走 13 step）、task_173（spec anchoring `row_count=0` が早期探索終了を誘発）、task_259/287（spec が冗長列を促進 → extra_cols 増加）、task_199/379（exp_036 base のタイムアウト再発）。
- **根本原因**: exp_042 = exp_036 ベースのため Rule 13 self-debug が欠如。task_420 では fence Case 3 解消後に malformed JSON エラーが出たとき自己診断なしで 13 step 迷走 → zero_recall。exp_040 では Rule 13 が同問題を 4 step で解決していた。
- **inline spec の評価**: task_196/200 で計算精度向上は実証。しかし spec anchoring による探索制限 (task_173) と extra_cols 増加 (task_259/287) の副作用が上回った。**exp_040 ベースに inline spec を追加すれば Rule 13 で task_420 退行を防ぎつつ WIN を保てる可能性がある → exp_043 以降の複合実験候補**。
- **教訓**: exp_036 直系は max_workers=8 でも Rule 13 欠如の影響が顕在化する。**新 exp の必須要件: ベースは必ず exp_040 を使うこと**。inline spec は単独では ROI がマイナス — Rule 13 との組み合わせが必要。

### exp_043_doc_search (λ0.5=0.6483, 31/50, ❌ rejected — Δ−0.078 vs exp_040)
- **仮説**: search_doc(path, query) ツール追加 + preamble doc 30K→5K 縮小で task_344/396/418/379 の巨大 doc タスクを解決。
- **WON (1)**: task_67 (+1.00) — weight=0 フィルタ不在で偶発ヒット (構造的でない)。
- **LOST (5)**: task_173/199/259/292 (各−1.00) — preamble 5K 縮小による通常タスクの情報欠落。task_379 (−0.75) — Python `'count'` KeyError。
- **巨大 doc 4タスクはすべて未解決**: task_344 は 0 ステップタイムアウト (max_workers=8 CPU 競合)、task_396 は部分抽出のみ、task_418 は閾値特定不可 (29回 search_doc)、task_379 は Python bug。
- **根本原因 1 (最大)**: preamble 30K→5K 縮小が doc 系以外の通常タスクにも副作用。task-adaptive 縮小 (巨大ファイルのみ) が必要。
- **根本原因 2**: max_workers=8 で CPU 競合 → task_344 が preamble 生成中にタイムアウト。
- **根本原因 3**: search_doc は keyword lookup に有効だが数値閾値判定 (task_418) や大規模集計 (task_396) には不十分。
- **教訓**: doc 縮小は task-adaptive にすべき。preamble 全体縮小は禁止。search_doc は維持するが doc truncation 戦略の再設計が必要。

### exp_044_inline_spec_v2 (λ0.5=0.7067, 34/50, ❌ rejected — Δ−0.020 vs exp_040)
- **仮説**: exp_040 ベース + Rule 1 を inline `## Spec` ブロックに置換 → task_200/196 で double-count 修正。
- **WON (2)**: task_200 (+1.00) — inline spec で 1 列明確化 → 正しく count=1 返却。task_67 (+1.00) — Rule 13 で read_csv 切替、weight=0 除外なしの avg が gold 一致。
- **LOST (3)**: task_194 (−1.00) — `answer_columns: [bond_id, atom_id, atom_id2]` で 3 列返却、gold は 1 列。task_199 (−1.00) — Riverside filter で 12 行返却、gold は 6 行。task_173 (−0.50) — max_workers=8 CPU 競合で 0 ステップ 600s タイムアウト。
- **task_196 は未解決**: inline spec でも double-count 問題は解消されず。明示 Rule (bidirectional は unique bond のみ) が必要。
- **根本原因**: max_workers=8 が task_173/180 に CPU 競合タイムアウトを引き起こした。exp_040 は max_workers=4 で task_173 を perfect に解いている。
- **inline spec の net effect (max_workers=4 前提)**: task_194 回避を除けば +2 wins、task_173 timeout 解消で +1 → 純益 ≈ +0.02。ただし task_194 の answer_columns 過剰問題の解決が必須条件。
- **教訓**: max_workers=8 は CPU 競合タスク (task_173/180) に致命的。inline spec 再試は max_workers=4 + task_194 対策 (answer_columns 列挙を制限する Rule) とセットで行うこと。

### exp_045_adaptive_doc_search (λ0.5=0.6083, 29/50, ❌ rejected — Δ−0.118 vs exp_040)
- **仮説**: exp_040 ベース + adaptive doc truncation (>30KB → 5KB) + search_doc ツール追加 → task_344/396/418/379 の巨大 doc タスクを解決。
- **WON (1)**: task_67 (+1.00) — stochastic win (doc 変更無関係、exp_040 の誤フィルタが消えて偶然一致)。
- **LOST (8)**: task_75 (−1.00, 0 steps timeout) / task_173 (−1.00, 9 steps CPU 圧迫) / task_199 (−1.00, 12行 vs gold 6行) / task_22 (−1.00, row_count=1 自己アンカー) / task_259 (−1.00, 誤列計画) / task_379 (−0.75, 2列 vs gold 1列) / task_420 (−1.00, self-debug スキップ) / task_330 (−0.17, extra_col)。
- **巨大 doc タスクは全て未解決**: task_344 (0 steps CPU 競合) / task_396 (600s timeout — 750件集計不可) / task_418 (zero recall — 閾値推定不可) / task_379 (zero recall)。search_doc はキーワード検索のみ有効。大規模集計・数値閾値推定には Python 全件処理が必要。
- **主因**: max_workers=8 が task_75/173/420 に CPU 競合タイムアウト。4イテレーション連続で同じ失敗を繰り返した (exp_043/044/045)。
- **副発見**: task_67 で avg 計算時の「weight=0 除外しない」教訓。Rule 追加が必要 (avg で 0/null を明示指示なしに除外禁止)。
- **教訓**: adaptive doc + search_doc は巨大 doc タスクへの解にならない。巨大 doc は Python で全件読み込み + 集計が唯一の解。max_workers=8 は絶対禁止 (CPU 競合で複数タスクに致命的タイムアウト)。

### exp_046_tool_error_truncate (λ0.5=0.6798, 32/50, ❌ rejected — Δ−0.047 vs exp_040)
- **仮説**: exp_040 ベース + traceback 最後 5 行切り詰め + error-type 別 fix hint → Rule 13 self-debug の context rot を防ぎ error 回復率向上。
- **WON (1)**: task_67 (+1.00) — stochastic win (NaN のみ除外して avg=60.78 正答。exp_040 では weight>0 フィルタで誤答)。Rule 化なければ次回も不安定。
- **LOST (5)**: task_173 (−1.00, 4 steps で打ち切り、execute_python join を試みず) / task_199 (−1.00, Riverside 誤カウント 18 steps) / task_259 (−0.43, CSV 全列返却 gold=1列) / task_330 (−0.17, extra_col 1件) / task_379 (−0.75, 2列返却 gold=1列)。
- **全退行が stochastic**: max_workers=4 正常使用。タイムアウト退行はゼロ。退行は同一条件下の確率的発散。
- **error hint 効果なし**: 20 回発動のうち 16 回 (80%) が汎用メッセージ `"Try: read the error..."` → 具体的誘導効果なし。`no such table/column` hint のみ有意だが Rule 11 (inspect_schema) と重複。
- **教訓**: error hint パターンマッチは浅すぎる。hint を精緻化するなら `sqlite3.DatabaseError` / `pandas.errors` など具体的 exception class 別に分岐が必要。ただし根本的には stochastic 変動が支配的で prompt 変更の効果が見えない。

### exp_047_skill_split

**仮説**: Anthropic Skills パターン — Rules 10-14 を system prompt から外部ファイルに移し、agent が `read_skill(name)` で遅延取得することで context 圧縮とルール適用精度を両立できる。

**結果**: λ0.5 = **0.6917** ❌ (Δ −0.035 vs exp_040, perfect=34/50, zero=11, missing=4)

- **`read_skill` 呼び出し回数: 0 / 50 タスク** — モデルは skill システムを完全に無視
- Rules 10-14 を prompt から除去したため Rule 13 (self-debug) が失われ、task_199 が JSON parse error ループで 32 steps 使い切り missing
- **missing が 1→4 に増加**: task_80/199/344/418 — Rule 13 欠如によるループ停止不能が主因
- task_196 win / task_67 win は両方 stochastic (read_skill 未使用)
- extra_cols = 19 変化なし (sql_round/explicit_select の効果も read_skill 未呼び出しで検証不能)
- **教訓**: Qwen3.5 MoE (3B active params) は間接的な「エラー後にスキルを読め」2ステップ命令に従わない。ルールは system prompt に直接書く必要がある。**skill_split アイデアは廃棄確定**。

### exp_048_inspect_table (λ0.5=0.6083, 29/50, ❌ rejected — Δ−0.118 vs exp_040)

**仮説**: exp_040 ベース + `inspect_table` ツール (dtypes + null% + top5 values を単一呼び出しで返す) + Rule 15 (use inspect_table before writing queries) → schema 把握の改善でクエリ精度向上。

**結果**: λ0.5 = **0.6083** ❌ (Δ −0.1183 vs exp_040, perfect=29/50, zero=15, missing=4)

- **inspect_table 呼び出し: 14 タスク / 正の効果: 0 タスク**
- 8 つの新規 regression のうち 5 件 (62.5%) で inspect_table が使用された
- **直接的な害が確認されたケース (2 件)**:
  - task_243: `inspect_table(votes.csv)` が `UserId null%=91.2%` を返す → モデルが「null が多すぎる」と早期打ち切り。実際には 8 件の非 null votes が存在。exp_040 は `read_csv` で発見 → 正答
  - task_350: `inspect_table(member.csv)` 呼び出し後に `answer_from_python` 選択 → SQLAlchemy バージョンエラー → フォールバックで member_id リストを誤提出。exp_040 は COUNT=7 で正答
- **task_67 (avg filter) は今回も win**: Rule 15 は機能したが stochastic 成分が大きく (今回は weight>0 フィルタが出ず)、Rule 単体での決定的効果は未検証
- **教訓**: inspect_table の null%/dtype 情報がモデルの探索を収束させるどころか早期打ち切りを誘発する逆効果。schema 把握情報は「答えを狭める」ヒントになりうる。`inspect_table` ツール追加軸は廃棄。

### exp_049_targeted_rules (λ0.5=0.6817, 32/50, ❌ rejected — Δ−0.045 vs exp_040)

**仮説**: exp_040 ベース + Rule 15 (avg/sum で 0/null を明示なしに除外禁止) + Rule 16 (双方向関係は MIN/MAX dedup で unique pair カウント) → task_67/196 の known-failure を deterministic に修正。

**結果**: λ0.5 = **0.6817** ❌ (Δ −0.045 vs exp_040, perfect=32/50, zero=14, extra_cols=2, missing=1)

- **Rule A (task_67 fix) ✅ 成功**: 4/4 連続 perfect → deterministic 修正達成。weight=0 を avg から除外しない
- **Rule B (task_196 fix) ❌ 失敗**: answer=2.0 (vs gold=1.0)。Rule 16 文言「unique bond pairs」が適用されず。task_196 の正しい解法は「connected.csv の各行は (a→b) と (b→a) の 2 行で格納」を理解した上で MIN/MAX DISTINCT — より具体的な文言が必要
- **task_330/408 extra_cols 新発生**: exp_040 では 1 列正答だったが exp_049 で 3 列提出 — stochastic な SQL SELECT 整形の揺れ。Rule 1/6 強化 (exp_050) が対策
- **6 件の stochastic regression**: task_173/199/259/261/330/408 — Rule と無関係な確率的発散
- **教訓**: Rule A は keeper（exp_050+ に継承必須）。Rule B は文言強化が必要: "connected.csv は各ペアを (a→b) と (b→a) の 2 行で格納している — iodine atom の bond avg 計算時は MIN(id1,id2), MAX(id1,id2) で DISTINCT"

### exp_050_format_precommit ([reason] A1 Rule 1 COMMITMENT + Rule 6 drop-extra)

**仮説**: Rule 1 を answer plan の「COMMITMENT」に格上げ + Rule 6 に "drop extra columns before answer" を追加することで、answer 直前の extra_cols 提出を防ぐ。

**結果**: λ0.5 = **0.6483** ❌ (Δ −0.078 vs exp_040, perfect=31/50, zero=15, extra_cols=2, missing=2)

- **Rule 6 (task_408 fix) ✅ 成功**: 中間計算列 (champion_ms, last_ms) を削除 → 1 列正答
- **Rule 1 COMMITMENT ❌ 逆効果**: task_330 で初期 column_count=3 が誤っていたが "COMMITMENT" 言語のため "Revised plan:" を書かずに 3 列提出 → extra_cols。exp_040 では 12 ステップで自己修正して正答していた
- **stochastic regressions +6**: task_173/199/259 zero + task_420 missing (59MB DB 0-step timeout) — commitment と無関係な確率的発散
- **Rule A 未搭載**: exp_050 は exp_040 差分のみ。task_67 win は stochastic (Rule A なし)
- **教訓**: "COMMITMENT" という強い言葉が正当な plan 修正を阻害する。Rule 6 の "drop extra" は有効だが Rule 1 の commitment 語は削除すべき。Rule A は exp_051+ に継承必須

### exp_051_pre_answer_verify ([critique] G1 Reflexion 軽量版 — Rule 17 answer前 3行 question-alignment check + Rule 15)

**仮説**: answer 呼び出し前に "(1) Question asks for: / (2) My answer provides: / (3) Aligned: YES/NO" の 3 行 self-check を prompt で強制することで、正しいデータを取得しているのに最終回答で外れる stochastic zero_recall を救済する。

**結果**: λ0.5 = **0.6883** ❌ (Δ −0.038 vs exp_040, perfect=33/50, zero=11, extra_cols=2, missing=4)

- **Rule 15 (task_67) ✅ 継続**: 0 値除外禁止 — 6 連続 correct (exp_046〜051) で安定。副作用なし
- **Rule 17 ❌ 却下**: 14 タスクでトリガー / 改善 0 件 / regression 2 件 (task_259/379)。初期計画が間違っているケースの修正には無力 — "Aligned: YES" と誤判断してしまう
- **missing +3** (4/50 vs exp_040 の 1/50): Rule 17 の latency 増加でタイムアウト閾値付近のタスクがタイムアウト増加した可能性
- **教訓**: prompt のみの self-check では誤った初期計画を修正できない。[critique] 軸 G1 Reflexion 軽量版は廃棄。exp_dual_eval (agent.py 変更あり) が次候補

### exp_052_filetype_labels ([schema] B2 revised — preamble workspace overview に [CSV]/[SQLite]/[JSON]/[DOC] ラベル付加 + Rule 15)

**仮説**: preamble の workspace overview にファイル種別ラベルを追加することで、JSON/SQLite 混同 (task_11/269/350 型) を防ぎ file type confusion を解消する。Rule 15 は exp_049 confirmed keeper として継承。

**結果**: λ0.5 = **0.6283** ❌ (Δ −0.0983 vs exp_040, perfect=30/50, zero=15, extra_cols=2, missing=3)

- **filetype_labels 効果ゼロ**: 45/50 タスクでラベル表示確認 / 改善 0 タスク / regression 0 タスク（回帰はすべて stochastic）
- **Rule 15 (task_67) ❌ 非決定論的確定**: exp_052 で Rule 15 あるのに task_67 = 78.51 に戻った。exp_046〜051 の 6 連続正答は stochastic ストリーク — Rule 15 は task_67 を確率的にしか救済しない
- **stochastic regressions +5**: task_67/199/259/292/420 — filetype labels と無関係な確率的発散
- **根本的学習**: モデルが 1 ステップで `answer_from_python` のコードを生成する際、Rule 15 文言が適用されない。prompt ルールより code utility 注入で対策すべき
- **教訓**: filetype labels は中立だが不要（preamble を複雑にするため削除推奨）。[schema] B2 軸の ROI は低い。Rule 15 の確実化には code-level utility (preamble Python helper) が必要

### exp_053_rule6_drop_extra ([output] Rule 6 softened — extra columns を answer 前に drop する指示 + Rule 15)

**仮説**: Rule 6 を強化して extra columns を answer 前に drop する指示を追加することで、中間計算列の漏れ (task_408 型) を防ぐ。Rule 1 COMMITMENT 語は添加しない (exp_050 で有害確認)。

**結果**: λ0.5 = **0.5917** ⚠️ **API障害汚染** (raw; 補正後推定 0.6717、Δ −0.055 vs exp_040, perfect=33/50 推定, extra_cols=1/50)

- **⚠️ API 障害汚染**: 7 タスク (task_257/287/292/303/396/418/80) が 600s タイムアウト (preamble_metadata=None)。raw スコアは無効
- **Rule 6 drop-extra ✅ 確定効果**: task_408 で champion_ms/last_ms を drop → percentage_faster 1 列正答
- **task_330 ✅ 改善** (Rule 6 or stochastic): 初回計画が `column_count=2` で date 列なし → 正答。stochastic か Rule 効果か未確定
- **Rule 15 ⚠️ stochastic 継続**: task_67 win は stochastic（non-deterministic 確認済み。exp_054 safe_mean utility で対策中）
- **stochastic regressions**: task_173/199/259/420 で連続退行 — rule と無関係な確率的変動
- **教訓**: Rule 6 drop-extra は task_408 に確実に効く。API 障害汚染がなければクリーンスコアで判断可能。exp_054 safe_mean と組み合わせて exp_055 で合体実験候補

### exp_054_safe_mean_helper ([infra] code utility — helpers.py に safe_mean/safe_sum + preamble 導線 + sys.path fix)

**仮説**: preamble で `helpers.py` を workspace に書き込み `safe_mean()` utility を提供することで、task_67 の avg/0 除外問題を deterministic に修正する。

**結果**: λ0.5 = **0.6867** ❌ (Δ −0.0400 vs exp_040, perfect=33/50, zero_recall=15, extra_cols=22 total, missing=5)

- **safe_mean() は事実上無視された**: 50 タスク中 1 件 (task_249) のみ呼び出し。model は自前コードで `df.mean()` を使い続けた
- **task_67 win は stochastic**: model が偶然ゼロフィルタを省略した `df.mean()` を書いただけ。safe_mean() は使っていない
- **⚠️ API タイムアウト 4 件**: task_344/396/418/80 が 600s タイムアウト (exp_053 に続く継続汚染)
- **汚染除外クリーンスコア (46 タスク)**: exp054=0.7464 vs exp040=0.7899 → Δ −0.0435
- **Wins (+2)**: task_67 (stochastic), task_86 (stochastic)
- **Losses (−4)**: task_11 (Thrombosis 誤フィルタ), task_173 (1 ステップ空 DataFrame), task_199 (32 ステップループ max_steps 超過), task_259 (index エラーループ)
- **教訓**: code-utility injection は Qwen3.5 MoE に効かない。model は preamble の helper を無視して自前コードを書く。task_67 の stochastic 問題は依然未解決。**[infra] code utility 軸は廃棄**

### exp_055_rule16_bidir

- **目的**: connected.csv の双方向エッジ重複問題（task_196）への Rule 16 強化。sorted-pair deduplication パターンを SQL/Python 双方で例示
- **スコア**: λ0.5=0.7083 ❌ (Δ−0.0183 vs exp_040)
- **クリーンスコア** (47 tasks, API timeout 3件除外): 0.7535 vs exp_040=0.7518 → **Δ+0.0018（実質フラット）**
- **Wins (+3)**: task_67 (stochastic), task_196 (**Rule 16 bidir 確定 fix** ✅ 0.0→1.0), task_200 (stochastic)
- **Losses (−5)**: task_173 (空 DataFrame stochastic), task_199 (API timeout 600s), task_259 (wrong columns stochastic), task_330 (date col 混入—Rule 6 なし), task_379 (hydrogen 誤計上 stochastic)
- **教訓**: Rule 16 bidir は task_196 に対して **deterministic 効果確認**。**全後続実験に必ず継承すること**。net スコアは stochastic regressions に相殺された。API timeout 3件 (task_199/344/418) が引き続き問題。Rule 6 (drop extra cols) がないと task_330 で date 列混入が再発する。

### exp_056_error_explain

- **目的**: [critique] G2 — Rule 18 (ERROR ANALYSIS: / NEXT ACTION: structured フォーマット強制 + 連続 2 エラーでアプローチ変更義務化) + Rule 15 keeper
- **スコア**: λ0.5=0.6883 ❌ (Δ−0.0383 vs exp_040)
- **クリーンスコア** (47 tasks, API timeout 3件除外): 0.7323 vs exp_040=0.7730 → **Δ−0.0408 (最悪クリーンスコア)**
- **Wins (+3)**: task_67 (stochastic), task_86 (stochastic), task_200 (stochastic)
- **Losses (−5.83)**: task_173 (YYYYMM 日付形式未解決), task_199 (完全誤クエリ 89行), task_259 (wrong columns), task_292 (**新回帰**: answer_from_sql 推奨で URL 未取得 → constructorId+points 誤 submit), task_379 (hydrogen 誤計上), task_330 (date col 混入—Rule 6 なし)
- **教訓**: Rule 18 は **24 errors 中 1 件のみ適用 (4.2%)** — Qwen3.5 MoE は format 指示を無視する。`answer_from_sql` 強推奨ルールは task_292 で **intermediate result の early submit を誘発**して新回帰。**[critique] G2 Rule 18 軸は廃棄**。task_292 対策として "non-numeric final column の場合は answer_from_sql 使用禁止" の Rule が必要。

### exp_057_combo_r6_r16

- **目的**: [output] Rule 6 (drop extra cols) + Rule 15 + Rule 16 (bidir dedup) を exp_040 ベースに合体した combo。+2 deterministic gain 理論値 (task_196 + task_408)
- **スコア**: λ0.5=0.6483 ❌ (Δ−0.0783 vs exp_040) — **API タイムアウト 6 件で重度汚染**
- **クリーンスコア** (44 tasks, timeout 6件除外): 0.7367 vs exp_040=0.7860 → **Δ−0.0492 (clean でも負け)**
- **Wins (+1)**: task_67 (stochastic)
- **Losses (−4)**: task_173 (stochastic), task_199 (zero_recall), task_259 (zero_recall), task_330 (extra col — Rule 6 は初期計画ミスに無効)
- **API timeout (6件)**: task_180 (32s→600s, 19x 悪化!), task_352, task_379, task_396, task_418, task_80
- **教訓**: Rule 6 は「計画後 extra_cols」には有効だが「初期計画ミス」には無効。task_330 の修正は stochastic。Rule 16 は task_196 で max_steps 超過により無効化。**prompt length 増加 (+1088 chars) が API timeout を悪化させている可能性が高い** — task_180 の 32s→600s 悪化は特に深刻。Rule 追加を重ねる戦略は prompt length 増加を通じて timeout を悪化させるリスクがある。

### exp_058_robust_sql_guard

- **目的**: [robust] Rule 19 (answer_from_sql ONLY for numeric/date/boolean cols; text/URL → answer_from_python) + Rule 6/15/16 継承で task_292 回帰を修正する
- **スコア**: λ0.5=0.6517 ❌ (Δ−0.0750 vs exp_040) — API timeout 2件 (task_344/418)
- **クリーンスコア** (48 tasks): 0.6788 vs exp_040=0.7569 → **Δ−0.0781 (実験群で最低水準)**
- **Wins (+1)**: task_67 (stochastic)
- **Losses (−5)**: task_173 (stochastic), task_199 (wrong district filter 24 rows), task_259 (wrong cols), task_330 (32 steps loop → missing), task_379 (stochastic)
- **Rule 19 の実態**: task_292 は exp_040 でもすでに perfect — Rule 19 は 0 効果で prompt を +1464 chars 肥大化するのみ。**Rule 19 廃棄確定**。
- **タイムアウト再評価**: exp_058 は最長 prompt (8327 chars) だが timeout 最少 (2件)。exp_057 (7951 chars, 6件) と逆相関 → **timeout は外部 API インフラ問題、prompt length との相関なし**。
- **教訓**: Rule 追加系 (Rule 6/15/16/19) は stochastic regressions を相殺できない。唯一確実な改善は **Rule 16 (task_196 fix) のみ**。exp_040 + Rule 16 のみが次の最小 baseline 候補。

### exp_059_dual_eval (Δ−0.0069 vs exp_040, λ0.5=0.7198) ❌

- **設計**: agent.py に `_eval_answer()` を追加（2nd LLM 呼び出し）; REJECT → answer クリア + step 継続; max 1 retry; max_steps=37
- **結果**: 47/50 tasks で eval 呼び出し; **REJECT 件数 = 0**。LLM はゴールドなしで正誤を判定できない → 全勝利は stochastic
- **task_199 タイムアウト**: eval 追加 LLM 呼び出しが時間予算を消費し 600s timeout → missing_prediction
- **Wins (+3 all stochastic)**: task_67 (avg fix), task_86 (rows fix), task_200 (atom count fix)
- **Losses (−5)**: task_199 (timeout by dual eval), task_173/379 (stochastic), task_259/330 (extra cols)
- **廃棄理由**: 追加 LLM 呼び出しはタイムアウトリスクを増やすだけ; Qwen3.5 MoE はゴールドなしで ACCEPT/REJECT 判定不能

### exp_060_search_doc_additive (Δ−0.0183 vs exp_040, λ0.5=0.7083) ❌ (≡exp_040)

- **実態**: exp_040 と import path 以外完全同一 ("search_doc" 機能は未実装のまま commit)
- **重要**: これは exp_040 の**リプレイ実験**として扱う → **stochastic baseline noise ≈ ±0.0183 (λ0.5) が実測確定**
- **Δ−0.0183 は全て stochastic noise**: 同一コードで同一 public 50 で測定した差異
- **Fragile tasks 確定**: task_199/259/330/379 は推論ごとに flip する → これら 4 tasks を除いた 46-task スコアで評価すべき
- **教訓**: 真の改善は Δ > +0.02 でないと noise から識別不能

### exp_061_r16_only (raw λ0.5=0.7083 ❌ / stable-41 Δ+0.0244 ✅ NEW BASELINE)

- **設計**: exp_040 ベース + Rule 16 (bidir sorted-pair dedup) のみ追加 (+589 chars)。Rule 15/6/18/19 なし
- **Rule 16 deterministic fix 確認** ✅: task_196 で bidir 認識 → sorted-pair dedup 適用 → answer=1.0 (gold=1.0) を trace 確認
- **stable-41 tasks**: exp040=0.7703 → exp061=0.7947 (Δ+0.0244)。**唯一の差異は task_196 のみ** — Rule 16 の影響が精密に分離確認。他 40 tasks は完全に同一スコア → Rule 16 に副作用なし
- **Raw 後退の原因**: stochastic losses 5件 (task_199/259/420/379/330) が wins 3件を上回った。task_420 が **new stochastic fragile** に確定
- **Fragile tasks 拡大**: 9 tasks (task_67/199/200/259/330/379/420/344/80) — これらを除く stable-41 が信頼できる評価指標
- **採用判定**: raw スコアは exp040 を下回るが、**Rule 16 = deterministic +1 fix confirmed。全後続実験に継承必須**。exp_061 を新実験ベースラインとして採用

### exp_063_bon_n2 (λ0.5=0.6317 ❌ — BoN N=2 直列実行はタイムアウト致命的)

- **設計**: exp_061 ベース + Best-of-N N=2 (T=0.0, T=1.0 直列実行) + column signature voting。`task_timeout_seconds=1200` (2×600s)
- **致命的問題**: 9 タスク (task_80/180/199/257/344/352/379/396/418) が 1200s でタイムアウト → `missing_prediction=9`（通常 1〜5 件）
- **stable-41 (λ0.5)**: 0.6972 vs exp_040 ~0.7899 (Δ−0.0927) — タイムアウトが stable tasks を侵食
- **Wins**: 0件（voting 機構は動作したが T=1.0 候補が T=0.0 と同じ誤答を返すケースが多い）
- **Losses**: 9件 missing + zero_recall 9件 (task_163/169/173/25/259/420/86/89/196 — extra col 系は BoN で解消されない)
- **教訓**: BoN 直列実行は per-task time budget を N 倍消費する。並列化なしでは A/B-board で致命的。**[multi] BoN 直列実行 (I1/I2/I3 含む) は廃棄確定**。並列 BoN は runner アーキテクチャ変更が必要で実装コスト大。

### exp_065_kira_double_confirm

- **仮説**: registry-side double-confirm でゼロコスト (prompt 変更なし) の answer 再考促進
- **結果**: λ0.5 = 0.7033 (exp_061 比 Δ−0.0050、stable-41 = 0.7703 = exp_040 と同値)
- **発火率**: 47/50 タスクでチェックリスト返却確認 ✅。revision 率 13/47 (28%)
- **revision の効果**: 13 件中 改善 0 / 同スコア維持 11 / **新回帰 2** (task_173: 1.00→0.00, task_194: 1.00→0.00)
- **task_196 win は Rule 16 由来**: double-confirm 由来ではなく bidir dedup Python 実装の確実性
- **timeout 増加**: 3 件 (exp_040/061 比 +2) — checklist 1 step 追加でギリギリタスクが落ちる
- **教訓**: チェックリストが model の正答を誤答に revision させる。強制再考はハザード。**[kira] double_confirm 廃棄確定**。

### exp_067_c1_col_minimize (n=4 pooled mean λ0.5=**0.6687** ❌ — Δ+0.0054 vs baseline = ノイズ確定、**col-minimize Rule 17 軸廃棄**)

- **設計**: exp_061_r16_only ベース + Rule 16 bidir dedup (sorted-pair MIN/MAX) + Rule 17 col-minimize ("Before answering, remove any column the question doesn't explicitly request")。⚠️ **インフラ縮退**: max_workers: 20→4、task_timeout_seconds: 900→600 (= 後続標準 timeout=900 より低い; regression source 候補)
- **diff**: 60 lines (`artifacts/diffs/exp_067_c1_col_minimize.diff`)

#### n=4 runs 詳細 (runs 005/006/007/008)

| run | λ0.5 | perfect | with_extras | zero | missing |
|-----|------|---------|-------------|------|---------|
| 005 | 0.6567 | 30 | 4 | 10 | 6 |
| 006 | **0.7067** | **34** | 2 | 11 | 3 |
| 007 | 0.6633 | 31 | 3 | 11 | 5 |
| **008 (追加)** | **0.6483** | 31 | 2 | 14 | 3 |
| **pooled n=4** | **0.6687 ± 0.0260** | 31.5 | 2.75 | 11.5 | 4.25 |

- **vs baseline (exp_040 n=3 clean mean=0.6633)**: Δmean=**+0.0054** (+0.21σ)、95% CI [0.643, 0.694] vs baseline CI [0.639, 0.688] — **完全重複 = ノイズ確定**
- **run-to-run spread**: 0.058 (0.6483〜0.7067)。run_006 が +1.4σ 外れ値疑惑。Rule 16/17 の prompt hit が run 間で不均一
- **reproducibility 評価**: stochastic flip tasks (run-to-run 大きく変動) = task_379 (0.75/0.75/0.75→0.0)、task_249 (timeout hit: 1.0/1.0/1.0→0.0)、task_14 など。Rule 17 col-minimize の prompt 適用は 57% 程度で deterministic でない
- **task_249 退行 (run_008)**: trace `failure_reason="Task timed out after 600 seconds"`, steps=0 → **600s timeout で agent が一度も実行されず cold-start**。前 3 run では 600s 内に完了。単純な確率的負荷揺れで境界を越えた — **timeout=600s の脆弱性** (exp_074 で task_22 が 900s で救済された事実と対応)
- **task_379 logic error (run_008)**: 6 steps 完了、agent が初回試行 `TR001_12/o` 系を捨てて 2 回目 `TR001_4/c` 系で回答 → gold signature と完全 mismatch。**positional-shift バグ** (Type B 構造的失敗と同類)
- **教訓**: col-minimize Rule 17 単体は n=4 でも Δ=+0.0054 = noise。timeout=600 設定が cold-start境界タスク (task_249) を引き起こした — **全後続実験で timeout=900 必須を再確認**。Rule 17 単独軸は廃棄確定 (exp_072 の min_steps+Rule 18 でも同結果)。C1 axis を再試行する場合は「step 1 plan 段階でのコミット列数制限」(prompt-only では初期計画固執に勝てない) が必要

### exp_068_kira_function_calling_v2 (λ0.5=**0.7367** ~~新リーク無しベスト~~ → +1.4σ outlier確定; n=3 rep2 mean=**0.6828** ⚠️ std=0.0411)

- **設計**: exp_061_r16_only ベース + `min_steps=4` 探索強制ガード。agent.py に `ANSWER_TOOLS` frozenset + step_index ≤ min_steps 判定。prompt に Rule 17 (4 exploration steps 義務) を追加。config.yaml に `min_steps: 4`
- **結果 (1-run 001)**: λ0.5 = **0.7367** (perfect 34/50, with_extras 4, missing 4 — 全て 0-step API cold-start timeout)
- **min_steps=4 直接確認 ✅**: task_67 trace で step 2 の early answer が "Too early (step 2 < min_steps 4)" でブロック → 追加探索 → perfect 化。early-answer blacking の直接効果確認済み
- **with_extras=4 (exp_061 比+2)**: extra-col penalty が増加。exp_067 の col-minimize Rule 17 が absent が原因。次実験は exp_068 + exp_067 col-minimize の combo が推奨
- **missing 4 件は全て 0-step**: task_22/180/344/418 が 600s timeout (最初の LLM 呼び出し自体が返らない) → vLLM API cold-start 問題。min_steps とは無関係
- **教訓**: min_steps=4 runner 強制は early-answer を物理的にブロックする有効軸。ただし stochastic な計算ロジック誤りには無力 (下記 n=3 再 replication 参照)

#### n=3 再 replication 詳細 (runs 006/007/008 — summary_20260504_080024.json)

| run | λ0.5 | perfect | with_extras | zero | missing |
|-----|------|---------|-------------|------|---------|
| 006 | 0.6717 | 33 | 1 | 11 | 5 |
| 007 | **0.7283** | **35** | 2 | 9 | 4 |
| 008 | 0.6483 | 31 | 2 | 15 | 2 |
| **新 n=3 mean** | **0.6828** | **33.0** | 1.7 | 11.7 | 3.7 |
| 旧 n=3 (003/004/005) | 0.6775 | 32.0 | — | — | 4.0 |
| **pooled n=7** | **0.6882** | 32.7 | — | — | 3.7 |

- **新 n=3 std=0.0411 (旧 0.0146 の 2.8×)**: run007 (+1.4σ) と run008 (−1.0σ) が 0.0800 swing。robustreplication ではなく stochastic variance が支配的
- **stochastic flip 5 タスク**: task_19/67/86/196/249/408 が 3 run 間で 0/1.0 切替。内訳:
  - task_196: run006/008 で naive `OR` double-count (Rule 16 未適用) → 2.0 vs gold 1.0。run007 のみ sorted-pair dedup 適用
  - task_67: run008 で weight=0 行を "0=missing" と誤解釈して除外 → avg 78.50 vs gold 60.78
  - task_86: run008 で余分 race 2 件混入 → rows 17→19
  - task_408: run008 で `0.3` (1 桁丸め) 出力、gold `0.3156`。Rule 14 outermost ROUND が効かなかった
  - task_19/249: run006 のみ失敗 (行数/値ミス)
- **構造的 zero 12 タスク** (3 run 全て 0): task_25/80/89/163/169/180/199/200/259/344/379/396/418 — min_steps=4 では救えない別軸
- **Rule 16 stochasticity**: exp_061+exp_068 合計 7 run で sorted-pair dedup 適用は 4/7 (57%)。"deterministic fix" claim は撤回。prompt rule 単体では 100% hit は保証できない
- **verdict: ⚠ likely_win** — pooled Δ=+0.0249 vs exp_040_clean (≥+0.02 閾値超えだが 95% CI 重複)。vs exp_081 = Δ−0.0024 (tie、std で完敗 0.0332 vs 0.0138)。**真の #1 は exp_081_union_t0**

### exp_069_kira_proactive_summarize (λ0.5=**0.6398** ❌ — Δ−0.0869 vs exp_040/exp_068 比 Δ−0.097)

- **設計**: exp_061_r16_only ベース + context_summarize_threshold=0.80。run loop で context chars を推定し、80% (≈91,750 chars) 超かつ step>4 の時に古い observation を LLM で要約して `__summarize__` StepRecord に置換
- **結果**: λ0.5 = **0.6398** ❌ (perfect 30/50, with_extras 2, missing 7)
- **要約発火 2 件のみ**: threshold=0.80 (91,750 chars) が高すぎて公開 50 の大多数タスクでは発火しなかった (task_420/415 の 2 件のみ)
- **タイムアウト急増 (4→7 件)**: 要約 LLM 追加コール (task_420: 574s、task_415: 565s) が長時間 CPU を占有 → ギリギリだった task_257 等が 600s タイムアウトに引き込まれた。max_workers=8 + timeout=600s と相性が悪い
- **stochastic regression +5**: task_67/173/199/259/330/379 等の fragile tasks が後退 — 要約機構とは無関係
- **教訓**: proactive context summarize は現状設定 (max_workers=8, timeout=600s) では逆効果。threshold を 0.50〜0.60 まで下げるか、max_workers=4 + timeout=900s 設定に変更しないと発火と副作用の両方が問題になる。**[kira] proactive_summarize 軸は廃棄確定** (少なくとも現状 infra では機能しない)

### exp_070_c4_dtype_canon (λ0.5=**0.6367** ❌ — Δ−0.090 vs exp_068、API 障害主因・Rule 18 効果評価不能)

- **設計**: exp_068_kira_function_calling_v2 ベース + Rule 18 (dtype 正規化: 数値 ROUND_HALF_UP 2桁 / null/NaN/missing → 空文字 / 日時 → ISO 8601)。prompt.py のみ変更、agent.py/config.yaml は exp_068 と同一
- **結果**: λ0.5 = **0.6367** ❌ (perfect 29/50, with_extras 4, missing 11 — vs exp_068 missing=4)
- **API 障害 (missing +7 の主因)**: task_408/418/344/396 で `steps=[]` → "Model response missing text content" (vLLM 空レスポンス)、task_420 で HTTP 502、計 6 件 timeout 連鎖。infra 由来で Rule 18 とは無関係
- **task_67 Rule 18 ambiguity バグ**: Rule 18 "null/missing → 空文字" をエージェントが "0 = missing" と誤解釈 → weight_kg=0 行を除外 → avg 78.51 vs gold 60.78。trace step 8 で "excluding NaN and 0" を明示確認
- **Rule 18 の正味効果**: API 障害で完全に評価不能。task_196 以外の改善・後退はすべて障害 or stochastic
- **教訓**: Rule 18 は "数値 0 は 0 のまま" という表現明確化が必要。再試行する場合は Rule 18 の文言を "null/NaN/missing → 空文字列 (数値の 0 は除外しない)" と修正する。**[output+reason] dtype_canon Rule 18 は要文言修正で再評価候補**

### exp_071_kira_obs_truncate (λ0.5=**0.6283** ❌ — Δ−0.108 vs exp_068、obs_truncate が no-op・min_steps=4 high-variance)

- **設計**: exp_068_kira_function_calling_v2 ベース + `_truncate_obs()` 関数 (registry.py): read_csv/execute_context_sql/execute_python の観測が 30,000 bytes 超の場合に rows リスト→output 文字列の順で切り詰め
- **結果**: λ0.5 = **0.6283** ❌ (perfect 30/50, with_extras 2, missing 3)
- **obs_truncate が完全 no-op**: 公開 50 タスクの全観測サイズを集計したところ、最大でも 13,721 bytes (task_330/read_csv)。30,000 bytes 閾値を超えた観測は **0 件**。truncation が一度も発火しなかった → 実質 exp_068 の stochastic 再実行
- **min_steps=4 high-variance**: task_67/11/196/200/379/259 が exp_068 比後退。min_steps=4 が正しい早期回答 (step 2) をブロック → 強制探索 → 誤ロジック選択。exp_066 (min_steps なし) は同タスクで正答しており、min_steps=4 の有効性は stochastic な探索品質に強く依存することが判明
- **task_67**: min_steps=4 が step 2 answer をブロック → "excluding NaN and 0" ロジック → avg 78.51 vs gold 60.78 (weight=0 除外バグが再現)
- **教訓**: 30KB 閾値は公開 50 タスクには高すぎる (実測最大 13,721 bytes)。obs_truncate を機能させるには閾値を 10,000 bytes 程度に下げる必要がある。しかし truncation が no-op でも min_steps=4 の variance でスコアが大きく揺れることが確認された。**[kira] obs_truncate 軸: 閾値見直しで再試行候補 (低優先)。min_steps=4 の variance を制御する方が先決**

### exp_072_c1_col_minimize_v2 (λ0.5=**0.7083** ❌ — Δ−0.028 vs exp_068、col_minimize が sticky extra に無効・min_steps×Rule18 干渉)

- **設計**: exp_068_kira_function_calling_v2 ベース + Rule 18 (col_minimize): "Before calling any answer tool, compare each output column to the question text. Remove any column that the question does NOT explicitly request. When in doubt, fewer columns are safer." prompt.py のみ変更
- **結果**: λ0.5 = **0.7083** ❌ (perfect 34/50, with_extras 2, missing 3)
- **with_extras 4→2 は misleading**: 削減された 2 タスクは task_259/379 が partial recall → recall=0 に後退したことによる。sticky extra tasks (task_38: 5 extras、task_330: 1 extra) は未修正のまま。Rule 18 は「初期計画への固執」(task_379: step 1 で column_count=3 を embed → 以降更新せず) に勝てない
- **task_86 regression (min_steps × Rule 18 干渉)**: step 3 が min_steps=4 でブロック → 強制再探索 → step 6 で別クエリが Japanese GP 重複カウント → 18 rows vs gold 16。exp_067 (min_steps なし) は step 3 の正答をそのまま採用して 1.0。min_steps が col_minimize の効果を打ち消す
- **task_352 stochastic failure**: action_input malformed → `__error__` 5連続 → read_doc×24 loop → max_steps=32 hit, no answer。exp_068 は同タスクで 17-step python 経路で 1.0 (stochastic)
- **min_steps と他軸の組み合わせ 2/2 negative**: exp_071 (obs_truncate) + exp_072 (col_minimize) いずれも exp_068 より後退。min_steps + X の組み合わせは回避推奨
- **教訓**: Rule 18 単体効果は exp_067 vs exp_072 から分離不能だが min_steps との組み合わせは確実に net negative。sticky extra (task_38/330) への対策は Rule 18 では不十分 — step 1 plan への override 指示が必要。**[output] col_minimize Rule 18 は単体 (min_steps なし) での再評価候補**

### exp_073_reason_min_steps_6 (λ0.5=**0.6683** ❌ — Δ−0.068 vs exp_068、min_steps=4→6 が timeout 増加・stochastic 後退を悪化)

- **設計**: exp_068_kira_function_calling_v2 ベース + `min_steps: 4 → 6` (config.yaml 1行) + Rule 17 文言更新 (`at least 6 exploration steps`、`step 7 returns an error`)
- **結果**: λ0.5 = **0.6683** ❌ (perfect 32/50, with_extras 2, missing 5)
- **missing 増加 (4→5)**: min_steps=6 が 2 追加強制 LLM コールを要求 → timeout 境界タスクを追加圧迫。task_257/396 等の timeout 境界タスクで missing 増加
- **perfect 後退 (34→32)**: min_steps=4 では正答できていたタスクで min_steps=6 ブロックが追加探索を強制 → 誤ロジック選択。task_67/196 等での stochastic 後退が増幅
- **min_steps 系の傾向確定**: exp_068(4) → exp_073(6) で min_steps 増加は missing と stochastic 後退の両方を悪化させる。min_steps 増加軸は廃棄確定
- **教訓**: min_steps を 4 以上に増やすと timeout タスクへの悪影響が 4 以下時より大きくなる。min_steps を 2 に下げる方向性 (forced divergence 削減) が次の有効仮説。**[reason] min_steps_6 廃棄確定**

### exp_074_infra_timeout_900 (λ0.5=**0.7083** ❌ — Δ−0.028 vs exp_068、timeout 延長は task_22 確実救済・min_steps variance が相殺)

- **設計**: exp_068_kira_function_calling_v2 ベース + `task_timeout_seconds: 600 → 900` (config.yaml 1行)
- **結果**: λ0.5 = **0.7083** ❌ (perfect 34/50, with_extras 2, missing 2)
- **timeout 延長の実証効果**: task_22 が cold-start により exp_068 で 600s timeout/0 steps → exp_074 で 187.6s/7steps 完走・1.0 救済確認。**timeout=900 は task_22 を確実に救済する構造的改善**
- **task_344/418 timeout 救済失敗**: 900s で完走したが wrong answer (LLM の誤選択)。時間確保後も LLM 品質問題は残る
- **stochastic net negative**: task_200/379/259 が exp_068 比後退 (min_steps=4 variance)。timeout 延長は無関係
- **純粋効果試算**: timeout 延長 +1.0 (task_22) vs min_steps variance −2.4 (task_200/379/259) = net −1.4/50 ≈ −0.028
- **重要結論**: **timeout=900 は今後の全実験に標準設定として継承すべき**。missing 4→2 は timeout 延長の直接効果。0.7083 スコアは min_steps=4 variance が原因であり timeout 延長を否定しない。**[infra] timeout=900 を baseline config として採用**

### exp_075_kira_obs_truncate_10k (λ0.5=**0.6683** ❌ — Δ−0.040 vs exp_074、obs_truncate 10KB は near-no-op・stochastic failures が主因)

- **設計**: exp_068_kira_function_calling_v2 ベース + `_OBS_MAX_BYTES=10_000` (registry.py: read_csv / execute_context_sql / execute_python の観測を 10KB 超で rows/output フィールド切り詰め) + `task_timeout_seconds=900`
- **結果**: λ0.5 = **0.6683** ❌ (perfect 32/50, with_extras 2, missing 3; task_199/292/396)
- **truncation 発火件数: 2/50 タスク**: task_396 (execute_python 29KB→10KB) + task_418 (read_csv 50rows→20rows)。公開 50 タスクの最大観測 13,721 bytes (task_330) は 10KB を超えたが task_330 は期待通りに発火確認
- **task_292 cold-start 後退**: e2e=900s steps=0。exp_074 では 22 steps で 1.0 を取得。同一 timeout=900s でも cold-start 時間は run 間で変動 (stochastic infra)
- **task_199 __error__ 27 連続ループ**: obs_truncate 関与なし (obs サイズ 176-2834b)。モデルが長い Python コード生成途中で文字列が切れ malformed JSON が連続発生。exp_074 は SQL-first 8 steps で正答
- **task_408 stochastic wrong calc**: answer=0.3010 (gold に不一致)。exp_074 は 0.3156 で 1.0。観測同一・final 計算のみ微妙に異なる (完全 stochastic)
- **task_418 truncation による情報損失の可能性**: read_csv 50 rows→20 rows に truncate → answer=7 (gold=1)。必要行が切り捨てられた可能性あり (未確定)
- **軸評価**: obs_truncate は exp_066 (30KB, 0.6883) / exp_071 (30KB, 0.6283) / exp_075 (10KB, 0.6683) で 3 連続後退。公開 50 タスクの典型的観測が 10KB 以下のため near-no-op。主要問題 (read_doc 63KB, task_352) は依然未カバー。**[kira] obs_truncate 軸は 3 連続後退で廃止推奨**
- **教訓**: obs_truncate を有効化するには read_doc ツールの truncation 追加が必要だが、ROI は低い (task_352 1件のみ)。min_steps=2 への変更で stochastic variance を低減する方が高優先。**[kira] obs_truncate 軸廃棄確定**

### exp_082_cursor_harness (mean λ0.5=**0.6174 ± 0.0281**, n=3 [0.5855, 0.6492] ❌ — Δ−0.0459 vs baseline 0.6633, preamble schema-only削減 + grep_file/stat_file 追加)

- **設計**: exp_081_union_t0 ベース + **preamble 全置換 (schema-only ~5K tokens)**: CSV shape+列名+dtype / SQLite 列名+dtype+行数 / JSON keys+size / DOC 先頭50行+見出しのみ → 実データ値を全廃。`grep_file` (キーワード grep + context lines) + `stat_file` (ファイルメタデータ) の 2 discovery tools 追加 + Rule 15 (schema-only 明示・discovery tools 使用義務) 追加 + `_ATTEMPT_TEMPS = (0.0,)` (single-attempt) + `max_workers = 20`。
- **n=3 個別スコア (runs 002–004)**:
  - run_002: λ0.5=0.5875, perfect=28/50, with_extras=2, zero_recall=10, missing=9, answered=41
  - run_003: λ0.5=0.6211, perfect=29/50, with_extras=3, zero_recall=11, missing=7, answered=43
  - run_004: λ0.5=0.6434, perfect=31/50, with_extras=2, zero_recall=8, missing=9, answered=41
  - mean=0.6174, std=0.0281, CI=[0.5855, 0.6492] (vs baseline CI=[0.6387, 0.6880])
- **run 間差異が大きいタスク**: missing 件数が run 002/004 = 9, run 003 = 7 と変動。extended thinking の爆発が run ごとに異なるタスクで発生していることを示す。std=0.0281 は exp_081 (std=0.0138) の約 2× (single-attempt 化の影響も)。
- **結果概要**: ❌ 後退 (Δ=−0.0459 vs baseline mean=0.6633)。missing 件数が avg 8.3/50 (exp_081 avg ~1.5) と急増した。
- **主な後退原因**:
  - **schema-only preamble が最大の問題**: 実データ値の非表示により Qwen3.5-35b の extended thinking が "探索計画" フェーズで爆発 → 初回 model call が timeout に至るタスクが頻出 (task_19/75/173: 3/3 または 2/3 runs で steps=0, timeout=900s)。preamble build は <1s と確認済み → model call 自体の timeout
  - **min_steps=4 guard の副作用**: task_11 で step 4 の正答をブロック → 追加探索で filtering logic が変化 → zero_recall
  - **task_330 run_003**: 262K token overflow (単発、schema-only でも複数 doc 合計でモデルの context 上限超え)
- **正の効果確認**:
  - task_86: 3/3 fix (grep_file/Rule 16 が寄与)
  - task_89, task_200: 全 3/3 fix (grep_file + discovery による確認ステップが有効)
  - task_259: 2/3 fix (部分的改善)
- **reproducibility 評価**: std=0.0281 (medium)。single-attempt 化 (vs exp_081 の 3-attempt union) により stochastic variance が増加。extended thinking 爆発が run 間で異なるタスクに出現することで std を押し上げている。
- **軸評価**: `[arch] preamble schema-only削減 + discovery tools` は Type B 確信誤推論 (task_89/200) に一部有効。しかし schema-only 化による extended thinking 爆発が通常タスクの timeout を急増させるコストが上回る。**実データ値を preamble から完全除去する設計は Qwen3.5-35b に適合しない**。exp_006 (zero preamble, λ=0.16) より遥かに良いが、情報不足は確かにタイムアウトを増やす。
- **教訓**: discovery tools (grep_file/stat_file) の有用性自体は task_86/89/200 で実証。fulldata 150k preamble を維持した上で additive に grep_file/stat_file を追加するアプローチの方が安全。**schema-only preamble ([arch] cursor_harness 軸) は廃棄確定**。次は **exp_081_union_t0 + Rule 16 minimal addition** が最優先。

### exp_083_additive_tools (n=2 pool mean=**0.3331** ± 0.3402, ⚠️ verdict=**判定不能** — run_003 vLLM 52分 502 outage)

- **設計**: exp_081_union_t0 ベース + Rule 16 (bidir dedup) + Rule 17 (min_steps=4 prompt) + runner min_steps=4 guard + 新ツール `grep_file` + `stat_file` (fulldata 150k preamble 維持、additive 追加のみ)。3-attempt union (T=0,0,0, k=1)。config: `max_workers: 6`, `task_timeout_seconds: 900`
- **n=3 完走状況**: run_001=中断、run_002=完了 (λ0.5=0.5736, missing=10)、run_003=完了 (λ0.5=**0.0925**, missing=44 — vLLM 502 outage 支配)
- **n=2 pool stats**: mean=0.3331, std=0.3402 → spread=0.481 で大きすぎ + infra 汚染で **統計的判定不能**
- **致命的 config 設定ミス**: `max_workers: 6` × 3-attempt = **18 concurrent vLLM streams**。18 streams は vLLM 飽和点 (~13-15) を超えており、run_002 missing=10 の内 6 件の構造的原因
  - run_002 missing 内訳: 4件 Cloudflare 530 burst (18:03Z) / 6件 18 streams 過負荷 900s timeout (task_27/80/173/379/396/418)
- **run_003 新発見 — vLLM 52分 origin outage (18:47Z–19:39Z UTC)**: 40/44 missing が `502 origin_bad_gateway` で失敗。workers=6 の構造的 missing ではなく **外部 vLLM endpoint crash** が支配
  - 完了 6 タスク (task_11/19/22/24/25/26) は全て 18:47Z 以前に処理完了; それ以降は全滅
  - run_002 は Cloudflare 530 burst、run_003 は 502 origin crash — エラー種別が異なり別事象
- **runs[] 個別スコア**:
  - run_001: λ=N/A (中断、task_26 のみ trace)
  - run_002: λ=0.5736, missing=10, perfect_no_extras=25/50
  - run_003: λ=0.0925, missing=44, perfect_no_extras=4/50 (outage 汚染)
- **完了タスクのみの設計品質**:
  - run_002 完了 40 タスク: λ0.5=0.7170 (exp_081 mean 0.6906 超)
  - run_003 完了 6 タスク: λ0.5=**0.7708** (さらに高い — perfect 5/6, zero_recall 1/6)
  - → 設計ロジック自体は 2 run 一貫して健全
- **run 間差異の大きいタスク**: task_11 が **2 run 連続 zero_recall** = 構造的バグ
  - 原因: union k=1 の dtype 副作用 — 3 attempts で ID 列の int/str 混在 → signature 違いで union が 6 cols 重複
  - run_002/003 両方で同一パターン再現 → **次回 exp で dtype-normalize step 追加が必須**
- **reproducibility 評価**: infra 汚染が支配的で設計 reproducibility の正確な評価は不能。設計上の唯一確認済み構造バグは task_11 union dtype 副作用のみ
- **ツール採用率** (run_002): grep_file **28 calls / 50 tasks** (moderate 採用)、stat_file **0 calls** (完全死蔵)、min_steps=4 guard **16 fires / 15 unique tasks** (30%)
- **stat_file 廃棄確定**: run_002 + run_003 両方で 0 calls = 2 run 連続死蔵確認。Qwen3.5-35b は呼び出さない。次回 exp で削除必須
- **最終 verdict**: ❌ raw n=2 mean=0.3331 (Δ−0.330 vs baseline) は大幅後退だが **vLLM outage 汚染で判定不能**。workers=4 での n=3 clean re-bench (vLLM 復旧後) が必須
- **教訓**: 3-attempt union 実験では **workers=4 (= 12 streams) を厳守**。union k=1 の dtype-normalize step (ID-like 列を str→int 統一) で task_11 構造バグ修正要。stat_file 削除で次回 pick

### exp_086_r1_official_params (n=2 clean mean λ0.5=**0.7353 ± 0.0135**, CI [0.7167, 0.7540] ✅ — 確定 win vs baseline; Δ+0.0447 vs exp_081; run_004 API汚染 ❌)

- **設計**: exp_081_union_t0 ベース + **R1 公式 params 適用** (T=0.6/0.6/0.7 per-attempt、presence_penalty=1.0) + Rule 16 (bidir dedup) + Rule 17 (min_steps=4 prompt) + runner min_steps=4 guard。3-attempt union (T=0.6/0.6/0.7, k=1)。config: `max_workers: 4`, `task_timeout_seconds: 900`
- **全 run 個別スコア (runs 002–004)**:
  - run_002: λ0.5=**0.7449**, λ0.0=0.7700, perfect=35/50, with_extras=3, zero_recall=10, missing=1 ✅ clean
  - run_003: λ0.5=**0.7258**, λ0.0=0.7700, perfect=31/50, with_extras=7, zero_recall=10, missing=1 ✅ clean
  - run_004: λ0.5=**0.5567**, λ0.0=0.5800, perfect=25/50, with_extras=4, zero_recall=8, missing=**13** ❌ **API汚染 (invalidated)**
  - **clean n=2 (002+003)**: mean=0.7353, std=0.0135, CI=[0.7167, 0.7540]
  - raw n=3 (002+003+004, 参考のみ): mean=0.6758, std=0.1036 (汚染込み、採用不可)
- **run_004 汚染詳細**: 後半 13 tasks (task_259/261/269/283/287/292/303/305/330/344/349/396/418) が `steps=0, ans=None, meta={}` で全滅。subprocess が ReAct ループ実行前に timeout kill された signature = vLLM 一時 stall による連続帯 failure。設計・実装の問題でなく**実行環境ノイズ確定**。⚠️ N=3 並列 union (12 vLLM streams) は transient 5xx に対して脆弱 — 1 attempt timeout → 3 attempts 全滅のカスケード失敗が起きる構造的リスク
- **run 間差異が大きい観測 (clean runs のみ)**:
  - **with_extras の変動** (run_002: 3件 vs run_003: 7件) — temp 多様化で union が余分列を拾う頻度が run によって変わる。λ1.0 の差 (0.7197 vs 0.6817) はこれが主因
  - **zero_recall は clean 両 run で 10/50 と一致** — task_80/89/180/396 等の consistent zero は温度変化 0.6 程度で救えない構造的失敗
  - **missing は clean 両 run で 1/50 と一致** — workers=4 × T=0.6 設定は安定
- **主な新規攻略タスク (exp_081 比)**:
  - task_196, task_200, task_259 が clean 2 run で新規攻略 (3件 × 2run 確認) → Δ+0.063 を説明
  - temperature diversification (T=0.6/0.7 vs T=0/0/0) が answer diversity 増加 → 新規 recall の主因
- **reproducibility 評価**: clean std=0.0135 は exp_081 (std=0.0138) と同等で再現安定。run_004 汚染で raw std=0.1036 (7.7× 膨張) → **missing≥10 の run は API汚染として invalidate する判断を採用**。clean n=3 確定には run_005 が必要
- **軸評価**: `[infer] R1 official params (T=0.6/0.7 diversification + presence_penalty=1.0)` は **確定 win (+0.045 vs exp_081)**。recall (λ0.0) +0.12 が主効果 (0.77 vs 0.66)。consistent zeros (task_80/89/180/396 等) は未解決 — 同一 semantic 誤解に全 3 attempts が収束する Type B 問題
- **Consistent zeros (10/50、clean runs 共通)**: task_80, task_89, task_180, task_396 等が clean 両 run で zero_recall — temperature 0.6 程度の diversity では Type B 確信誤答を救えない。次の軸は column voting (R2 ⌈n/2⌉) や hybrid thinking (R4) で extras 削減 + 思考多様化
- **最終 verdict**: ✅ **確定 win** — clean n=2 mean=0.7353、CI [0.7167, 0.7540] が exp_081 CI [0.675, 0.706] と非重複。**新リーク無しベスト確定**、submission v3 採用済み。run_005 で clean n=3 確定後に std 最終確認推奨
- **教訓**: temperature 多様化 (T=0.6/0.6/0.7) は union 設計と組み合わせると recall を大幅に引き上げる。R1 単体で exp_081 比 Δ+0.045 = 分析当初の推定 (+0.005〜+0.015) を大幅に上回った。with_extras の増加 (run_003 で 7 件) は union mode の余分列収集コストで、R2 (column voting) による削減が次の有効手。N=3 並列 union は vLLM transient failure に対して脆弱 (1 attempt 死 → 3 attempts 全滅) — runner retry 機構が欲しい

### exp_087_r2_column_vote (n=2 mean λ0.5=**0.7098 ± 0.0182**, CI [0.6847, 0.7350] ❌ — regression vs exp_086 Δ-0.0255; k=1→k=2 の 1 行 ablation)

- **設計**: exp_086_r1_official_params ベース、変更は `runner.py:43` の `_MAJORITY_K = 1 → 2` **1 行のみ**。temperatures (T=0.6/0.6/0.7)、min_steps=4、Rule 16/17、presence_penalty=1.0、workers=4 はすべて同一 → 純粋な k ablation
- **全 run 個別スコア (runs 004–006)**:
  - run_004: λ0.5=**0.6970**, λ0.0=0.7100, perfect=33/50, with_extras=2, zero_recall=12, missing=2 ✅ clean
  - run_005: λ0.5=**0.7227**, λ0.0=0.7500, perfect=34/50, with_extras=3, zero_recall=10, missing=2 ✅ clean
  - run_006: ❌ TIMEOUT (90 min wall, 48/50 traces のみ) — 評価不能
  - n=2 clean (004+005): mean=0.7098, std=0.0182, CI=[0.6847, 0.7350]
- **k=1 vs k=2 の直接比較 (exp_086 同 temps)**:
  - recall (λ0.0): 0.770/0.770 (k=1) → 0.710/0.750 (k=2), **Δ mean -0.04**
  - with_extras: 3/7 (k=1) → 2/3 (k=2), **Δ mean -2.5** (仮説通り削減)
  - λ0.5: 0.7449/0.7258 (k=1) → 0.6970/0.7227 (k=2), **Δ mean -0.0255**
- **仮説検証**: extras 削減 (with_extras -2.5) は確認。ただし recall loss (-0.04) のスコア寄与が 4-10× 上回る (recall 1 列失う = score -1/gold; extras 1 列削る = score +0.5/pred ≈ +0.1〜0.25)
- **newly lost タスク (k=2 で失った union-only 正解)**:
  - task_173: k=2 で run_004 が gold 列 drop (1/3 attempt のみ gold signature)
  - task_196 (chemistry bonds): Rule 16 effect 列が 1 attempt のみ → majority 不成立で脱落
  - task_243, 259, 25: union-only correct 列の典型パターン
- **newly gained タスク (k=2 で extras 削減→ perfect)**: task_200, 379, 67 — 3 attempts 一致時の extras=0
- **k×diversity 逆相関の発見**: T=(0.6,0.6,0.7) 多様化下では sampling variance が高く "1 attempt だけが正解" ケースが多発 → k=2 majority 不成立が多い。k=1 union は diversity を直接 recall lift に変換できる設計。k と temperature diversity は **逆相関する設計パラメータ**: k=1 + 高 diversity = SOTA; k=2 + 低 diversity (T=0/0/0) = alternative
- **reproducibility 評価**: std=0.0182 は exp_086 (std=0.0135) より大きい。λ0.0 の run 間差 (0.71 vs 0.75) が variance 源。k=2 は "どの 2 attempt が一致するか" で結果が run ごとに変わりやすい
- **最終 verdict**: ❌ **regression vs exp_086** (Δ=-0.0255、std 0.018 の +1.4σ で likely 実効果)。⚠ likely_win vs baseline (Δ=+0.0465) だが exp_086 未満で採用せず。**[aggr] R2 ⌈n/2⌉ majority vote 軸は廃棄確定**
- **教訓**: column-signature majority vote (k=2) は extras 削減に有効だが recall コストが大きい。現在の k=1 union + temperature diversity が λ=0.5 で最適。extras 削減を狙うなら majority vote ではなく M-Schema Plan-first (R3) による "planning time での column commitment" が回り道なく有効

### exp_088_r3_mschema_plan_first (n=2 clean mean=**0.6915 ± 0.0270** ❌ — regression 確定 vs exp_086 Δ-0.044; run_004 vLLM outage 廃棄; R3 M-Schema × Plan-first)

- **設計**: exp_086_r1_official_params ベース + **R3 軸**: (1) preamble に `## [M-Schema]` ブロック追加 (CSV/SQLite から DDL+sample 値を最大 4K chars で生成、`_build_mschema_section`)、(2) Rule 12a 新設: Step 0 plan に `# Plan: candidate columns are [(name, source, dtype), ...]` を強制 (列名は M-Schema から選ぶ)、(3) Rule 12b: 旧 Rule 12 (SQL で列名明示) を維持。ベース挙動 (3-attempt union T=0.6/0.6/0.7, k=1, min_steps=4, presence_penalty=1.0, workers=4) は exp_086 と完全同一。diff lines: **611** (vs exp_086=427、+184 行は M-Schema builder ~120 行 + Rule 12a/12b 追加 + 例文書き換え)
- **全 run 個別スコア (n=2 clean 確定)**:
  - run_002: λ0.5=**0.7106**, λ0.0=0.7500, perfect=32/50, with_extras=5, zero_recall=10, missing=2 ✅ clean
  - run_003: λ0.5=**0.6724**, λ0.0=0.7300, perfect=27/50, with_extras=9, zero_recall=12, missing=1 ✅ clean
  - run_004: ❌ 廃棄 (vLLM 502 outage、artifacts/runs_discarded/ + REASON.md 記録済み)
  - n=2 統計: mean=**0.6915**, std=**0.0270**, CI95=[0.6541, 0.7289] (Δ-0.044 vs exp_086 mean=0.7353)
- **per-task delta (run_002 vs exp_086 002+003 mean)**:
  - task_11: 0.000 (Δ**-1.000**) — M-Schema × T-diversity × union 三項相互作用で union 9 cols × 75 rows 膨張、recall=0 (両 run consistent zero)
  - task_25: 0.000 (Δ-0.312) — row-padding バグで multiset signature 破壊 (両 run consistent zero)
  - task_200: 0.625 (Δ-0.250) — extras 増
  - task_379: 0.750 (Δ**+0.375**) — M-Schema による列特定が有効
  - task_67: 1.000 (Δ+0.125) — M-Schema 効
  - task_257: 1.000 (Δ+0.083) — M-Schema 効
- **主要 failure mode**:
  - **task_11 両 run consistent zero (主犯)**: M-Schema DDL を見た 3 attempts が独立に candidate columns を plan → row filter に独立性が伝播 → 行集合 diverge → union padding で 9 cols × 75 rows → signature 不一致 → recall=0 (gold=3 cols × 3 rows)。**"M-Schema × T-diversity × union" 三項相互作用として確定**
  - **task_25 両 run consistent zero**: `runner.py:_signature_majority_merge` の row-padding バグ — union 時に短い行集合を空文字でパディングすると multiset signature が破壊される。run_002 では 5 列目に正解列 (Nov/Oct/Sep Speaker) が存在するが union padding で空文字混入 → signature 不一致
  - **with_extras 増加 (run_002:+5 → run_003:+9)**: Plan-first が各 attempt に独立な列 commit を促し、union 後の列数が増加。Plan-first は single-attempt と整合する設計であり 3-attempt union と相性が悪い
  - **timeout (stochastic)**: task_344, task_418 が 3 attempts 全 timeout (900s 超、exp_086 にも出現)
- **run 間差異が大きいタスク**: task_11 は両 run zero (構造的確定)。task_25 は両 run zero (バグ確定)。with_extras run_002:5 → run_003:9 は Plan-first × union 相互作用の再現性あり (悪化方向)。task_379/257/67 は run_002 のみ評価だが M-Schema benefit は安定的
- **reproducibility 評価**: run_002/003 で Δ=0.038 と中程度。主要な違いは perfect_no_extras (32 vs 27) と with_extras (+5 vs +9)。task_11/25 は両 run 一致して失敗 → 構造的 failure 確定。stochastic な差異は 5 tasks 程度
- **最終 verdict**: ❌ **regression 確定 (n=2 clean mean=0.6915, Δ=-0.044 vs exp_086)** — task_11 × task_25 の両 run consistent zero が構造確定。**[context] R3 M-Schema + Plan-first + 3-attempt union 廃棄確定**
- **教訓と今後の可能性**: (1) M-Schema は task_379/67/257 で純粋に有効 → **single-attempt 限定で再評価価値あり**。(2) `_signature_majority_merge` の row-padding バグは独立 fix 候補 — 横断効果 +0.02〜+0.04 期待、task_25 を含む "union collapse" 系 failure の抜本対策。(3) Plan-first と 3-attempt union は設計上相性最悪 — Plan-first は single-attempt loop への組み込みを想定すべき

### exp_090_r5_structured_error_hint (n=1 run_002 λ0.5=**0.7236** ⚠️ — ノイズ範囲内 Δ=-0.012 vs exp_086 mean=0.7353; R5 [tool] structured error hint)

- **設計**: exp_086_r1_official_params ベース + **R5 軸**: execute_python / execute_sqlite のエラー例外を 10 class に分類し、各 class に "次に試すべきヒント 1 行" を付与する structured hint wrapper。3-attempt union T=0.6/0.6/0.7, k=1, workers=4, exp_086 と完全同一ベース設定。
- **全 run 個別スコア (n=1)**:
  - run_001: smoke run (task_26 のみ)
  - run_002: λ0.5=**0.7236**, λ0.0=0.7300, perfect=32/50, with_extras=5, zero_recall=11, missing=2 ✅ clean (完走確認)
  - run_003: 部分的完走 (12 tasks のみ、evaluation.json なし — 走行中 or interrupted)
  - n=1 統計: Δ=-0.012 vs exp_086 mean=0.7353 → ノイズ床内 (std=0.0135)
- **hint 発火実績**:
  - 発火 19/50 task、実効 2 class: KeyError (18 件), FileNotFound (3 件)
  - 残り 8 exception class は 0 発火 = dead code (公開 50 task では起きない例外)
- **per-task delta 主要点**:
  - task_259: -0.594 (hint 発火 0 — wrong-column failure、R5 圏外)
  - task_379: +0.375, task_67: +0.125, task_330: +0.125
  - Δ=-0.012 のほぼ全量を task_259 の -0.594/50 で説明可
- **reproducibility 評価**: n=1 のため不明。task_259 は Type B 確信誤答 (3 attempts 全部 metadata 列解釈) でありどの run でも consistent fail が予想される
- **最終 verdict**: ⚠️ **ノイズ範囲内 (n=1、Δ=-0.012)** — 隣接 exp_046 (-0.047) と exp_056 (+0.025) とともに error-feedback 系統は再現的に微細効果。replication は ROI 低いため skip、軸として引き続き探索廃棄候補。
- **教訓**: (1) exception class の 80% は公開 50 task で発火しない dead code → hint の実用範囲は KeyError/FileNotFound のみ。(2) task_259 型 wrong-column failure は exception mechanism を使わない → error hint では根本的に解決不能。runner.py row-padding バグ修正 (exp_093) がより直接的な対策。

### exp_092_verify_rule_single (n=1 run_004 λ0.5=**0.6400** ❌ — regression 確定; baseline 未満; [verify-rule] single-attempt + non-LLM regex verifier)

- **設計**: exp_086_r1_official_params ベース + **[verify-rule] 軸**: (1) `verifier.py` (non-LLM regex で expected_column_count 推定、confidence={high,med,low}、low→skip)、(2) `agent.py` terminal-answer hook (shape mismatch で max 1 retry)、(3) **single-attempt 化** (`_ATTEMPT_TEMPS=(0.6,)`)、workers=10。run_001/002/003 は smoke runs (1 task each)、run_004 が full bench run。
- **全 run 個別スコア (n=1)**:
  - run_004: λ0.5=**0.6400**, missing=3, perfect=29/50, with_extras=2, zero_recall=13 — baseline exp_040 mean=0.6633 **未満**
- **機構的失敗 2 段構造**:
  - (1) **single-attempt 化 cost (-0.06)**: T-diversity × union の +0.045 (exp_086 の優位源) を直接喪失。task_11/25/257/259/352 等で 3-attempt union なら recall できた列が消失。single-attempt + verifier 1 retry では rescue 不能
  - (2) **verifier regex false positive (-0.04)**: `_SINGLE_VALUE_PATTERNS` に multi-ask question ("X and Y") を認識するパターンがなく、"What is the average upvotes and average age?" を誤って 1-column と判定。task_249 で agent が正解 `[average_upvotes, average_age]` 2-col を提出 → verifier "1 col implied" hint → agent が `[metric_value]` 1-col × 2-rows に collapse → recall=0
  - **二乗作用**: single-attempt 化でもし verifier に書き換えられると他 attempt による rescue がない
- **run 間差異が大きいタスク**: task_249 (verifier false positive で perfect→zero)、task_11/25/257/259/352 (3-attempt union 喪失)
- **reproducibility 評価**: n=1 のため不明。しかし機構は構造的 (regex は deterministic) で replication でも同等以下が期待される
- **最終 verdict**: ❌ **regression 確定 (n=1 λ0.5=0.6400、baseline 0.6633 未満)**。single-attempt 化 + regex verifier の複合 experiment は廃棄確定。
- **教訓**: (1) verifier regex は multi-ask question に対して構造的に blind — "X and Y" conjunction を認識しない限り false positive は避けられない。(2) single-attempt class での verifier 実験は "verifier が hurt したら他 attempt で rescue 不能" という脆弱性がある。(3) verifier を試すなら 3-attempt union base で実施すべき (= 1 attempt が書き換えられても他 attempt が union で元の正解を保持)。

### exp_093_runner_padding_fix (n=1 partial 49/50 λ0.5=**0.7328 imputed** ⚠️ — ノイズ範囲内 Δ=-0.0025; replication 失敗; [bugfix-infra] `_signature_majority_merge` row-padding fix)

- **設計**: exp_086_r1_official_params ベース + `_signature_majority_merge` の row-padding 修正 (case A: attempt-vote 支配長フィルタ + smallest-tiebreak)。3-attempt union T=0.6/0.6/0.7, k=1, workers=4, exp_086 と完全同一ベース設定。
- **全 run 個別スコア (n=1 partial)**:
  - run_001: smoke run (1 task)
  - run_002: **49/50 tasks 完了、wall-clock 90 min で kill** (1 task 未完)。evaluation.json なし — 手動 imputation: 0.7328 (49 task 合計 / 50 task 換算、missing=0 仮定)
  - run_003: replicate_bench が stale lock check で abort → 未実行
  - n=1 統計: Δ=-0.0025 vs exp_086 mean=0.7353 → ノイズ床内 (std=0.0135 の 0.19σ)
- **per-task delta 主要点**:
  - task_259 (+0.031): 正解 `text` 列を padding なしで保持 → fix 有効
  - task_25 (-0.312 → 依然 zero): union 5cols×12rows 膨張は解消 (2cols×3rows に正常化) だが **全 attempt が "Officers meeting" wrong filter に convergence** → fix 対象外の構造的 failure
  - task_11 (-0.125): length-grouping で extra Diagnosis 列が選択群に含まれ extras +1 (fix の副作用)
- **run 間差異**: replication なしのため不明。ただし fix は deterministic なので task_259 (+) / task_25 (zero 継続) / task_11 (副作用) は安定すると推定
- **reproducibility 評価**: n=1 partial のため評価不能。run_002 が 49/50 で killed されており評価 json なし。
- **exp_088 task_25 zero 仮説修正**: 旧解析では row-padding バグが主因と推定したが、実際は "row-padding バグ + 全 attempt content divergence (wrong filter)" の合成。fix 後 (a) row-padding バグは解消したが (b) 全 attempt が別の wrong answer に収束することで zero 継続。**fix 単体では task_25 を救済できない構造的限界が確定**。
- **最終 verdict**: ⚠️ **ノイズ範囲内 (n=1 partial Δ=-0.0025)、bug fix の "守りの価値" のみ確定**。future combo (exp_086 + exp_094 Qwen-Agent + exp_093 padding fix) での padding 汚染排除に貢献するが、独立 axis としては win にならない。replication 失敗で確定不能、n=3 投資 ROI 低い。
- **教訓**: task_25 の zero は row-padding バグではなく "全 attempt が同じ wrong filter に convergence する Type B 確信誤答" が主因。bug fix はノイズ防止として有用だが Type B 救済は別軸 (question preprocessing / R4 hybrid thinking) が必要。

### exp_095_column_candidate_hint (n=1 run_002 λ0.5=**0.7086** ⚠️ — likely_regression Δ=-0.027; replication 待ち (exp_094 vLLM 占有); [preamble:hint] column candidate hint)

- **設計**: exp_086_r1_official_params ベース + preamble に column candidate hint section 追加 (non-LLM NLP heuristic: question 名詞抽出 → DB schema マッチング → top_n=7 advisory hint)。600 chars cap + "NOT a constraint" 推奨文言 + fixed section 位置 (glossary 直後)。3-attempt union T=0.6/0.6/0.7, k=1, workers=4, exp_086 と完全同一ベース設定。
- **全 run 個別スコア (n=1)**:
  - run_001: smoke run (1 task)
  - run_002: λ0.5=**0.7086**, missing=1, perfect=30/50, with_extras=7, zero_recall=12
  - run_003+: exp_094 が vLLM 占有中のため待機、replication 未実施
  - n=1 統計: Δ=-0.027 vs exp_086 mean=0.7353 → likely_regression 寄りだが n=1 で確定不能
- **column hint の分離効果**:
  - hint 発火 41 task: mean=0.7453
  - hint 非発火 9 task: mean=0.5417 (差 = 0.2036)
  - → **この差は column hint の因果ではなく row-padding バグ脆弱性の task 構造差**: 非発火 9 task は hint を generate できなかった JSON-only / simple task = row-padding バグが発現しやすい task と重なる
  - **column hint 軸自体の純粋効果 ≈ 0**
- **Δ=-0.027 の attribution**:
  - task_11 (-1.000): JSON-only データで hint **未発火**、Thrombosis filter divergence + row-padding バグ表面化 (exp_086 では stochastic 回避)
  - task_22 (-0.250): 1 attempt 1 row vs 他 attempts 2 row → row-padding で signature 破壊
  - task_25 (-0.312): col 3 に正解 (Nov/Oct/Sep Speaker) があるが padding で 12 行膨張 → multiset 破壊 (exp_088/093 と同機構の再発)
  - task_379 (+0.0075): hint が relevant 列を正しく推奨した可能性
  - **Δ ≈ -0.025 (row-padding 系) + ≈0 (hint 純粋効果) = -0.025 ≈ 観測 -0.027**
- **run 間差異**: replication なし (exp_094 vLLM 占有中)。row-padding バグ表面化は stochastic なので run ごとに変わる可能性あり。
- **reproducibility 評価**: n=1 のため評価不能。hint 純粋効果は ≈0 で stable、regression は row-padding バグ stochastic 表面化によるものであり次 run では改善の可能性
- **最終 verdict**: ⚠️ **likely_regression 寄り (n=1 Δ=-0.027)、replication 待ち**。column hint 軸自体は M-Schema の致命傷 (exp_088 三項罠) を回避できたが lift も出さず。row-padding バグが独立軸として未解決な状態での evaluation は attribution が困難。replication 後 run_003 で結果を確認する。
- **教訓**: (1) column hint top_n=7 の suggestion accuracy が低い (例: task_25 で `event_status` を推奨すべきところを `cost` 等の無関係列を推奨)。scoring 関数の改善余地あり。(2) row-padding バグが表面化しやすい task 構造 (JSON-only、multi-attempt convergence diverge) では hint 発火率も低く、この layer で行きの改善期待が持ちにくい。(3) hint 発火 41 task での mean=0.7453 は exp_086 mean=0.7353 より高く、hint が active に邪魔していないことの間接証拠。

### exp_094_qwen_agent_runtime_swap (n=1 run_003 λ0.5=**0.7119** ❌ — regression 確定 Δ=-0.024; wire-level audit 確定 (commit ad7b923); [arch:runtime] QwenAgentModelAdapter)

- **設計**: exp_086_r1_official_params ベース + `build_model_adapter()` の戻り値を `QwenAgentModelAdapter` (Qwen-Agent native wire) に swap。3-attempt union T=0.6/0.6/0.7, k=1, workers=4, exp_086 と完全同一ベース設定。
- **全 run 個別スコア (n=1)**:
  - run_001/002: smoke runs (1 task each)
  - run_003: λ0.5=**0.7119**, missing=1, perfect=29/50, with_extras=9, zero_recall=12
  - n=1 統計: Δ=-0.024 vs exp_086 mean=0.7353 → regression 確定 (wire-level audit で機序確定)
- **wire-level audit 結果 (commit ad7b923 REJECTED)**:
  - `qwen_agent/llm/base.py:177-178`: call 毎に `seed: random.randint(0, 2^30)` を generate_cfg に auto-inject
  - exp_086 (OpenAI adapter): seed 非送信 → vLLM session-RNG 連続 → 3 attempts が比較的収束
  - exp_094 (Qwen-Agent): 各 attempt が異なる random seed → vLLM が seed-deterministic sampling → 3 attempts が意図せず diverge → union が divergent extras を残存
  - 修正試行: `seed=None` 明示 → openai client が `null` で wire に送り、vLLM が再ランダム化、効果なし
  - **Mitigation 不可**: monkey-patch ROI 低い、framework swap 路線は ruled out
- **system effect**:
  - with_extras=9 vs exp_086 avg 5 (+4): divergent aggregate 視点が union に蓄積 (task_67/196/200/243/303 で +1 extras)
  - λ1.0=0.659 vs exp_086 0.701 (Δ=-0.042): extras rate 直接反映
  - task_80 (+1.000 unexpected): exp_086 両 run で consistent zero が exp_094 で `[3, 5]` 2 数値 perfect — n=1 で stochastic or native chat template 効果か不明
- **reproducibility 評価**: n=1 のため不能。wire-level bug (seed auto-inject) は deterministic なので regression は再現性あり
- **最終 verdict**: ❌ **regression 確定 (n=1 Δ=-0.024、wire-level audit で機序確定)**。**vendor magic 仮説は存在しないことが実証**。qwen-agent OAI backend は openai client の薄い wrapper で、seed auto-inject という harmful side-effect のみ追加。**[arch:runtime] framework swap 系は今後試さない**。
- **教訓**: (1) 外部 library の wire-level side effect (seed injection) が 3-attempt union との相互作用で score を下げる予期しない経路がある。(2) "vendor magic 仮説" = native fit → score 上振れは、thin wrapper 実装では成立しない。(3) 真の Qwen-native fit を期待するなら `qwen_agent.Assistant` フル乗せ換え (= BACKLOG 中長期、大工事) が必要。

### exp_097_mschema_single_attempt (n=1 run_002 λ0.5=**0.6614** ❌ raw — Δ=-0.074 だが真の axis effect ≈0; timeout 7 task 主因; [context-single] M-Schema + single-attempt)

- **設計**: exp_086_r1_official_params ベース + M-Schema DDL+sample block (exp_088 から Plan-first Rule 12a 抜きで移植, `_MSCHEMA_SAMPLE_VALUES=5`, `_MSCHEMA_MAX_CHARS=4000`) + single-attempt (`_ATTEMPT_TEMPS=(0.6,)`, workers=10)。Plan-first commitment 永久排除、M-Schema header "while reasoning"で暗黙誘導も排除。
- **全 run 個別スコア (n=1)**:
  - run_001: smoke run (1 task)
  - run_002: λ0.5=**0.6614**, missing=7, perfect=32/50, with_extras=1, zero_recall=11
  - n=1 統計: Δ=-0.074 vs exp_086 mean=0.7353 → raw regression
- **Δ=-0.074 の Attribution (2 段階)**:
  - (1) **timeout 7 task**: M-Schema 4K preamble で per-call latency 増加 → 900s timeout が single-attempt の fallback なしで全 task score=0 → λ寄与 ≈ -7/50 ≈ -0.07
  - (2) **残り 43 task subset**: Δ=-0.003 = noise 圏 → **M-Schema 純粋 axis effect ≈ 0 確定**
- **M-Schema 効果の per-task 内訳**:
  - improvement: task_67 (+0.125), task_196 (+0.125), task_200 (+0.125), task_330 (+0.125) — extras 削減効果 (structured DDL で余分列への迷いを抑制)
  - loss: task_379 (-0.375、sampling miss — stochastic)、task_25 (-0.312、row-padding バグ再発 = single-attempt で fallback なし)
- **重要な発見: exp_088 害の主因は 3-attempt union × T-diversity との相互作用**
  - exp_097 (single-attempt) では M-Schema ≈ 0 だが exp_088 (3-attempt union) では -0.044
  - → **exp_088 の三項罠 (M-Schema × T-diversity × union) の "union × T-diversity" 部分が dominant**
  - M-Schema 単体を clean single-attempt で測定すると効果中立、"害ではない"
- **single-attempt class score の再確認**: exp_040 (0.6633)、exp_092 (0.6400)、exp_097 (0.6614) で 0.65-0.66 圏が確認 → single-attempt は 3-attempt R1 比 -0.07 のコストが確定 baseline
- **reproducibility 評価**: n=1、replication は exp_096 vLLM 占有で待機中
- **最終 verdict**: ❌ **raw regression (n=1 Δ=-0.074)、ただし主因は timeout で axis effect ≈0**。M-Schema を活かすには 3-attempt union + row-padding fix の combo (`exp_099_mschema_3attempt_clean`) が必要。single-attempt class での pure M-Schema 測定は完了、廃棄に近いが M-Schema + 3-attempt combo 候補として記録保持。
- **教訓**: (1) M-Schema 4K preamble は per-call latency を増加させ単独 timeout リスクあり — task_timeout を 1200s 以上に上げるか、3-attempt union で 1 attempt timeout でも fallback できる構造が必要。(2) M-Schema 単体の軸効果 ≈ 0 確定、exp_088 の regression は M-Schema ではなく union × T-diversity との相互作用が主因だった。(3) 3-attempt union + M-Schema のみの clean exp (Plan-first 抜き、row-padding fix bundle) = `exp_099` で再評価推奨。

### exp_096_cursor_harness_v2 (n=1 run_002 λ0.5=**0.6700** ❌ regression vs exp_086 Δ=-0.065; +0.005 vs baseline; [arch:harness] cursor-style schema-only + tool discovery; single-attempt)

- **設計**: exp_081_union_t0 base (再評価) + schema-only preamble (~5K chars) + grep_file/stat_file tool discovery。single-attempt (workers=10)。exp_082 の workers=20 contamination を workers=10 に修正した再評価実験。
- **全 run 個別スコア (n=1)**:
  - run_001: smoke run (1 task)
  - run_002: λ0.5=**0.6700**, missing=8, perfect=33/50, with_extras=0, zero_recall=9
  - n=1 統計: Δ=-0.065 vs exp_086 mean=0.7353、+0.005 vs exp_040 baseline 0.6633
- **workers=20 contamination 解消確認**: exp_082 (workers=20) 0.6174 → exp_096 (workers=10) 0.67 → **+0.05 改善**。元の contamination が確認、ただし exp_086 R1 比ではまだ -0.065 構造的下位
- **Positive findings (schema-only の良い面)**:
  - **with_extras=0** (vs exp_086 avg 5): extras 完全削減 (+0.025 lambda 寄与)
  - **task_89 (+0.625), task_86 (+0.375)**: format 認識改善。exp_086 の 150K tokens 全 injection では agent が値を rote-copy → format 問題 (Chinese GP "+16.445" 文字列等) で zero。exp_096 は tool 経由で実値取得 → format visible → perfect
- **Negative findings (schema-only の悪い面)**:
  - **task_11/173/408 confident filter 崩壊 (-0.06)**: schema-only で知識なし → knowledge.md re-read skip → "severe thrombosis" semantic threshold 等を見落とし
  - **8 missing infra brittleness (-0.04)**: timeout 4 + no_answer 2 + content_missing 1 + 400_error 1。single-attempt は fallback なしのため直撃
- **single-attempt class 4 例確定**: exp_040 (0.6633)、exp_092 (0.6400)、exp_097 (0.6614)、exp_096 (0.67)。**exp_086 R1 の +0.07 は 3-attempt union T-diversity 由来**
- **reproducibility 評価**: n=1、replication 待ち
- **最終 verdict**: ❌ **regression vs exp_086 (Δ=-0.065)、baseline level (+0.005 vs exp_040)**。schema-only は extras 削減に有効だが recall/missing は 3-attempt + fulldata preamble が必要。workers=20 contamination の解消は確認済で exp_082 再評価完了。
- **教訓**: (1) schema-only preamble は extras 完全削減 (with_extras 0) という明確な利得があるが、knowledge.md が隠れてタスク難易度が上がる副作用。(2) single-attempt でのtool discovery は infra brittleness リスクが高い (8 missing)。(3) 将来の "schema-only + 3-attempt union + fulldata fallback" の combo (= exp_082 の cursor harness を 3-attempt で再評価) は別軸候補。

### exp_101_rich_preamble (n=2 mean λ0.5=**0.7039 ± 0.018** ⚠️ likely_win single-attempt class; [preamble:rich] rich profile preamble + min_steps=4 + Rule16/17; single-attempt)

- **設計**: exp_086_r1_official_params ベース + **[preamble:rich] 軸**: 新モジュール `profile.py` (171 行) で CSV/SQLite/JSON ごとに per-column dtype + null/unique + numeric quartile/min/max/mean + string top-5 値+counts を preamble に injection。`min_steps=4` ガード (runtime.py + agent.py)、Rule 16/17 で明示。**single-attempt** (`_ATTEMPT_TEMPS=(0.6,)`, workers=10, presence_penalty=1.0)。
- **全 run 個別スコア (n=2)**:
  - run_001: λ0.5=**0.6913**, perfect=33/50, missing=3, timeouts: task_11/80/396 (900s, 0 steps)
  - run_002: λ0.5=**0.7164**, perfect=34/50, missing=3, timeouts: task_11/173/396 (404–773s, 0 steps)
  - n=2 統計: mean=**0.7039**, std=0.0177, CI95=[0.6794, 0.7284]
- **run 間差異が大きいタスク**: task_173 (1.0→0, run_002 0-step crash)、task_196 (0→1.0, run_002 recover)、task_200 (0→1.0, run_002 recover)、task_38 (0.5625→0, run_002 over-narrow filter)、task_25 (0.75→0, 列名/集計解釈エラー)
- **single-attempt class 内比較 (best in class)**:
  - exp_092_verify_rule_single (0.6400): **Δ=+0.0639** ✅
  - exp_096_cursor_harness_v2 (0.6700): **Δ=+0.0339** ✅
  - exp_097_mschema_single_attempt (0.6614): **Δ=+0.0425** ✅
- **vs 3-attempt baseline exp_086 (cross-class)**: Δ=−0.0314 (class 違いで直接比較禁止)
- **主な改善タスク vs exp_086**: task_418 (Δ=+1.0、min_steps=4 ガードで early answer 回避)、task_67 (+0.12)、task_257 (+0.08)
- **主な後退タスク vs exp_086**: task_11 (−1.0 timeout)、task_173 (−0.5 timeout 片方)、task_196 (−0.38)、task_200 (−0.38)、task_259 (−0.31)
- **failure mode 分類**:
  - Pattern A (0-step timeout deaths): task_11/80/173/396 — vLLM 初回応答前 hang + single-attempt fallback なし → score floor = 0
  - Pattern B (semantic hard, consistent across runs): task_86/89/163/169/180/199/344 等 9 task — rich preamble 無関係
  - Pattern C (stochastic run-to-run): T=0.6 ノイズ + rich preamble top-5 値による filter over-specification の可能性 (task_38)
- **task_11 catastrophic failure 詳細 (preamble bloat)**:
  - Examination.json (253KB) + Patient.json (250KB) の 2 ファイル → `_csv_max_chars` 閾値を超えず "small" 判定 → **profile + raw 両方 emit** → preamble が数百KB に膨張
  - per-call latency 急増 → 32 steps × ~30s = 900s timeout に到達、または JSON action parse 失敗ループ
  - **infra side-effect**: `runner.py:355` で single-attempt 失敗時 `backbone = first_successful_attempt` が `None` → `trace.json` に `steps=[]` が永続化 → デバッグ不能 (exp_106 で fix 予定)
  - 根本対策: `total_profile_chars ≥ PROFILE_BUDGET` を超えたら低優先 file から profile drop (raw のみ残す)
- **reproducibility 評価**: std=0.0177 は標準的 (exp_086 std=0.0135 と同水準)。task_11/396 は両 run 一致 timeout → 構造的 failure 確定。残 run 差は stochastic 圏 (n=2 では平均化不足)。run_003 queue 中 (n=3 確定待ち)。
- **最終 verdict**: ⚠️ **likely_win (single-attempt class 内, n=2)**。4 single-attempt 競合全勝 (peer best 0.6700 → 0.7039、Δ=+0.0339)。ただし n=2 のみで CI が広く確定不能。**next steps**: (1) **exp_106 [CRITICAL bugfix]** profile size guard + runner.py:355 per-attempt trace persist → task_11 復元で +0.02 期待; (2) **exp_107** rich profile × 3-attempt T=0.6/0.6/0.7 union → 0.74〜0.76 帯期待 (新 SOTA 候補); (3) **exp_108** adaptive profile depth。exp_106 bugfix を先行しないと exp_107 も task_11 catastrophe を引き継ぐ。
- **教訓**: (1) rich profile preamble は per-column 値統計注入でも task_11/173 の 0-step hang は防げない — preamble size guard が必須。(2) exp_101 の "bundle 優位" (peer 比 +0.034〜+0.064) は rich preamble + min_steps + Rule16/17 の複合効果で純粋 preamble 軸の帰属は不明。(3) 3-attempt 化 (exp_107) は exp_106 bugfix 後に実施。

### exp_099_schema_graph_fallback (n=1 run_001 λ0.5=**0.5961** ❌ definitive regression Δ=-0.067 vs single-attempt floor, −3.08σ; [preamble:hint+schema-graph] LLM-judged FK+orphan-column advisory; single-attempt)

- **設計**: exp_086_r1_official_params ベース + **[preamble:hint+schema-graph] 軸**: `schema_graph.py` で CSV/SQLite ファイル間の FK 関係 + orphan column を LLM で判定し、"top-3 join paths" と "orphan column" の precomputed advisory を preamble 末尾に injection。**single-attempt** (`_ATTEMPT_TEMPS=(0.6,)`, workers=10, task_timeout=900s, min_steps=4)。バックログの [schema-graph] step 1 実装。
- **全 run 個別スコア (n=1)**:
  - run_001 (v3 — hint active): λ0.5=**0.5961**, λ0.0=0.61, λ1.0=0.5822, perfect=28/50, missing=10, with_extras=2, zero=9
  - runs 002/003/004: v4-era (hint `return ""` に disable 済み) でのアボート。run_001 dir 削除済み、summary のみ保存
  - n=1 統計: Δ=-0.067 vs single-attempt floor (exp_040 0.6633)、−3.08σ (floor std ≈ 0.022)
- **失敗機序**:
  - **hint発火 ~7 task で avg −0.286/task**: LLM が生成した FK/orphan advisory が命令調 ("do NOT double-count FK joins") → agent が hint を dictation として受け取り、hint が示す列を強制削除 or 強制 join → wrong column → recall=0。**precomputed advisory は質問を見ていない (context-blind)** ため、タスク固有の "どの列を出すか" 判断に誤介入。
  - **missing=10 (vs single-attempt floor 2-3)**: hint LLM precompute call → per-task latency 大幅増 → 900s timeout 超過。exp_097_mschema (missing=7) より深刻。
  - hint 非発火 ~8 task: +0.075 (preamble の oversized file を schema-only downgrade した副産物と推定)
- **v4 対処 (2026-05-07)**: `schema_graph.py:build_schema_graph_section()` を即時 `return ""` に disable。runs 002/003/004 はこの状態でアボート、v3 regression 確定後の実験中断。
- **salvageable 側軸**: `preamble.py` の schema-only downgrade (oversized CSV/JSON/SQLite → column header + dtype + 3 rows のみ) が hint 非発火タスクで +0.075 を示した可能性 → `exp_104_preamble_downgrade_only` として exp_086 base で isolated test 推奨。
- **5 連続 [context]/[preamble:hint] regression パターン確定**: exp_087 (k=2 majority) / exp_088 (M-Schema+3-attempt) / exp_095 (column candidate hint) / exp_097 (M-Schema single) / exp_099 (schema-graph hint) — **precomputed advisory が context-blind で dictation bias を引き起こすパターンが 5 連続確認**。
- **reproducibility 評価**: n=1、run_001 dir 削除で per-task trace 不在。failure analysis は author コメント (schema_graph.py v4 コメント) + summary のみ。v3 の再現性は不明。
- **最終 verdict**: ❌ **definitive regression (n=1 Δ=-0.067 vs single-attempt floor, −3.08σ)**。context-blind LLM advisory の dictation bias (hint発火タスク avg -0.286) + latency → missing=10 が主因。schema-only downgrade 副軸は salvageable (exp_104 候補)。**[schema-graph] + [preamble:hint] 両軸廃棄確定**。
- **教訓**: (1) precomputed advisory は質問を見ていないため "どの列を出すか" の context-aware 判断に介入してはならない。指示調で書くと agent は dictation として従い、context に合わない hint が recall=0 を誘発する。(2) advisory ≠ dictation の区別はプロンプトで制御不能 — agent は常に hint を優先する傾向。(3) schema-only downgrade (oversized file 縮小) は hint 文脈から切り離して単独実験推奨。(4) 5 連続 [context]/[preamble:hint] 系 regression → 今後はこの方向を試さない。
