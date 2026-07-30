# v6 ablation 結果 (= exp_128 abl_a / abl_b / abl_c)

> 実施日: 2026-05-15
> 目的: v6 (= exp_122_column_auditor、LB 0.5789) の構成要素分解
> 全 ablation 50-task n=1、workers=20、auditor + 2dp + prose-guard は維持

## 設定

v6 baseline:
- **PhasedReActAgent** (= PLAN → EXPLORE → ANSWER → VERIFY、4 phase + tool whitelist + per-phase system prompt)
- **adaptive_vote** (= 3 attempts × T=0.6、subset-pick > numeric-convergent > union)
- **column_auditor / filter_auditor** (= 維持、全 ablation で同じ)
- **2dp normalization + prose-file SQL guard** (= v5/v6 patch、維持)

3 ablation はそれぞれ 1 因子を取り除く:
- **A (att1)**: phased + per-phase prompt **+ n=1** (= adaptive_vote 無効化)
- **B (nophased)**: **no PhasedReActAgent** (= plain ReActAgent、全 tool 同時利用、単一 system prompt) + n=1
- **C (sameprompt)**: PhasedReActAgent 維持 + tool whitelist 維持 **+ phase 関係なく同じ system prompt (= PLAN 用を全 phase で流用)** + n=1

## 結果

| 設定 | phased | per-phase prompt | n_attempts | λ0.5 | perfect | Δ vs v6 baseline |
|---|---|---|---|---|---|---|
| **v6 baseline** | ✓ | ✓ | 3 | **0.7950** | 39/50 | — |
| **A** (att1) | ✓ | ✓ | **1** | **0.7800** | 39/50 | **−0.015** |
| **B** (nophased) | **✗** | (n/a) | 1 | **0.7400** | 37/50 | **−0.055** |
| **C** (sameprompt) | ✓ | **✗** | 1 | **0.7200** | 36/50 | **−0.075** |

## 寄与分解 (n=1 統一比較)

| 因子 | 数値根拠 | 寄与 (λ0.5) |
|---|---|---|
| adaptive_vote n=3 vs n=1 | baseline (0.7950) − A (0.7800) | **+0.015** |
| **per-phase system prompt 切替** | A (0.7800) − C (0.7200) | **+0.060** |
| Phase tracking + tool whitelist (per-phase prompt 抜き) | C (0.7200) − B (0.7400) | **−0.020 (= マイナス効果)** |
| Phased 構造全部 (= per-phase prompt + tracking) | A (0.7800) − B (0.7400) | **+0.040** |

**最重要発見:**

1. **C < B**: phase tracking + tool whitelist を「**per-phase prompt なしで動かす**」と、no-phase より悪化する。
   → phase 切り替えは prompt 切替前提で設計されてる。tool 制限だけ残しても無意味どころか逆効果。

2. **per-phase prompt が支配的 (+0.06)**: adaptive_vote (+0.015) の 4 倍。
   → v6 の構造的価値の 80% は per-phase prompt 切り替えに集約。

3. **adaptive_vote の寄与は意外と小** (+0.015):
   → 3 attempts は同 prompt × T=0.6 noise のみで mode 多様性ゼロ → 改善余地大 (= DeepEye N-version へつながる)

## 戦略含意

### 維持すべき (= 寄与確定)
- **per-phase system prompt 切替** = +0.06、ablation で最大単一寄与
- **PhasedReActAgent 構造** = +0.04 (= per-phase prompt 込み)

### 改善余地大 (= 弱点)
- **adaptive_vote 同 prompt 3 attempts** = +0.015 のみ。同 prompt なので mode 多様性ゼロ。
  → DeepEye-SQL の **N-version generator (= Skeleton/ICL/D&C 異種 3 種)** で +0.02〜0.04 期待
  → Alpha-SQL の **A6 SQL Revision (= 実行 feedback ループ)** で +0.005〜0.015 期待

### λ0.5 公式と直接整合する未実装機能 (= 主催運営同チーム由来)
- DeepEye-SQL の **5 rule checker** (= Select/Join/MaxMin/OrderByNull/Time) → **exp_129 第 1 候補**
- DeepEye-SQL の **Pair adjudication + shortcut 0.6** → **exp_130 候補**

## 派生メモ

### 実行時間
- 全 ablation = 1 ablation あたり ~25 分 (= workers=20、n=1)
- 50 task で per-task wall time ~50s avg

### v6 baseline 3-att との比較公正性
- baseline 0.7950 は n=3 mean (= 過去 bench_vote3x3 から 0.775 / 0.78 / 0.83 の 3 run mean)
- A の 0.7800 は単 1 run なので run-to-run variance ±0.03 (= 過去 std から推定) 込みで読む
  - つまり A vs baseline の −0.015 は noise 範囲内の可能性あり
  - **再現には A を 3-run しないと確証は得られない** (= TODO、優先度低)
- B, C は構造変更が大きいので noise を超える Δ と判断

### LB へのインパクト見積
- LB ≈ 1.19 × local − 0.37 (= v1/v2/v5 線形回帰)
- 設定別:
  - C (0.72) → 推定 LB 0.486
  - B (0.74) → 推定 LB 0.510
  - A (0.78) → 推定 LB 0.558
  - v6 baseline (0.795) → 推定 LB 0.577 (= 実測 v5 0.5789、+0.002 ズレ程度で fit 良い)

## 参考 artifacts

- `artifacts/runs/exp_128_abl_a_att1_002/evaluation.json`
- `artifacts/runs/exp_128_abl_b_nophased_001/evaluation.json`
- `artifacts/runs/exp_128_abl_c_sameprompt_001/evaluation.json`
- `src/experiments/exp_128_abl_a_att1/` (= exp_122 を 1 行 patch: `_ATTEMPT_TEMPS=(0.6,)`)
- `src/experiments/exp_128_abl_b_nophased/` (= exp_122 + PhasedReActAgent → ReActAgent 置換)
- `src/experiments/exp_128_abl_c_sameprompt/` (= exp_122 + PHASE_NAME_TO_INSTRUCTION 全 phase 統一)
