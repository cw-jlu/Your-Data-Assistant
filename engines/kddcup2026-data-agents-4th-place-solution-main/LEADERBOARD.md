# Leaderboard ledger

公式 LB スコアの記録 + ローカル public-50 スコアとの **gap 追跡**。

> **注意**: 下の「提出履歴」表は **Phase 1 系列** (exp_068/086/122、2026-05、スコア帯 0.45–0.58)。
> 現行の **Phase 2 A-board** は別系列 (下の専用セクション)。同じ vN 番号でも別ボード。

## Phase 2 A-board 提出履歴 (現行フェーズ、gdrive team1418_vN)

評価ホスト: 16 vCPU / 64GB / **GPU なし** (LLM は主催者 EP)、壁時間あり。A-board ノイズ床 **~±0.02** (同一コード v1↔v5 で 0.3664↔0.34916)。**local は A-board を予測しない (むしろ逆相関) — local mean で提出候補を選ぶのは罠。**

| ver | date | exp | git | A-board LB | 備考 |
|----:|------|-----|-----|----------:|------|
| v1 | 2026-06-09 | `exp_149_modality` | ff96358 | **0.3664** | v6 まで最高、workers=8 |
| v2 | 2026-06-10 | `exp_152_final_sql_guard` | — | ≈0.301 | workers=12 で壁死退行 |
| v3 | — | `exp_153_audio_asr` | — | 0.3466 | ≈v1 (ノイズ内) |
| v4 | 2026-06-15 | `exp_154_v1_audio_asr` | da2d43f | 0.3358 | ASR 単体=効果ゼロ〜微マイナス |
| v5 | 2026-06-18 | `exp_149` 再提出 | ff96358 | 0.34916 | v1 と同一コード→ノイズ実測 |
| v6 | 2026-06-23 | `exp_166_domain_rules` | f771ebe | 0.4289 | 🎯 新最高、prior best +0.0625 (ノイズ床超え=本物) |
| **v7** | **2026-06-25** | **`exp_167_domain_pruner`** | **09519e2** | **0.4614** | **🎯🎯 連続新最高、v6 +0.0325 (本物)、EC2 3-run 0.7500** |
| v8 | 2026-06-28 | `exp_171_prose_router` | 7300acc | 0.4647 | ⚠️ v7 +0.0033 = **ノイズ床内＝実質フラット**。local +0.033 は A-board に非転移 (prose-decoy 構造が hidden set に乏しい疑い)。gate-then-select、EC2 3-run 0.7833、task_14 退行修復 |
| **v9** | **2026-07-01** | **`exp_172_ehr_distinct`** | **4a090c0** | **0.5833** | **🎯🎯🎯 過去最大ジャンプ +0.1186、A-board 最終2位で確定** (2026-07-06、3位1573=0.5812と0.0021差; 1位1547=0.6540)。B-board は auto-fallback で v9 が走る。v8 + EHR ドメイン限定 DISTINCT ガード外しのみ→ **hidden A-board は EHR set-valued (route/method) 質問を大量に含む** (demo-60 に無い型=demoは EHR 非代表)。外部 EHRSQL 検証 (32.8% DISTINCT) 起点の補償 fix が最大転移。finance/other は v8 同一なので差分は全て EHR |
| v10 | 2026-07-03 | `exp_172_ehr_distinct` (=v9) + 防御パック | 38eda6d | 0.4947 | **v9 + マルチ動画対応 + 空prediction 2周目** (イメージdiffで差分がこの2点+未使用domain_db除去だけであることをバイトレベル検証済)。①マルチ動画: 旧実装は最初の1本のみ (2本目以降は完全不可視、demo-60は全タスク1本で未露見)→全動画をper-video keyframe note化し `## video:` ヘッダで連結、1本時はv9と同一挙動。合成2動画E2E=score 1.0。②2周目: SUBMISSION_RETRY_EMPTY=1 + BOARD=AUTO (タスク数判別) + 波数対応timeout。E2E: 初回90s全滅→2周目3/3回復全1.0。ENV差分3行のみ。⚠️ 初代v10ドラフト (exp_173 ICL=同源gold SQL同梱) は未提出のままルール抵触判明で撤回、Drive差替 (メール送信前なので合法)。exp_173コードもイメージから除外済 |
| v11 | 2026-07-04 | v10 と同一ビット (tag のみ変更, config 71b0b1a9df50) | 38eda6d | 0.5358 | **分散測定のための再提出** (締切40分前)。v10↔v11 = 同一コードで 0.041 差 → run 間ノイズ σ≈0.03-0.04 を実測、0.4947=コイン裏を裏付け。ただし v10 ビットは 2/2 で v9 の draw 未満 (平均0.515 vs 0.583、~1.4σ・非有意)。**B-board 最終判断: メール送らず自動フォールバック = v9** (一発勝負でモデル不確実性の少ない側、実績ビット、運用リスクゼロ) |

**v6→v7 連続ブレイクスルー (2026-06-27 判明):** v1 に ASR/pruner/guard を足しても A-board は動かなかった (v2–v5 全て v1±ノイズ) が、ここから **2連続で本物の改善**:
- **v6** = 決定論 domain router + value-fidelity preamble + modality note → **0.4289 (+0.0625)**。「domain 別スキーマ/出力ルール注入」が初めて隠し A-board に転移。
- **v7** = v6 + 出力忠実性パッケージ (DISTINCT literal-only ハードガード + datetime/区切り保持 + column auditor ドメインゲート + math-gate 撤回) → **0.4614 (+0.0325)**。「行は合うが値が違う」失点の狙撃も転移した。
- 累計 v1→v7 = **+0.0950** (0.3664→0.4614)。教訓更新: **demo-tuned な pruner/guard は転移しない (旧) が、汎用的な value-fidelity / DISTINCT 規律 / domain ノートは転移する**。per-task ハック ≠ 汎用 fidelity ルール、の線引きが鍵。
- 注: v7 は当日複数 REBUILD (f4f573c→e8e1ac9→09519e2)。0.4614 がどのビルドかは主催者 pull 時刻次第だが、現 gdrive v7=09519e2 は最新(DISTINCT ハードガード込み)で e8e1ac9 の strict superset。

## なぜ追跡するか

- ローカル public-50 (50 タスク手元評価) と公式 LB (隠し test set) は **同じ分布ではない**
- ローカル best と LB best が一致するとは限らない (= local overfit を疑う必要がある)
- 提出ごとに gap を測ることで「local +Δ が LB に何 % 反映されるか」の **相関係数** が見えてくる
- 提出は時間がかかるので、毎回提出する代わりに **gap の歴史** から LB 期待値を予測する判断材料を貯める

## 競争状況 (2026-05-15 更新、v5 結果判明)

**直近 LB スナップショット (2026-05-07 時点、新規 LB は確認次第更新):**

| rank | team | LB | 提出 ver |
|---:|---:|---:|---|
| 1 | 1384 | 0.6311 | v2 |
| 2 | 1227 | 0.5906 | v3 |
| 3 | 1326 | 0.5807 | v2 |
| 4 | 1671 | 0.5098 | v1 |
| **5** | **1418 (= us)** | **0.4969** | **v2** |
| 6 | 1213 | 0.4939 | v1 |
| 7 | 1160 | 0.4693 | v2 |

**v5 = 0.5789** が反映されれば top 3 圏内 (= 当時 3 位 1326 の 0.5807 とほぼ並ぶ、top 1 との gap 0.052)。
他チームも v5 で score 更新している可能性ありなので、最新 LB は要再確認。

**戦略含意 (= 2026-05-07 更新):**
- v1→v2 で **94% LB transfer** (= local +0.047 → LB +0.044) を実証 → local chasing 戦略は valid
- 4 位までは **+0.013 LB = +0.014 local** で届く (= exp_109 微増でも到達可能性大)
- 3 位までは **+0.084 LB = +0.125 local** (= exp_109 + Tier 2 軸が必要)
- top までは **+0.134 LB = +0.20 local** (= 0.94 必要、現 harness 上限 0.78-0.82 を超える、要 Tier 3 大改造)

## 提出履歴

| ver | date_submitted | exp | local 1-run | local n=N mean | local n | **LB** | rank | gap (LB − local_mean) |
|----:|----------------|-----|------------:|---------------:|--------:|-------:|-----:|---------------------:|
| **v1** | 2026-05-03 | `exp_068_kira_function_calling_v2` | 0.7367 (run 005, +1.4σ outlier) | 0.6882 ± 0.0332 | n=7 | **0.4526** | **9/~100** | **−0.2356** |
| **v2** | 2026-05-05 | `exp_086_r1_official_params` (R1: Qwen3.5 公式 sampling) | 0.7449 (run 002) | **0.7353 ± 0.0135** | n=2 | **0.4969** | **5/7+** | **−0.2384** |
| ~~v3~~ | 2026-05-07 | _(チームメイト提出、我々の系譜外)_ | — | — | — | — | — | — |
| ~~v4~~ | — | _(チームメイト提出、我々の系譜外)_ | — | — | — | — | — | — |
| **v5** | 2026-05-13 | `exp_122_column_auditor` (preamble + plan-first + Rule 18 + column auditor + adaptive_vote) | 0.8300 (run high) | **0.7950 ± 0.0304** | n=3 | **0.5789** | _(順位待ち)_ | **−0.2161** |

### v1 → v2 の改善 transfer 分析 (2026-05-07 v2 LB 判明後)

| 指標 | v1 | v2 | Δ |
|---|---:|---:|---:|
| local n=N mean | 0.6882 | 0.7353 | **+0.0471** |
| LB | 0.4526 | 0.4969 | **+0.0443** |
| transfer 効率 | — | — | **94%** (= 0.0443 / 0.0471) |
| LB / local 比 | 65.8% | 67.6% | +1.8pt |
| gap (LB − local) | −0.2356 | −0.2384 | −0.003 (= ほぼ一定) |

### v2 → v5 の改善 transfer 分析 (2026-05-15 v5 LB 判明後)

| 指標 | v2 | v5 | Δ |
|---|---:|---:|---:|
| local n=N mean | 0.7353 | 0.7950 | **+0.0597** |
| LB | 0.4969 | 0.5789 | **+0.0820** |
| transfer 効率 | — | — | **137%** (= 0.0820 / 0.0597) ⬆️ |
| LB / local 比 | 67.6% | 72.8% | **+5.2pt** ⬆️ |
| gap (LB − local) | −0.2384 | −0.2161 | **+0.022 (= 縮小)** ⬆️ |

**結論 (= 2026-05-15 update):**
- v1→v2 で 94% transfer、**v2→v5 で 137% transfer** (= local の改善 +0.060 に対し LB +0.082 と上回り)
- **gap が初めて縮小** (−0.239 → −0.216) → public-50 改善が hidden test により強く効いている
- LB / local 比が **72.8%** まで上昇 (過去最良) → public-50 と LB の domain bias が縮まりつつある
- → exp_122 系 (= rich preamble + plan-first + Rule 18 + column auditor + adaptive_vote) の改善は **真の汎化** (= LB 用に汎用的に効く)
- LB 期待値式更新: `LB ≈ local × 0.73` または `LB ≈ local − 0.22` (= v5 ベース)
- local 0.82-0.83 を維持できれば LB ≈ 0.60〜0.62 が見え、top 0.6311 に届く射程

## v1 — exp_068_kira_function_calling_v2

**Local 詳細:**
- 1-run record (run 005): λ0.5 = **0.7367** (= +1.4σ outlier、後で再評価で否定された "1-run record")
- n=3 rep1 (runs 005-007 の最初の 3): mean ≈ 0.71 (記憶頼り、要確認)
- n=3 rep2 (runs 006/007/008): mean = 0.6828, std = 0.0411
- **n=7 pooled mean = 0.6882, std = 0.0332** (最も信頼度の高い local 値)

**LB:** 0.4526 (公式提出)

**Gap 観察:**
- LB − pooled_mean = **−0.2356** (local が 0.236 高い、相対的に 34% 高め評価)
- LB − 1-run_record = **−0.2841** (1-run の 0.7367 を信じると 0.28 過大評価)
- → **public-50 の 1-run 値で意思決定すると LB から見て大幅 overshoot する**
- → n=7 pooled でもまだ +0.236 high — public-50 自体が LB より易しい可能性が高い

**仮説 (gap の原因):**
1. **Distribution shift**: 公開 50 が LB 隠し test より簡単な domain bias を持つ
2. **Leak (Tier 1/2)**: pretrain corpus に類題を見ているため public-50 で score が浮く (= リーク無し統制でも実は弱いリークあり)
3. **Submission infra**: Docker / 提出時の時間制限 / API rate / 環境差で local と挙動が違う
4. **n=7 → infinite gap**: 公開 50 の noise を平均しても hidden test の真値には収束しない (= sample 不一致)

## 今後 submission で記録するもの

各提出時に以下を埋める:
- `date_submitted`: ISO 日付
- `exp`: 提出した EXPERIMENT_NAME (Dockerfile)
- `local 1-run`: 最高 1-run スコア (replicate のうち)
- `local n=N mean`: 同実験の n-replication mean ± std
- `LB`: 公式 LB の数値
- 派生: `gap = LB − local_n=N_mean`

## 期待される観察

- **gap が実験ごとに 安定** (例: 常に −0.20〜−0.25) → public-50 と LB は equally biased、local 改善は LB に係数で transfer する
- **gap が広がる方向** (例: local が伸びるほど LB との差が増える) → local overfitting、public-50 だけに効く改善を追ってる
- **gap が縮まる方向** → local は厳しめ、改善が LB に増幅される (一番嬉しいケース、稀)

## トップとの比較 (top 0.65 から逆算)

| 仮説 | top の local 想定 | 我々の改善方向 |
|------|-----------------:|-------------|
| **gap 定数 (−0.236)** | local 0.886 が必要 | 現実的でない、transfer 効率の見直し必要 |
| **gap 比例 (LB×1.52)** | local 0.988 | これも非現実的 |
| **top は gap が小さい (例 −0.10)** | local 0.75 | top チームは local overfit が少ない構造 — 我々の public-50 改善が空振り |

→ 一番現実的なのは **(C) top チームは local-LB transfer が高効率**。同じ local 値でも我々より LB に変換できる構造を持っている。

**含意となる戦略変更:**
- local public-50 で +0.05〜+0.10 取りに行くより、**local 値を維持しつつ LB に transfer する設計** に重心を置く
- 具体的には:
  1. **public-50 の特定タスクパターンに依存しない** generic な改善 (qwen3 paper の thinking budget / FC format / YaRN 64K のような **モデル本来の能力を解放する**方向)
  2. **異なる data domain でも頑健**な実験 (= 公開 50 にだけ効く Rule の追加は LB transfer しにくい)
  3. **次の submission で gap を測る** — 1〜2 回 submit して transfer 係数を確定させる
- exp_081 union (現状 best) を提出して LB 確認するのは **十分価値あり** — gap データ点が増える

## 注意

- LB スコアは **公式 dashboard の数字をそのまま貼る** こと。手元での再計算と混ぜない
- 1 提出の LB 値は **noise 入り** (= LB 内部 split の sampling noise)。gap の方向を見るには 3〜5 提出が最低必要
- 毎回提出するのはコスト高 — local で大幅 +Δ が出た時だけ提出する戦略が無難
