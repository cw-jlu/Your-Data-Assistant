# Priorities (human-only injection)

This file lets a human steer the experiment loop without editing code.
The `pick` step of `experiment-iter.yaml` reads this file at each
iteration and gives `[URGENT]` items the highest priority.

**📋 LEADERBOARD ledger**: `LEADERBOARD.md` に LB スコアと local の gap を記録中。
v1 (exp_068) LB=0.4526 vs local n=7=0.6882, gap=−0.236。**新 submission 出したら必ず
LEADERBOARD.md に行追加** (date / exp / local 1-run / local n=N / LB / gap)。提出判断は
「local n=3 で明確に +Δ > 0.02 出てる実験」のみが推奨。

## How to use

- Add a single bullet like `- [URGENT] Try max_workers=24 ...`
- Remove or comment out the line once the loop has consumed it.
- One urgent item at a time keeps the workflow predictable.


## Current

- [URGENT] **⏱️ task_timeout_seconds = 900 を新基準として固定 (2026-05-03 user 指示)**
  - exp_074_infra_timeout_900 検証結果: λ=0.7083 (Δ−0.028 vs exp_068)。Type A 5 タスク (task_180/344/379/396/418) のうち **4 件は 900s 内に完走したが誤答** (= 実は Type B ロジック失敗)、残り 1 件 (task_180) は 900s でも依然 timeout。
  - 900s でスコアは下がったが **global miss が 4→2 に改善** (スループット安定化、ノイズ要因 1 つ除去)。
  - **方針**: `src/experiments/exp_040_selfdbg_fence/config.yaml` の clone 元基準値を 900 に更新済 (今後の cp -r で自動継承)。
  - **新 exp の `config.yaml` で `task_timeout_seconds: 900` を確認**すること。submission Docker は別管理 (`MAX_STEPS` env 経由で制御)。


- [URGENT] **🔒 Global single-bench lock (2026-05-03 user 指示)**:
  `launch_bench.sh` を改修: 他の bench が走行中なら **30s ポーリングで blocking wait**。
  workers=8 を内部並列にしつつ、bench process 自体は **常に 1 つだけ**。
  
  **理由**: 並行 2 bench (workers=8 × 2 = 16 同時 API) で vLLM 飽和 → per-task latency 増 →
  missing 増 → スコア −0.05〜−0.10 depressed。replicate exp_040 で実証 (0.7267 → 0.62 圏)。

  **影響**:
  - takt は launch_bench でブロック wait → 並行 discovery がシーケンシャル化 (体感 sequential workflow に近い)
  - 1 イテレーション ~45-50 min (旧 pipeline 30 min から増)
  - 1 day = ~30 runs (旧 40)
  - **score ノイズ ↓、信頼度 ↑** — 1-run benchmark の数値が信頼できる
  - 将来的に過去スコアの再検証必要 (concurrent 期の 1-run 値は割引)



- [URGENT] **🔧 max_workers = 10 に revert (2026-05-05 user 指示) — 5/4 の workers=20 化は誤り**

  ### 経緯
  5/4 にローカル 20 vCPU 化を理由に workers=4→20 へ bump したが、**ローカル CPU は元々ボトルネックではなく**、vLLM (`gpu-host.internal`) 側の同時処理容量が真のボトルネックだった。20 同時 stream は vLLM 飽和点 (~12-15 streams) を超え、per-task latency が 2.4x に悪化、900s cap で timeout missing が 10x 増加した。

  ### 実測 (2026-05-05)
  | 構成 | workers | streams | per-task median | missing/50 | score |
  |---|---:|---:|---:|---:|---:|
  | exp_068 fc_v2 | 8 | 8 | 120s | 3.7 | 0.683 |
  | exp_081 union | 4 | **12** | 248s | **1.0** | **0.693** |
  | exp_082 cursor | 20 | 20 | 283s | 8.3 | 0.617 ❌ |
  | exp_083 additive | 6 | 18 | 473s | 10.0 | 0.574 ❌ |

  ### 新基準 (2026-05-05〜)
  - **single-attempt 実験**: `max_workers: 10` (= 10 streams, vLLM 飽和点 ~13-15 の安全側)
  - **3-attempt union 実験 (exp_081 base)**: `max_workers: 4` (= 4×3 = 12 streams, exp_081 で実証済)
  - `submission/Dockerfile`: `MAX_WORKERS=10` (revert 済)
  - `src/experiments/exp_040_selfdbg_fence/config.yaml`: `max_workers: 10` (revert 済)

  ### 履歴
  - workers=4 (5/2 まで): miss=1 / λ=0.7267 (exp_040 元値、lock なし、solo)
  - workers=8 (5/3-5/4): miss=3-7 / λ=0.60-0.69 (lock あり、solo, replication 多数)
  - workers=20 (5/4-5/5): **誤った設定**、exp_082/083 のスコアは過小評価
  - workers=8 (5/5 以降): revert、新基準


- [URGENT] **⚠️ Qwen3.5 thinking 制御は `extra_body` のみ — `/think` `/no_think` soft switch は非対応 (2026-05-05 重要訂正)**

  Qwen3.5-35B-A3B 公式 model card 明記:
  > "No official soft switches (`/think`, `/no_think` are NOT supported for Qwen3.5)"

  Qwen3 (旧モデル) では使えたが **Qwen3.5 で非対応**。user message に
  `/think` や `/no_think` を suffix 付加しても **ただの文字列として扱われる**
  (= 機能しない、ただし error も出ない silent failure)。

  ### 正しい per-call 制御方法
  vLLM の `extra_body.chat_template_kwargs.enable_thinking` のみ:
  ```python
  client.chat.completions.create(
      ...,
      extra_body={"chat_template_kwargs": {"enable_thinking": True/False}}
  )
  ```

  ### kobushi_core/model.py での対応 (2026-05-05 完了)
  `OpenAIModelAdapter.complete()` と `complete_with_tools()` の両方に
  **per-call `enable_thinking: bool | None` 引数**を追加済み:
  - `None` (default) → instance の self.enable_thinking (= True)
  - `True` / `False` → per-call で override

  使い方 (例: R4 hybrid 切替):
  ```python
  # Plan step (deep thinking 必要):
  resp = adapter.complete_with_tools(msgs, tools, enable_thinking=True)
  # Tool exec step (即実行、thinking 不要):
  resp = adapter.complete_with_tools(msgs, tools, enable_thinking=False)
  ```

  ### takt design_next への注意
  - **R4 「hybrid /think 切替」軸の実装は user message suffix を使わないこと**
  - **必ず `enable_thinking` 引数で制御**する
  - `agent.py` の `model.complete*()` 呼び出しに条件付きで `enable_thinking=False` (or True) を渡す
  - smoke で「latency が `/no_think` step で本当に短縮されてるか」確認 (= 1 step 単位の latency log で判定可能)


- [URGENT] **❌ aggregation-scale 軸 (n=4+ attempts) は採用しない (2026-05-06 user 指示)**

  N attempts を増やす方向 (例: 3 → 5、union diversity 拡大) は **採用候補から外す**:
  - **理由 1 (cost)**: N=5 は 3-attempt の 1.67x、wall-clock 同比率増加
  - **理由 2 (低 ROI)**: R2 (k=2) で「集約方法を変えても base 効果以上は出ない」既に判明
  - **理由 3 (戦略不一致)**: big-effect (+0.05+) 狙いに対し aggregation-scale は +0.005〜0.02 効果、最大化軸でない

  - 設計済 exp_096_aggr_scale_5attempt は **削除済** (2026-05-06)
  - 今後の design_next は **n=4+ attempts 系を pick しない**
  - 集約軸 (k 値変更、temperature 分布変更も含む) は当面保留


- [URGENT] **🚀 big-effect 軸狙い + 探索 default single-attempt 化 (2026-05-06 user 指示、戦略切替)**

  ### 背景: 微増軸が出尽くしつつある
  3-attempt 同 class 内比較:
  | 軸 | 結果 |
  |---|---|
  | R1 公式 sampling | ✅ **+0.045** (exp_086) |
  | R2 column-vote k=2 | ❌ −0.026 |
  | R3 M-Schema + Plan-first | ❌ −0.044 |
  | R5 structured error (n=1) | −0.012 (noise 内、低い) |

  → 微増軸は枯渇方向。LB top 0.65 vs 我々 v2 推定 ~0.50 の **0.15 gap は微増では埋まらない**。

  ### 新方針 (2026-05-06 user 指示):
  **「微増狙いやめ、+0.05 以上の big-effect 軸のみ追求」**

  ### Default class 切替: 新軸の探索 = single-attempt
  - 1 軸あたり cost 1/3 (~75 min vs ~225 min)、1 日試せる軸数 **3x 増**
  - **+0.05 以上の big-effect** は single-attempt の noise (std ~0.04) でも検出可
  - **+0.01-0.02 微妙な軸はそもそも狙わない** (今回は勝てない)
  - 探索 winner のみ 3-attempt 化して submission v3 候補に昇格

  ### 候補 big-effect 軸 (takt design_next 参考に)

  **(A) Architecture rebuild ← 最優先**
  - **exp_082 cursor_harness 再評価**: workers=20 期で infra 汚染されていた値 0.6174。clean infra で再 bench → schema-only + grep_file/stat_file は frontier 系で +0.05 期待
  - **execute_python の super-tool 化**: 細かい tool を `execute_python(code)` 1 本に統合
  - **Plan-Act-Verify with rule-based verifier (non-LLM)**: 過去 G1/G2 self-bias で失敗、代わりに column 数 / row 数を question から推定して mismatch なら retry 要求 (= LLM 自己評価ではなく code 検証)

  **(B) Question preprocessing**
  - **質問構造化前処理**: 自然言語 question を「列リスト + フィルタ + 集計タイプ」に LLM で事前変換、agent に構造化 input を渡す
  - **column candidate hint**: question から「答えに含まれそうな列名」を事前推測して preamble に injection

  **(C) Tool / Discovery 強化**
  - **smart inspect_table v2**: 過去 inspect_table は null% 早期打ち切り罠で失敗。dtypes + sample のみで null% 排除版
  - **execute_sql vs execute_python の明確化**: SQL タスクは execute_sql、analytical は execute_python に強制割り当て

  **(D) Qwen-Agent native runtime — ❌ 廃案 (2026-05-06、実験 + audit で否定)**

  検証結果 (exp_094_qwen_agent_runtime_swap, n=1):
  - λ=0.7119 / miss=1 vs exp_086 (n=2 mean=0.7353) → Δ = **-0.024** (劣化)
  - dup_cols rate: 18% vs exp_086 10% (3-attempt union が dup-cols を多く keep)

  **真因 audit**: `qwen_agent/llm/base.py:177-178` で **call ごとに `random.randint(0, 2^30)` を auto-inject**:
  - exp_086 (OpenAI adapter): seed 非送信 → vLLM session-RNG 連続 → 3 attempts が比較的収束
  - exp_094 (Qwen-Agent): 各 attempt が異なる random seed → vLLM が seed-deterministic sampling → 3 attempts が **意図せず diverge** → union が dup-cols 増加 → score -0.024

  修正試行: `seed=None` を generate_cfg に明示 → openai client が `null` で wire に送り、vLLM が再ランダム化、効果なし。完全削除には qwen-agent 内部 monkey-patch が必要、ROI 悪い。

  **結論**: qwen-agent OAI backend は openai client の薄い wrapper。seed auto-inject 以外は完全同一の wire 動作。**vendor magic 仮説は存在しない**ことが実証された。framework swap 系は今後試さない。

  以下、廃案前の議論履歴 (= 学習素材):

  **(D-historical) Qwen-Agent native runtime (= 同 ReAct logic、wire は Qwen-team 製 library に)** ← user 提案 2026-05-06
  - **仮説**: Qwen-Agent は **Qwen team が直接書いた Qwen 特化 library**。我々の自作 OpenAI-compat 経由よりも、tool description format / chat template / multi-turn history の組み立て方が **Qwen3.5 の training distribution に native fit** している → 同じロジックでも model にとって自然な input になり、score が上振れする。
  - 実装: 我々の `ReActAgent` / tool registry / prompt design はそのまま、`OpenAIModelAdapter` を `Qwen-Agent.Assistant` の LLM wrapper に置き換えるだけ
  - 効果サイズ予想:
    - Chat template native fit: +0.005〜+0.02
    - Tool description format optimization: +0.005〜+0.02
    - Hidden vendor tweaks (= 公開 model card 外の knowledge): +0.0〜+0.03
    - **合算予想 +0.01〜+0.07** ← 下限でも foundation、上限なら big-effect 軸
  - 実装コスト: 半日〜1 日 (adapter 1 つ書き換え、agent.py ロジック温存)
  - 失敗時 fallback: adapter 戻すだけ = 低リスク
  - 副次効果: 以後の小改善が「公式 native」の上に乗るので乗算的に効く可能性

  ### ⚠️ 重要: 結果が出なくても「効かなかった」と即断しない
  Qwen-Agent runtime swap は **wire-level の差し替え**なので、score が出ない場合の原因は **多くが impl bug**。analyze_prev / pick_next は廃止判定する前に必ず以下を全部確認:

  1. **API 呼び出しの diff**: 旧 OpenAIModelAdapter と新 Qwen-Agent wrapper で送ってるリクエストを log diff (= messages, tools, extra_body, etc.)
  2. **Tool schema 比較**: tool description が JSON Schema format に正しく変換されてるか、引数の型注釈が落ちてないか
  3. **Chat history の strip**: thinking content が次 turn に leak してないか (= Qwen-Agent の Jinja2 と我々の `strip_thinking()` が二重で同じ仕事してる、or どちらも欠けてる)
  4. **Response parse**: tool_calls の取り出し、reasoning_content の扱い、content=null 時の挙動
  5. **Sampling params 維持**: max_tokens=32K, presence_penalty=0, top_p=0.95 等が Qwen-Agent 経由でも届いてるか (引数名の変更可能性)
  6. **Authentication**: CF Access headers が API 呼び出しに乗ってるか
  7. **Model name**: `qwen3.5-35b-a3b` が正しく渡ってるか (Qwen-Agent が model registry を持ってると別名になる可能性)
  8. **Tool execute 結果の return format**: tool 実行結果を model に戻すときの format (string vs dict、JSON エンコード)

  → これらを diff log で 1 つずつ潰しても **全く同等** なら、初めて「framework swap は効果なし」と結論。実装 bug を残したまま「効かなかった」と廃棄するのが最悪 (= 真の効果が見えない上に、bug fix の機会も失う)。

  ### 推奨運用
  - smoke test で 1-3 task 走らせて trace.json を旧 adapter と diff
  - 数値が違ったら原因分析 (= 上記 8 項目の差分洗い出し)
  - bench に出すのは smoke で動作一致確認後
  - bench 結果が低い → 即廃棄じゃなく、**「3 attempts のうち 1 つだけ adapter 替えて A/B」** 等の細粒度比較も検討
  - **Qwen-Agent full migration (= 全構造移行)**: BACKLOG 中長期候補、
    公式 chat template + MCP + 内蔵 R1/R4/R5、一気に +0.05+ 期待。大工事 1-2 日
  - **DS-STAR style multi-agent**: Planner + Coder + Verifier + Router、
    frontier 系で +20pp 実績、qwen3.5-35b で transfer 効くか不明

  ### 既存 queue の扱い
  - **R5 (exp_090) は完走させる** (run 1 sunk cost、続行 ROI 良)
  - **R4 (exp_089 hybrid think) は廃止 or 後回し**: cost 削減目的で score boost 期待小、big-effect 路線と合わない
  - **C5 (exp_091) は内容確認**: 大改造系なら継続、微増系なら廃止

  ### 比較ルール (前 URGENT 継承、変更なし)
  - 同 attempts-class 内のみ score 直接比較
  - single-attempt < 3-attempt は class 違いで failure 判定しない
  - single-attempt 新軸の有望性は **「exp_086 - 0.04 ≈ 0.69」を pseudo-base** として比較


- [URGENT] **🎯 探索 = single-attempt、確認 = 3-attempt — attempts-class で公平比較 (2026-05-05 user 指示)**

  ### 問題
  3-attempt union は **+0.03〜0.05 の加算 boost を任意の base に提供**するが、**3x time + 3x cost**。
  毎実験に default で乗せると:
  - 新軸の **真の効果が探索段階で見えにくく**なる (boost に埋もれる)
  - **bench 時間 3 倍** で試行数が 1/3 に減る
  - 単発実験を 3-attempt と直接比較すると、「class が違う」だけで誤って廃棄しがち

  ### 解決: 「attempts-class」で実験を分類 (= 概念的なガイダンス、固定 base は強制しない)
  | Class | 用途 | 識別方法 |
  |---|---|---|
  | **single-attempt** | **新軸探索 (cheap)** | runner.py の `_ATTEMPT_TEMPS` 長 == 1、または name 末尾 `_single` |
  | **3-attempt** | **submission 候補 (expensive)** | `_ATTEMPT_TEMPS` 長 >= 2、または name 末尾 `_union`/`_3x`/`_majority` |

  3-attempt 現王者: `exp_086_r1_official_params` (n=2 mean=0.7353)
  → 新軸を実装するときに「single-attempt 化するか / 3-attempt 維持か」は **takt の design_next が状況に応じて判断**。

  ### 比較ルール (analyze_prev / pick_next が必ず守る)
  1. **同 class 内**: 直接 score 比較 OK (axis 効果の判定)
  2. **cross-class 比較は禁止 (= 厳密比較しない)**: 「3-attempt ≈ single + ~0.04 boost」を暗黙仮定で粗推定だけ
  3. **single-attempt が 3-attempt より低くても "失敗" と判定しない**: class が違うだけ
  4. **新軸の有望性は single-attempt class 内 base (exp_089) との Δ で判断** ← 最重要

  ### 推奨される判断パターン (固定義務ではなくガイダンス)
  - 新軸の **初回試行** は single-attempt class で速く回す案を検討
    (cost 1/3、1 日で 3x 多くの軸を試せる)
  - **+Δ > 0.02** のシグナルが見えた軸は 3-attempt 化して submission 候補に昇格
  - 既存 `exp_086 / 087 / 088` は **3-attempt class でこのまま続行** (廃棄しない、submission 戦力)
  - ただし「単発が低分散の 3-attempt より検出感度低い (std 0.04 vs 0.014)」点は要注意。微妙な軸の判定では 3-attempt を選ぶ価値がある場合も多い

  → 結局は **状況依存**。design_next (Opus) が:
  - 過去結果の variance / 期待効果サイズ
  - 残り bench 時間予算 (= submission deadline)
  - その軸が「明確に大きい効果」期待か「微妙な差」期待か
  
  を踏まえて class を選ぶ。

  ### 注意
  - 既存 single-attempt 実験 (exp_068 等) は R1 sampling 入ってないので **R1 適用後の単発実験とは別 sub-class**。直接比較は注意。
  - exp_088 (R3 M-Schema、launch 待機中) は 3-attempt 設計済、そのまま完走させて R3 in 3-attempt class の data point にする。


- [URGENT] **🧹 vLLM 502 outage 汚染 run の判定 + 廃棄ルール (2026-05-05 user 指示)**

  **現象:** `missing_prediction_count` が異常に高く、trace.json `failure_reason` に **"Error code: 502" や timeout** が頻出する run は **vLLM 鯖落ちで汚染**された infra ノイズ。スコアは無意味、むしろ n=N mean を引き下げて誤った判断を誘発する。

  **判定基準 (どれかに該当で疑う):**
  - `missing_prediction_count >= 5` (= clean run の典型は 0-2)
  - 複数 missing の `e2e_elapsed_seconds` が **481s 前後に集中** (openai client max_retries=8 の budget exhaust 値)
  - trace.json failure_reason が `Error code: 502` を含む
  - 同じ exp の他 run 比 **−0.1 以上スコアが低い**

  **対応:**
  1. `artifacts/runs/<exp>_<NNN>/` を `artifacts/runs_discarded/<exp>_<NNN>/` に **移動**
  2. 移動先に `REASON.md` (= 廃棄理由、観察症状、outage 時刻)
  3. `artifacts/replications/<exp>/summary_*.json` を **n を減らして再計算**
  4. dashboard refresh: `uv run python scripts/update_docs.py`
  5. 真の n が <3 なら、vLLM 健全時に **clean replicate を 1 回追加**して n=3 完成

  **既知の廃棄 run:**
  - `exp_086_r1_official_params_004` — λ=0.5567, miss=13, 502 outage 16:00-17:25 期間内に被弾。
    真の n=2 (run_002 + run_003) mean=**0.7353 ± 0.0135** が exp_086 R1 の正値。

  **2026-05-05 vLLM outage 履歴:**
  - 04:00 頃 (~3.5 hr): exp_083 run_003 (λ=0.0925/miss=44) 廃棄
  - 16:00-17:25 頃 (~1.5 hr): exp_086 run_004 廃棄
  - → **同日 2 回の鯖落ち**。gpu-host.internal の vLLM プロセス安定性に懸念。GPU OOM / メモリリーク / 自動再起動疑い。user 側で監視・systemd 設定見直し余地。


- [URGENT] **♻️ workers=20 期 (5/4-5/5) のスコア再評価キュー**

  以下の実験は workers=20 (= vLLM 飽和) 条件下で評価され、**真のスコアより過小評価**されている可能性が高い。workers=8 に戻して再 replicate (n=3) する必要あり:

  1. **exp_082_cursor_harness**: 公式記録 n=3 mean=0.6174 (Δ-0.046 vs baseline)。**再評価必須** — もしかすると schema-only 設計自体は中立で、missing 8.3 が score 全部の責任の可能性あり。  
     - 再 replicate 設定: `max_workers: 8`, single-attempt のまま
     - 期待: missing が 1-3 に戻れば score +0.05〜+0.10 上振れ可能性
     - 結果が baseline (0.6633) を回復するなら **schema-only 軸の廃棄判断は撤回**

  2. **exp_083_additive_tools**: ✅ run_002 + run_003 両方完了。n=2 pool mean=0.3331 ± 0.3402 → **判定不能 (infra 汚染)**。
     - run_002: λ0.5=0.5736, missing=10 (530 burst ×4 + workers=6 timeout ×6)
     - run_003: λ0.5=**0.0925**, missing=44 — **vLLM 52分 502 outage (18:47Z–19:39Z) が支配**、workers=6 の構造問題とは別事象
     - 有効完了タスクのみ: run_002 40t→λ=0.7170 / run_003 6t→λ=**0.7708** → 設計は一貫健全
     - **task_11 が 2 run 連続 zero_recall** = union k=1 dtype 副作用の構造的バグ (修正必須)
     - stat_file=0 calls が **2 run 連続確認** = 廃棄確定 → 次回 exp で必ず削除
     - **次の action**: vLLM 復旧確認 → workers=4 修正 + dtype-normalize + stat_file 削除 → n=3 clean replicate

  3. **過去の workers=8 → workers=20 期 reanalysis 不要**: exp_068 など 5/3-5/4 期は workers=8 で評価済み (= 飽和点内)、再評価不要。

  ### 実施順
  1. ✅ run_002 完了・分析済み
  2. ✅ run_003 完走確認 (λ0.5=0.0925, missing=44 — vLLM 502 outage 汚染、評価不能)
  3. ✅ analyze_prev + log_prev で vLLM outage + workers=6 config error を記録 (EXPERIMENTS.md + BACKLOG.md 更新)
  4. **[NEXT]** vLLM 復旧後、workers=4 + dtype-normalize + stat_file 削除で n=3 clean replicate を実施

  ### 副次的影響
  - submission/main.py が exp_068 (workers=8 baseline) なので **submission 側は健全**、A-board に影響なし
  - replication summary (artifacts/replications/) の exp_082 サマリは破棄、exp_083 完走後の summary も再 replicate 後に上書き


- [URGENT] **🚫 リーク禁止ルール拡張 (永続)**: 詳細は `.takt/facets/knowledge/kddcup2026-state.md`

- [URGENT] **🚫 リーク禁止ルール拡張 (永続)**: 詳細は `.takt/facets/knowledge/kddcup2026-state.md`
  の「🚫 厳守ルール: テストデータ・リーク禁止」を **必ず全文読む**。

  **重要な拡張 (2026-05-02)**: 「リーク」は verbatim コピーだけでなく **構造リーク (Tier 2) も含む**。
  今後の fewshot は **公開 50 のどのドメイン (student_club / formula1 / financial / molecule
  / cards など) とも被らないスキーマ**を使うこと。

  **真のリーク無しベスト (n=3 confirmed) = exp_081_union_t0 (mean=0.6906, std=0.0138)** (2026-05-04 更新)。submission v2 候補。`submission/main.py` の切り替えは A-board 120s 制約での local smoke 確認後に実施。
  **既知の汚染ベスト** (rank 対象外): exp_026/027 (Tier 1)、exp_017/023/029 系列 (Tier 2)。

  **新 exp の implement / smoke で必ず行う検証**:
    1. system prompt / preamble に公開 50 タスクの question (40 文字以上) が含まれない
       (`grep -F` で確認)
    2. fewshot 例があれば DB スキーマが student_club/formula1/financial 等と被らない
    3. 違反検出時は smoke FAIL → implement に戻る


- [URGENT] **📚 Deep research 由来の Tier 1 改善 5 軸 (2026-05-05、外部 deep research レポート反映)**

  外部 deep research が同じく Qwen3-30B-A3B + DABench で詳細調査。我々の現状実測と整合する事実 + 公式仕様 + 関連論文 ablation の数値を踏まえた **Tier 1 (期待効果 vs 実装コスト 良)** 5 軸:

  ### 軸 R1 [infer] **inference parameter 公式化** (即効、最低コスト) [+0.005〜+0.015]

  Qwen3-30B-A3B 公式 model card は **T=0.0 (greedy decoding) を明示禁止**:
  > "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions"

  **現状の attempt 1 が T=0.0** で運用 → 公式に反する。修正:
  - attempt 1: **T=0.6, top_p=0.95, top_k=20, min_p=0, /think on, presence_penalty=1.0**
  - attempt 2: **T=0.7, top_p=0.8, top_k=20, /no_think, presence_penalty=1.5**
  - attempt 3: **T=0.6, top_p=0.95, /think on, presence_penalty=1.0** (alternative-interp prompt)
  - 実装コスト: runner の API call 3 行修正
  - ベース: exp_081_union_t0
  - 期待: thinking-loop 故障モード激減、attempt 1 の hit 率上昇

  ### 軸 R2 [aggr] **column-signature vote (⌈n/2⌉ threshold)** [+0.02〜+0.04]

  DABench scoring は **column content signature ベース** で照合 (公式 dataagent.top/rules#section-6-2)。これは Spider/BIRD と異なる本ベンチ固有の構造で、execution-based clustering が極めて効きやすい。

  現状の k=1 union → **⌈n/2⌉ confidence-weighted column voting** に置換:
  - 各 attempt の各列を値 multiset signature 化 → canonical column に正規化
  - canonical column ごとに「何 attempt に出現したか」を count → ⌈n/2⌉ 未満は drop
  - これで extra_cols ratio 直接削減 → λ penalty 直接縮小
  - 根拠: ReFoRCE arXiv 2502.00675 「removing majority voting -1.83〜-2.01pp」, RSL-SQL arXiv 2411.00073 「binary mode vote +1.37pp」, XiYan-SQL 2411.08599 「selection なしの SC は ~3pp 落ちる」
  - ベース: exp_081_union_t0

  ### 軸 R3 [context] **M-Schema 風 preamble + Plan-first column declaration** [+0.02〜+0.04]

  XiYan-SQL (arXiv 2411.08599, BIRD test 75.63%) は **DDL → M-Schema の semi-structured 変換だけで一貫した利得**:
  ```
  [DB_ID] task_NNN
  ├── # Table sales (販売トランザクション)
  │   ├── (sale_id, INT, PK, 例: 1001..9999)
  │   ├── (region, TEXT, 例: "East Asia", "EMEA", "NA")
  │   └── (revenue, REAL, 例: 12345.67)
  └── [Foreign Keys] sales.region = targets.region
  ```
  + **step 0 で `# Plan: candidate columns are [c1, c2, ...]` を必ず出力**させる枠。Rule 12 (Plan-first +0.030) の延長で、「question から答えに必要そうな列を 3〜7 列宣言」させる。

  ### 軸 R4 [thinking] **thinking mode の段階別切替 (hybrid)** [+0.01〜+0.03、token -40%]

  P1 (`/think` 全 ON) ではなく、**ステップごとに切替** が最適:
  - step 0 (Plan): `/think on`, T=0.6 — 計画は深く考える
  - step 1+ (tool call / exec): `/no_think`, T=0.7 — 単純実行
  - step final (answer): `/no_think`

  根拠: Kunal Ganglani 2026 (Qwen3-32B 実測): **「cut my total token usage by roughly 40% with no measurable accuracy loss」** verbatim。

  これにより、同じ予算で **n=3 → n=5 attempt** に拡張可能 → 軸 R2 の voting 精度↑。

  ### 軸 R5 [tool] **structured execution feedback (guided error hint 簡易版)** [+0.01〜+0.025]

  我々の exp_046 「error hint truncation 80% generic で誘導効果ゼロ」は、SQL-of-Thought (arXiv 2509.00581) の発見と整合: **実行 trace の 95-99% は既に valid syntax で、生 traceback では誘導できない**。

  解決: execute_python wrapper で例外を **分類** して、各分類に「次に試すべきヒント 1 行」付与:
  - `ColumnNotFound: "X" → HINT: available columns are [Y, Z, W]`
  - `TypeMismatch: cannot compare str and int → HINT: cast to int with int()`
  - `EmptyResult: filter returned 0 rows → HINT: relax filter, check value distribution`
  - `Timeout / OOM → HINT: reduce data scope, sample first`

  これは exp_046 の延長で、「exception class 別分岐」を実装 (前回提案した方向と整合)。

  ### 実施順序 (推奨)

  1. ~~**R1 (infer params)**~~ ✅ **消化済み (exp_086, 2026-05-05)** — n=2 mean=0.7353 ± 0.0135, 確定 win (+0.045 vs exp_081)
  2. ~~**R2 (⌈n/2⌉ vote)**~~ ❌ **廃棄 (exp_087, 2026-05-05)** — n=2 mean=0.7098 (Δ-0.0255 vs exp_086)。extras 削減は確認だが recall loss が 4-10× 上回る。k+diversity 逆相関確認: k=1+高diversity が SOTA
  3. ~~**R3 (M-Schema preamble + Plan-first)**~~ ❌ **廃棄確定 (exp_088, 2026-05-06)** — n=2 clean mean=0.6915 ± 0.0270 (Δ-0.044 vs exp_086)。runs: 0.7106/0.6724 (run_004 vLLM outage 廃棄)。task_11 × task_25 両 run consistent zero (三項相互作用 + `_signature_majority_merge` row-padding バグ)。**M-Schema は single-attempt 限定で再評価候補**。`_signature_majority_merge` row-padding バグ修正が独立 fix 価値 (+0.02〜+0.04 横断効果)。
  4. **R4 (hybrid /think 切替)** ← exp_089 で実施中 (bench 走行中)
  5. ~~**R5 (structured error hint)**~~ ⚠️ **n=1 ノイズ範囲内 (exp_090, 2026-05-06)** — λ0.5=0.7236 (Δ=-0.012 vs exp_086)。hint 発火 19/50 だが実効 2 class のみ (KeyError/FileNotFound)。error-feedback 系統 3 連続微細効果 (exp_046/056/090)。replication 廃棄推奨。

  R1 確定 win (+0.045)。R2 廃棄確定。R3 実施中。R3+R4+R5 全部入れて **0.73 → 0.78〜0.80** が目標。extras 削減は R3 Plan-first 軸で攻める。

  ### 罠 (試さない確定):

  - **Reflexion / Self-Refine 単純導入**: CHASE-SQL ablation で BIRD dev -4.17pp (= self-bias、我々の exp_051/056 失敗と整合)
  - **Multi-agent (Schema Selector / Verifier 分離)**: 中規模 MoE では sub-agent も同じモデル、レイテンシ増のみ
  - **Tool 多本化 (Cursor 流 20 本+)**: 中規模はtool-selection 精度が下がる
  - **execution trace そのまま feedback**: 95-99% は valid、誘導効果ゼロ

  ### 注意 (deep research caveat):

  - **Qwen3-30B-A3B-Instruct-2507 は thinking mode が無い**ので、もし vLLM がそのバージョンを serve してたら R1/R4 が失効。**現在の serve は Qwen3-30B-A3B (Apr 2025 初版) であることを確認**してから着手。
  - vLLM hermes parser の streaming `}` 切れバグ (vllm-project/vllm Issue #19056) → non-streaming 推奨
  - DABench Phase 1 の評価対象は **A-board ~60 + B-board ~320 = ~380 タスク**。public 50 で +Δ 出ても LB transfer は割引で見積もる (gap 0.236 が一定とは限らない)

  ### 詳細根拠

  メモリーの `project_qwen3_paper_hints.md` + `project_lb_gap.md` + `project_harness_principles_vs_qwen.md` 参照。論文出典: XiYan-SQL 2411.08599, CHASE-SQL 2410.01943, RSL-SQL 2411.00073, ReFoRCE 2502.00675, SQL-of-Thought 2509.00581, CHESS 2405.16755, DS-STAR 2509.21825, Universal Self-Consistency 2311.17311, Qwen3 paper 2505.09388.


- [URGENT] **🧠 Qwen3 paper-derived 未活用レバー 3 軸 (2026-05-05 user 指示)**

  公式 Qwen3 technical report (arxiv 2505.09388) によると、Qwen3-30B-A3B (= 実 serve `qwen3.5-35b-a3b` の最も近い公式モデル) の **公式評価条件と現状実装にギャップ**がある。これらは「frontier 模倣の削り」ではなく **公式設計に揃える方向**の実験で、ROI 期待値が高い。

  **背景データ (paper Table 15):**
  - BFCL v3 (function calling): thinking ON = **69.1**, non-thinking = 58.6 (**+10.5pt 差**)
  - 公式評価は **FC format + YaRN 64K context** で multi-turn 評価
  - thinking budget scaling (Fig 2): thinking トークン増 → 精度 smooth に上昇

  **軸 P1 [reason] `/think` flag 明示** (低リスク高 ROI):
  - 仮説: qwen3 は hybrid thinking model で `/think` `/no_think` フラグ制御。デフォルト thinking ON だが、multi-turn で誤って `/no_think` 状態に逸れるリスク。
  - 実装: system prompt に `Always think step by step before tool calls. (Qwen3 thinking mode)` を明示、各 user turn の末尾に `/think` を append。または OpenAI-compat API で `extra_body={"chat_template_kwargs":{"enable_thinking":true}}` を送る。
  - 期待: thinking ON が確実化されれば BFCL レベル (+10pt) の効果あり得る
  - 実装コスト: 軽 (system prompt + runner の API call 1 箇所)
  - ベース: exp_081_union_t0

  **軸 P2 [thinking] thinking budget 拡張 (max_tokens 32K)** (中リスク中 ROI):
  - 仮説: paper Fig 2 で thinking 長 = 精度 が smooth scaling。現状 vLLM の max_tokens は推測 8K 前後で、難タスク (task_344/396/418/379 等の巨大 doc) で thinking が打ち切られている可能性。
  - 実装: vLLM `--max-model-len 32768` 確認 (既に 32K 以上なら不要) + agent 側 `max_tokens` の per-call cap を 8K → 24K に引き上げ。
  - 期待: 巨大 doc タスク 4 件 (task_344/396/418/379) のうち 1-2 件が解ける可能性。task_22 の cold-start 系も thinking 余地増で安定化期待。
  - リスク: latency 増 → timeout 増の懸念。task_timeout=900s と組合せ。
  - ベース: exp_081_union_t0

  **軸 P3 [context] YaRN 64K context 化** (高リスク高 ROI):
  - 仮説: 公式 BFCL 評価は **YaRN scaling factor=4 で 64K context**。現状 32K で運用 → 公式条件と乖離。**obs_truncate (10K cap) は逆方向**で危険。
  - 実装: vLLM 起動時 `--rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'` で 64K 化、experiment 側で obs_truncate 撤去 + preamble 縮小撤去。
  - 期待: 巨大 task で context 不足を解消、先の exp_082 cursor_harness 失敗 (schema-only 5K) の逆方向検証になる。
  - リスク: vLLM 再起動が必要 (= 走行中 bench との衝突)。本番投入前に smoke 必須。
  - 注意: vLLM 側変更を伴うので **user 確認後に実施**。
  - ベース: exp_081_union_t0

  **実施順序**:
  1. P1 (`/think` 明示) ← 最小コスト、これだけ先に
  2. P2 (thinking budget 32K) ← P1 とセットでも可
  3. P3 (YaRN 64K) ← vLLM サーバー側変更必要、user 承認後

  **詳細根拠**: `.claude/projects/.../memory/project_qwen3_paper_hints.md` 参照。


## Recently consumed

- **❌ [preamble:hint+schema-graph] schema-graph FK+orphan advisory definitive regression (2026-05-07) — exp_099_schema_graph_fallback**
  - n=1 λ0.5=0.5961 (Δ=-0.067 vs single-attempt floor 0.6633、−3.08σ)
  - missing=10 (vs floor 2-3): hint LLM precompute call の latency 増 → 900s timeout 頻発
  - **context-blind dictation bias 確定**: precomputed advisory (質問を見ていない) が命令調 → agent が hint 通りに wrong join/drop → hint 発火 ~7 task で avg −0.286/task
  - hint 非発火 ~8 task: +0.075 (schema-only downgrade 副産物 — salvageable)
  - run_001 dir 削除、runs 002–004 は v4 (hint disabled, `return ""`) でアボート
  - **5 連続 [context]/[preamble:hint] regression 確定** (exp_087/088/095/097/099)
  - **[schema-graph] + [preamble:hint] 両軸廃棄確定**
  - **次 step**: `exp_104_preamble_downgrade_only` — schema-only downgrade (oversized file → column+dtype+3rows) 副産物を exp_086 base で isolated test

- **⚠️ [preamble:rich] rich profile preamble + min_steps=4 single-attempt likely_win (2026-05-07, 分析完了) — exp_101_rich_preamble**
  - n=2 mean=0.7038 ± 0.0178, CI=[0.6794, 0.7284] (Δ=+0.0405 vs exp_040 baseline 0.6633、+1.86σ)
  - **single-attempt class 内最高**: exp_096 (0.6700) Δ+0.0339、exp_097 (0.6614) Δ+0.0425、exp_092 (0.6400) Δ+0.0639
  - missing=3/run (timeout: task_11/396 両 run、task_80/173 片方) — single-attempt fallback なし
  - **task_11 catastrophic (-1.0 両 run)**: 250KB×2 JSON → "small" 閾値誤判定 → profile + raw 同時 emit → preamble overflow → 0-step trace / 900s timeout
  - **infra side-effect**: `runner.py:355` で single-attempt 失敗時 `steps=[]` 永続化 → デバッグ不能 (exp_106 で fix 必須)
  - **軸 bundle**: rich profile preamble + min_steps=4 + Rule16/17 の複合、純粋 preamble 軸効果は分離不可
  - **🚨 next steps (REVISED)**: (1) **exp_106** [CRITICAL bugfix] profile size guard + runner.py:355 trace persist; (2) **exp_107** rich profile × 3-attempt T=0.6/0.6/0.7 union → 0.74〜0.76 帯期待; (3) **exp_108** adaptive profile depth — exp_107 は exp_106 bugfix 先行必須

- **❌ [arch:harness] cursor-style schema-only + tool discovery regression (2026-05-06) — exp_096_cursor_harness_v2**
  - n=1 λ0.5=0.6700 (Δ=-0.065 vs exp_086、+0.005 vs baseline)、wext=0 ✅、missing=8
  - workers=20 contamination 解消確認: exp_082 (0.6174) → exp_096 (0.67) で +0.05 改善
  - **positive**: with_extras=0 (extras 完全削減)、task_89/86 format 認識で zero→perfect
  - **negative**: knowledge.md semantic skip (-0.06)、8 missing (single-attempt fallback なし)
  - **schema-only 軸確定**: extras 削減有効だが recall/missing は 3-attempt + fulldata 必要

- **❌ [context-single] M-Schema + single-attempt timeout regression (2026-05-06) — exp_097_mschema_single_attempt**
  - n=1 λ0.5=0.6614 (raw Δ=-0.074)、missing=7 (timeout)、wext=1
  - **真の axis effect ≈0**: timeout 除外 43-task Δ=-0.003
  - **重要発見**: exp_088 の害は M-Schema ではなく 3-attempt union × T-diversity の相互作用が dominant
  - **M-Schema 単体は無害 but no-gain** — 3-attempt + row-padding fix との combo で再評価が必要
  - single-attempt class score ≈ 0.66 確定 (exp_040/092/097 一致)
  - **次 step**: `exp_099_mschema_3attempt_clean` (Plan-first 抜き + row-padding fix bundle + 3-attempt)

- **❌ [arch:runtime] Qwen-Agent wire swap 廃棄確定 (2026-05-06) — exp_094_qwen_agent_runtime_swap**
  - n=1 λ0.5=0.7119 (Δ=-0.024 vs exp_086 mean=0.7353)、with_extras=9 (vs exp_086 avg 5)
  - **wire-level audit 確定**: `qwen_agent/llm/base.py:177-178` が call 毎に random seed inject → 3 attempts intentional diverge → extras 増
  - Mitigation 不可 (seed=None 効果なし)、monkey-patch ROI 低
  - **vendor magic 仮説の否定**: thin wrapper = seed harmful side-effect のみ
  - **[arch:runtime] framework swap 系は今後試さない** (Qwen-native fit = qwen_agent.Assistant フル乗せ換えのみ)

- **⚠️ [preamble:hint] column candidate hint n=1 likely_regression (2026-05-06) — exp_095_column_candidate_hint**
  - n=1 λ0.5=0.7086 (Δ=-0.027 vs exp_086 mean=0.7353)。replication 待ち (exp_094 vLLM 占有中)
  - **column hint 純粋効果 ≈ 0**: Δ=-0.027 の主因は row-padding バグ stochastic 表面化 (task_11/22/25)
  - hint 発火 41 task mean=0.7453 — hint が active に邪魔していない間接証拠
  - M-Schema の致命傷 (exp_088 三項罠) を回避できた ✅
  - **純粋効果測定は row-padding fix との combo (exp_098) で再評価推奨**
  - wext=7 (+2 vs exp_086 mean 5) — 軽度の extras 増、hint が間違った列推奨をしていた可能性あり

- **⚠️ [bugfix-infra] runner.py row-padding fix ノイズ範囲内 (2026-05-06) — exp_093_runner_padding_fix**
  - n=1 partial (49/50 wall-clock kill) λ0.5=0.7328 imputed (Δ=-0.0025 vs exp_086 mean=0.7353)
  - task_259 (+0.031): fix 有効。task_25 (zero 継続): wrong filter convergence が主因で fix 対象外。task_11 (-0.125): 副作用 extras +1
  - replication 失敗 (stale lock abort) → n=1 確定不能
  - **task_25 zero 仮説修正**: row-padding バグ主因 → 合成失敗 (fix 後も zero 継続)
  - **bug fix の "守りの価値" のみ確定**。独立 axis win にはならない。future combo での padding 汚染排除に貢献

- **❌ [verify-rule] non-LLM regex verifier + single-attempt 廃棄確定 (2026-05-06) — exp_092_verify_rule_single**
  - n=1 λ0.5=0.6400 (Δ=-0.095 vs exp_086、baseline 未満)
  - 二段失敗: (1) single-attempt 化 → 3-attempt union recall 喪失 (-0.06); (2) verifier regex multi-ask blind → false positive (-0.04)
  - task_249: 正解 2-col → verifier "1 col implied" hint → agent collapse → recall=0
  - **[verify-rule] non-LLM regex verifier は廃棄確定**。再挑戦は 3-attempt union base + LLM judge or multi-ask pattern 追加で別軸

- **⚠️ [tool] R5 structured error hint n=1 ノイズ範囲内 (2026-05-06) — exp_090_r5_structured_error_hint**
  - n=1 λ0.5=0.7236 (Δ=-0.012 vs exp_086 mean=0.7353)
  - hint 発火 19/50 task、実効 2 class のみ (KeyError×18, FileNotFound×3)、残り 8 class dead code
  - task_259 (-0.594) の wrong-column failure が Δ の全量 — R5 は圏外
  - **error-feedback 系統 3 連続ノイズ床 (exp_046/056/090)、軸効果 ±0.005 と確認**
  - replication ROI 低い → 廃棄推奨。dead code 削除軽量版のみ次 iter バンドル候補

- **❌ [context] R3 M-Schema preamble + Plan-first 廃棄確定 (2026-05-06) — exp_088_r3_mschema_plan_first**
  - n=2 clean mean=0.6915 ± 0.0270, CI=[0.6541, 0.7289] (Δ-0.044 vs exp_086 mean=0.7353)
  - runs: run_002=0.7106 (missing=2), run_003=0.6724 (missing=1), run_004 廃棄 (vLLM 502 outage)
  - 主因: task_11 (M-Schema × T-diversity × union 三項相互作用崩壊、両 run zero) + task_25 (`_signature_majority_merge` row-padding バグ、両 run consistent zero)
  - 副作用: with_extras 増加 (Plan-first → attempt 別 column commit → union 後列数増)
  - **M-Schema 単体 + single-attempt なら有望** (task_379/67/257 で純粋 benefit 確認)
  - **独立 fix 価値**: `runner.py:_signature_majority_merge` row-padding バグ修正 (+0.02〜+0.04)

- **❌ [aggr] R2 ⌈n/2⌉ column-signature majority vote 廃棄 (2026-05-05) — exp_087_r2_column_vote**
  - **軸: Deep research Tier 1 R2** — k=1 union → k=2 majority (1 行変更 ablation、exp_086 base)
  - **結果: ❌ regression** — n=2 mean=0.7098 ± 0.0182, CI=[0.6847, 0.7350] (Δ-0.0255 vs exp_086, Δ+0.0465 vs baseline)
  - extras 削減 (with_extras 5→2.5) は確認。しかし recall loss (-0.04) がスコア寄与で 4-10× 上回る
  - **重要な発見**: k と temperature diversity は逆相関設計パラメータ。k=1+高diversity = SOTA。k=2+低diversity = alternative
  - **[aggr] R2 k=2 majority 軸廃棄確定**。extras 削減は R3 Plan-first column commitment で攻める

- **✅ [infer] R1 公式 inference params 消化 (2026-05-05) — exp_086_r1_official_params**
  - **軸: Deep research Tier 1 R1** — T=(0.6, 0.6, 0.7) per-attempt, presence_penalty=1.0 適用 (Qwen3.5 official "precise mode")
  - **結果: ✅ 確定 win** — n=2 mean=0.7353 ± 0.0135, CI=[0.7167, 0.7540] (Δ+0.0447 vs exp_081 mean=0.6906)
  - recall (λ0.0): 0.66→0.77 (+0.11)。新規攻略 task_196/200/259。consistent zeros (10/50) は T=0.6 では救えない
  - with_extras が run_003 で 7 件 (run_002: 3 件) に増加 → R2 column voting で対策予定
  - **新リーク無しベスト確定。submission main.py + Dockerfile を exp_086 に更新済み**

- **🧪 Cursor / Claude Code / Codex 流フロンティア原則からの未試行軸 (2026-05-03) — 軸 A 完了 (2026-05-05)**
  - **軸 A: exp_082_cursor_harness** ❌ **後退確定** — n=3 mean=0.6174 ± 0.0281, CI=[0.5855, 0.6492] (Δ−0.046 vs baseline 0.6633)
    - 根本原因: schema-only preamble (~5K) が extended thinking 爆発を引き起こし missing avg 8.3/50 (通常 1-2)
    - 正の効果: task_86/89/200 が 3/3 fix (grep_file/stat_file は有効)
    - **[arch] schema-only preamble 軸は廃棄確定**。実データ値の完全除去は Qwen3.5-35b に適合しない
    - grep_file/stat_file は fulldata preamble ベースで additive 追加を次の候補とする
  - **軸 B: exp_smart_errors** — 軸 A が後退のため優先度降格。次の実験サイクルで再検討
  - **軸 C: exp_deterministic_sql** — 軸 A が後退のため優先度降格。Rule 16 が既に similar effect
  - **次の推奨**: exp_081_union_t0 + grep_file/stat_file additive (fulldata preamble 維持 + discovery tools 追加のみ)

- **✅ ♻️ 並行影響を受けた過去実験の再評価キュー — 完了 (2026-05-04)**: exp_040 clean baseline を n=3 (runs 014/015/016) で確定。mean=**0.6633** ± 0.0218 (旧 0.5622 は API 障害汚染で miss=11、棄却)。items 2-5 (exp_055/044/059/060) は exp_081 n=3 mean=0.6906 確定により不要 — clean baseline 比 Δ > +0.02 を超える候補は exp_081 のみ。**真ベスト = exp_081_union_t0**、他は全て likely_win 留まり。

- **🎯 真のリーク無しベスト (要再検証)** — ✅ **exp_068 n=3 re-replication (runs 006/007/008) + exp_081 n=3 で解決 (2026-05-04)**。
  真ベスト = **exp_081_union_t0** (n=3 mean=0.6906, std=0.0138)。exp_068 pooled mean=0.6882 は tie だが std=0.0332 で完敗。exp_040 の "1-run 0.7267" は outlier であり n=3 clean mean=0.6633。submission v1 (exp_068) は exp_081 への切替が recommended だが A-board 120s 制約での local smoke 確認が先決。

- **🤖 KIRA harness (Krafton AI) — 残 0 件、全 4 軸消化完了 (2026-05-04)** — ✅ 全項目実装・評価済み。
  - double_confirm (exp_065): ❌ revision が正答を誤答に変えた、廃棄
  - function_calling_v2 (exp_068): ⚠️ 1-run +0.010 有効 (min_steps=4 guard) だが n=3 std=0.0411 (likely_win)
  - obs_truncate (exp_071/075): ❌ 公開 50 タスクで no-op、廃棄
  - proactive_summarize (exp_069): ❌ threshold 高すぎ・missing 増加、廃棄
  ❌ 移植不可: Marker-based polling / Image read tool / tmux session 操作 (terminal 専用)

- **📚 Cursor / Anthropic ハーネス研究 — 残 0 件 (2026-05-04)** — 全 ❌ (試さない確定)。
  Sub-agent / Static→Dynamic 全面切替 / Initializer+Coder / Multi-session compaction / skill_split / dual_eval — すべて過去実験で否定済み。

- exp_068_kira_function_calling_v2 (axis: [kira] min_steps=4 runner guard — 最初の 4 step は answer 系ツールをブロック、探索強制) — ✅ **実装・評価済み**。
  ⚠️ **結果 (再評価後): 1-run λ0.5=0.7367 は +1.4σ outlier。n=3 rep2 (runs 006/007/008) mean=0.6828, std=0.0411。pooled n=7 mean=0.6882, std=0.0332。verdict=likely_win (vs exp_040 Δ+0.0249、vs exp_081 Δ−0.0024 tie)**。"新リーク無しベスト" タグ撤回。min_steps=4 early-answer ブロックは有効軸だが stochastic ロジック誤りには無力。Rule 16 prompt hit率 4/7 (57%)。

- exp_069_kira_proactive_summarize (axis: [kira] proactive context compression — context > 80% 時に過去 observation を LLM で要約) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.6398 (Δ−0.087 vs exp_040)**。要約発火 2 件のみ (threshold=0.80 高すぎ)。追加 LLM コールで missing 4→7 件に急増。max_workers=8 + timeout=600s と根本的に相性が悪い。**[kira] proactive_summarize 廃棄確定**。

- exp_070_c4_dtype_canon (axis: [output+reason] C4 dtype 正規化 — Rule 18: 数値 2dp / null→空文字 / ISO 日時) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.6367 (Δ−0.090 vs exp_068)**。API 障害 (vLLM 空レスポンス×4 + HTTP502×1 + timeout 連鎖) が missing 4→11 件の主因。Rule 18 の正味効果は評価不能。task_67 で "null/missing→空文字" を "0=missing" と誤解釈し weight=0 行除外バグ確認。Rule 18 は文言修正 ("数値 0 は 0 のまま") の上でクリーン run での再評価候補。

- exp_071_kira_obs_truncate (axis: [kira] selective observation truncation — 30KB 超過時のみ rows→output 順で切り詰め、exp_068 base) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.6283 (Δ−0.108 vs exp_068)**。obs_truncate が公開 50 タスクで完全 no-op (最大観測 13,721 bytes < 30,000 bytes 閾値)。スコア低下は min_steps=4 stochastic variance が原因 (task_67/11/196/200/379/259 後退)。**[kira] obs_truncate 低優先・閾値見直しで再評価候補**。

- exp_072_c1_col_minimize_v2 (axis: [output] C1 col-minimize Rule 18 — answer 前に question と照合し不要列削除、exp_068 base) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.7083 (Δ−0.028 vs exp_068)**。with_extras 4→2 は misleading (task_259/379 が recall=0 後退による偽陽性)。sticky extra (task_38/330) 未修正。min_steps × Rule 18 干渉 (task_86 後退確認)。Rule 18 単体 (min_steps なし) での再評価候補。

- exp_073_reason_min_steps_6 (axis: [reason] min_steps=4→6 引き上げ — 4ステップ後の誤ロジック選択タスクへ追加探索強制、exp_068 base) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.6683 (Δ−0.068 vs exp_068)**。missing 4→5 (timeout 境界タスク追加圧迫)、perfect 34→32 (min_steps=6 forced divergence)。min_steps 増加軸は廃棄確定。

- exp_075_kira_obs_truncate_10k (axis: [kira] obs_truncate 10KB — 閾値 30KB→10KB で確実発火を狙う、exp_068 base + timeout=900) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.6683 (Δ−0.040 vs exp_074、Δ−0.068 vs exp_068)**。発火 2/50 件。主後退は stochastic failures (task_292 cold-start / task_199 malformed JSON ループ)。obs_truncate 3連続後退 (exp_066/071/075) で **[kira] obs_truncate 軸廃棄確定**。

- exp_074_infra_timeout_900 (axis: [infra] timeout=600→900s — cold-start timeout タスク救済、exp_068 base) — ✅ **実装・評価済み**。
  ⚠ **結果: λ0.5=0.7083 (Δ−0.028 vs exp_068、純粋効果 +0.020)**。task_22 cold-start 救済確認 (0→1.0、187.6s完走)。stochastic net negative (task_200/379/259 後退)。**timeout=900 を全後続実験の baseline config として採用**。

- exp_065_kira_double_confirm (axis: [kira] double-confirm registry — 1st answer returns checklist, 2nd call accepted) — ✅ **実装・評価済み**。
  ❌ **結果: λ0.5=0.7033 (Δ−0.0050 vs exp_061、stable-41=0.7703 = exp_040 と同値)**。発火率 94%。revision 率 28% (13/47)。**revision が正答を誤答に変えた** (task_173: 1.00→0.00, task_194: 1.00→0.00)。改善 0 件。timeout +2 (checklist step 追加)。**[kira] double_confirm 廃棄確定**。強制再考はモデルに対してハザードが大きい。

- exp_046_tool_error_truncate (axis: [tool] Cursor context rot — traceback 5行切り詰め + error hint) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6798 (−0.047 vs exp_040)**。error hint 20 回発動のうち 16 回 (80%) が汎用メッセージ → 具体的誘導効果ゼロ。全退行は stochastic 変動。**error hint 軸は廃棄**。task_67 win は stochastic — Rule A (avg で 0/null 除外禁止) で確実化が必要。

- exp_047_skill_split (axis: [prompt] Anthropic Skills パターン — Rule 10-14 外部化 + read_skill ツール) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6917 (−0.035 vs exp_040)**。`read_skill` 呼び出し 0/50 タスク — モデルは間接的な「エラー後にスキルを読め」命令に従わない。Rules 10-14 除去により Rule 13 欠如 → missing 1→4 (+3)。**skill_split / tool-mediated rule delivery 軸は廃棄確定**。

- exp_048_inspect_table (axis: [tool] Cursor inspect_table — CSV/SQLite 共通 schema+dtypes+sample ワンコール) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6083 (−0.118 vs exp_040)**。inspect_table 14 タスク呼び出し / 正効果 0 タスク。null% 情報がモデルの探索を早期打ち切りさせる逆効果 (task_243)。通常パスから外れる経路を誘発 (task_350)。**schema 情報提示ツール追加の [tool] 軸は廃棄**。exp_049 は [output] Rule 追加で軸転換済み。

- exp_049_targeted_rules (axis: [output] Rule 15+16 追加 — avg 0/null 除外禁止 + 双方向ペア unique pair) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6817 (−0.045 vs exp_040)**。Rule A (task_67 avg/0 fix) deterministic 成功 — **Rule A (Rule 15) は全 exp に継承必須**。Rule B (task_196 bidir) 文言不十分で未解決。6 件 stochastic regression。exp_050 は [reason] Rule 1/6 強化で軸転換済み。

- exp_050_format_precommit (axis: [reason] A1 format pre-commitment — Rule 1 COMMITMENT + Rule 6 drop extra cols) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6483 (−0.078 vs exp_040)**。Rule 6 "drop extra" は task_408 に有効だが Rule 1 "COMMITMENT" 語が task_330 の正当な column_count 修正を阻害 → extra_cols。stochastic regression +6 (task_173/199/259/420 等)。Rule A 未搭載のため task_67 win は stochastic。**`[reason] A1` COMMITMENT 語は廃棄。Rule 6 drop のみ次世代に継承候補**。exp_051 は [critique] self-check で軸転換済み。

- exp_051_pre_answer_verify (axis: [critique] G1 Reflexion 軽量版 — Rule 17 answer前 3行 question-alignment check + Rule 15 keeper) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6883 (−0.038 vs exp_040)**。Rule 17 は 14 タスクでトリガー / 改善 0 件 / regression 2 件 (task_259/379) / missing +3 (latency timeout)。初期計画が間違っているケースで "Aligned: YES" と誤判断する根本欠陥。**Rule 17 および [critique] 軸 G1 Reflexion 軽量版は廃棄**。Rule 15 は副作用なし継続。exp_052 は [schema] filetype_labels で軸転換済み。

- exp_052_filetype_labels (axis: [schema] B2 revised — preamble workspace overview に [CSV]/[SQLite]/[JSON]/[DOC] ラベル付加 + Rule 15 keeper) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6283 (−0.0983 vs exp_040)**。filetype labels 改善 0 タスク。**Rule 15 非決定論的確定**: task_67 = 78.51 ❌ (Rule 15 ありでも wrong)。exp_046〜051 の 6 連続正答は stochastic ストリークだった。stochastic regressions +5。**[schema] B2 filetype labels 廃棄。Rule 15 確実化には code utility が必要**。

- exp_053_rule6_drop_extra (axis: [output] Rule 6 強化 — extra columns を answer 前に drop する指示 + Rule 15 keeper) — ✅ **実装済み**。
  ⚠️ **結果: λ0.5=0.5917 (API障害汚染、補正後推定 0.6717、Δ −0.055 vs exp_040)**。7 タスクが 600s タイムアウト (preamble_metadata=None)。**Rule 6 drop-extra は task_408 で確定効果** — 中間計算列 drop。task_330 改善（stochastic か Rule 効果か未確定）。クリーン再実行が必要。**Rule 6 drop-extra 軸は有効、exp_055 で継続候補**。

- exp_054_safe_mean_helper (axis: [infra] code utility — helpers.py に safe_mean/safe_sum + preamble 導線 + sys.path fix) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6867 (−0.0400 vs exp_040)**。safe_mean() 呼び出し 1/50 件のみ。model は preamble hint を無視して自前 df.mean() を書き続けた。task_67 win は stochastic (safe_mean 未使用)。API タイムアウト 4 件汚染。クリーンスコア: Δ−0.0435。**[infra] code utility injection 軸は廃棄確定**。

- exp_055_rule16_bidir (axis: [reason] Rule 16 強化 — undirected edge 双方向格納パターン説明 + MIN/MAX dedup SQL/Python 例 + Rule 15 keeper) — ✅ **実装済み**。
  ⚠️ **結果: λ0.5=0.7083 (−0.0183 vs exp_040)**。Rule 16 bidir は task_196 に対して **deterministic 効果確認** (0.0→1.0) → **全後続実験に継承必須**。net 後退の原因は stochastic regression 3件 + API timeout 3件。クリーンスコア (47 tasks): 0.7535 vs exp040=0.7518 → Δ+0.0018 (実質フラット)。**Rule 16 bidir は採用確定、task_196 を deterministic に修正**。

- exp_056_error_explain (axis: [critique] Rule 18 — ERROR ANALYSIS:/NEXT ACTION: structured thought 義務化 + 連続 2 エラーでアプローチ変更強制 + Rule 15 keeper) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6883 (−0.0383 vs exp_040)**。Rule 18 は 24 errors 中 1件 (4.2%) しか適用されず — model は format 指示を無視。`answer_from_sql` 強推奨ルールが task_292 で新回帰 (URL → constructorId+points)。クリーンスコア: 0.7323 vs exp040=0.7730 (Δ−0.0408)。**[critique] G2 Rule 18 軸廃棄確定**。

- exp_057_combo_r6_r16 (axis: [output] Rule 6 drop-extra + Rule 15/16 — exp_053/055 両確定 fix を exp_040 ベースに合体) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6483 (Δ−0.0783 vs exp_040) — API タイムアウト 6件 (task_180/352/379/396/418/80) で重度汚染**。クリーンスコア (44 tasks): 0.7367 vs exp040=0.7860 (Δ−0.0492)。Rule 6/16 の効果検証不可。**⚠️ prompt length +1088 chars が task_180 の 32s→600s 悪化の原因可能性** — Rule 追加戦略は timeout を悪化させる副作用がある。廃棄。

- exp_058_robust_sql_guard (axis: [robust] Rule 19 — answer_from_sql guard: text/URL final col 時は answer_from_python 使用強制 + Rule 6/15/16 継承) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.6517 (Δ−0.0750 vs exp_040)**。クリーンスコア (48 tasks): 0.6788 vs exp040=0.7569 (Δ−0.0781)、実験群で最低水準。Rule 19 は task_292 に 0 効果 (exp_040 でもすでに perfect)。timeout は外部 API インフラ問題と確認 (prompt 最長でも最少 timeout)。**[robust] Rule 19 軸廃棄確定。Rule 6/15 廃棄確定**。

- exp_059_dual_eval (axis: [multi] eval 2段階 loop — 1st answer REJECT 時に max +5 steps retry; exp_040 ベース; prompt 変更なし) — ✅ **実装済み**。
  ❌ **結果: λ0.5=0.7198 (Δ−0.0069 vs exp_040)**。REJECT 発火 = 0件 (47/50 tasks eval 呼び出し)。Qwen3.5 MoE はゴールドなしで ACCEPT/REJECT 判定不能。task_199 が 600s タイムアウト (dual eval 追加 LLM 呼び出しが時間予算消費)。**[multi] dual eval 廃棄確定**。

- exp_061_r16_only (axis: [reason] Rule 16 bidir dedup のみ追加 — exp_040 ベース + sorted-pair MIN/MAX dedup) — ✅ **実装済み**。
  ⚠️ **結果: raw λ0.5=0.7083 (Δ−0.0183 vs exp_040)。ただし stable-41 tasks: Δ+0.0244 (0.7703→0.7947)**。Rule 16 は task_196 を deterministic に修正 (trace 確認済)。stochastic losses 5件 (task_420 new fragile 含む) が raw を押し下げ。stable tasks の差異は task_196 のみ → Rule 16 に副作用ゼロ確認。**Rule 16 は全後続実験に継承必須。exp_061 を新実験ベースラインとして採用**。

- exp_042_doc_search + adaptive follow-up (axis: [tool] D-rag — search_doc keyword grep + doc truncate) — ✅ **exp_043 + exp_045 として実装済み**。
  ❌ **exp_043 結果: λ0.5=0.6483 (−0.078 vs exp_040)**。preamble 全体縮小が通常タスクの情報欠落を引き起こした。
  ❌ **exp_045 結果: λ0.5=0.6083 (−0.118 vs exp_040)**。adaptive 縮小に改善しても巨大 doc 4 タスク 0/4 解決。max_workers=8 が task_75/173/420 に CPU 競合タイムアウト。
  **軸凍結確定**: search_doc はキーワード検索のみ有効。大規模集計・閾値推定には Python 全件読み込みが必要。search_doc / adaptive doc truncation はこれ以上試さない。

- exp_035_fk_map_fence_fix (axis: [infra+schema] fence fix + FK map) — λ0.5 = **0.5539** ❌ large regression (Δ −0.099 vs exp_031).
  **失敗原因 1 (FK Map が extra_cols 悪化)**: preamble FK Map が JOIN 関係を過剰明示 → total_extra_cols 31→41（+10）、λ=0.5 では exp_034 よりさらに悪化。FK Map は **廃止確定**。
  **fence fix 自体は有効**: trailing `}"` ~25→2 ✅、unclosed fence 0 ✅。総 `__error__` 87→69。fence Case 3 は exp_031 ベースへの持ち込み価値あり。
  **タイムアウト継続 (2→6)**: max_steps=48 継続が worker 飢餓の根本原因。**max_steps=32 回帰が最優先**。
  **ベスト更新なし**: exp_031 (0.6531) が引き続きリーク無しベスト。

- exp_034_no_manual_answer_validated (axis: [critique G + output] 手書き answer 撤廃 + output validation) — λ0.5 = **0.5756** ❌ large regression (Δ −0.078 vs exp_031).
  **失敗原因 (新種: f-string JSON エスケープ失敗 70 件)**: モデルが `answer_from_python` コード文字列中に f-string を書くと JSON `"` escape が壊れ `"Expecting ',' delimiter"` エラー。closing fence / trailing `}"` は解消、代わりに f-string エスケープが露出。
  **4 tasks 新規クリア**: task_173/196/379/420 が answer_from_python/duckdb 経由でクリア — answer 撤廃の恩恵。
  **タイムアウト継続 (2→5)**: max_steps=48 × execute_python ループ（task_418 が 48 ステップ全消費）。
  **ベスト更新なし**: exp_031 (0.6531) が引き続きリーク無しベスト。

- exp_033_profile_table (axis: [tool] D3 profile_table) — λ0.5 = **0.5917** ❌ large regression (Δ −0.061 vs exp_031).
  **失敗原因 1 (closing fence なし — 新種)**: モデルが ` ```json {...}` を出力するが閉じる ` ``` ` を生成しないためパースエラー 71 回発生。task_420 は 48/48 全ステップがこのエラー → missing。
  **失敗原因 2 (タイムアウト急増)**: missing 2→11 (+9)。3 件は posts.db (137MB) サンプリング遅延、残りはパースエラーループによるリソース飢餓（preamble スレッドが CPU を得られない）。
  **profile_table は 4/50 タスクのみ使用**: 探索効率向上効果はほぼゼロ。exp_032 の structured spec ベースのため closing fence バグが顕在化。
  **ベスト更新なし**: exp_031 (0.6531) が引き続きリーク無しベスト。

- exp_032_cascade_self_correct (axis: [multi] F + [critique] G hybrid) — λ0.5 = **0.6067** ❌ large regression (Δ −0.046 vs exp_031).
  **失敗原因**: JSON パースエラー 93 回（vs exp_031: ~25）。複雑な structured spec prompt がモデルの出力を肥大化させ `__error__` を量産。task_80 は 40 step 中 38 回が `__error__`。self-correction が逆効果 (task_292)。missing 2→7。
  **G 系列 (self-correction) の教訓**: validation ループ自体は正しい設計だが、prompt 複雑化によるパースエラー増加が全利得を打ち消す。exp_034 では prompt を簡素化して再試行。
  **ベスト更新なし**: exp_031 (0.6531) が引き続きリーク無しベスト。

- exp_031_deterministic_cascade (axis: [multi] F deterministic-cascade) — λ0.5 = **0.6531** ✅ 新リーク無しベスト (+0.010 vs exp_030).
  `preamble.py` 全面書き換え。トークン上限 8k→150k。CSV全文/SQLite100行/JSON全文/doc全文。
  `_budget_cut()` で優先削減（json > csv > doc > sqlite > knowledge_full）。
  wext +4 が課題（大 context が余分列バイアス）→ exp_032 の structured spec で修正狙い。

- exp_030_leak_free_baseline (axis: clean baseline) — λ0.5 = **0.6431** ✅ リーク無しベース確立.
  exp_029 から FIXED_EXAMPLES（student_club priming）を完全削除。Rule 11 (schema-first) 維持。
  **新 exp のベースとして確立**（exp_031 以降はこれから派生）。


- exp_023_glossary_preamble (axis: schema B3) — λ0.5 = **0.6837** ✅ 新ベスト (+0.0085 vs exp_017).
  `preamble.py` のみ変更: `_extract_glossary()` + `_build_glossary_section()` で knowledge.md から `- **term**: def` パターンを抽出、800 chars truncate して Workspace overview ↔ File details 間に挿入。
  perfect +2（task_196/330 救済）、zero_recall −2、ただし missing +3（task_352 が doc/budget.md を JSON と誤認し 32 ステップ消尽して退行）。
  **B3 axis: 有効（新ベスト）だが task_352 退行要因の修正が次課題**。次候補: I3 BoN GenSelect / B1 schema-first。

- exp_022_cascade_v3 (axis: multi/cascade F revival) — λ0.5 = 0.3117, large regression −0.3635 vs exp_017.
  Fix #1〜5 全適用（ANSWER_MAX_STEPS=12 / soft warning / DuckDB cheatsheet 5パターン / self-correction / filter-back）したが、Phase 2 の唯一ツール DuckDB が失敗した際の退路（answer_from_python / answer）がないため EXTRA_COLS 17 件 + MISSING 5 件が発生。zero_recall +18（11→29）。
  **F 系列 cascade は設計欠陥確定。再試行不要。** シングルエージェント (exp_017 系列) に回帰し B3/I3 を次候補とする。

- exp_021_cascade_v3 が実際に実装された名称は exp_022_cascade_v3（ワークフローが pick 時に PRIORITIES.md の実験名を exp_021 → exp_022 に変更して割り当てた）。

- exp_019_column_count_coercion (axis: tool D6) — λ0.5 = 0.6087, regression −0.0665 vs exp_017.
  agent thought の column_count を registry で強制 truncate → zero_recall +5 件で大幅後退。
  agent の計画ミス時に truncate が損失を大幅に上回る。D6 設計欠陥確定。

- exp_018_fix_select_star (axis: output C2 改良) — λ0.5 = 0.6620, regression −0.0132 vs exp_017.
  `answer` 手書き禁止（Rule 10 追加 + soft warning）を試みたが `answer` 呼び出し回数は変化なし（33 回）。
  zero_recall の根本原因は wrong_value（計算誤り 7 件）+ wrong_rows（行数誤り 5 件）で、`answer` 起因は 0 件だった。
  task_352 で prompt 制約が過剰探索を誘発して退行。D 系列の直接制約アプローチは ROI 低い。
- exp_015_cascade_json_fix (axis: multi/cascade F) — λ0.5 = 0.3320, large regression −0.3390 vs exp_012.
  **3 つの設計バグ**が同時に炸裂: ①ANSWER_MAX_STEPS=4 不足（DuckDB エラー自己修正に不十分）、②Shape mismatch guard ok=False がリトライループ → missing +14、③DuckDB JSON `unnest` 構文の未教示（Binder Error 10 件）。
  cascade アーキテクチャ自体の評価はまだできていない（task_200 の 1 件のみ改善確認）。
  次サイクルで cascade 続けるなら ①ANSWER_MAX_STEPS 8〜12、②Shape guard soft warning（ok=True + 警告文）、③DuckDB JSON チートシート修正 の 3 点を同時適用すること。
- exp_013_sample_rich_preamble (axis: schema B2) — λ0.5 = 0.6400, regression −0.031 vs exp_012.
  value_sample ブロックがファイル種別ラベルなしで JSON/SQLite を混同させ task_11/269 でクエリ失敗。
  extra_column = 0 達成（λ0.0=λ0.5=λ1.0）はポジティブ。B 系列再挑戦には種別ラベルが必須。
- exp_012_tool_tolerance (axis: tool) — λ0.5 = **0.6710** ✅ 新ベスト（+0.020 vs exp_007）。
  `action_input` 文字列→`{"code": str}` 自動変換で `__error__` −78%（204→44）。
  missing 6→4、perfect 32→33。残課題: wrong_value/rows 各4件、JSON特殊文字(task_344)。
- exp_011_spec_driven (axis: output+tool) — λ0.5 = 0.5270, regression −0.124 vs exp_007.
  JSON 崩壊ループが主因（`action_input` 文字列渡しで `__error__` +89%）。
  計画フェーズ自体は有効だが D1（tool input tolerance）を先に解決しないと機能しない。
  再評価は exp_012 完了後。
- exp_010_verify_before_execute (axis: reason) — λ0.5 = 0.4820, regression −0.169
  vs exp_007. Negative result; 系列 A (reason) は当面凍結。

