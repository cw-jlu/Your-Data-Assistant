# Backlog — KDD Cup 2026 DataAgent-Bench

実験アイデアの蓄積。新しい発見はこのファイルにまず溜め、採用したものを `src/experiments/exp_NNN_*/` に落とす。

公式: <https://dataagent.top> / <https://dataagent.top/rules>
スコア式: `Score = Recall − λ · (Extra Columns / Predicted Columns)`、各タスクで 0 にクランプ、タスク間で平均。

---

## 中長期 (大改造、別軸検討)

- **Qwen-Agent ベースに乗せ換え** (2026-05-05 user 言及、Tier 2 候補)
  Qwen 公式が agent 構築に [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) を推奨。MCP 対応・tool 統合済・公式 chat template に最適化済 (`/strip thinking content from history` などを Jinja2 で自動処理)。我々の自作 ReAct + 自作 union を Qwen-Agent ベースに移植すると公式設計と整合し、R1/R4/R5 軸を Qwen-Agent 内蔵機能で代替可能。
  - メリット: 公式 best practice 取り込み、tool calling フォーマット最適化、`qwen3_coder` parser との整合
  - コスト: 既存 ReAct loop / 3-attempt union / 既存 tool セット (grep_file, execute_python, etc.) を全部書き換え (= 大工事)
  - ROI 判定: R1〜R5 を個別実装した結果と比較してから判断。短期は個別実装で進め、Tier 1 効果が頭打ちになった段階で移行検討。

- **[schema-graph] SchemaGraphSQL: Heterogeneous Schema Graph + MinHash value_overlap** (2026-05-04 外部アドバイス採用、Tier 1B 候補)
  論文ベース: **SchemaGraphSQL (arXiv 2505.18363, May 2025)** — BIRD MiniDev recall 90.56%, F1 91.68%、training-free。CSV/JSON/SQLite の table×column を node、宣言済 FK と **MinHash で検出した column 値集合の Jaccard 類似度** を edge とする graph を preamble 段で構築。datasketch (CPU-only、軽量) で全 column ペアの Jaccard を近似し、閾値 (例 ≥0.5) を超えたものを `value_overlap` edge として preamble に注入する。
  - **論文の元構造**: `[FK で graph 構築] → [LLM 1-call で src/dst 抽出] → [Dijkstra/k-shortest path] → [prompt に inject]`。我々は src/dst 抽出を ReAct 内 1 step に統合できる。
  - **DABench 特化の優位性**: 公式に declared FK のない CSV ファイル間の **暗黙の join 関係** を統計的に発見できる。これは現行の `inspect_sqlite_schema` / Glossary / sample_rich preamble がカバーしていない領域。**SchemaGraphSQL 原論文も FK 前提で DABench は CSV 間 FK が暗黙、value_overlap edge を自前で追加する必要がある** (= 我々の独自部分)。
  - **過去の失敗との切り分け**:
    - `exp_088 R3 M-Schema` 失敗 (-0.044): あれは「DDL 風 preamble に置き換える」純粋構造変更で、新情報ゼロ → 同じ情報を重い形で入れただけ。本案は **新情報 (暗黙 FK)** を **既存 preamble に additive で追加** する点が異なる。
    - `exp_035 FK Map` 失敗 (-0.099): 当時は schema linking 単独で **column scope を絞らせる prompt** を組合わせていたため extra_cols バイアスが悪化。本案は graph を「hint」として提示し scope を絞らせない。
  - **段階実装**:
    1. smoke: top-3 join paths と orphan columns hint だけを preamble 末尾に追加 (~10 行)、3-attempt union baseline からの delta 観測
    2. 効くなら subgraph injection (top-k path 全体) に拡張
    3. さらに効くなら CSV-CSV value_overlap edge の Jaccard 閾値で ablation (0.3/0.5/0.7)
    4. さらに src/dst LLM 抽出 → k-shortest path 注入 (= SchemaGraphSQL full)
  - **コスト**: ~300 行 (networkx + datasketch 依存追加)、smoke までは 1 日。bench 1run。
  - **期待効果**: +0.01〜0.03 (SchemaGraphSQL 単独)、value_overlap 追加で +0.02〜0.05 上乗せ余地。
  - **リスク / 失敗条件**: (a) MinHash の偽 positive が多すぎて agent が誤った join を試す、(b) hint が長くて context を圧迫、(c) 既に grep_file / execute_python で十分カバーされている場合は重複投資。

- **[schema-bidir] RSL-SQL bidirectional schema linking** (2026-05-04 外部アドバイス採用、Tier 1 候補)
  論文ベース: **RSL-SQL (arXiv 2411.00073, Oct 2024)** — BIRD dev strict recall **94%**, **column reduction 83%**, EX 67.21%。`[forward: question→tables→columns]` と `[backward: question→preliminary SQL→schema]` を **両方並行に走らせて union → contextual augmentation → binary mode voting** (= forward と backward の合意を取る)。
  - **DABench への直接効果**: column reduction 83% は **DABench の λ penalty (extra_cols 削減) に直結**。zero_recall を増やさずに wext を削れる構造。
  - **過去の失敗との切り分け**:
    - `exp_087 k=2 majority voting` 失敗 (-0.026): あれは **3-attempt 全部の最終列リストの k=2 majority** で、attempt の framing が同質なため voting 効果が薄かった。本案の binary mode voting は **forward / backward 2 つの異なる framing** の合意を取るので情報源が独立、より sound。
    - `exp_036 explicit SELECT cols` 成功 (+0.030) との相補: あれは output 段の制約。bidirectional は preamble 段の制約 → orthogonal、combo 可能。
  - **段階実装**:
    1. forward only でまず baseline + question→relevant_cols 1 LLM call、preamble に「relevant_cols ヒント」として追加。これだけで効くか smoke。
    2. 効けば backward (= question から擬似 SQL を 1 call 生成 → SQL 内 schema を抽出) を追加し、forward∪backward を preamble に。
    3. binary mode voting は **3-attempt union とは独立 layer** で実装 (= forward∩backward が high-confidence、forward△backward が low-confidence ヒント)。
  - **コスト**: ~500 行、smoke 1 日。
  - **期待効果**: +0.02〜0.04 (Qwen3.5 で discount しても、column reduction 効果は中規模 LLM ほど大きい — `arXiv 2408.07702 "Death of Schema Linking"` に「reasoning が強いほど false positive 抑制効果は減る」とあり、frontier 帯でしか不要論は成立しないという反証あり)。
  - **リスク**: (a) backward の擬似 SQL 生成が実 schema と乖離、(b) forward/backward 不一致の処理ロジックが複雑化。

- **[schema-iter] AutoLink: iterative schema expansion agent** (2026-05-04 外部アドバイス、**実験的**、Tier 2 future)
  論文ベース: **AutoLink (arXiv 2511.17190, AAAI 2026, Nov 2025)** — Bird-Dev strict recall **97.4% SOTA**, Spider 2.0-Lite recall 91.2% SOTA, token consumption -87.7% vs 2 位。コード公開: github.com/wzy416/AutoLink。`[full schema 投げない] → [agent loop で iterative に schema 探索 + 拡張]` で 3000+ columns でも recall 維持。
  - **DABench fit**: ✅ agent-driven (既存 ReAct と親和)、✅ training-free、✅ extreme split (>128K) に強い (DABench の context 圧迫タスク向き)、✅ token 効率良い。
  - **棄却ではなく後回しの理由**:
    - コードが新しすぎ (Nov 2025)、stability 未検証
    - 元論文は **multi-agent 設計** で、3B active MoE での安定性に懸念
    - 単一 agent loop に簡略化可能だが、簡略化版の効果は未測定
    - 我々の harness と機能重複 (= 既に ReAct が iterative)、純増効果が読みにくい
  - **再評価条件**: SchemaGraphSQL + RSL-SQL bidirectional を実装後、まだ +0.05 ヘッドルームがある場合に試す。実装は 2-3 日大工事。

- **[verify-rule] Non-LLM rule-based plan-final consistency check** (2026-05-04 外部アドバイス採用、Tier 2 候補)
  agent の plan に書かれた予定列と、最終 answer の列の Jaccard を **コードで** 計算し、不整合 (Jaccard < 0.5 等) なら 1 回だけ verifier prompt を発火する。
  - **過去の失敗との切り分け**:
    - `exp_051/056` LLM Reflexion 失敗: あれは LLM 自身に答えを批評させたため、(a) 自己肯定バイアスで誤答を通す、(b) 批評ステップで step 予算消費の二重ペナルティ。
    - `exp_092 [verify-rule] regex` 失敗 (multi-ask blind): あれは **single-attempt + multi-ask** 構造で attempt 多様性を失ったのが主因。本案は **3-attempt union を維持したまま** rule check を最終 layer に薄く挟む。
  - **設計**: agent の plan 抽出 → final answer の列名抽出 → Python の set Jaccard → 閾値外なら verifier prompt (=「plan で X を予定したが final に Y がある、どちらが正しい?」) を 1 turn だけ追加。
  - **コスト**: ~100 行、smoke 1 日。
  - **期待効果**: +0.01〜0.03 (= zero_recall 数件を recall に転化)
  - **リスク**: (a) plan 抽出 regex が脆弱、(b) verifier prompt が 1 turn 加わるため step 予算圧迫。

---

## 次の実験キュー (axis 分散版)

> ⚠️ **軸重複警告 (2026-05-02 更新)**: 直近 3 イテレーション: `[infra+schema: exp_035 fence fix + FK map]` → `[output: exp_036 explicit SELECT cols]` → `[robust: exp_037 lenient JSON]`。
> exp_033（D3 profile_table）: λ0.5 = **0.5917** −0.061 ❌。exp_034（answer_from_* のみ + output validation）: λ0.5 = **0.5756** −0.078 ❌。exp_035（fence fix + FK map）: λ0.5 = **0.5539** −0.099 ❌ worst yet。
> **exp_036 新ベスト確定**: λ0.5 = **0.6833** +0.030 ✅ リーク無しベスト更新！total_extra_cols 31→18（−42%）、zero_recall 14→9、λ0.0=0.700。Rule 12（先行列計画 + SELECT 明示）が extra_cols 削減に直接効果。trailing_brace エラー 0→9（unclosed fence Case 3 欠落）が残課題 → exp_037 で修正。
> **next の優先候補**: exp_037 (lenient JSON) — exp_036 ベース + unclosed fence Case 3 + trailing junk trimmer 改良。trailing_brace 9→0 を狙う。
>
> **Negative result (exp_035)**: fence fix + FK map は **−0.099 の最大後退**。FK Map が extra_cols バイアスを悪化。fence Case 3 単体の寄与は正だが FK Map が上回った。max_steps=48 が 6 タスクをタイムアウトに追い込んだ。
> **Negative result (exp_034)**: 手書き answer 撤廃 + output validation は **−0.078 の大幅後退**。f-string エスケープ失敗 70 回 + timeout +3。answer_from_python が主力である限りエスケープ問題は継続する。4 tasks 新規クリア（+4）は価値あり — answer 撤廃コンセプト自体は正しい。
> **Negative result (exp_033)**: profile_table ツール追加は **−0.061 の大幅後退**。closing fence バグ 71 回 + timeout +9。D3 ツール追加は parse バグ解消後に再評価すること。
> **Negative result (exp_032)**: structured spec prompt + 診断的 retry は **−0.046 の大幅後退**。JSON パースエラーが +68 ステップ、missing +5。G 系列 self-correction を retry するなら prompt 簡素化が必須。
> **Positive result (exp_036)**: Rule 12（先行列計画 + SELECT 明示 + SELECT * 禁止）は **+0.030 有効 — 新リーク無しベスト**。total_extra_cols 31→18（−42%）が直接効果。zero_recall 14→9（−5）、λ0.0=0.700。残課題: unclosed fence Case 3 欠落（trailing_brace 9 件再発）・task_408 型 SQL 早期丸め・task_259 型 wrong-column 選択。
> **Positive result (exp_031)**: 決定論的 Python preamble（150k tokens 全データ流し込み）は **+0.010 有効**。pure single-agent に比べ missing 減少・wrong_value 改善。ただし wext +4（extra_columns 増加）が新課題。大きな context が余分列バイアスを誘発する模様。
> **Positive result (exp_030)**: Rule 11 (schema-first) は汚染なし clean baseline として確立。リーク無し exp のベースとして機能。
>
> **Negative result (exp_037 lenient JSON)**: unclosed fence Case 3 修正は **−0.112 の大幅後退**（0.5717）。DOTALL 貪欲マッチが正常 JSON を破壊。**fence Case 3 の再実装は非貪欲マッチ必須**。trailing junk trimmer 強化も単独では逆効果。
> **Negative result (exp_038 function_calling)**: OpenAI tool-calling プロトコルへの全面移行は **−0.107 の大幅後退**（0.5764）。思考ブロックを tool 引数に埋め込めず Rule 12 列計画が機能しない。extra_cols 再増。**function calling 軸は凍結**。
> **Negative result (exp_039 answer_validation)**: exp_037 ベース継承 + validation で **−0.272 の最悪後退**（0.4117、missing=22）。exp_037 fence バグを引き継いだ連鎖障害。
> **Mixed result (exp_039 preamble_sqlite_cap)**: SQLite 10MB cap は実装正しいが exp_038 ベース継承のため exp_036 比 −0.092。**cap 自体は exp_036 直系で再検証が必要**。
> **Mixed result (exp_041 inline_spec)**: inline spec ブロックは **task_408 の ROUND precision を解決（実証済み）** だが exp_038 ベース継承のタイムアウト増（missing +3）が利得を上回り −0.056。**exp_036 直系 + inline spec のみ追加**で WIN を保持できる可能性が高い — 高優先。
> **Positive result (exp_040 selfdbg_fence)**: fence Case 3 (非DOTALL) + Rule 13 self-debug + Rule 14 ROUND outermost の 3 合体は **+0.043 有効 — 新リーク無しベスト (0.7267)**。missing 6→1 (−5)、perfect 32→35 (+3)。task_408/420/259/199/330 の 5 WIN。残課題: task_344/396/418/379 の巨大 doc truncate (→ exp_043 search_doc ツールで対策)、task_196/67 の計算ロジック regression (→ Rule 追加で対策可能)。**新 exp のベースは exp_040 を使うこと**。
> **Negative result (exp_042 inline_spec_clean)**: exp_036 直系 + inline spec は **exp_040 比 −0.075 の後退（0.6517）**。Rule 13 self-debug 欠如が致命的 — task_420 で malformed JSON error 後に 13 step 迷走。spec anchoring 副作用 (task_173 探索打ち切り)・extra_cols 増加 (task_259/287) も追加コスト。**inline spec は Rule 13 と組み合わせないと ROI がマイナス。exp_040 ベース + inline spec の複合実験が正しいアプローチ**。
> **Negative result (exp_043 doc_search)**: exp_040 ベース + search_doc ツール + preamble doc 5K 縮小は **exp_040 比 −0.078 の後退（0.6483）**。巨大 doc 4 タスク (task_344/396/418/379) は 0/4 解決。**最大の問題は preamble 全体縮小 (30K→5K) が通常タスクの情報欠落を引き起こした**こと (task_173/199/259/292 退行)。search_doc ツール自体は動作するが keyword lookup 以上 (閾値判定・大規模集計) には不十分。task-adaptive preamble 縮小 (巨大ファイルのみ縮小) が必要。**preamble 全体縮小は禁止**。
> **Negative result (exp_044 inline_spec_v2)**: exp_040 ベース + inline `## Spec` ブロックは **exp_040 比 −0.020 の後退（0.7067）**。WINS: task_200/67 (+2)。LOSSES: task_194 (answer_columns 3 列 → gold 1 列)、task_199 (行数 12 → gold 6)、task_173/180 (max_workers=8 CPU 競合タイムアウト)。**max_workers=8 が task_173/180 で CPU 競合を起こす** — exp_040 は max_workers=4 でこの問題を回避していた。inline spec 再試は **max_workers=4** + **task_194 対策 Rule (answer_columns は最終列のみ、JOIN 中間列は含めない)** とセットが必須。

> **Negative result (exp_045 adaptive_doc_search)**: exp_040 ベース + adaptive doc truncation (>30KB → 5KB) + search_doc ツールは **exp_040 比 −0.118 の大幅後退（0.6083）**。巨大 doc 4 タスク (task_344/396/418/379) は **0/4 解決**。仮説は完全否定。主因 1: max_workers=8 が task_75/173/420 に CPU 競合タイムアウトを引き起こし、task_75 (0 steps)/task_420 (2 steps で self-debug フェーズに到達できず) など致命的回帰。主因 2: search_doc はキーワード検索のみ有効 — 大規模集計 (task_396: 750件)・閾値推定 (task_418) には不十分。巨大 doc タスクには Python での全件読み込み + 集計が必要。**search_doc / adaptive doc truncation 軸は凍結**。副発見: task_67 で avg 計算時に weight=0 を誤除外しない Rule が必要。

> **Negative result (exp_046 tool_error_truncate)**: exp_040 ベース + traceback 5 行切り詰め + error-type 別 fix hint は **exp_040 比 −0.047 の後退（0.6798）**。error hint 20 回発動のうち 16 回 (80%) が汎用メッセージ `"Try: read the error..."` → 具体的誘導効果ゼロ。全退行 (task_173/199/259/330/379) は stochastic 変動でプロンプト変更との因果関係なし。task_67 は stochastic win (NaN 除外のみ = avg 正答) — Rule 化で確実に取れる: **avg 計算で 0/null を明示指示なしに除外禁止**。**error hint 軸は廃棄**。精緻化するなら exception class 別分岐が必要だが効果は小。

> **Negative result (exp_047 skill_split)**: **skill_split アイデアは廃棄確定**。Qwen3.5 MoE (3B active params) は `read_skill` を 0/50 タスクで呼び出した。Rule 13 欠如で missing が 1→4 に増加。ルールは system prompt に直接書く必要がある。

> **Negative result (exp_048 inspect_table)**: exp_040 ベース + `inspect_table` ツール (dtypes + null% + top5) + Rule 15 (use before queries) は **exp_040 比 −0.118 の大幅後退（0.6083）**。14 タスクで呼び出し / 正効果 0 タスク。null% 情報がモデルの探索を早期打ち切りさせる逆効果 (task_243: null%=91.2% → 探索放棄)。`inspect_table` は通常パスから外れる経路を誘発 (task_350)。**schema 情報提示ツールの [tool] 軸は廃棄**。

> **Mixed result (exp_049 targeted_rules)**: exp_040 ベース + Rule 15 (avg 0/null 除外禁止) + Rule 16 (bidir unique pair) は **exp_040 比 −0.045 の後退（0.6817）**。Rule A (task_67 fix) は 4/4 連続 perfect で deterministic 修正確定 — **Rule A は全 exp に引き続き継承必須**。Rule B (task_196 fix) は文言が弱く未解決 — "connected.csv は (a→b)+(b→a) 2 行格納" の具体的説明が必要。6 件の stochastic regression は Rule と無関係。task_330/408 で extra_cols 新発生 → exp_050 の Rule 1/6 強化が対策。

> **Negative result (exp_050 format_precommit)**: exp_040 ベース + Rule 1 COMMITMENT + Rule 6 "drop extra columns" は **exp_040 比 −0.078 の大幅後退（0.6483）**。Rule 6 drop は task_408 で有効（中間計算列削除）だが、Rule 1 "COMMITMENT" 語が task_330 で正当な column_count 修正を阻害 → extra_cols。stochastic regression +6 (task_173/199/259/420 等)。**Rule A 未搭載のため task_67 win は stochastic**。結論: "COMMITMENT" 語を削除し Rule 6 drop 単体を次世代に持ち込む候補。`[reason] A1` 軸は「COMMITMENT 文言有効性」が否定されたため廃棄。

> **Negative result (exp_051 pre_answer_verify)**: exp_040 ベース + Rule 15 (keeper) + Rule 17 (answer前 3行 Q/A alignment check) は **exp_040 比 −0.038 の後退（0.6883）**。Rule 17 は 14 タスクでトリガーされたが改善 0 件 / regression 2 件 (task_259/379)。missing +3 (latency 増加による timeout 増)。初期計画が間違っているケースで "Aligned: YES" と誤判断する根本的欠陥。**Rule 17 および [critique] 軸 G1 Reflexion 軽量版は廃棄**。Rule 15 (task_67 keeper) は副作用なしで継続。

> **Negative result (exp_052 filetype_labels)**: exp_040 ベース + Rule 15 + preamble `[CSV]/[SQLite]/[JSON]/[DOC]` ラベルは **exp_040 比 −0.0983 の大幅後退（0.6283）**。ラベル改善 0 タスク。**重要発見: Rule 15 が非決定論的と確定** — exp_052 で Rule 15 ありでも task_67 = 78.51 ❌（exp_046〜051 の 6 連続正答は stochastic ストリークだった）。filetype labels は中立だが preamble 複雑化のため削除推奨。**[schema] B2 filetype labels 軸は廃棄**。Rule 15 確実化には prompt ルールではなく code utility (preamble Python helper `safe_mean()` など) が必要。

> **Mixed result (exp_053 rule6_drop_extra)**: exp_040 ベース + Rule 6 softened (extra cols drop、COMMITMENT なし) + Rule 15 は **API 障害汚染で raw λ0.5=0.5917 (無効)、補正後推定 0.6717 (Δ −0.055)**。**Rule 6 drop-extra は task_408 で確定効果** — champion_ms/last_ms を drop して percentage_faster 1 列正答。task_330 も改善（stochastic か Rule 効果か未確定）。stochastic regression +4 (task_173/199/259/420)。7 タスク API 障害 missing。**クリーン再実行が必要**。Rule 6 drop-extra は有効軸として継続 — exp_055 での合体実験候補。

> **Negative result (exp_054 safe_mean_helper)**: exp_040 ベース + helpers.py (safe_mean/safe_sum) workspace 注入 + preamble 導線 + sys.path fix は **exp_040 比 −0.0400 の後退 (λ0.5=0.6867)**。safe_mean() 呼び出し 1/50 タスク — model は preamble hint を無視して自前 `df.mean()` を書き続けた。task_67 win は stochastic (safe_mean 未使用)。API タイムアウト 4 件 (task_344/396/418/80) が汚染。クリーンスコア比較: exp054=0.7464 vs exp040=0.7899 (Δ−0.0435)。**[infra] code utility injection 軸は廃棄確定** — Qwen3.5 MoE は外部 helper を使わない。task_67 対策は Rule 記述での禁止文アプローチへ転換が必要。

> **Mixed result (exp_055 rule16_bidir)**: exp_040 ベース + Rule 16 strengthened (undirected edge 2-row storage + sorted-pair dedup SQL/Python examples) は **exp_040 比 −0.0183 の後退 (λ0.5=0.7083)**。ただし **Rule 16 bidir は task_196 に対して deterministic 効果確認** (0.0→1.0、sorted-pair ロジック trace 確認済) — **全後続実験に必ず継承**。クリーンスコア (47 tasks, API timeout 3件除外): 0.7535 vs exp040=0.7518 → Δ+0.0018 (実質フラット)。net 後退の原因は stochastic regression 3件 (task_173/379/259) + API timeout 3件 (task_199/344/418)。**Rule 16 bidir は valid fix、task_196 対策として採用確定**。[reason] 軸での単独 Rule 追加は stochastic variance に埋もれる → 複合実験か別軸との組み合わせが必要。

> **Negative result (exp_056 error_explain)**: exp_040 ベース + Rule 18 (ERROR ANALYSIS:/NEXT ACTION: structured フォーマット強制 + 連続 2 エラーでアプローチ変更義務化) + Rule 15 は **exp_040 比 −0.0383 の後退 (λ0.5=0.6883)**。Rule 18 は 24 errors 中 **1 件 (4.2%) しか適用されず** — model は format 指示を無視。`answer_from_sql` 強推奨ルールが task_292 で **intermediate result の early submit を誘発**して新回帰 (URL → constructorId+points)。クリーンスコア (47 tasks): 0.7323 vs exp040=0.7730 (Δ−0.0408)。**[critique] G2 Rule 18 軸廃棄確定**。Qwen3.5 MoE は structured format 指示に従わない。`answer_from_sql` 推奨ルール (現 exp_040 以降の prompt 末尾) は non-numeric final col タスクで有害になる可能性 → 要注意。

> **Negative result (exp_057 combo_r6_r16)**: exp_040 ベース + Rule 6 drop-extra + Rule 15 + Rule 16 bidir の combo は **API タイムアウト 6 件 (task_180/352/379/396/418/80) で重度汚染、raw λ0.5=0.6483 (Δ−0.0783)**。クリーンスコア (44 tasks): 0.7367 vs exp040=0.7860 (Δ−0.0492)。**clean でも負け**。Rule 6 は「初期計画ミス」には無効 (task_330 stochastic 継続)。Rule 16 は task_196 で max_steps 超過により無効化。**⚠️ prompt length 増加 (+1088 chars) が task_180 の elapsed を 32s→600s (19x) に悪化させた可能性が高い** — Rule 追加戦略は timeout を悪化させる副作用がある。clean wins = task_67 (stochastic) 1件のみ。**[output] combo 軸は現時点では廃棄 — Rule 追加による prompt 肥大化は避けるべき**。

> **Negative result (exp_058 robust_sql_guard)**: exp_040 ベース + Rule 6+15+16+19 (answer_from_sql guard) は **raw λ0.5=0.6517 (Δ−0.0750)**、クリーンスコア (48 tasks): 0.6788 vs exp040=0.7569 (Δ−0.0781)、**実験群で最低水準**。Rule 19 は task_292 に 0 効果 (exp_040 でもすでに perfect)、prompt を +1464 chars 肥大化するのみ。**⚠️ timeout は外部 API インフラ問題と確認** — exp_058 は最長 prompt (8327 chars) だが timeout 最少 (2件)、exp_057 (7951 chars, 6件) と逆相関。Rule 追加が stochastic regressions を上回る改善を出せない構造的問題が明確。**唯一確実な改善は Rule 16 (task_196 deterministic fix) のみ** → exp_040 + Rule 16 のみが next baseline 候補。**[robust] Rule 19 軸廃棄確定**。Rule 6/15 も廃棄確定。

> **Negative result (exp_059 dual_eval)**: exp_040 ベース + agent.py 2nd LLM eval pass (REJECT → retry, max 1回) は **raw λ0.5=0.7198 (Δ−0.0069)**。**REJECT 発火 = 0 件** (47/50 tasks で eval 呼び出し)。LLM はゴールドなしで正誤判定不能 → 全 wins は stochastic。**task_199 が 600s タイムアウト** — dual eval 追加呼び出しが時間予算消費。stochastic net: +3 wins, −5 losses。**[multi] dual eval 軸廃棄確定** — 追加 LLM 呼び出しはタイムアウトリスクのみ増やす。Qwen3.5 MoE を judge として使用不可。

> **stochastic baseline noise 確定 (exp_060 = exp_040 replay)**: exp_060 は exp_040 と import path 以外完全同一、Δ−0.0183 は全て stochastic noise。**真の改善は Δ > +0.02 (λ0.5) でないと noise から識別不能**。Fragile tasks 確定: task_199/259/330/379 は推論ごとに flip → 46-task clean スコアで評価を推奨。

> **Mixed result (exp_061 r16_only)**: exp_040 ベース + Rule 16 bidir dedup のみ (+589 chars) は **raw λ0.5=0.7083 (Δ−0.0183)**。ただし **stable-41 tasks (fragile 9 + timeout 除外): Δ+0.0244** (0.7703→0.7947)。stable tasks の唯一の差異は task_196 のみ (0→1.0) — **Rule 16 は task_196 を deterministic に修正することを trace 確認済み**。raw 後退の原因は stochastic losses (task_199/259/420/379/330)。task_420 が new stochastic fragile 確定。**Rule 16 bidir は全後続実験に継承必須** (副作用ゼロ確認)。**[reason] Rule 16 のみの実験 = 採用・ベースライン更新**。

> ⚠️ **直近 4 イテレーション (exp_042/043/044/045) すべてで max_workers=8 による CPU 競合問題が観測**。exp_040 が max_workers=4 で task_173/180 を perfect に解いている事実は確定。**すべての新 exp で max_workers=4 を使うこと。max_workers=8 は絶対禁止**。

> ⚠️ **直近 5 イテレーション軸重複警告**: `[output] c1_col_minimize_v2(exp_072)` ❌廃棄 → `[reason] min_steps_6(exp_073)` ❌廃棄 → `[infra] timeout_900(exp_074)` ⚠️ (timeout 効果確認・stochastic net negative) → `[kira] obs_truncate_10k(exp_075)` ❌廃棄 (3連続後退・axis廃止)。次は `[reason] min_steps=2` (step 1 のみブロック + timeout=900 base) が最優先。`[reason]` min_steps 増加 (4以上) 廃棄確定。**[infra] timeout=900 を全後続実験の標準 config として採用**。`[kira]` proactive_summarize 廃棄、`[multi]` BoN 廃棄、`[kira]` double_confirm 廃棄、`[schema]` B4 廃棄、`[tool]` 廃棄、`[reason] A1/A2/A4` 廃棄、`[critique] G1/G2` 廃棄、`[schema] B2` 廃棄、`[output]` combo Rule 廃棄、`[robust] Rule19` 廃棄。**[multi] dual eval 廃棄確定。[multi] BoN 直列実行 廃棄確定**。Rule 6/15/18/19 廃棄確定 (Rule 18 は単体・min_steps なしで再評価候補)。**Rule 16 prompt は 4/7 run (57%) hit でありもはや "deterministic" ではない → runner-level post-process が必要**。Fragile tasks 9: task_67/199/200/259/330/379/420/344/80 → stable-41 metric を推奨評価基準とする。**[kira] obs_truncate 3連続後退で廃棄確定**。
>
> ⚠️ **直近 3 イテレーション軸重複警告 (2026-05-07 更新 v11)**: `[preamble:rich] rich_preamble(exp_101)` ⚠️ **likely_win (分析完了)** → `[preamble:hint+schema-graph] schema_graph_fallback(exp_099)` ❌ **definitive regression (Δ=-0.067, −3.08σ)** → `[context-single] mschema_single_attempt(exp_097)` ❌ **timeout dominant regression**。軸の重複なし (3 軸とも異なる preamble/context 系統)。**⚠️ 重大警告: [context]/[preamble:hint] 5 連続 regression** (exp_087/088/095/097/099) — precomputed advisory 軸は廃棄確定。**🚨 [CRITICAL] exp_106 が最優先**: exp_101 の task_11 catastrophic failure (250KB JSON preamble overflow + runner.py:355 0-step trace bug) を修正しないと exp_107 (3-attempt combo) でも同じ崩壊が起きる。**追加回避制約**: ⑯ **`[preamble:rich]` profile + raw 同時 emit (size guard なし)** — 250KB× 複数ファイルで preamble overflow → 0-step trace (task_11 catastrophic 再現確定)。exp_107 は exp_106 (profile_budget guard + runner.py:355 fix) 後にのみ実施。⑮ **`[preamble:hint]` + `[schema-graph]` + `[context]` 系 (precomputed advisory)** — context-blind dictation bias が 5 連続確認、再挑戦禁止。⑭ **`[preamble:rich]` size guard なしの単純 3-attempt 化** — exp_106 の profile_budget 修正が先決。⑬ **`[arch:harness]` schema-only + single-attempt** (exp_082/096 で 2 連続で exp_086 比 regression 確定)。⑫ **`[context-single]` M-Schema + single-attempt** (timeout リスク、3-attempt combo 必須)。⑪ **`[arch:runtime]` framework swap 系** (vendor magic 仮説否定)。⑩ ~~**`[preamble:hint]` + row-padding バグ共存の combo 再評価**~~ → exp_099 で [preamble:hint] 自体が廃棄確定、combo 再評価も取り消し。**次の `pick_next` では以下を回避すること**: ① `[reason]` Rule 単体追加 (n=3 replication で Δ > +0.02 を達成できない構造確認)、② `[kira]` 系 (全 3 軸廃棄済)、③ `[infra]` 単独 (timeout は標準化済で追加投資 ROI 低)、④ `[arch]` preamble 削減 (exp_082 で廃棄確定)、⑤ **`[aggr]` k=2 majority (exp_087 で廃棄確定)**、⑥ `[context]` **R3 M-Schema + 3-attempt union 組み合わせ** (exp_088 n=2 confirmed)、⑦ **`runner.py` row-padding バグ未修正のまま 3-attempt union 系を継続** (独立 axis `exp_093_runner_padding_fix` として先行実施済)、⑧ **`[tool]` error-feedback 系統を n=3 replication に投資** (exp_046/056/090 で 3 連続ノイズ床)、⑨ **`[verify-rule]` non-LLM regex verifier + single-attempt の再挑戦** (exp_092 で multi-ask blind + single-attempt 喪失の二段構造失敗確定、やるなら 3-attempt union base + LLM-judge verifier で別軸)、⑩ **`[preamble:hint]` + `[schema-graph]` + context-blind precomputed advisory 系** (exp_087/088/095/097/099 の 5 連続 regression で廃棄確定)。

> **❌ Negative result 確定 (exp_099_schema_graph_fallback — [preamble:hint+schema-graph] LLM-judged FK+orphan-column advisory, 2026-05-07)**: n=1 run_001 λ0.5=0.5961 (Δ=-0.067 vs single-attempt floor 0.6633、−3.08σ)。missing=10、with_extras=2、perfect=28。**context-blind dictation bias**: precomputed advisory (質問を見ていない) が命令調 → hint 発火 ~7 task で avg −0.286/task (wrong join/drop 強制)。missing=10 は hint LLM precompute call の latency 増で 900s timeout 頻発 (exp_097 missing=7 より深刻)。hint 非発火 ~8 task: +0.075 (schema-only downgrade 副産物)。run_001 dir 削除、runs 002–004 は v4 (hint disabled) でアボート。**salvageable**: preamble schema-only downgrade (oversized file → column+dtype+3rows) は `exp_104_preamble_downgrade_only` として独立実験推奨。**5 連続 [context]/[preamble:hint] regression (exp_087/088/095/097/099) で廃棄確定**。precomputed advisory の context-blind 設計欠陥が根本原因。
>
> **⚠️ likely_win 確定 (exp_101_rich_preamble — [preamble:rich] rich profile preamble + min_steps=4, 2026-05-07, 分析完了)**: n=2 mean=0.7038 ± 0.0178, CI=[0.6794, 0.7284] (Δ=+0.0405 vs exp_040 baseline 0.6633、+1.86σ)。runs: run_001=0.6913 / run_002=0.7164。missing=3/run (timeout: task_11/396 両 run、task_80/173 片方)。**single-attempt class 最高** (peer best exp_096: 0.6700、Δ=+0.0339)。**wins vs exp_040 baseline**: task_418 (+1.0)、task_196 (+0.5)、task_25 (+0.375)、task_379 (+0.375)、task_259 (+0.286)、task_22/243/287/292/420 (+0.333 each)。**catastrophic loss**: task_11 (−1.0 両 run): Examination.json (253KB) + Patient.json (250KB) を "small" 閾値で誤判定 → **profile + raw 同時 emit → preamble overflow → 0-step trace (900s/405s timeout)**。**infra side-effect**: `runner.py:355` で single-attempt 失敗時 `backbone=None` → `trace.json` に `steps=[]` 永続化 → デバッグ不能 (exp_106 bugfix 対象)。**reproducibility**: both-run-zero=10 task (task_11 含む構造的 failure)、stochastic-large-swing=7 task (T=0.6 noise)。**推奨 next axes**: (1) **[CRITICAL] exp_106**: profile size guard (total_profile_chars ≥ 8K budget で低優先 file から profile drop) + runner.py:355 per-attempt trace persist → task_11 +1.0 回収期待; (2) **exp_107**: rich profile × 3-attempt T=0.6/0.6/0.7 union (exp_086 base) → 0.74〜0.76 帯期待 (新 SOTA 候補); (3) **exp_108**: adaptive profile depth (question keywords で relevance top-N 列 full + 残り dtype+unique のみ)。⚠️ exp_107 は exp_106 (size guard) 先行なしには task_11 catastrophe を引き継ぐ。
>
> **❌ Negative result (exp_096_cursor_harness_v2 — [arch:harness] cursor-style schema-only + tool discovery, 2026-05-06)**: n=1 run_002 λ0.5=0.6700 (Δ=-0.065 vs exp_086 mean=0.7353、+0.005 vs baseline)。missing=8、wext=0。**positive**: with_extras=0 (extras 完全削減)、task_89/86 format 認識で zero→perfect (+0.04)。**negative**: task_11/173/408 knowledge.md semantic skip (-0.06)、8 missing infra brittleness (single-attempt fallback なし)。**workers=20 contamination 解消確認**: exp_082 (0.6174) → exp_096 (0.67) で +0.05 改善。ただし exp_086 R1 比では -0.065 構造的下位。**schema-only 軸確定判定**: extras 削減有効、recall/missing 確保には 3-attempt + fulldata preamble 必要、単独 single-attempt では fix 不可。
>
> **❌ Negative result (exp_097_mschema_single_attempt — [context-single] M-Schema + single-attempt, 2026-05-06)**: n=1 run_002 λ0.5=0.6614 (raw Δ=-0.074 vs exp_086 mean=0.7353)、missing=7、wext=1。**主因: timeout 7 task** (M-Schema 4K preamble で per-call latency 増加 → 900s timeout で single-attempt fallback なし → score=0)。**真の axis effect 確定**: timeout 除外 43-task Δ=-0.003 = noise 圏。**重要発見**: exp_088 (-0.044) の害は M-Schema ではなく **3-attempt union × T-diversity との attempt-divergence 増幅** が dominant — M-Schema 単体は clean single-attempt で ≈ 0。improvement: task_67/196/200/330 (+0.125 each、extras 削減)。loss: task_379 (-0.375 stochastic)、task_25 (-0.312 row-padding バグ)。**M-Schema を活かすには 3-attempt union + row-padding fix combo で再設計** (= `exp_099_mschema_3attempt_clean`)。single-attempt class の標準 score ≈ 0.66 再確認 (exp_040/092/097 一致)。
>
> **❌ Negative result 確定 (exp_094_qwen_agent_runtime_swap — [arch:runtime] Qwen-Agent wire swap, 2026-05-06)**: n=1 run_003 λ0.5=0.7119 (Δ=-0.024 vs exp_086 mean=0.7353)。missing=1, perfect=29, wext=9。**wire-level audit (commit ad7b923) で機序確定**: `qwen_agent/llm/base.py:177-178` が call 毎に `random.randint(0, 2^30)` を seed として inject → vLLM が seed-deterministic sampling → 3 attempts が意図せず diverge → union が divergent extras を保持 → with_extras=9 (vs exp_086 avg 5)、λ1.0=0.659 (vs exp_086 0.701)。`seed=None` 明示で修正試みたが no effect (vLLM が再ランダム化)。**vendor magic 仮説の否定**: qwen-agent OAI backend は openai client の薄い wrapper、seed auto-inject という harmful side-effect のみ追加。Mitigation 不可、monkey-patch ROI 低い。**[arch:runtime] framework swap 系は今後試さない**。Qwen-native fit を期待するなら qwen_agent.Assistant フル乗せ換え (BACKLOG 中長期、1-2 日大工事) のみ。
>
> **⚠️ Mixed result (exp_095_column_candidate_hint — [preamble:hint] column candidate hint, 2026-05-06)**: n=1 run_002 λ0.5=0.7086 (Δ=-0.027 vs exp_086 mean=0.7353)。missing=1, perfect=30, wext=7 (wext 増加: exp_086 mean 5 → 095: 7)。**column hint 軸自体の純粋効果 ≈ 0**: hint 発火 41 task mean=0.7453 vs 非発火 9 task mean=0.5417 の差は行列バグ脆弱性 task 構造差に起因。Δ=-0.027 の主因 = **row-padding バグ stochastic 表面化**: task_11 (-1.000、hint 未発火、row-padding バグ表面化) + task_22 (-0.250) + task_25 (-0.312)。hint 純粋効果 ≈ 0 だが M-Schema の致命傷 (exp_088 三項罠) を回避。**replication 待ち** (exp_094 vLLM 占有中)。column hint scoring 関数の精度低 (task_25 で wrong 列推奨)。**純粋効果測定には row-padding fix との combo が必要** (exp_093 fix + exp_095 hint を同時投入した exp_098 で再評価推奨)。
>
> **⚠️ Mixed result (exp_093_runner_padding_fix — [bugfix-infra] row-padding fix, 2026-05-06)**: n=1 partial (49/50 wall-clock kill) λ0.5=0.7328 imputed (Δ=-0.0025 vs exp_086 mean=0.7353)。期待 +0.02〜+0.04 は**出ず**。task_259 (+0.031): fix 有効。task_25 (依然 zero): 全 attempt "wrong filter convergence" が主因 = **row-padding バグは副因**。task_11 (-0.125): length-grouping 副作用。**exp_088 task_25 zero の仮説修正**: "row-padding バグ主因" → "row-padding バグ + 全 attempt wrong convergence の合成、fix 後も zero 継続"。replication 失敗 (run_003 stale lock abort)。bug fix の "守りの価値" (future combo での padding 汚染排除) のみ確定。axis としては win にならない。
>
> **❌ Negative result 確定 (exp_092_verify_rule_single — [verify-rule] non-LLM regex verifier + single-attempt, 2026-05-06)**: n=1 run_004 λ0.5=0.6400 (Δ=-0.095 vs exp_086 mean=0.7353、baseline exp_040 mean=0.6633 **未満**)。二段失敗: (1) **single-attempt 化 (-0.06)**: 3-attempt union の T-diversity recall を直接喪失 (task_11/25/257/259/352)。(2) **verifier regex false positive (-0.04)**: `_SINGLE_VALUE_PATTERNS` が multi-ask ("X and Y") を blind; task_249 で正解 2-col → "1 col implied" hint → agent が 1-col に collapse → recall=0。hits 5 task 中 4 が hurt。**二乗作用**: single-attempt 化でも verifier が hurt したら他 attempt による rescue 不能。教訓: non-LLM verifier は multi-ask blind 根本欠陥があり廃棄確定。verifier を試すなら 3-attempt union base + LLM judge (or multi-ask pattern 追加) で再設計。
>
> **⚠️ Negative result (exp_090_r5_structured_error_hint — R5 structured error hint, 2026-05-06)**: n=1 run_002 λ0.5=0.7236 (Δ=-0.012 vs exp_086 mean=0.7353)。hint 発火 19/50 task、実発火 2 class (KeyError×18, FileNotFound×3)、残り 8 class dead code。**task_259 (-0.594) の wrong-column failure が Δの全量を説明** — R5 は exception trigger なしには動作しないため wrong-column 系は圏外。隣接 exp_046 (-0.047) / exp_056 (+0.025 noise) と合わせ **[tool] error-feedback 系統は 3 連続ノイズ床 (軸効果 ±0.005)** と判定。replication ROI 低い。独立 action: dead code 8 class 削除 + KeyError/FileNotFound のみ残す軽量版 (+10 行) は次の iter でバンドル可能だが single-attempt class でのみ純粋効果測定可。
>
> **❌ Negative result 確定 (exp_088_r3_mschema_plan_first — R3 M-Schema preamble + Plan-first, 2026-05-06)**: n=2 clean mean=0.6915 ± 0.0270 (Δ-0.044 vs exp_086 mean=0.7353)。runs: run_002=0.7106 (missing=2), run_003=0.6724 (missing=1), run_004 廃棄 (vLLM 502 outage)。主犯 2 軸: (1) **task_11 両 run consistent zero** — M-Schema × T-diversity × union 三項相互作用で 3 attempts が独立 filter → union padding 9 cols × 75 rows → recall=0。(2) **task_25 両 run consistent zero** — `runner.py:_signature_majority_merge` の row-padding バグ: union 時に短い行集合を空文字でパディングすると multiset signature が破壊される。with_extras 増加 run_002:+5 → run_003:+9 は Plan-first × 3-attempt union 相互作用 (Plan-first は attempt 別 column commit を促す → union 後列数増)。M-Schema は task_379/67/257 で +0.375/+0.125/+0.083 の純粋 benefit 確認 → **single-attempt 化なら有望**。**[context] R3 M-Schema + Plan-first + 3-attempt union 三項相互作用廃棄確定**。独立 fix 価値: `_signature_majority_merge` row-padding バグ修正 (+0.02〜+0.04 横断効果、task_25 系 consistent fail の抜本対策)。exp_087 の "k × T-diversity 逆相関" の続編 = **"M-Schema × T-diversity × union" 三項相互作用として確定記録**。
>
> **⚠️ Negative result (exp_083 config error — workers=6 × 3-attempt = 18 vLLM streams 飽和)**: exp_083_additive_tools run_002 で workers=6 × 3-attempt = 18 同時 streams が起動し、vLLM 飽和点 (~13-15 streams) を超えたことで missing=10/50 に急増 (λ0.5=0.5736)。**3-attempt union 実験の正しい config は `max_workers: 4` (= 12 streams)** — exp_040 baseline config.yaml が明記済みだったが実装時に無視された。完了 40 tasks のみの λ0.5=0.7170 は exp_081 mean を上回るため **設計自体は健全**。exp_082 (20 streams, missing=8.3) と exp_083 (18 streams, missing=10) で同じ infra failure pattern を確認 → **18 streams 以上は禁止**。**stat_file: 0 calls (完全死蔵)** — Qwen3.5-35b では使われない。次回は削除。
>
> **⚠️ Negative result (exp_083 run_003 — vLLM 52分 502 origin outage で壊滅)**: run_003 は workers=6 config error のまま再実行したが、**vLLM endpoint 自体が 18:47Z–19:39Z (52分) 502 origin_bad_gateway で完全停止**していたため missing=44/50 (λ0.5=0.0925) に崩壊。workers=6 構造的 missing (+6) とは別の infra 事象 — **n=2 pool mean=0.3331 は判定に使用不可**。有効完了タスクのみの品質: run_003 6 タスク mean=0.7708 → 設計健全性は 2 run 一貫確認。**task_11 が run_002/003 両方で zero_recall = union k=1 dtype 副作用の構造的バグ確定** (ID 列 int/str 混在 → signature 不一致で union 6 cols 重複)。**stat_file 0 calls が 2 run 連続確認 → 廃棄確定、次回 exp で削除**。exp_084 smoke (19:15–19:47Z) も同じ 502 で全失敗 → **vLLM 復旧まで新 bench 起動は無意味**。次の action: (1) vLLM health check / 再起動 (2) workers=4 修正 + union dtype-normalize 追加 + stat_file 削除 + n=3 clean re-bench。
>
> **✅ Positive result (exp_086_r1_official_params — R1 公式 params, 2026-05-05)**: T=(0.6, 0.6, 0.7) per-attempt + presence_penalty=1.0 の公式設定は **確定 win (n=2 clean mean=0.7353, Δ+0.0447 vs exp_081 mean=0.6906)**。CI [0.7167, 0.7540] が exp_081 CI [0.675, 0.706] と非重複。recall (λ0.0) が 0.66→0.77 に +0.11 上昇。新規攻略タスク: task_196/200/259 の 3 件で +0.063。run 間で with_extras が 3→7 件と変動 → λ1.0 に差 (0.720 vs 0.682)。union の余分列増加 (R2 column voting で対策予定)。consistent zeros (task_80/89/180/396 等 10/50) は T=0.6 程度では救えない Type B 問題。**新リーク無しベスト確定。submission main.py + Dockerfile を exp_086 に更新済み**。
>
> **❌ Negative result (exp_087_r2_column_vote — R2 ⌈n/2⌉ majority vote, 2026-05-05)**: k=2 majority は extras 削減に有効 (with_extras 5→2.5) だが **recall loss (-0.04) がスコア寄与で 4-10× 上回る** (recall 1 列 -1.0/gold vs extras 1 列 +0.1-0.25/task)。n=2 mean=0.7098 (Δ=-0.0255 vs exp_086, +1.4σ, likely 実回帰)。k=1→k=2 の 1 行 ablation で temperature 多様化 (T=0.6/0.6/0.7) と相性が悪いことを実証: diversity が高いほど "1 attempt だけ正解" ケースが多発し majority 不成立 → union-only 正解が脱落。**k と temperature diversity は逆相関する設計パラメータ**: k=1+高diversity = 現 SOTA。k=2+低diversity (T=0/0/0) は alternative として将来試す価値あり (exp_080/078 と直接比較できる条件)。**[aggr] R2 k=2 majority vote 軸廃棄確定**。extras 削減を狙うなら majority ではなく R3 Plan-first column commitment が有効。
>
> **⚠️ Infra observation (exp_086 run_004 — N=3 並列 union の vLLM transient failure 脆弱性)**: run_004 で後半 13 tasks が `steps=0/ans=None` で全滅 (missing=13/50, λ0.5=0.5567)。原因: 3-attempt 並列が vLLM 一時 stall に当たると "1 attempt timeout → 3 attempts 全滅" のカスケード failure が起きる。workers=4 × 3 = 12 streams という正しい config でも vLLM transient 5xx には無力。**runner に "attempt=0/None → fallback to surviving attempts" retry logic がない**ことが根本原因。exp_082 (missing avg 8.3) や exp_068 (1 run miss=11) とも類似パターン。対策: `missing>=10 run は API汚染で invalidate` 判断を採用 (今回適用済み)。中長期: `_run_single_task_with_timeout` に transient failure → attempt 再利用 logic (exp_088_robust_union_retry 候補)。
>
> **🆕 Negative result (Rule 追加軸 — n=3 clean baseline 再確定により全候補降格 2026-05-04)**: exp_040 clean n=3 baseline = 0.6633 ± 0.0218 と確定 (旧 0.5622 は API 汚染)。これにより全候補の Δmean を clean baseline 比で再計算: exp_081 Δ+0.0273 (+1.25σ)、exp_068 Δ+0.0195、exp_067 Δ+0.0054 (n=4 更新、旧 n=3: +0.0123)、exp_061 Δ+0.0100 — **全て CI 重複 (likely_win のみ、confirmed_win = 0)**。単発 Rule 追加実験は n=3 replication でも Δ > +0.02 を確実に超えられない構造的限界が改めて確認された。
> **🆕 Negative result (exp_067 c1_col_minimize n=4 確定 — 2026-05-04)**: run_008 追加で n=4 pooled mean=0.6687 ± 0.0260、vs exp_040 clean baseline Δ+0.0054 (+0.21σ) = CI 完全重複でノイズ確定。run-to-run spread=0.058 (0.6483〜0.7067)。col-minimize Rule 17 単独軸は **廃棄確定** (n=4 で効果なし)。副発見: timeout=600 設定が task_249 cold-start timeout を誘発 (run_008 のみ; 前 3 run は 600s 内完走) — **timeout=600s は境界タスクで stochastic failure を引き起こす。全後続実験で timeout=900 必須**。
> **🆕 Negative result (12 always-zero タスクの構造的失敗)**: exp_040 n=3 cross-run 分析で 3 カテゴリの失敗が確定: Type A timeout (task_80/180/344/418) = 既存アプローチ全滅。Type B 確信誤答 (task_89/169/259/379/396) = prompt Rule では無力、answer 直前 sanity-check のみ有望。Type C schema 揺れ (task_25/163/199) = plan 段階の schema commitment 問題。Type B 対策 (exp_value_audit) は diff が小さく実装コスト低・期待 +0.02〜+0.04。
>
> **Mixed result (exp_068 kira_function_calling_v2 — 再 replication n=3 runs 006/007/008 追加分析)**: 1-run 0.7367 は +1.4σ 外れ値と確定。新 n=3 mean=0.6828, std=**0.0411** (旧 n=3 std=0.0146 の 2.8×)。pooled n=7 mean=0.6882, std=0.0332。vs exp_081_union_t0 = Δ−0.0024 (tie、std で完敗)。**"新リーク無しベスト" タグ撤回。min_steps=4 early-answer ブロックは有効だが stochastic ロジック誤りには無力**。stochastic flip 5 タスク (task_19/67/86/196/249/408) が run-to-run swing を支配。
>
> **Negative result (Rule 追加軸の構造的限界 — n=3 replication 横断確認)**: Rule 16 bidir prompt は exp_061+exp_068 合計 7 run で 4/7 (57%) しか sorted-pair dedup を適用しなかった。"deterministic fix" claim は撤回。**prompt-only Rule 追加は小 diff (~60-142 chars) で mean 0.67〜0.69 圏に集中し、単独で Δ>+0.02 を超えられない構造的限界が全 n=3 比較で確認された**。真の mean break-through は exp_081 の "推論ループアーキテクチャ変更" (diff=403) 系のみ達成中。Rule 追加軸は combo 効果期待を含めて評価すること。
>
> **Negative result (exp_069 kira_proactive_summarize)**: context_summarize_threshold=0.80 の LLM 要約圧縮は **−0.087 の大幅後退 (0.6398)**。threshold 高すぎて発火 2 件のみ (task_420/415)。要約追加 LLM コールが CPU を長時間占有し missing 4→7 件に急増。max_workers=8 + timeout=600s 設定と根本的に相性が悪い。**[kira] proactive_summarize 廃棄確定**。threshold 下げ + workers 削減 + timeout 延長を全て変更しないと機能しない — 複合インフラ変更のコスト大。
>
> **Negative result (exp_070 c4_dtype_canon)**: Rule 18 (dtype 正規化: 数値 2dp / null→空 / ISO 日時) は **−0.090 の大幅後退 (0.6367)**。ただし API 障害 (vLLM 空レスポンス×4 + HTTP502×1 + timeout 連鎖 = missing +7) が score 崩壊の主因であり Rule 18 の正味効果は評価不能。task_67 では "null/missing → 空文字" の表現を "0 = missing" と誤解釈し weight=0 行を除外するバグが確認された。Rule 18 を再試行する場合は "数値の 0 は 0 のまま (除外しない)" と文言を明示する必要あり。**[output+reason] dtype_canon Rule 18 は文言修正の上でクリーン run で再評価候補**。
>
> **Negative result (exp_071 kira_obs_truncate)**: obs_truncate (30KB cap) は **−0.108 の後退 (0.6283)**。最大実測観測サイズが 13,721 bytes (task_330/read_csv) であり、30,000 bytes 閾値を超えた観測が **公開 50 タスク中 0 件** — 完全 no-op。スコア低下は min_steps=4 の stochastic variance が原因 (task_67/11/196/200/379/259 が exp_068 比後退)。obs_truncate を機能させるには閾値を 10,000 bytes 以下に下げる必要があるが、現状の公開 50 タスクで効果を計測できない可能性が高い。**[kira] obs_truncate 軸は低優先・閾値見直しで再評価候補**。min_steps=4 の高分散が引き続き懸念。
>
> **Negative result (exp_072 c1_col_minimize_v2)**: col_minimize Rule 18 (answer 前に不要列削除) は **−0.028 の後退 (0.7083)**。with_extras 4→2 は misleading — 削減された 2 件は task_259/379 が recall=0 に後退したことによる偽陽性。sticky extra (task_38: 5 extras / task_330: 1 extra) は未修正のまま。task_379 は step 1 で column_count=3 を固定後 Rule 18 を無視し続けた「初期計画固執」問題。min_steps × Rule 18 の干渉 (task_86: min_steps=4 ブロック後の強制再探索が誤クエリを選択) が確認された。**min_steps + 他軸の組み合わせは exp_071/072 の 2/2 で後退 → 回避推奨**。Rule 18 を再試行する場合は min_steps なし (exp_067 ベース) での単体評価が必要。
>
> **Negative result (exp_073 reason_min_steps_6)**: min_steps=4→6 への引き上げは **−0.068 の大幅後退 (0.6683)**。missing 4→5 (timeout 境界タスクを追加圧迫) + perfect 34→32 (min_steps=6 ブロックが追加探索を強制 → 誤ロジック選択増加)。min_steps 増加は timeout と stochastic の両方を悪化させる。**[reason] min_steps 増加 (4以上) 軸は廃棄確定**。min_steps を 2 に下げる方向 (step 1 のみブロック、forced divergence 削減) が次の有効仮説。
>
> **Mixed result (exp_074 infra_timeout_900)**: timeout=900s は **task_22 の cold-start 救済を実証確認** (0→1.0、実行時間 187.6s)。ただし min_steps=4 variance で task_200/379/259 が後退し net で −0.028 (0.7083)。timeout 延長自体は構造的に正しい改善で pure effect = +0.020。**[infra] timeout=900 を全後続実験の baseline config として採用推奨**。missing 4→2 は timeout 延長の直接効果。
>
> **Negative result (exp_075 kira_obs_truncate_10k)**: obs_truncate 閾値 30KB→10KB への引き下げは **−0.040 の後退 (0.6683)**。発火件数 2/50 タスクのみ (task_396: execute_python 29KB + task_418: read_csv 50rows)。主要な大観測 (task_352/read_doc 63KB) は依然未カバー。後退の主因は obs_truncate ではなく stochastic failures (task_292 cold-start 0 steps / task_199 malformed JSON 27連続 / task_408 wrong final calc)。**obs_truncate 3連続後退 (exp_066: 0.6883 / exp_071: 0.6283 / exp_075: 0.6683) で [kira] obs_truncate 軸廃棄確定**。read_doc truncation で task_352 を狙う場合は別軸 (ROI 低) として独立評価が必要だが、min_steps=2 の方が優先度高い。

`pick` ステップが **直近 3 イテレーションで連続して同じ軸を試したら別軸に切り替える** ルールに従って、ここから 1 つ選ぶ。各エントリは「軸タグ」で分類済み。エントリ消化後は EXPERIMENTS.md に記録され、ここからは消す/打ち消し線にする。

> 軸タグ凡例: `[reason]` 単一エージェントの推論強化 / `[schema]` スキーマ理解 / `[output]` DABench メトリクス特化 / `[tool]` ツール設計 / `[fewshot]` 公開 gold 利用 / `[multi]` マルチエージェント / `[critique]` 自己批評ループ / `[robust]` 観測やり直しなど

### 系列 A: single-agent reasoning [reason]
- **A1: ReFoRCE-style format pre-commitment** — plan の先頭で、出力 column 名 / dtype / row_count を 1 ブロックに固定 commit。実行中に header を作り直すのを禁止。期待: zero_recall の "extra column" / "row count off-by-one" を抑制。
- **A2: Reflection-on-failure** — 1 候補目で answer 出した後に「私の答えは question のすべての制約を満たすか？ 行数は？ 単位は？」を自己問答する 1 ステップ追加。Reflexion 風。
- **A3: Plan validation gate** — `agent.py` parser が thought に `quoted_definitions` / `column_source_check` / `cardinality_check` の 3 タグが揃わない場合は tool_error を返してリトライ強制。**exp_010 が成功したらその上に重ねる強化版**。
- **A4: Question paraphrase first** — 最初の thought で question を 2 通りに言い換え、解釈の差を明示してから 1 つを選ぶ。task_163/169/89 のような「systematic 解釈ミス」を抑制。

### 系列 B: schema linking / data exploration [schema]
- **B1: Schema-first agent** — 任意の `execute_*` の前に `inspect_sqlite_schema` か preamble の DDL 確認を必ず通す。schema を見ずにクエリを書く agent を強制矯正。
- ~~**B2: Sample-rich preamble** ✅ **消化済み (exp_013)**~~ — λ0.5 = 0.6400（−0.031 vs exp_012）。ファイル種別ラベルなしで JSON/SQLite が混同された。**再試行するなら `[CSV]`/`[SQLite]`/`[JSON]` の明示ラベルが必須**。
- ~~**B3: knowledge.md term extraction** ✅ **消化済み (exp_023)**~~ — λ0.5 = **0.6837**（+0.0085 vs exp_017、新ベスト）。perfect +2、zero_recall −2。ただし task_352 退行（missing +3 副作用）。doc/ ファイル含むタスクで JSON 誤認リスクあり。
  - **追加改修**: question に出てくる名詞を Glossary とマッチングし、最初の thought に「question で 'type' とあるが、Glossary によれば events.type 列を指す（expense は別概念）」を強制引用させる
  - **ベース**: exp_012 から派生。期待効果: task_163 救済 + interpretation 系の zero_recall 1〜3 件削減 → +0.02〜0.06 score。
- **B4: Cross-table foreign-key map** — 全 SQLite/JSON について `<table>.<col> → <table>.<col>` の参照関係を preamble に明示。task_26 のような relational JSON で効きそう。

### 系列 C: DABench メトリクス特化 [output]
- **C1: Final column minimization pass** — answer 直前に必ず 1 ターン挟み、「question の signature に含まれない列があれば削除」を agent 自身に問わせる。extras_ratio 罰則を直接攻める。
- **C2: Multi-row plan flexibility** — superlative 語 (`lowest`, `highest`, `least`, `most`) を含む question では plan の row_count を「単数 or 複数（同率の有無を要確認）」と二段で書く。task_25 (lowest cost) の救済。
- **C3: SQL-first preference** — system prompt に「`*.db` ファイルが存在する場合は `answer_from_sql` を最初に検討せよ」を明記。SQLite 含む 27 タスク (54%) でツール選択の最短化。
- **C4: dtype canonicalization on answer** — answer 直前に「数値は decimal、日付は ISO、null は空文字」に正規化することを agent に義務付け。

### 系列 D: ツール設計改善 [tool]
- ~~**D1: Tool input tolerance** ⚡ **最優先（消化済み: exp_012）**~~ — **✅ exp_012 で実装・採用。λ0.5 0.6510→0.6710（+0.020、新ベスト）。** `execute_python` / `answer_from_python` / `answer_from_sql` が string `action_input` を受けたら自動で `{"code": str}` 等に変換。フォーマットエラー連鎖の根絶に成功（`__error__` −78%）。**新 exp のベースは exp_012 から派生させること**。
- **D2: Truncate-and-offload** ⚡ — `read_csv` / `read_json` / `read_doc` が大きな結果を返すとき、`/tmp/<task>/<step>.txt` に書いて observation には `path + 200 char preview` だけ返す。context 32k 圧迫の根本対策。
  - **特に有効**: 8 実験を通して常に missing になる **task_344 / task_418 / task_420** の 3 件。これらは answer 不到達 = 0 点 → どんな小さな進展でも +1〜3 perfect。
  - **実装範囲**:
    - `tools/filesystem.py` の `read_csv_preview` / `read_json_preview` / `read_doc_preview` の戻り値 size を計算
    - 4KB 超の場合: `/tmp/kobushi_offload_<task_id>_<step>_<basename>.txt` に書き出し
    - observation には `{"preview": "<最初 200char>...", "offload_path": "/tmp/...", "full_size": N}` を返す
    - 新ツール `read_offloaded(path)` を追加。agent が必要なら部分的に再取得
  - **system prompt 改修**: 「巨大なファイルは `read_offloaded` で行/列限定して再取得せよ。全文を context に取り込むな」
  - **ベース**: exp_012 から派生。期待効果: missing 4 → 1〜2、score +0.04〜0.08。
- **D3: Combined `profile_table` tool** — dtypes / sample / null 率 / 値 cardinality / min-max を 1 call で返す。`list_context` → `read_csv` × N の往復削減。
- **D4: `answer_from_sql` の自動 SQL 修復** — SQL が syntax error なら 1 回だけ自動的に「Python で SQL を sqlite に投げて parse エラーをキャッチして prompt にフィードバック」する補助ループ。
- **D5: Forbid manual `answer` for computed results** ⚡ — exp_012 失敗 17 件のうち **手書き `answer` 起因が 7 件以上**（task_163/173/25/86/89/196/200/379）。agent が `execute_python` の出力を目で読んで JSON 化する過程で row 数や列数がずれる。改修: system prompt に「Python/SQL を実行した結果から答えを作る場合は **必ず `answer_from_sql` か `answer_from_python` を使う**。`answer` 手書きは「単一定数（"yes" や "no" のような literal）」のみ許可」を明記。さらに registry で `answer` を呼ばれたら「もしかして `answer_from_python` を使うべきでは？」の警告を返す soft gate を追加してもよい。**ベースは exp_012、最小変更**。期待効果: zero_recall 12→5、score +0.10〜0.14。

### 系列 E: public gold from few-shot [fewshot]
- **E1: 3-shot fixed example** — `data/public/output/task_*/gold.csv` のうち、難易度 easy/medium で gold が 1col×1row / 1col×多 row / 2col の代表 3 件を選び、(質問 + reasoning + gold) を system prompt 末尾に固定挿入。
- **E2: dynamic few-shot retrieval** — 現タスクの question を embedding して、最も類似する公開タスクの (Q, reasoning, gold) を引いて挿入。実装重い、後回し。

### 系列 F: マルチエージェント [multi]
- **F1: Decomposer + Executor** — 「decomposer」persona が question を sub-query に分解、「executor」が各 sub-query を解く。MAC-SQL の Selector / Decomposer / Refiner の light 版。複雑だが 32 step 制約に効く可能性。
- **F2: Critic LLM** — answer 提出前に別の system prompt で「critic」役の LLM が「この答えは question を満たすか？」を批評し、yes なら通す / no なら revise を求める。CHASE-SQL pairwise judge の単体版。
- **F3: Sub-agent for `answer_from_python`** — `answer_from_python` を子 agent で実行し、子の探索ログを親 context に持ち込まない。LangGraph / Claude Code 流の sub-agent isolation。

### 系列 G: 自己批評ループ [critique]
- **G1: Reflexion (multi-attempt)** — 1 試行 失敗時、agent が「何が悪かったか」を自然言語で書いてから再試行。max 2 attempts。
- **G2: Self-debug on tool error** — `__error__` が出たとき、必ず「stderr の何が問題か」を 1 段落書いてから次の action を出す。observation 読み飛ばし防止。

### 系列 H: 計算予算配分 [robust]
- **H1: Question-level early stopping** — 32 step 中に 2 連続で同じ tool result を得たら「ループに入った」判定して即 answer に進む or 別 tool を強制。
- **H2: Hard task triage** — preamble token 数が >10k のタスクは hard と判定し、N=1 で短時間予算 → 軽実装で諦める。空いた時間を easy/medium に再配分。

### 系列 I: BoN の救済（exp_009 の修正版）

> ❌ **[multi] BoN 直列実行 廃棄確定** (exp_063_bon_n2, λ0.5=0.6317, Δ−0.0950 vs exp_040): BoN N=2 直列実行は per-task time budget を 2 倍消費。9 タスクがタイムアウト → `missing_prediction=9`（通常 1〜5）。T=1.0 候補の多様性も低く（T=0.0 と同じ誤答多発）、voting の恩恵ゼロ。**並列化なしの BoN I1/I2/I3 系列は全廃棄**。並列 BoN は runner アーキテクチャ変更が必要で実装コスト大 → 優先度低。

- **I1: BoN with diverse system prompts** ❌廃棄 — 直列実行のため timeout 問題は解決しない
- **I2: BoN tie-break by row count** ❌廃棄 — 直列実行のため timeout 問題は解決しない
- **I3: GenSelect** ❌廃棄 — 追加 LLM 呼び出しが timeout をさらに悪化させる

### Negative results (試して効果なし)

| 軸 | exp | 結果 | 教訓 |
|---|---|---|---|
| `[arch]` preamble_schema-only | exp_082_cursor_harness | n=3 mean 0.6174 ± 0.0281 (Δ−0.046 vs baseline 0.6633) | schema-only preamble (5K) が extended thinking 爆発を引き起こし missing avg 8.3/50 (通常 1-2)。discovery tools (grep_file/stat_file) は task_86/89/200 で有効だが cost を上回れず。実データ値を preamble から完全除去は Qwen3.5-35b に適合しない。**全 file type 統一 schema-only 軸は廃棄確定**。grep_file/stat_file は fulldata preamble ベースで additive 追加を推奨 |
| `[reason]` | exp_010 verify-before-execute | λ0.5 0.4820 (−0.169 vs exp_007) | thought 肥大で step 切れ。系列 A は当面凍結 |
| `[output+tool]` | exp_011 spec-driven 3-phase | λ0.5 0.5270 (−0.124 vs exp_007) | D1 解決後に再評価（exp_012 ベースで再試行推奨） |
| `[infra]` | thinking off (exp_008) | smoke 即死 | Qwen3.5 + ReAct は thinking 必須。自己修正に使っている |
| `[schema]` B2 | exp_013 sample-rich preamble | λ0.5 0.6400 (−0.031 vs exp_012) | ファイル種別ラベルなしで JSON/SQLite 混同。再挑戦には `[CSV]`/`[SQLite]`/`[JSON]` ラベル必須 |
| `[multi]` F cascade | exp_015 cascade_json_fix | λ0.5 0.3320 (−0.3390 vs exp_012) | **Shape mismatch guard (ok=False) + ANSWER_MAX_STEPS=4 の組み合わせが missing +14 の主因**。修正版 exp_022 で再評価済み → 依然大幅 regression |
| `[tool]` | exp_047 skill_split | λ0.5 0.6917 (−0.035 vs exp_040) | **`read_skill` 呼び出し 0/50 タスク** — Qwen3.5 MoE (3B active) は間接的な「エラー後にスキルを読め」2ステップ命令に従わない。Rules 10-14 を prompt から除去したため Rule 13 欠如で missing 1→4 (+3)。**skill_split / tool-mediated rule delivery 軸は廃棄確定**。ルールは system prompt に直接書くこと |
| `[output+critique]` | exp_034 no_manual_answer_validated | λ0.5 0.5756 (−0.078 vs exp_031) | **新種バグ「f-string JSON エスケープ失敗」70 回発生**。`answer_from_python` コード中の f-string が JSON escape を壊す。4 tasks 新規クリア（answer 撤廃の恩恵）は有効。**answer 撤廃コンセプト自体は正しい**が parse エラー完全解消が先決。max_steps=48 + execute_python ループで timeout 増加 |
| `[tool]` D5 | exp_016 forbid_manual_answer | λ0.5 0.6510 (−0.020 vs exp_012) | **`answer` 手書き禁止は効果なし**。`answer` 呼び出し回数は変化なし（33 回→33 回）。zero_recall の根本原因は wrong_value（計算誤り 7 件）と wrong_rows（行数誤り 5 件）であり `answer` 呼び出し起因は 0 件だった。task_352 で逆に prompt 制約が過剰探索を誘発して退行。D 系列の直接プロンプト制約アプローチは ROI 低い |
| `[multi]` BoN | exp_063_bon_n2 | λ0.5 0.6317 (Δ−0.0950 vs exp_040) | **BoN N=2 直列実行でタイムアウト 9 件**。T=1.0 候補の多様性低く voting 恩恵ゼロ。I1/I2/I3 系列を含む直列 BoN 廃棄確定 |
| `[kira]` double_confirm | exp_065_kira_double_confirm | λ0.5 0.7033 (Δ−0.0050 vs exp_061) | **チェックリスト後の revision が正答を誤答に変える** (task_173/194 各 1.00→0.00)。revision 率 28% (13/47) のうち改善 0 件。強制再考はハザード大。**[kira] double_confirm 廃棄確定** |
| `[output]` C1 | exp_067_c1_col_minimize | n=4 mean 0.6687 ± 0.0260, Δ+0.0054 vs baseline = noise | Rule 16+17 combo だが col-minimize Rule 17 は n=4 でも効果ゼロ。run-to-run spread=0.058。timeout=600 が task_249 cold-start を誘発。**[output] C1 col-minimize Rule 17 単独軸廃棄確定**。step 1 plan commit 制限が必要 |
| `[output]` C2 改良 | exp_018 fix_select_star | λ0.5 0.6620 (−0.0132 vs exp_017) | **`SELECT *` 禁止は wext を根本解消できない**。task_194 で「Never SELECT *」指示が 1 列→3 列の過剰反応を誘発（perfect→zero）。wext の根本は answer_plan column_count と SELECT 列数の不一致であり、prompt 禁止制約では解消不可。C2 系列は 2 回連続 regression — 次は C1 か D2 へ |
| `[tool]` D6 | exp_019 column_count_coercion | λ0.5 0.6087 (−0.0665 vs exp_017) | **agent の column_count 宣言を信頼した外部 truncate は逆効果**。agent の計画ミス時に truncate が zero_recall を +5 件誘発。wext 4→2 の部分改善はあるが損失がはるかに大きい。D6 設計欠陥確定 — 再試行不要 |
| `[tool]` D2 | exp_020 truncate_offload | λ0.5 0.6402 (−0.035 vs exp_017) | **デフォルトの preview 化は agent の習慣を破壊**。4KB 閾値で `read_csv_preview` 等が offload+preview 化したが、agent が preview だけで answer に走り wrong_value/zero_recall を +3 件誘発。missing 4→3 の救済はあるが損失が上回る。D2 を再挑戦するなら閾値を 50KB+ に引き上げて稀なケース限定にする必要あり |
| `[reason]` A4 | exp_021 question_paraphrase | λ0.5 0.6617 (−0.0135 vs exp_017) | **paraphrase 強制二択は誤選択リスクが利得を上回る**。Rule 11 で Paraphrase A/B/Chosen 形式を強制したが、wext −1 / missing −1 / zero_recall +2 で net regression。agent が Paraphrase A vs B のうち誤った方を選ぶ案件あり。A 系列 [reason] axis は exp_010 verify-before-execute と同じ失敗パターンで凍結確定。**再試行不要** |
| `[multi]` F cascade v3 | exp_022 cascade_v3 | λ0.5 0.3117 (−0.3635 vs exp_017) | **DuckDB 専用 Phase 2 は退路なし設計の致命欠陥**。3 つのバグ修正（ANSWER_MAX_STEPS=12, soft warning, JSON cheatsheet）を全適用したが、DuckDB 失敗時の `answer_from_python` / `answer` フォールバックがないため EXTRA_COLS 17 件 + MISSING 5 件発生（zero_recall +18）。**F 系列 cascade は設計欠陥確定 — 再試行不要** |
| `[schema]` B3 follow-up | exp_024 glossary_v2 | λ0.5 0.6233 (−0.0604 vs exp_023) | **doc_hint が全タスクに適用される設計欠陥**。全 50 タスクが knowledge.md を持つため「use read_doc」注記が全タスクに追加され、task_249 等 5 タスクで不要な JSON parse error ループを誘発（zero_recall +6）。task_352 ↔ task_249 スワップにすぎなかった。**B3 系列拡張は凍結** |
| `[fewshot]` E1 | exp_025 fewshot_3ex | λ0.5 0.6517 (−0.0320 vs exp_023) | **fewshot 例示内のパスに `context/` プレフィックスを含めたため全タスクで模倣エラー 10 件**。コンセプト自体は有効（task_86/352 救済）。パス表記を `database.sqlite` 形式に修正した exp_026 を試すべき。巨大 context タスク（task_249/259、150MB+）の 0 ステップタイムアウトも悪化 |
| `[robust]` H1 | exp_027 loop_break | λ0.5 0.7067 (−0.0175 vs exp_026) | **同一 observation 検出は実際のループを捉えられない**（task_330 型は observation が微変化するため不発）。発火は task_420 の 1 件のみで効果なし。さらに Example C = task_173 の完全一致問題が発覚（モデルがショートカット SQL 実行 → データなし → 誤答）。**H1 系列凍結、fewshot 例をテストセット外タスクに差し替える必要あり** |
| `[fewshot]` E1 synthetic | exp_028 fewshot_synthetic | λ0.5 0.6631 (−0.0611 vs exp_026) | **合成例が全て `answer_from_sql` 一辺倒 → JSON/Python 処理タスクで大幅退行**（zero_recall +5）。task_11/196/199/200 が新規退行。task_173 ショートカット問題は解消されたが代償が大きすぎた。**fewshot 例はツール多様性（SQL + Python + doc 各 1 件以上）が必須**。E1 軸は設計制約が厳しく追加投資の ROI 低い — I3 / B1 への切り替えを推奨 |
| `[schema]` B1 schema-first | exp_029 schema_first | λ0.5 0.7067 (+0.0230 vs exp_023 leak-clean) | **Rule 11（inspect_sqlite_schema 必須）は task_257 に有効**。ただし exp_023 ベースのため FIXED_EXAMPLES なし → task_86/199/200/259 が exp_026 比で退行。**Rule 11 単体の効果は確認済み**。Rule 10（superlative filter-back）は task_80 のような exact-match には不適用。**次の候補**: exp_026 ベース + Rule 11 の組み合わせ（LEAK 問題残存）か deterministic cascade（PRIORITIES.md URGENT） |
| `[preamble:hint+schema-graph]` | exp_099_schema_graph_fallback | n=1 λ0.5 0.5961 (Δ=-0.067 vs single-attempt floor, −3.08σ) | **context-blind precomputed advisory が dictation bias を引き起こした**。hint 発火 ~7 task で avg −0.286/task (命令調 FK/orphan hint → agent が wrong join/drop 強制)。missing=10 (LLM precompute call の latency 増で 900s timeout 頻発)。hint 非発火 ~8 task では +0.075 (schema-only downgrade 副産物)。**5 連続 [context]/[preamble:hint] regression (exp_087/088/095/097/099) で廃棄確定**。schema-only downgrade (oversized file → column+dtype+3rows) は独立候補 (`exp_104_preamble_downgrade_only`) |

### Positive results (採用・ベスト更新)

| 軸 | exp | 結果 | 教訓 |
|---|---|---|---|
| `[tool]` D1 | exp_012 tool_tolerance | λ0.5 **0.6710** (+0.020 vs exp_007) | `action_input` 文字列→`{"code":str}` 自動変換で `__error__` −78%。**全新 exp のベースに** |
| `[output]` C2 | exp_017 tie_aware_superlative | λ0.5 **0.6752** (+0.0042 vs exp_012), λ0.0=0.7000 | filter-back 例示で task_352 救済。ただし wext +3 の副作用（task_259/330 退行）。例示 SQL に `SELECT *` を含めると余分列選択を誘発する |
| `[schema]` B3 | exp_023 glossary_preamble | λ0.5 **0.6837** (+0.0085 vs exp_017) | knowledge.md から glossary 抽出・preamble 挿入で perfect +2、zero −2。missing +3 の副作用（task_352 退行、doc/ ファイルを JSON と誤認）。task_352 を再び完走できれば +0.02 の余地あり |
| `[fewshot]` E1 | exp_026 fewshot_pathfix | λ0.5 **0.7242** ⚠ LEAK (+0.0405 vs exp_023) | 実タスク固有パス（db/event.db 等）を例示し「パスを模倣するな」注記追加。task_173/199/86 が新規 perfect。task_257 退行（大規模 JSON 混在タスク、短絡誤答）。wext ±0、missing −2。 |
| `[output]` Rule12 | exp_036 explicit_select_cols | λ0.5 **0.6833** ✅ **新clean best** (+0.030 vs exp_031) | Rule 12（先行列計画 + SELECT 列明示 + SELECT * 禁止）で total_extra_cols 31→18（−42%）。zero_recall 14→9、λ0.0=0.700。残課題: unclosed fence Case 3 欠落・task_408 早期丸め・task_259 wrong-column |

---

## 0. 制約と勝ち筋の仮説

**動かせない前提**
- モデルは **qwen3.5（35B）固定**、API 経由。
- **学習・蒸留・ファインチューニング不可**。

**動かせる変数（＝勝つために攻める所）**
1. モデルが **見るもの**（プロンプト、ツール返値、注入する DDL / 値 / few-shot）
2. モデルが **次にやること**（スキャフォルド：ReAct / CodeAct / plan-execute / reflection）
3. **どう集約するか**（best-of-N、column-signature 投票、GenSelect 風 selector）
4. **計算予算の配り方**（簡単タスクは早期停止、難しいタスクに N を寄せる）

**仮説（thesis）**
- 同一固定モデルで勝敗を決めるのは「**情報を front-load する**」「**多様なサンプルを集める**」「**メトリクスに沿って選ぶ・整える**」の 3 点。AIMO-2（NemoSkills / imagination-research）も Spider 2.0 / BIRD 上位もすべて、固定モデル前提でこの 3 軸を最大化していた。
- 公式メトリクスは **列 signature 一致** + **余剰列にペナルティ**。投票や integration の単位は「行」ではなく「列 signature の集合」になる。AIMO の整数答え多数決とは集約軸が違うので、ここを **公式 normalizer (`kobushi_core/eval/csv_compare.py:_column_signature`) を再利用して評価ベースで投票** するのが筋。
- 現状ベースライン (`exp_001_react_baseline`, max_steps=16) は失敗の大半が「16 ステップ以内に `answer` を呼べず」。**まず観測情報を front-load して step 浪費を減らす**のがすべての施策の前提条件。
- **AIMO 3 の null result（Nitarach 2026, arXiv:2603.27844）の警告**: 高 T 多サンプルだけで pairwise error correlation は ρ≈−0.12 まで下がり、プロンプト多様化の追加価値は小さい。**pass@N → maj@N の 6pt selection-loss は selector でしか埋まらない**。我々の設定でもプロンプト多様化に時間をかけるより、**最終答えを選ぶ仕組み（GenSelect / entropy-weighted vote / signature 投票）に投資**すべき。

**現状ベースライン実測 (`exp_001_react_baseline_002`)**
- official_score_lambda_0_5_mean = **0.2650**
- 内訳：perfect 13 / partial 1 / zero 2 / missing 34（= 16 step 切れ）
- λ を振っても score はほぼ動かない（0.27 → 0.26）→ **今は extras より recall（完走率）が支配的**。最小化パス（§2.1）は完走率が上がってから効いてくる。

---

## 1. 最優先：観測 front-load で step 予算を確保（exp_002 候補）

> 動機：現状失敗の支配的要因は「`max_steps` 切れ」。1 ターン目から有用な情報を全部投入できれば、後段の改良すべてが効きやすくなる。

### 1.1 Deterministic preamble — `answer` までの定型探索を 1 ターンで済ませる
- ReAct を回す前に、決定論的な前処理で以下を全部集めて system / user prompt の先頭に埋める：
  - `context/` 全ファイルの再帰リスト（拡張子・サイズつき）
  - `knowledge.md` の全文（**全タスクに同梱されている重要ヒント**）
  - 全 CSV: dtypes / null 率 / cardinality / min-max / 先頭 5 行
  - 全 JSON: keys と先頭の構造（`json.dumps` の `indent=2` で 50 行）
  - 全 SQLite/DB: テーブル一覧、カラム DDL、各テーブルの 5 行サンプル
- これだけで現状の `list_context` → `read_csv` → `inspect_sqlite_schema` → `read_csv` × N の往復が消える。
- 配置: `experiments/exp_002_preamble/` を新規実験で切り、`runner.py` でタスクごとに preamble を生成 → 初期 system message に注入。

### 1.2 `profile(path)` 単発ツール
- 1 回呼ぶだけで dtypes / null 率 / cardinality / min-max / unique 例 / 先頭 N 行を返す。
- preamble 後でも、agent が「このカラムを深掘り」と判断したときの 1-call 完結ツールとして残す。
- 配置: `experiments/exp_001_react_baseline/tools/profile.py`（preamble と相補）。

### 1.3 DuckDB 横断クエリツール
- CSV/JSON/Parquet/SQLite を 1 つのエンジンで join できる。`read_csv_auto` / `read_json_auto` で型推定も任せられる。
- ツール名: `execute_duckdb(sql)`。preamble の中で全ファイルを virtual table として登録するスクリプトを示しておく。
- 期待: `execute_python` を都度書くより少ないトークン・少ないターンで複雑な集計が書ける。

### 1.4 max_steps を 16 → 32 に
- 1.1〜1.3 で大半は不要にしたいが、それでも届かないタスク用に保険として上げる。
- 配置: 既存 yaml の 1 行変更で済むので preamble 実験と同居させる。

---

## 2. DABench メトリクス特化（exp_003 候補）

> 動機：本コンペは Score = Recall − λ·(extras/pred)。**他コンペにない「列を増やすと損する」**特性。ここに最適化するだけでスコアが直接動く。

### 2.1 出力カラム最小化パス
- `answer` 直前に「質問文と現在の候補テーブルから、不要な列を落とす」確認ステップを挟む。
- 実装: agent に「現在の output_columns: [...]。質問に答えるのに **必須でない** 列を列挙し、最終 answer から除外せよ」と問う 1 ターンを差し込み。
- DABench で最も再現性のある +α 候補。**ReAct はデフォルトで「念のため」列を増やしがち**で、そこに直接ペナルティが乗る。

### 2.2 Signature-aware self-consistency（§3 と統合）
- N 候補の最終 CSV を **公式 `_column_signature` で正規化** してから列集合を比較。
- 「過半数候補に出現する列」のみを最終 answer に採用 → recall を維持しつつ extras を切る。
- 通常の str-eq 投票だと dtype/空白で取りこぼすので、必ず公式 normalizer 経由。

### 2.3 λ 推定の感度分析
- 公式 λ は非開示。我々の評価器は λ ∈ {0.0, 0.1, 0.3, 0.5, 1.0} を全部出している。
- **推論時の閾値（最小化パスがどこまで列を落とすか）を λ=0.3 と λ=0.5 の中間で最適化**する設計にすると、公式 λ がどこにあってもロバスト。

---

## 3. Best-of-N + 集約（exp_004 候補、最大の +α 期待 / AIMO 3 教訓を反映）

> 動機：固定モデルで上限を突破する唯一の道は「同じタスクを複数回別アプローチで解いて選ぶ」。AIMO 上位、Spider 2.0 / BIRD 上位の共通項。
> **AIMO 3 (Nitarach 2026) のキー知見**: 高 T (=1.0) で 8 並列回せば pairwise error correlation は ρ≈−0.12 まで下がる（自然に decorrelate される）。**追加のプロンプト多様化はほぼ効かない**。pass@N と maj@N の差（selection loss）が AIMO 3 で 6pt → これを埋める selector が最重要レバー。

### 3.1 高温度シンプル並列（差別化プロンプト最小限）
- N=5–8、temperature=1.0、**プロンプト 1 種**で並列。
- ただし DABench は SQL/pandas どちらでも書ける問題が多いので、**ツール選択の多様性は残す**：
  - 半分: DuckDB SQL を促すヒント、もう半分: pandas を促すヒント
- imagination-research の AIMO-2 2nd「7 CoT + 8 Code」は本コンペでも 2 系統だけ残す（純 NL 系列は要らない、必ずツール経由）。
- ⚠ 3 系統以上のプロンプト混合は AIMO 3 の null result により ROI 低下が見込まれる、最初は試さない。

### 3.2 集約 A：**列 signature 投票**（DABench 特化、最重要）
- 各候補の最終 CSV を `kobushi_core/eval/csv_compare.py:_column_signature` で正規化 → 列集合を K/N 多数決。
- K/N の閾値で recall vs. extras を制御。最初は K = ceil(N/2)。
- 行の値は signature 一致候補の中で **セル単位多数決**（公式 normalizer 経由で str-eq 投票）。
- **AIMO の整数答え majority を CSV テーブルに焼き直したもの**。これがコア。

### 3.3 集約 B：**Entropy-weighted aggregation**（AIMO 3 公開ノート 43/50 で採用）
- 各候補に対し最終トークン群（`answer` 引数の columns/rows）の log-prob からエントロピーを計算 → 重み = 1/entropy で投票。
- 実装：OpenAI 互換 API なら logprobs を取得 → answer ツール呼び出し時の token-level prob 平均でスコア化。
- plain majority よりわずかに上、実装コスト低。§3.2 と組み合わせて signature 投票内のタイブレークに使う。

### 3.4 集約 C：**GenSelect / Pairwise selector**（pass@N → maj@N の 6pt 差を埋める本命）
- qwen3.5 自身を judge にし、N 候補から最良を選ばせる。実装パターン 2 種：
  - (a) **GenSelect** (NVIDIA, arXiv:2507.17797): 全候補一覧 + 質問 + コンテキスト要約 を 1 プロンプトに入れて「最良はどれか」を生成
  - (b) **Pairwise judge** (CHASE-SQL): 候補ペアごとに勝者選択 → トーナメント
- 第 1 段で §3.2 の signature 投票が割れたタスクのみ呼べばコスト 1 タスクあたり +1 API call。
- AIMO 3 の最大の教訓：**「selector に予算を回す」がプロンプト工夫より高 ROI**。

### 3.5 Question-level early stopping
- K/N 一致で残りロールアウトを打ち切り、浮いた予算を hard task に回す（NemoSkills 4/12, imagination 5/7）。
- 簡単タスクで 5/8 一致したら即停止 → 残り 3 ロールアウト分を hard task の追加サンプリングへ。
- `runner.py` のスケジューラを拡張：簡単タスクの早期完了で空いた worker を未完の hard task に再配分。
- imagination-research AIMO-2 の `adjust_speed` を CSV ベンチに移植する位置付け。

---

## 4. スキーマ理解の強化（exp_005 候補）

### 4.1 公開タスクからの few-shot 構築
- `data/public/output/task_*/gold.csv` が手元にある（**学習禁止だが in-context demo 利用は可**のはず ← 要規約再確認）。
- 簡単タスク 3-5 件を「(質問 + コンテキスト要約 + 正解 CSV)」形式で system prompt に固定挿入 → DABench の答え方の **format/disposition** をモデルに学ばせる。
- few-shot は固定 1 セットで十分。動的検索しなくても効く（プロンプト長短縮優先）。

### 4.2 スキーマリンキング（広い DB 用）
- DuckDB に登録した全テーブルから、質問語と類似度の高い列だけを候補抽出 → prompt に DDL を限定注入。
- 手段: rapidfuzz で質問語×カラム名、＋カラム値サンプルへのフレーズ完全一致検索。
- BIRD/Spider 系上位は全員前段でやっている。広いスキーマで効く。

### 4.3 値検索（CHESS 風）
- 質問に固有名詞や enum 候補がありそうな場合、全テーブルの全列を 1 度走査して「この語が出現する列」を返す。
- 列特定の精度が劇的に上がる（CHESS 論文）。

---

## 5. スキャフォルド改善（exp_006 以降）

### 5.1 CodeAct スキャフォルドへの移行
- ReAct (JSON-tool) を捨て、**実行可能 Python を直接吐かせる + 永続カーネル** へ。
- 中間 DataFrame をカーネル変数として持ち越せる → step あたりの情報量が増える。
- OpenHands の ACI 設計を踏襲：「ツールは人ではなくモデルが使いやすい」設計。

### 5.2 Plan-Execute 分解
- Step 0 で「YAML 形式の plan（最大 6 ステップ）」を吐かせ、その後の各ターンで plan の 1 行ずつを実行。
- 16-32 step 制約の中で「迷走」が減る。DataInterpreter / OpenInterpreter-Data の構造。

### 5.3 反射ループ（execution-feedback）
- ツールエラー or 空結果 → 次ターンで「原因仮説 + 別アプローチ」を強制する分岐。
- Reflexion / Self-Debug。2-3 反復で頭打ち、それ以上は無駄。

---

## 6. 評価インフラ

### 6.1 公式ルール準拠の追加カバレッジ
- 完了：`null` 正規化 / 0 フロア（PR #11）
- TODO：日時 TZ エッジケース（夏時間・ナノ秒・`+00:00` ⇄ `Z`）
- TODO：分割名 ⇄ 結合名マッチングが「3 列以上の連結」も許すか公式に未確認。実装は隣接 2 列のみ。

### 6.2 タスク難易度別ダッシュボード
- `task.json:difficulty` で集計 → どの帯で落ちているか可視化。打ち手選定の最重要根拠。

### 6.3 シード変動測定
- 同一設定 3-5 ラン × 50 タスクで標準誤差を測る。AIMO-1 は 5-10 seed で 1-3% 分散と報告。
- 50 タスクは小さいので公開 LB のブレが大きいはず → 内部評価では必ず複数シードで判断。

### 6.4 失敗モード自動分類
- `trace.json` を後段で正規表現分類：`max_steps_exhausted` / `tool_error` / `empty_result` / `wrong_columns` / `wrong_dtype`。
- 改善実験のたびにこの分布が動くかを見る。

---

## 7. 適用しない（このコンペでは禁じ手 or 効果薄）

- **Frontier model 切替**（Claude/GPT/Gemini）: モデル固定の制約に違反。
- **ファインチューニング / 蒸留 / DPO**: 学習不可の制約に違反。
- **R1 系 reasoning model 切替**: 同上。仮に許されても 16-32 step 予算では verbose CoT が割に合わない（AIMO-2 は 5h 持っていた）。
- **整数答え前提の majority vote**: DABench は CSV テーブル → 列 signature ベース投票が必須。
- **量子化 / GPU 最適化**: API 利用なので無関係。

---

## 8. 優先順位（提案、現状実測 0.2650 / 完走率 32% を踏まえて）

1. **exp_002**: Deterministic preamble + max_steps=32 — 完走率を上げる土台。**最も支配的な失敗要因（34/50 が step 切れ）への直接対処**。
2. **exp_003**: Best-of-N (§3.1) + 列 signature 投票 (§3.2) — 上限突破の本命。preamble で完走率が上がってから初めて投票が機能する。
3. **exp_004**: GenSelect / Pairwise selector (§3.4) — AIMO 3 の最大の教訓「selection loss を selector で埋める」。§3.2 で割れたタスクにのみ適用。
4. **exp_005**: Output column minimization pass (§2.1) — extras ペナルティが効くフェーズに入ってから。今 (extras=0) は ROI ゼロ。
5. **exp_006**: 公開タスクから few-shot を 3-5 件固定挿入 — フォーマット学習。
6. **exp_007+**: CodeAct スキャフォルド・スキーマリンキング・entropy-weighted vote — 残り余地。

各実験は前段に積み上げる形で評価。**preamble は他すべての実験の土台**で、効果が無くても残す。`exp_003` 以降は前段を含めて測る。

---

## 9. リファレンス

### 主要論文 / 実装
- CHASE-SQL — <https://arxiv.org/abs/2410.01943>（多生成器 + pairwise selector）
- XiYan-SQL — <https://arxiv.org/abs/2411.08599>
- CHESS — <https://arxiv.org/abs/2405.16755>（値検索 + 列フィルタ）
- MAC-SQL — <https://arxiv.org/abs/2312.11242>
- Spider-Agent — <https://arxiv.org/abs/2408.05109>
- CodeAct — <https://arxiv.org/abs/2402.01030>
- OpenHands — <https://github.com/All-Hands-AI/OpenHands>
- SWE-agent ACI — <https://swe-agent.com>
- Reflexion — <https://arxiv.org/abs/2303.11366>
- Self-Debug — <https://arxiv.org/abs/2304.05128>
- DSPy — <https://dspy.ai>
- Chain-of-Table — <https://arxiv.org/abs/2401.04398>

### Kaggle 勝者解法（横展開元、固定モデル時代の集約・予算配分の参考に）
- **AIMO 3 (2025-2026)** — winner 公開未了 (May 2026)
  - "Every Improvement Failed" / Nitarach — <https://arxiv.org/abs/2603.27844> / <https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/every-improvement-failed-aimo3>（**プロンプト多様化 23 通り全敗、selection loss 6pt の null result**）
  - Entropy-weighted self-consistency 公開ノート (43/50) — <https://www.kaggle.com/code/kurianbenoy/43-50-aimo-3-gpt-oss-120b-weighted-entropy>
  - GenSelect (NVIDIA) — <https://arxiv.org/abs/2507.17797>（**生成的 best-of-N selector の標準実装**）
  - 競技ページ — <https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/overview>
- AIMO 2 (2024-2025)
  - NemoSkills 1st — <https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills>（質問単位 early stop 4/12）
  - imagination-research 2nd — <https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/imagination-research-2nd-place-solution-team-imagi>（7 CoT + 8 Code 混合 + 5/7 早期停止 + adjust_speed）
  - 横断サマリ — <https://medium.com/@lucamassaron/learning-from-kaggle-competitions-using-gemini-2-5-ai-mathematical-olympiad-progress-prize-2-805b5b8d87f2>
- AIMO 1 (2024)
  - Numina 1st — <https://huggingface.co/blog/winning-aimo-progress-prize>（SC-TIR の元祖）

### 公開ベンチマーク現況参考
| 名前 | 規模 | 現 SOTA |
|---|---|---|
| Spider 2.0-Lite | 企業級 SQL multi-dialect | ~35–40% EX |
| BIRD-SQL | text-to-SQL 95 DBs | ~75% EX |
| BIRD-CRITIC | 多ターン対話/デバッグ | ~30–40% |
| DABench (本コンペ) | CSV/JSON/SQLite/text → CSV | 公開 SOTA 未定 |
| DA-Code | code-grounded EDA/ML | ~30% |
| ScienceAgentBench | 科学タスク E2E | 32–42% |
| TableBench | 表 fact-check + 数値推論 | 60–70% |
