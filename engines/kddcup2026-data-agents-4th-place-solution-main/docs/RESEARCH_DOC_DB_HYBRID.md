# Research Findings: Knowledge-Grounded Data-Analysis Agents (2024-2026)

> 2026-05-07 に web research agent で収集した、**doc + DB hybrid task** に対する SOTA アプローチの整理。
> DABench (Qwen3.5-35B-A3B + ReAct + 3-attempt union, knowledge.md + narrative.md + DB の混合タスク) に対する **適用可能性ランキング**付き。

---

## 1. 最重要発見

**DABench の `knowledge.md` は BIRD benchmark の `evidence` field と構造的に同一**:
- BIRD: query ごとに `evidence` (= 数値推論ヒント / domain knowledge / 同義語 / 値例) を付与
- DABench: task ごとに `knowledge.md` (= Glossary / Metric Definition / Examples / Ambiguity Resolution)

→ **2024-2026 の BIRD 系 SOTA paper の知見が、ほぼそのまま DABench に転用可能**。これは戦略的に大きい。

---

## 2. SOTA 系の知識注入アプローチ

### 2.1 BIRD evidence-aware text-to-SQL ベースライン

- **元論文**: BIRD ([NeurIPS 2023](https://openreview.net/forum?id=dI4wzAE6uV))
- **方法**: `evidence` field を質問の直後 / schema の前に inject
- **適用度**: ⭐⭐⭐⭐⭐ — DABench の knowledge.md を BIRD evidence と同じ位置に置く方針は確実に valid

### 2.2 SEED — auto-evidence synthesis ([arXiv 2506.07423](https://arxiv.org/abs/2506.07423), ICDEW 2025)

- **方法**: schema profiling + description mining + value sampling から LLM で synthetic evidence を生成
- **結果**: BIRD/Spider で **human evidence 提供版より良い場合あり** (= human evidence は noisy / incomplete)
- **適用度**: ⭐⭐⭐⭐ — knowledge.md にヒットしない task で synthetic evidence 生成の前段階。POC 価値あり。

### 2.3 RubikSQL — agentic lifelong KB ([arXiv 2508.17590](https://arxiv.org/abs/2508.17590), Aug 2025)

- **方法**: DB profiling → structured info extraction → agentic rule mining → SQL profiling、Unified Knowledge Format (UKF) で indexing
- **結果**: SOTA on KaggleDBQA + BIRD Mini-Dev
- **適用度**: ⭐⭐ — lifelong learning は DABench (= task 独立) では ROI 低い、ただし structured-extraction stage は narrative parse の参考に

### 2.4 BIRD-INTERACT — knowledge as queryable hierarchical KB ([arXiv 2510.05318](https://arxiv.org/abs/2510.05318), ICLR 2026 Oral)

- **方法**: DB と階層 KB をペアにし、agent が knowledge action を retrieve
- **結果**: GPT-5 でも 17% (= 最難レベル)
- **適用度**: ⭐⭐⭐ — knowledge.md がさらに大型化したら有効、現状は直接 inject で十分

### 2.5 LiveSQLBench — DAG-structured knowledge ([livesqlbench.ai](https://livesqlbench.ai/))

- **方法**: HKB を DAG 形式 + JSON / Document の dual format
- **適用度**: ⭐⭐⭐⭐ — narrative.md を JSON-mirror に変換、両方を agent に渡すパターンが直接参考

### 2.6 Patwardhan et al. — DB-level domain statements ([arXiv 2510.02394](https://arxiv.org/abs/2510.02394), Oct 2025)

- **方法**: per-query NL evidence は非現実的とし、structured DB-level 文 (= "WBC unit = ×10⁹/L; normal 4-9") を index して sub-string match で retrieve
- **適用度**: ⭐⭐⭐⭐ — 我々の knowledge.md を atomic statement に分解 → retrieval、明確な precedent

---

## 3. Multi-Source Reasoning Pipelines

### 3.1 CHESS ([arXiv 2405.16755](https://arxiv.org/abs/2405.16755), ICML 2025)

- **構成**: Information Retriever (LSH + vector DB over column descriptions) → Schema Selector → Candidate Generator → Unit Tester
- **結果**: 65-66.7% EX on BIRD
- **適用度**: ⭐⭐⭐⭐⭐ — IR stage の設計 (LSH + vector DB) を knowledge.md indexing に直接転用可能

### 3.2 MAC-SQL ([arXiv 2312.11242](https://arxiv.org/abs/2312.11242), COLING 2025)

- **構成**: Selector + Decomposer + Refiner (= 実行 feedback loop)
- **適用度**: ⭐⭐⭐ — Refiner pattern は exp_109 ReAct に内在、Decomposer は別軸

### 3.3 RSL-SQL — bidirectional schema linking + CIA ([arXiv 2411.00073](https://arxiv.org/abs/2411.00073))

- **方法**: bidirectional pruning (94% recall, 83% column reduction) + Contextual Information Augmentation (= column descriptions injection)
- **結果**: 67.2% on BIRD
- **適用度**: ⭐⭐⭐⭐⭐ — **CIA = knowledge.md Glossary を column 単位で inject** という設計、直接適用

### 3.4 AutoLink — agentic iterative schema expansion ([arXiv 2511.17190](https://arxiv.org/abs/2511.17190), Nov 2025)

- **方法**: agent が schema subset を iterative に拡張
- **結果**: BIRD-Dev recall **97.4%**, Spider-2.0-Lite 91.2%, 34.9% EX (= #2 on leaderboard)
- **適用度**: ⭐⭐⭐⭐ — task_330 (115 col) のような wide CSV に直接効く

### 3.5 SchemaGraphSQL ([arXiv 2505.18363](https://arxiv.org/abs/2505.18363), May 2025)

- **方法**: FK graph 構築 → LLM で source/dest 抽出 → 古典 path-finding
- **適用度**: ⭐⭐⭐ — DABench は CSV 主体で declared FK 不在、value_overlap edge を補う方向

### 3.6 Spider 2.0 / ReFoRCE ([arXiv 2502.00675](https://arxiv.org/pdf/2502.00675))

- **背景**: Spider 2.0 は 3000+ column、metadata + dialect docs + project codebases 必要、GPT-4o 10.1%、o1-preview 17.1%
- **ReFoRCE**: Spider 2.0 leading agent
- **適用度**: ⭐⭐⭐⭐ — DABench は Spider 2.0 と同質の "doc-heavy heterogeneous" benchmark、ReFoRCE recipe 追跡価値高い

### 3.7 OpenSearch-SQL — Query-CoT-SQL cache ([arXiv 2502.14913](https://arxiv.org/abs/2502.14913))

- **方法**: self-taught dynamic few-shot (Query, CoT, SQL) を cache、新質問で retrieve
- **結果**: 69.3% BIRD dev, 72.28% test
- **適用度**: ⭐⭐⭐⭐ — 50 task 横断 cache で SQL-example のない task を救済する道

---

## 4. Narrative-to-Structured Extraction

### 4.1 AIE — Hybrid Long Document Extraction ([arXiv 2412.20072](https://arxiv.org/html/2412.20072v1), Dec 2024)

- **方法**: hybrid long doc (text+tables) を segment 化 → iterative LLM extraction → schema-driven aggregation
- **適用度**: ⭐⭐⭐⭐⭐ — 80-280KB narrative.md (Patient/superhero/races/etc) の前処理に直接適用、我々の POC 失敗 (single-shot) を救う設計

### 4.2 Markdown-as-extraction-target

- **手法**: narrative → markdown → segment-aware chunking → LLM extraction
- **適用度**: ⭐⭐⭐⭐ — DABench narrative は既に markdown、heading 構造を chunking 境界に活用

---

## 5. Ambiguity Resolution

### 5.1 AmbiSQL ([arXiv 2508.15276](https://arxiv.org/abs/2508.15276), SIGMOD'26 demo)

- **方法**: term → multiple columns 検出 (= "Fresno" → City|County)、XiYan-SQL 組み合わせで disambiguation
- **結果**: **42.5% → 92.5%** on 40-Q ambiguous benchmark (= +50pp!)
- **適用度**: ⭐⭐⭐⭐⭐ — task_25 (cost), task_163 (type) のような ambiguous noun を直接攻略可能。**3-attempt union を temp branching ではなく deliberate ambiguity branching に置換** という設計が示唆

### 5.2 NL2SQL Schema-Ambiguity Recommender ([arXiv 2505.19302](https://arxiv.org/pdf/2505.19302), May 2025)

- **方法**: ambiguous schema mapping 識別 + 候補 query ranking
- **適用度**: ⭐⭐⭐ — multi-attempt の ranker として有用

### 5.3 Pre-computed disambiguation table

- **手法** (RubikSQL UKF, SEED 共通): term → schema element の static dictionary
- **適用度**: ⭐⭐⭐⭐⭐ — knowledge.md からの自動構築は実装 1 hour、即効

---

## 6. Cross-Task Knowledge Transfer

### 6.1 SAFE-SQL — self-augmented in-context learning ([EMNLP 2025](https://aclanthology.org/2025.emnlp-main.962.pdf))

- **方法**: 自前 demonstration 生成 + quality filter
- **適用度**: ⭐⭐⭐ — 既存 SQL example 不足時の補完

### 6.2 RASL — retrieval-augmented schema linking ([Amazon Science](https://assets.amazon.science/1b/95/8f62e89647348f4c4836f6c3040d/rasl-retrieval-augmented-schema-linking-for-massive-database-text-to-sql.pdf))

- **方法**: schema + descriptions の RAG
- **適用度**: ⭐⭐ — 我々の DB は小規模、過剰

---

## 7. **DABench への適用可能性ランキング**

| # | アイデア | 出典 | 実装コスト | 期待 +λ | 備考 |
|---|---|---|---|---|---|
| **1** | **Pre-computed glossary `{term: column}` from knowledge.md** | RubikSQL / SEED / Patwardhan | 低 (~1h) | +small-mid | 最速で試せる、precedent 強い |
| **2** | **Narrative.md → atomic-fact JSON (chunk + extract)** | AIE 2412.20072 / LiveSQLBench | 低-中 (~半日) | +mid (narrative-heavy task) | task_344/396/418 直接攻略 |
| **3** | **Ambiguity-branched 3-attempt union** | AmbiSQL 2508.15276 | 低 (~半日) | +mid | temp branching を ambig branching に置換 |
| **4** | **Column-description injection (CIA-style)** | RSL-SQL 2411.00073 / CHESS | 低 | +small | 既存 knowledge.md を schema 近接に inject |
| 5 | Query-CoT-SQL example cache | OpenSearch-SQL 2502.14913 | 中 | +mid (LB) | cross-task transfer、実 LB に効く可能性大 |
| 6 | Iterative agentic schema expansion | AutoLink 2511.17190 | 中 | +mid (wide-table) | task_330 の 115 col 対策 |
| 7 | Auto-evidence synthesis | SEED 2506.07423 | 中 | +small-mid | 知識欠落 task で synthetic evidence |
| 8 | FK pathfinding | SchemaGraphSQL 2505.18363 | 中 | +small | 過去 exp_099 で失敗、value_overlap で改良必要 |
| 9 | Multi-hop DAG knowledge | LiveSQLBench / BIRD-INTERACT | 高 | unclear | knowledge.md が現状小さい、過剰 |
| 10 | Lifelong KB across runs | RubikSQL | 高 | low | DABench task 独立、ROI 低 |

---

## 8. **やらない方が良い** ことの確認

研究 agent と DOC_TEMPLATE_ANALYSIS の両方の結論として:

| 避けるべき | 理由 |
|---|---|
| 全 knowledge.md を毎 step dump | CHESS/RSL-SQL は targeted retrieval を選ぶ。我々も既に Glossary 抽出は preamble にあるが、より精選すべき |
| Multi-hop DAG knowledge bases | knowledge.md が現状 5KB 程度、overengineering |
| Lifelong KB across runs | DABench は task 独立、cache の hit 率不明 |
| M-Schema / k=2 vote / Reflexion | exp_088 / exp_087 / exp_051 で失敗確定、2025 文献も反証なし |
| Single-shot LLM で narrative 全文抽出 | POC で失敗確認、chunked + iterative が必須 |

---

## 9. 推奨される次の実験軸

### Phase 1 (即効、~1 日)
**Items #1 + #4 stack**: knowledge.md → `{term: column}` glossary + Question-Relevant CIA section を inject
- 出典: RubikSQL/SEED/Patwardhan + RSL-SQL
- 狙い: ambiguity-resolved schema linking 強化

### Phase 2 (~2-3 日)
**Item #2**: narrative.md → atomic-fact JSON (chunk + iterative extract)
- 出典: AIE 2412.20072
- 狙い: task_344/396/418 等の Type F 救済

### Phase 3 (~半日)
**Item #3**: 3-attempt union を **ambiguity-branched** に置換
- 出典: AmbiSQL 2508.15276 (= +50pp 実証)
- 狙い: task_25/163 等の ambiguous noun 救済

### 累計期待効果
- Phase 1: +0.02-0.04
- Phase 2: +0.02-0.05
- Phase 3: +0.03-0.06

→ **合計 +0.07-0.15 local → LB +0.07-0.14 = 3 位射程 (= 0.58 over)**

---

## 10. メモ

- BIRD 系 SOTA を借用するのは valid だが、**評価 metric が違う** (= BIRD は EX = SQL execution match、DABench は λ = 列値マッチ)。SQL form の正確性より値 multiset 一致が重要。
- DABench の **narrative.md は decoy fluff 混入**が特徴 (= BIRD evidence にはない)。AIE などの精密抽出が必要。
- 我々の **3-attempt union** は AmbiSQL の "deliberate disambiguation branching" として再利用可能。temperature diversification より ambiguity-axis diversification が ROI 高い。

---

## 11. 関連 paper / source URL リスト

- BIRD: https://bird-bench.github.io/ , https://openreview.net/forum?id=dI4wzAE6uV
- SEED: https://arxiv.org/abs/2506.07423 , https://github.com/felix01189/SEED
- Patwardhan: https://arxiv.org/abs/2510.02394
- RubikSQL: https://arxiv.org/abs/2508.17590
- BIRD-INTERACT: https://arxiv.org/abs/2510.05318 , https://github.com/bird-bench/BIRD-Interact
- LiveSQLBench: https://livesqlbench.ai/ , https://github.com/bird-bench/livesqlbench
- CHESS: https://arxiv.org/abs/2405.16755 , https://github.com/ShayanTalaei/CHESS
- MAC-SQL: https://arxiv.org/abs/2312.11242
- RSL-SQL: https://arxiv.org/abs/2411.00073
- AutoLink: https://arxiv.org/abs/2511.17190
- SchemaGraphSQL: https://arxiv.org/abs/2505.18363
- Spider 2.0: https://arxiv.org/abs/2411.07763 , https://github.com/xlang-ai/Spider2
- ReFoRCE: https://arxiv.org/pdf/2502.00675
- OpenSearch-SQL: https://arxiv.org/abs/2502.14913
- AIE: https://arxiv.org/html/2412.20072v1
- AmbiSQL: https://arxiv.org/abs/2508.15276
- NL2SQL Ambiguity Recommender: https://arxiv.org/pdf/2505.19302
- SAFE-SQL: https://aclanthology.org/2025.emnlp-main.962.pdf
- RASL: https://assets.amazon.science/1b/95/8f62e89647348f4c4836f6c3040d/rasl-retrieval-augmented-schema-linking-for-massive-database-text-to-sql.pdf
