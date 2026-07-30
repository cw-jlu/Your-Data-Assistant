# DABench Public Tasks: Doc / knowledge.md テンプレート構造分析

> 50 task の `knowledge.md` と各 task が持つ `doc/*.md` (= narrative) の **テンプレート構造**を抽出し、
> 失敗 type との相関を整理。POC 実験 (= `scripts/test_doc_preprocess.py`) で得た境界条件も記録。

---

## 1. knowledge.md は **完全テンプレート** (= 100% 一致)

### 1.1 Top-level sections (6 つ、全 50 task で出現)

```
## 1. Introduction
## 2. Core Entities & Fields
## 3. Metric Definitions
## 4. Constraints & Conventions
## 5. Exemplar Use Cases
## 6. Ambiguity Resolution
```

各 section が全 50 task の knowledge.md に **必ず存在** する (= scaffolding がデータベース題材を変えても固定)。

### 1.2 Sub-section パターン

| 出現 sub-section | 出現率 |
|---|---:|
| `### Key Performance Indicators (KPIs)` | 100% |
| `### Potentially Ambiguous Fields` | 96% |
| `### Filtering Criteria` | 94% |
| `### Recommended Usage` | 78% |
| `### Calculation Logic` | 52% |
| `### Temporal Boundaries` | 40% |
| `### Currency Formatting` | 30% |
| `### Common Filters` / `### Common Conventions` | 28% |
| `### Unit Conversions` | 26% |
| `### Example N: <title>` (例示節) | 92% (= 5 個 × 46 task) |

### 1.3 Examples (= "Use Cases" or "Example N:") の構造

```markdown
### Example N: <Metric Title>
- **Metric** / **Natural Language**: <自然言語での metric 定義>
- **SQL**: `SELECT ... FROM ... WHERE ...` ← optional
- **Explanation**: <意義 / business context>
```

**重要な variation:** SQL block の有無

| 群 | task 数 | knowledge.md の Example の SQL 有無 |
|---|---:|---|
| 🟢 SQL example あり | 23/50 (= 46%) | 全 example に concrete SQL 記載 (= task_25, task_38 等) |
| 🟡 部分的 SQL | 8 | 一部 example のみ SQL 付き |
| 🔴 SQL example なし | 27/50 (= 54%) | metric 名 + 自然言語記述のみ (= task_169, task_180, task_396 等) |

### 1.4 Glossary (= Core Entities 内の用語定義)

各 entity section 配下に `- **term**: definition` 形式で **平均 28.3 用語**が定義されてる。

```markdown
### Customers
- **CustomerID**: Unique identifier for each customer.
- **Segment**: Classification of customers into segments (SME, LAM, KAM).
- **Currency**: Denotes the currency used by the customer.
```

→ agent は `_build_glossary_section` で先頭 800 chars 抽出して preamble に追加済。

### 1.5 Ambiguity Resolution section の存在

全 50 task に `## 6. Ambiguity Resolution` 節が存在。`### Potentially Ambiguous Fields` 形式で:

```markdown
- **name**: Always specify whether referring to event_name or member_name.
- **amount**: Clarify whether budgeted amount, spent amount, or funds received.
```

→ **gold は故意にこれら ambiguous field を使う傾向**あり (= task_25 の "cost", task_163 の "type")。

---

## 2. Knowledge.md の質と solve rate の相関

### 2.1 SQL example density × カテゴリ

| カテゴリ | avg SQL examples | avg λ0.5 |
|---|---:|---:|
| 🟢 always_solved (17) | 2.65 | 0.938 |
| 🟡 often_solved (15) | 2.33 | 0.835 |
| 🟠 variable (5) | 2.00 | 0.399 |
| 🔴 rarely_solved (9) | 2.22 | 0.126 |
| ⚫ never_solved (4) | **1.25** | 0.000 |

→ **never_solved の SQL example density が圧倒的に低い**(= 0.5x of always_solved)。

### 2.2 失敗 task は **質問にマッチする Example はあるが SQL がない** ことが多い

代表例 (task_169):

```
質問: "What was the average monthly consumption of customers in SME for the year 2013?"

knowledge.md:
  ### Example 2: Average Monthly Consumption for SME in 2013   ← 質問とほぼ完全一致!
  - **Metric**: Determine the average monthly consumption for SME customers in 2013.
  - **Explanation**: Provides insights into monthly consumption patterns...
                                                          ← SQL は伏せられてる
```

= 設計者は **意図的に "解答ヒント" Example を置きつつ、operational SQL を伏せて挑発**している構造。

### 2.3 Metric formula の operational ambiguity

LaTeX 数式が抽象的:

```latex
\[
  \text{Average Monthly Consumption} = \frac{\text{Total Annual Consumption}}{12}
\]
```

この `Total Annual Consumption` の **scope** (= 全 SME の合計 / per-customer 集計) が指定されていない:

- agent 解釈: `SUM(Consumption WHERE SME) / 12` = **82M** (= 全顧客集計)
- gold 解釈: `AVG(per-customer-yearly-total) / 12` = **460** (= per-customer)

→ agent の解釈は読み方として妥当だが、gold は別解釈。

---

## 3. Narrative docs (= `doc/*.md`) は **per-domain template**

### 3.1 Inventory (= 大型 narrative top 12)

| task | file | size_KB | 一行目スタイル |
|---|---|---:|---|
| task_418 | Laboratory.md | 280 | `### Hepatobiliary Enzyme Signatures...` |
| task_396 | superhero.md | 174 | `### The Vanguard Registry: A Comprehensive Catalog...` |
| task_415 | races.md | 90 | `### Chronological Performance Dossier...` |
| task_408 | races.md | 84 | `### Official Steward's Summary...` |
| task_418 | Patient.md | 83 | `### Patient Health Summary...` |
| task_349 | major.md | 73 | `MEMORANDUM...` |
| task_352 | budget.md | 61 | `### Strategic Review of Educational Program...` |
| task_350 | event.md / event_event.md | 57-59 | `### Annual Engagement & Operations Review...` |
| task_420 | legalities.md | 56 | `### Portfolio of Sanctioned Assets: A Legality Compendium` |
| task_344 | Patient.md | 54 | `### A Retrospective Cohort Analysis...` |
| task_379 | molecule.md | 36 | `### Computational Analysis and Carcinogenic Potential` |
| task_257 | League.md | ~20 | `### Continental Proving Grounds: A Strategic Overview` |

### 3.2 文体: prose narrative + 構造データ + decoy fluff

各 entity が **散文段落** で記述される。例 (= Patient.md):

```
The initial case under review pertains to the patient registered under
Case ID 43003. This individual is a male subject whose date of birth is
documented as November 24th, 1937. His clinical file was formally
initiated on the eighth of March, 1994, marking his entry into the
observational cohort. The patient's occupational history indicates a
long career in archival management, which he often cited as
contributing to a sedentary lifestyle. This subject's file also notes a
family history of hypertension on his paternal side.
```

= 構造抽出可能フィールド:
- ID = `43003`
- SEX = `male`
- Birthday = `November 24th, 1937` → ISO `1937-11-24`
- File initiated = `eighth of March, 1994` → `1994-03-08`

= 構造データ + **意図的な decoy fluff** (= occupational history, hobbies, family medical history) が入り混じる。

### 3.3 per-domain template variation

| domain | 各 entity の field 出現順 | 文体特徴 |
|---|---|---|
| **Patient.md** | ID → SEX → Birthday → file_init → fluff | 医学コホート分析体 |
| **superhero.md** | name → registration number → full name (or unconfirmed) → operational history | スパイ風 dossier 体 |
| **races.md** | name → date → venue → outcomes | championship report 体 |
| **budget.md** | budget_id (= alphanumeric) → amount → spent → remaining → campaign | 財務 audit 体 |
| **event.md** | event_name → date → status → notes | 業務 brief 体 |

→ 各 doc は **同 domain 内で template ベース**だが、domain を跨ぐと完全に別 format。**汎用 extractor は不可、domain 別 parser が必要**。

### 3.4 Regex 抽出の reliability (= task_344 Patient.md)

| field | regex 検出率 |
|---|---:|
| ID (= numeric, "Case ID NNN" or "Patient NNN") | 100% (= ID 自体は確実) |
| SEX (= "male" / "female" 直接表現) | **48%** (残りは "a man", "individual" 等の variant) |
| Birthday (= "born on", "date of birth") | **47%** (残りは "second quarter of 1937" 等の indirect 表記) |
| File init date | **29%** (= 多様な phrasing) |

→ **regex 単独で 50% 抽出** + **LLM で残り 50% を逐次抽出** が現実的攻略。

### 3.5 narrative doc の challenge: 重要情報の **不在**

task_344 (= 280KB Patient.md + 54KB Laboratory.md) では:
- patient demographics (= ID, SEX) は narrative にあり抽出可能
- **WBC / Fibrinogen の "normal range" 数値閾値**は **どの doc にも記載なし**

→ **doc を完璧に抽出しても解けない外因 task** が存在 (= Type G の根本)。
   → 医学共通知識 (= WBC normal 4-9, Fibrinogen 1.5-4.0 g/L) の **外部 reference 必須**。

---

## 4. POC 実験で得た境界条件 (= 2026-05-07)

`scripts/test_doc_preprocess.py` で task_169 / task_344 に対し 2 種類の preprocessing を試行:

### 4.1 Pass 1: Knowledge disambiguator (= LLM call で knowledge.md を operational SQL に展開)

**Result:** task_169 で **wrong interpretation を高自信で commit**。
- LLM 出力: `SELECT SUM(annual_total) / 12 ...` (= 全顧客集計、間違い)
- gold formula: `AVG(per-customer-yearly-total) / 12` (= per-customer)

**境界条件 (= 学んだ前提):**
- knowledge.md の inherent ambiguity は **LLM 再読込では解消しない**
- LLM は同じ情報源から同じ判断 → 同じ間違いを**自信を持って** commit する
- → **agent の不確実性を奪い、wrong direction を強化**するリスク
- = Pass 1 単独は **無価値〜逆効果**

### 4.2 Pass 2: Narrative structurer (= LLM call で narrative → CSV 抽出)

**Result:** task_344 Patient.md (= 80KB cap) で **CSV header のみ生成、データ行ゼロ**。

**境界条件:**
- 80KB narrative + 32K-128K thinking budget でも **LLM 単発抽出は完成しない**
- 抽出が partial / 完全失敗、agent は augmented preamble の空テーブルを参照しても役立たず
- = 単発 LLM call では narrative は extract できない、**chunked / iterative** な抽出戦略が必須

### 4.3 結論

両アプローチとも **0.0 → 0.0** で改善せず。doc preprocessing は naive 設計では機能しない。

**しかし学んだこと (= 構造的 insight):**

- knowledge.md は 100% templated (= 6 section + Example 構造)
- narrative docs は per-domain templated (= regex で半分抽出可能)
- **真の障壁は LLM の再読込能力ではなく、原始情報側の operational specificity 不足**

---

## 5. 失敗 type と doc 性質の対応関係

| Failure Type | doc 側の性質 | 対処の可否 |
|---|---|---|
| **B 計算ロジック誤り** (task_169/396/408) | knowledge.md の Example が question 一致するが **SQL なし** | 🟡 LLM disambiguator では救えない (= Pass 1 失敗で確認)、**多解釈 union** が現実的 |
| **C Filter scope** (task_180/199) | knowledge.md が解釈の余地を残す | 🟡 同上、解釈 enumeration 必要 |
| **E 質問解釈 bias** (task_163/25/173) | Ambiguity Resolution 節で "potentially ambiguous fields" を予告するが、agent はそれを系統的に活用していない | 🟢 ambiguity resolution 節を **強化注入** (= prompt に固定で出す) で改善可能性 |
| **F narrative doc parse** (task_344/396/418) | 80-280KB prose narrative、per-domain templated だが 50% は variant 表現 | 🟢 hybrid regex + LLM iterative chunked extraction で攻略可能 |
| **G 外部知識 gap** (task_344/418) | 必要情報が **doc にも knowledge.md にもない** | 🔴 外部 reference embed (= 医学/業界閾値 DB) 大改造 |

---

## 6. 実装可能性の優先順位 (= 2026-05-07 知見アップデート版)

### Tier 1: 既に実装済 / 軽微改修

- ✅ Rich preamble の per-col profile (= exp_101)
- ✅ Padding fix, column verify, row_count 撤廃 (= exp_109)

### Tier 2: 新規だが構造的に実装可能

| 軸 | 期待 +λ | 実装コスト |
|---|---|---|
| **Ambiguity Resolution 節の強制注入** (= 既存 knowledge.md 内容を prompt 固定セクション化) | +0.01〜+0.03 | 低 (~2 hours) |
| **Narrative doc hybrid extractor** (= regex + LLM chunked) | +0.02〜+0.04 | 中 (~1-2 day) |
| **多解釈 union** (= Type B/C 救済、Phase 1 plan で 2-3 候補生成 → 並列実行) | +0.03〜+0.05 | 中 (~2 day) |

### Tier 3: 大改造 / 外部依存

| 軸 | 期待 +λ | コスト |
|---|---|---|
| 医学/業界 knowledge embed (= Type G) | +0.02 | 高 (~3 day) |
| Examples の SQL を LLM 補完してから query 実行 | +0.01 | 中 (= question-matching example だけ補完) |

### POC で **実装すべきでない** と判明

- ❌ Knowledge.md disambiguator (Pass 1) — LLM の wrong commit 助長
- ❌ Single-shot narrative extraction (Pass 2) — 物理的に不可能

---

## 7. doc 性質 × success rate の **fine-grained 相関分析**

50 task の knowledge.md / question / narrative-doc 性質を bool / numeric features に encode、mean λ0.5 と相関を取る。

### 7.1 強い相関を持つ feature

| feature | True 群 mean | False 群 mean | Δ |
|---|---:|---:|---:|
| **q_has_filter** (質問に "where/among/only/than/between/during" 含む) | 0.518 (n=16) | 0.686 (n=34) | **-0.168** ← 最大の負相関 |
| **n_sql_examples ≥ 4** (knowledge.md に concrete SQL example が多い) | 0.695 (n=23) | 0.578 (n=27) | **+0.117** ← 最大の正相関 |
| **has_narrative_doc** (大型 narrative 持ち) | 0.575 (n=12) | 0.650 (n=38) | -0.075 |
| **has_question_matching_example** (= 質問と Example title が strong match) | **0.490 (n=5)** | 0.648 (n=45) | **-0.158** ← 逆相関、trap 仕掛け確認 |

### 7.2 解釈

#### **q_has_filter** が最大ネック

multi-condition filter (= "among X who are Y, count Z") を要する質問が **16/50 (32%)** あり、平均 0.52 と低い。

該当 task の代表:
- task_180 ("among people who paid more than 29.00 per unit of product 5, give August 2012 consumption")
- task_344 ("among male patients with normal WBC, count abnormal fibrinogen")
- task_418 ("among patients with abnormal creatinine, count those under 70")

= **Type C (filter scope) + B (calc) + E (interpretation) が並列発生する難 task 群**。

#### **question-matching Example は trap**

knowledge.md に質問と一致する Example title があるタスク (= task_25, task_38, task_169, task_259, task_408) で:
- mean = 0.490 (= 全体平均 0.633 から -0.143)
- 設計者が "答えに見せて SQL を伏せた" 構造を意図的に置いてる

→ agent が Example のタイトルだけ参照して **abstract metric を盲信**するリスク。

#### **SQL example 多量 + filter なし** が成功条件

```
最高成績群: SQL 4+ examples + no filter language (= aggregate/lookup 系)
        → mean ~0.80 (= always_solved の代表パターン)

最悪群: SQL 0件 + filter language あり + narrative doc 大型
       → mean ~0.10 (= never_solved + 一部 rarely_solved)
```

### 7.3 改善軸への含意 (= 相関分析からの逆算)

**filter 系 task の救済が最大 ROI:**
- 16 task × 平均 -0.168 の gap = 全体 score への寄与 -0.054 程度
- これを半分埋めれば **+0.027 全体 score 改善** 可能

具体策候補:
1. **Multi-condition filter の verifier** = "filter A, then within those filter B, then aggregate C" を逐次検証
2. **Per-step row count 観測** = filter 後に row 数が想定通りかチェック
3. **Filter 解釈の多重化** = filter 順序や境界条件 (= inclusive vs exclusive) を 2-3 通り試す

**question-matching Example の trap を逆手に取る:**
- 質問と Example title の semantic overlap を検出した task では、agent に **明示警告**:
  > "Example N has a similar title but no SQL. Do NOT assume the SQL form. Derive from data."
- → 抽象 metric の盲信を抑制。

---

## 8. 結論

### 8.1 doc 構造の特徴

1. **knowledge.md は 100% templated** (= 6 section + Example 構造)、ただし **SQL example 有無で質に variation**
2. **narrative docs は per-domain templated**、prose generation + decoy fluff、hybrid 抽出で 50%+ 可能
3. **設計上の挑発 (= question-matching Example で SQL を伏せる)** が一部 task の難度を意図的に上げてる

### 8.2 相関分析が示す主要ネック

1. **filter 系質問** (= -0.168 全体寄与 -0.054): 救済の ROI 最大
2. **narrative doc 大型 task** (= -0.075): hybrid extract で部分救済可能
3. **question-matching Example トラップ** (= -0.158、5 task のみ): 警告で抑制可能
4. **SQL example の不在** (= -0.117): 多解釈 union or 外部 SQL 補完で対応

### 8.3 真の障壁

LLM 再読込 (= POC で検証) では doc の inherent ambiguity / 不完全性は解消しない。**操作可能な領域** は:

- ✅ **prompt-level**: ambiguity-resolution 節の固定注入、question-Example overlap の警告
- ✅ **runner-level**: hybrid regex + LLM chunked extraction for narratives
- ✅ **multi-attempt**: 解釈空間の enumeration + union (= 単 attempt の自由度を信頼性に置換)

これら Tier 2 が最も現実的、合計 **+0.04〜+0.08** の改善余地と推定。

---

## 9. Deep Dive 続編 (= 追加 fine-grained 分析)

### 9.1 Gold answer header の SQL 形式露出

50 task の gold.csv header を分析:

| header pattern | 出現数 | 意味 |
|---|---:|---|
| plain column (= `event_name`, `count` 等) | 40 | 単純 lookup / count |
| `COUNT(...)` | 12 | aggregate |
| `multi_col` (= 2+ col) | 10 | join 結果 |
| `CAST(... AS REAL)` | 8 | percentage / ratio |
| `arithmetic / N` (= `/ COUNT`, `/ 12`) | 8 | 二段集計 |
| `CASE WHEN` | 5 | conditional count |
| `arithmetic *100` | 5 | percentage |

**Complex header (= CAST/CASE/arithmetic) を持つ task の category 別分布:**

| category | total | complex_header | 解ける率 (complex) | 解ける率 (simple) |
|---|---:|---:|---:|---:|
| always_solved | 17 | 1 | 0.956 | 0.937 |
| often_solved | 15 | 5 | 0.812 | 0.846 |
| variable | 5 | 1 | 0.456 | 0.384 |
| rarely_solved | 9 | 1 | 0.008 | 0.141 |
| never_solved | 4 | 1 | 0.000 | 0.000 |

→ complex header が必ずしも難度に直結しない (= often_solved にも 5 件)。
→ **gold は SQL 形式を露出してるが eval は値マッチ**ベース、column header の不一致は致命的でない。

### 9.2 Database-level 内変動 (= same DB, different question)

同じデータベース・schema・knowledge.md なのに solve rate が劇的に違う = **data 問題ではなく質問構造**。

| database | tasks | avg score | range | 失敗 task |
|---|---:|---:|---:|---|
| **student_club** | 12 | 0.757 | **0.97 → 0.00** (= range 0.97!) | task_163 (= type 解釈), task_25 (= ties) |
| **formula_1** | 9 | 0.634 | 0.97 → 0.01 | task_80/89 (= ties / column 名) |
| **superhero** | 8 | 0.805 | 0.97 → 0.01 | task_396 (= height range + percentage) |
| **debit_card** | 3 | 0.097 | 0.29 → 0.00 | 全 3 task: 169 / 173 / 180 |
| **toxicology** | 4 | 0.494 | 0.92 → 0.21 | 379 (= 4th atom), 200 (= triple-bond) |
| **thrombosis_pred** | 3 | 0.253 | 0.70 → 0.00 | 344 (= WBC 閾値), 418 (= 創傷年齢) |
| **california_schools** | 2 | 0.473 | 0.92 → 0.02 | 199 (= Riverside multi-condition) |
| **the (= stack overflow)** | 5 | 0.729 | 0.91 → 0.24 | 259 (= comment max in view range) |

→ **toxicology, debit_card, thrombosis** はデータベース全体が難 (= avg ≤ 0.5)。
→ **student_club** は 12 task 中 10 が成功、2 が失敗 (= 質問構造で別れる)。

### 9.3 質問 opening pattern × solve rate

| opening | n | mean λ0.5 |
|---|---:|---:|
| **How many ...** | 4 | **0.901** |
| What is ... | 12 | 0.699 |
| **list / provide / give** | 7 | 0.724 |
| **Among X, ...** | 7 | 0.570 |
| **For X, ...** | 4 | 0.615 |
| **identify / calculate / tally** | 6 | 0.542 |
| **Which X has Y** | 3 | **0.374** |
| **In X, ...** | 1 | 0.008 |

→ **"How many"** = 単純 scalar count = 高 solve rate
→ **"Which X has Y"** = ties + 単一値曖昧性 = 困難 (= task_25/86/250 等)
→ **"identify/calculate/tally"** = verb 自体 ambiguous = 困難

#### "Among X" 群の細分

```
解ける (0.85+): "Among X with attribute Y who attended Z" (= 単純 join + filter)
   - task_303 (Among European Grand Prix races, percentage of Germany hosted) = 0.90
   - task_350 (Among students who attended Women's Soccer, T-shirt size) = 0.92
   - task_145 (Among events with > 10 members, count meetings) = 0.95

解けない (0.0): "Among X with abnormal/normal Y"  (= 医学/業界閾値)
   - task_344 (Among male with normal WBC, count abnormal fibrinogen) = 0.00
   - task_418 (Among patients with abnormal creatinine, count age < 70) = 0.06
   - task_396 (= "In superheroes with height between 150 to 180...") = 0.01
```

→ **数値閾値が exposed か否かが分水嶺**。具体数値 (= "more than 10") なら解ける、抽象 (= "abnormal") なら解けない。

### 9.4 Ambiguity Resolution section の **未活用** 問題

knowledge.md の section 6 "Ambiguity Resolution" は 50/50 task で存在し、設計者が把握してる ambiguity を列挙してる。失敗 task で警告内容を確認:

| task | 失敗の真因 | section 6 で警告されてるか |
|---|---|---|
| task_25 (= cost 解釈) | "cost" を expense.cost / budget.amount のどちらか曖昧 | ✅ **YES** — "amount: budgeted amount, spent amount, or funds received" と明示 |
| task_169 (= scope) | "Total Annual Consumption" の scope (= 全集計 vs per-customer) | ❌ NO — Date format / Segment は警告だが scope は未言及 |
| task_344 (= 医学閾値) | WBC/Fibrinogen normal/abnormal の数値 | ❌ NO — Diagnosis vs Disease 等は警告だが threshold 未言及 |
| task_396 (= height + percentage) | 範囲解釈 | ❌ NO — 関連 entity の曖昧性は警告だが質問固有は未 |
| task_163 (= "type of expenses") | event の type vs expense category | ❌ NO — section 6 は別の field を扱う |

→ **task_25 は唯一 section 6 が当該 ambiguity を直接警告してる**が、agent はそれを系統的に enumeration して試してない。
→ **section 6 を prompt-level で agent に強制注入** (= "Before answering, identify if any question term matches Ambiguity Resolution flagged terms; if so, enumerate alternatives") は task_25 で +0.5 救済の可能性。

### 9.5 失敗 task の 5 ケース詳細

#### Case 1: task_169 (= 0.0, scope ambiguity)
```
Question:    "What was the average monthly consumption of customers in SME for the year 2013?"
Gold:         AVG(T2.Consumption) / 12 = 459.96
Agent:        SUM(annual_total) / 12 = 82M  (= scope 取り違え)
KM section 6: Date format / Segment / Currency  → **scope 未言及**
KM example 2: "Average Monthly Consumption for SME in 2013" タイトル一致だが SQL なし
TRAP:         knowledge.md は "Total Annual Consumption" と書くが per-customer 集計が gold
```
→ 救済策: **多解釈 union** (= AVG vs SUM scope 両方試す) + **オーダー sanity** (= 82M vs 460 で 5 桁ずれ → 警告)

#### Case 2: task_25 (= 0.04, cost interpretation + ties)
```
Question:    "Which event has the lowest cost?"
Gold:         3 行 tied (November/October/September Speaker)
Agent:        3 行 tied (Officers meeting series)  ← cost=0 の events を 含めた
KM section 6: ✅ "amount: budgeted, spent, or funds received" 明示
TRAP:         "lowest" + ambiguous "cost" + ties = 3 重ハードル
```
→ 救済策: **section 6 注入 + 多解釈 union** (= "lowest cost" を "lowest budget.amount" と "lowest sum(expense.cost)" で並列実行)

#### Case 3: task_344 (= 0.0, missing thresholds)
```
Question:    "Among male patients with normal WBC, how many have abnormal fibrinogen?"
Gold:         4
Agent:        0 (= 過剰フィルタ) or 50+ (= filter 失敗) 
KM:          LDH > 500 / UA > 8.0 等は記載、WBC/FG 未記載
narrative:   Patient.md (54KB) は demographics narrative、検査値 narrative なし
Lab.md:      実は Laboratory.md (= task_418 が持つが task_344 は持ってない)
TRAP:         必要情報が doc にない (= Type G)
```
→ 救済策: **外部医学知識 embed** (= "WBC normal 4-9 ×10⁹/L, FG normal 1.5-4.0 g/L") prompt 固定追加

#### Case 4: task_180 (= 0.0, multi-condition filter scope)
```
Question:    "For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in August 2012."
Gold:         9 行 × 1 col (Consumption only)
Agent:        153 行 × 2 col (CustomerID + Consumption)  ← filter scope 緩い + 余分 col
KM section 6: Date YYYYMM 警告のみ
TRAP:         "per unit" = transactions.Price 列、これが pivot point
```
→ 救済策: **filter scope verifier** (= 想定 row count vs 実 row count 比較) + column scope verify (Rule 18 強化)

#### Case 5: task_163 (= 0.0, "type" 解釈 bias)
```
Question:    "Identify the type of expenses and their total value approved for 'October Meeting' event."
Gold:         1 行 ("Meeting", 175.39)
Agent:        3 行 (Pizza, Posters, Water+chips)
KM section 6: name/amount/date 等は警告だが "type" は未
TRAP:         "type of expenses" = expense_description ではなく event.type (= "Meeting")
              これは単なる counter-intuitive な gold-side bias
```
→ 救済策: **多解釈 union** (= event.type vs expense_description 両方試す)

### 9.6 改善軸の優先 (= 深掘り後の確定版)

| 優先 | 軸 | 救済対象 | 期待 +λ | 実装コスト |
|---|---|---|---|---|
| **1** | **多解釈 attempt union** (= scope/type 曖昧性を 2-3 通り並列実行) | task_25/163/169/180 + 部分救済他 | +0.04〜+0.07 | 中 (= runner 改修 + plan diversification) |
| **2** | **Ambiguity Resolution 強制注入** (= section 6 を prompt 固定で出す) | task_25 確実、他は probabilistic | +0.01〜+0.03 | 低 (= preamble 改修 1h) |
| **3** | **filter scope verifier** (= row count 想定値との比較) | task_180/344/418 部分 | +0.01〜+0.03 | 中 (= verifier rule layer) |
| **4** | **外部知識 embed** (= 医学/業界閾値) | task_344/418 | +0.02〜+0.04 | 高 (= per-domain knowledge curation) |
| **5** | **narrative hybrid extract** (= regex + LLM chunked) | task_344/396 部分 | +0.02 | 中-高 |

合計上振れ **+0.10〜+0.19**、現実的見込み **+0.05〜+0.10**。LB transfer 94% で **LB +0.05〜+0.09 = 4-3 位射程**。

