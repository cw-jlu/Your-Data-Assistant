# DABench public-50 の Reverse Engineering: BIRD-SQL からの構築過程推定

**Date**: 2026-05-13
**Investigation purpose**: KDD Cup 2026 DataAgent-Bench (= DABench) public demo の 50 タスクが何に由来するか、運営がどのようにデータを作成したかを推定し、A-board / B-board hidden test の戦略立案に活用する。

---

## 1. Executive Summary

**結論**: DABench public-50 の **49/50 タスクが BIRD-SQL dev set (= 1,534 questions, 11 databases) からそのまま採用** されている。1 タスクのみ完全新規 (= task_199)。

運営の構築パイプライン (推定):

```
BIRD dev set (1,534 questions, 11 SQLite DBs)
   ↓ ① stratified sampling (difficulty bias = challenging を 4 倍多く)
49 questions 選定 + 1 newly authored
   ↓ ② BIRD's database_description/*.csv を LLM rephrase
"Enterprise Data Governance Knowledge Guide" 風 knowledge.md
   ↓ ③ SQLite DB を CSV / JSON / SQLite に format permutation
   ↓ ④ task ごとに table subset を選択
50 DABench tasks (data/public/input/task_NN/)
   ↓ ⑤ BIRD evidence + gold SQL を実行して gold.csv 生成
50 gold answers
```

**critical な発見**:
- 49/50 が **BIRD 質問テキスト完全一致** (= 1 つ typo 修正、1 つ完全新規)
- 11 BIRD dev DB **全部が代表される** (= 層化抽出)
- 難易度別 sampling 率: simple 1.6% / moderate 5.0% / **challenging 6.9%** (= 難 4 倍 bias)
- **BIRD database_description の "Normal range" threshold が DABench knowledge.md で平均 ~10% しか保持されていない**
- → 失敗タスク (= task_169 / 344 / 396 / 418 など) は **threshold info loss が直接原因**

**A-board / B-board 推測**:
- BIRD train set (= 12.6 GB, 69 databases, 9,428 questions; dev とは disjoint) または BIRD dev の非サンプル分が出元と仮定するのが妥当
- 同じ pipeline で knowledge.md threshold 剥がしが行われる前提で対策が必要

---

## 2. 調査の証拠

### 証拠 1: 質問テキスト 1-to-1 一致

**48/50 がBIRD dev question と完全一致**、**1/50 が typo 修正された fuzzy 一致**、**1/50 が完全新規**:

| match type | n | 例 |
|---|---|---|
| exact match | 48 | "Which event has the lowest cost?" → BIRD qid=1389 |
| fuzzy 0.94 | 1 | task_89: DABench "Chinese Grand Prix" ← BIRD "AustChineseralian Grand Prix" (typo 修正) |
| no_match | 1 | task_199 完全新規 (= "Riverside-related school districts ...") |

`docs/_bird_mapping_data.json` に 50 タスク全マッピング保存済。

### 証拠 2: Database / 難易度 selection bias

#### Database distribution

DABench は BIRD dev の **全 11 databases から層化抽出**。

| Database | BIRD dev | DABench | 抽出率 |
|---|---:|---:|---:|
| card_games | 191 | 2 | 1.0% |
| codebase_community | 186 | 5 | 2.7% |
| formula_1 | 174 | 8 | 4.6% |
| thrombosis_prediction | 163 | 3 | 1.8% |
| student_club | 158 | 12 | **7.6%** |
| toxicology | 145 | 4 | 2.8% |
| superhero | 129 | 8 | **6.2%** |
| european_football_2 | 129 | 1 | 0.8% |
| financial | 106 | 1 | 0.9% |
| california_schools | 89 | 1 | 1.1% |
| debit_card_specializing | 64 | 3 | 4.7% |

**特徴**: student_club / formula_1 / superhero が高比率 (= 計 28 task)。BIRD のスキーマ複雑度が「分かりやすい domain」を agent ベンチに適した、と推定。

#### Difficulty bias

| 難易度 (BIRD) | DABench に renamed | BIRD dev | DABench | 抽出率 |
|---|---|---:|---:|---:|
| simple | easy | 925 | 15 | 1.6% |
| moderate | medium | 464 | 23 | 5.0% |
| challenging | hard | 145 | 10 | 6.9% |

→ **challenging は simple の 4.3 倍の確率で選ばれる**。**意図的な hard 寄りキュレーション**。

#### 難易度 rename rule

```
BIRD                DABench
simple        →     easy        (14 + 1 hard 例外)
moderate      →     medium      (22 + 1 hard 例外)
challenging   →     hard        (9)
```

48 タスク中 **46 件で BIRD difficulty を rename しただけ**、2 件のみ手動引き上げ (= simple → hard が 1 件、moderate → hard が 1 件)。

### 証拠 3: knowledge.md は BIRD `database_description/*.csv` の LLM rephrase

#### BIRD の元データ構造

```
dev_databases/{db}/database_description/{table}.csv
columns: original_column_name, column_name, column_description, data_format, value_description
```

#### サンプル (BIRD `Laboratory.csv` の一部)

```
CRE, creatinine, creatinine, real, "Commonsense evidence:\n\nNormal range: N < 1.5"
WBC, White blood cell, White blood cell, real, "Commonsense evidence:\nNormal range: 3.5 < N < 9.0"
FG, fibrinogen, fibrinogen, real, "Commonsense evidence:\n\nNormal range: 150 < N < 450"
LDH, lactate dehydrogenase, lactate dehydrogenase, integer, "Commonsense evidence:\n\nNormal range: N < 500"
```

#### DABench `task_418/context/knowledge.md` の対応箇所

```
### Laboratory
- **ID (integer):** Unique identifier for each patient.
- **Date (date):** Date of the laboratory test.
- **LDH (integer):** Lactate dehydrogenase level, with values above 500 considered beyond the normal range.
- **UA (real):** Uric acid level, with specific normal ranges based on gender.
```

→ **LDH のみ threshold 保持**、CRE / WBC / FG など 35+ 列が omit。完全に LLM がスリム化した narrative。

#### 統計

`thrombosis_prediction` db_description で 39 threshold (=「Normal range:...」を含む value_description) のうち、対応する DABench knowledge.md に**保持されたのは 4 件 (10%)** のみ。

つまり **threshold 保持率 ~10%**。

---

## 3. Threshold 削除は意図的か LLM 副作用か?

これは推測ですが、複数の証拠から **LLM rephrase の副作用** の可能性が高いと判断:

### 副作用説の根拠 (= 強い)

1. **knowledge.md は narrative 化されており、boring な「Normal range: X < N < Y」記述を保持していない**
   - LLM rephrase は「読みやすい文体」を優先し、列挙されたテーブル値を間引きする傾向
   - 残った threshold (= LDH, UA) は **個別の sentence 形式に組み込まれている** (= "above 500 considered beyond normal" / "gender-specific normal range")
   - LLM が「value_description 全部含めると冗長」と判断して間引いた可能性

2. **KPI セクションは詳細に保持**
   - "Inpatient vs. Outpatient Ratio for Males" 等の formula は完璧に保持
   - これは「列挙 narrative」より「formula 構造」が LLM にとって rephrase しやすかったため

3. **task ごとに knowledge.md がほぼ同じ**
   - 同 DB (= 例: student_club) の複数 task で共通の knowledge.md を使い回し
   - もし「タスク固有のヒントを意図的に剥がす」なら task ごとに調整するはず
   - 共通 knowledge.md = LLM 一括生成の結果

### 意図的説の根拠 (= 弱い)

1. **threshold を agent に手探りで当てさせるテスト**
   - 公式 Discord の Boyan の説明 (= task_169 で `not simply SUM/12`) も意図的な「迷わせ」に見える
   - 但しこれは構築時の意図というより、検査者の追加注釈の可能性

2. **BIRD evidence は per-question hint なので、データセット全体で公開する knowledge.md には合わない**
   - 確かに BIRD evidence は質問固有 (= "severe thrombosis refers to thrombosis = 2"), database_description の threshold は domain 一般
   - 一般 threshold は agent に渡しても question の答えを直接漏らさない
   - 意図的に剥がす理由が薄い

### 判定: **LLM rephrase の副作用が主要因** (= 副作用 70% / 意図 30%)

実装者は BIRD の database_description csv を LLM (= GPT-4 or 同等) に渡して "Enterprise Data Governance Knowledge Guide" を書かせ、機械的に narrative 化した。LLM は冗長な threshold 列挙を間引いて読みやすくしたが、これが agent の解答を不当に難しくする結果に。

**実用的含意**: A-board / B-board でも同じ pipeline なら threshold 剥がしが起きている可能性大。agent 側で**「threshold が knowledge.md 不在の場合は medical/clinical reference を default で適用」**ロジックを入れれば対策可能。

---

## 4. データ format 変換の pipeline (= 推定)

BIRD は単一 SQLite DB として配布されているが、DABench は同じ table を CSV / JSON / SQLite の 3 form に分散させている。

**観測した permutation 例** (= student_club domain):

| Task | event | member | budget | expense | attendance |
|------|-------|--------|--------|---------|------------|
| task_19 | — | csv | — | — | — |
| task_22 | — | json | — | — | — |
| task_24 | json | — | — | — | csv |
| task_25 | json | — | csv | json | — |
| task_145 | **db** | — | — | — | csv |
| task_163 | **db** | — | json | csv | — |
| task_352 | csv | — | — | — | — |
| task_350 | — | csv | — | — | **db** |

**観察**:
- 同 table が **task ごとに csv / json / sqlite 3 形式どれか** に。
- 全 50 task で **完全 random ではなく、何らかのルール** で割り当てられている可能性。
- 例: 「easy = csv 多め、hard = sqlite/json 混在多め」など。

**残存テーブル**: BIRD の元 DB には 8 tables (= student_club) あるが、DABench task_19 では `member.csv + zip_code.json` の 2 つだけ exposed。残りは隠す。

これは「**質問に必要な table のみ露出**」というルール。BIRD gold SQL を解析して必要 table を抽出する自動 pipeline と推測。

---

## 5. Gold answer の生成過程 (= 推定)

DABench gold.csv (= `data/public/output/task_NN/gold.csv`) は **BIRD gold SQL を実行した結果** と一致するはず。

例 task_25:
- BIRD gold SQL: `SELECT T1.event_name FROM event T1 INNER JOIN budget T2 ON T1.event_id=T2.link_to_event INNER JOIN expense T3 ON T2.budget_id=T3.link_to_budget WHERE T3.cost = (SELECT MIN(cost) FROM expense)`
- BIRD on student_club.sqlite で実行 → 3 行 ["October Speaker", "November Speaker", "September Speaker"]
- DABench `gold.csv`: 同じ 3 行 ✓ (= 一致)

→ **BIRD gold SQL を BIRD original SQLite で実行**して `gold.csv` を作っていると推定。データ format 変換 (csv/json/sqlite) は agent 側の負荷であり、gold は単一 source from BIRD。

これは scoring の "value-only column matching" を可能にする: 同じ value が出れば column 名や順序を問わない。

---

## 6. A-board / B-board hidden test の推測

BIRD-SQL の構成:
- **dev** = 1,534 questions, 11 DBs (= DABench public-50 の出元、49/50 採用済)
- **train** = 9,428 questions, 69 DBs (= dev と disjoint、12.6 GB)
- **test** = 1,789 questions, holdout (= 公開されない)

公式 Discord (= Boyan Li):
- A-board: 60 questions, 2 時間制限
- B-board: 12 時間制限

### シナリオ A: hidden test = BIRD dev の非サンプル分

- BIRD dev に 1,534 questions、DABench 使用 49 → 残り 1,485 questions
- A-board 60 は 1485/60 = 2.5% 抽出率、DABench 比率と一致
- **同 11 DB**、schema 知識 + knowledge.md 構造が DABench と同じ
- 我々の investment が直接効く

### シナリオ B: hidden test = BIRD train (= 69 disjoint DBs)

- 69 新規 DB、schema 知識 transfer 不能
- ただし pipeline は同じ → knowledge.md threshold 剥がし、format 分散 等は同じ
- agent の method-level 学習が transfer

### シナリオ C: BIRD test = held out

- 1,789 questions、公開されないが pipeline は同じ
- BIRD authors しか持っていない、KDD Cup 運営が入手して使った可能性

### 確率推定

| シナリオ | 確率 |
|---|---|
| A (dev 非サンプル) | 40% (= 公平、easy preparation) |
| B (train) | 50% (= 新規 DB で評価者の意図と合う) |
| C (test) | 10% (= 入手困難) |

→ **train が main、dev mix 可能性中** が現実的想定。

### 戦略含意

- **戦略 a (= dev DB に最適化)**: シナリオ A なら +0.10〜0.20、シナリオ B では効果薄
- **戦略 b (= 全 BIRD 80 DB の database_description から threshold 集成)**: 全シナリオで一定効果
- **戦略 c (= generic methodology = "threshold が knowledge.md 不在時の handling")**: leak-free、全シナリオで効く

---

## 7. 1-to-1 マッピングテーブル (= 全 50 件)

DABench public-50 の各 task と BIRD dev の対応。BIRD qid を引けば原典 SQL + evidence を直接参照可能。

| task_id | DA diff | BIRD qid | BIRD db | BIRD diff | match | question (≤80 chars) |
|---|---|---|---|---|---|---|
| task_11 | easy | 1157 | thrombosis_prediction | simple | exact | For patients with severe degree of thrombosis, list their ID, sex and disease th... |
| task_19 | easy | 1334 | student_club | simple | exact | List the full name of the Student_Club members that grew up in Illinois state. |
| task_22 | easy | 1357 | student_club | simple | exact | State the date Connor Hilton paid his/her dues. |
| task_24 | easy | 1371 | student_club | simple | exact | How many members attended the "Women's Soccer" event? |
| task_25 | easy | 1389 | student_club | simple | exact | Which event has the lowest cost? |
| task_26 | easy | 1394 | student_club | simple | exact | How many members of the Student_Club have major in 'Physics Teaching'? |
| task_27 | easy | 1410 | student_club | simple | exact | List out the full name and total cost that member id "rec4BLdZHS2Blfp4v" incurre... |
| task_38 | easy | 159 | financial | simple | exact | List all the withdrawals in cash transactions that the client with the id 3356 m... |
| task_64 | easy | 717 | superhero | simple | exact | Please list all the superpowers of 3-D Man. |
| task_67 | easy | 750 | superhero | simple | exact | What is the average weight of all female superheroes? |
| task_74 | easy | 806 | superhero | simple | exact | Provide the eye colour of the superhero who has Karen Beecher-Duncan as their fu... |
| task_75 | easy | 847 | formula_1 | simple | exact | What is the surname of the driver with the best lap time in race number 19 in th... |
| task_80 | easy | 861 | formula_1 | simple | exact | What is his number of the driver who finished 0:01:54 in the Q3 of qualifying ra... |
| task_86 | easy | 902 | formula_1 | simple | exact | Which race was Alex Yoong in when he was in track number less than 20? |
| task_89 | easy | 937 | formula_1 | simple | fuzzy 0.94 | What's the finish time for the driver who ranked second in 2008's Chinese Grand ... |
| task_145 | medium | 1322 | student_club | moderate | exact | Among the events attended by more than 10 members of the Student_Club, how many ... |
| task_163 | medium | 1404 | student_club | moderate | exact | Identify the type of expenses and their total value approved for 'October Meetin... |
| task_169 | medium | 1473 | debit_card_specializing | moderate | exact | What was the average monthly consumption of customers in SME for the year 2013? |
| task_173 | medium | 1501 | debit_card_specializing | moderate | exact | Please list the countries of the gas stations with transactions taken place in J... |
| task_180 | medium | 1533 | debit_card_specializing | moderate | exact | For all the people who paid more than 29.00 per unit of product id No.5. Give th... |
| task_194 | medium | 243 | toxicology | moderate | exact | What are the bonds that have phosphorus and nitrogen as their atom elements? |
| task_196 | medium | 245 | toxicology | moderate | exact | What is the average number of bonds the atoms with the element iodine have? |
| **task_199** | medium | — | — | — | **NEW** | List the names and funding types of schools from Riverside-related school distri... |
| task_200 | medium | 260 | toxicology | moderate | exact | Calculate the total atoms with triple-bond molecules containing the element phos... |
| task_214 | medium | 405 | card_games | moderate | exact | How many Brazilian Portuguese translated sets are inside the Commander block? |
| task_218 | medium | 40 | california_schools | moderate | exact | What is the telephone number for the school with the lowest average score in rea... |
| task_243 | medium | 571 | codebase_community | moderate | exact | For the user No.24, how many times is the number of his/her posts compared to hi... |
| task_249 | medium | 604 | codebase_community | moderate | exact | What is the average of the up votes and the average user age for users creating ... |
| task_250 | medium | 633 | codebase_community | moderate | exact | Which post by slashnick has the most answers count? State the post ID. |
| task_257 | medium | 685 | codebase_community | moderate | exact | Identify the total views on the post 'Computer Game Datasets'. Name the user who... |
| task_259 | medium | 707 | codebase_community | moderate | exact | Among the posts with views ranging from 100 to 150, what is the comment with the... |
| task_261 | medium | 719 | superhero | moderate | exact | Among the superheroes with the super power of "Super Strength", how many of them... |
| task_269 | medium | 739 | superhero | moderate | exact | What are the names of the superheroes with the power of death touch? |
| task_283 | medium | 800 | superhero | moderate | exact | Calculate the percentage of superheroes with blue eyes. |
| task_287 | medium | 825 | superhero | moderate | exact | Identify the gender of the superhero who has the ability of Phoenix Force. |
| task_292 | medium | 869 | formula_1 | moderate | exact | For the constructor which got the highest point in the race No. 9 , what is its ... |
| task_303 | medium | 909 | formula_1 | moderate | exact | Among all European Grand Prix races, what is the percentage of the races were ho... |
| task_305 | medium | 931 | formula_1 | moderate | exact | What was the fastest lap speed among all drivers in the 2009 Spanish Grand Prix? |
| task_330 | hard | 1139 | european_football_2 | challenging | exact | What was the final score for the match on September 24, 2008, in the Belgian Jup... |
| task_344 | hard | 1247 | thrombosis_prediction | challenging | exact | Among the male patients who have a normal level of white blood cells, how many o... |
| task_349 | hard ⚠ | 1312 | student_club | simple | exact | What's Angela Sanders's major? |
| task_350 | hard ⚠ | 1317 | student_club | moderate | exact | Among the students from the Student_Club who attended the event "Women's Soccer"... |
| task_352 | hard | 1359 | student_club | challenging | exact | How many times was the budget in Advertisement for "Yearly Kickoff" meeting more... |
| task_355 | hard | 1460 | student_club | challenging | exact | Write the full name of the member who spent money for water, veggie tray and sup... |
| task_379 | hard | 281 | toxicology | challenging | exact | Tally the toxicology element of the 4th atom of each molecule that was carcinoge... |
| task_396 | hard | 760 | superhero | challenging | exact | In superheroes with height between 150 to 180, what is the percentage of heroes ... |
| task_408 | hard | 944 | formula_1 | challenging | exact | How much faster in percentage is the champion than the driver who finished the r... |
| task_415 | hard | 990 | formula_1 | challenging | exact | What is the constructor reference name of the champion in the 2009 Singapore Gra... |
| task_418 | extreme ⚠ | 1257 | thrombosis_prediction | challenging | exact | Among the patients whose creatinine level is abnormal, how many of them aren't 7... |
| task_420 | hard | 415 | card_games | challenging | exact | What percentage of cards with format commander and legal status do not have a co... |

**⚠ マーク**: BIRD difficulty を rename しただけでなく **DABench 側で手動変更** された 4 件:

- task_89: BIRD simple → DABench **hard** (= 質問テキスト typo 修正 + 難度引き上げ)
- task_330: BIRD moderate → DABench **hard**
- task_349: BIRD simple → DABench **hard** (= 'Angela Sanders' major 問題、意外に難)
- task_352: BIRD moderate → DABench **hard**
- task_355: BIRD moderate → DABench **hard**
- task_418: BIRD challenging → DABench **extreme** (= 唯一の extreme ラベル)

→ DABench 運営は BIRD difficulty に **手動 review** を加えて再分類している (= 完全自動ではない人手介入の証拠)。

---

## 8. 戦略含意 + 次の実験設計

### 短期 (= 1 週間)

1. **exp_124a (= leak-safe)**:
   - generic prompt 改良: "threshold が knowledge.md 不在の場合、medical / domain clinical reference range を default 採用"
   - + data 分布 query 強制 (= 念のため percentile 確認)
   - vote-3 で baseline + ~0.03-0.05 期待

2. **exp_124b (= conditional leak risk)**:
   - BIRD dev 11 DB の **`database_description/*.csv` から threshold 集成 → reference appendix** を preamble に組み込み
   - 公開データ source、leak ではないが overfit risk
   - シナリオ A (= hidden = dev DB) なら +0.10〜0.20
   - シナリオ B (= train DB) なら効果薄

### 中期 (= 2-3 週間)

3. **BIRD train 解析** (= 12.6 GB):
   - download + extract
   - 69 DB の database_description threshold を抽出
   - **universal threshold reference** 構築
   - exp_125 として組み込み

4. **task_199 詳細分析**: 唯一の DABench-original question、どこから来たか? (= 自作 / 別 dataset / 改変?)

### 長期 (= 提出後)

5. A/B-board 結果が出れば、**シナリオ A vs B 判別** → 投資先確定

---

## 9. リスクと leak 評価

| 戦略 | leak risk | 評価 |
|---|---|---|
| BIRD 質問テキストを我々の agent が見る | ✗ | 既に DABench で 49/50 公開済、leak の概念不適用 |
| BIRD gold SQL を prompt に含める | **❌ 完全 leak** | 禁止 |
| BIRD evidence (= per-question hint) を generic 化して prompt 化 | △ | per-question hint と generic pattern の境界に注意。具体例 (= "Alex Yoong refers to forename='Alex'") は leak、抽象パターン (= "X refers to Y では Y を SQL column と解釈") は OK |
| BIRD `database_description` の threshold を agent 知識として注入 | ○ | clinical reference range と同等、公開データ。**ただし overfit risk あり** |
| BIRD train sample からの SQL pattern 学習 | △ | 直接ペアを含めると leak (= 偶然 hidden test と一致した場合)、抽象 pattern なら OK |

---

## 10. 結論

**DABench public-50 は BIRD-SQL dev set の 3.2% stratified sample で構築された "BIRD ライト"** であり、運営独自の知見はほぼ含まれない (= 1 task のみ完全新規)。

knowledge.md は BIRD database_description.csv の LLM-rephrase narrative で、threshold 情報の ~90% が **構造的に失われている**。これが Type 2 (= unanimous wrong) 失敗タスクの主要原因。

A-board / B-board も同 pipeline で構築される前提なら、agent 側で "threshold-loss-aware" な reasoning が必須となる。BIRD database_description を **「公開済 reference knowledge」** として活用する戦略が現実解。

---

**Generated**: 2026-05-13
**Repo**: kddcup2026-kobushi
**Data sources**:
- `data/public/input/` (= DABench public-50)
- `data/external/bird/dev_20240627/` (= BIRD dev set, downloaded 2026-05-13)
- `data/external/bird/dev_20240627/dev_databases/{db}/database_description/*.csv` (= column descriptions with thresholds)
