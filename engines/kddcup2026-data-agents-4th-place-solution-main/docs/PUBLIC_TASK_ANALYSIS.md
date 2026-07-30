# DABench Public 50 タスク全分析

> 生成: `scripts/build_task_analysis_md.py` (基礎データ: `scripts/analyze_all_tasks.py`)
> 履歴ソース: `artifacts/runs/*/evaluation.csv` (leak / partial / 502retry を除外)

## カテゴリ分布

| カテゴリ | 説明 | task 数 |
|---|---|---:|
| never_solved | ⚫ 一度も解けない (never-solved, 0% perfect) | 4 |
| rarely_solved | 🔴 滅多に解けない (rarely-solved, 1-10% perfect) | 9 |
| variable | 🟠 揺れる (variable, 10-50% perfect) | 5 |
| often_solved | 🟡 ほぼ解ける (often-solved, 50-90% perfect) | 15 |
| always_solved | 🟢 常に解ける (always-solved, ≥90% perfect) | 17 |

## TOC

- **⚫ 一度も解けない (never-solved, 0% perfect)** (4 task)
  - [task_163](#task-163)  medium · n=130 · mean=0.0 · perfect=0%
  - [task_169](#task-169)  medium · n=130 · mean=0.0 · perfect=0%
  - [task_180](#task-180)  medium · n=130 · mean=0.0 · perfect=0%
  - [task_344](#task-344)  hard · n=130 · mean=0.0 · perfect=0%
- **🔴 滅多に解けない (rarely-solved, 1-10% perfect)** (9 task)
  - [task_379](#task-379)  hard · n=130 · mean=0.2135 · perfect=0%
  - [task_38](#task-38)  easy · n=130 · mean=0.521 · perfect=0%
  - [task_80](#task-80)  easy · n=130 · mean=0.0077 · perfect=1%
  - [task_396](#task-396)  hard · n=130 · mean=0.0077 · perfect=1%
  - [task_25](#task-25)  easy · n=130 · mean=0.0385 · perfect=1%
  - [task_89](#task-89)  easy · n=130 · mean=0.0231 · perfect=2%
  - [task_199](#task-199)  medium · n=130 · mean=0.0231 · perfect=2%
  - [task_418](#task-418)  extreme · n=130 · mean=0.0596 · perfect=5%
  - [task_259](#task-259)  medium · n=130 · mean=0.2427 · perfect=7%
- **🟠 揺れる (variable, 10-50% perfect)** (5 task)
  - [task_86](#task-86)  easy · n=130 · mean=0.2 · perfect=20%
  - [task_330](#task-330)  hard · n=130 · mean=0.6619 · perfect=20%
  - [task_173](#task-173)  medium · n=130 · mean=0.2923 · perfect=29%
  - [task_200](#task-200)  medium · n=130 · mean=0.3837 · perfect=34%
  - [task_196](#task-196)  medium · n=130 · mean=0.4558 · perfect=39%
- **🟡 ほぼ解ける (often-solved, 50-90% perfect)** (15 task)
  - [task_420](#task-420)  hard · n=129 · mean=0.6279 · perfect=63%
  - [task_11](#task-11)  easy · n=130 · mean=0.7 · perfect=70%
  - [task_352](#task-352)  hard · n=130 · mean=0.7615 · perfect=76%
  - [task_257](#task-257)  medium · n=130 · mean=0.8051 · perfect=78%
  - [task_249](#task-249)  medium · n=130 · mean=0.8026 · perfect=79%
  - [task_67](#task-67)  easy · n=130 · mean=0.8365 · perfect=81%
  - [task_292](#task-292)  medium · n=130 · mean=0.8538 · perfect=85%
  - [task_408](#task-408)  hard · n=130 · mean=0.8692 · perfect=85%
  - [task_22](#task-22)  easy · n=130 · mean=0.8673 · perfect=86%
  - [task_27](#task-27)  easy · n=130 · mean=0.9023 · perfect=87%
  - [task_243](#task-243)  medium · n=130 · mean=0.9077 · perfect=87%
  - [task_250](#task-250)  medium · n=130 · mean=0.8846 · perfect=88%
  - [task_287](#task-287)  medium · n=130 · mean=0.8904 · perfect=88%
  - [task_303](#task-303)  medium · n=130 · mean=0.8962 · perfect=88%
  - [task_355](#task-355)  hard · n=130 · mean=0.9167 · perfect=88%
- **🟢 常に解ける (always-solved, ≥90% perfect)** (17 task)
  - [task_19](#task-19)  easy · n=130 · mean=0.9 · perfect=90%
  - [task_349](#task-349)  hard · n=130 · mean=0.9205 · perfect=92%
  - [task_269](#task-269)  medium · n=130 · mean=0.9212 · perfect=92%
  - [task_194](#task-194)  medium · n=130 · mean=0.9231 · perfect=92%
  - [task_218](#task-218)  medium · n=130 · mean=0.9231 · perfect=92%
  - [task_261](#task-261)  medium · n=130 · mean=0.9231 · perfect=92%
  - [task_350](#task-350)  hard · n=130 · mean=0.9231 · perfect=92%
  - [task_214](#task-214)  medium · n=130 · mean=0.9308 · perfect=93%
  - [task_74](#task-74)  easy · n=130 · mean=0.9385 · perfect=94%
  - [task_415](#task-415)  hard · n=130 · mean=0.9423 · perfect=94%
  - [task_24](#task-24)  easy · n=130 · mean=0.9462 · perfect=95%
  - [task_75](#task-75)  easy · n=130 · mean=0.9462 · perfect=95%
  - [task_145](#task-145)  medium · n=130 · mean=0.9462 · perfect=95%
  - [task_283](#task-283)  medium · n=130 · mean=0.9564 · perfect=95%
  - [task_26](#task-26)  easy · n=174 · mean=0.9655 · perfect=97%
  - [task_64](#task-64)  easy · n=130 · mean=0.9692 · perfect=97%
  - [task_305](#task-305)  medium · n=130 · mean=0.9692 · perfect=97%

---

## ⚫ 一度も解けない (never-solved, 0% perfect)  (4 task)

### task_163  (medium, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0, std=0.0, perfect_rate=0%, zero_rate=100%

**質問**:
> Identify the type of expenses and their total value approved for 'October Meeting' event.

**入力データ**:
- **CSV** `csv/expense.csv` (2.9KB): shape sampled=[32, 7], cols=[expense_id, expense_description, expense_date, cost, approved, link_to_member, link_to_budget]
- **SQLite** `db/event.db` (20.0KB): tables=['event']
    - table `event`: cols=[event_id, event_name, event_date, type, notes, location, status]
- **SQLite** `event.db` (0.0KB): tables=[]
- **JSON** `json/budget.json` (11.5KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 2 cols**
- header: `['type', 'SUM(T3.cost)']`
- first rows:
  - `['Meeting', '175.39']`

**正解までの reasoning steps**:
1. 'October Meeting' を `event.event_name` で検索 → event_id + `event.type` 値 (= "Meeting") を取得
2. budget.json で `link_to_event = <event_id>` の budget_id 群を取得
3. expense.csv で `link_to_budget IN (...) AND approved = 'true'` をフィルタ
4. cost の SUM 算出 = 175.39
5. 出力 = 1 行 × 2 col: `(event の type 値, SUM(approved cost))`

**失敗モード / 過去 trace の傾向**:
- **解釈ミス確定**: agent が "type of expenses" を **event の `type` 列値 (= "Meeting")** ではなく **expense_description ごとのカテゴリ (Pizza, Posters, Water+chips...)** と読む
- 観測例 (exp_086): `[[Pizza, 51.81], [Posters, 54.25], [Water+chips, 69.33]]` (= 3 行に分解、recall 0)
- **130 run で 0% perfect** = 解釈の失敗が普遍的、temperature 揺らぎでも回避できない

**メモ**:
- gold col header `SUM(T3.cost)` は SQL alias literal、eval は値マッチで救済可能 (header 不一致は致命的でない)
- 真の問題は **行レベルの解釈** (= "type" が何を指すか)
- ⚠️ 質問が ambiguous: 字義的には expense_description のカテゴライズが自然 (agent 解釈の方が理にかなう面あり)。**gold-side bias** に近い変則 task
- 救済困難: Rule 19 強化でも質問構造的に多解釈、構造的勝ち目薄い
- 投資優先度: **低** (= 質問の bias を理解しても 1 task のみ)

---

### task_169  (medium, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0, std=0.0, perfect_rate=0%, zero_rate=100%

**質問**:
> What was the average monthly consumption of customers in SME for the year 2013?

**入力データ**:
- **CSV** `csv/yearmonth.csv` (8017.9KB): shape sampled=[1000, 3], cols=[CustomerID, Date, Consumption]
- **SQLite** `customers.db` (0.0KB): tables=[]
- **SQLite** `db/customers.db` (908.0KB): tables=['customers']
    - table `customers`: cols=[CustomerID, Segment, Currency]
- **DOC** `knowledge.md` (4.7KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['AVG(T2.Consumption) / 12']`
- first rows:
  - `['459.9562642871061']`

**正解までの reasoning steps**:
1. customers.db の `customers` から `Segment = 'SME'` の CustomerID 群を取得
2. yearmonth.csv で `Date LIKE '2013%'` (= 6 桁 YYYYMM、年 2013) AND `CustomerID IN (SME 集合)` をフィルタ
3. **per-customer の年合計** を取り、それを 12 で割って **月平均** を customer ごとに計算
4. 全 SME 顧客の月平均値を **AVG** で集計 → スカラー `459.96`
5. **= AVG(yearly_total / 12) = (平均年合計) / 12**

**失敗モード / 過去 trace の傾向**:
- **計算式の組み立てミスが普遍的**: 130 run 全てで 0 perfect
- 観測値 (exp_086): `82,027,220.30` ← 月平均ではなく **総消費 (= SUM)** を返してる
- 別パターン: `1138.93` (= avg of consumption rows、month 概念抜け)
- **/ 12 ぶんで割るステップが抜ける** or `AVG()` を全 row に適用 (= 既に集計済 row の avg = 月当たり avg)
- gold の `AVG(T2.Consumption) / 12` という SQL 表現自体が示唆: **per-customer の年合計を出してから 12 で割って AVG する** という二段集計が必要

**メモ**:
- gold value `459.96` は agent 観測値 `82M` から **5 桁ずれ** ← 「単位 / 期間 / 集約レベル」のサニティチェックがあれば検出可能
- 必要な reasoning: "monthly consumption" の解釈を **「月ごとの値」ではなく「月単位での平均」** と読み替える + 二段 aggregation
- 救済策: 計算結果のオーダー比較 verify step (= 「全消費 vs 1 顧客の予想範囲」) を追加すれば agent が再計算へ進む可能性
- 投資優先度: **中** (= scaler answer なので比較的低リスクで verify-rule 系の検出ターゲットに使える)

---

### task_180  (medium, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0, std=0.0, perfect_rate=0%, zero_rate=100%

**質問**:
> For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.

**入力データ**:
- **CSV** `csv/yearmonth.csv` (8017.9KB): shape sampled=[1000, 3], cols=[CustomerID, Date, Consumption]
- **SQLite** `db/transactions_1k.db` (64.0KB): tables=['sqlite_sequence', 'transactions_1k']
    - table `sqlite_sequence`: cols=[name, seq]
    - table `transactions_1k`: cols=[TransactionID, Date, Time, CustomerID, CardID, GasStationID, ProductID, Amount, Price]
- **DOC** `knowledge.md` (4.7KB)

**Gold answer**:
- shape: **9 rows × 1 cols**
- header: `['Consumption']`
- first rows:
  - `['1903.2']`
  - `['88265.39']`
  - `['1129.2']`

**正解までの reasoning steps**:
1. transactions_1k.db で `ProductID = 5 AND Price > 29.00` のレコードから CustomerID 群を抽出 (= "paid more than 29.00 per unit of product 5" の人々)
2. yearmonth.csv で `CustomerID IN (上記) AND Date = '201208'` (= 2012 年 8 月) をフィルタ
3. **Consumption 列のみ** を出力 (= CustomerID は不要)
4. 出力 = 9 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- **典型パターン**: 153 行 × 2 col (CustomerID, Consumption) を出力 → recall=1.0 (= 値は全部含む) だが extra col + extra row で λ0.5 = 0.0
- **filter scope のミス**: `Price > 29.00` の条件抜けで 8 月の全顧客の Consumption を返す
- 130 run 中で 0 perfect = **「単価 > 29」フィルタの適用失敗が普遍**
- gold の 9 値はだいたい pred 内に存在することが多い (= 値レベルは正解、scope が広すぎ)

**メモ**:
- 質問構造が複雑: "paid more than 29.00 **per unit** of product id No.5" の **per unit = transactions の Price** という解釈が必要。Amount でも cost でもなく Price 列。
- knowledge.md には Price 概念が明示されておらず agent は推測必要
- 救済策: Rule 18 column verify は `[CustomerID, Consumption]` → `[Consumption]` の縮小に有効。さらに **filter logic verify** が必要
- 投資優先度: **中** (= rule-based plan-final verify で部分救済可能)

---

### task_344  (hard, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0, std=0.0, perfect_rate=0%, zero_rate=100%

**質問**:
> Among the male patients who have a normal level of white blood cells, how many of them have an abnormal fibrinogen level?

**入力データ**:
- **CSV** `csv/Laboratory.csv` (1495.3KB): shape sampled=[1000, 44], cols=[ID, Date, GOT, GPT, LDH, ALP, TP, ALB...]
- **CSV** `patient_sex.csv` (0.9KB): shape sampled=[92, 2], cols=[ID, SEX]
- **DOC** `doc/Patient.md` (54.5KB)
- **DOC** `knowledge.md` (5.2KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(DISTINCT T1.ID)']`
- first rows:
  - `['4']`

**正解までの reasoning steps**:
1. patient_sex.csv で `SEX = 'M'` の ID 群を取得
2. Laboratory.csv で `WBC` (= White Blood Cells) と `FG` (= Fibrinogen) の値を取得
3. **正常 / 異常の閾値判定**:
   - WBC 正常: 4.0 ≤ WBC ≤ 9.0 (= 単位 ×10⁹/L、医学標準)
   - FG 異常: FG < 1.5 OR FG > 4.0 (= 単位 g/L)
4. `male AND WBC normal AND FG abnormal` の DISTINCT 患者 ID 数 = 4

**失敗モード / 過去 trace の傾向**:
- **130 run 全 0**: 正解の 4 人を識別できず大半が `0` 出力
- 観測例 (exp_086): `[[count, 0]]` (= 全患者を弾いてしまう過剰フィルタ)
- 別パターン: 数十〜数百の count (= 閾値が反対 / WBC/FG の単位混乱)
- 根本原因: knowledge.md に **WBC / FG の数値閾値が記載されていない** (= LDH/UA/PLT の閾値はあるが WBC/FG は欠落)
- agent は医学的常識で推測する必要があるが、複数の医学標準 (= ×10³ vs ×10⁹ の単位差) があり一意に決められない

**メモ**:
- knowledge.md (4 KB) は WBC/FG 閾値を **記述していない** = データから推定不可能な情報空白
- doc/Patient.md (54 KB) は患者プロファイルの **narrative 文** (= ID 単位の記述) で閾値情報なし
- gold value = 4 人 → 全患者数 ~92 (patient_sex.csv) なので **4/92 ≈ 4%** = 標準閾値で見ると妥当だが agent には判明しない
- 救済策:
  - 外部 medical reference を agent に与える (= 大改修)
  - knowledge から推測ヒントを LLM に再質問 (= rule-based threshold extraction の Tier 4 軸)
- 投資優先度: **低** (= 知識欠落による外因失敗、prompt や preamble 改善では救えない)

---

## 🔴 滅多に解けない (rarely-solved, 1-10% perfect)  (9 task)

### task_379  (hard, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.2135, std=0.3397, perfect_rate=0%, zero_rate=72%

**質問**:
> Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.

**入力データ**:
- **CSV** `csv/atom.csv` (213.6KB): shape sampled=[1000, 3], cols=[atom_id, molecule_id, element]
- **DOC** `carcinogenic_mols.txt` (0.6KB)
- **DOC** `doc/molecule.md` (35.6KB)
- **DOC** `knowledge.md` (6.5KB)

**Gold answer**:
- shape: **7 rows × 1 cols**
- header: `['element']`
- first rows: `['c']`, `['br']`, `['cl']`

**正解までの reasoning steps**:
1. carcinogenic_mols.txt から発癌性分子の molecule_id 群を取得
2. atom.csv で `molecule_id IN (carcinogenic 集合)` のレコードを取得
3. 各 molecule 内で **atom_id の昇順 4 番目** (= 4th atom) を選択
4. element 列だけを出力 (= count や数値カラムは不要)
5. 出力 = 7 行 (= carcinogenic 分子 7 つ) × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 7r × **2c** (`element, count`) — 行数は正しいが count 列が余分
- 別パターン: 100 r × 3c (= 全 carcinogenic atoms を返す + count + element) — 4 番目絞り込み失敗
- 130 run のうち 72% で zero、28% で部分点
- agent が "tally" を **集計** (= count) と解釈して count 列を追加するのが常習化

**メモ**:
- "Tally the X" は「Xをリストアップ」の意味だが agent は GROUP BY で count を取るのが直感
- gold col header `element` だけ → 質問通り「element をそのまま並べる」が正解
- 救済策: Rule 5 (column_count 制約) と Rule 18 (column verify) で count 列除外を強制
- 投資優先度: **高** (= column_count discipline で直接救済可能、簡単な勝ち)

---

### task_38  (easy, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.521, std=0.1748, perfect_rate=0%, zero_rate=10%

**質問**:
> List all the withdrawals in cash transactions that the client with the id 3356 makes.

**入力データ**:
- **CSV** `csv/disp.csv` (116.2KB): shape sampled=[1000, 4], cols=[disp_id, client_id, account_id, type]
- **CSV** `csv/trans.csv` (58374.1KB): shape sampled=[1000, 10], cols=[trans_id, account_id, date, type, operation, amount, balance, k_symbol...]
- **JSON** `json/account.json` (569.2KB)
- **JSON** `json/client.json` (611.7KB)
- **DOC** `knowledge.md` (4.8KB)

**Gold answer**:
- shape: **140 rows × 1 cols**
- header: `['trans_id']`
- first rows:
  - `['816173']`
  - `['816174']`
  - `['816175']`

**正解までの reasoning steps**:
1. JSON `account.json` で client_id=3356 の account_id を解決
2. JSON `client.json` も併用して client→account の関係確認
3. trans.csv で `account_id IN (上記)` AND `type='VYDAJ' (= withdrawal)` AND `operation='VYBER' (= cash)` をフィルタ
4. **trans_id 列のみ**を出力 (= account_id, date, amount は不要)
5. 出力 = 140 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 140r × **6c** (`trans_id, account_id, date, type, operation, amount`) — 行は正しいが余分列 5 つで wext 罰則
- 130 run で 0 perfect (= 列スコープ規律の構造的失敗)
- gold 値 (trans_id) は予測内に存在 → recall 高いが extras_ratio が λ0.5 を 0 に潰す
- 質問 "List all the withdrawals" → agent は SELECT * 風で関連列全部返す傾向

**メモ**:
- VYDAJ / VYBER はチェコ語で **withdrawal / cash** (= knowledge.md に記載)
- gold col=trans_id 1 個のみ → Rule 12 (explicit SELECT) + Rule 18 (column verify) で救済可能
- 投資優先度: **高** (= 列削減だけで perfect、easy task の取りこぼし救済)

---

### task_80  (easy, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0077, std=0.0877, perfect_rate=1%, zero_rate=99%

**質問**:
> What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903?

**入力データ**:
- **CSV** `csv/qualifying.csv` (270.9KB): shape sampled=[1000, 9], cols=[qualifyId, raceId, driverId, constructorId, number, position, q1, q2...]
- **JSON** `json/drivers.json` (232.2KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **2 rows × 1 cols**
- header: `['number']`
- first rows:
  - `['3']`
  - `['5']`

**正解までの reasoning steps**:
1. qualifying.csv で raceId=903 のレコードを取得
2. q3 列が `'0:01:54'` (or 1:54.xxx 等の format バリエーション) のレコードをフィルタ
3. **driverId は答えではなく** `number` 列を取得 (= ドライバーゼッケン番号)
4. **同タイム複数あり** (= ties)、両方の number を返す = 3 と 5
5. 出力 = 2 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × 1c (`3` のみ) — **2 番目の tied driver `5` を取りこぼす**
- 130 run で 99% zero、1% perfect (= ties 救済が稀)
- LIMIT 1 / `==` 一致のみで filter-back せず単一 driver 返却が常習
- 一部 run では時間 format 違いで recall 0 (`0:01:54.000` vs `0:01:54`)

**メモ**:
- gold が 2 行 → Rule 10 (filter-back for ties) が直接効く対象
- ただし superlative 単語 (lowest 等) を含まないので Rule 10 トリガーに引っかからない
- "finished 0:01:54" = 完了タイムが正確に 0:01:54 のドライバー (= 等値マッチ、ties あり)
- 救済策: 等値マッチでも tied 結果を予期する rule 追加 (= Rule 10 を 拡張)
- 投資優先度: **中** (= rule extension で救済可能、1 task のみ)

---

### task_396  (hard, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0077, std=0.0877, perfect_rate=1%, zero_rate=99%

**質問**:
> In superheroes with height between 150 to 180, what is the percentage of heroes published by Marvel Comics?

**入力データ**:
- **JSON** `json/publisher.json` (1.7KB)
- **DOC** `doc/superhero.md` (173.8KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `["CAST(COUNT(CASE WHEN T2.publisher_name = 'Marvel Comics' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)"]`
- first rows:
  - `['54.83870967741935']`

**正解までの reasoning steps**:
1. doc/superhero.md からヒーロー情報抽出 (= name, height, publisher_id)
2. publisher.json から publisher_id → publisher_name のマッピング
3. height が 150-180 の範囲のヒーローを集める (= 母数集合)
4. その中で publisher_name = 'Marvel Comics' のヒーロー数 / 母数 × 100 = 54.84%
5. 出力 = 1 行 × 1 col のスカラー値

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × **2c** (`percentage, percentage`) で値 60.87, 48.15 — 2 列の重複 + 値も間違い
- 質問の "between 150 to 180" の境界解釈ぶれ (= inclusive vs exclusive)
- doc/superhero.md は narrative 文 (= 173KB) でデータ構造化が困難
- 130 run で 99% zero
- gold value 54.83870967741935 は agent が再現困難な精度

**メモ**:
- doc/superhero.md は agent が parse しづらい narrative 形式 (= LLM extraction が必要)
- gold col header `CAST(...) * 100 / COUNT(...)` は SQL alias literal、value match の余地あり
- height filter の境界解釈 (150 ≤ h ≤ 180 vs 150 < h < 180) で母数が変わる可能性
- 救済策: doc parsing の精度向上 (= 別軸の大改造)
- 投資優先度: **低** (= 文書 parsing の根本問題、prompt fix では改善困難)

---

### task_25  (easy, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0385, std=0.1646, perfect_rate=1%, zero_rate=95%

**質問**:
> Which event has the lowest cost?

**入力データ**:
- **CSV** `csv/budget.csv` (3.5KB): shape sampled=[52, 7], cols=[budget_id, category, spent, remaining, amount, event_status, link_to_event]
- **JSON** `json/event.json` (10.9KB)
- **JSON** `json/expense.json` (8.5KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **3 rows × 1 cols**
- header: `['event_name']`
- first rows:
  - `['November Speaker']`
  - `['October Speaker']`
  - `['September Speaker']`

**正解までの reasoning steps**:
1. expense.json で `approved=true` の cost を event 単位で集計
2. budget.csv の link_to_event を介して event.json と join
3. event ごとの total_cost を計算
4. **最低 cost を持つ event を全部** (= ties あり) 取得
5. event_name 列のみ出力 = November Speaker / October Speaker / September Speaker (3 行 tied at 6.0)

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 3r × **2c** (`event_name, total_cost`) で値 `Officers meeting - November` 等 — **行数 3 は当たってるが event 自体が違う**
- 値が違う原因: budget.amount を使うか expense.cost を使うかの解釈ぶれ
- 130 run で 95% zero、行数 1 で出すケースが大多数 (= ties 取りこぼし)
- exp_109 の row_count 撤廃 + Rule 10 filter-back で `+0.31` 改善傾向あり (= 救済可能性高い)

**メモ**:
- "lowest cost" の cost = expense.cost の SUM か budget.amount かで解釈分岐
- gold は **expense の SUM** ベース (= Officers meeting series ではなく Speaker series)
- ties が 3 つ完全一致なので filter-back (Rule 10) が効く
- exp_109 で実際に救済されている (= row_count 撤廃が効果)
- 投資優先度: **高** (= 既に救済路線確立、定着させたい)

---

### task_89  (easy, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0231, std=0.1507, perfect_rate=2%, zero_rate=98%

**質問**:
> What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?

**入力データ**:
- **CSV** `csv/results.csv` (1200.3KB): shape sampled=[1000, 18], cols=[resultId, raceId, driverId, constructorId, number, grid, position, positionText...]
- **JSON** `json/races.json` (238.6KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['time']`
- first rows:
  - `['+16.445']`

**正解までの reasoning steps**:
1. races.json で 2008 年の Chinese Grand Prix の raceId を取得
2. results.csv で `raceId=該当 AND positionOrder=2` の driver を特定
3. その driver の `time` 列を取得 = `+16.445`
4. 出力 = 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × 1c で `+14.925` — **driver が違う / 別 race と混同 / position 列の取り違え**
- gold value `+16.445` は 2 位ドライバーの finish time
- 130 run で 98% zero
- agent が `position` (= grid 位置) と `positionOrder` (= 終了順位) を混同する典型ケース

**メモ**:
- results.csv 列が多数 (positionOrder, position, positionText, points, ... 18 列) で agent が `position` を直感的に使うが正解は `positionOrder`
- gold value 取れた稀な perfect run はうまく positionOrder を使った例
- 救済策: knowledge.md で position/positionOrder の意味を明示
- 投資優先度: **中** (= column 名理解の問題、knowledge enrichment で改善可能)

---

### task_199  (medium, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0231, std=0.1507, perfect_rate=2%, zero_rate=98%

**質問**:
> List the names and funding types of schools from Riverside-related school districts where the average SAT math score across schools exceeds 400.

**入力データ**:
- **CSV** `csv/frpm.csv` (2770.0KB): shape sampled=[1000, 29], cols=[CDSCode, Academic Year, County Code, District Code, School Code, County Name, District Name, School Name...]
- **SQLite** `db/satscores.db` (260.0KB): tables=['satscores']
    - table `satscores`: cols=[cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite...]
- **DOC** `knowledge.md` (5.0KB)

**Gold answer**:
- shape: **6 rows × 2 cols**
- header: `['sname', 'Charter Funding Type']`
- first rows:
  - `['Arlington High', '']`
  - `['John W. North High', '']`
  - `['Martin Luther King Jr. High', '']`

**正解までの reasoning steps**:
1. frpm.csv で `District Name LIKE '%Riverside%'` の school 群取得
2. satscores.db の `satscores` で `cds IN (上記 school)` AND `AvgScrMath > 400` をフィルタ
3. group by district、district の avg(AvgScrMath) > 400 の district を選択
4. 該当 district の **school name** と **Charter Funding Type** (= frpm の列) を取得 (2 列)
5. 出力 = 6 行 × 2 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): **506r × 6c** (= School Name 列が 3 つ重複 + 別文字列の Charter Funding Type 列も 3 つ)
- 完全に scope mismatch + 列重複の双方
- 130 run で 98% zero
- "Riverside-related school districts" の解釈 (= District Name に Riverside を含む) で district を絞らず、 school 単位で集計

**メモ**:
- 質問が複合的: "Riverside-related" + "average across schools" + "names AND funding types" (2 列)
- gold col `Charter Funding Type` の値が空文字列 (`''`) の行あり (= NULL に相当) → agent が空セル表現を間違える
- 救済策: 多段 reasoning + 列スコープ verify
- 投資優先度: **中** (= 多列複合 task の代表例、verify 系 rule で部分救済期待)

---

### task_418  (extreme, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.0596, std=0.2346, perfect_rate=5%, zero_rate=94%

**質問**:
> Among the patients whose creatinine level is abnormal, how many of them aren't 70 yet?

**入力データ**:
- **DOC** `doc/Laboratory.md` (279.8KB)
- **DOC** `doc/Patient.md` (83.0KB)
- **DOC** `knowledge.md` (5.2KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(DISTINCT T1.ID)']`
- first rows:
  - `['1']`

**正解までの reasoning steps**:
1. doc/Patient.md から患者の生年月日抽出 → 年齢計算 (current year - birth year, 70 未満)
2. doc/Laboratory.md から creatinine (= CRE) 値抽出
3. CRE 異常閾値で患者をフィルタ
4. 上記両条件 AND の **DISTINCT 患者 ID 数** = 1
5. 出力 = 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × 1c で値 `2` — **count が違う**
- gold = 1
- 130 run で 94% zero、5% perfect
- 失敗原因: CRE の "abnormal" 閾値が unclear (= male: >1.3 / female: >1.0 等) + 年齢 70 未満の境界 (年齢 < 70 vs ≤ 70) の解釈ぶれ
- doc 形式 (= narrative) で患者情報抽出が不安定

**メモ**:
- creatinine の正常範囲: 男性 0.6-1.2 mg/dL、女性 0.5-1.1 mg/dL (医学標準)
- doc/Laboratory.md (279KB) は narrative、構造化抽出困難
- gold = 1 の極端な少なさ → 厳密な閾値定義必要
- 救済策: 医学閾値を knowledge に明記 + doc parsing 強化
- 投資優先度: **低** (= 知識欠落 + 文書 parsing の二重困難)

---

### task_259  (medium, 深掘り)

**履歴統計**: n=130, λ0.5 mean=0.2427, std=0.3426, perfect_rate=7%, zero_rate=65%

**質問**:
> Among the posts with views ranging from 100 to 150, what is the comment with the highest score?

**入力データ**:
- **CSV** `csv/comments.csv` (46994.3KB): shape sampled=[1000, 7], cols=[Id, PostId, Score, Text, CreationDate, UserId, UserDisplayName]
- **SQLite** `db/posts.db` (139384.0KB): tables=['posts']
    - table `posts`: cols=[Id, PostTypeId, AcceptedAnswerId, CreaionDate, Score, ViewCount, Body, OwnerUserId, LasActivityDate, Title...]
- **SQLite** `posts.db` (0.0KB): tables=[]
- **DOC** `knowledge.md` (5.4KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['Text']`
- first rows:
  - `['Welcome to Cross Validated David P, for more information about the site and how to ask questions, you can check out the [FAQ](http://stats.stackexchange.com/faq). Is your question just about whether the calculation is right or wrong (it is wrong, with the numbers you give on your second line the result of the operation is 0.187, rounded)? Otherwise, yes, a percentage can be as close to zero as you can imagine because it is bounded between 0 and 100 and continuous on that interval.']`

**正解までの reasoning steps**:
1. SQLite posts.db で ViewCount BETWEEN 100 AND 150 の Post.Id 群を取得
2. comments.csv で `PostId IN (上記)` のコメントをフィルタ
3. Score が最高のコメントを選択 (= ties あれば全部)
4. **Text 列のみ**を出力
5. 出力 = 1 行 × 1 col (gold は 1 行だが構造的には ties 可能)

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × **4c** (`comment, comment_id, score, post_id`) — **値は正しいが列が 3 つ余分**
- 130 run で 65% zero、7% perfect、28% partial (= 部分点取れること多い)
- score=14 の Welcome message が gold 値と一致する run はあるが extras で λ0.5 削られる
- Text 列の長文 + 改行を含む CSV エスケープ問題でも稀に escape ミス

**メモ**:
- gold col header `Text` のみ → Rule 5 で他列削除すれば perfect
- 値は正しい場合が多いので **column verify (Rule 18)** が直接効く
- 巨大ファイル (comments.csv 47 MB, posts.db 139 MB) で preamble truncation も影響
- exp_101 rich preamble の profile で改善期待大
- 投資優先度: **高** (= column verify + rich preamble combo で確実救済可能)

---

## 🟠 揺れる (variable, 10-50% perfect)  (5 task)

### task_86  (easy, 中)

**履歴統計**: n=130, λ0.5 mean=0.2, std=0.4015, perfect_rate=20%, zero_rate=80%

**質問**:
> Which race was Alex Yoong in when he was in track number less than 20?

**入力データ**:
- **CSV** `answer.csv` (0.4KB): shape sampled=[18, 1], cols=[name]
- **CSV** `csv/driverStandings.csv` (807.8KB): shape sampled=[1000, 7], cols=[driverStandingsId, raceId, driverId, points, position, positionText, wins]
- **CSV** `csv/races.csv` (98.0KB): shape sampled=[976, 8], cols=[raceId, year, round, circuitId, name, date, time, url]
- **JSON** `json/drivers.json` (232.2KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **16 rows × 1 cols**
- header: `['name']`
- first rows:
  - `['Australian Grand Prix']`
  - `['Malaysian Grand Prix']`
  - `['Brazilian Grand Prix']`

**正解までの reasoning steps**:
1. drivers.json で `Alex Yoong` の driverId 取得
2. results.csv で `driverId=該当 AND grid<20` (= 出走 grid 順位 20 未満) のレコード抽出
3. raceId の集合から races.json で race name を取得
4. **name 列のみ** 出力 (= 順位や date は不要)
5. 出力 = 16 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 18r × **2c** (`name, name`) — 列重複 (= 同じ列が 3-attempt union から残る) + 行 1〜2 余分
- 130 run で 80% non-perfect、20% perfect (= ばらつき大)
- "track number" の解釈ぶれ (= grid number? track position? race number?) で別フィルタになることあり
- attempts diversity が高い task で union 後の wext / extra row が増える典型

**メモ**:
- "track number" は results.csv の `grid` 列 (= スタート grid 位置) を指すと推定
- gold 16 行は Alex Yoong の career の特定期間 (= grid<20 の race 群)
- 救済策: padding fix (= exp_093 同等) + Rule 18 column verify
- 投資優先度: **中** (= 既存 fix で部分救済期待、padding fix 効果検証中)

---

### task_330  (hard, 中)

**履歴統計**: n=130, λ0.5 mean=0.6619, std=0.3376, perfect_rate=20%, zero_rate=12%

**質問**:
> What was the final score for the match on September 24, 2008, in the Belgian Jupiler League between the home team and the away team?

**入力データ**:
- **CSV** `csv/Match.csv` (272864.4KB): shape sampled=[1000, 115], cols=[id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id...]
- **DOC** `doc/League.md` (12.4KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 2 cols**
- header: `['home_team_goal', 'away_team_goal']`
- first rows:
  - `['1', '1']`

**正解までの reasoning steps**:
1. Match.csv で `date='2008-09-24'` AND League が 'Belgian Jupiler League' のレコードを取得
2. country.json / league.json で league_id 解決
3. 該当 match の `home_team_goal, away_team_goal` 2 列を取得
4. 出力 = 1 行 × 2 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × 2c で `date, home_team_goal` — **away_team_goal 列が抜け、date が混入**
- 130 run で 80% non-perfect、20% perfect
- 巨大 CSV (Match.csv 279 MB) の preamble truncation で agent が away_team_goal 列を発見できないケース
- exp_101 rich preamble (= 列リスト全部表示) で改善期待大
- date 列を答えに含めるのは Rule 12 (explicit SELECT) の弱さ

**メモ**:
- exp_101 rich preamble の profile section が **115 列全部表示**して救済される代表例
- gold = (1, 1) の引き分け試合
- 救済策: rich preamble + column verify combo
- 投資優先度: **高** (= rich preamble の effect ターゲット、winning 路線)

---

### task_173  (medium, 中)

**履歴統計**: n=130, λ0.5 mean=0.2923, std=0.4566, perfect_rate=29%, zero_rate=71%

**質問**:
> Please list the countries of the gas stations with transactions taken place in June, 2013.

**入力データ**:
- **CSV** `csv/yearmonth.csv` (8017.9KB): shape sampled=[1000, 3], cols=[CustomerID, Date, Consumption]
- **SQLite** `db/transactions_1k.db` (64.0KB): tables=['sqlite_sequence', 'transactions_1k']
    - table `sqlite_sequence`: cols=[name, seq]
    - table `transactions_1k`: cols=[TransactionID, Date, Time, CustomerID, CardID, GasStationID, ProductID, Amount, Price]
- **JSON** `json/gasstations.json` (626.3KB)
- **DOC** `knowledge.md` (4.7KB)

**Gold answer**:
- shape: **2 rows × 1 cols**
- header: `['Country']`
- first rows:
  - `['CZE']`
  - `['SVK']`

**正解までの reasoning steps**:
1. transactions テーブル (= 大型 .db か CSV) で `Date LIKE '2013-06%'` の transactions を取得
2. transaction の GasStationID から gas station の Country を引く
3. **DISTINCT Country** を出力
4. gold = `['CZE', 'SVK']` (2 国)

**失敗モード / 過去 trace の傾向**:
- 観測: 2 行 1 列で正しい構造の run はあるが、Country 値が誤る (= e.g. 'Czech Republic' vs 'CZE')
- 130 run で 71% non-perfect、29% perfect
- 値の format (= ISO code vs full name) で一致しないケース
- knowledge.md には `'CZE' (Czech Republic), 'SVK' (Slovakia)` 表記あり、agent が ISO に正規化できれば perfect

**メモ**:
- knowledge.md の Country 値 hint = ISO 3-letter code が正解の format
- agent が `'Czech Republic'` で出すと value match で 0.0
- 救済策: knowledge.md の value format を Rule で参照させる (= 既存 R3 系で対応)
- 投資優先度: **低** (= format 統一のみ、稀な失敗)

---

### task_200  (medium, 中)

**履歴統計**: n=130, λ0.5 mean=0.3837, std=0.4757, perfect_rate=34%, zero_rate=60%

**質問**:
> Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.

**入力データ**:
- **CSV** `csv/atom.csv` (213.6KB): shape sampled=[1000, 3], cols=[atom_id, molecule_id, element]
- **SQLite** `db/bond.db` (568.0KB): tables=['bond']
    - table `bond`: cols=[bond_id, molecule_id, bond_type]
- **JSON** `json/molecule.json` (20.8KB)
- **DOC** `knowledge.md` (6.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(T1.atom_id)']`
- first rows:
  - `['1']`

**正解までの reasoning steps**:
1. molecule の triple-bond 分子を特定 (= bond テーブルで bond_type='triple' を含む molecule)
2. その molecule の atom テーブルで `element='p' (phosphorus) OR element='br' (bromine)` をフィルタ
3. atom_id を **COUNT** = 1
4. 出力 = 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × **2c** (`count, count`) で `4, 1` — 重複 col + 値解釈分岐
- 130 run で 66% non-perfect、34% perfect
- 解釈ぶれ: 「triple-bond を含む molecule の全 atoms」 vs 「triple-bond 結合に直接参加する p/br atoms」
- gold = 1 (= 厳密に triple-bond participation の p/br atom は 1 個のみ)

**メモ**:
- bond_type の表記ぶれ (= 'triple' vs '#' vs '3') で agent 困惑
- 重複 col はやはり Rule 18 column verify で削減可能
- 投資優先度: **中** (= 解釈の一貫性 + column verify 必要)

---

### task_196  (medium, 中)

**履歴統計**: n=130, λ0.5 mean=0.4558, std=0.4837, perfect_rate=39%, zero_rate=52%

**質問**:
> What is the average number of bonds the atoms with the element iodine have?

**入力データ**:
- **CSV** `csv/connected.csv` (708.2KB): shape sampled=[1000, 3], cols=[atom_id, atom_id2, bond_id]
- **SQLite** `db/atom.db` (504.0KB): tables=['atom']
    - table `atom`: cols=[atom_id, molecule_id, element]
- **DOC** `knowledge.md` (6.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['CAST(COUNT(T2.bond_id) AS REAL) / COUNT(T1.atom_id)']`
- first rows:
  - `['1.0']`

**正解までの reasoning steps**:
1. atom テーブルで `element='i'` (= iodine) の atom_id 群を取得
2. bond / connected テーブルで該当 atom_id を含む bond 数を集計
3. AVG(bond_count per atom) または COUNT(bond)/COUNT(atom) でスカラー値計算 = 1.0
4. 出力 = 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 観測 (exp_086): 1r × **2c** (`average_bonds, average_bonds`) で値 `1.0, 2.0` — 重複 col + 異なる解釈の値
- 130 run で 61% non-perfect、39% perfect
- 解釈ぶれ: 「avg bonds per atom」 vs 「total bonds / count atoms」 (= ratio)
- 3-attempt union で 2 つの解釈が混在 → padding 起きないが col 重複

**メモ**:
- gold value 1.0 = 1 atom あたり平均 1 bond (= iodine の性質)
- 重複 col は exp_109 の column verify で解決可能
- 投資優先度: **中** (= column verify 直接効果、padding fix 不要)

---

## 🟡 ほぼ解ける (often-solved, 50-90% perfect)  (15 task)

### task_420  (hard, 簡潔)

**履歴統計**: n=129, λ0.5 mean=0.6279, std=0.4852, perfect_rate=63%, zero_rate=37%

**質問**:
> What percentage of cards with format commander and legal status do not have a content warning?

**入力データ**:
- **SQLite** `db/cards.db` (60136.0KB): tables=['cards', 'sqlite_sequence']
    - table `cards`: cols=[id, artist, asciiName, availability, borderColor, cardKingdomFoilId, cardKingdomId, colorIdentity, colorIndicator, colors...]
    - table `sqlite_sequence`: cols=[name, seq]
- **DOC** `doc/legalities.md` (55.5KB)
- **DOC** `knowledge.md` (5.2KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['CAST(SUM(CASE WHEN T1.hasContentWarning = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id)']`
- first rows:
  - `['100.0']`

**正解までの reasoning steps**:
1. SQLite で `format='commander' AND legality='Legal'` の cards
2. hasContentWarning=0 の比率 × 100 = 100.0
= 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 63% perfect (= often-solved 中で最低)、巨大 SQLite (61 MB) で preamble 制約
- format string の case sensitivity (= 'commander' vs 'Commander') でフィルタずれ

**メモ**:
- exp_101 rich preamble で改善期待
- 投資優先度: 中 (= 巨大 SQL の preamble 改善で救済可)

---

### task_11  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.7, std=0.46, perfect_rate=70%, zero_rate=30%

**質問**:
> For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.

**入力データ**:
- **JSON** `json/Examination.json` (247.2KB)
- **JSON** `json/Patient.json` (244.0KB)
- **DOC** `knowledge.md` (5.2KB)

**Gold answer**:
- shape: **3 rows × 3 cols**
- header: `['ID', 'SEX', 'Diagnosis']`
- first rows:
  - `['163109', 'F', 'SLE']`
  - `['2803470', 'F', 'SLE']`
  - `['4395720', 'F', 'SLE']`

**正解までの reasoning steps**:
1. examination で `Thrombosis=2` (severe) の DISTINCT ID
2. patient_sex.csv で SEX、Patient.md で Diagnosis (= narrative parse)
= 3 行 × 3 col

**失敗モード / 過去 trace の傾向**:
- 70% perfect だが残り 30% で run-to-run variance (= padding bug, attempts diversity)
- 値はだいたい合うが行数誤差や extras で wext 罰
- exp_103 で pred 75r×7c の blow-up 観測 (= padding バグ顕在化)

**メモ**:
- padding fix (exp_093) で blow-up 解消の対象 task
- Patient.md の narrative parse が安定性を左右

---

### task_352  (hard, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.7615, std=0.4278, perfect_rate=76%, zero_rate=24%

**質問**:
> How many times was the budget in Advertisement for "Yearly Kickoff" meeting more than "October Meeting"?

**入力データ**:
- **CSV** `answer.csv` (0.0KB): shape sampled=[1, 1], cols=[times_more]
- **CSV** `csv/event.csv` (4.6KB): shape sampled=[42, 7], cols=[event_id, event_name, event_date, type, notes, location, status]
- **DOC** `doc/budget.md` (61.2KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `["CAST(SUM(CASE WHEN T2.event_name = 'Yearly Kickoff' THEN T1.amount ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.event_name = 'October Meeting' THEN T1.amount ELSE 0 END)"]`
- first rows:
  - `['2.727272727272727']`

**正解までの reasoning steps**:
1. budget で `category='Advertisement' AND event='Yearly Kickoff'` の amount SUM
2. 同 event='October Meeting' の amount SUM
3. 比 = 2.727
= 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 76% perfect、event_name の表記ぶれで分岐失敗
- gold col header が複雑 SQL CASE、agent は value だけ合えば救済

**メモ**:
- M-Schema 系 exp_088 で改善見せた task、preamble 改善対象

---

### task_257  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8051, std=0.3875, perfect_rate=78%, zero_rate=17%

**質問**:
> Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time.

**入力データ**:
- **SQLite** `db/postHistory.db` (269976.0KB): tables=['postHistory']
    - table `postHistory`: cols=[Id, PostHistoryTypeId, PostId, RevisionGUID, CreationDate, UserId, Text, Comment, UserDisplayName]
- **JSON** `json/posts.json` (162380.6KB)
- **JSON** `json/users.json` (19060.7KB)
- **DOC** `knowledge.md` (5.4KB)

**Gold answer**:
- shape: **1 rows × 2 cols**
- header: `['ViewCount', 'DisplayName']`
- first rows:
  - `['1708', 'mbq']`

**正解までの reasoning steps**:
1. posts で `Title='Computer Game Datasets'` の Id, ViewCount 取得
2. last edit / last poster の DisplayName を取得
= 1 行 × 2 col (`ViewCount, DisplayName`)

**失敗モード / 過去 trace の傾向**:
- 78% perfect、観測 fail: 1r × **3c** (`total_views, last_editor_display_name, last_poster_display_name`) — last editor と last poster の 2 つを agent が両方含めて wext
- 巨大 SQLite (139 MB) で preamble 影響

**メモ**:
- 質問 "posted it last time" は last poster 限定だが agent が editor も含める
- Rule 18 column verify で 3→2 col 削減可能

---

### task_249  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8026, std=0.3946, perfect_rate=79%, zero_rate=18%

**質問**:
> What is the average of the up votes and the average user age for users creating more than 10 posts?

**入力データ**:
- **SQLite** `db/users.db` (7444.0KB): tables=['users']
    - table `users`: cols=[Id, Reputation, CreationDate, DisplayName, LastAccessDate, WebsiteUrl, Location, AboutMe, Views, UpVotes...]
- **JSON** `json/posts.json` (162380.6KB)
- **DOC** `knowledge.md` (5.4KB)

**Gold answer**:
- shape: **1 rows × 2 cols**
- header: `['AVG(T1.UpVotes)', 'AVG(T1.Age)']`
- first rows:
  - `['182.2832618025751', '34.083333333333336']`

**正解までの reasoning steps**:
1. users で post 数 > 10 の UserId
2. AVG(UpVotes), AVG(Age) を 2 列で出力
= 1 行 × 2 col

**失敗モード / 過去 trace の傾向**:
- 79% perfect、列順 (UpVotes vs Age) の入れ替えで value match 失敗するケース
- 巨大 JSON file (166 MB) で preamble truncation 影響

**メモ**:
- exp_101 rich preamble の profile section で救済期待 task の 1 つ

---

### task_67  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8365, std=0.3613, perfect_rate=81%, zero_rate=15%

**質問**:
> What is the average weight of all female superheroes?

**入力データ**:
- **CSV** `csv/superhero.csv` (35.4KB): shape sampled=[750, 12], cols=[id, superhero_name, full_name, gender_id, eye_colour_id, hair_colour_id, skin_colour_id, race_id...]
- **JSON** `json/gender.json` (0.2KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['AVG(T1.weight_kg)']`
- first rows:
  - `['60.77956989247312']`

**正解までの reasoning steps**:
1. superhero で `gender='Female'` の id
2. weight_kg の AVG
= 1 行 × 1 col スカラー

**失敗モード / 過去 trace の傾向**:
- 81% perfect、観測 fail (exp_086): 1r × **2c** (`average_weight, average_weight_kg`) 重複
- 単位 conversion (lbs vs kg) で値ぶれもあり

**メモ**:
- 重複 col は Rule 18 column verify で解決可能
- gold col header `AVG(T1.weight_kg)` は SQL alias、value match で救済

---

### task_292  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8538, std=0.3546, perfect_rate=85%, zero_rate=15%

**質問**:
> For the constructor which got the highest point in the race No. 9 , what is its introduction website?

**入力データ**:
- **SQLite** `db/constructorResults.db` (168.0KB): tables=['constructorResults', 'sqlite_sequence']
    - table `constructorResults`: cols=[constructorResultsId, raceId, constructorId, points, status]
    - table `sqlite_sequence`: cols=[name, seq]
- **JSON** `json/constructors.json` (39.4KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['url']`
- first rows:
  - `['http://en.wikipedia.org/wiki/Red_Bull_Racing']`

**正解までの reasoning steps**:
1. results で raceId=9 の constructor で SUM(points) 最大
2. constructors で url 取得
= 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 85% perfect、ties や filter-back 抜けで稀に異 constructor

**メモ**:
- url 値が完全一致必要、URL format ぶれで失敗するケースもあり

---

### task_408  (hard, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8692, std=0.3307, perfect_rate=85%, zero_rate=12%

**質問**:
> How much faster in percentage is the champion than the driver who finished the race last in the 2008 Australian Grand Prix?

**入力データ**:
- **SQLite** `db/results.db` (984.0KB): tables=['results', 'sqlite_sequence']
    - table `results`: cols=[resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points...]
    - table `sqlite_sequence`: cols=[name, seq]
- **DOC** `doc/races.md` (84.2KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['(CAST((SELECT time_seconds FROM last_driver_incremental) AS REAL) * 100) / (SELECT time_seconds + (SELECT time_seconds FROM last_driver_incremental) FROM champion_time)']`
- first rows:
  - `['0.31555732286030097']`

**正解までの reasoning steps**:
1. results で 2008 Australian Grand Prix のチャンピオン (positionOrder=1) の time
2. 最終 driver (positionOrder=max) の time
3. (last - champion) / last × 100 計算
= 0.31555%

**失敗モード / 過去 trace の傾向**:
- 85% perfect、time の format parse + 計算式の組み立てで稀に miss

**メモ**:
- gold value 0.3155 は %, agent が round ぶれで誤差

---

### task_22  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8673, std=0.3384, perfect_rate=86%, zero_rate=13%

**質問**:
> State the date Connor Hilton paid his/her dues.

**入力データ**:
- **CSV** `csv/income.csv` (2.2KB): shape sampled=[36, 6], cols=[income_id, date_received, amount, source, notes, link_to_member]
- **JSON** `json/member.json` (9.9KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **2 rows × 1 cols**
- header: `['date_received']`
- first rows:
  - `['2019-10-02']`
  - `['2019-09-12']`

**正解までの reasoning steps**:
1. members で `full_name='Connor Hilton'` の member_id
2. dues / payments で支払日 (date_received) を取得
= 2 行 × 1 col (= 複数回支払いあり)

**失敗モード / 過去 trace の傾向**:
- 86% perfect、稀に 1 行のみ返して ties 1 つ取りこぼし

**メモ**:
- gold 2 行 = 複数回 dues 支払い、Rule 10 filter-back が効く

---

### task_27  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9023, std=0.2902, perfect_rate=87%, zero_rate=9%

**質問**:
> List out the full name and total cost that member id "rec4BLdZHS2Blfp4v" incurred?

**入力データ**:
- **JSON** `json/expense.json` (8.5KB)
- **JSON** `json/member.json` (9.9KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 3 cols**
- header: `['first_name', 'last_name', 'SUM(T2.cost)']`
- first rows:
  - `['Sacha', 'Harrison', '866.25']`

**正解までの reasoning steps**:
1. members で member_id=該当 の first_name, last_name
2. expense (or budget) で member_id=該当 の cost SUM
= 1 行 × 3 col

**失敗モード / 過去 trace の傾向**:
- 87% perfect、稀に SUM 値の ROUND ぶれ or 余分列

**メモ**:
- 安定性高い、列スコープが gold と一致しやすい

---

### task_243  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9077, std=0.2712, perfect_rate=87%, zero_rate=8%

**質問**:
> For the user No.24, how many times is the number of his/her posts compared to his/her votes?

**入力データ**:
- **CSV** `csv/votes.csv` (1022.0KB): shape sampled=[1000, 6], cols=[Id, PostId, VoteTypeId, CreationDate, UserId, BountyAmount]
- **SQLite** `db/posts.db` (139384.0KB): tables=['posts']
    - table `posts`: cols=[Id, PostTypeId, AcceptedAnswerId, CreaionDate, Score, ViewCount, Body, OwnerUserId, LasActivityDate, Title...]
- **DOC** `knowledge.md` (5.4KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['CAST(COUNT(DISTINCT T2.Id) AS REAL) / COUNT(DISTINCT T1.Id)']`
- first rows:
  - `['0.375']`

**正解までの reasoning steps**:
1. badges/posts で UserId=24 の post 数 (T1)、vote 数 (T2)
2. post / vote の比 = 0.375
= 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 87% perfect、観測 fail: extra col (= count of posts や count of votes 個別を含めるパターン)

**メモ**:
- 比率計算、Rule 18 column verify でほぼ完全救済

---

### task_250  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8846, std=0.3207, perfect_rate=88%, zero_rate=12%

**質問**:
> Which post by slashnick has the most answers count? State the post ID.

**入力データ**:
- **CSV** `csv/postHistory.csv` (223576.4KB): shape sampled=[1000, 9], cols=[Id, PostHistoryTypeId, PostId, RevisionGUID, CreationDate, UserId, Text, Comment...]
- **SQLite** `db/users.db` (7444.0KB): tables=['users']
    - table `users`: cols=[Id, Reputation, CreationDate, DisplayName, LastAccessDate, WebsiteUrl, Location, AboutMe, Views, UpVotes...]
- **JSON** `json/posts.json` (162380.6KB)
- **DOC** `knowledge.md` (5.4KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['PostId']`
- first rows:
  - `['351']`

**正解までの reasoning steps**:
1. users で `display_name='slashnick'` の UserId
2. posts で OwnerUserId=該当、AnswerCount 最大の post (filter-back)
3. PostId 出力
= 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 88% perfect、稀に PostId vs Id 列名混乱

**メモ**:
- 単純 filter-back + lookup、安定

---

### task_287  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8904, std=0.3113, perfect_rate=88%, zero_rate=11%

**質問**:
> Identify the gender of the superhero who has the ability of Phoenix Force.

**入力データ**:
- **CSV** `csv/hero_power.csv` (44.5KB): shape sampled=[1000, 2], cols=[hero_id, power_id]
- **SQLite** `db/superhero.db` (40.0KB): tables=['superhero']
    - table `superhero`: cols=[id, superhero_name, full_name, gender_id, eye_colour_id, hair_colour_id, skin_colour_id, race_id, publisher_id, alignment_id...]
- **JSON** `json/gender.json` (0.2KB)
- **JSON** `json/superpower.json` (10.8KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['gender']`
- first rows:
  - `['Female']`

**正解までの reasoning steps**:
1. hero_power で `power_name='Phoenix Force'` の hero_id
2. gender 取得 (= Female)
= 1 行 × 1 col

**失敗モード / 過去 trace の傾向**:
- 88% perfect、稀に gender 値の format ぶれ

**メモ**:
- 単純 lookup、稀な失敗のみ

---

### task_303  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.8962, std=0.3015, perfect_rate=88%, zero_rate=10%

**質問**:
> Among all European Grand Prix races, what is the percentage of the races were hosted in Germany?

**入力データ**:
- **SQLite** `db/races.db` (208.0KB): tables=['races', 'sqlite_sequence']
    - table `races`: cols=[raceId, year, round, circuitId, name, date, time, url]
    - table `sqlite_sequence`: cols=[name, seq]
- **JSON** `json/circuits.json` (20.5KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `["CAST(COUNT(CASE WHEN T1.country = 'Germany' THEN T2.circuitID END) AS REAL) * 100 / COUNT(T2.circuitId)"]`
- first rows:
  - `['52.17391304347826']`

**正解までの reasoning steps**:
1. races で `name='European Grand Prix'` の circuit
2. circuits で country='Germany' の数 / 全 circuit 数 × 100
= 52.17%

**失敗モード / 過去 trace の傾向**:
- 88% perfect、percentage 計算、Round ぶれで稀に value mismatch

**メモ**:
- gold col header = SQL CAST, value match で救済

---

### task_355  (hard, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9167, std=0.2681, perfect_rate=88%, zero_rate=8%

**質問**:
> Write the full name of the member who spent money for water, veggie tray and supplies and include the cost of it.

**入力データ**:
- **CSV** `csv/expense.csv` (2.9KB): shape sampled=[32, 7], cols=[expense_id, expense_description, expense_date, cost, approved, link_to_member, link_to_budget]
- **DOC** `doc/member.md` (32.5KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 3 cols**
- header: `['first_name', 'last_name', 'cost']`
- first rows:
  - `['Elijah', 'Allen', '28.15']`

**正解までの reasoning steps**:
1. expense で `expense_description LIKE '%water%veggie%supplies%'`
2. members join で first_name, last_name + cost
= 1 行 × 3 col

**失敗モード / 過去 trace の傾向**:
- 88% perfect、expense_description の正規表現マッチで稀に miss

**メモ**:
- 安定、稀な失敗のみ

---

## 🟢 常に解ける (always-solved, ≥90% perfect)  (17 task)

### task_19  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9, std=0.3012, perfect_rate=90%, zero_rate=10%

**質問**:
> List the full name of the Student_Club members that grew up in Illinois state.

**入力データ**:
- **CSV** `csv/member.csv` (3.5KB): shape sampled=[33, 9], cols=[member_id, first_name, last_name, email, position, t_shirt_size, phone, zip...]
- **CSV** `illinois_members.csv` (0.1KB): shape sampled=[3, 2], cols=[first_name, last_name]
- **JSON** `json/zip_code.json` (7268.8KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **3 rows × 2 cols**
- header: `['first_name', 'last_name']`
- first rows:
  - `['Trent', 'Smith']`
  - `['Tyler', 'Hewitt']`
  - `['Annabella', 'Warren']`

**正解までの reasoning steps**:
1. members.json で `state='Illinois'` の member_id 群を取得
2. first_name, last_name 列を出力
= 3 行 × 2 col

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- easy task、knowledge.md の state hint と member.json で完結。

---

### task_349  (hard, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9205, std=0.2684, perfect_rate=92%, zero_rate=8%

**質問**:
> What's Angela Sanders's major?

**入力データ**:
- **CSV** `csv/member.csv` (3.5KB): shape sampled=[33, 9], cols=[member_id, first_name, last_name, email, position, t_shirt_size, phone, zip...]
- **DOC** `doc/major.md` (73.2KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['major_name']`
- first rows:
  - `['Business']`

**正解までの reasoning steps**:
1. members で `first_name='Angela' AND last_name='Sanders'`
2. major join で major_name

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 単純 lookup、92% perfect。

---

### task_269  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9212, std=0.2678, perfect_rate=92%, zero_rate=8%

**質問**:
> What are the names of the superheroes with the power of death touch?

**入力データ**:
- **CSV** `csv/hero_power.csv` (44.5KB): shape sampled=[1000, 2], cols=[hero_id, power_id]
- **SQLite** `db/superhero.db` (40.0KB): tables=['superhero']
    - table `superhero`: cols=[id, superhero_name, full_name, gender_id, eye_colour_id, hair_colour_id, skin_colour_id, race_id, publisher_id, alignment_id...]
- **JSON** `json/superpower.json` (10.8KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **7 rows × 1 cols**
- header: `['superhero_name']`
- first rows:
  - `['Black Flash']`
  - `['Blackwulf']`
  - `['Hela']`

**正解までの reasoning steps**:
1. hero_power で `power_name='Death Touch'` の hero_id
2. superhero_name 列 = 7 行

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 92% perfect、複数行返答だが安定。

---

### task_194  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9231, std=0.2675, perfect_rate=92%, zero_rate=8%

**質問**:
> What are the bonds that have phosphorus and nitrogen as their atom elements?

**入力データ**:
- **CSV** `csv/connected.csv` (708.2KB): shape sampled=[1000, 3], cols=[atom_id, atom_id2, bond_id]
- **SQLite** `db/atom.db` (1352.0KB): tables=['atom', 'connected']
    - table `atom`: cols=[atom_id, molecule_id, element]
    - table `connected`: cols=[atom_id, atom_id2, bond_id]
- **DOC** `knowledge.md` (6.5KB)

**Gold answer**:
- shape: **6 rows × 1 cols**
- header: `['bond_id']`
- first rows:
  - `['TR032_2_3']`
  - `['TR032_3_5']`
  - `['TR058_1_3']`

**正解までの reasoning steps**:
1. atom で `element='p' OR element='n'` の atom_id
2. connected で 両 atom_id を含む bond を抽出 = 6 行

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- exp_088 M-Schema regression が起きた task の 1 つ、preamble 改造に注意。

---

### task_218  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9231, std=0.2675, perfect_rate=92%, zero_rate=8%

**質問**:
> What is the telephone number for the school with the lowest average score in reading in Fresno Unified?

**入力データ**:
- **SQLite** `db/satscores.db` (260.0KB): tables=['satscores']
    - table `satscores`: cols=[cds, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite...]
- **JSON** `json/schools.json` (25169.2KB)
- **DOC** `knowledge.md` (5.0KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['Phone']`
- first rows:
  - `['(559) 248-5100']`

**正解までの reasoning steps**:
1. satscores で `dname='Fresno Unified'` の AvgScrRead 最低 school
2. frpm or schools で Phone 取得

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- filter-back + cross-table join、92% perfect。

---

### task_261  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9231, std=0.2675, perfect_rate=92%, zero_rate=8%

**質問**:
> Among the superheroes with the super power of "Super Strength", how many of them have a height of over 200cm?

**入力データ**:
- **CSV** `csv/superhero.csv` (35.4KB): shape sampled=[750, 12], cols=[id, superhero_name, full_name, gender_id, eye_colour_id, hair_colour_id, skin_colour_id, race_id...]
- **SQLite** `db/hero_power.db` (72.0KB): tables=['hero_power']
    - table `hero_power`: cols=[hero_id, power_id]
- **JSON** `json/superpower.json` (10.8KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(T1.id)']`
- first rows:
  - `['56']`

**正解までの reasoning steps**:
1. hero_power で `power='Super Strength'` の hero_id
2. superhero で height > 200 の COUNT

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 92% perfect、フィルタ + COUNT。

---

### task_350  (hard, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9231, std=0.2675, perfect_rate=92%, zero_rate=8%

**質問**:
> Among the students from the Student_Club who attended the event "Women's Soccer", how many of them want a T-shirt that's in medium size?

**入力データ**:
- **CSV** `csv/member.csv` (3.5KB): shape sampled=[33, 9], cols=[member_id, first_name, last_name, email, position, t_shirt_size, phone, zip...]
- **SQLite** `db/attendance.db` (44.0KB): tables=['attendance']
    - table `attendance`: cols=[link_to_event, link_to_member]
- **DOC** `doc/event.md` (57.4KB)
- **DOC** `doc/event_event.md` (58.9KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(T1.event_id)']`
- first rows:
  - `['7']`

**正解までの reasoning steps**:
1. attendance で `event_name="Women's Soccer"`
2. members.t_shirt_size と join、特定 size の COUNT

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- multi-condition filter、92% perfect。

---

### task_214  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9308, std=0.2548, perfect_rate=93%, zero_rate=7%

**質問**:
> How many Brazilian Portuguese translated sets are inside the Commander block?

**入力データ**:
- **CSV** `csv/set_translations.csv` (42.8KB): shape sampled=[1000, 4], cols=[id, language, setCode, translation]
- **SQLite** `db/sets.db` (108.0KB): tables=['sets', 'sqlite_sequence']
    - table `sets`: cols=[id, baseSetSize, block, booster, code, isFoilOnly, isForeignOnly, isNonFoilOnly, isOnlineOnly, isPartialPreview...]
    - table `sqlite_sequence`: cols=[name, seq]
- **DOC** `knowledge.md` (5.2KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(T1.id)']`
- first rows:
  - `['7']`

**正解までの reasoning steps**:
1. set_translations で `language='Portuguese (Brazil)'`
2. sets で `block='Commander'`
3. join して COUNT

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- magic the gathering データ、93% perfect。

---

### task_74  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9385, std=0.2412, perfect_rate=94%, zero_rate=6%

**質問**:
> Provide the eye colour of the superhero who has Karen Beecher-Duncan as their full name.

**入力データ**:
- **CSV** `csv/superhero.csv` (35.4KB): shape sampled=[750, 12], cols=[id, superhero_name, full_name, gender_id, eye_colour_id, hair_colour_id, skin_colour_id, race_id...]
- **JSON** `json/colour.json` (1.9KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['colour']`
- first rows:
  - `['Brown']`

**正解までの reasoning steps**:
1. superhero で `full_name='Karen Beecher-Duncan'`
2. colour テーブル join で eye_colour 取得

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 単純 lookup、94% perfect。

---

### task_415  (hard, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9423, std=0.2299, perfect_rate=94%, zero_rate=5%

**質問**:
> What is the constructor reference name of the champion in the 2009 Singapore Grand Prix? Please give its website.

**入力データ**:
- **SQLite** `db/results.db` (984.0KB): tables=['results', 'sqlite_sequence']
    - table `results`: cols=[resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points...]
    - table `sqlite_sequence`: cols=[name, seq]
- **JSON** `json/constructors.json` (39.4KB)
- **DOC** `doc/races.md` (90.2KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 2 cols**
- header: `['constructorRef', 'url']`
- first rows:
  - `['mclaren', 'http://en.wikipedia.org/wiki/McLaren']`

**正解までの reasoning steps**:
詳細不明、データ未取得

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 94% perfect。

---

### task_24  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9462, std=0.2266, perfect_rate=95%, zero_rate=5%

**質問**:
> How many members attended the "Women's Soccer" event?

**入力データ**:
- **CSV** `csv/attendance.csv` (11.8KB): shape sampled=[326, 2], cols=[link_to_event, link_to_member]
- **JSON** `json/event.json` (10.9KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(T2.link_to_member)']`
- first rows:
  - `['17']`

**正解までの reasoning steps**:
1. event テーブルで `event_name="Women's Soccer"` の event_id 取得
2. attendance テーブルで `event_id=該当` の DISTINCT member 数 = COUNT

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- COUNT スカラー値、構造的に robust。

---

### task_75  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9462, std=0.2266, perfect_rate=95%, zero_rate=5%

**質問**:
> What is the surname of the driver with the best lap time in race number 19 in the second qualifying period?

**入力データ**:
- **CSV** `csv/qualifying.csv` (270.9KB): shape sampled=[1000, 9], cols=[qualifyId, raceId, driverId, constructorId, number, position, q1, q2...]
- **JSON** `json/drivers.json` (232.2KB)
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['surname']`
- first rows:
  - `['Räikkönen']`

**正解までの reasoning steps**:
1. qualifying で raceId=19 の q2 列で最小時間 (filter-back) → driverId
2. drivers で surname

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- Rule 10 filter-back が効く例、大体安定。

---

### task_145  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9462, std=0.2266, perfect_rate=95%, zero_rate=5%

**質問**:
> Among the events attended by more than 10 members of the Student_Club, how many of them are meetings?

**入力データ**:
- **CSV** `csv/attendance.csv` (11.8KB): shape sampled=[326, 2], cols=[link_to_event, link_to_member]
- **SQLite** `db/event.db` (20.0KB): tables=['event']
    - table `event`: cols=[event_id, event_name, event_date, type, notes, location, status]
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(*)']`
- first rows:
  - `['4']`

**正解までの reasoning steps**:
1. attendance で event ごとの member 数集計 > 10
2. event テーブル join で `type='Meeting'` の COUNT

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 多段 aggregation だが安定、95% perfect。

---

### task_283  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9564, std=0.1964, perfect_rate=95%, zero_rate=4%

**質問**:
> Calculate the percentage of superheroes with blue eyes.

**入力データ**:
- **SQLite** `db/superhero.db` (40.0KB): tables=['superhero']
    - table `superhero`: cols=[id, superhero_name, full_name, gender_id, eye_colour_id, hair_colour_id, skin_colour_id, race_id, publisher_id, alignment_id...]
- **JSON** `json/colour.json` (1.9KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `["CAST(COUNT(CASE WHEN T2.colour = 'Blue' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)"]`
- first rows:
  - `['31.2']`

**正解までの reasoning steps**:
1. colour テーブル join で eye colour='Blue' をカウント
2. ratio × 100 = percentage

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 95% perfect、percentage 計算は agent 安定。

---

### task_26  (easy, 簡潔)

**履歴統計**: n=174, λ0.5 mean=0.9655, std=0.183, perfect_rate=97%, zero_rate=3%

**質問**:
> How many members of the Student_Club have major in 'Physics Teaching'?

**入力データ**:
- **JSON** `json/major.json` (23.6KB)
- **JSON** `json/member.json` (9.9KB)
- **DOC** `knowledge.md` (5.3KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['COUNT(T2.member_id)']`
- first rows:
  - `['1']`

**正解までの reasoning steps**:
1. major テーブルで `major_name='Physics Teaching'` の major_id
2. members で `major_id=該当` の COUNT

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 単純 COUNT、agent が直感的に解決。

---

### task_64  (easy, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9692, std=0.1734, perfect_rate=97%, zero_rate=3%

**質問**:
> Please list all the superpowers of 3-D Man.

**入力データ**:
- **CSV** `csv/hero_power.csv` (44.5KB): shape sampled=[1000, 2], cols=[hero_id, power_id]
- **CSV** `csv/superpower.csv` (3.1KB): shape sampled=[167, 2], cols=[id, power_name]
- **JSON** `json/superhero.json` (234.6KB)
- **DOC** `knowledge.md` (5.5KB)

**Gold answer**:
- shape: **4 rows × 1 cols**
- header: `['power_name']`
- first rows:
  - `['Agility']`
  - `['Super Strength']`
  - `['Stamina']`

**正解までの reasoning steps**:
1. superhero で `superhero_name='3-D Man'` の id 取得
2. hero_power で hero_id=該当 の power_id
3. superpower で power_name 取得 = 4 行

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- exp_109 で 96% perfect、典型的 join task。

---

### task_305  (medium, 簡潔)

**履歴統計**: n=130, λ0.5 mean=0.9692, std=0.1734, perfect_rate=97%, zero_rate=3%

**質問**:
> What was the fastest lap speed among all drivers in the 2009 Spanish Grand Prix?

**入力データ**:
- **CSV** `csv/results.csv` (1200.3KB): shape sampled=[1000, 18], cols=[resultId, raceId, driverId, constructorId, number, grid, position, positionText...]
- **SQLite** `db/races.db` (208.0KB): tables=['races', 'sqlite_sequence']
    - table `races`: cols=[raceId, year, round, circuitId, name, date, time, url]
    - table `sqlite_sequence`: cols=[name, seq]
- **DOC** `knowledge.md` (5.9KB)

**Gold answer**:
- shape: **1 rows × 1 cols**
- header: `['fastestLapSpeed']`
- first rows:
  - `['202.484']`

**正解までの reasoning steps**:
1. races で `year=2009 AND name='Spanish Grand Prix'` の raceId
2. results.fastestLapSpeed の MAX

**失敗モード / 過去 trace の傾向**:
- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。

**メモ**:
- 97% perfect、MAX スカラー、安定。

---

## 横断的失敗パターン (Phase 3)

per-task 分析から抽出した失敗モードを類型化。各 task は 1 つ以上の type に該当する。

### Type A: 列スコープ違反 (= extra columns penalty)

**症状**: gold col 1 個に対し agent が 2-6 列出力、value match で recall=1.0 取れても extras_ratio で λ0.5 が大幅減点。

**該当 task** (代表):
- task_38 (140r×6c vs 1c) — withdrawal の 6 列全部
- task_180 (153r×2c vs 1c) — CustomerID + Consumption
- task_259 (1r×4c vs 1c) — comment + 関連 metadata 3 列
- task_67 (1r×2c vs 1c) — 重複 col `average_weight, average_weight_kg`
- task_196/200/243 — 重複 col の典型 (= 3-attempt union から残る)

**根本原因**: agent が "これも答えに関係ありそう" と関連列を含める / 3-attempt 間で異なる列名 → union 後に重複。

**救済軸**: Rule 18 column verify (= exp_109)、Rule 12 explicit SELECT cols 強化、3-att padding fix。

---

### Type B: 計算ロジック誤り (= formula step missed)

**症状**: 必要な計算 step (= 除算、CASE WHEN、二段集計) を抜く / 解釈違いで完全に値が外れる。

**該当 task**:
- task_169 (avg_monthly = annual/12 のはずが annual SUM そのまま)
- task_396 (% percentage 計算で母数違い)
- task_408 (last/(last+champion) の比率式で typo / 順序ミス)

**根本原因**: 長い数式の組み立てで 1 step 抜け、または「monthly = 月単位」「monthly = 月平均」等の意味解釈分岐。

**救済軸**: 計算結果の **オーダー比較 verify** (= 「億単位の値で 月平均か?」のサニティチェック)、knowledge.md の formula を **inline embed** で agent に強制参照。

---

### Type C: Filter scope 過剰 / 過少

**症状**: 条件式の組み立てで row 数が gold と乖離 (= 100x or 1/10x 級)。

**該当 task**:
- task_180 (Price>29 フィルタ抜け → 153 行 vs gold 9 行)
- task_199 (district 集計レベル間違い → 506 行 vs gold 6 行)
- task_344/418 (medical 閾値不明で 0 件 or 過剰)

**根本原因**: 多段フィルタの順序ぶれ + 条件解釈の曖昧さ + 数値閾値の knowledge gap。

**救済軸**: filter logic の rule-based verifier (= row count のオーダー想定値との比較)、医学/業界閾値の knowledge embed。

---

### Type D: Tied-rows 取りこぼし

**症状**: gold が 2-9 行 (tied results) なのに agent が 1 行返す。

**該当 task**:
- task_25 (lowest cost、3 行 tied)
- task_80 (qualifying time、2 driver tied)
- task_75/218 (filter-back で ties 含む)
- task_269/379 (list 系で 7 行 gold)

**根本原因**: SQL で LIMIT 1 / `==` 一致のみ、filter-back パターン未使用。row_count を 1 と guess して切り捨て。

**救済軸**: Rule 10 filter-back (`WHERE col = (SELECT MAX(col) FROM t)`) を SQL/Python 両方で強制。**row_count predeclaration を撤廃** (= exp_109 で実装)。

---

### Type E: 質問解釈の bias / ambiguity

**症状**: agent が文字通りに読むが gold は別解釈、または逆。

**該当 task**:
- task_163 ("type of expenses" = event の type vs expense category)
- task_25 (cost = expense vs budget.amount)
- task_173 (Country format = ISO code vs full name)

**根本原因**: 質問の自然言語が複数解釈、gold の選好が文脈依存。

**救済軸**: Rule 19 INTERPRET line (= exp_109)、複数解釈生成 → union (= cost expensive)。

---

### Type F: 大型ファイル + preamble truncation

**症状**: 巨大 CSV / SQLite (50 MB+) で sample preamble に列名や全体像が反映されず agent が schema 把握不能。

**該当 task**:
- task_259/420/249/250 (= comments.csv 47 MB, posts.db 139 MB 等)
- task_330 (= Match.csv 279 MB)
- task_38 (= trans.csv 58 MB)

**根本原因**: 旧 preamble の per-file cap (50 KB CSV / 20 KB JSON) で sample 行のみ、全 115 列の見渡しなし。

**救済軸**: rich preamble (= profile section、exp_101 で +0.045 確認)、column-list 明示。

---

### Type G: Knowledge gap (= 外部参照必要)

**症状**: 質問が業界標準値 (医学閾値、化学命名規則) を要求するが knowledge.md に記載なし。

**該当 task**:
- task_344 (WBC / FG normal range)
- task_418 (creatinine abnormal threshold)

**根本原因**: knowledge.md の data dictionary が不完全、agent が pre-training 知識で推測するも単位ぶれで失敗。

**救済軸**: knowledge enrichment (大改造)、または LLM precompute で外部標準値を rule で埋める。

---

### Type H: Multiprocessing union の row-padding バグ (= 既知 / fix 済)

**症状**: 3-attempt の row 数違いで `_signature_majority_merge` が None pad → CSV 書き出しで signature 破壊 → 正しい列も recall 0。

**該当 task**:
- task_11 (3 行 gold が 75 行 padding で死ぬ)
- task_379 (7 行 gold が 100 行 padding で死ぬ)
- task_25/259 (関連 padding シナリオ)

**根本原因**: `_signature_majority_merge` が `max_len` まで None pad する旧実装。

**救済軸**: ✅ **exp_093 padding fix** (= mode by attempt-count length group + shortest tie)。exp_109 に移植済、validated。

---

## 推奨改善軸 (Phase 3)

per-task 分析から導出した改善軸の **ROI ランキング**:

### Tier 1: 確実な改善 (= 実装コスト低 + 構造的効果)

| 軸 | 救済対象 (合計推定 +λ) | 状態 |
|---|---|---|
| **A1: padding fix (exp_093)** | task_11/379/25/259 (+0.05〜+0.08) | ✅ exp_109 で実装済、検証中 |
| **A2: Column verify (Rule 18, exp_109)** | task_38/259/67/196/200/243/257 (+0.04〜+0.07) | ✅ exp_109 で実装済 |
| **A3: row_count predeclaration 撤廃** | task_25/80/180 (+0.02〜+0.03) | ✅ exp_109 で実装済 |
| **A4: rich preamble (exp_101)** | task_259/330/420/249/38 (+0.02〜+0.04) | ✅ 単攻略 +0.045 確認、3-att はまだ |

→ exp_109 で全 Tier 1 が組み込まれた状態、n=3 結果で +0.04〜+0.10 の構造的勝ちが見えるかが本命。

### Tier 2: 中効果 (= 中規模実装、対象範囲限定)

| 軸 | 救済対象 (合計推定 +λ) | 実装コスト |
|---|---|---|
| **B1: 計算結果オーダー verify** | task_169/396/408 (+0.02〜+0.04) | 中 (= rule-based magnitude check + 1 turn re-prompt) |
| **B2: Filter logic verify** | task_180/199 (+0.02) | 中 (= row count vs question scope の妥当性チェック) |
| **B3: 質問解釈 union** | task_163/25/173 (+0.01〜+0.03) | 中-高 (= multi-interpret 並列実行) |
| **B4: filter-back 強制 (Rule 10 拡張)** | task_80 (+0.01) | 低 |

### Tier 3: 大改造 / 知識拡張

| 軸 | 救済対象 | 実装コスト |
|---|---|---|
| **C1: 医学/業界閾値 knowledge enrichment** | task_344/418 (+0.02〜+0.04) | 高 (= per-task knowledge curation) |
| **C2: doc/Patient.md narrative parse 強化** | task_344/418 + 数 task | 高 (= LLM-based extraction step 追加) |
| **C3: 多解釈 attempts diversification** | task_163/25 | 高 (= attempt-level prompt diversification) |

---

### 提出戦略への影響

- **Tier 1 全部組み込んだ exp_109** が n=3 で **0.78+ で着地できれば LB v3 候補**
- それでも **never-solved 4 task は Tier 3 でしか救えない** = LB の絶対上限を意識
- 50 task 中 **約 13 task (= rarely+never) が「上限近い改善が必要」** でこれらは Tier 1 単独では救済困難
- Tier 2 (B1/B2 = verify 系) を exp_110 候補として実装すると **+0.03〜+0.05 上乗せ可能性**

---

## 生成スクリプト

このファイルは下記スクリプト群で生成・更新可能:

```sh
.venv/bin/python scripts/analyze_all_tasks.py     # 履歴集計 → artifacts/task_analysis/per_task_history.json
.venv/bin/python scripts/build_task_analysis_md.py # 骨格生成 (= TOC + per-task 入力データ + Gold)
# scripts/_task_fills.py を編集して narrative 追記
.venv/bin/python scripts/fill_task_analysis.py    # narrative を MD に流し込む
```
