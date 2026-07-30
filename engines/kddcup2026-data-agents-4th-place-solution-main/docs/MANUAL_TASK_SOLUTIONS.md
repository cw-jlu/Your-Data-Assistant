# 50 タスク手解きレポート (2026-05-14 〜 2026-05-15)

各 public 50 task を手動で解き、gold と一致する答えを出した記録。

形式: **Q** / **diff (BIRD diff)** / **source** / **アプローチ** / **結果** / **メモ**

## 集計サマリ

| 結果 | Count | 内訳 |
|------|-------|------|
| ✅ gold 一致 | 46 / 50 | 全 easy (15) + medium (25) + hard 6 |
| ⚠️ context-insufficient / LLM extraction 必須 | 4 / 50 | task_173, task_344, task_396, task_418 |

**全 49/50 タスクは BIRD mini-dev / dev / train データから来ている**。1 タスク (task_199 phrasing 等) は微妙だが基本は BIRD 由来。

## 失敗パターン分類

### A. Context-insufficient (= public input data に不足)
- **task_173** (medium, debit_card_specializing): "June 2013 transactions" が transactions_1k.db に存在せず (= 1k subset は Aug 2012 のみ)。gold (CZE, SVK) は full BIRD で計算済み。エージェントは "gas station の全 country" 出力で偶然一致。
- **task_344** (hard, thrombosis_prediction): Patient.md prose は 100 ID のみ網羅、Laboratory には 302 distinct ID。約 200 patient の sex 情報なし。full BIRD では gold=4 だが public 入力では到達不可。

### B. LLM-extraction 必須 (= regex では不十分)
- **task_396** (hard, superhero): 750 hero × 3 prose セクション (codename/height/publisher) を ID anchor で結合。regex で 16/17 Marvel hero までは抽出可、最後の 1 件で off-by-one。
- **task_418** (extreme, thrombosis_prediction): Patient.md + Laboratory.md の 2 prose を ID anchor で結合 + CRE 閾値が knowledge.md に未記載 (= BIRD evidence の domain knowledge 必要)。

## エージェントが躓くパターン (10 件抽出)

| パターン | 例 | 罠 |
|----------|-----|------|
| BIRD evidence-only field semantics | task_25 (= MIN(cost) on expense, not budget.spent) | column 選択ミスで 0 検出 |
| description case-mismatch | task_257 ("Computer Game Datasets" vs "Computer game datasets") | exact match で 0 件 |
| connected directional rows | task_196 (= 各 bond が双方向で 2 行) | DISTINCT 漏れで AVG が 2x |
| BIRD 単価 vs 合計 | task_180 (= Price/Amount > 29, NOT Price>29) | 153 → 9 になる差 |
| budget amount vs spent | task_352 (= allocated 値 != spent 値) | 比率 2.72 → 2.25 |
| atom_id 数値ソート | task_379 (= "_4" 番目, csv 順では 'h' 混入) | 'h' が 4 番目に多発 |
| "Riverside-related" 解釈狭め | task_199 (= Riverside Unified のみ、Riverside Co Office 除外) | 5 校 → 12 校 |
| Patient.md subset 性 | task_344 (= prose 100/302 ID のみ) | sex 情報部分欠落 |
| transaction_1k subset 性 | task_173 (= 1k 行 Aug 2012 のみ) | June 2013 一致不能 |
| prose 多セクション結合 | task_396 (= 3 セクション ID anchor) | regex で 1-off |

## V6 / V7 提案: 修正効果が見込めるエリア

| 修正 | 該当タスク | 期待効果 |
|------|------------|----------|
| BIRD-style description fuzzy match (= case-insensitive substring) | task_257 など | +1 task |
| connected.csv 双方向対応 (= DISTINCT bond_id) | task_196 等 | knowledge.md 追記済 |
| 単価計算式 (= Price/Amount) のテンプレ化 | task_180 等 | 既存 evidence |
| atom_id 数値ソート (= LPAD or CAST) | task_379 | knowledge.md 1 行追記で対応 |
| prose 構造化抽出 sub-agent (= extract_structured) | task_396, task_418 | exp_127 で実装中 |
| amount vs spent 区別 prompt | task_352 等 | プロンプト 1 行 |

---

## task_11 (easy / BIRD simple, thrombosis_prediction)
**Q**: For patients with severe degree of thrombosis, list their ID, sex and disease.
**source**: Examination.json, Patient.json
**アプローチ**: BIRD evidence "severe = Thrombosis=2"。Examination.Thrombosis=2 の DISTINCT ID → Patient JOIN → DISTINCT (ID, SEX, Diagnosis).
**結果**: ✅ `(163109,F,SLE), (2803470,F,SLE), (4395720,F,SLE)`
**メモ**: knowledge.md sec 2 が "1=most severe, 2=severe" と書く → 両方含める罠あり。BIRD evidence の解釈固定が必須。

---

## task_19 (easy / BIRD simple, student_club)
**Q**: List the full name of the Student_Club members that grew up in Illinois state.
**source**: csv/member.csv, json/zip_code.json
**アプローチ**: zip_code.state='Illinois' の zip 集合 → member.zip が含まれる人の (first_name, last_name).
**結果**: ✅ `Trent Smith, Tyler Hewitt, Annabella Warren`
**メモ**: zip → state 変換が必要、member 単体では解けない。

---

## task_22 (easy / BIRD simple, student_club)
**Q**: State the date Connor Hilton paid his/her dues.
**source**: csv/income.csv, json/member.json
**アプローチ**: member で Connor Hilton の id → income で link_to_member=id AND source='Dues' の date_received.
**結果**: ✅ `2019-09-12, 2019-10-02` (= 2 回支払い)
**メモ**: source 'Dues' フィルタ必須 (= 他 source 種類あり)。

---

## task_24 (easy / BIRD simple, student_club)
**Q**: How many members attended the "Women's Soccer" event?
**source**: csv/attendance.csv, json/event.json
**アプローチ**: event_name="Women's Soccer" の event_id → attendance で link_to_event=id の COUNT.
**結果**: ✅ `17`
**メモ**: DISTINCT 不要 (= gold は COUNT)。

---

## task_25 (easy / BIRD simple, student_club)
**Q**: Which event has the lowest cost?
**source**: csv/budget.csv, json/event.json, json/expense.json
**アプローチ**: expense.cost で MIN → 該当 expense の link_to_budget → budget の link_to_event → event_name. **3 event tie at $6.00**.
**結果**: ✅ `November Speaker, October Speaker, September Speaker`
**メモ**: 初回 budget.spent を SUM した → 全 event 0.0 で誤答。正解は **expense.cost (= 個別 expense 行)** の MIN。BIRD evidence: "lowest cost means MIN(cost)"。gold SQL は `ORDER BY T3.cost LIMIT 1` だが gold csv は 3 行 (= 同 cost 値共存)。

---

## task_26 (easy / BIRD simple, student_club)
**Q**: How many members of the Student_Club have major in 'Physics Teaching'?
**source**: json/major.json, json/member.json
**アプローチ**: major.major_name='Physics Teaching' の major_id → member で link_to_major=id COUNT.
**結果**: ✅ `1`
**メモ**: 単純 lookup + count.

---

## task_27 (easy / BIRD simple, student_club)
**Q**: List out the full name and total cost that member id "rec4BLdZHS2Blfp4v" incurred?
**source**: json/member.json, json/expense.json
**アプローチ**: member で id → expense.link_to_member=id の cost 合計。
**結果**: ✅ `Sacha, Harrison, 866.25`
**メモ**: 単純 GROUP BY 集計。

---

## task_38 (easy / BIRD simple, financial)
**Q**: List all the withdrawals in cash transactions that the client with the id 3356 makes.
**source**: csv/trans.csv, csv/disp.csv
**アプローチ**: disp.client_id=3356 → account_id (= 2779) → trans.account_id=2779 AND type='VYDAJ' AND operation='VYBER'.
**結果**: ✅ 140 trans_ids (816173 .. 816342)
**メモ**: "withdrawal in cash" = `type='VYDAJ' AND operation='VYBER'`。client→account→trans の 2 段 JOIN。

---

## task_64 (easy / BIRD simple, superhero)
**Q**: Please list all the superpowers of 3-D Man.
**source**: json/superhero.json, csv/hero_power.csv, csv/superpower.csv
**アプローチ**: superhero で 3-D Man id (=1) → hero_power.hero_id=1 の power_ids → superpower.id JOIN で power_name.
**結果**: ✅ `Agility, Super Strength, Stamina, Super Speed`
**メモ**: superhero.id は int 型なので str 比較注意。

---

## task_67 (easy / BIRD simple, superhero)
**Q**: What is the average weight of all female superheroes?
**source**: csv/superhero.csv, json/gender.json
**アプローチ**: gender.gender='Female' の id → superhero.gender_id=id の weight_kg 平均 (= AVG)。
**結果**: ✅ `60.77956989247312`
**メモ**: weight_kg=0 のレコードも含める (= BIRD AVG は NULL のみ除外、0 は含む)。n=186 全件で計算。

---

## task_74 (easy / BIRD simple, superhero)
**Q**: Provide the eye colour of the superhero who has Karen Beecher-Duncan as their full name.
**source**: csv/superhero.csv, json/colour.json
**アプローチ**: superhero.full_name='Karen Beecher-Duncan' → eye_colour_id → colour.id JOIN。
**結果**: ✅ `Brown`
**メモ**: 単純 2 段 JOIN。

---

## task_75 (easy / BIRD simple, formula_1)
**Q**: What is the surname of the driver with the best lap time in race number 19 in the second qualifying period?
**source**: csv/qualifying.csv, json/drivers.json
**アプローチ**: qualifying.raceId=19, q2 列 (= 2nd qual period) の min time → driverId → drivers.surname。
**結果**: ✅ `Räikkönen`
**メモ**: q2 を時間 string でソート (`M:SS.mmm` → 秒換算)。NULL/'\\N' 除外。

---

## task_80 (easy / BIRD simple, formula_1)
**Q**: What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903?
**source**: csv/qualifying.csv, json/drivers.json
**アプローチ**: qualifying.raceId=903, q3 LIKE '1:54%' → driverId → drivers.number。**2 driver tie at 1:54.x**.
**結果**: ✅ `3, 5`
**メモ**: "0:01:54" は H:MM:SS 表記、qualifying.q3 は M:SS 表記 → `1:54.xxx` で start。tied 2 driver。

---

## task_86 (easy / BIRD simple, formula_1)
**Q**: Which race was Alex Yoong in when he was in track number less than 20?
**source**: csv/driverStandings.csv, csv/races.csv, json/drivers.json
**アプローチ**: drivers.forename='Alex' AND surname='Yoong' → driverId → driverStandings.position<20 の raceId → races.name DISTINCT。
**結果**: ✅ 16 races (Australian, Austrian, Belgian, Brazilian, British, Canadian, European, French, German, Hungarian, Italian, Malaysian, Monaco, San Marino, Spanish, United States GP)
**メモ**: "track number" は driverStandings.position (= 順位)。results.csv はこの task に含まれていない点に注意。

---

## task_89 (easy / BIRD simple, formula_1)
**Q**: What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?
**source**: csv/results.csv, json/races.json
**アプローチ**: races.year=2008 AND name='Chinese Grand Prix' → raceId → results.raceId=id AND rank=2 → time。
**結果**: ✅ `+16.445`
**メモ**: BIRD evidence "ranked second = rank=2"。results.rank 列は完走者順位。

---

## task_145 (medium / BIRD moderate, student_club)
**Q**: Among the events attended by more than 10 members of the Student_Club, how many of them are meetings?
**source**: csv/attendance.csv, db/event.db
**アプローチ**: attendance.link_to_event で COUNT > 10 の event_id 集合 (=14) → event.type='Meeting' の交集合 COUNT.
**結果**: ✅ `4`
**メモ**: HAVING COUNT > 10 = strict, =10 含まず。

---

## task_163 (medium / BIRD moderate, student_club)
**Q**: Identify the type of expenses and their total value approved for 'October Meeting' event.
**source**: csv/expense.csv, json/budget.json, db/event.db
**アプローチ**: event_name='October Meeting' → event.type='Meeting'。budget.link_to_event の budget_ids → expense.link_to_budget AND approved='true' の SUM(cost)。
**結果**: ✅ `Meeting, 175.39`
**メモ**: 罠: "type of expenses" は event.type を返す (= "Meeting")、budget.category (= Advertisement/Food) ではない。approved=true フィルタ必須。

---

## task_169 (medium / BIRD moderate, debit_card_specializing)
**Q**: What was the average monthly consumption of customers in SME for the year 2013?
**source**: csv/yearmonth.csv, db/customers.db
**アプローチ**: customers.Segment='SME' の CustomerID → yearmonth.Date LIKE '2013%' AND CustomerID∈SME の AVG(Consumption) / 12.
**結果**: ✅ `459.9562642870894`
**メモ**: knowledge.md `Average Monthly Consumption = Total Annual / 12` 公式。Date は 'YYYYMM' 文字列。/12 は最後に。

---

## task_173 (medium / BIRD moderate, debit_card_specializing)
**Q**: Please list the countries of the gas stations with transactions taken place in June, 2013.
**source**: csv/yearmonth.csv, json/gasstations.json, db/transactions_1k.db
**アプローチ**: gold = `CZE, SVK`。**transactions_1k.db には 2012-08 のみで June 2013 データなし** (= 1k sample subset)。
**結果**: ⚠️ context insufficient: 厳密 SQL では空集合。gasstations の全 country (= CZE, SVK 2 種) が gold 正解と一致。
**メモ**: BIRD gold は full transactions table で計算済み、サンプル DB では再現不可能。エージェントは「gasstations の全 country」を返すしかない (= ヒューリスティック)。

---

## task_180 (medium / BIRD moderate, debit_card_specializing)
**Q**: For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.
**source**: csv/yearmonth.csv, db/transactions_1k.db
**アプローチ**: transactions_1k.ProductID=5 AND `Price/Amount > 29` の DISTINCT CustomerID → yearmonth.Date='201208' AND CustomerID∈cust の Consumption.
**結果**: ✅ 9 rows (1903.2, 88265.39, 1129.2, 126157.7, 58.19, 1142.95, 8878.07, 69331.72, 45937.22)
**メモ**: 罠: "paid more than 29 per unit" は **Price/Amount > 29** (= 単価)、Price > 29 ではない。Price=合計, Amount=数量。

---

## task_194 (medium / BIRD moderate, toxicology)
**Q**: What are the bonds that have phosphorus and nitrogen as their atom elements?
**source**: csv/connected.csv, db/atom.db
**アプローチ**: connected.csv (atom_id, atom_id2, bond_id) → bond ごとに atom set 構築 → 'p' AND 'n' 両方含む bond_id を抽出。
**結果**: ✅ 6 bonds (TR032_2_3, TR032_3_5, TR058_1_3, TR058_1_4, TR058_1_5, TR298_1_5)
**メモ**: element は lowercase ('p', 'n'); 1 bond = 2 directional rows なので set 構築必要。

---

## task_196 (medium / BIRD moderate, toxicology)
**Q**: What is the average number of bonds the atoms with the element iodine have?
**source**: csv/connected.csv, db/atom.db
**アプローチ**: iodine atoms (= element='i', 6 件) → connected で iodine 関与 bonds の **distinct** count (= 6) → 6/6 = 1.0.
**結果**: ✅ `1.0`
**メモ**: connected.csv は (atom_id, atom_id2) 両方向行入り → distinct bond_id で重複除去必須 (= 重複すると 2.0 になる)。

---

## task_199 (medium / BIRD moderate, california_schools)
**Q**: List the names and funding types of schools from Riverside-related school districts where the average SAT math score across schools exceeds 400.
**source**: csv/frpm.csv, db/satscores.db
**アプローチ**: satscores で `rtype='S' AND dname='Riverside Unified' AND AvgScrMath>400` → cds → frpm.CDSCode JOIN で 'Charter Funding Type'.
**結果**: ✅ 5 schools (Arlington High, John W. North High, Martin Luther King Jr. High, Polytechnic High, Ramona High) 全 CFT 空欄
**メモ**: 罠: "Riverside-related" = `dname='Riverside Unified'` のみ (= "Riverside County Office of Education" は除外)。"avg SAT math across schools exceeds 400" は per-school filter (= 各校 AvgScrMath>400)。CFT は非 charter 校で空欄。

---

## task_200 (medium / BIRD moderate, toxicology)
**Q**: Calculate the total atoms with triple-bond molecules containing the element phosphorus or bromine.
**source**: csv/atom.csv, json/molecule.json, db/bond.db
**アプローチ**: bond.bond_type='#' (triple) の DISTINCT molecule_id (= 4) → atom.csv で molecule_id ∈ triple AND element ∈ {'p','br'} の COUNT.
**結果**: ✅ `1` (= TR499_2, element='p')
**メモ**: "total atoms with triple-bond molecules" → atom in triple-bond molecule。"containing the element P or Br" → atom 自身の element が P/Br。triple-bond molecules = TR041/TR377/TR447/TR499。

---

## task_214 (medium / BIRD moderate, card_games)
**Q**: How many Brazilian Portuguese translated sets are inside the Commander block?
**source**: csv/set_translations.csv, db/sets.db
**アプローチ**: sets.block='Commander' の code 集合 (= 23) → set_translations.language='Portuguese (Brazil)' AND setCode ∈ commander の COUNT.
**結果**: ✅ `7`
**メモ**: 罠: set_translations の join key は `setCode` (= sets.code)、`id` ではない。

---

## task_218 (medium / BIRD moderate, california_schools)
**Q**: What is the telephone number for the school with the lowest average score in reading in Fresno Unified?
**source**: json/schools.json, db/satscores.db
**アプローチ**: satscores で `rtype='S' AND dname='Fresno Unified'` の MIN(AvgScrRead) → cds → schools.CDSCode JOIN で Phone。
**結果**: ✅ `(559) 248-5100` (= McLane High, AvgScrRead=370)
**メモ**: AvgScrRead IS NOT NULL の filter 重要。

---

## task_243 (medium / BIRD moderate, codebase_community)
**Q**: For the user No.24, how many times is the number of his/her posts compared to his/her votes?
**source**: csv/votes.csv, db/posts.db
**アプローチ**: posts.OwnerUserId=24 → DISTINCT post count (= 3)。votes.UserId=24 → DISTINCT vote count (= 8)。比率 = 3/8 = 0.375.
**結果**: ✅ `0.375`
**メモ**: gold header `COUNT(posts) / COUNT(votes)` → posts/votes。逆比率にしない。

---

## task_249 (medium / BIRD moderate, codebase_community)
**Q**: What is the average of the up votes and the average user age for users creating more than 10 posts?
**source**: json/posts.json, db/users.db
**アプローチ**: posts で OwnerUserId GROUP BY HAVING COUNT > 10 (= 1166 user) → users.Id ∈ big の AVG(UpVotes), AVG(Age)。
**結果**: ✅ `182.2832618025751, 34.083333333333336`
**メモ**: Age NULL 除外で n=24 程度に絞られる (= AVG(Age) 計算用)、UpVotes NULL 除外で n=1165。

---

## task_250 (medium / BIRD moderate, codebase_community)
**Q**: Which post by slashnick has the most answers count? State the post ID.
**source**: csv/postHistory.csv, json/posts.json, db/users.db
**アプローチ**: users.DisplayName='slashnick' → Id=16 → posts.OwnerUserId=16 → 1 件のみ (= Id=351)。
**結果**: ✅ `351`
**メモ**: slashnick は posts 1 件のみ、答えは自明。AnswerCount=None でも MAX 取り。

---

## task_257 (medium / BIRD moderate, codebase_community)
**Q**: Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it last time.
**source**: json/posts.json, json/users.json, db/postHistory.db
**アプローチ**: posts.Title='Computer game datasets' (= lowercase 'g'! のtitle) → Id=8222, ViewCount=1708 → postHistory で PostId=8222 の最新 UserId (=88) → users.Id=88.DisplayName.
**結果**: ✅ `1708, mbq`
**メモ**: 罠: 質問 "Computer Game Datasets" は実 title "Computer game datasets" と case mismatch → 大文字小文字無視 or 部分一致必須。

---

## task_259 (medium / BIRD moderate, codebase_community)
**Q**: Among the posts with views ranging from 100 to 150, what is the comment with the highest score?
**source**: csv/comments.csv, db/posts.db
**アプローチ**: posts.ViewCount BETWEEN 100 AND 150 (= 5088 件) → comments.PostId ∈ 集合 (= 11817 件) → MAX Score の Text。
**結果**: ✅ "Welcome to Cross Validated David P, ..." (Score=14)
**メモ**: BETWEEN は inclusive。Text を完全一致比較するので 1 件目を Score 降順 LIMIT 1 で取得。

---

## task_261 (medium / BIRD moderate, superhero)
**Q**: Among the superheroes with the super power of "Super Strength", how many of them have a height of over 200cm?
**source**: csv/superhero.csv, json/superpower.json, db/hero_power.db
**アプローチ**: superpower.power_name='Super Strength' (= id=18) → hero_power の DISTINCT hero_id (= 358) → superhero.height_cm > 200 で COUNT。
**結果**: ✅ `56`
**メモ**: > 200 (strict)。NULL/'' は除外。

---

## task_269 (medium / BIRD moderate, superhero)
**Q**: What are the names of the superheroes with the power of death touch?
**source**: csv/hero_power.csv, json/superpower.json, db/superhero.db
**アプローチ**: superpower.power_name='Death Touch' (= id=37) → hero_power の hero_id (= 7 件) → superhero.id JOIN で name。
**結果**: ✅ 7 heroes (Black Flash, Blackwulf, Hela, Living Tribunal, One-Above-All, Poison Ivy, Spectre)
**メモ**: power_name は title-case ('Death Touch')、質問の "death touch" lowercase でも対応。

---

## task_283 (medium / BIRD moderate, superhero)
**Q**: Calculate the percentage of superheroes with blue eyes.
**source**: json/colour.json, db/superhero.db
**アプローチ**: colour.colour='Blue' (= id=7) → superhero.eye_colour_id=7 の COUNT / total * 100 = 234 / 750 * 100 = 31.2。
**結果**: ✅ `31.2`
**メモ**: 単純比率計算、NULL は eye_colour_id 0 か別 id で除外不要。

---

## task_287 (medium / BIRD moderate, superhero)
**Q**: Identify the gender of the superhero who has the ability of Phoenix Force.
**source**: csv/hero_power.csv, json/gender.json, json/superpower.json, db/superhero.db
**アプローチ**: superpower.power_name='Phoenix Force' (= id=163) → hero_power の hero_id (= 534, Phoenix) → superhero.gender_id → gender.gender。
**結果**: ✅ `Female`
**メモ**: 1 hero のみ (Phoenix)。gender_id=1 が Female。

---

## task_292 (medium / BIRD moderate, formula_1)
**Q**: For the constructor which got the highest point in the race No. 9, what is its introduction website?
**source**: json/constructors.json, db/constructorResults.db
**アプローチ**: constructorResults.raceId=9 ORDER BY points DESC LIMIT 1 → constructorId=9 → constructors.url。
**結果**: ✅ `http://en.wikipedia.org/wiki/Red_Bull_Racing`
**メモ**: points DESC (= 18.0 が最大)。

---

## task_303 (medium / BIRD moderate, formula_1)
**Q**: Among all European Grand Prix races, what is the percentage of the races were hosted in Germany?
**source**: json/circuits.json, db/races.db
**アプローチ**: races.name='European Grand Prix' (= 23 races) → circuitId 集合 → circuits.country='Germany' に該当する数 (= 12) / 23 * 100.
**結果**: ✅ `52.17391304347826`
**メモ**: EGP は色んな国で開催 → Germany 比率を計算。

---

## task_305 (medium / BIRD moderate, formula_1)
**Q**: What was the fastest lap speed among all drivers in the 2009 Spanish Grand Prix?
**source**: csv/results.csv, db/races.db
**アプローチ**: races.year=2009 AND name='Spanish Grand Prix' → raceId=5 → results.fastestLapSpeed の MAX (NULL/'\\N' 除外)。
**結果**: ✅ `202.484`
**メモ**: '\\N' は NULL 表現。CAST/REAL ソート必須。

---

## task_330 (hard / BIRD challenging, european_football_2)
**Q**: What was the final score for the match on September 24, 2008, in the Belgian Jupiler League between the home team and the away team?
**source**: csv/Match.csv, doc/League.md
**アプローチ**: League.md から "Belgium Jupiler League" の registry code (= league_id=1) を抽出 → Match.date='2008-09-24' AND league_id=1 → home_team_goal, away_team_goal。
**結果**: ✅ `1, 1`
**メモ**: League.md prose 解析必須 (= 専用 league テーブル無し)。"Belgian" vs "Belgium" でゆるめに正規表現必要。

---

## task_344 (hard / BIRD challenging, thrombosis_prediction)
**Q**: Among the male patients who have a normal level of white blood cells, how many of them have an abnormal fibrinogen level?
**source**: csv/Laboratory.csv, doc/Patient.md
**アプローチ**: Patient.md prose → male IDs 抽出。Lab で normal WBC (3.5-9.0) AND abnormal FG (<150 OR >400)。
**結果**: ⚠️ best-attempt = 3 (gold 4)。**Patient.md は 100 ID のみ網羅 (全 male)、Lab には 302 distinct ID あり (= 21 lab patients が male confirmed)**。
**メモ**: context insufficient: 残り 281 lab IDs の sex 不明 (prose は subset のみ). full BIRD Patient table では gold=4。Lab 閾値 (WBC 3.5-9, FG<150/>400) も BIRD evidence からのみ取得可能。

---

## task_349 (hard / BIRD challenging, student_club)
**Q**: What's Angela Sanders's major?
**source**: csv/member.csv, doc/major.md
**アプローチ**: member.first_name='Angela' AND last_name='Sanders' → link_to_major (= recxK3MHQFbR9J5uO) → major.md prose で該当 ID の major_name (= Business)。
**結果**: ✅ `Business`
**メモ**: major.md prose 解析: ID → "Business (Registry ID: recxK3MHQFbR9J5uO)" 形式。

---

## task_350 (hard / BIRD challenging, student_club)
**Q**: Among the students from the Student_Club who attended the event "Women's Soccer", how many of them want a T-shirt that's in medium size?
**source**: csv/member.csv, db/attendance.db, doc/event.md, doc/event_event.md
**アプローチ**: event.md prose で "Women's Soccer" → event_id=rec2N69DMcrqN9PJC → attendance.db で link_to_event=id の member_ids (= 17) → member.t_shirt_size='Medium' の COUNT。
**結果**: ✅ `7`
**メモ**: event.md は narrative prose、Apostrophe 表記注意 ('s vs 's)。t_shirt_size は大文字小文字注意 ("Medium" 正確)。

---

## task_352 (hard / BIRD challenging, student_club)
**Q**: How many times was the budget in Advertisement for "Yearly Kickoff" meeting more than "October Meeting"?
**source**: csv/event.csv, doc/budget.md
**アプローチ**: event.csv で YK/OM の event_id 取得 → budget.md prose で **Advertisement** category かつ event_id 紐付けの budget レコード抽出 → **amount field (= "allocated"/budgeted 値)** の SUM 比率 → 150/55 = 2.7272.
**結果**: ✅ `2.727272727272727`
**メモ**: **罠**: amount=allocated value (= 55, 150)、spent value (= 54.25, 122.06) ではない。"was allocated 55" / "revised upward in the final budget to an amount of 150" 表現要解析。

---

## task_355 (hard / BIRD challenging, student_club)
**Q**: Write the full name of the member who spent money for water, veggie tray and supplies and include the cost of it.
**source**: csv/expense.csv, doc/member.md
**アプローチ**: expense.expense_description='Water, Veggie tray, supplies' (= 1 row) → link_to_member=recro8T1MPMwRadVH, cost=28.15 → member.md prose で該当 ID = "Elijah Allen"。
**結果**: ✅ `Elijah, Allen, 28.15`
**メモ**: description は AND ではなく "Water, Veggie tray, supplies" 全部一行 (= 単一 expense item)。

---

## task_379 (hard / BIRD challenging, toxicology)
**Q**: Tally the toxicology element of the 4th atom of each molecule that was carcinogenic.
**source**: carcinogenic_mols.txt, csv/atom.csv, doc/molecule.md
**アプローチ**: carcinogenic_mols.txt (= 100 molecules) → atom.csv で molecule ごとに atom_id 数値ソート (= TR000_1, TR000_2, ...) → 4 番目の atom.element → DISTINCT。
**結果**: ✅ 7 elements (c, br, cl, s, o, n, f)
**メモ**: 罠: atom_id ソートは **数値順** (TR000_1 < TR000_10)、CSV 行順 or alpha ソートは間違い (= 'h' が混入)。SELECT DISTINCT element (= 順序不問)。

---

## task_396 (hard / BIRD challenging, superhero)
**Q**: In superheroes with height between 150 to 180, what is the percentage of heroes published by Marvel Comics?
**source**: json/publisher.json, doc/superhero.md
**アプローチ**: superhero.md 3 セクション (codename / biometrics / publisher affiliation) を ID anchor で結合 → height 150-180 (= 31) → publisher_id=13 (Marvel) フィルタ → 17/31 = 54.84%.
**結果**: ⚠️ best-attempt 51.6% (= 16/31)、gold 54.84% (= 17/31). regex 抽出で 1 Marvel hero 漏れ。LLM 抽出必須。
**メモ**: 750 hero × 3 セクション prose → 構造化抽出。Marvel id=13、heights between (inclusive)。regex は heroes 全数カバー困難 (sub-agent extract_structured 推奨)。

---

## task_408 (hard / BIRD challenging, formula_1)
**Q**: How much faster in percentage is the champion than the driver who finished the race last in the 2008 Australian Grand Prix?
**source**: db/results.db, doc/races.md
**アプローチ**: races.md prose で "Australian Grand Prix" + 2008 → raceId=18 → results.raceId=18 ORDER BY positionOrder。champion (pos=1, time="1:34:50.616" = 5690.616s)、last finishing (pos=5, gap="+18.014") → gap*100/(champion+gap) = 18.014/5708.630 = 0.31555...
**結果**: ✅ `0.31555732286030097`
**メモ**: "last" = 完走者の最後 (pos 6-22 は DNF/None time)。incremental = 5 位の gap, total = champion + gap。

---

## task_415 (hard / BIRD challenging, formula_1)
**Q**: What is the constructor reference name of the champion in the 2009 Singapore Grand Prix? Please give its website.
**source**: json/constructors.json, db/results.db, doc/races.md
**アプローチ**: races.md prose "Singapore Grand Prix (Race ID: 14)" + 2009 → raceId=14 → results.raceId=14 AND positionOrder=1 の constructorId=1 → constructors で constructorRef + url。
**結果**: ✅ `mclaren, http://en.wikipedia.org/wiki/McLaren`
**メモ**: races.md は "(Race ID: N)" 明示 → 抽出容易。

---

## task_418 (extreme / BIRD challenging, thrombosis_prediction)
**Q**: Among the patients whose creatinine level is abnormal, how many of them aren't 70 yet?
**source**: doc/Laboratory.md, doc/Patient.md
**アプローチ**: Patient.md (= 92 patients prose) → ID, sex, birthyear 抽出。Laboratory.md (= 1696 lines prose) → CRE 抽出 → 異常 CRE patient → 年齢計算 (= 2026 - 生年月日年) < 70 で COUNT。
**結果**: ⚠️ best-attempt = 1 (gold 1)、ただし abnormal CRE 閾値が BIRD evidence では不明 (= CRE>=1.2 で gold 一致、CRE>=1.5 で 0)。LLM extraction 必須レベル。
**メモ**: 2 つの prose doc を ID anchor で結合。age = 2026-birth_year (= 現在年基準)、< 70 strict。BIRD 閾値は機関ごとに違うが knowledge.md 記載なし。

---

## task_420 (hard / BIRD challenging, card_games)
**Q**: What percentage of cards with format commander and legal status do not have a content warning?
**source**: db/cards.db, doc/legalities.md
**アプローチ**: legalities.md prose で 各 ID の format/status と linkage to cards_id 抽出 → Commander+Legal の cards_id 集合 (= 93) → cards.db で hasContentWarning=0 ratio。
**結果**: ✅ `100.0` (= 全 93 cards に warning なし)
**メモ**: legalities.md は ID と cards_id を別パラグラフで分離記述 → 2 stage join 必要。Commander+Legal はほぼ全 commander format card がカバー。

---
