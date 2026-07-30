"""Per-task analysis content for PUBLIC_TASK_ANALYSIS.md.

Each entry: task_id → {steps, failure, memo}, all multi-line markdown.
Edit this dict and re-run scripts/fill_task_analysis.py to regenerate.
"""

_BRIEF_SOLVED = "- 90%+ で perfect、構造的問題なし。1-2 attempt で正解可能。"

FILLS: dict[str, dict[str, str]] = {
    # === always-solved (17 tasks) — brief notes ===
    "task_19": {
        "steps": "1. members.json で `state='Illinois'` の member_id 群を取得\n2. first_name, last_name 列を出力\n= 3 行 × 2 col",
        "failure": _BRIEF_SOLVED,
        "memo": "- easy task、knowledge.md の state hint と member.json で完結。",
    },
    "task_24": {
        "steps": "1. event テーブルで `event_name=\"Women's Soccer\"` の event_id 取得\n2. attendance テーブルで `event_id=該当` の DISTINCT member 数 = COUNT",
        "failure": _BRIEF_SOLVED,
        "memo": "- COUNT スカラー値、構造的に robust。",
    },
    "task_26": {
        "steps": "1. major テーブルで `major_name='Physics Teaching'` の major_id\n2. members で `major_id=該当` の COUNT",
        "failure": _BRIEF_SOLVED,
        "memo": "- 単純 COUNT、agent が直感的に解決。",
    },
    "task_64": {
        "steps": "1. superhero で `superhero_name='3-D Man'` の id 取得\n2. hero_power で hero_id=該当 の power_id\n3. superpower で power_name 取得 = 4 行",
        "failure": _BRIEF_SOLVED,
        "memo": "- exp_109 で 96% perfect、典型的 join task。",
    },
    "task_74": {
        "steps": "1. superhero で `full_name='Karen Beecher-Duncan'`\n2. colour テーブル join で eye_colour 取得",
        "failure": _BRIEF_SOLVED,
        "memo": "- 単純 lookup、94% perfect。",
    },
    "task_75": {
        "steps": "1. qualifying で raceId=19 の q2 列で最小時間 (filter-back) → driverId\n2. drivers で surname",
        "failure": _BRIEF_SOLVED,
        "memo": "- Rule 10 filter-back が効く例、大体安定。",
    },
    "task_145": {
        "steps": "1. attendance で event ごとの member 数集計 > 10\n2. event テーブル join で `type='Meeting'` の COUNT",
        "failure": _BRIEF_SOLVED,
        "memo": "- 多段 aggregation だが安定、95% perfect。",
    },
    "task_194": {
        "steps": "1. atom で `element='p' OR element='n'` の atom_id\n2. connected で 両 atom_id を含む bond を抽出 = 6 行",
        "failure": _BRIEF_SOLVED,
        "memo": "- exp_088 M-Schema regression が起きた task の 1 つ、preamble 改造に注意。",
    },
    "task_214": {
        "steps": "1. set_translations で `language='Portuguese (Brazil)'`\n2. sets で `block='Commander'`\n3. join して COUNT",
        "failure": _BRIEF_SOLVED,
        "memo": "- magic the gathering データ、93% perfect。",
    },
    "task_218": {
        "steps": "1. satscores で `dname='Fresno Unified'` の AvgScrRead 最低 school\n2. frpm or schools で Phone 取得",
        "failure": _BRIEF_SOLVED,
        "memo": "- filter-back + cross-table join、92% perfect。",
    },
    "task_261": {
        "steps": "1. hero_power で `power='Super Strength'` の hero_id\n2. superhero で height > 200 の COUNT",
        "failure": _BRIEF_SOLVED,
        "memo": "- 92% perfect、フィルタ + COUNT。",
    },
    "task_269": {
        "steps": "1. hero_power で `power_name='Death Touch'` の hero_id\n2. superhero_name 列 = 7 行",
        "failure": _BRIEF_SOLVED,
        "memo": "- 92% perfect、複数行返答だが安定。",
    },
    "task_283": {
        "steps": "1. colour テーブル join で eye colour='Blue' をカウント\n2. ratio × 100 = percentage",
        "failure": _BRIEF_SOLVED,
        "memo": "- 95% perfect、percentage 計算は agent 安定。",
    },
    "task_305": {
        "steps": "1. races で `year=2009 AND name='Spanish Grand Prix'` の raceId\n2. results.fastestLapSpeed の MAX",
        "failure": _BRIEF_SOLVED,
        "memo": "- 97% perfect、MAX スカラー、安定。",
    },
    "task_349": {
        "steps": "1. members で `first_name='Angela' AND last_name='Sanders'`\n2. major join で major_name",
        "failure": _BRIEF_SOLVED,
        "memo": "- 単純 lookup、92% perfect。",
    },
    "task_350": {
        "steps": "1. attendance で `event_name=\"Women's Soccer\"`\n2. members.t_shirt_size と join、特定 size の COUNT",
        "failure": _BRIEF_SOLVED,
        "memo": "- multi-condition filter、92% perfect。",
    },
    "task_415": {
        "steps": "詳細不明、データ未取得",
        "failure": _BRIEF_SOLVED,
        "memo": "- 94% perfect。",
    },
    # === often-solved (15 tasks) ===
    "task_11": {
        "steps": "1. examination で `Thrombosis=2` (severe) の DISTINCT ID\n2. patient_sex.csv で SEX、Patient.md で Diagnosis (= narrative parse)\n= 3 行 × 3 col",
        "failure": "- 70% perfect だが残り 30% で run-to-run variance (= padding bug, attempts diversity)\n- 値はだいたい合うが行数誤差や extras で wext 罰\n- exp_103 で pred 75r×7c の blow-up 観測 (= padding バグ顕在化)",
        "memo": "- padding fix (exp_093) で blow-up 解消の対象 task\n- Patient.md の narrative parse が安定性を左右",
    },
    "task_22": {
        "steps": "1. members で `full_name='Connor Hilton'` の member_id\n2. dues / payments で支払日 (date_received) を取得\n= 2 行 × 1 col (= 複数回支払いあり)",
        "failure": "- 86% perfect、稀に 1 行のみ返して ties 1 つ取りこぼし",
        "memo": "- gold 2 行 = 複数回 dues 支払い、Rule 10 filter-back が効く",
    },
    "task_27": {
        "steps": "1. members で member_id=該当 の first_name, last_name\n2. expense (or budget) で member_id=該当 の cost SUM\n= 1 行 × 3 col",
        "failure": "- 87% perfect、稀に SUM 値の ROUND ぶれ or 余分列",
        "memo": "- 安定性高い、列スコープが gold と一致しやすい",
    },
    "task_67": {
        "steps": "1. superhero で `gender='Female'` の id\n2. weight_kg の AVG\n= 1 行 × 1 col スカラー",
        "failure": "- 81% perfect、観測 fail (exp_086): 1r × **2c** (`average_weight, average_weight_kg`) 重複\n- 単位 conversion (lbs vs kg) で値ぶれもあり",
        "memo": "- 重複 col は Rule 18 column verify で解決可能\n- gold col header `AVG(T1.weight_kg)` は SQL alias、value match で救済",
    },
    "task_243": {
        "steps": "1. badges/posts で UserId=24 の post 数 (T1)、vote 数 (T2)\n2. post / vote の比 = 0.375\n= 1 行 × 1 col",
        "failure": "- 87% perfect、観測 fail: extra col (= count of posts や count of votes 個別を含めるパターン)",
        "memo": "- 比率計算、Rule 18 column verify でほぼ完全救済",
    },
    "task_249": {
        "steps": "1. users で post 数 > 10 の UserId\n2. AVG(UpVotes), AVG(Age) を 2 列で出力\n= 1 行 × 2 col",
        "failure": "- 79% perfect、列順 (UpVotes vs Age) の入れ替えで value match 失敗するケース\n- 巨大 JSON file (166 MB) で preamble truncation 影響",
        "memo": "- exp_101 rich preamble の profile section で救済期待 task の 1 つ",
    },
    "task_250": {
        "steps": "1. users で `display_name='slashnick'` の UserId\n2. posts で OwnerUserId=該当、AnswerCount 最大の post (filter-back)\n3. PostId 出力\n= 1 行 × 1 col",
        "failure": "- 88% perfect、稀に PostId vs Id 列名混乱",
        "memo": "- 単純 filter-back + lookup、安定",
    },
    "task_257": {
        "steps": "1. posts で `Title='Computer Game Datasets'` の Id, ViewCount 取得\n2. last edit / last poster の DisplayName を取得\n= 1 行 × 2 col (`ViewCount, DisplayName`)",
        "failure": "- 78% perfect、観測 fail: 1r × **3c** (`total_views, last_editor_display_name, last_poster_display_name`) — last editor と last poster の 2 つを agent が両方含めて wext\n- 巨大 SQLite (139 MB) で preamble 影響",
        "memo": "- 質問 \"posted it last time\" は last poster 限定だが agent が editor も含める\n- Rule 18 column verify で 3→2 col 削減可能",
    },
    "task_287": {
        "steps": "1. hero_power で `power_name='Phoenix Force'` の hero_id\n2. gender 取得 (= Female)\n= 1 行 × 1 col",
        "failure": "- 88% perfect、稀に gender 値の format ぶれ",
        "memo": "- 単純 lookup、稀な失敗のみ",
    },
    "task_292": {
        "steps": "1. results で raceId=9 の constructor で SUM(points) 最大\n2. constructors で url 取得\n= 1 行 × 1 col",
        "failure": "- 85% perfect、ties や filter-back 抜けで稀に異 constructor",
        "memo": "- url 値が完全一致必要、URL format ぶれで失敗するケースもあり",
    },
    "task_303": {
        "steps": "1. races で `name='European Grand Prix'` の circuit\n2. circuits で country='Germany' の数 / 全 circuit 数 × 100\n= 52.17%",
        "failure": "- 88% perfect、percentage 計算、Round ぶれで稀に value mismatch",
        "memo": "- gold col header = SQL CAST, value match で救済",
    },
    "task_352": {
        "steps": "1. budget で `category='Advertisement' AND event='Yearly Kickoff'` の amount SUM\n2. 同 event='October Meeting' の amount SUM\n3. 比 = 2.727\n= 1 行 × 1 col",
        "failure": "- 76% perfect、event_name の表記ぶれで分岐失敗\n- gold col header が複雑 SQL CASE、agent は value だけ合えば救済",
        "memo": "- M-Schema 系 exp_088 で改善見せた task、preamble 改善対象",
    },
    "task_355": {
        "steps": "1. expense で `expense_description LIKE '%water%veggie%supplies%'`\n2. members join で first_name, last_name + cost\n= 1 行 × 3 col",
        "failure": "- 88% perfect、expense_description の正規表現マッチで稀に miss",
        "memo": "- 安定、稀な失敗のみ",
    },
    "task_408": {
        "steps": "1. results で 2008 Australian Grand Prix のチャンピオン (positionOrder=1) の time\n2. 最終 driver (positionOrder=max) の time\n3. (last - champion) / last × 100 計算\n= 0.31555%",
        "failure": "- 85% perfect、time の format parse + 計算式の組み立てで稀に miss",
        "memo": "- gold value 0.3155 は %, agent が round ぶれで誤差",
    },
    "task_420": {
        "steps": "1. SQLite で `format='commander' AND legality='Legal'` の cards\n2. hasContentWarning=0 の比率 × 100 = 100.0\n= 1 行 × 1 col",
        "failure": "- 63% perfect (= often-solved 中で最低)、巨大 SQLite (61 MB) で preamble 制約\n- format string の case sensitivity (= 'commander' vs 'Commander') でフィルタずれ",
        "memo": "- exp_101 rich preamble で改善期待\n- 投資優先度: 中 (= 巨大 SQL の preamble 改善で救済可)",
    },
    # === variable (5 tasks) ===
    "task_86": {
        "steps": """1. drivers.json で `Alex Yoong` の driverId 取得
2. results.csv で `driverId=該当 AND grid<20` (= 出走 grid 順位 20 未満) のレコード抽出
3. raceId の集合から races.json で race name を取得
4. **name 列のみ** 出力 (= 順位や date は不要)
5. 出力 = 16 行 × 1 col""",
        "failure": """- 観測 (exp_086): 18r × **2c** (`name, name`) — 列重複 (= 同じ列が 3-attempt union から残る) + 行 1〜2 余分
- 130 run で 80% non-perfect、20% perfect (= ばらつき大)
- "track number" の解釈ぶれ (= grid number? track position? race number?) で別フィルタになることあり
- attempts diversity が高い task で union 後の wext / extra row が増える典型""",
        "memo": """- "track number" は results.csv の `grid` 列 (= スタート grid 位置) を指すと推定
- gold 16 行は Alex Yoong の career の特定期間 (= grid<20 の race 群)
- 救済策: padding fix (= exp_093 同等) + Rule 18 column verify
- 投資優先度: **中** (= 既存 fix で部分救済期待、padding fix 効果検証中)""",
    },
    "task_173": {
        "steps": """1. transactions テーブル (= 大型 .db か CSV) で `Date LIKE '2013-06%'` の transactions を取得
2. transaction の GasStationID から gas station の Country を引く
3. **DISTINCT Country** を出力
4. gold = `['CZE', 'SVK']` (2 国)""",
        "failure": """- 観測: 2 行 1 列で正しい構造の run はあるが、Country 値が誤る (= e.g. 'Czech Republic' vs 'CZE')
- 130 run で 71% non-perfect、29% perfect
- 値の format (= ISO code vs full name) で一致しないケース
- knowledge.md には `'CZE' (Czech Republic), 'SVK' (Slovakia)` 表記あり、agent が ISO に正規化できれば perfect""",
        "memo": """- knowledge.md の Country 値 hint = ISO 3-letter code が正解の format
- agent が `'Czech Republic'` で出すと value match で 0.0
- 救済策: knowledge.md の value format を Rule で参照させる (= 既存 R3 系で対応)
- 投資優先度: **低** (= format 統一のみ、稀な失敗)""",
    },
    "task_196": {
        "steps": """1. atom テーブルで `element='i'` (= iodine) の atom_id 群を取得
2. bond / connected テーブルで該当 atom_id を含む bond 数を集計
3. AVG(bond_count per atom) または COUNT(bond)/COUNT(atom) でスカラー値計算 = 1.0
4. 出力 = 1 行 × 1 col""",
        "failure": """- 観測 (exp_086): 1r × **2c** (`average_bonds, average_bonds`) で値 `1.0, 2.0` — 重複 col + 異なる解釈の値
- 130 run で 61% non-perfect、39% perfect
- 解釈ぶれ: 「avg bonds per atom」 vs 「total bonds / count atoms」 (= ratio)
- 3-attempt union で 2 つの解釈が混在 → padding 起きないが col 重複""",
        "memo": """- gold value 1.0 = 1 atom あたり平均 1 bond (= iodine の性質)
- 重複 col は exp_109 の column verify で解決可能
- 投資優先度: **中** (= column verify 直接効果、padding fix 不要)""",
    },
    "task_200": {
        "steps": """1. molecule の triple-bond 分子を特定 (= bond テーブルで bond_type='triple' を含む molecule)
2. その molecule の atom テーブルで `element='p' (phosphorus) OR element='br' (bromine)` をフィルタ
3. atom_id を **COUNT** = 1
4. 出力 = 1 行 × 1 col""",
        "failure": """- 観測 (exp_086): 1r × **2c** (`count, count`) で `4, 1` — 重複 col + 値解釈分岐
- 130 run で 66% non-perfect、34% perfect
- 解釈ぶれ: 「triple-bond を含む molecule の全 atoms」 vs 「triple-bond 結合に直接参加する p/br atoms」
- gold = 1 (= 厳密に triple-bond participation の p/br atom は 1 個のみ)""",
        "memo": """- bond_type の表記ぶれ (= 'triple' vs '#' vs '3') で agent 困惑
- 重複 col はやはり Rule 18 column verify で削減可能
- 投資優先度: **中** (= 解釈の一貫性 + column verify 必要)""",
    },
    "task_330": {
        "steps": """1. Match.csv で `date='2008-09-24'` AND League が 'Belgian Jupiler League' のレコードを取得
2. country.json / league.json で league_id 解決
3. 該当 match の `home_team_goal, away_team_goal` 2 列を取得
4. 出力 = 1 行 × 2 col""",
        "failure": """- 観測 (exp_086): 1r × 2c で `date, home_team_goal` — **away_team_goal 列が抜け、date が混入**
- 130 run で 80% non-perfect、20% perfect
- 巨大 CSV (Match.csv 279 MB) の preamble truncation で agent が away_team_goal 列を発見できないケース
- exp_101 rich preamble (= 列リスト全部表示) で改善期待大
- date 列を答えに含めるのは Rule 12 (explicit SELECT) の弱さ""",
        "memo": """- exp_101 rich preamble の profile section が **115 列全部表示**して救済される代表例
- gold = (1, 1) の引き分け試合
- 救済策: rich preamble + column verify combo
- 投資優先度: **高** (= rich preamble の effect ターゲット、winning 路線)""",
    },
    # === rarely-solved (9 tasks) ===
    "task_38": {
        "steps": """1. JSON `account.json` で client_id=3356 の account_id を解決
2. JSON `client.json` も併用して client→account の関係確認
3. trans.csv で `account_id IN (上記)` AND `type='VYDAJ' (= withdrawal)` AND `operation='VYBER' (= cash)` をフィルタ
4. **trans_id 列のみ**を出力 (= account_id, date, amount は不要)
5. 出力 = 140 行 × 1 col""",
        "failure": """- 観測 (exp_086): 140r × **6c** (`trans_id, account_id, date, type, operation, amount`) — 行は正しいが余分列 5 つで wext 罰則
- 130 run で 0 perfect (= 列スコープ規律の構造的失敗)
- gold 値 (trans_id) は予測内に存在 → recall 高いが extras_ratio が λ0.5 を 0 に潰す
- 質問 "List all the withdrawals" → agent は SELECT * 風で関連列全部返す傾向""",
        "memo": """- VYDAJ / VYBER はチェコ語で **withdrawal / cash** (= knowledge.md に記載)
- gold col=trans_id 1 個のみ → Rule 12 (explicit SELECT) + Rule 18 (column verify) で救済可能
- 投資優先度: **高** (= 列削減だけで perfect、easy task の取りこぼし救済)""",
    },
    "task_80": {
        "steps": """1. qualifying.csv で raceId=903 のレコードを取得
2. q3 列が `'0:01:54'` (or 1:54.xxx 等の format バリエーション) のレコードをフィルタ
3. **driverId は答えではなく** `number` 列を取得 (= ドライバーゼッケン番号)
4. **同タイム複数あり** (= ties)、両方の number を返す = 3 と 5
5. 出力 = 2 行 × 1 col""",
        "failure": """- 観測 (exp_086): 1r × 1c (`3` のみ) — **2 番目の tied driver `5` を取りこぼす**
- 130 run で 99% zero、1% perfect (= ties 救済が稀)
- LIMIT 1 / `==` 一致のみで filter-back せず単一 driver 返却が常習
- 一部 run では時間 format 違いで recall 0 (`0:01:54.000` vs `0:01:54`)""",
        "memo": """- gold が 2 行 → Rule 10 (filter-back for ties) が直接効く対象
- ただし superlative 単語 (lowest 等) を含まないので Rule 10 トリガーに引っかからない
- "finished 0:01:54" = 完了タイムが正確に 0:01:54 のドライバー (= 等値マッチ、ties あり)
- 救済策: 等値マッチでも tied 結果を予期する rule 追加 (= Rule 10 を 拡張)
- 投資優先度: **中** (= rule extension で救済可能、1 task のみ)""",
    },
    "task_396": {
        "steps": """1. doc/superhero.md からヒーロー情報抽出 (= name, height, publisher_id)
2. publisher.json から publisher_id → publisher_name のマッピング
3. height が 150-180 の範囲のヒーローを集める (= 母数集合)
4. その中で publisher_name = 'Marvel Comics' のヒーロー数 / 母数 × 100 = 54.84%
5. 出力 = 1 行 × 1 col のスカラー値""",
        "failure": """- 観測 (exp_086): 1r × **2c** (`percentage, percentage`) で値 60.87, 48.15 — 2 列の重複 + 値も間違い
- 質問の "between 150 to 180" の境界解釈ぶれ (= inclusive vs exclusive)
- doc/superhero.md は narrative 文 (= 173KB) でデータ構造化が困難
- 130 run で 99% zero
- gold value 54.83870967741935 は agent が再現困難な精度""",
        "memo": """- doc/superhero.md は agent が parse しづらい narrative 形式 (= LLM extraction が必要)
- gold col header `CAST(...) * 100 / COUNT(...)` は SQL alias literal、value match の余地あり
- height filter の境界解釈 (150 ≤ h ≤ 180 vs 150 < h < 180) で母数が変わる可能性
- 救済策: doc parsing の精度向上 (= 別軸の大改造)
- 投資優先度: **低** (= 文書 parsing の根本問題、prompt fix では改善困難)""",
    },
    "task_25": {
        "steps": """1. expense.json で `approved=true` の cost を event 単位で集計
2. budget.csv の link_to_event を介して event.json と join
3. event ごとの total_cost を計算
4. **最低 cost を持つ event を全部** (= ties あり) 取得
5. event_name 列のみ出力 = November Speaker / October Speaker / September Speaker (3 行 tied at 6.0)""",
        "failure": """- 観測 (exp_086): 3r × **2c** (`event_name, total_cost`) で値 `Officers meeting - November` 等 — **行数 3 は当たってるが event 自体が違う**
- 値が違う原因: budget.amount を使うか expense.cost を使うかの解釈ぶれ
- 130 run で 95% zero、行数 1 で出すケースが大多数 (= ties 取りこぼし)
- exp_109 の row_count 撤廃 + Rule 10 filter-back で `+0.31` 改善傾向あり (= 救済可能性高い)""",
        "memo": """- "lowest cost" の cost = expense.cost の SUM か budget.amount かで解釈分岐
- gold は **expense の SUM** ベース (= Officers meeting series ではなく Speaker series)
- ties が 3 つ完全一致なので filter-back (Rule 10) が効く
- exp_109 で実際に救済されている (= row_count 撤廃が効果)
- 投資優先度: **高** (= 既に救済路線確立、定着させたい)""",
    },
    "task_89": {
        "steps": """1. races.json で 2008 年の Chinese Grand Prix の raceId を取得
2. results.csv で `raceId=該当 AND positionOrder=2` の driver を特定
3. その driver の `time` 列を取得 = `+16.445`
4. 出力 = 1 行 × 1 col""",
        "failure": """- 観測 (exp_086): 1r × 1c で `+14.925` — **driver が違う / 別 race と混同 / position 列の取り違え**
- gold value `+16.445` は 2 位ドライバーの finish time
- 130 run で 98% zero
- agent が `position` (= grid 位置) と `positionOrder` (= 終了順位) を混同する典型ケース""",
        "memo": """- results.csv 列が多数 (positionOrder, position, positionText, points, ... 18 列) で agent が `position` を直感的に使うが正解は `positionOrder`
- gold value 取れた稀な perfect run はうまく positionOrder を使った例
- 救済策: knowledge.md で position/positionOrder の意味を明示
- 投資優先度: **中** (= column 名理解の問題、knowledge enrichment で改善可能)""",
    },
    "task_199": {
        "steps": """1. frpm.csv で `District Name LIKE '%Riverside%'` の school 群取得
2. satscores.db の `satscores` で `cds IN (上記 school)` AND `AvgScrMath > 400` をフィルタ
3. group by district、district の avg(AvgScrMath) > 400 の district を選択
4. 該当 district の **school name** と **Charter Funding Type** (= frpm の列) を取得 (2 列)
5. 出力 = 6 行 × 2 col""",
        "failure": """- 観測 (exp_086): **506r × 6c** (= School Name 列が 3 つ重複 + 別文字列の Charter Funding Type 列も 3 つ)
- 完全に scope mismatch + 列重複の双方
- 130 run で 98% zero
- "Riverside-related school districts" の解釈 (= District Name に Riverside を含む) で district を絞らず、 school 単位で集計""",
        "memo": """- 質問が複合的: "Riverside-related" + "average across schools" + "names AND funding types" (2 列)
- gold col `Charter Funding Type` の値が空文字列 (`''`) の行あり (= NULL に相当) → agent が空セル表現を間違える
- 救済策: 多段 reasoning + 列スコープ verify
- 投資優先度: **中** (= 多列複合 task の代表例、verify 系 rule で部分救済期待)""",
    },
    "task_418": {
        "steps": """1. doc/Patient.md から患者の生年月日抽出 → 年齢計算 (current year - birth year, 70 未満)
2. doc/Laboratory.md から creatinine (= CRE) 値抽出
3. CRE 異常閾値で患者をフィルタ
4. 上記両条件 AND の **DISTINCT 患者 ID 数** = 1
5. 出力 = 1 行 × 1 col""",
        "failure": """- 観測 (exp_086): 1r × 1c で値 `2` — **count が違う**
- gold = 1
- 130 run で 94% zero、5% perfect
- 失敗原因: CRE の "abnormal" 閾値が unclear (= male: >1.3 / female: >1.0 等) + 年齢 70 未満の境界 (年齢 < 70 vs ≤ 70) の解釈ぶれ
- doc 形式 (= narrative) で患者情報抽出が不安定""",
        "memo": """- creatinine の正常範囲: 男性 0.6-1.2 mg/dL、女性 0.5-1.1 mg/dL (医学標準)
- doc/Laboratory.md (279KB) は narrative、構造化抽出困難
- gold = 1 の極端な少なさ → 厳密な閾値定義必要
- 救済策: 医学閾値を knowledge に明記 + doc parsing 強化
- 投資優先度: **低** (= 知識欠落 + 文書 parsing の二重困難)""",
    },
    "task_259": {
        "steps": """1. SQLite posts.db で ViewCount BETWEEN 100 AND 150 の Post.Id 群を取得
2. comments.csv で `PostId IN (上記)` のコメントをフィルタ
3. Score が最高のコメントを選択 (= ties あれば全部)
4. **Text 列のみ**を出力
5. 出力 = 1 行 × 1 col (gold は 1 行だが構造的には ties 可能)""",
        "failure": """- 観測 (exp_086): 1r × **4c** (`comment, comment_id, score, post_id`) — **値は正しいが列が 3 つ余分**
- 130 run で 65% zero、7% perfect、28% partial (= 部分点取れること多い)
- score=14 の Welcome message が gold 値と一致する run はあるが extras で λ0.5 削られる
- Text 列の長文 + 改行を含む CSV エスケープ問題でも稀に escape ミス""",
        "memo": """- gold col header `Text` のみ → Rule 5 で他列削除すれば perfect
- 値は正しい場合が多いので **column verify (Rule 18)** が直接効く
- 巨大ファイル (comments.csv 47 MB, posts.db 139 MB) で preamble truncation も影響
- exp_101 rich preamble の profile で改善期待大
- 投資優先度: **高** (= column verify + rich preamble combo で確実救済可能)""",
    },
}
