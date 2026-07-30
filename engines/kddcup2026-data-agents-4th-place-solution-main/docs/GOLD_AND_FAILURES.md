# 50-Task gold SELECT expressions & failure cases (= exp_120 baseline)

50 tasks total. Mean score: 0.7469

Notes:
- **Gold SELECT expression**: the column header in `gold.csv` is the BIRD-style SELECT clause from the gold SQL (= e.g. `CAST(COUNT(CASE WHEN ...) AS REAL) * 100 / COUNT(*)`). The full FROM/JOIN/WHERE clauses are NOT shipped in the public dataset — only the SELECT expression + result values.
- **Plain header**: when the gold header is just a column name (= `'name'`, `'Country'`, etc.) without SQL functions, the gold SQL was a simple `SELECT name FROM ...`. These rows show `(plain SELECT)`.
- **Agent SQL**: the SQL our exp_120 agent submitted at the answer_from_sql terminal step (= from `trace.json`).

## ✅ Correct (32 tasks, score≥0.99)

| task | gold SELECT expression | gold first row | question | agent SQL |
|---|---|---|---|---|
| task_11 | `ID,SEX,Diagnosis` (plain SELECT) | 163109, F, SLE | For patients with severe degree of thrombosis, list their ID, sex and disease the patient | <code>SELECT p.ID, p.SEX, p.Diagnosis FROM Patient p JOIN Examination e ON p.ID = e.ID WHERE e.Thrombosis IN (1, 2)</code> |
| task_145 | `COUNT(*)` | 4 | Among the events attended by more than 10 members of the Student_Club, how many of them ar | <code>(manual answer with cols=['meeting_count'])</code> |
| task_19 | `first_name,last_name` (plain SELECT) | Trent, Smith | List the full name of the Student_Club members that grew up in Illinois state. | <code>(manual answer with cols=['first_name', 'last_name'])</code> |
| task_194 | `bond_id` (plain SELECT) | TR032_2_3 | What are the bonds that have phosphorus and nitrogen as their atom elements? | <code>SELECT DISTINCT c.bond_id FROM connected c JOIN atom a1 ON c.atom_id = a1.atom_id JOIN atom a2 ON c.atom_id2 = a2.atom_id WHERE (a1.element = 'p' AND a2.element = 'n') OR (a1.element = 'n' AND a2.element = 'p') ORDER BY c.bond_id</code> |
| task_200 | `COUNT(T1.atom_id)` | 1 | Calculate the total atoms with triple-bond molecules containing the element phosphorus or | <code>(manual answer with cols=['total_atoms'])</code> |
| task_214 | `COUNT(T1.id)` | 7 | How many Brazilian Portuguese translated sets are inside the Commander block? | <code>SELECT COUNT(DISTINCT s.code) as count FROM sets s JOIN set_translations st ON s.code = st.setCode WHERE s.block = 'Commander' AND st.language = 'Portuguese (Brazil)'</code> |
| task_218 | `Phone` (plain SELECT) | (559) 248-5100 | What is the telephone number for the school with the lowest average score in reading in Fr | <code>(manual answer with cols=['Phone'])</code> |
| task_22 | `date_received` (plain SELECT) | 2019-10-02 | State the date Connor Hilton paid his/her dues. | <code>SELECT i.date_received FROM income i JOIN member m ON i.link_to_member = m.member_id WHERE m.first_name = 'Connor' AND m.last_name = 'Hilton' AND i.source = 'Dues'</code> |
| task_24 | `COUNT(T2.link_to_member)` | 17 | How many members attended the "Women's Soccer" event? | <code>SELECT COUNT(*) as attendee_count FROM attendance a JOIN event e ON a.link_to_event = e.event_id WHERE e.event_name = 'Women''s Soccer'</code> |
| task_243 | `CAST(COUNT(DISTINCT T2.Id) AS REAL) / COUNT(DISTINCT T1.Id)` | 0.375 | For the user No.24, how many times is the number of his/her posts compared to his/her vote | <code>(manual answer with cols=['ratio'])</code> |
| task_249 | `AVG(T1.UpVotes),AVG(T1.Age)` | 182.2832618025751, 34.083333333333336 | What is the average of the up votes and the average user age for users creating more than | <code>(no SQL)</code> |
| task_25 | `event_name` (plain SELECT) | November Speaker | Which event has the lowest cost? | <code>WITH event_costs AS (SELECT e.event_id, e.event_name, SUM(b.spent) as total_cost FROM event e JOIN budget b ON e.event_id = b.link_to_event GROUP BY e.event_id, e.event_name), min_cost AS (SELECT MIN(total_cost) as min_val FROM event_costs) SELECT ec.event_name, ec.total_cost FROM event_costs ec, mi</code> |
| task_250 | `PostId` (plain SELECT) | 351 | Which post by slashnick has the most answers count? State the post ID. | <code>(no SQL)</code> |
| task_257 | `ViewCount,DisplayName` (plain SELECT) | 1708, mbq | Identify the total views on the post 'Computer Game Datasets'. Name the user who posted it | <code>(manual answer with cols=['total_views', 'last_editor_name'])</code> |
| task_26 | `COUNT(T2.member_id)` | 1 | How many members of the Student_Club have major in 'Physics Teaching'? | <code>(manual answer with cols=['count'])</code> |
| task_261 | `COUNT(T1.id)` | 56 | Among the superheroes with the super power of "Super Strength", how many of them have a he | <code>SELECT COUNT(s.id) AS count FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id JOIN superpower sp ON hp.power_id = sp.id WHERE sp.power_name = 'Super Strength' AND s.height_cm > 200</code> |
| task_269 | `superhero_name` (plain SELECT) | Black Flash | What are the names of the superheroes with the power of death touch? | <code>SELECT s.superhero_name FROM superhero s JOIN hero_power hp ON s.id = hp.hero_id JOIN superpower sp ON hp.power_id = sp.id WHERE sp.power_name = 'Death Touch'</code> |
| task_283 | `CAST(COUNT(CASE WHEN T2.colour = 'Blue' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)` | 31.2 | Calculate the percentage of superheroes with blue eyes. | <code>SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM superhero), 2) as percentage FROM superhero s JOIN colour c ON s.eye_colour_id = c.id WHERE c.colour = 'Blue'</code> |
| task_287 | `gender` (plain SELECT) | Female | Identify the gender of the superhero who has the ability of Phoenix Force. | <code>(manual answer with cols=['gender'])</code> |
| task_292 | `url` (plain SELECT) | http://en.wikipedia.org/wiki/Red_Bull_Racing | For the constructor which got the highest point in the race No. 9 , what is its introducti | <code>(manual answer with cols=['url'])</code> |
| task_303 | `CAST(COUNT(CASE WHEN T1.country = 'Germany' THEN T2.circuitID END) AS REAL) * 100 / COUNT(T2.circuitId)` | 52.17391304347826 | Among all European Grand Prix races, what is the percentage of the races were hosted in Ge | <code>(manual answer with cols=['percentage'])</code> |
| task_305 | `fastestLapSpeed` (plain SELECT) | 202.484 | What was the fastest lap speed among all drivers in the 2009 Spanish Grand Prix? | <code>SELECT MAX(fastestLapSpeed) AS fastest_lap_speed FROM results WHERE raceId = 5</code> |
| task_349 | `major_name` (plain SELECT) | Business | What's Angela Sanders's major? | <code>(manual answer with cols=['major_name'])</code> |
| task_350 | `COUNT(T1.event_id)` | 7 | Among the students from the Student_Club who attended the event "Women's Soccer", how many | <code>SELECT COUNT(*) as count FROM attendance a JOIN member m ON a.link_to_member = m.member_id WHERE a.link_to_event = 'rec2N69DMcrqN9PJC' AND m.t_shirt_size = 'Medium'</code> |
| task_355 | `first_name,last_name,cost` (plain SELECT) | Elijah, Allen, 28.15 | Write the full name of the member who spent money for water, veggie tray and supplies and | <code>(manual answer with cols=['full_name', 'cost'])</code> |
| task_408 | `(CAST((SELECT time_seconds FROM last_driver_incremental) AS REAL) * 100) / (SELECT time_seconds + (SELECT time_seconds FROM last_driver_incremental) FROM champion_time)` | 0.31555732286030097 | How much faster in percentage is the champion than the driver who finished the race last i | <code>SELECT ROUND(((MAX(milliseconds) - MIN(milliseconds)) * 100.0 / MAX(milliseconds)), 2) AS percentage_faster FROM results WHERE raceId = 18 AND milliseconds IS NOT NULL</code> |
| task_415 | `constructorRef,url` (plain SELECT) | mclaren, http://en.wikipedia.org/wiki/McLaren | What is the constructor reference name of the champion in the 2009 Singapore Grand Prix? P | <code>(manual answer with cols=['constructorRef', 'url'])</code> |
| task_420 | `CAST(SUM(CASE WHEN T1.hasContentWarning = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.id)` | 100.0 | What percentage of cards with format commander and legal status do not have a content warn | <code>SELECT ROUND(CAST(SUM(CASE WHEN hasContentWarning = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*), 2) as percentage FROM cards WHERE leadershipSkills LIKE '%commander%True%'</code> |
| task_64 | `power_name` (plain SELECT) | Agility | Please list all the superpowers of 3-D Man. | <code>(manual answer with cols=['power_name'])</code> |
| task_67 | `AVG(T1.weight_kg)` | 60.77956989247312 | What is the average weight of all female superheroes? | <code>SELECT AVG(s.weight_kg) AS avg_weight FROM superhero s JOIN gender g ON s.gender_id = g.id WHERE g.gender = 'Female' AND s.weight_kg IS NOT NULL</code> |
| task_74 | `colour` (plain SELECT) | Brown | Provide the eye colour of the superhero who has Karen Beecher-Duncan as their full name. | <code>(manual answer with cols=['eye_colour'])</code> |
| task_75 | `surname` (plain SELECT) | Räikkönen | What is the surname of the driver with the best lap time in race number 19 in the second q | <code>(manual answer with cols=['surname'])</code> |

## ⚠️ Partial credit (8 tasks)

| task | score | gold SELECT | gold preview | pred header | pred preview | question | agent SQL |
|---|---|---|---|---|---|---|---|
| task_27 | 0.88 | `first_name,last_name,SUM(T2.cost)` | Sacha, Harrison, 866.25 (1r) | `full_name,total_cost,first_name,last_name` | Sacha Harrison, 866.25, Sacha, Harrison (1r) | List out the full name and total cost that member id "rec4BLdZHS2Blfp4v" incurre | <code>SELECT m.first_name \|\| ' ' \|\| m.last_name AS full_name, SUM(e.cost) AS total_cost FROM member m JOIN expense e ON m.member_id = e.link_to_member WHERE m.member_id = 'rec4BLdZHS2Blfp4v' GROUP BY m.member_id, m.first_name, m.last_name</code> |
| task_196 | 0.75 | `CAST(COUNT(T2.bond_id) AS REAL) / COUNT(T1.atom_id)` | 1.0 (1r) | `avg_bonds,avg_bonds` | 1.0, 1.2 (1r) | What is the average number of bonds the atoms with the element iodine have? | <code>(manual answer with cols=['avg_bonds'])</code> |
| task_352 | 0.75 | `CAST(SUM(CASE WHEN T2.event_name = 'Yearly Kickoff' THEN T1.amount ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.event_name = 'October Meeting' THEN T1.amount ELSE 0 END)` | 2.727272727272727 (1r) | `count,ratio` | 1, 2.73 (1r) | How many times was the budget in Advertisement for "Yearly Kickoff" meeting more | <code>(manual answer with cols=['count'])</code> |
| task_379 | 0.75 | `element` (plain) | c (7r) | `element,count` | br, 4 (7r) | Tally the toxicology element of the 4th atom of each molecule that was carcinoge | <code>(manual answer with cols=['element', 'count'])</code> |
| task_418 | 0.75 | `COUNT(DISTINCT T1.ID)` | 1 (1r) | `count,count` | 1, 7 (1r) | Among the patients whose creatinine level is abnormal, how many of them aren't 7 | <code>(manual answer with cols=['count'])</code> |
| task_259 | 0.67 | `Text` (plain) | Welcome to Cross Validated David P, for more information about the site and how (1r) | `Text,comment_id,comment_score` | Welcome to Cross Validated David P, for more information abo, 90813, 14 (1r) | Among the posts with views ranging from 100 to 150, what is the comment with the | <code>(manual answer with cols=['Text'])</code> |
| task_38 | 0.56 | `trans_id` (plain) | 816173 (140r) | `trans_id,account_id,date,type,operation,amount,balance,k_sym` | 816173, 2779, 1993-12-02, VYDAJ, VYBER, 800, 42072, , (140r) | List all the withdrawals in cash transactions that the client with the id 3356 m | <code>SELECT t.trans_id, t.account_id, t.date, t.type, t.operation, t.amount, t.balance, t.k_symbol, t.bank, t.account FROM trans t JOIN disp d ON t.account_id = d.account_id WHERE d.client_id = 3356 AND t.type = 'VYDAJ' AND t.operation = 'VYBER'</code> |
| task_330 | 0.25 | `home_team_goal,away_team_goal` (plain) | 1, 1 (1r) | `home_team_goal,date` | 1, 2008-09-24 00:00:00 (1r) | What was the final score for the match on September 24, 2008, in the Belgian Jup | <code>(manual answer with cols=['home_team_goal', 'away_team_goal'])</code> |

## ❌ Wrong (10 tasks, score≤0.01)

| task | gold SELECT | gold preview | pred header | pred preview | question | agent SQL |
|---|---|---|---|---|---|---|
| task_163 | `type,SUM(T3.cost)` | Meeting, 175.39 (1r) | `expense_description,total_cost` | Pizza, 51.81 (3r) | Identify the type of expenses and their total value approved for 'October Meetin | <code>(manual answer with cols=['expense_description', 'total_cost'])</code> |
| task_169 | `AVG(T2.Consumption) / 12` | 459.9562642871061 (1r) | `average_monthly_consumption` | 82027220.3041686 (1r) | What was the average monthly consumption of customers in SME for the year 2013? | <code>SELECT SUM(ym.Consumption) / 12 as average_monthly_consumption FROM yearmonth ym JOIN customers c ON ym.CustomerID = c.CustomerID WHERE c.Segment = 'SME' AND ym.Date >= 201301 AND ym.Date <= 201312</code> |
| task_173 | `Country` (plain) | CZE (2r) | `Country` | (empty) (0r) | Please list the countries of the gas stations with transactions taken place in J | <code>SELECT DISTINCT g.Country FROM transactions_1k t JOIN gasstations g ON t.GasStationID = g.GasStationID WHERE CAST(t.Date AS VARCHAR) LIKE '2013-06%' ORDER BY g.Country</code> |
| task_180 | `Consumption` (plain) | 1903.2 (9r) | `CustomerID,Consumption` | 49148, 965.87 (153r) | For all the people who paid more than 29.00 per unit of product id No.5. Give th | <code>SELECT y.CustomerID, y.Consumption FROM yearmonth y INNER JOIN (SELECT DISTINCT CustomerID FROM transactions_1k WHERE ProductID=5 AND Price > 29.00) t ON y.CustomerID = t.CustomerID WHERE y.Date = 201208</code> |
| task_199 | `sname,Charter Funding Type` (plain) | Arlington High, (6r) | `School Name,Charter Funding Type` | Arlington High, (60r) | List the names and funding types of schools from Riverside-related school distri | <code>SELECT DISTINCT "School Name", "Charter Funding Type" FROM frpm f INNER JOIN (SELECT dname FROM satscores WHERE cname = 'Riverside' AND AvgScrMath IS NOT NULL GROUP BY dname HAVING AVG(AvgScrMath) > 400) d ON f."District Name" = d.dname ORDER BY "School Name"</code> |
| task_344 | `COUNT(DISTINCT T1.ID)` | 4 (1r) | `` | (empty) (0r) | Among the male patients who have a normal level of white blood cells, how many o | <code>(no SQL)</code> |
| task_396 | `CAST(COUNT(CASE WHEN T2.publisher_name = 'Marvel Comics' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.id)` | 54.83870967741935 (1r) | `` | (empty) (0r) | In superheroes with height between 150 to 180, what is the percentage of heroes | <code>(no SQL)</code> |
| task_80 | `number` (plain) | 3 (2r) | `number` | 3 (1r) | What is his number of the driver who finished 0:01:54 in the Q3 of qualifying ra | <code>(no SQL)</code> |
| task_86 | `name` (plain) | Australian Grand Prix (16r) | `name` | Japanese Grand Prix (18r) | Which race was Alex Yoong in when he was in track number less than 20? | <code>SELECT r.name FROM driverStandings ds JOIN races r ON ds.raceId = r.raceId WHERE ds.driverId = 62 AND r.round < 20 ORDER BY r.year, r.round</code> |
| task_89 | `time` (plain) | +16.445 (1r) | `time` | +14.925 (1r) | What's the finish time for the driver who ranked second in 2008's Chinese Grand | <code>(manual answer with cols=['time'])</code> |

## Failure root-cause categorization

| task | category | note |
|---|---|---|
| task_27 | A. structural variance | concat 'Sacha Harrison' vs split 'Sacha'+'Harrison' |
| task_38 | A. column extras | list X → trans_id 1 col, agent gave 6 cols |
| task_259 | A. column extras | Text only, agent gave Text+comment_id+score |
| task_352 | A. column extras (union) | ratio (gold) vs count+ratio |
| task_379 | A. column extras | element only (gold) vs element+count |
| task_418 | A. union artifact | 1 col gold vs count,count = 1,7 (= union ガチャ) |
| task_86 | A'. row extras | 16 races gold vs 18 pred |
| task_80 | B. tied set miss | 2 drivers (3,5) gold vs 1 (3) pred |
| task_173 | C. empty result accepted | 0 rows pred、agent が transactions_1k のみ見て諦め |
| task_89 | D. wrong cell value | +16.445 gold vs +14.925 pred |
| task_330 | D. wrong column | (home, away) vs (home, date) |
| task_199 | D. filter not applied | Riverside-related で絞れず 60 schools 返答 (gold 6) |
| task_180 | D. hidden filter | 9 rows gold vs 153 rows pred |
| task_163 | E. interpretation ambiguity | type='Meeting' (gold event.type) vs expense_description × 3 |
| task_169 | E. knowledge.md mismatch | AVG/12=459 (gold) vs SUM/12=82M |
| task_196 | E. calculation interpretation | ratio 1.0 (gold) vs per-atom AVG 2.0 |
| task_344 | F. knowledge.md range missing | WBC/FG normal range 未記載 |
| task_396 | G. prose-MD parse fail | superhero_id を MD prose から抽出失敗 |
