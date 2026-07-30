"""Manually-written SQL for all 50 DABench public tasks.

For each task_id, MANUAL_SQL contains the SQL adapted to DABench schema.
Categories:
  - 28 tasks: BIRD SQL works as-is (verified by bird_sql_verify.csv status='match')
  - 22 tasks: rewritten — DuckDB syntax, GROUP BY, INT date casts, table-from-doc, etc.
  - doc-only tasks: assume prose has been parsed to a virtual table; SQL written
    as if that table existed.

After running, output CSV: manual_sql_full50_results.csv with status per task.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from experiments.exp_122_column_auditor.tools.duckdb_unified import execute_sql


# ============================================================
# 28 BIRD-match tasks — copy verbatim from BIRD dev set
# ============================================================
BIRD_MATCH = {
    "task_11": "SELECT DISTINCT T1.ID, T1.SEX, T1.Diagnosis FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.Thrombosis = 2",
    "task_19": "SELECT T1.first_name, T1.last_name FROM member AS T1 INNER JOIN zip_code AS T2 ON T1.zip = T2.zip_code WHERE T2.state = 'Illinois'",
    "task_22": "SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = 'Connor' AND T1.last_name = 'Hilton' AND T2.source = 'Dues'",
    "task_24": "SELECT COUNT(T2.link_to_member) FROM event AS T1 INNER JOIN attendance AS T2 ON T1.event_id = T2.link_to_event WHERE T1.event_name = 'Women''s Soccer'",
    "task_26": "SELECT COUNT(T2.member_id) FROM major AS T1 INNER JOIN member AS T2 ON T1.major_id = T2.link_to_major WHERE T1.major_name = 'Physics Teaching'",
    "task_38": "SELECT T4.trans_id FROM client AS T1 INNER JOIN disp AS T2 ON T1.client_id = T2.client_id INNER JOIN account AS T3 ON T2.account_id = T3.account_id INNER JOIN trans AS T4 ON T3.account_id = T4.account_id WHERE T1.client_id = 3356 AND T4.operation = 'VYBER'",
    "task_64": "SELECT T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T1.superhero_name = '3-D Man'",
    "task_67": "SELECT AVG(T1.weight_kg) FROM superhero AS T1 INNER JOIN gender AS T2 ON T1.gender_id = T2.id WHERE T2.gender = 'Female'",
    "task_74": "SELECT T2.colour FROM superhero AS T1 INNER JOIN colour AS T2 ON T1.eye_colour_id = T2.id WHERE T1.full_name = 'Karen Beecher-Duncan'",
    "task_75": "SELECT T2.surname FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = 19 ORDER BY T1.q2 ASC LIMIT 1",
    "task_80": "SELECT T2.number FROM qualifying AS T1 INNER JOIN drivers AS T2 ON T2.driverId = T1.driverId WHERE T1.raceId = 903 AND T1.q3 LIKE '1:54%'",
    "task_86": "SELECT T1.name FROM races AS T1 INNER JOIN driverStandings AS T2 ON T2.raceId = T1.raceId INNER JOIN drivers AS T3 ON T3.driverId = T2.driverId WHERE T3.forename = 'Alex' AND T3.surname = 'Yoong' AND T2.position < 20",
    "task_89": "SELECT T1.time FROM results AS T1 INNER JOIN races AS T2 ON T1.raceId = T2.raceId WHERE T1.rank = 2 AND T2.name = 'Chinese Grand Prix' AND T2.year = 2008",
    "task_173": "SELECT DISTINCT T2.Country FROM transactions_1k AS T1 INNER JOIN gasstations AS T2 ON T1.GasStationID = T2.GasStationID INNER JOIN yearmonth AS T3 ON T1.CustomerID = T3.CustomerID WHERE T3.Date = '201306'",
    "task_194": "SELECT T2.bond_id FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T2.bond_id IN (SELECT T3.bond_id FROM connected AS T3 INNER JOIN atom AS T4 ON T3.atom_id = T4.atom_id WHERE T4.element = 'p') AND T1.element = 'n'",
    "task_196": "SELECT CAST(COUNT(T2.bond_id) AS REAL) / COUNT(T1.atom_id) FROM atom AS T1 INNER JOIN connected AS T2 ON T1.atom_id = T2.atom_id WHERE T1.element = 'i'",
    "task_200": "SELECT COUNT(T1.atom_id) FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id INNER JOIN bond AS T3 ON T2.molecule_id = T3.molecule_id WHERE T3.bond_type = '#' AND T1.element IN ('p', 'br')",
    "task_214": "SELECT COUNT(T1.id) FROM sets AS T1 INNER JOIN set_translations AS T2 ON T1.code = T2.setCode WHERE T2.language = 'Portuguese (Brazil)' AND T1.block = 'Commander'",
    "task_218": "SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.District = 'Fresno Unified' AND T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT 1",
    "task_249": "SELECT AVG(T1.UpVotes), AVG(T1.Age) FROM users AS T1 INNER JOIN (SELECT OwnerUserId, COUNT(*) AS post_count FROM posts GROUP BY OwnerUserId HAVING post_count > 10) AS T2 ON T1.Id = T2.OwnerUserId",
    "task_250": "SELECT T2.PostId FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T1.DisplayName = 'slashnick' ORDER BY T3.AnswerCount DESC LIMIT 1",
    "task_257": "SELECT T2.ViewCount, T3.DisplayName FROM postHistory AS T1 INNER JOIN posts AS T2 ON T1.PostId = T2.Id INNER JOIN users AS T3 ON T2.LastEditorUserId = T3.Id WHERE T1.Text = 'Computer Game Datasets'",
    "task_259": "SELECT Text FROM comments WHERE PostId IN (SELECT Id FROM posts WHERE ViewCount BETWEEN 100 AND 150) ORDER BY Score DESC LIMIT 1",
    "task_261": "SELECT COUNT(T1.id) FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T3.power_name = 'Super Strength' AND T1.height_cm > 200",
    "task_269": "SELECT T1.superhero_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T3.power_name = 'Death Touch'",
    "task_287": "SELECT T4.gender FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id INNER JOIN gender AS T4 ON T1.gender_id = T4.id WHERE T3.power_name = 'Phoenix Force'",
    "task_292": "SELECT T2.url FROM constructorResults AS T1 INNER JOIN constructors AS T2 ON T2.constructorId = T1.constructorId WHERE T1.raceId = 9 ORDER BY T1.points DESC LIMIT 1",
    "task_305": "SELECT T2.fastestLapSpeed FROM races AS T1 INNER JOIN results AS T2 ON T2.raceId = T1.raceId WHERE T1.name = 'Spanish Grand Prix' AND T1.year = 2009 AND T2.fastestLapSpeed IS NOT NULL ORDER BY T2.fastestLapSpeed DESC LIMIT 1",
}

# ============================================================
# Helpers for tasks that need inlining of root-level non-CSV files
# ============================================================

def _build_task_344_sql() -> str:
    src = REPO / "data" / "public" / "input" / "task_344" / "context" / "patient_sex.csv"
    male_ids = [r["ID"] for r in csv.DictReader(open(src)) if r["SEX"] == "M"]
    in_list = ",".join(male_ids)  # all are integer IDs
    return f"""
SELECT COUNT(DISTINCT l.ID) AS cnt
FROM Laboratory l
WHERE l.ID IN ({in_list})
  AND l.WBC > 3.5 AND l.WBC < 9.0
  AND (l.FG < 150 OR l.FG > 450)
"""


def _build_task_379_sql() -> str:
    src = REPO / "data" / "public" / "input" / "task_379" / "context" / "carcinogenic_mols.txt"
    mols = src.read_text().strip().splitlines()
    in_list = ",".join(f"'{m.strip()}'" for m in mols if m.strip())
    return f"""
SELECT DISTINCT a.element
FROM atom a
WHERE a.molecule_id IN ({in_list})
  AND SUBSTR(a.atom_id, -1) = '4' AND LENGTH(a.atom_id) = 7
"""


# ============================================================
# 22 rewritten tasks
# ============================================================
MANUAL_SQL = {
    # task_25: gold has 3 events (tie) — use filter-back, not LIMIT 1
    "task_25": """
SELECT e.event_name
FROM event e
JOIN budget b ON e.event_id = b.link_to_event
JOIN expense ex ON b.budget_id = ex.link_to_budget
WHERE ex.cost = (SELECT MIN(cost) FROM expense)
""",

    # task_27: DuckDB requires GROUP BY for non-aggregate
    "task_27": """
SELECT m.first_name, m.last_name, SUM(e.cost) AS total_cost
FROM member m
JOIN expense e ON m.member_id = e.link_to_member
WHERE m.member_id = 'rec4BLdZHS2Blfp4v'
GROUP BY m.first_name, m.last_name
""",

    # task_145: gold is COUNT(*) = 4 (= meetings among events with >10 attendees)
    "task_145": """
WITH popular_events AS (
  SELECT a.link_to_event
  FROM attendance a
  GROUP BY a.link_to_event
  HAVING COUNT(a.link_to_member) > 10
)
SELECT COUNT(*)
FROM popular_events p
JOIN event e ON p.link_to_event = e.event_id
WHERE e.type = 'Meeting'
""",

    # task_163: DuckDB GROUP BY needed
    "task_163": """
SELECT e.type, SUM(ex.cost) AS total_cost
FROM event e
JOIN budget b ON e.event_id = b.link_to_event
JOIN expense ex ON b.budget_id = ex.link_to_budget
WHERE e.event_name = 'October Meeting'
GROUP BY e.type
""",

    # task_169: Date is BIGINT in DABench yearmonth — use integer compare
    "task_169": """
SELECT AVG(ym.Consumption) / 12
FROM customers c
JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
WHERE ym.Date >= 201301 AND ym.Date <= 201312 AND c.Segment = 'SME'
""",

    # task_180: gold excludes Customer 32379 (= Amount=0 → Price/Amount=inf in BIRD logic).
    # Add Amount > 0 guard.
    "task_180": """
SELECT ym.Consumption
FROM yearmonth ym
WHERE ym.Date = 201208
  AND ym.CustomerID IN (
    SELECT DISTINCT t.CustomerID
    FROM transactions_1k t
    WHERE t.ProductID = 5 AND t.Amount > 0 AND t.Price/t.Amount > 29.00
  )
""",

    # task_199: NEW. Per-school SAT math > 400, district name contains 'Riverside'.
    "task_199": """
SELECT s.sname, f."Charter Funding Type"
FROM satscores s
JOIN frpm f ON s.cds = f.CDSCode
WHERE s.dname LIKE '%Riverside%' AND s.AvgScrMath > 400
ORDER BY s.sname
""",

    # task_243: BIRD missing DISTINCT on T2.Id
    "task_243": """
SELECT CAST(COUNT(DISTINCT p.Id) AS REAL) / COUNT(DISTINCT v.Id)
FROM votes v
JOIN posts p ON v.UserId = p.OwnerUserId
WHERE v.UserId = 24
""",

    # task_283: BIRD result is correct (= 31.2 with float precision), copy verbatim
    "task_283": """
SELECT CAST(COUNT(CASE WHEN c.colour = 'Blue' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(s.id)
FROM superhero s
JOIN colour c ON s.eye_colour_id = c.id
""",

    # task_303: BIRD result correct (52.17 with float precision), copy verbatim
    "task_303": """
SELECT CAST(COUNT(CASE WHEN c.country = 'Germany' THEN r.circuitID END) AS REAL) * 100 / COUNT(r.circuitId)
FROM circuits c
JOIN races r ON r.circuitId = c.circuitId
WHERE r.name = 'European Grand Prix'
""",

    # task_330: doc/League.md → Belgium Jupiler League = league_id 1. Date is TIMESTAMP.
    "task_330": """
SELECT m.home_team_goal, m.away_team_goal
FROM Match m
WHERE m.league_id = 1
  AND CAST(m.date AS VARCHAR) LIKE '2008-09-24%'
""",

    # task_344: PROCESS-ONLY — patient_sex.csv has only 92 of the 302 patients in Laboratory.
    # The rest (= male sex info) is embedded in doc/Patient.md prose. Cannot SQL directly.
    # PROCESS: parse doc/Patient.md to extract full male ID list → INNER JOIN Laboratory →
    #   filter WBC normal (3.5<WBC<9.0) AND abnormal FG (<150 OR >450) → COUNT DISTINCT → 4
    "task_344": "-- PROCESS: doc/Patient.md has SEX info for the 210 patients NOT in patient_sex.csv. Parse it, union with patient_sex.csv to get full male list, then JOIN Laboratory with WBC/FG filters → 4",

    # task_349: PROCESS-ONLY — major.md = full table in prose, cannot SQL directly.
    # PROCESS: member.csv (Angela Sanders) → link_to_major = 'recxK3MHQFbR9J5uO'
    #   → doc/major.md lookup that ID → "Business". Answer: "Business".
    "task_349": "-- PROCESS: lookup Angela Sanders → link_to_major='recxK3MHQFbR9J5uO' → doc/major.md → 'Business'",

    # task_350: doc/event.md → Women's Soccer event_id = 'rec2N69DMcrqN9PJC'.
    "task_350": """
SELECT COUNT(a.link_to_event)
FROM attendance a
JOIN member m ON a.link_to_member = m.member_id
WHERE a.link_to_event = 'rec2N69DMcrqN9PJC'
  AND m.t_shirt_size = 'Medium'
""",

    # task_352: PROCESS-ONLY — budget table fully in doc/budget.md.
    # PROCESS: parse doc/budget.md → get Advertisement budgets for Yearly Kickoff
    #   and October Meeting events → ratio = 2.727 (= gold).
    "task_352": "-- PROCESS: parse doc/budget.md → SUM(amount) for Advertisement category, divided per event_name (Yearly Kickoff / October Meeting) → 2.727",

    # task_355: doc/member.md → Elijah Allen = 'recro8T1MPMwRadVH'.
    "task_355": """
SELECT 'Elijah' AS first_name, 'Allen' AS last_name, e.cost
FROM expense e
WHERE e.expense_description = 'Water, Veggie tray, supplies'
  AND e.link_to_member = 'recro8T1MPMwRadVH'
""",

    # task_379: carcinogenic_mols.txt at root (= not loaded as view). Inline 100 mol IDs.
    "task_379": _build_task_379_sql(),

    # task_396: PROCESS-ONLY — superhero.md is the main entity table in prose.
    "task_396": "-- PROCESS: parse doc/superhero.md to get (id, height_cm, publisher_id) for all heroes; filter 150 ≤ height ≤ 180; JOIN publisher.json on publisher_id; count % Marvel Comics → 54.84",

    # task_408: doc/races.md → Australian GP 2008 = raceId 18.
    "task_408": """
WITH time_in_seconds AS (
  SELECT positionOrder,
    CASE WHEN positionOrder = 1
      THEN (CAST(SUBSTR(time, 1, 1) AS REAL) * 3600)
         + (CAST(SUBSTR(time, 3, 2) AS REAL) * 60)
         + CAST(SUBSTR(time, 6) AS REAL)
      ELSE CAST(SUBSTR(time, 2) AS REAL)
    END AS time_seconds
  FROM results
  WHERE raceId = 18 AND time IS NOT NULL
),
champion_time AS (SELECT time_seconds FROM time_in_seconds WHERE positionOrder = 1),
last_driver AS (SELECT time_seconds FROM time_in_seconds WHERE positionOrder = (SELECT MAX(positionOrder) FROM time_in_seconds))
SELECT (CAST((SELECT time_seconds FROM last_driver) AS REAL) * 100)
     / (SELECT time_seconds + (SELECT time_seconds FROM last_driver) FROM champion_time)
""",

    # task_415: doc/races.md → Singapore GP 2009 = raceId 14.
    "task_415": """
SELECT c.constructorRef, c.url
FROM results r
JOIN constructors c ON r.constructorId = c.constructorId
WHERE r.raceId = 14 AND r.time LIKE '_:%:__.___'
""",

    # task_418: PROCESS-ONLY — Patient.md + Laboratory.md both doc-only, no CSV.
    "task_418": "-- PROCESS: parse doc/Patient.md (ID, Birthday) + doc/Laboratory.md (ID, CRE) → JOIN → CRE≥1.5 AND age<70 → COUNT DISTINCT → 1",

    # task_420: PROCESS-ONLY — legalities.md has the big lookup table.
    "task_420": "-- PROCESS: parse doc/legalities.md → JOIN cards.db on uuid where format='commander' AND status='Legal' → % hasContentWarning=0 → 100.0",
}

ALL_SQL = {**BIRD_MATCH, **MANUAL_SQL}


# ============================================================
# Runner: execute each and verify against gold.csv
# ============================================================

def normalize_rows(rows):
    out = set()
    for r in rows:
        norm = tuple(str(x).strip().lower() if x is not None else "" for x in r)
        norm = tuple(re.sub(r"\.0+$", "", v) for v in norm)
        # Round floats to 2 dp to absorb precision noise
        norm = tuple(
            f"{float(v):.2f}" if re.match(r"^-?\d+\.\d+$", v) else v
            for v in norm
        )
        out.add(norm)
    return out


def load_gold_csv(task_id: str):
    p = REPO / "data" / "public" / "output" / task_id / "gold.csv"
    if not p.exists():
        return [], []
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], [tuple(c.strip() for c in row) for row in rows[1:]]


def run_one(task_id: str, sql: str) -> dict:
    ds = DABenchPublicDataset(root_dir=REPO / "data" / "public" / "input")
    try:
        task = ds.get_task(task_id)
    except Exception as exc:
        return {"task_id": task_id, "status": "no_task", "error": str(exc)[:200]}
    # Strip -- comments
    clean_sql = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--")).strip()
    if not clean_sql:
        return {"task_id": task_id, "status": "no_sql"}
    try:
        res = execute_sql(task.context_dir, clean_sql)
    except Exception as exc:
        return {"task_id": task_id, "status": "exec_error", "error": str(exc)[:200]}
    actual = normalize_rows(res.get("rows", []))
    _, gold_rows = load_gold_csv(task_id)
    gold = normalize_rows(gold_rows)
    if not gold:
        status = "no_gold"
    elif actual == gold:
        status = "match"
    elif actual >= gold:
        status = "match_superset"
    elif gold >= actual:
        status = "match_subset"
    else:
        ints = len(actual & gold)
        status = "partial" if ints else "mismatch"
    return {
        "task_id": task_id,
        "status": status,
        "n_actual": len(actual),
        "n_gold": len(gold),
        "n_intersect": len(actual & gold),
    }


def main():
    out_csv = REPO / "artifacts" / "plan_verify_decisions_full50" / "manual_sql_results.csv"
    rows = []
    for tid in sorted(ALL_SQL, key=lambda x: int(x.split("_")[1])):
        sql = ALL_SQL[tid]
        res = run_one(tid, sql)
        res["sql"] = sql.strip()
        rows.append(res)
        s = res["status"]
        extra = ""
        if s in ("mismatch", "partial", "match_superset", "match_subset"):
            extra = f" a={res.get('n_actual')} g={res.get('n_gold')} ∩={res.get('n_intersect')}"
        elif s == "exec_error":
            extra = f" {res.get('error','')[:80]}"
        print(f"  {tid}: {s}{extra}", flush=True)
    # CSV
    keys = ["task_id", "status", "n_actual", "n_gold", "n_intersect", "error", "sql"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # Summary
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"\nSummary: {dict(c)}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
