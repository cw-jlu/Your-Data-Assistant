"""POC v2: LOOSE filter check (= 必要条件のリストアップ + 存在確認).

A. soft_filter_lister(question) → list of free-text filter concepts
B. presence_checker(concepts, sql) → for each concept: present? Y/N + reason

Test on filter-missing tasks:
  - task_180: Amount>0 guard missing (= real failure)
  - Synthetic: take a passing task's SQL, remove ONE filter, check if checker catches
"""
from __future__ import annotations

import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter, ModelMessage


_LISTER_SYS = """List the filter conditions a SQL query MUST implement to correctly answer
this question. Each condition is a short natural-language phrase (= not column-level).

Examples (= unrelated topics):
Q: "Find books published before 1990 in the fiction genre"
["published before 1990", "fiction genre"]

Q: "Top albums where average track score exceeds 8.0"
["average track score exceeds 8.0"]

Q: "Active employees in HR with bonus paid this year"
["active employee", "HR department", "bonus paid this year"]

Output: JSON array of short phrases. ONLY the JSON array."""


_CHECKER_SYS = """For each filter concept, check if the SQL implements it.
Output JSON:
  {"checks": [{"concept": "<phrase>", "present": true/false, "evidence": "<SQL fragment or 'not found'>"}],
   "all_present": true/false,
   "missing": ["<list of missing concepts>"]}

Be LENIENT (= if a SQL fragment plausibly implements the concept, mark present=true).
Be STRICT only when a concept is clearly absent from the SQL.

ONLY the JSON."""


# Test cases: (id, question, sql, expected_missing_count)
TESTS = [
    # Real failure case
    ("task_180_real",
     "For all the people who paid more than 29.00 per unit of product id No.5. Give their consumption status in the August of 2012.",
     "SELECT ym.Consumption FROM yearmonth ym WHERE ym.Date = 201208 AND ym.CustomerID IN (SELECT t.CustomerID FROM transactions_1k t WHERE t.ProductID = 5 AND t.Price/t.Amount > 29.00)",
     1),  # missing: Amount>0 guard (= divide-by-zero)
    # Synthetic: take task_22 SQL, remove the source='Dues' filter
    ("task_22_synth_remove_dues",
     "State the date Connor Hilton paid his/her dues.",
     "SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = 'Connor' AND T1.last_name = 'Hilton'",
     1),  # missing: source='Dues' filter
    # Synthetic: task_67 (AVG female weight), drop female
    ("task_67_synth_remove_female",
     "What is the average weight of all female superheroes?",
     "SELECT AVG(weight_kg) FROM superhero",
     1),  # missing: female filter
    # Synthetic: task_24 (Women's Soccer count), drop event_name filter
    ("task_24_synth_remove_event",
     "How many members attended the Women's Soccer event?",
     "SELECT COUNT(link_to_member) FROM attendance",
     1),  # missing: event_name='Women's Soccer'
    # Sanity: correct SQL — should report all_present=true
    ("task_22_sanity_correct",
     "State the date Connor Hilton paid his/her dues.",
     "SELECT T2.date_received FROM member AS T1 INNER JOIN income AS T2 ON T1.member_id = T2.link_to_member WHERE T1.first_name = 'Connor' AND T1.last_name = 'Hilton' AND T2.source = 'Dues'",
     0),
    ("task_67_sanity_correct",
     "What is the average weight of all female superheroes?",
     "SELECT AVG(s.weight_kg) FROM superhero s JOIN gender g ON s.gender_id = g.id WHERE g.gender = 'Female'",
     0),
    # Extra sanity (= correct SQLs from real passing tasks)
    ("task_11_sanity",
     "For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.",
     "SELECT DISTINCT T1.ID, T1.SEX, T1.Diagnosis FROM Patient T1 INNER JOIN Examination T2 ON T1.ID = T2.ID WHERE T2.Thrombosis = 2",
     0),
    ("task_19_sanity",
     "List the full name of the Student_Club members that grew up in Illinois state.",
     "SELECT T1.first_name, T1.last_name FROM member T1 INNER JOIN zip_code T2 ON T1.zip = T2.zip_code WHERE T2.state = 'Illinois'",
     0),
    ("task_26_sanity",
     "How many members of the Student_Club have major in 'Physics Teaching'?",
     "SELECT COUNT(T2.member_id) FROM major T1 INNER JOIN member T2 ON T1.major_id = T2.link_to_major WHERE T1.major_name = 'Physics Teaching'",
     0),
    ("task_75_sanity",
     "What is the surname of the driver with the best lap time in race number 19 in the second qualifying period?",
     "SELECT T2.surname FROM qualifying T1 INNER JOIN drivers T2 ON T2.driverId = T1.driverId WHERE T1.raceId = 19 ORDER BY T1.q2 ASC LIMIT 1",
     0),
    ("task_173_sanity",
     "Please list the countries of the gas stations with transactions taken place in June 2013.",
     "SELECT DISTINCT T2.Country FROM transactions_1k T1 INNER JOIN gasstations T2 ON T1.GasStationID = T2.GasStationID INNER JOIN yearmonth T3 ON T1.CustomerID = T3.CustomerID WHERE T3.Date = '201306'",
     0),
    ("task_218_sanity",
     "What is the telephone number for the school with the lowest average score in reading in Fresno?",
     "SELECT T2.Phone FROM satscores T1 INNER JOIN schools T2 ON T1.cds = T2.CDSCode WHERE T2.District = 'Fresno Unified' AND T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT 1",
     0),
    ("task_269_sanity",
     "What are the names of the superheroes with the power of death touch?",
     "SELECT T1.superhero_name FROM superhero T1 INNER JOIN hero_power T2 ON T1.id = T2.hero_id INNER JOIN superpower T3 ON T2.power_id = T3.id WHERE T3.power_name = 'Death Touch'",
     0),
    ("task_287_sanity",
     "Identify the gender of the superhero who has the ability of Phoenix Force.",
     "SELECT T4.gender FROM superhero T1 INNER JOIN hero_power T2 ON T1.id = T2.hero_id INNER JOIN superpower T3 ON T2.power_id = T3.id INNER JOIN gender T4 ON T1.gender_id = T4.id WHERE T3.power_name = 'Phoenix Force'",
     0),
    ("task_305_sanity",
     "What was the fastest lap speed among all drivers in the 2009 Spanish Grand Prix?",
     "SELECT T2.fastestLapSpeed FROM races T1 INNER JOIN results T2 ON T2.raceId = T1.raceId WHERE T1.name = 'Spanish Grand Prix' AND T1.year = 2009 AND T2.fastestLapSpeed IS NOT NULL ORDER BY T2.fastestLapSpeed DESC LIMIT 1",
     0),
]


def make_model():
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ["AGENT_API_BASE"],
        api_key=os.environ["AGENT_API_KEY"],
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


def list_filters(question, model):
    r = model.complete(
        [ModelMessage(role="system", content=_LISTER_SYS),
         ModelMessage(role="user", content=f"Q: {question}\n\nFilters:")],
        enable_thinking=False, max_tokens=512,
    )
    m = re.search(r"\[.*\]", r, re.DOTALL)
    if not m: return []
    try: return json.loads(m.group(0))
    except: return []


def check_presence(concepts, sql, model):
    r = model.complete(
        [ModelMessage(role="system", content=_CHECKER_SYS),
         ModelMessage(role="user", content=f"Concepts: {json.dumps(concepts)}\n\nSQL: {sql}\n\nCheck:")],
        enable_thinking=False, max_tokens=1024,
    )
    m = re.search(r"\{.*\}", r, re.DOTALL)
    if not m: return {"all_present": "?", "missing": []}
    try: return json.loads(m.group(0))
    except: return {"all_present": "?", "missing": [], "raw": r[:200]}


def run_one(tid, question, sql, expected_missing):
    model = make_model()
    concepts = list_filters(question, model)
    verdict = check_presence(concepts, sql, model)
    actual_missing = len(verdict.get("missing", []))
    correct = (
        (expected_missing == 0 and verdict.get("all_present") is True)
        or (expected_missing > 0 and verdict.get("all_present") is False)
    )
    return {
        "id": tid, "q": question[:60],
        "concepts": concepts,
        "missing_expected": expected_missing,
        "missing_detected": actual_missing,
        "missing_list": verdict.get("missing", []),
        "all_present": verdict.get("all_present"),
        "correct": correct,
    }


def main():
    print(f"=== loose filter check POC, {len(TESTS)} cases ===\n")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, *t): t[0] for t in TESTS}
        results = [fut.result() for fut in as_completed(futs)]

    print(f"{'id':35s} | exp_miss | got_miss | all_pres | correct | missing")
    print("-" * 130)
    for r in sorted(results, key=lambda x: x["id"]):
        m = "✓" if r["correct"] else "✗"
        print(f"  {r['id']:33s} | {r['missing_expected']:^8d} | {r['missing_detected']:^8d} | {str(r['all_present']):^8s} | {m:^7s} | {r['missing_list']}")
    n_ok = sum(1 for r in results if r["correct"])
    print(f"\naccuracy: {n_ok}/{len(results)}")
    Path("artifacts/filter_check_loose_poc").mkdir(parents=True, exist_ok=True)
    with open("artifacts/filter_check_loose_poc/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
