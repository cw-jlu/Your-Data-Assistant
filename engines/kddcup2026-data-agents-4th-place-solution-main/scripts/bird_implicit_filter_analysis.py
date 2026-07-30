"""Deep analysis: what IMPLICIT filters does BIRD gold SQL add that aren't
in the question text?

Categories to mine:
  - IS NOT NULL (null guards)
  - NULLIF (division guards)
  - Range constraints (> 0, BETWEEN with no Q hint)
  - Inner join implicit row filter
  - COALESCE
  - DISTINCT scope
  - GROUP BY structure not implied by Q
  - LIMIT / ORDER BY style for superlatives
  - Evidence field overlap (= what BIRD evidence says vs Q says vs SQL has)

Output: per-pattern frequency, examples, and Q→SQL correlation (= when do
implicit filters appear, what's the trigger phrase?).
"""
from __future__ import annotations

import json, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIRD = REPO / "data" / "external" / "bird" / "dev_20240627" / "dev.json"


def main():
    data = json.load(open(BIRD))
    print(f"Loaded {len(data)} BIRD examples\n")

    # ============ 1. IS NOT NULL frequency ============
    not_null_count = 0
    not_null_examples = []
    for ex in data:
        sql = ex["SQL"]
        q = ex["question"]
        ev = ex.get("evidence", "")
        if re.search(r"IS\s+NOT\s+NULL", sql, re.IGNORECASE):
            not_null_count += 1
            # Identify the columns marked NOT NULL
            cols = re.findall(r"(\w+)\.?\w*\s+IS\s+NOT\s+NULL", sql, re.IGNORECASE)
            not_null_examples.append({
                "q": q[:60], "ev": ev[:60], "cols": cols, "sql": sql[:200]
            })
    print(f"=== IS NOT NULL ===")
    print(f"  appears in {not_null_count}/{len(data)} = {not_null_count/len(data)*100:.1f}%")
    print(f"  sample (Q does NOT mention NULL):")
    for e in not_null_examples[:5]:
        print(f"    Q: {e['q']}")
        print(f"    ev: {e['ev']}")
        print(f"    cols: {e['cols']}")
        print()

    # ============ 2. NULLIF / COALESCE (division guards) ============
    nullif_count = sum(1 for ex in data if re.search(r"NULLIF\s*\(", ex["SQL"], re.IGNORECASE))
    coalesce_count = sum(1 for ex in data if re.search(r"COALESCE\s*\(", ex["SQL"], re.IGNORECASE))
    print(f"=== NULLIF / COALESCE ===")
    print(f"  NULLIF: {nullif_count}/{len(data)} = {nullif_count/len(data)*100:.1f}%")
    print(f"  COALESCE: {coalesce_count}/{len(data)} = {coalesce_count/len(data)*100:.1f}%")
    print()

    # ============ 3. DISTINCT vs no DISTINCT ============
    distinct_count = sum(1 for ex in data if re.search(r"\bDISTINCT\b", ex["SQL"], re.IGNORECASE))
    # When does DISTINCT appear? Q has "list" / "names" / etc?
    distinct_q_features = Counter()
    no_distinct_q_features = Counter()
    for ex in data:
        is_distinct = bool(re.search(r"\bDISTINCT\b", ex["SQL"], re.IGNORECASE))
        q = ex["question"].lower()
        feats = []
        if re.search(r"\b(list|names|all of|every)\b", q): feats.append("list_word")
        if re.search(r"\b(how many|count)\b", q): feats.append("count_word")
        if re.search(r"\bunique\b", q): feats.append("unique_word")
        for f in feats:
            if is_distinct: distinct_q_features[f] += 1
            else: no_distinct_q_features[f] += 1
    print(f"=== DISTINCT ===")
    print(f"  appears in {distinct_count}/{len(data)} = {distinct_count/len(data)*100:.1f}%")
    print(f"  Q features when DISTINCT used:")
    for f, c in distinct_q_features.most_common():
        no = no_distinct_q_features.get(f, 0)
        total = c + no
        print(f"    Q has '{f}': DISTINCT in {c}/{total} = {c/total*100:.0f}%")
    print()

    # ============ 4. ORDER BY ... LIMIT 1 (superlative) ============
    order_limit_count = 0
    filter_back_count = 0
    for ex in data:
        sql = ex["SQL"]
        q = ex["question"].lower()
        is_superlative = bool(re.search(r"\b(highest|lowest|maximum|minimum|most|least|top|best|worst)\b", q))
        if not is_superlative: continue
        if re.search(r"ORDER\s+BY.+?LIMIT\s+1\b", sql, re.IGNORECASE | re.DOTALL):
            order_limit_count += 1
        if re.search(r"WHERE.+?=\s*\(\s*SELECT\s+(MIN|MAX)", sql, re.IGNORECASE | re.DOTALL):
            filter_back_count += 1
    print(f"=== Superlative handling (Q has highest/lowest/etc) ===")
    superlative_n = sum(1 for ex in data if re.search(r"\b(highest|lowest|maximum|minimum|most|least|top|best|worst)\b", ex["question"], re.IGNORECASE))
    print(f"  total superlative Qs: {superlative_n}")
    print(f"  ORDER BY ... LIMIT 1: {order_limit_count}/{superlative_n} = {order_limit_count/superlative_n*100:.0f}%")
    print(f"  filter-back (= col = (SELECT MIN/MAX)): {filter_back_count}/{superlative_n} = {filter_back_count/superlative_n*100:.0f}%")
    print()

    # ============ 5. INNER JOIN as implicit row filter ============
    inner_join_count = sum(1 for ex in data if re.search(r"INNER\s+JOIN", ex["SQL"], re.IGNORECASE))
    left_join_count = sum(1 for ex in data if re.search(r"LEFT\s+JOIN", ex["SQL"], re.IGNORECASE))
    print(f"=== JOIN preference ===")
    print(f"  INNER JOIN: {inner_join_count}/{len(data)} = {inner_join_count/len(data)*100:.1f}%")
    print(f"  LEFT JOIN: {left_join_count}/{len(data)} = {left_join_count/len(data)*100:.1f}%")
    print(f"  → BIRD strongly prefers INNER JOIN (= rows without match excluded implicitly)")
    print()

    # ============ 6. Evidence field → SQL pattern correlation ============
    # If evidence mentions "Y means X", the SQL likely filters on Y=X
    evidence_eq_count = 0
    for ex in data:
        ev = ex.get("evidence", "")
        if re.search(r"\bmeans\b|\brefers to\b|\bequals\b|\bidentifier\b", ev, re.IGNORECASE):
            evidence_eq_count += 1
    print(f"=== Evidence field: 'X means Y' pattern ===")
    print(f"  evidence has equality hint: {evidence_eq_count}/{len(data)} = {evidence_eq_count/len(data)*100:.1f}%")
    print(f"  → BIRD evidence frequently defines SQL filter values")
    print()

    # ============ 7. Per-domain implicit-filter density ============
    domain_implicit = defaultdict(lambda: {"total": 0, "implicit_filter": 0})
    for ex in data:
        db = ex["db_id"]
        sql = ex["SQL"]
        domain_implicit[db]["total"] += 1
        # Implicit filter = IS NOT NULL OR (Q has no number/word but SQL has WHERE)
        has_implicit = (
            bool(re.search(r"IS\s+NOT\s+NULL", sql, re.IGNORECASE)) or
            bool(re.search(r"NULLIF\b", sql, re.IGNORECASE))
        )
        if has_implicit:
            domain_implicit[db]["implicit_filter"] += 1
    print(f"=== Implicit-filter density by domain ===")
    rows = []
    for db, d in domain_implicit.items():
        rate = d["implicit_filter"] / d["total"] if d["total"] else 0
        rows.append((db, d["total"], d["implicit_filter"], rate))
    rows.sort(key=lambda x: -x[3])
    print(f"  {'domain':35s} | total | implicit | rate")
    for db, t, i, r in rows:
        print(f"    {db:33s} | {t:>5d} | {i:>8d} | {r*100:>4.1f}%")


if __name__ == "__main__":
    main()
