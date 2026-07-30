"""Mine question→SQL universal patterns from BIRD dev 1534 examples.

For each example, extract:
  - Question features (= ~30 regex patterns matching common phrasings)
  - SQL features (= ~20 regex patterns matching SQL structures)

Then compute conditional probabilities P(sql_feat | q_feat) to identify
universal heuristics. High precision + decent support = reliable rule.

Output: ranked rules table.
"""
from __future__ import annotations

import json, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIRD = REPO / "data" / "external" / "bird" / "dev_20240627" / "dev.json"


# ============ Question patterns ============
Q_PATTERNS = {
    "Q_percent": r"\b(percent|percentage|%)\b",
    "Q_ratio": r"\b(ratio|times more|how many times|times compared to)\b",
    "Q_per_unit": r"\bper\s+(unit|item|day|hour|minute|capita|person|customer|user|month|year)\b",
    "Q_avg_monthly": r"\baverage\s+(monthly|yearly|daily|hourly|weekly|annual)\b",
    "Q_avg_word": r"\baverage\b|\bavg\b|\bmean\b",
    "Q_total": r"\btotal\b|\bsum of\b",
    "Q_count": r"\b(how many|count of|number of|tally)\b",
    "Q_max": r"\b(highest|maximum|most|largest|best|top)\b",
    "Q_min": r"\b(lowest|minimum|least|smallest|worst|cheapest)\b",
    "Q_ranked": r"\branked\s+(first|second|third|\d+|1st|2nd|3rd)\b",
    "Q_list_of_X": r"\b(list|name|give|provide|identify|show|state)\b",
    "Q_full_name": r"\bfull\s*name\b",
    "Q_difference": r"\bdifference\b|\bdiffer(s|ence)?\b|\bgap\b",
    "Q_growth": r"\bgrowth\b|\bincrease\b|\bgrowth rate\b",
    "Q_year_range": r"\bbetween\s+\d{4}\s+and\s+\d{4}\b|\bfrom\s+\d{4}\s+to\s+\d{4}\b",
    "Q_year_single": r"\b(in|for|during)\s+(\d{4})\b",
    "Q_age": r"\bage\b|\byears\s+old\b|\bborn\b",
    "Q_quoted_value": r"'[^']+'|\"[^\"]+\"",
    "Q_among": r"\bamong\b",
    "Q_compared": r"\bcompared\s+to\b",
    "Q_distinct": r"\bdistinct\b|\bdifferent\b|\bunique\b",
    "Q_or_logic": r"\b(or|either)\b",
    "Q_and_logic": r"\b(and|both|with both|all of)\b",
    "Q_not_logic": r"\b(not|aren't|isn't|without|except)\b",
}

# ============ SQL patterns ============
SQL_PATTERNS = {
    "S_cast_real": r"CAST\s*\([^)]+AS\s+(REAL|FLOAT|DOUBLE)\)",
    "S_mul_100": r"\*\s*100(\.0|\b)",
    "S_div_col": r"/\s*[a-zA-Z_]\w*",  # division by a column-like identifier
    "S_div_number": r"/\s*[\d.]+",  # division by literal number
    "S_avg": r"\bAVG\s*\(",
    "S_sum": r"\bSUM\s*\(",
    "S_count_star": r"\bCOUNT\s*\(\s*\*\s*\)",
    "S_count_distinct": r"\bCOUNT\s*\(\s*DISTINCT\s+",
    "S_max": r"\bMAX\s*\(",
    "S_min": r"\bMIN\s*\(",
    "S_limit_1": r"\bLIMIT\s+1\b",
    "S_filter_back": r"WHERE.+?=\s*\(\s*SELECT\s+(MIN|MAX|AVG)",
    "S_case_when": r"\bCASE\s+WHEN\b",
    "S_distinct": r"\bDISTINCT\b",
    "S_group_by": r"\bGROUP\s+BY\b",
    "S_having": r"\bHAVING\b",
    "S_subquery": r"\(\s*SELECT\b",
    "S_join": r"\bJOIN\b",
    "S_in_list": r"\bIN\s*\(",
    "S_or_clause": r"\bOR\b",
    "S_and_clause": r"\bAND\b",
    "S_not_clause": r"\b(NOT|<>|!=)\b",
    "S_like": r"\bLIKE\b",
    "S_strftime": r"\bSTRFTIME\b|\bSUBSTR\(",
    "S_div_12": r"/\s*12\b",
}


def features(text: str, patterns: dict) -> set:
    return {name for name, pat in patterns.items() if re.search(pat, text, re.IGNORECASE | re.DOTALL)}


def main():
    data = json.load(open(BIRD))
    print(f"Loaded {len(data)} BIRD examples")

    # Compute (q_features, s_features) per example
    rows = []
    for ex in data:
        q = ex["question"]
        sql = ex["SQL"]
        rows.append({
            "q_feats": features(q, Q_PATTERNS),
            "s_feats": features(sql, SQL_PATTERNS),
        })

    # Co-occurrence: for each (q_feat, s_feat) compute precision = P(s|q)
    # and support = count(q)
    q_counts = Counter()
    cooccur = Counter()  # (q, s) → count
    for r in rows:
        for q in r["q_feats"]:
            q_counts[q] += 1
            for s in r["s_feats"]:
                cooccur[(q, s)] += 1

    # Build rules table: q → s with precision >= 0.7 and support >= 30
    rules = []
    for (q, s), c in cooccur.items():
        n_q = q_counts[q]
        precision = c / n_q
        # baseline: P(s) overall
        n_s = sum(1 for r in rows if s in r["s_feats"])
        baseline = n_s / len(rows)
        lift = precision / baseline if baseline > 0 else 0
        rules.append({
            "if_q": q, "then_s": s,
            "support": n_q,
            "precision": precision,
            "baseline": baseline,
            "lift": lift,
            "count": c,
        })

    # Multi-tier reporting
    strong = [r for r in rules if r["precision"] >= 0.6 and r["support"] >= 20 and r["lift"] >= 1.5]
    strong.sort(key=lambda r: (-r["lift"], -r["precision"]))

    print(f"\n=== STRONG RULES (precision≥0.6, support≥20, lift≥1.5) ===")
    print(f"{'IF q has':24s} | {'THEN sql has':22s} | {'prec':>5s} | {'lift':>5s} | {'supp':>5s}")
    print("-" * 80)
    for r in strong[:30]:
        print(f"  {r['if_q']:22s} | {r['then_s']:22s} | {r['precision']:>5.2f} | {r['lift']:>5.2f} | {r['support']:>5d}")

    # Output to file
    OUT = REPO / "artifacts" / "bird_pattern_mining"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rules.json", "w") as f:
        json.dump(strong, f, indent=2)
    print(f"\nSaved: {OUT / 'rules.json'}")


if __name__ == "__main__":
    main()
