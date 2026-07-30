"""Mine BIRD examples where SQL has DIVISION but Q has no explicit /, per, ratio.
Find what Q phrasings trigger implicit division.
"""
from __future__ import annotations

import json, re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIRD = REPO / "data" / "external" / "bird" / "dev_20240627" / "dev.json"


EXPLICIT_DIV_Q = re.compile(r"\b(per\b|ratio|/|divided by|times more|how many times)", re.IGNORECASE)


def has_division_in_sql(sql: str) -> tuple[bool, str]:
    """Detect division and classify type."""
    # 1. / operator
    if re.search(r"\b\w+\s*/\s*[a-zA-Z_(]\w*", sql):
        return True, "div_operator"
    # 2. AVG()
    if re.search(r"\bAVG\s*\(", sql, re.IGNORECASE):
        return True, "AVG"
    # 3. /literal
    if re.search(r"/\s*\d+", sql):
        return True, "div_literal"
    return False, ""


def main():
    data = json.load(open(BIRD))
    implicit_div = []  # (q, sql, division_type)
    for ex in data:
        q = ex["question"]
        sql = ex["SQL"]
        has_div, div_type = has_division_in_sql(sql)
        if not has_div: continue
        if EXPLICIT_DIV_Q.search(q): continue  # explicit, skip
        implicit_div.append((q, sql, div_type, ex.get("evidence",""), ex.get("db_id","")))

    print(f"Implicit division (= SQL has /, AVG, /N but Q has no explicit per/ratio): {len(implicit_div)}/{len(data)} = {len(implicit_div)/len(data)*100:.1f}%\n")

    # Count division types
    type_counts = Counter(t for _,_,t,_,_ in implicit_div)
    print(f"Division type breakdown:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")
    print()

    # Extract Q-trigger phrases (= bigrams / trigrams common to these Qs)
    q_phrases = Counter()
    for q, _, _, _, _ in implicit_div:
        ql = q.lower()
        # Common trigger phrases
        triggers = [
            ("average", r"\baverage\b"),
            ("percentage", r"\bpercentage\b"),
            ("percent_word", r"\bpercent\b"),
            ("percent_sign", r"%"),
            ("mean", r"\bmean\b"),
            ("rate", r"\brate\b"),
            ("share", r"\bshare\b"),
            ("portion", r"\bportion\b"),
            ("proportion", r"\bproportion\b"),
            ("density", r"\bdensity\b"),
            ("ratio_word", r"\bratio\b"),  # already filtered explicit but check anyway
            ("frequency", r"\bfrequency\b"),
            ("how_often", r"\bhow often\b"),
            ("for_each", r"\bfor each\b|\bfor every\b"),
            ("normalised", r"\bnormali[sz]ed\b"),
            ("ratio_explicit", r":"),  # X:Y notation
        ]
        for name, pat in triggers:
            if re.search(pat, ql):
                q_phrases[name] += 1

    print(f"Q trigger words / phrases in implicit-division examples:")
    for name, c in q_phrases.most_common():
        pct = c / len(implicit_div) * 100
        print(f"  {name:20s}: {c:>5d} ({pct:.1f}%)")
    print()

    # Sample examples per type
    print(f"=== SAMPLE EXAMPLES (= 3 per category) ===")
    for cat in ["AVG", "div_operator", "div_literal"]:
        cat_examples = [(q, sql, ev) for q, sql, t, ev, db in implicit_div if t == cat][:5]
        print(f"\n--- {cat} ---")
        for q, sql, ev in cat_examples:
            print(f"  Q: {q[:120]}")
            print(f"  EV: {ev[:120]}")
            print(f"  SQL: {sql[:200]}")
            print()


if __name__ == "__main__":
    main()
