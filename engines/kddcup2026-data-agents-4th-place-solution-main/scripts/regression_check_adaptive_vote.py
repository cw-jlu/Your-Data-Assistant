"""Regression check: simulate vote strategies on synthetic + real attempts.

Compares:
  - majority (= most common signature)
  - union (= column-merge across attempts)
  - adaptive_v3 (= token-set divergence detection + plurality)

Pass criteria: adaptive_v3 must NOT score worse than majority in any pattern.

Usage:
    uv run python scripts/regression_check_adaptive_vote.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import _evaluate_task, EvaluationOptions


# ============================================================================
# 3 vote strategies
# ============================================================================

def _norm(v):
    if v is None: return (0, "")
    s = str(v).strip()
    try: return (1, round(float(s), 3))
    except: return (2, s.lower())


def _signature(a):
    if not a or not a.columns: return ("__empty__",)
    n_cols = len(a.columns)
    rows = []
    for r in a.rows or []:
        if len(r) != n_cols: continue
        rows.append(tuple(_norm(v) for v in r))
    rows.sort()
    return (n_cols, len(rows), tuple(rows))


def _majority(answers):
    s2i = {}
    for i, a in enumerate(answers):
        s2i.setdefault(_signature(a), []).append(i)
    best = max(s2i, key=lambda k: len(s2i[k]))
    return answers[s2i[best][0]]


def _union(answers):
    if not answers: return None
    sig_to_rep = {}
    for a in answers:
        sig = _signature(a)
        if sig not in sig_to_rep:
            sig_to_rep[sig] = a
    if len(sig_to_rep) == 1:
        return next(iter(sig_to_rep.values()))
    unique_cols = []
    seen = set()
    for a in answers:
        for ci, c in enumerate(a.columns):
            col_vals = [r[ci] if ci < len(r) else "" for r in a.rows]
            key = tuple(sorted(_norm(v) for v in col_vals))
            if key in seen: continue
            seen.add(key)
            unique_cols.append((c, col_vals))
    if not unique_cols: return answers[0]
    max_rows = max(len(c[1]) for c in unique_cols)
    cols = [c[0] for c in unique_cols]
    rows = [[(uc[1][ri] if ri < len(uc[1]) else "") for uc in unique_cols] for ri in range(max_rows)]
    return AnswerTable(columns=cols, rows=rows)


def _token_set(a):
    if not a or not a.rows: return frozenset()
    tokens = set()
    for row in a.rows:
        for v in row:
            if v is None: continue
            s = str(v).strip().lower()
            try: tokens.add(round(float(s), 3))
            except: tokens.update(s.split())
    return frozenset(tokens)


def _adaptive_v3(answers):
    non_empty = [a for a in answers if a and a.rows]
    if not non_empty: return answers[0] if answers else None
    groups = {}
    for a in non_empty:
        ts = _token_set(a)
        groups.setdefault(ts, []).append(a)
    sorted_g = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    largest = sorted_g[0]
    n = len(non_empty)
    if len(sorted_g) == 1:
        return _majority(non_empty)
    if len(largest[1]) > n // 2:
        return _majority(largest[1])
    return _union(non_empty)


def _adaptive_v4(answers):
    """Strict: any token-set divergence → union. Don't trust majority."""
    non_empty = [a for a in answers if a and a.rows]
    if not non_empty: return answers[0] if answers else None
    groups = {}
    for a in non_empty:
        ts = _token_set(a)
        groups.setdefault(ts, []).append(a)
    if len(groups) == 1:
        return _majority(non_empty)
    return _union(non_empty)


# ============================================================================
# Score helper: compares an answer to a gold CSV using the official metric
# ============================================================================

def score_against_gold(answer, gold_csv_text):
    if answer is None: return 0.0
    pred_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    gold_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    w = csv.writer(pred_path); w.writerow(answer.columns)
    for r in answer.rows: w.writerow([str(v) for v in r])
    pred_path.close()
    gold_path.write(gold_csv_text); gold_path.close()
    try:
        e = _evaluate_task(task_id="x", prediction_path=Path(pred_path.name),
                           gold_path=Path(gold_path.name), options=EvaluationOptions())
        return e.official_score_lambda_0_5
    except Exception:
        return 0.0
    finally:
        Path(pred_path.name).unlink(missing_ok=True)
        Path(gold_path.name).unlink(missing_ok=True)


# ============================================================================
# Test cases
# ============================================================================

CASES = []

def AT(cols, rows): return AnswerTable(columns=cols, rows=rows)

# Pattern 1: 全 attempts 同じ単一数値 (= COUNT 系)
CASES.append({
    "name": "P1: 全同 単一数値 (3-0)",
    "gold": "count\n42",
    "attempts": [AT(["count"], [["42"]]), AT(["count"], [["42"]]), AT(["count"], [["42"]])],
})
# Pattern 2: 単一数値 attempts 分岐
CASES.append({
    "name": "P2: 単一数値 分岐 (1.0 vs 2.0 vs 2.0)",
    "gold": "count\n1.0",
    "attempts": [AT(["x"], [["1.0"]]), AT(["x"], [["2.0"]]), AT(["x"], [["2.0"]])],
})
# Pattern 3: 全同 multi-row 数値
CASES.append({
    "name": "P3: multi-row 数値 全同",
    "gold": "n\n10\n20\n30",
    "attempts": [AT(["n"], [["10"],["20"],["30"]])]*3,
})
# Pattern 4: multi-row 数値 行ずれ
CASES.append({
    "name": "P4: multi-row 数値 行ずれ (一部 attempt が違う rows)",
    "gold": "n\n10\n20\n30",
    "attempts": [
        AT(["n"], [["10"],["20"],["30"]]),
        AT(["n"], [["10"],["20"],["30"]]),
        AT(["n"], [["10"],["20"],["40"]]),  # 1 row diff
    ],
})
# Pattern 5: 単一文字列 全同
CASES.append({
    "name": "P5: 単一文字列 全同",
    "gold": "name\nAlice",
    "attempts": [AT(["name"], [["Alice"]]), AT(["name"], [["Alice"]]), AT(["name"], [["Alice"]])],
})
# Pattern 6: 単一文字列 分岐
CASES.append({
    "name": "P6: 単一文字列 分岐 (= Speaker vs Officers tied set)",
    "gold": "event_name\nNovember Speaker\nOctober Speaker\nSeptember Speaker",
    "attempts": [
        AT(["event_name"], [["November Speaker"], ["October Speaker"], ["September Speaker"]]),
        AT(["event_name"], [["Officers meeting - November"], ["Officers meeting - October"], ["Officers meeting - September"]]),
        AT(["event_name"], [["Officers meeting - November"], ["Officers meeting - October"], ["Officers meeting - September"]]),
    ],
})
# Pattern 7: multi-row 文字列 全同
CASES.append({
    "name": "P7: 文字列 list 全同 (= 7 superheroes)",
    "gold": "superhero_name\nA\nB\nC\nD\nE\nF\nG",
    "attempts": [AT(["superhero_name"], [[c] for c in "ABCDEFG"])]*3,
})
# Pattern 8: 構造ブレ (= concat vs split)
CASES.append({
    "name": "P8: 構造ブレ (Sacha Harrison vs Sacha+Harrison)",
    "gold": "first_name,last_name,total_cost\nSacha,Harrison,866.25",
    "attempts": [
        AT(["full_name", "cost"], [["Sacha Harrison", "866.25"]]),
        AT(["first_name", "last_name", "cost"], [["Sacha", "Harrison", "866.25"]]),
        AT(["first_name", "last_name", "cost"], [["Sacha", "Harrison", "866.25"]]),
    ],
})
# Pattern 9: 1 attempt empty (= 0行)
CASES.append({
    "name": "P9: 1 attempt empty",
    "gold": "trans_id\n100\n200\n300",
    "attempts": [
        AT(["trans_id"], []),
        AT(["trans_id"], [["100"], ["200"], ["300"]]),
        AT(["trans_id"], [["100"], ["200"], ["300"]]),
    ],
})
# Pattern 10: 多列 numeric (= avg_x, avg_y)
CASES.append({
    "name": "P10: 2列 numeric (avg, avg)",
    "gold": "avg_a,avg_b\n12.34,56.78",
    "attempts": [AT(["avg_a", "avg_b"], [["12.34", "56.78"]])]*3,
})
# Pattern 11: extras 列を 1 attempt が含む
CASES.append({
    "name": "P11: 1 attempt が extras 列を含む (= 残りは 1 列)",
    "gold": "trans_id\n100\n200\n300",
    "attempts": [
        AT(["trans_id"], [["100"], ["200"], ["300"]]),
        AT(["trans_id"], [["100"], ["200"], ["300"]]),
        AT(["trans_id", "amount", "date"], [["100", "10.0", "2020-01-01"], ["200", "20.0", "2020-01-02"], ["300", "30.0", "2020-01-03"]]),
    ],
})
# Pattern 12: 三者三様 (= 1-1-1 split)
CASES.append({
    "name": "P12: 三者三様 (= 1-1-1 split, gold = first attempt)",
    "gold": "x\nA",
    "attempts": [AT(["x"], [["A"]]), AT(["x"], [["B"]]), AT(["x"], [["C"]])],
})


# ============================================================================
# Run
# ============================================================================

def main():
    print(f"{'pattern':<60} {'maj':>6} {'uni':>6} {'v3':>6} {'v4':>6} {'best':>6}")
    print('-' * 100)
    sum_maj = sum_uni = sum_v3 = sum_v4 = 0.0
    v4_regr = []
    for c in CASES:
        a_maj = _majority(c["attempts"])
        a_uni = _union(c["attempts"])
        a_v3 = _adaptive_v3(c["attempts"])
        a_v4 = _adaptive_v4(c["attempts"])
        s_maj = score_against_gold(a_maj, c["gold"])
        s_uni = score_against_gold(a_uni, c["gold"])
        s_v3 = score_against_gold(a_v3, c["gold"])
        s_v4 = score_against_gold(a_v4, c["gold"])
        sum_maj += s_maj; sum_uni += s_uni; sum_v3 += s_v3; sum_v4 += s_v4
        flag = "✓" if s_v4 >= max(s_maj, s_uni) else f"v4<best by {max(s_maj,s_uni)-s_v4:.2f}"
        if s_v4 < max(s_maj, s_uni):
            v4_regr.append((c['name'], max(s_maj, s_uni) - s_v4))
        print(f"  {c['name']:<60} {s_maj:>6.2f} {s_uni:>6.2f} {s_v3:>6.2f} {s_v4:>6.2f} {flag:>6}")
    print()
    print(f"sum: maj={sum_maj:.2f}  uni={sum_uni:.2f}  v3={sum_v3:.2f}  v4={sum_v4:.2f}")
    print()
    print(f"v4 regressions: {len(v4_regr)}")
    for name, delta in v4_regr:
        print(f"  - {name}: -{delta:.3f}")

if __name__ == "__main__":
    main()
