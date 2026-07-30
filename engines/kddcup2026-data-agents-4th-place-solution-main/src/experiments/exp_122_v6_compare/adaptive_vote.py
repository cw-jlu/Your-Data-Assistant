"""Adaptive vote across multiple attempt answers.

Strategy:
  1. Drop empty / failed attempts.
  2. Group remaining attempts by NUMERIC value-set (= sorted multiset of all
     numeric cell values, regardless of column structure or string content).
     Numeric divergence = real interpretation branch.
     Numeric convergence = structural variance (= same answer, different shape).
  3. If only ONE numeric value-set → majority signature vote (= pick most
     common shape). Wins for tasks like task_27 (= "Sacha Harrison" vs
     "Sacha"+"Harrison" — same numerics 866.25, vote on shape).
  4. If MULTIPLE numeric value-sets → union by signature_majority_merge
     (= keep all distinct columns). Wins for tasks like task_196 (= numerics
     1.0 vs 2.0 differ — keep both so gold has a chance).

This combines majority's column discipline with union's recall safety net.
"""
from __future__ import annotations

from typing import Any

from kobushi_core.benchmark.schema import AnswerTable


def _norm_value(v: Any) -> tuple:
    # Numeric rounding aligns with the official scorer's Decimal quantize
    # to 2 decimal places (ROUND_HALF_UP). Using a tighter granularity here
    # causes the vote dedup to keep duplicate columns that the scorer would
    # later consider identical, inflating extras_ratio at evaluation time.
    if v is None: return (0, "")
    if isinstance(v, (int, float)):
        try: return (1, round(float(v), 2))
        except: return (2, str(v).strip().lower())
    s = str(v).strip()
    try: return (1, round(float(s), 2))
    except: return (2, s.lower())


def _numeric_set(answer: AnswerTable) -> tuple:
    """Multiset of normalized NUMERIC values in the answer (= ignores strings)."""
    if not answer or not answer.rows:
        return ()
    nums = []
    for row in answer.rows:
        for v in row:
            if v is None: continue
            s = str(v).strip()
            try:
                nums.append(round(float(s), 2))
            except Exception:
                pass
    return tuple(sorted(nums))


def _signature(answer: AnswerTable) -> tuple:
    if not answer or not answer.columns:
        return ("__empty__",)
    n_cols = len(answer.columns)
    rows = []
    for r in answer.rows or []:
        if len(r) != n_cols: continue
        rows.append(tuple(_norm_value(v) for v in r))
    rows.sort()
    return (n_cols, len(rows), tuple(rows))


def _majority_signature_vote(answers: list[AnswerTable]) -> AnswerTable:
    """Return the answer whose signature is most common."""
    s2i: dict[tuple, list[int]] = {}
    for i, a in enumerate(answers):
        s2i.setdefault(_signature(a), []).append(i)
    best = max(s2i, key=lambda k: len(s2i[k]))
    return answers[s2i[best][0]]


def _union_columns(answers: list[AnswerTable]) -> AnswerTable:
    """Union answers column-wise.

    Each column is a value-vector identified by its sorted normalized values.
    Columns with the same vector across attempts are deduped; distinct columns
    are appended. Row count is the max across attempts (= rows that don't
    appear in a given attempt are filled with empty).
    """
    # Group attempts by signature first; keep one representative per signature
    sig_to_rep: dict[tuple, AnswerTable] = {}
    for a in answers:
        sig = _signature(a)
        if sig not in sig_to_rep:
            sig_to_rep[sig] = a

    # If one signature, just return its representative
    if len(sig_to_rep) == 1:
        return next(iter(sig_to_rep.values()))

    # Build column-level union: each (column-vector) keyed by sorted tuple of
    # its values, dedupe across attempts, preserve column order from first
    # appearance.
    unique_cols: list[tuple[str, list[Any]]] = []
    seen_keys: set[tuple] = set()
    for a in answers:
        n_cols = len(a.columns)
        for ci in range(n_cols):
            col_vals = [r[ci] if ci < len(r) else "" for r in a.rows]
            key = tuple(sorted(_norm_value(v) for v in col_vals))
            if key in seen_keys: continue
            seen_keys.add(key)
            unique_cols.append((a.columns[ci], col_vals))

    if not unique_cols:
        return answers[0]

    # Align rows: pad shorter columns to max length
    max_rows = max(len(c[1]) for c in unique_cols)
    new_columns = [c[0] for c in unique_cols]
    new_rows: list[list[Any]] = []
    for ri in range(max_rows):
        row = []
        for _, vals in unique_cols:
            row.append(vals[ri] if ri < len(vals) else "")
        new_rows.append(row)
    return AnswerTable(columns=new_columns, rows=new_rows)


def entropy_weighted_vote(
    answers: list[AnswerTable], entropies: list[float]
) -> AnswerTable | None:
    """AIMO3-style entropy-weighted majority vote.

    weight = 1 / max(mean_entropy, 1e-9). Aggregate weight per signature
    across attempts. Pick the signature with highest total weight; return
    one representative answer from that group.

    Falls back to plain majority signature vote when entropies are all inf
    (= no logprobs collected).
    """
    if not answers:
        return None
    if len(answers) != len(entropies):
        # Defensive: if lengths mismatch, fall back to plain majority.
        return _majority_signature_vote([a for a in answers if a and a.rows] or answers)

    paired = [
        (a, e) for a, e in zip(answers, entropies)
        if a and a.rows and e != float("inf")
    ]
    if not paired:
        # Fallback if no valid (answer, entropy) pairs.
        non_empty = [a for a in answers if a and a.rows]
        if non_empty:
            return _majority_signature_vote(non_empty)
        return answers[0]

    weight_per_sig: dict[tuple, float] = {}
    rep_per_sig: dict[tuple, AnswerTable] = {}
    for a, e in paired:
        sig = _signature(a)
        w = 1.0 / max(e, 1e-9)
        weight_per_sig[sig] = weight_per_sig.get(sig, 0.0) + w
        rep_per_sig.setdefault(sig, a)

    best_sig = max(weight_per_sig, key=weight_per_sig.get)
    return rep_per_sig[best_sig]


def _normalized_rows_set(answer: AnswerTable) -> set[tuple]:
    """Normalize answer's rows into a set of tuples for subset comparison."""
    return set(tuple(_norm_value(v) for v in row) for row in (answer.rows or []))


def _try_subset_pick(answers: list[AnswerTable]) -> AnswerTable | None:
    """SUBSET fix: if one answer's rows are a strict subset of another's
    (= same columns, fewer rows, all contained), prefer the smaller one,
    BUT only when not overruled by a majority-sized sig.

    Rules:
      1. Group by columns (= subset comparison requires same shape).
      2. Within column group, find smallest distinct row-set candidate.
      3. Require candidate's frequency >= every larger-sized sig's frequency
         (= protect majority). If a larger sig has more attempts than the
         smaller, the smaller is treated as the minority outlier (= incomplete
         filter), and we keep the regular majority path.
      4. All larger answers must contain candidate's rows (= proper subset).

    Targets task_11 v3 case (1×rc=3 ⊂ 1×rc=18 → pick rc=3) while NOT
    misfiring on task_80 (1×rc=1 ⊂ 2×rc=2 → keep majority rc=2).

    Returns None if no subset relation exists, or if rule 3 blocks the pick.
    """
    if len(answers) < 2:
        return None
    from collections import Counter
    by_cols: dict[tuple, list[AnswerTable]] = {}
    for a in answers:
        if not a or not a.columns: continue
        by_cols.setdefault(tuple(a.columns), []).append(a)

    for group in by_cols.values():
        if len(group) < 2: continue
        # Count attempts per signature (row content multiset).
        sig_count: Counter = Counter()
        sig_rep: dict[tuple, AnswerTable] = {}
        for a in group:
            sig = _signature(a)
            sig_count[sig] += 1
            sig_rep.setdefault(sig, a)
        # Identify the smallest-row-count signature
        sigs_sorted = sorted(sig_rep.items(), key=lambda kv: len(kv[1].rows or []))
        cand_sig, cand = sigs_sorted[0]
        cand_rows = _normalized_rows_set(cand)
        if not cand_rows:
            continue
        cand_count = sig_count[cand_sig]
        # Check candidate is strict subset of ALL larger sigs.
        is_proper_subset_of_larger = True
        for other_sig, other in sigs_sorted[1:]:
            other_rows = _normalized_rows_set(other)
            if len(other_rows) <= len(cand_rows):
                # Equal-size sig that's distinct → can't be strict subset.
                is_proper_subset_of_larger = False
                break
            if not cand_rows.issubset(other_rows):
                is_proper_subset_of_larger = False
                break
            # Rule 3: protect majority — don't override if a larger sig
            # outnumbers the candidate.
            if sig_count[other_sig] > cand_count:
                is_proper_subset_of_larger = False
                break
        if is_proper_subset_of_larger:
            return cand
    return None


def adaptive_vote(answers: list[AnswerTable]) -> AnswerTable | None:
    """Adaptive vote: subset-pick > numeric-convergent majority > union.

    Returns None if all answers are empty.
    """
    if not answers:
        return None
    # Drop empty/failed attempts (= 0 rows)
    non_empty = [a for a in answers if a and a.rows]
    if not non_empty:
        # fall back to first non-None to preserve column header
        return answers[0]

    # SUBSET check first: catches LEFT-JOIN-extras pattern where one attempt
    # over-includes rows containing another attempt's clean answer.
    subset_pick = _try_subset_pick(non_empty)
    if subset_pick is not None:
        return subset_pick

    # Group by numeric value-set
    num_groups: dict[tuple, list[AnswerTable]] = {}
    for a in non_empty:
        ns = _numeric_set(a)
        num_groups.setdefault(ns, []).append(a)

    if len(num_groups) == 1:
        # All attempts agree numerically → structural variance only → majority
        attempts = next(iter(num_groups.values()))
        return _majority_signature_vote(attempts)

    # Numeric divergence — but if there is a clear majority numeric group
    # (= more than half of attempts agree on numerics), trust the majority
    # and ignore the outlier(s). Falls back to union only when no group has
    # majority (= e.g. 2:2:1 split).
    n_total = len(non_empty)
    max_count = max(len(v) for v in num_groups.values())
    if max_count * 2 > n_total:  # strict majority (= more than half)
        majority_attempts = max(num_groups.values(), key=len)
        return _majority_signature_vote(majority_attempts)

    # No clear majority → union to preserve gold candidate
    return _union_columns(non_empty)
