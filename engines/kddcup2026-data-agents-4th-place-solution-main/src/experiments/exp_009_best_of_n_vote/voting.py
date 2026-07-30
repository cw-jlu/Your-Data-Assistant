"""Best-of-N selection by self-similarity over column signatures.

Generates N candidate answer tables, normalizes each column with the same
rules the official scorer uses, computes pairwise multiset-overlap of column
signatures across candidates, and picks the candidate whose answer agrees
most with the others. This uses the official metric itself as the verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from kobushi_core.benchmark.schema import AnswerTable
from kobushi_core.eval.csv_compare import _column_signature, _normalize_cell


@dataclass(frozen=True, slots=True)
class CandidateScore:
    index: int
    has_answer: bool
    score: int  # sum of multiset overlap with all other candidates' signature lists
    signatures: tuple[tuple[str, ...], ...]


def _column_signatures(answer: AnswerTable | None) -> tuple[tuple[str, ...], ...]:
    if answer is None:
        return tuple()
    column_count = len(answer.columns)
    sigs: list[tuple[str, ...]] = []
    for column_index in range(column_count):
        normalized: list[str] = []
        for row in answer.rows:
            cell = row[column_index] if column_index < len(row) else ""
            normalized.append(_normalize_cell(str(cell)))
        sigs.append(_column_signature(normalized))
    return tuple(sigs)


def _multiset_overlap(
    a: Sequence[tuple[str, ...]], b: Sequence[tuple[str, ...]]
) -> int:
    """Number of signatures from `a` that match a unique signature in `b`."""
    if not a or not b:
        return 0
    consumed = [False] * len(b)
    matched = 0
    for sig in a:
        for idx, other in enumerate(b):
            if not consumed[idx] and sig == other:
                consumed[idx] = True
                matched += 1
                break
    return matched


def score_candidates(answers: list[AnswerTable | None]) -> list[CandidateScore]:
    sigs_list = [_column_signatures(answer) for answer in answers]
    scores: list[CandidateScore] = []
    for i, sig_i in enumerate(sigs_list):
        score_i = 0
        for j, sig_j in enumerate(sigs_list):
            if i == j or not sig_j:
                continue
            score_i += _multiset_overlap(sig_i, sig_j)
        scores.append(
            CandidateScore(
                index=i,
                has_answer=answers[i] is not None,
                score=score_i,
                signatures=sig_i,
            )
        )
    return scores


def select_best_index(answers: list[AnswerTable | None]) -> int:
    """Pick the candidate index with highest pairwise signature overlap.

    Tie-breakers (in order):
      1. has_answer (skip ones with no answer)
      2. higher overlap score
      3. more columns (favor higher recall)
      4. lowest index (deterministic)
    """
    if not answers:
        raise ValueError("no candidates to score")
    scores = score_candidates(answers)
    return max(
        range(len(answers)),
        key=lambda i: (
            scores[i].has_answer,
            scores[i].score,
            len(scores[i].signatures),
            -i,
        ),
    )
