"""PLAN target-noun verifier.

Pipeline (called after PLAN → EXPLORE transition):
  1. Sub-agent extracts target nouns from the question
  2. Rule-based regex search of schema columns for literal matches
  3. Check if PLAN's `<table>.<column>` references include any literal match
  4. Return list of flagged nouns (= unused exact column matches)

POC validated on real PLAN outputs: TP=2/3, FP=0/5 (= literal-match-class
failures, perfect specificity).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kobushi_core.model import OpenAIModelAdapter, ModelMessage


_NOUN_EXTRACTOR_SYS = """You extract NOUNS from a question that the SQL answer must touch.
Focus on:
  - the SUBJECT (= what entity/attribute to return)
  - the OBJECT (= what entity/attribute to filter on)
  - any AGGREGATE term (= "total", "average", "count" - keep as a separate noun)

Rules:
  - Lowercase, singular (e.g., "expenses" → "expense").
  - Drop articles (a, the).
  - Keep multi-word concepts as a single string if they're a unit (e.g., "first name", "monthly consumption").
  - Output ONLY a JSON array of nouns, no other text.

Examples:
  Q "Which event has the lowest cost?" → ["event", "cost"]
  Q "Identify the type of expenses and their total value approved." → ["type", "expense", "total value", "approved"]
  Q "What is the average weight of female superheroes?" → ["average weight", "female", "superhero"]
"""

_TABLE_DOT_COL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


@dataclass
class PlanFlag:
    noun: str
    unused_matches: list[tuple[str, str]]  # [(table, col), ...]


@dataclass
class VerifyResult:
    nouns: list[str]
    flags: list[PlanFlag]

    @property
    def has_issue(self) -> bool:
        return bool(self.flags)


def extract_target_nouns(question: str, model: OpenAIModelAdapter) -> list[str]:
    try:
        response = model.complete(
            [
                ModelMessage(role="system", content=_NOUN_EXTRACTOR_SYS),
                ModelMessage(role="user", content=f"Q: {question}\n\nOutput the JSON array:"),
            ],
            enable_thinking=False,
            max_tokens=256,
        )
    except Exception:
        return []
    m = re.search(r"\[.*?\]", response, re.DOTALL)
    if not m:
        return []
    try:
        result = json.loads(m.group(0))
        if isinstance(result, list):
            return [str(x).lower().strip() for x in result if x]
    except Exception:
        pass
    return []


def find_literal_matches(noun: str, schema: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    """Return [(table, col, match_type)]; match_type ∈ {exact, plural_singular, contains}."""
    noun_lower = noun.lower().strip()
    if not noun_lower:
        return []
    words = noun_lower.split()
    candidates = [noun_lower] + (words if len(words) > 1 else [])
    matches: list[tuple[str, str, str]] = []
    for table, cols in schema.items():
        for col in cols:
            col_norm = col.lower().replace("_", "").replace(" ", "")
            for cand in candidates:
                cand_norm = cand.lower().replace("_", "").replace(" ", "")
                if not cand_norm:
                    continue
                if col_norm == cand_norm:
                    matches.append((table, col, "exact")); break
                if col_norm == cand_norm + "s" or col_norm + "s" == cand_norm:
                    matches.append((table, col, "plural_singular")); break
                if len(cand_norm) >= 4 and cand_norm in col_norm:
                    matches.append((table, col, "contains")); break
                if len(col_norm) >= 4 and col_norm in cand_norm:
                    matches.append((table, col, "contains")); break
    seen, out = set(), []
    for m in matches:
        if (m[0], m[1]) not in seen:
            seen.add((m[0], m[1])); out.append(m)
    return out


def extract_chosen_refs(plan_text: str, schema: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Prefer DECISIONS block (= machine-readable) when present; else scan full text."""
    valid = {(t.lower(), c.lower()) for t, cs in schema.items() for c in cs}
    # Look for DECISIONS: block; use only its content if found
    m = re.search(r"DECISIONS\s*:\s*(.+?)(?=\n\s*\n|\Z)", plan_text, re.DOTALL | re.IGNORECASE)
    scan_text = m.group(1) if m else plan_text
    return {
        (mm[0].lower(), mm[1].lower())
        for mm in _TABLE_DOT_COL_RE.findall(scan_text)
        if (mm[0].lower(), mm[1].lower()) in valid
    }


def verify_plan(
    question: str,
    plan_text: str,
    schema: dict[str, list[str]],
    model: OpenAIModelAdapter,
) -> VerifyResult:
    """Run the full verifier. Returns flags for unused EXACT/plural literal matches."""
    table_names = {t.lower() for t in schema}
    nouns = extract_target_nouns(question, model)
    chosen = extract_chosen_refs(plan_text, schema)
    flags: list[PlanFlag] = []
    for noun in nouns:
        noun_norm = noun.lower().replace(" ", "")
        # Skip noun that names a table
        if noun_norm in table_names or noun_norm.rstrip("s") in table_names:
            continue
        matches = find_literal_matches(noun, schema)
        strong = [m for m in matches if m[2] in ("exact", "plural_singular")]
        if not strong:
            continue
        unused = [(t, c) for (t, c, _) in strong if (t.lower(), c.lower()) not in chosen]
        used = [(t, c) for (t, c, _) in strong if (t.lower(), c.lower()) in chosen]
        if unused and not used:
            flags.append(PlanFlag(noun=noun, unused_matches=unused))
    return VerifyResult(nouns=nouns, flags=flags)
