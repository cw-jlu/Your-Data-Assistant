"""Schema-link sub-agent — extract column/value candidates from the question.

Pipeline:
  1. Sub-agent receives the question
  2. Outputs JSON: {column_candidates: [{phrase, candidates}], value_candidates: [...]}
  3. Match column_candidates against actual schema column names (= fuzzy)
  4. Match value_candidates against cell values (= via value_match.retrieve)
  5. Format as preamble block

The key value-add over the question-as-query VALUE HINTS:
  - Disambiguates phrases like "type of expenses" → multiple column candidates
  - Surfaces column names that don't appear in cell values (e.g. status, approved)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from kobushi_core.model import ModelMessage, OpenAIModelAdapter

# Tunables
MAX_COL_CANDIDATES_PER_PHRASE = 5
MAX_VALUE_CANDIDATES_PER_PHRASE = 3
MAX_HINT_ENTRIES = 25


_EXTRACTOR_SYS = """You are a schema-linking assistant for text-to-SQL.

Given a user question about a database, output a JSON object identifying:
  - column_candidates: phrases in the question that likely refer to a column.
    For each phrase, list MULTIPLE candidate column names (= snake_case or
    camelCase). Include alternatives for ambiguous terms.
  - value_candidates: phrases in the question that likely refer to a cell
    value (= proper nouns, quoted strings, specific entities).

Be EXHAUSTIVE for ambiguous terms. Example:
  Q: "Identify the type of expenses for the October Meeting event."
  → "type" could mean expense.type, event.type, or expense_description
  → List ALL THREE.

Output JSON EXACTLY in this format, no other text:
{
  "column_candidates": [
    {"phrase": "string from question", "candidates": ["col_a", "col_b"]}
  ],
  "value_candidates": [
    {"phrase": "string from question", "candidates": ["literal value"]}
  ]
}

Examples (= unrelated synthetic domains):

Q: "How many movies released in 1985 had a runtime over 120 minutes?"
A: {"column_candidates": [
     {"phrase": "movies", "candidates": ["movie_id", "title", "name"]},
     {"phrase": "released", "candidates": ["release_year", "release_date", "year"]},
     {"phrase": "runtime", "candidates": ["runtime", "duration", "length_minutes"]}
   ],
   "value_candidates": [
     {"phrase": "1985", "candidates": ["1985", "1985-01-01"]},
     {"phrase": "120 minutes", "candidates": ["120"]}
   ]}

Q: "Which species of fish has the most occurrences in the catch logbook?"
A: {"column_candidates": [
     {"phrase": "species", "candidates": ["species", "species_name", "fish_type", "kind"]},
     {"phrase": "occurrences", "candidates": ["count", "occurrences", "frequency"]},
     {"phrase": "catch", "candidates": ["catch_id", "log_id", "record_id"]}
   ],
   "value_candidates": []}

Q: "List the airports with average delays above 30 minutes in 2019."
A: {"column_candidates": [
     {"phrase": "airports", "candidates": ["airport_id", "iata_code", "airport_name", "code"]},
     {"phrase": "delays", "candidates": ["delay", "delay_minutes", "departure_delay", "arrival_delay"]}
   ],
   "value_candidates": [
     {"phrase": "30 minutes", "candidates": ["30"]},
     {"phrase": "2019", "candidates": ["2019"]}
   ]}

For AMBIGUOUS terms (e.g. "type", "size", "status"), list ALL plausible column
naming conventions across schemas (= type, *_type, category, kind, etc.).

Output ONLY the JSON. No preamble, no markdown fence, no explanation."""


@dataclass(frozen=True, slots=True)
class LinkCandidate:
    phrase: str
    candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SchemaLinkResult:
    column_candidates: tuple[LinkCandidate, ...]
    value_candidates: tuple[LinkCandidate, ...]


def _make_model() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        model=os.environ.get("AGENT_MODEL", "qwen3.5-35b-a3b"),
        api_base=os.environ.get("AGENT_API_BASE") or os.environ.get("MODEL_API_URL"),
        api_key=os.environ.get("AGENT_API_KEY") or os.environ.get("MODEL_API_KEY"),
        temperature=0.0,
        extra_headers={
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_link_candidates(question: str, model: OpenAIModelAdapter | None = None) -> SchemaLinkResult:
    """Call sub-agent to extract column/value candidates from the question."""
    if model is None:
        model = _make_model()
    try:
        reply = model.complete(
            [
                ModelMessage(role="system", content=_EXTRACTOR_SYS),
                ModelMessage(role="user", content=f"Q: {question}\nA:"),
            ],
            enable_thinking=False,
            max_tokens=1024,
        )
    except Exception:
        return SchemaLinkResult(column_candidates=(), value_candidates=())
    # Parse JSON (= tolerate leading/trailing noise)
    m = _JSON_RE.search(reply or "")
    if not m:
        return SchemaLinkResult(column_candidates=(), value_candidates=())
    try:
        data = json.loads(m.group(0))
    except Exception:
        return SchemaLinkResult(column_candidates=(), value_candidates=())
    cols = []
    for entry in data.get("column_candidates", []) or []:
        if not isinstance(entry, dict):
            continue
        phrase = str(entry.get("phrase", "")).strip()
        cands = entry.get("candidates", []) or []
        if not phrase or not isinstance(cands, list):
            continue
        clean = tuple(str(c).strip() for c in cands[:MAX_COL_CANDIDATES_PER_PHRASE] if c)
        if clean:
            cols.append(LinkCandidate(phrase=phrase, candidates=clean))
    vals = []
    for entry in data.get("value_candidates", []) or []:
        if not isinstance(entry, dict):
            continue
        phrase = str(entry.get("phrase", "")).strip()
        cands = entry.get("candidates", []) or []
        if not phrase or not isinstance(cands, list):
            continue
        clean = tuple(str(c).strip() for c in cands[:MAX_VALUE_CANDIDATES_PER_PHRASE] if c)
        if clean:
            vals.append(LinkCandidate(phrase=phrase, candidates=clean))
    return SchemaLinkResult(column_candidates=tuple(cols), value_candidates=tuple(vals))


def _match_columns_to_schema(
    candidates: list[str], schema_cols: list[tuple[str, str]], question: str
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
    """Match candidate column names against actual (view, column) pairs.

    Returns:
      - hits: list of (view, col, candidate, match_kind) where match_kind is
        "LITERAL" (= question word == column name, case-insensitive) or
        "SUBSTRING" / "CANDIDATE" (= weaker match).
      - literal_hits_question: pairs of (question_word, view.col) where the
        question contains the column name verbatim, regardless of candidates.
    """
    hits: list[tuple[str, str, str, str]] = []
    cand_lc = [c.lower() for c in candidates]
    # Pre-tokenize question to test literal column-name appearance
    q_tokens = set()
    for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]+", question):
        q_tokens.add(tok.lower())
    literal_hits_q: list[tuple[str, str]] = []
    for view, col in schema_cols:
        col_lc = col.lower()
        # LITERAL: question token == column name
        if col_lc in q_tokens:
            literal_hits_q.append((col_lc, f"{view}.{col}"))
        matched = False
        for c, c_lc in zip(candidates, cand_lc):
            if c_lc == col_lc:
                hits.append((view, col, c, "LITERAL"))
                matched = True
                break
        if not matched:
            for c, c_lc in zip(candidates, cand_lc):
                if c_lc in col_lc or col_lc in c_lc:
                    hits.append((view, col, c, "SUBSTRING"))
                    break
    return hits, literal_hits_q


def build_schema_link_block(
    question: str,
    context_dir: Path,
    model: OpenAIModelAdapter | None = None,
) -> str | None:
    """One-shot: extract candidates, match against schema + values, format."""
    extracted = extract_link_candidates(question, model)
    if not extracted.column_candidates and not extracted.value_candidates:
        return None
    # Gather actual schema columns
    try:
        from experiments.exp_140_agentar_icl.tools.duckdb_unified import get_catalog
        catalog = get_catalog(context_dir)
        schema_cols = [(e["view"], c) for e in catalog if e.get("view")
                       for c in (e.get("columns") or [])]
    except Exception:
        schema_cols = []
    # Match value candidates via existing dense+BM25 retriever
    try:
        from experiments.exp_140_agentar_icl.value_match import retrieve
    except Exception:
        retrieve = None

    lines: list[str] = [
        "# SCHEMA LINK HINTS (= extracted column/value candidates by sub-agent)",
        (
            "A schema-linking sub-agent identified the following phrases in the "
            "question as potentially mapping to specific columns or cell values. "
            "Use these as guidance for your filter and SELECT choices.\n"
        ),
    ]
    n_hint = 0
    # First: scan the WHOLE question for literal column-name appearances.
    # This catches cases the LLM extractor missed (= "type" verbatim in
    # question → `event.type`). BIRD-style benchmarks strongly prefer the
    # literal column name over semantic alternatives.
    all_literal: list[tuple[str, str]] = []
    for view, col in schema_cols:
        col_lc = col.lower()
        # Use word-boundary match on question
        if re.search(r"\b" + re.escape(col_lc) + r"\b", question.lower()):
            all_literal.append((col_lc, f"`{view}`.`{col}`"))
    if all_literal:
        lines.append("## ⚠ LITERAL COLUMN MATCH (= question contains column name verbatim)")
        lines.append(
            "These columns appear LITERALLY in the question. BIRD-style benchmarks "
            "STRONGLY prefer the literal column over semantically-related alternatives. "
            "Default to selecting these columns unless EXPLORE proves them irrelevant."
        )
        seen_kw = set()
        for kw, ref in all_literal[:8]:
            if kw in seen_kw:
                continue
            seen_kw.add(kw)
            lines.append(f"  - question word `{kw}` → use column {ref}")
            n_hint += 1
    if extracted.column_candidates:
        lines.append("\n## Column candidates (= phrases → likely columns in your schema)")
        for cc in extracted.column_candidates:
            matched, _ = _match_columns_to_schema(list(cc.candidates), schema_cols, question)
            if matched:
                # Sort LITERAL hits first
                matched.sort(key=lambda x: 0 if x[3] == "LITERAL" else 1)
                hits_str = ", ".join(
                    f"`{v}`.`{c}`" + (" [LITERAL]" if k == "LITERAL" else "")
                    for v, c, _, k in matched[:6]
                )
                lines.append(f"  - phrase `{cc.phrase}` → matched: {hits_str}")
                n_hint += 1
            else:
                cand_str = ", ".join(f"`{c}`" for c in cc.candidates[:5])
                lines.append(f"  - phrase `{cc.phrase}` → guessed: {cand_str} (no exact match in schema)")
                n_hint += 1
            if n_hint >= MAX_HINT_ENTRIES:
                break
    if extracted.value_candidates and n_hint < MAX_HINT_ENTRIES:
        lines.append("\n## Value candidates (= phrases → likely cell values)")
        for vc in extracted.value_candidates:
            if retrieve:
                # Use first candidate as query (= usually the most specific)
                for cand in vc.candidates[:2]:
                    hits = retrieve(cand, context_dir, top_k_out=3) or []
                    if hits:
                        hits_str = "; ".join(
                            f"`{h.view}`.`{h.column}` = '{h.value}'" for h in hits[:3]
                        )
                        lines.append(f"  - phrase `{vc.phrase}` → cell hits: {hits_str}")
                        n_hint += 1
                        break
                else:
                    lines.append(f"  - phrase `{vc.phrase}` → guessed values: {vc.candidates[:3]}")
                    n_hint += 1
            else:
                lines.append(f"  - phrase `{vc.phrase}` → guessed values: {vc.candidates[:3]}")
                n_hint += 1
            if n_hint >= MAX_HINT_ENTRIES:
                break
    return "\n".join(lines)
