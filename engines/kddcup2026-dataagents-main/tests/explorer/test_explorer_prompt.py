from __future__ import annotations

from agents.explorer.prompt import EXPLORER_SYSTEM_PROMPT


def test_explorer_prompt_requires_knowledge_md_mapping_evidence() -> None:
    prompt = EXPLORER_SYSTEM_PROMPT
    lower = prompt.lower()

    assert "if `inspect_files` lists `knowledge.md`, you must read it before `report`" in lower
    assert "a knowledge.md mapping/disambiguation/example" in lower
    assert "column mapping" in lower
    assert "disambiguation rule" in lower
    assert "example/use-case sql" in lower
    assert "knowledge_md_evidence" in prompt
    assert "trading volume" in prompt
    assert "turnoverdeals" in prompt


def test_explorer_prompt_allows_local_sqlite_indexes_for_discovery() -> None:
    prompt = EXPLORER_SYSTEM_PROMPT
    lower = prompt.lower()

    assert "discovery-only" in lower
    assert "virtual-context database copy" in lower
    assert "create index if not exists" in lower
    assert "you are read-only" not in lower


def test_explorer_prompt_mentions_sqlite_preview_row_counts() -> None:
    prompt = EXPLORER_SYSTEM_PROMPT

    assert "returns CREATE TABLE plus row_count" in prompt


def test_explorer_prompt_reports_on_demand_etl_sources() -> None:
    prompt = EXPLORER_SYSTEM_PROMPT
    lower = prompt.lower()

    assert "`etl_sources`" in prompt
    assert "run_etl" in prompt
    assert "do not include `knowledge.md`" in lower
    assert "grouped record-id entity paragraph samples" in prompt
    assert "For PDFs, `inspect_files` already" in prompt
    assert "same-stem document" in lower
    assert "not treat a similarly named structured table" in lower
    assert "reporting `etl_sources: []` in this situation is invalid" in lower
    assert "name similarity is irrelevant" in lower
    assert "mf_fmscaleanalysisn" in prompt
    assert "mf_fmretscaleanalysis" in prompt
