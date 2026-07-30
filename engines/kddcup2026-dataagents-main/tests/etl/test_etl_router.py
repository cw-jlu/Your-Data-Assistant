from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from agents.etl._router import route_prose_files
from agents.llm.types import ModelMessage, ModelResponse


class _SchemaAwareRouterAdapter:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        del tools, kwargs
        content = messages[-1].content
        assert isinstance(content, str)
        self.prompts.append(content)
        if "mf_netvalueperformancehis" in content and "AnnualizedRRSinceStart" in content:
            return ModelResponse(content='["mf_fundarchives.md"]')
        return ModelResponse(content='["mf_fundreturnrank.md"]')


def test_router_prompt_includes_structured_source_schemas(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    doc_dir = context_dir / "doc"
    csv_dir = context_dir / "csv"
    db_dir = context_dir / "db"
    doc_dir.mkdir(parents=True)
    csv_dir.mkdir()
    db_dir.mkdir()

    archives = doc_dir / "mf_fundarchives.md"
    return_rank = doc_dir / "mf_fundreturnrank.md"
    archives.write_text("fund archive prose", encoding="utf-8")
    return_rank.write_text("return rank prose", encoding="utf-8")
    (csv_dir / "mf_other.csv").write_text("InnerCode,OtherMetric\n1,2\n", encoding="utf-8")

    db_path = db_dir / "sub_db.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE mf_netvalueperformancehis (
                InnerCode INTEGER,
                AnnualizedRRSinceStart REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO mf_netvalueperformancehis VALUES (?, ?)",
            (1, 12.5),
        )

    adapter = _SchemaAwareRouterAdapter()

    selected = route_prose_files(
        adapter,
        (
            "Group all funds by fund type and count funds meeting the annualized "
            "return threshold since inception."
        ),
        [archives, return_rank],
        "",
    )

    assert selected == [archives]
    assert len(adapter.prompts) == 1
    prompt = adapter.prompts[0]
    assert "Structured sources already available" in prompt
    assert "csv/mf_other.csv (CSV): columns: InnerCode, OtherMetric" in prompt
    assert "db/sub_db.sqlite (SQLite):" in prompt
    assert "table mf_netvalueperformancehis: columns: InnerCode, AnnualizedRRSinceStart" in prompt
    assert "master/archive/dimension prose documents" in prompt
