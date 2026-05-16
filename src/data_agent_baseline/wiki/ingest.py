"""
Wiki ingest - extracts knowledge from completed tasks and writes wiki pages.

Implements the "Ingest" operation from Karpathy's LLM Wiki pattern:
after a task is solved, extract structured knowledge (schemas, patterns,
domain concepts) and integrate them into the wiki with cross-references.

This is the "compilation" step: raw task data is compiled into persistent,
interlinked wiki pages that compound over time.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.wiki.engine import WikiEngine, WikiPage


@dataclass
class TaskKnowledge:
    """Extracted knowledge from a single task run."""
    task_id: str
    difficulty: str
    question: str
    data_sources: list[dict[str, Any]] = field(default_factory=list)
    sql_queries: list[str] = field(default_factory=list)
    python_patterns: list[str] = field(default_factory=list)
    column_schemas: list[dict[str, Any]] = field(default_factory=list)
    domain_entities: list[str] = field(default_factory=list)
    answer_columns: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    error_patterns: list[str] = field(default_factory=list)


class WikiIngestor:
    """
    Extracts knowledge from completed tasks and writes wiki pages.

    Two-phase ingest:
      1. Extract: analyze task context + agent trace to identify knowledge
      2. Write: create/update wiki pages with cross-references
    """

    def __init__(self, engine: WikiEngine) -> None:
        self.engine = engine

    def ingest_task(
        self,
        task: PublicTask,
        run_result: dict[str, Any] | None = None,
    ) -> list[WikiPage]:
        """
        Ingest a completed task into the wiki.

        Phase 1: Extract knowledge from task context + agent trace
        Phase 2: Write wiki pages with deduplication and cross-references
        All writes are batched with a single index rebuild at the end.
        """
        knowledge = self._extract_knowledge(task, run_result)
        pages: list[WikiPage] = []

        # 1. Source summary page (per-task, always new)
        source_page = self._create_source_page(knowledge)
        pages.append(source_page)

        # 2. Entity pages (upsert: merge if exists, create if new)
        for entity in knowledge.data_sources:
            name = entity.get("name", "")
            if not name:
                continue
            schema_content = self._build_entity_schema_content(entity)
            entity_type = entity.get("type", "unknown")
            tags = [entity_type, knowledge.difficulty]
            self.engine.upsert_entity_page(
                title=name,
                entity_type=entity_type,
                schema_content=schema_content,
                task_id=knowledge.task_id,
                question=knowledge.question,
                tags=tags,
                rebuild_index=False,
            )

        # 3. Concept pages (SQL patterns, Python patterns)
        for concept_page in self._create_concept_pages(knowledge):
            pages.append(concept_page)

        # 4. Schema pages (upsert)
        for schema in knowledge.column_schemas:
            source = schema.get("source", "")
            columns = schema.get("columns", [])
            source_type = schema.get("source_type", "")
            if not source or not columns:
                continue
            schema_content = self._build_schema_table_content(columns)
            self.engine.upsert_entity_page(
                title=f"Schema: {source}",
                entity_type=source_type,
                schema_content=schema_content,
                task_id=knowledge.task_id,
                question=knowledge.question,
                tags=["schema", source_type, knowledge.difficulty],
                rebuild_index=False,
            )

        # Batch save all remaining pages + single index rebuild
        self.engine.save_pages_batch(pages)

        # 5. Generate cross-task synthesis if enough tasks have been processed
        self._maybe_create_synthesis(knowledge)

        return pages

    # -- Knowledge extraction -------------------------------------------------

    def _extract_knowledge(
        self, task: PublicTask, run_result: dict[str, Any] | None
    ) -> TaskKnowledge:
        """Extract structured knowledge from a task and its run result."""
        knowledge = TaskKnowledge(
            task_id=task.task_id,
            difficulty=task.difficulty,
            question=task.question,
        )

        context_dir = task.context_dir
        if not context_dir.exists():
            return knowledge

        # Extract from CSV files
        for csv_file in context_dir.rglob("*.csv"):
            self._extract_csv_knowledge(csv_file, knowledge, context_dir)

        # Extract from SQLite databases
        for db_file in list(context_dir.rglob("*.sqlite")) + list(context_dir.rglob("*.db")):
            self._extract_sqlite_knowledge(db_file, knowledge, context_dir)

        # Extract from JSON files
        for json_file in context_dir.rglob("*.json"):
            if json_file.name != "task.json":
                self._extract_json_knowledge(json_file, knowledge, context_dir)

        # Extract from doc files
        for doc_file in list(context_dir.rglob("*.md")) + list(context_dir.rglob("*.txt")):
            self._extract_doc_knowledge(doc_file, knowledge, context_dir)

        # Extract from knowledge.md
        knowledge_md = context_dir / "knowledge.md"
        if knowledge_md.exists():
            self._extract_knowledge_md(knowledge_md, knowledge)

        # Extract from agent trace if available
        if run_result:
            self._extract_trace_knowledge(run_result, knowledge)

        # Extract domain entities from question
        knowledge.domain_entities = self._extract_entities_from_text(task.question)

        return knowledge

    def _extract_csv_knowledge(
        self, csv_path: Path, knowledge: TaskKnowledge, context_dir: Path
    ) -> None:
        """Extract schema and sample data from a CSV file."""
        rel_path = csv_path.relative_to(context_dir).as_posix()
        try:
            with csv_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if not headers:
                    return
                sample_rows = []
                for i, row in enumerate(reader):
                    if i >= 3:
                        break
                    sample_rows.append(row)

            knowledge.data_sources.append({
                "type": "csv",
                "path": rel_path,
                "name": csv_path.stem,
                "columns": headers,
                "sample_rows": sample_rows,
            })

            knowledge.column_schemas.append({
                "source": rel_path,
                "columns": headers,
                "source_type": "csv",
            })
        except Exception:
            pass

    def _extract_sqlite_knowledge(
        self, db_path: Path, knowledge: TaskKnowledge, context_dir: Path
    ) -> None:
        """Extract table schemas from a SQLite database."""
        rel_path = db_path.relative_to(context_dir).as_posix()
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()
            table_info = []
            for table_name, create_sql in tables:
                cursor.execute(f"PRAGMA table_info([{table_name}])")
                cols = cursor.fetchall()
                col_names = [c[1] for c in cols]
                table_info.append({
                    "table": table_name,
                    "columns": col_names,
                    "create_sql": create_sql,
                })
                knowledge.column_schemas.append({
                    "source": f"{rel_path}::{table_name}",
                    "columns": col_names,
                    "source_type": "sqlite",
                })

            knowledge.data_sources.append({
                "type": "sqlite",
                "path": rel_path,
                "name": db_path.stem,
                "tables": table_info,
            })
        except Exception:
            pass
        finally:
            if conn is not None:
                conn.close()

    def _extract_json_knowledge(
        self, json_path: Path, knowledge: TaskKnowledge, context_dir: Path
    ) -> None:
        """Extract structure from a JSON file."""
        rel_path = json_path.relative_to(context_dir).as_posix()
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            keys = list(data.keys()) if isinstance(data, dict) else []
            knowledge.data_sources.append({
                "type": "json",
                "path": rel_path,
                "name": json_path.stem,
                "top_keys": keys[:20],
            })
        except Exception:
            pass

    def _extract_doc_knowledge(
        self, doc_path: Path, knowledge: TaskKnowledge, context_dir: Path
    ) -> None:
        """Note the existence of documentation files."""
        rel_path = doc_path.relative_to(context_dir).as_posix()
        try:
            text = doc_path.read_text(encoding="utf-8")[:500]
            knowledge.data_sources.append({
                "type": "doc",
                "path": rel_path,
                "name": doc_path.stem,
                "preview": text[:200],
            })
        except Exception:
            pass

    def _extract_knowledge_md(self, path: Path, knowledge: TaskKnowledge) -> None:
        """Extract domain knowledge from knowledge.md."""
        try:
            text = path.read_text(encoding="utf-8")
            knowledge.key_findings.append(f"Background knowledge: {text[:300]}")
        except Exception:
            pass

    def _extract_trace_knowledge(
        self, run_result: dict[str, Any], knowledge: TaskKnowledge
    ) -> None:
        """Extract patterns from the agent's execution trace."""
        steps = run_result.get("steps", [])
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action", "")
            action_input = step.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            if action == "execute_context_sql":
                sql = action_input.get("sql", "")
                if sql:
                    knowledge.sql_queries.append(sql)

            elif action == "execute_python":
                code = action_input.get("code", "")
                if code:
                    knowledge.python_patterns.append(code[:300])

            elif action == "answer":
                columns = action_input.get("columns", [])
                if columns:
                    knowledge.answer_columns = columns

        # Check for errors
        failure = run_result.get("failure_reason")
        if failure:
            knowledge.error_patterns.append(failure[:200])

    def _extract_entities_from_text(self, text: str) -> list[str]:
        """Extract potential entity names from text using simple heuristics."""
        entities: list[str] = []
        # Capitalized words (potential proper nouns)
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for cap in caps:
            if cap not in {"The", "This", "That", "What", "Which", "How", "List", "Find",
                           "Show", "Give", "Calculate", "Determine", "Compare"}:
                entities.append(cap)
        return list(set(entities))[:10]

    # -- Wiki page creation ---------------------------------------------------

    def _create_source_page(self, knowledge: TaskKnowledge) -> WikiPage:
        """Create a source summary page for the task."""
        content_parts = [
            f"## Task: {knowledge.question}",
            "",
            f"**Difficulty:** {knowledge.difficulty}",
            f"**Task ID:** {knowledge.task_id}",
            "",
        ]

        if knowledge.data_sources:
            content_parts.append("## Data Sources")
            content_parts.append("")
            for src in knowledge.data_sources:
                src_type = src.get("type", "unknown")
                src_name = src.get("name", "unknown")
                src_path = src.get("path", "")
                content_parts.append(f"- **{src_name}** ({src_type}): `{src_path}`")
                if src_type == "csv" and "columns" in src:
                    cols = ", ".join(src["columns"][:10])
                    content_parts.append(f"  Columns: {cols}")
                elif src_type == "sqlite" and "tables" in src:
                    for tbl in src["tables"][:5]:
                        cols = ", ".join(tbl["columns"][:10])
                        content_parts.append(f"  Table `{tbl['table']}`: {cols}")
            content_parts.append("")

        if knowledge.sql_queries:
            content_parts.append("## SQL Queries Used")
            content_parts.append("")
            for i, sql in enumerate(knowledge.sql_queries[:5], 1):
                content_parts.append(f"```sql\n{sql}\n```")
                content_parts.append("")

        if knowledge.answer_columns:
            content_parts.append("## Answer Structure")
            content_parts.append("")
            content_parts.append(f"Columns: {', '.join(knowledge.answer_columns)}")
            content_parts.append("")

        # Add wikilinks to related entities
        links = []
        for src in knowledge.data_sources:
            name = src.get("name", "")
            if name:
                links.append(f"[[{name}]]")
        if links:
            content_parts.append("## Related")
            content_parts.append("")
            content_parts.append(", ".join(links))
            content_parts.append("")

        tags = [knowledge.difficulty] + [s.get("type", "") for s in knowledge.data_sources]
        tags = [t for t in tags if t]

        return WikiPage(
            title=f"Task {knowledge.task_id}: {knowledge.question[:60]}",
            page_type="source",
            tags=tags,
            sources=[knowledge.task_id],
            task_ids=[knowledge.task_id],
            difficulty=knowledge.difficulty,
            content="\n".join(content_parts),
        )

    @staticmethod
    def _build_entity_schema_content(entity: dict[str, Any]) -> str:
        """Build schema content section for an entity."""
        parts = [f"**Path:** `{entity.get('path', '')}`", ""]
        entity_type = entity.get("type", "unknown")

        if entity_type == "csv" and "columns" in entity:
            parts.append("## Schema")
            parts.append("")
            parts.append("| Column |")
            parts.append("|--------|")
            for col in entity["columns"]:
                parts.append(f"| {col} |")
            parts.append("")

        elif entity_type == "sqlite" and "tables" in entity:
            parts.append("## Tables")
            parts.append("")
            for tbl in entity["tables"]:
                parts.append(f"### {tbl['table']}")
                cols = ", ".join(tbl["columns"])
                parts.append(f"Columns: {cols}")
                parts.append("")

        elif entity_type == "json" and "top_keys" in entity:
            parts.append("## Structure")
            parts.append("")
            parts.append(f"Top keys: {', '.join(entity['top_keys'][:15])}")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _build_schema_table_content(columns: list[str]) -> str:
        """Build a markdown table for column listing."""
        parts = ["| # | Column Name |", "|---|-------------|"]
        for i, col in enumerate(columns, 1):
            parts.append(f"| {i} | {col} |")
        return "\n".join(parts)

    def _maybe_create_synthesis(self, current: TaskKnowledge) -> None:
        """
        If enough tasks have been ingested, create a cross-task synthesis page
        that summarizes common patterns across all processed tasks.
        """
        all_source_pages = self.engine.list_pages(page_type="source")
        if len(all_source_pages) < 3:
            return  # Not enough data for synthesis yet

        # Collect all task IDs and difficulties
        difficulties: dict[str, int] = {}
        all_tags: Counter[str] = Counter()
        for entry in all_source_pages:
            for tag in entry.tags:
                if tag in ("easy", "medium", "hard", "extreme"):
                    difficulties[tag] = difficulties.get(tag, 0) + 1
                else:
                    all_tags[tag] += 1

        # Build synthesis content
        content = "## Cross-Task Knowledge Synthesis\n\n"
        content += f"**Total tasks processed:** {len(all_source_pages)}\n\n"

        if difficulties:
            content += "### Difficulty Distribution\n\n"
            for diff, count in sorted(difficulties.items()):
                content += f"- {diff}: {count} tasks\n"
            content += "\n"

        # Common data source types
        if all_tags:
            content += "### Common Data Patterns\n\n"
            for tag, count in all_tags.most_common(10):
                content += f"- `{tag}`: appears in {count} tasks\n"
            content += "\n"

        # Links to all source pages
        content += "### All Processed Tasks\n\n"
        for entry in all_source_pages[:20]:
            content += f"- [[{entry.title}]]\n"
        if len(all_source_pages) > 20:
            content += f"- ... and {len(all_source_pages) - 20} more\n"

        synthesis = WikiPage(
            title="Knowledge Synthesis",
            page_type="synthesis",
            tags=["synthesis", "overview"],
            content=content,
        )
        self.engine.save_page(synthesis, rebuild_index=True)

    def _create_concept_pages(self, knowledge: TaskKnowledge) -> list[WikiPage]:
        """Create concept pages for significant patterns found."""
        pages: list[WikiPage] = []

        # Create a concept page for interesting SQL patterns
        if knowledge.sql_queries:
            sql_content = "## SQL Patterns\n\n"
            sql_content += "SQL queries used across tasks:\n\n"
            for i, sql in enumerate(knowledge.sql_queries[:5], 1):
                sql_content += f"### Pattern {i}\n```sql\n{sql}\n```\n\n"

            sql_content += f"## Related Tasks\n\n[[Task {knowledge.task_id}: {knowledge.question[:40]}]]\n"

            pages.append(WikiPage(
                title=f"SQL Patterns ({knowledge.task_id})",
                page_type="concept",
                tags=["sql", "query-pattern", knowledge.difficulty],
                task_ids=[knowledge.task_id],
                content=sql_content,
            ))

        # Create concept page for data analysis patterns
        if knowledge.python_patterns:
            py_content = "## Python Data Analysis Patterns\n\n"
            py_content += "Python code patterns used for data processing:\n\n"
            for i, code in enumerate(knowledge.python_patterns[:3], 1):
                py_content += f"### Pattern {i}\n```python\n{code}\n```\n\n"

            py_content += f"## Related Tasks\n\n[[Task {knowledge.task_id}: {knowledge.question[:40]}]]\n"

            pages.append(WikiPage(
                title=f"Python Patterns ({knowledge.task_id})",
                page_type="concept",
                tags=["python", "data-analysis", knowledge.difficulty],
                task_ids=[knowledge.task_id],
                content=py_content,
            ))

        return pages

