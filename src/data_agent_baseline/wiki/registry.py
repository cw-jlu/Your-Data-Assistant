"""
Wiki tool registry - tools exposed to the ReAct agent for wiki interaction.

Implements all three operations from Karpathy's LLM Wiki pattern:
  - Query: wiki_search, wiki_get_page, wiki_list_index
  - Lint: wiki_lint (find orphans, dead links, stale pages)
"""
from __future__ import annotations

from typing import Any

from data_agent_baseline.wiki.engine import WikiEngine, extract_summary
from data_agent_baseline.wiki.retrieval import WikiRetriever


# Shared retriever instance (persists IDF cache across tool calls)
_shared_retriever: WikiRetriever | None = None


def _get_retriever(engine: WikiEngine) -> WikiRetriever:
    """Return a shared WikiRetriever instance to preserve caches."""
    global _shared_retriever
    if _shared_retriever is None or _shared_retriever.engine is not engine:
        _shared_retriever = WikiRetriever(engine)
    return _shared_retriever


def wiki_search(engine: WikiEngine, action_input: dict[str, Any]) -> dict[str, Any]:
    """Search the wiki for pages relevant to a query."""
    query = str(action_input.get("query", ""))
    top_k = int(action_input.get("top_k", 3))
    if not query:
        return {"error": "query is required"}

    retriever = _get_retriever(engine)
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return {"results": [], "message": "No relevant wiki pages found."}

    output = []
    for r in results:
        page = r.page
        entry = {
            "title": page.title,
            "type": page.page_type,
            "score": r.score,
            "tags": page.tags,
            "summary": extract_summary(page.content, max_len=200),
        }
        if len(page.content) < 800:
            entry["content"] = page.content
        output.append(entry)

    return {"results": output}


def wiki_get_page(engine: WikiEngine, action_input: dict[str, Any]) -> dict[str, Any]:
    """Retrieve a specific wiki page by title."""
    title = str(action_input.get("title", ""))
    if not title:
        return {"error": "title is required"}

    page = engine.load_page_by_title(title)
    if not page:
        all_pages = engine.list_pages()
        matches = [p for p in all_pages if title.lower() in p.title.lower()]
        if matches:
            return {
                "error": f"Page '{title}' not found. Did you mean:",
                "suggestions": [m.title for m in matches[:5]],
            }
        return {"error": f"Page '{title}' not found."}

    return {
        "title": page.title,
        "type": page.page_type,
        "tags": page.tags,
        "content": page.content,
    }


def wiki_list_index(engine: WikiEngine, action_input: dict[str, Any]) -> dict[str, Any]:
    """List all pages in the wiki index."""
    page_type = action_input.get("type")
    pages = engine.list_pages(page_type=str(page_type) if page_type else None)

    entries = []
    for p in pages:
        entries.append({
            "title": p.title,
            "type": p.page_type,
            "path": p.path,
            "summary": p.summary[:100] if p.summary else "",
        })

    return {"pages": entries, "count": len(entries)}


def wiki_lint(engine: WikiEngine, action_input: dict[str, Any]) -> dict[str, Any]:
    """Health-check the wiki: orphans, dead links, empty pages, missing tags."""
    return engine.lint()


def create_wiki_tool_registry(
    engine: WikiEngine,
) -> dict[str, Any]:
    """Create wiki tool specs and handlers for integration with ToolRegistry."""
    specs = {
        "wiki_search": {
            "name": "wiki_search",
            "description": (
                "Search the knowledge wiki for pages relevant to a query. "
                "Use this to find prior knowledge about data sources, schemas, "
                "domain concepts, or analysis patterns from previously processed tasks."
            ),
            "input_schema": {
                "query": "search query text",
                "top_k": 3,
            },
        },
        "wiki_get_page": {
            "name": "wiki_get_page",
            "description": (
                "Retrieve a specific wiki page by its exact title. "
                "Use wiki_search first to find available page titles."
            ),
            "input_schema": {
                "title": "exact page title",
            },
        },
        "wiki_list_index": {
            "name": "wiki_list_index",
            "description": (
                "List all pages in the wiki index, optionally filtered by type. "
                "Types: entity, concept, source, synthesis, overview."
            ),
            "input_schema": {
                "type": "optional page type filter",
            },
        },
        "wiki_lint": {
            "name": "wiki_lint",
            "description": (
                "Health-check the wiki. Finds orphan pages (no inbound links), "
                "dead wikilinks (links to non-existent pages), empty pages, "
                "and pages missing tags."
            ),
            "input_schema": {},
        },
    }

    handlers = {
        "wiki_search": lambda _task, inp: wiki_search(engine, inp),
        "wiki_get_page": lambda _task, inp: wiki_get_page(engine, inp),
        "wiki_list_index": lambda _task, inp: wiki_list_index(engine, inp),
        "wiki_lint": lambda _task, inp: wiki_lint(engine, inp),
    }

    return {"specs": specs, "handlers": handlers}
