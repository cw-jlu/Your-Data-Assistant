"""
LLM Wiki module - a persistent, compounding knowledge base for data agents.

Based on Andrej Karpathy's LLM Wiki pattern (2026-04):
Instead of re-deriving knowledge from scratch on every task, the agent
incrementally builds and maintains a structured wiki of interlinked
markdown pages that grows richer with each task processed.

Three-layer architecture:
  1. Raw sources (task context) - immutable input data
  2. Wiki (markdown pages) - LLM-maintained, interlinked knowledge
  3. Schema (index + log) - navigation and audit trail
"""

from data_agent_baseline.wiki.engine import WikiEngine, WikiPage
from data_agent_baseline.wiki.retrieval import WikiRetriever
from data_agent_baseline.wiki.ingest import WikiIngestor
from data_agent_baseline.wiki.registry import create_wiki_tool_registry

__all__ = [
    "WikiEngine",
    "WikiPage",
    "WikiRetriever",
    "WikiIngestor",
    "create_wiki_tool_registry",
]
