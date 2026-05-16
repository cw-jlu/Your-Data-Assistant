"""
LLM Wiki core engine - manages a persistent, interlinked markdown wiki.

Implements Karpathy's LLM Wiki pattern:
- Wiki pages are markdown files with YAML frontmatter
- Pages are linked via [[wikilinks]] (Obsidian-compatible)
- An index.md catalogs all pages by category
- A log.md records all operations chronologically
- Knowledge compounds: each new source enriches existing pages
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Page schema
# ---------------------------------------------------------------------------

@dataclass
class WikiPage:
    title: str
    page_type: str  # entity, concept, source, synthesis, overview
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    task_ids: list[str] = field(default_factory=list)
    difficulty: str = ""
    confidence: str = "EXTRACTED"  # EXTRACTED, INFERRED, AMBIGUOUS, UNVERIFIED
    content: str = ""

    def to_markdown(self) -> str:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        created = self.created or now
        updated = now

        front = {
            "title": self.title,
            "type": self.page_type,
            "tags": self.tags,
            "sources": self.sources,
            "created": created,
            "updated": updated,
        }
        if self.task_ids:
            front["task_ids"] = self.task_ids
        if self.difficulty:
            front["difficulty"] = self.difficulty
        if self.confidence != "EXTRACTED":
            front["confidence"] = self.confidence

        lines = ["---"]
        for k, v in front.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: {v}")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)

    @staticmethod
    def from_markdown(text: str) -> WikiPage:
        text = text.strip()
        frontmatter: dict[str, Any] = {}
        content = text

        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1].strip()
                content = parts[2].strip()
                frontmatter = _parse_frontmatter(fm_text)

        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        sources = frontmatter.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        task_ids = frontmatter.get("task_ids", [])
        if isinstance(task_ids, str):
            task_ids = [task_ids]

        return WikiPage(
            title=frontmatter.get("title", "Untitled"),
            page_type=frontmatter.get("type", "concept"),
            tags=tags,
            sources=sources,
            created=frontmatter.get("created", ""),
            updated=frontmatter.get("updated", ""),
            task_ids=task_ids,
            difficulty=frontmatter.get("difficulty", ""),
            confidence=frontmatter.get("confidence", "EXTRACTED"),
            content=content,
        )


def _parse_frontmatter(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current_key is not None:
            if current_list is None:
                current_list = []
            current_list.append(stripped[2:].strip().strip('"').strip("'"))
            result[current_key] = current_list
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            current_key = key
            if val:
                current_list = None
                result[key] = val.strip('"').strip("'")
            else:
                current_list = None
                result[key] = []

    return result


# ---------------------------------------------------------------------------
# Index entry
# ---------------------------------------------------------------------------

@dataclass
class IndexEntry:
    title: str
    page_type: str
    path: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WikiEngine
# ---------------------------------------------------------------------------

WIKI_SUBDIRS = ("entities", "concepts", "sources", "syntheses", "overview")


def extract_summary(content: str, max_len: int = 150) -> str:
    """Extract the first meaningful line from markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
            return stripped[:max_len]
    return ""


class WikiEngine:
    """
    Manages the wiki directory structure and page lifecycle.

    Performance features:
    - Page cache: title -> (path, WikiPage), invalidated on write/delete
    - Batch operations: save_pages_batch with single index rebuild
    - Content-hash based change detection
    """

    def __init__(self, wiki_root: Path) -> None:
        self.root = wiki_root
        # Page cache: title -> (file_path, WikiPage)
        self._page_cache: dict[str, tuple[Path, WikiPage]] = {}
        self._cache_dirty = True
        self._ensure_structure()

    # -- Initialization -------------------------------------------------------

    def _ensure_structure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in WIKI_SUBDIRS:
            (self.root / subdir).mkdir(exist_ok=True)
        if not (self.root / "index.md").exists():
            self._write_index([])
        if not (self.root / "log.md").exists():
            (self.root / "log.md").write_text(
                "# Wiki Operation Log\n\n", encoding="utf-8"
            )

    # -- Page cache -----------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cache_dirty = True

    def _ensure_cache_loaded(self) -> None:
        """Load all pages into cache if dirty."""
        if not self._cache_dirty:
            return
        self._page_cache.clear()
        for md_file in self.root.rglob("*.md"):
            if md_file.name in ("index.md", "log.md"):
                continue
            page = self._read_page_file(md_file)
            if page:
                self._page_cache[page.title] = (md_file, page)
        self._cache_dirty = False

    @staticmethod
    def _read_page_file(path: Path) -> WikiPage | None:
        """Read and parse a single page file."""
        try:
            return WikiPage.from_markdown(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # -- CRUD operations ------------------------------------------------------

    def save_page(self, page: WikiPage, *, rebuild_index: bool = True) -> Path:
        subdir = self._subdir_for_type(page.page_type)
        safe_title = self._slugify(page.title)
        page_path = self.root / subdir / f"{safe_title}.md"
        page_path.write_text(page.to_markdown(), encoding="utf-8")
        # Update cache immediately
        self._page_cache[page.title] = (page_path, page)
        if rebuild_index:
            self._rebuild_index()
        self._append_log(f"save | {page.page_type} | {page.title}")
        return page_path

    def save_pages_batch(self, pages: list[WikiPage]) -> list[Path]:
        """Save multiple pages with a single index rebuild at the end."""
        paths: list[Path] = []
        for page in pages:
            paths.append(self.save_page(page, rebuild_index=False))
        self._rebuild_index()
        return paths

    def upsert_entity_page(
        self,
        title: str,
        entity_type: str,
        schema_content: str,
        task_id: str,
        question: str,
        tags: list[str] | None = None,
        *,
        rebuild_index: bool = True,
    ) -> WikiPage:
        """Create or merge an entity page. Deduplicates by title."""
        self._ensure_cache_loaded()
        existing_entry = self._page_cache.get(title)

        if existing_entry is not None:
            _, existing = existing_entry
            task_ref = f"[[Task {task_id}: {question[:40]}]]"
            if task_ref not in existing.content:
                existing.content += f"\n{task_ref}"
            if task_id not in existing.task_ids:
                existing.task_ids.append(task_id)
            existing.tags = list(set(existing.tags + (tags or [])))
            return self.save_page(existing, rebuild_index=rebuild_index)

        content = f"## {title}\n\n"
        content += f"**Type:** {entity_type}\n\n"
        content += schema_content
        content += f"\n## Used In\n\n[[Task {task_id}: {question[:40]}]]\n"

        page = WikiPage(
            title=title,
            page_type="entity",
            tags=tags or [entity_type],
            task_ids=[task_id],
            content=content,
        )
        return self.save_page(page, rebuild_index=rebuild_index)

    def load_page(self, page_path: Path) -> WikiPage | None:
        return self._read_page_file(page_path)

    def load_page_by_title(self, title: str) -> WikiPage | None:
        """O(1) cached lookup by title."""
        self._ensure_cache_loaded()
        entry = self._page_cache.get(title)
        return entry[1] if entry else None

    def list_pages(self, page_type: str | None = None) -> list[IndexEntry]:
        """List pages from cache (no disk reads if cache is warm)."""
        self._ensure_cache_loaded()
        entries: list[IndexEntry] = []
        for title, (path, page) in sorted(self._page_cache.items()):
            if page_type and page.page_type != page_type:
                continue
            rel_path = path.relative_to(self.root).as_posix()
            summary = self._extract_summary(page.content)
            entries.append(IndexEntry(
                title=page.title,
                page_type=page.page_type,
                path=rel_path,
                summary=summary,
                tags=page.tags,
            ))
        return entries

    def find_pages_by_tags(self, tags: list[str]) -> list[WikiPage]:
        self._ensure_cache_loaded()
        tag_set = set(tags)
        return [
            page for _, page in self._page_cache.values()
            if tag_set.intersection(page.tags)
        ]

    def find_pages_by_task(self, task_id: str) -> list[WikiPage]:
        self._ensure_cache_loaded()
        return [
            page for _, page in self._page_cache.values()
            if task_id in page.task_ids
        ]

    def get_all_page_texts(self) -> list[tuple[str, str, str]]:
        """Return (title, path, full_text) from cache."""
        self._ensure_cache_loaded()
        results: list[tuple[str, str, str]] = []
        for title, (path, page) in self._page_cache.items():
            rel_path = path.relative_to(self.root).as_posix()
            full_text = f"{page.title}\n{page.content}"
            results.append((page.title, rel_path, full_text))
        return results

    def delete_page(self, title: str) -> bool:
        self._ensure_cache_loaded()
        entry = self._page_cache.get(title)
        if entry is None:
            return False
        path, page = entry
        path.unlink()
        del self._page_cache[title]
        self._rebuild_index()
        self._append_log(f"delete | {page.page_type} | {title}")
        return True

    # -- Wikilink resolution --------------------------------------------------

    def resolve_wikilinks(self, content: str) -> list[str]:
        return re.findall(r"\[\[([^\]]+)\]\]", content)

    def has_page(self, title: str) -> bool:
        self._ensure_cache_loaded()
        return title in self._page_cache

    # -- Content hash ---------------------------------------------------------

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_content_hash(self) -> str:
        """Hash of all page content for cache invalidation detection."""
        self._ensure_cache_loaded()
        combined = "".join(
            f"{title}:{page.content}"
            for title, (_, page) in sorted(self._page_cache.items())
        )
        return self.content_hash(combined)

    # -- Internal helpers -----------------------------------------------------

    def _subdir_for_type(self, page_type: str) -> str:
        mapping = {
            "entity": "entities",
            "concept": "concepts",
            "source": "sources",
            "synthesis": "syntheses",
            "overview": "overview",
        }
        return mapping.get(page_type, "concepts")

    def _slugify(self, text: str) -> str:
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug[:80] or "untitled"

    def _extract_summary(self, content: str, max_len: int = 150) -> str:
        return extract_summary(content, max_len)

    def lint(self) -> dict[str, Any]:
        """Public API for wiki health check: orphans, dead links, empty pages, missing tags."""
        self._ensure_cache_loaded()
        page_titles = set(self._page_cache.keys())

        all_links: set[str] = set()
        page_link_map: dict[str, list[str]] = {}
        empty_pages: list[str] = []
        no_tag_pages: list[str] = []

        for title, (_, page) in self._page_cache.items():
            links = self.resolve_wikilinks(page.content)
            page_link_map[title] = links
            all_links.update(links)

            if not page.content.strip():
                empty_pages.append(title)
            if not page.tags:
                no_tag_pages.append(title)

        dead_links = sorted(all_links - page_titles)

        linked_titles: set[str] = set()
        for links in page_link_map.values():
            linked_titles.update(links)
        orphans = sorted(page_titles - linked_titles)

        issues = []
        if dead_links:
            issues.append({"type": "dead_links", "count": len(dead_links), "items": dead_links[:10]})
        if orphans:
            issues.append({"type": "orphans", "count": len(orphans), "items": orphans[:10]})
        if empty_pages:
            issues.append({"type": "empty_pages", "count": len(empty_pages), "items": empty_pages})
        if no_tag_pages:
            issues.append({"type": "no_tags", "count": len(no_tag_pages), "items": no_tag_pages})

        return {
            "total_pages": len(page_titles),
            "total_links": len(all_links),
            "issues": issues,
            "healthy": len(issues) == 0,
        }

    def _write_index(self, entries: list[IndexEntry]) -> None:
        lines = [
            "# Wiki Index",
            "",
            f"*Last updated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}*",
            "",
        ]
        by_type: dict[str, list[IndexEntry]] = {}
        for entry in entries:
            by_type.setdefault(entry.page_type, []).append(entry)

        type_order = ["overview", "entity", "concept", "source", "synthesis"]
        type_labels = {
            "overview": "Overview",
            "entity": "Entities (Data Sources & Tables)",
            "concept": "Concepts (Domain Knowledge & Patterns)",
            "source": "Sources (Task Summaries)",
            "synthesis": "Syntheses (Cross-Task Analysis)",
        }
        for ptype in type_order:
            group = by_type.get(ptype, [])
            if not group:
                continue
            label = type_labels.get(ptype, ptype.title())
            lines.append(f"## {label}")
            lines.append("")
            for entry in sorted(group, key=lambda e: e.title):
                tag_str = ""
                if entry.tags:
                    tag_str = f" `{'`, `'.join(entry.tags)}`"
                summary_str = f" — {entry.summary}" if entry.summary else ""
                lines.append(f"- [[{entry.title}]]{tag_str}{summary_str}")
            lines.append("")

        for ptype, group in sorted(by_type.items()):
            if ptype in type_order:
                continue
            lines.append(f"## {ptype.title()}")
            lines.append("")
            for entry in sorted(group, key=lambda e: e.title):
                lines.append(f"- [[{entry.title}]]")
            lines.append("")

        (self.root / "index.md").write_text("\n".join(lines), encoding="utf-8")

    def _rebuild_index(self) -> None:
        entries = self.list_pages()
        self._write_index(entries)

    def _append_log(self, entry: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log_path = self.root / "log.md"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{ts}] {entry}\n")
