from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from app.database import AppDatabase
from app.engines import RUNTIME_ROOT


WORKSPACES_ROOT = RUNTIME_ROOT / "workspaces"
WORKSPACE_ID_PATTERN = re.compile(r"^[a-z0-9-]{8,80}$")
MAX_FILE_BYTES = 1_500_000_000

ENGINE_EXTENSIONS: dict[str, frozenset[str]] = {
    "langgraph": frozenset(
        {
            ".csv",
            ".json",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".md",
            ".txt",
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".mp4",
            ".m4v",
            ".mov",
            ".mkv",
            ".webm",
            ".avi",
        }
    ),
    "kobushi": frozenset(
        {
            ".csv",
            ".tsv",
            ".json",
            ".jsonl",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".md",
            ".txt",
            ".pdf",
            ".html",
            ".htm",
            ".xml",
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
            ".avi",
        }
    ),
    "memory": frozenset(
        {
            ".csv",
            ".tsv",
            ".json",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".md",
            ".txt",
            ".pdf",
            ".xlsx",
            ".xlsm",
            ".parquet",
        }
    ),
    "mamba": frozenset(
        {
            ".csv",
            ".json",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".md",
            ".txt",
            ".pdf",
            ".mp4",
            ".m4v",
            ".mov",
            ".mkv",
            ".webm",
            ".avi",
        }
    ),
}

ENGINE_SUPPORT_NOTES: dict[str, tuple[str, ...]] = {
    "langgraph": (
        "图片仅支持 JPG/JPEG/PNG/WebP。",
        "视频依赖关键帧、ASR 和模型多模态能力。",
    ),
    "kobushi": (
        "音频 ASR 针对视频音轨，不支持独立 MP3/WAV。",
        "HTML/XML 通过文档读取器处理。",
    ),
    "memory": (
        "图片工具只读元数据，不计为内容理解支持。",
        "支持 XLSX/XLSM，不支持旧式 XLS。",
    ),
    "mamba": (
        "视频工具单文件硬限制为 100 MB。",
        "常规预览器原生支持 CSV/JSON/SQLite/PDF/Markdown/TXT。",
    ),
}

FILE_GROUPS: tuple[dict[str, object], ...] = (
    {
        "name": "表格与数据",
        "extensions": (
            ".csv",
            ".tsv",
            ".json",
            ".jsonl",
            ".xlsx",
            ".xlsm",
            ".parquet",
            ".db",
            ".sqlite",
            ".sqlite3",
        ),
    },
    {
        "name": "文档",
        "extensions": (".txt", ".md", ".pdf", ".html", ".htm", ".xml"),
    },
    {
        "name": "图片",
        "extensions": (".png", ".jpg", ".jpeg", ".webp"),
    },
    {
        "name": "视频",
        "extensions": (
            ".mp4",
            ".m4v",
            ".mov",
            ".mkv",
            ".webm",
            ".avi",
        ),
    },
)
ALLOWED_EXTENSIONS = frozenset().union(*ENGINE_EXTENSIONS.values())


def _now() -> str:
    return datetime.now(UTC).isoformat()


class WorkspaceManager:
    def __init__(self, database: AppDatabase | None = None) -> None:
        self._database = database
        WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)

    def capabilities(self) -> dict[str, object]:
        return {
            "groups": [
                {"name": group["name"], "extensions": list(group["extensions"])}
                for group in FILE_GROUPS
            ],
            "accept": ",".join(sorted(ALLOWED_EXTENSIONS)),
            "max_file_bytes": MAX_FILE_BYTES,
            "engine_support": {
                engine_id: {
                    "extensions": sorted(extensions),
                    "notes": list(ENGINE_SUPPORT_NOTES[engine_id]),
                }
                for engine_id, extensions in ENGINE_EXTENSIONS.items()
            },
        }

    def create(self) -> dict[str, object]:
        workspace_id = f"ws-{uuid.uuid4().hex[:16]}"
        root = self._root(workspace_id)
        self._context_dir(workspace_id).mkdir(parents=True, exist_ok=False)
        metadata = {
            "id": workspace_id,
            "created_at": _now(),
            "updated_at": _now(),
            "query": "",
        }
        self._write_metadata(root, metadata)
        return self.get(workspace_id)

    def _root(self, workspace_id: str) -> Path:
        if not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
            raise ValueError("非法工作区 ID")
        root = (WORKSPACES_ROOT / workspace_id).resolve()
        root.relative_to(WORKSPACES_ROOT.resolve())
        return root

    def _context_dir(self, workspace_id: str) -> Path:
        return self._root(workspace_id) / "input" / "task_1" / "context"

    def _metadata_path(self, root: Path) -> Path:
        return root / "workspace.json"

    def _write_metadata(self, root: Path, metadata: dict[str, object]) -> None:
        self._metadata_path(root).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_metadata(self, root: Path) -> dict[str, object]:
        try:
            payload = json.loads(self._metadata_path(root).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("工作区元数据损坏") from exc
        if not isinstance(payload, dict):
            raise ValueError("工作区元数据格式错误")
        return payload

    def _safe_filename(self, raw_name: str) -> str:
        name = Path(raw_name.replace("\\", "/")).name.strip()
        if not name or name in {".", ".."}:
            raise ValueError("文件名不能为空")
        if len(name) > 240:
            raise ValueError("文件名过长")
        extension = Path(name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"暂不支持 {extension or '无扩展名'} 文件")
        return name

    def add_file(
        self,
        workspace_id: str,
        raw_name: str,
        source: BinaryIO,
        content_length: int,
    ) -> dict[str, object]:
        if content_length < 0 or content_length > MAX_FILE_BYTES:
            raise ValueError("文件大小超出限制")
        root = self._root(workspace_id)
        if not root.is_dir():
            raise ValueError("工作区不存在")
        context_dir = self._context_dir(workspace_id)
        name = self._safe_filename(raw_name)
        destination = context_dir / name
        if destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            index = 2
            while destination.exists():
                destination = context_dir / f"{stem} ({index}){suffix}"
                index += 1

        remaining = content_length
        with destination.open("wb") as handle:
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    destination.unlink(missing_ok=True)
                    raise ValueError("上传内容提前结束")
                handle.write(chunk)
                remaining -= len(chunk)

        metadata = self._read_metadata(root)
        metadata["updated_at"] = _now()
        self._write_metadata(root, metadata)
        uploaded = {
            "name": destination.name,
            "size": destination.stat().st_size,
            "extension": destination.suffix.lower(),
        }
        if self._database is not None:
            self.get(workspace_id)
        return uploaded

    def remove_file(self, workspace_id: str, raw_name: str) -> dict[str, object]:
        context_dir = self._context_dir(workspace_id).resolve()
        name = Path(raw_name.replace("\\", "/")).name
        target = (context_dir / name).resolve()
        target.relative_to(context_dir)
        if not target.is_file():
            raise ValueError("文件不存在")
        target.unlink()
        return self.get(workspace_id)

    def get(self, workspace_id: str) -> dict[str, object]:
        root = self._root(workspace_id)
        if not root.is_dir():
            raise ValueError("工作区不存在")
        metadata = self._read_metadata(root)
        context_dir = self._context_dir(workspace_id)
        files = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "extension": path.suffix.lower(),
            }
            for path in sorted(context_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file()
        ]
        workspace = {
            **metadata,
            "files": files,
            "dataset_root": str(root / "input"),
        }
        if self._database is not None:
            self._database.save_workspace(workspace)
        return workspace

    def finalize(self, workspace_id: str, query: str) -> Path:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("请输入 Query")
        if len(clean_query) > 100_000:
            raise ValueError("Query 过长")
        root = self._root(workspace_id)
        context_dir = self._context_dir(workspace_id)
        if not context_dir.is_dir():
            raise ValueError("工作区不存在")
        if not any(path.is_file() for path in context_dir.iterdir()):
            raise ValueError("请至少上传一个文件")

        task_path = root / "input" / "task_1" / "task.json"
        task_path.write_text(
            json.dumps(
                {
                    "task_id": "task_1",
                    "difficulty": "unknown",
                    "question": clean_query,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata = self._read_metadata(root)
        metadata["query"] = clean_query
        metadata["updated_at"] = _now()
        self._write_metadata(root, metadata)
        if self._database is not None:
            self.get(workspace_id)
        return root / "input"

    def delete(self, workspace_id: str) -> None:
        root = self._root(workspace_id)
        if root.is_dir():
            shutil.rmtree(root)
        if self._database is not None:
            self._database.delete_workspace(workspace_id)
