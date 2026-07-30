from __future__ import annotations

from pathlib import Path, PurePosixPath

from data_agent_baseline.benchmark.schema import ContextAsset, PublicTask


def normalize_context_relative_path(relative_path: str) -> str:
    normalized = relative_path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("context/"):
        normalized = normalized[len("context/") :]
    if normalized == "context":
        return ""
    return normalized.strip("/")


def reject_unsafe_context_path(normalized_path: str, original_path: str) -> None:
    if not normalized_path:
        return
    parts = PurePosixPath(normalized_path).parts
    if normalized_path.startswith("/") or ".." in parts:
        raise ValueError(f"Path escapes context dir: {original_path}")


def iter_context_file_assets(task: PublicTask) -> list[ContextAsset]:
    context_view = task.assets.context_view
    if context_view is not None:
        return sorted(context_view.assets, key=lambda asset: asset.visible_path)

    assets: list[ContextAsset] = []
    for path in sorted(task.context_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(task.context_dir).as_posix()
        assets.append(
            ContextAsset(
                visible_path=rel_path,
                physical_path=path,
                source_path=rel_path,
                action="source",
                generated=False,
            )
        )
    return assets


def resolve_context_asset(task: PublicTask, relative_path: str) -> ContextAsset:
    normalized_path = normalize_context_relative_path(relative_path)
    reject_unsafe_context_path(normalized_path, relative_path)

    context_view = task.assets.context_view
    if context_view is None:
        path = resolve_context_path(task, normalized_path)
        rel_path = path.relative_to(task.context_dir).as_posix()
        return ContextAsset(
            visible_path=rel_path,
            physical_path=path,
            source_path=rel_path,
            action="source",
            generated=False,
        )

    for asset in context_view.assets:
        if asset.visible_path == normalized_path:
            if not asset.physical_path.exists():
                raise FileNotFoundError(
                    f"Missing context asset: {normalized_path or relative_path}. "
                    "All tool paths must be relative to the context directory."
                )
            return asset
    raise FileNotFoundError(
        f"Missing context asset: {normalized_path or relative_path}. "
        "All tool paths must be relative to the context directory."
    )


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    normalized_path = normalize_context_relative_path(relative_path)
    context_view = task.assets.context_view
    if context_view is not None:
        return resolve_context_asset(task, normalized_path).physical_path

    candidate = (task.context_dir / normalized_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing context asset: {normalized_path or relative_path}. "
            "All tool paths must be relative to the context directory."
        )
    return candidate
