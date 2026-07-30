from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_NAME = "DataAgent"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
FROZEN = bool(getattr(sys, "frozen", False))


def _bundle_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled).resolve() if bundled else SOURCE_ROOT


def _install_root() -> Path:
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return SOURCE_ROOT


def _data_root() -> Path:
    override = os.environ.get("DATA_AGENT_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if FROZEN:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return (base / APP_NAME).resolve()
    return SOURCE_ROOT / ".runtime"


BUNDLE_ROOT = _bundle_root()
INSTALL_ROOT = _install_root()
DATA_ROOT = _data_root()
RUNTIME_ROOT = DATA_ROOT
STATIC_ROOT = BUNDLE_ROOT / "static"
DATABASE_PATH = DATA_ROOT / "data-agent.db"

_engines_override = os.environ.get("DATA_AGENT_ENGINES_ROOT", "").strip()
if _engines_override:
    ENGINES_ROOT = Path(_engines_override).expanduser().resolve()
elif FROZEN and (INSTALL_ROOT / "engines").is_dir():
    ENGINES_ROOT = INSTALL_ROOT / "engines"
else:
    ENGINES_ROOT = BUNDLE_ROOT / "engines"


def uv_executable() -> str | None:
    override = os.environ.get("DATA_AGENT_UV", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return str(candidate.resolve()) if candidate.is_file() else None
    bundled = INSTALL_ROOT / ("uv.exe" if os.name == "nt" else "uv")
    if FROZEN and bundled.is_file():
        return str(bundled)
    return shutil.which("uv")


def ensure_data_directories() -> None:
    for path in (
        DATA_ROOT,
        RUNTIME_ROOT / "configs",
        RUNTIME_ROOT / "logs",
        RUNTIME_ROOT / "outputs",
        RUNTIME_ROOT / "traces",
        RUNTIME_ROOT / "workspaces",
    ):
        path.mkdir(parents=True, exist_ok=True)
