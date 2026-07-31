from __future__ import annotations

import ctypes
import json
import sys
import threading
import traceback
import urllib.request

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow

from app.paths import DATA_ROOT, ensure_data_directories
from app.server import create_server


class DataAgentWindow(QMainWindow):
    def __init__(self, url: str) -> None:
        super().__init__()
        self.setWindowTitle("Data Agent")
        self.resize(1440, 900)
        self.setMinimumSize(980, 680)

        profile = QWebEngineProfile.defaultProfile()
        profile.setPersistentStoragePath(str(DATA_ROOT / "webview" / "storage"))
        profile.setCachePath(str(DATA_ROOT / "webview" / "cache"))

        view = QWebEngineView(self)
        page = QWebEnginePage(profile, view)
        view.setPage(page)
        view.setUrl(QUrl(url))
        self.setCentralWidget(view)
        self._view = view


def _show_startup_error(message: str, show_dialog: bool = True) -> None:
    ensure_data_directories()
    error_path = DATA_ROOT / "startup-error.log"
    error_path.write_text(message, encoding="utf-8")
    if show_dialog and sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
            0,
            f"Data Agent 启动失败。\n\n错误详情已写入：\n{error_path}",
            "Data Agent",
            0x10,
        )


def self_test() -> None:
    """Exercise bundled imports, SQLite, static files, uv, and the local API."""
    ensure_data_directories()
    server = create_server("127.0.0.1", 0)
    host, port = server.server_address[:2]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=10) as response:
            health = json.load(response)
        with urllib.request.urlopen(f"http://{host}:{port}/api/engines", timeout=10) as response:
            engines = json.load(response)["engines"]
        if not health.get("ok"):
            raise RuntimeError("Local API health check failed")
        if len(engines) != 4 or not all(engine.get("ready") for engine in engines):
            raise RuntimeError("Expected all four bundled engines to be ready")
        result = {
            "ok": True,
            "database": health["database"],
            "engines": [engine["id"] for engine in engines],
        }
        (DATA_ROOT / "self-test.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)


def main() -> None:
    ensure_data_directories()
    server = create_server("127.0.0.1", 0)
    host, port = server.server_address[:2]
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="data-agent-local-server",
        daemon=True,
    )
    server_thread.start()

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    application = QApplication(sys.argv)
    application.setApplicationName("Data Agent")
    application.setOrganizationName("Your Data Assistant")
    window = DataAgentWindow(f"http://{host}:{port}")
    window.show()

    try:
        exit_code = application.exec()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)
    if exit_code:
        raise RuntimeError(f"Qt event loop exited with code {exit_code}")


def guarded_main() -> None:
    try:
        if "--self-test" in sys.argv:
            self_test()
        else:
            main()
    except Exception:  # noqa: BLE001 - desktop process boundary
        details = traceback.format_exc()
        _show_startup_error(details, show_dialog="--self-test" not in sys.argv)
        raise
