from __future__ import annotations

import ctypes
import sys
import threading
import traceback

import webview

from app.paths import DATA_ROOT, ensure_data_directories
from app.server import create_server


def _show_startup_error(message: str) -> None:
    ensure_data_directories()
    error_path = DATA_ROOT / "startup-error.log"
    error_path.write_text(message, encoding="utf-8")
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
            0,
            f"Data Agent 启动失败。\n\n错误详情已写入：\n{error_path}",
            "Data Agent",
            0x10,
        )


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

    try:
        webview.create_window(
            "Data Agent",
            f"http://{host}:{port}",
            width=1440,
            height=900,
            min_size=(980, 680),
            background_color="#F7F7F5",
            text_select=True,
        )
        webview.start(
            gui="edgechromium",
            debug=False,
            private_mode=False,
            storage_path=str(DATA_ROOT / "webview"),
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)


def guarded_main() -> None:
    try:
        main()
    except Exception:  # noqa: BLE001 - desktop process boundary
        details = traceback.format_exc()
        _show_startup_error(details)
        raise
