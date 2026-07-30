from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from app.database import AppDatabase
from app.engines import ENGINE_SPECS, list_tasks
from app.manager import RunManager
from app.paths import DATABASE_PATH, STATIC_ROOT, ensure_data_directories
from app.workspaces import WorkspaceManager


ensure_data_directories()
DATABASE = AppDatabase(DATABASE_PATH)
MANAGER = RunManager(DATABASE)
WORKSPACES = WorkspaceManager(DATABASE)


class ClientHandler(BaseHTTPRequestHandler):
    server_version = "UnifiedDataAgentClient/0.1"

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("请求体为空或过大")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self._json({"ok": True, "database": str(DATABASE_PATH)})
                return
            if path == "/api/engines":
                self._json({"engines": [engine.public_dict() for engine in ENGINE_SPECS]})
                return
            if path == "/api/upload-capabilities":
                self._json(WORKSPACES.capabilities())
                return
            if path.startswith("/api/workspaces/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3:
                    self._json(WORKSPACES.get(unquote(parts[2])))
                    return
            if path == "/api/tasks":
                query = parse_qs(parsed.query)
                root = query.get("dataset_root", [""])[0]
                tasks = list_tasks(root)
                self._json({"root": str(Path(root).expanduser().resolve()), "count": len(tasks), "tasks": tasks})
                return
            if path == "/api/runs":
                self._json({"runs": MANAGER.list_runs()})
                return
            if path.startswith("/api/runs/"):
                parts = path.strip("/").split("/")
                run_id = unquote(parts[2]) if len(parts) >= 3 else ""
                if len(parts) == 4 and parts[3] == "preview":
                    self._json(MANAGER.preview(run_id))
                elif len(parts) == 4 and parts[3] == "trace":
                    self._json(MANAGER.trace(run_id))
                else:
                    self._json(MANAGER.get(run_id, include_log=True))
                return
            self._serve_static(path)
        except KeyError:
            self._error("找不到运行记录", HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self._error(str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                payload = self._read_json()
                workspace_id = str(payload.get("workspace_id", "")).strip()
                if workspace_id:
                    dataset_root = WORKSPACES.finalize(
                        workspace_id,
                        str(payload.get("query", "")),
                    )
                    payload["dataset_root"] = str(dataset_root)
                    payload["mode"] = "batch"
                    payload["task_id"] = ""
                self._json({"runs": MANAGER.start_many(payload)}, HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/workspaces":
                self._json(WORKSPACES.create(), HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/workspaces/") and parsed.path.endswith("/files"):
                parts = parsed.path.strip("/").split("/")
                workspace_id = unquote(parts[2])
                query = parse_qs(parsed.query)
                filename = query.get("name", [""])[0]
                content_length = int(self.headers.get("Content-Length", "-1"))
                self._json(
                    WORKSPACES.add_file(
                        workspace_id,
                        filename,
                        self.rfile,
                        content_length,
                    ),
                    HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/stop"):
                parts = parsed.path.strip("/").split("/")
                self._json(MANAGER.stop(unquote(parts[2])))
                return
            self._error("未知 API", HTTPStatus.NOT_FOUND)
        except KeyError:
            self._error("找不到运行记录", HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self._error(str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/workspaces/") and parsed.path.endswith("/files"):
                parts = parsed.path.strip("/").split("/")
                query = parse_qs(parsed.query)
                filename = query.get("name", [""])[0]
                self._json(WORKSPACES.remove_file(unquote(parts[2]), filename))
                return
            self._error("未知 API", HTTPStatus.NOT_FOUND)
        except (ValueError, OSError) as exc:
            self._error(str(exc))

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        candidate = (STATIC_ROOT / relative).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error("非法路径", HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            candidate = STATIC_ROOT / "index.html"
        body = candidate.read_bytes()
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if media_type.startswith("text/") or media_type in {"application/javascript", "application/json"}:
            media_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[http] {self.address_string()} - {format % args}")


def create_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), ClientHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="KDD unified data-agent client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}"
    print(f"Unified client running at {url}")
    if args.open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
