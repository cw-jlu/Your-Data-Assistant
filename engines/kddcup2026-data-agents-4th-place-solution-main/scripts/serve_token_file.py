#!/usr/bin/env python3
"""Serve one static HTML file behind a token path.

Intended for Cloudflare Quick Tunnel demos where the tunnel hostname is public
but the app should not be available at the root path.
"""
from __future__ import annotations

import argparse
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def _norm_token(token: str) -> str:
    return token.strip().strip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--token", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8891, type=int)
    args = parser.parse_args()

    html_path = args.file.resolve()
    if not html_path.is_file():
        raise SystemExit(f"file not found: {html_path}")
    token = _norm_token(args.token)
    body = html_path.read_bytes()
    mime = mimetypes.guess_type(str(html_path))[0] or "text/html"

    class Handler(BaseHTTPRequestHandler):
        server_version = "KobushiTokenFile/1.0"

        def log_message(self, fmt: str, *fmt_args) -> None:  # noqa: N802
            print(f"{self.address_string()} - {fmt % fmt_args}", flush=True)

        def _send_headers(self, status: int, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            allowed = {f"/{token}", f"/{token}/", f"/{token}/{html_path.name}"}
            if path not in allowed:
                self._send_headers(404)
                self.wfile.write(b"not found\n")
                return
            self._send_headers(200, mime)
            self.wfile.write(body)

        def do_HEAD(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            allowed = {f"/{token}", f"/{token}/", f"/{token}/{html_path.name}"}
            self._send_headers(200 if path in allowed else 404, mime)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"serving {html_path} at http://{args.host}:{args.port}/{token}/",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
