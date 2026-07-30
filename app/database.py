from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


class AppDatabase:
    """SQLite persistence for task state, trace logs, and upload metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    engine_id TEXT NOT NULL,
                    engine_name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    task_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    exit_code INTEGER,
                    output_dir TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    stop_requested INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS run_logs (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    line TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    query TEXT NOT NULL DEFAULT '',
                    dataset_root TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workspace_files (
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    extension TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, name)
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_run_logs_run_id ON run_logs(run_id, sequence);
                """
            )
            current = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if current is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(current["version"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported database schema {current['version']}; expected {SCHEMA_VERSION}"
                )

    def recover_interrupted_runs(self, ended_at: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'interrupted',
                    ended_at = COALESCE(ended_at, ?),
                    error = COALESCE(error, '应用关闭时任务仍在运行')
                WHERE status IN ('queued', 'running')
                """,
                (ended_at,),
            )

    def save_run(self, run: Mapping[str, Any]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, engine_id, engine_name, mode, task_id, status, created_at,
                    started_at, ended_at, exit_code, output_dir, error, stop_requested
                ) VALUES (
                    :id, :engine_id, :engine_name, :mode, :task_id, :status, :created_at,
                    :started_at, :ended_at, :exit_code, :output_dir, :error, :stop_requested
                )
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    exit_code = excluded.exit_code,
                    output_dir = excluded.output_dir,
                    error = excluded.error,
                    stop_requested = excluded.stop_requested
                """,
                dict(run),
            )

    def append_run_log(self, run_id: str, sequence: int, line: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO run_logs(run_id, sequence, line) VALUES (?, ?, ?)",
                (run_id, sequence, line),
            )

    def trim_run_logs(self, run_id: str, keep_from: int) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM run_logs WHERE run_id = ? AND sequence < ?",
                (run_id, keep_from),
            )

    def load_runs(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                payload = dict(row)
                logs = connection.execute(
                    "SELECT line FROM run_logs WHERE run_id = ? ORDER BY sequence",
                    (payload["id"],),
                ).fetchall()
                payload["log_lines"] = [item["line"] for item in logs]
                payload["stop_requested"] = bool(payload["stop_requested"])
                result.append(payload)
            return result

    def save_workspace(self, workspace: Mapping[str, Any]) -> None:
        files = list(workspace.get("files", []))
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO workspaces(id, created_at, updated_at, query, dataset_root)
                VALUES (:id, :created_at, :updated_at, :query, :dataset_root)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    query = excluded.query,
                    dataset_root = excluded.dataset_root
                """,
                dict(workspace),
            )
            connection.execute(
                "DELETE FROM workspace_files WHERE workspace_id = ?",
                (workspace["id"],),
            )
            connection.executemany(
                """
                INSERT INTO workspace_files(workspace_id, name, size, extension)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        workspace["id"],
                        item["name"],
                        int(item["size"]),
                        item["extension"],
                    )
                    for item in files
                ],
            )

    def delete_workspace(self, workspace_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
