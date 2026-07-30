from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.database import AppDatabase
from app.engines import RUNTIME_ROOT, LaunchPlan, build_launch_plan


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class RunRecord:
    id: str
    engine_id: str
    engine_name: str
    mode: str
    task_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    output_dir: str = ""
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    log_sequence: int = 0
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    stop_requested: bool = False

    def public_dict(self, include_log: bool = False) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "mode": self.mode,
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "output_dir": self.output_dir,
            "error": self.error,
            "log_size": len(self.log_lines),
            "log_tail": self.log_lines[-12:],
        }
        if include_log:
            payload["log"] = self.log_lines
        return payload


class RunManager:
    def __init__(self, database: AppDatabase | None = None) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.RLock()
        self._database = database
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        if self._database is not None:
            self._database.recover_interrupted_runs(_now())
            for payload in self._database.load_runs():
                record = RunRecord(**payload)
                record.log_sequence = len(record.log_lines)
                self._runs[record.id] = record

    def _persist(self, record: RunRecord) -> None:
        if self._database is None:
            return
        self._database.save_run(
            {
                "id": record.id,
                "engine_id": record.engine_id,
                "engine_name": record.engine_name,
                "mode": record.mode,
                "task_id": record.task_id,
                "status": record.status,
                "created_at": record.created_at,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "exit_code": record.exit_code,
                "output_dir": record.output_dir,
                "error": record.error,
                "stop_requested": int(record.stop_requested),
            }
        )

    def start_many(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        engine_ids = payload.get("engine_ids")
        if not isinstance(engine_ids, list) or not engine_ids:
            raise ValueError("至少选择一个引擎")
        if len(engine_ids) > 4:
            raise ValueError("一次最多启动四个引擎")

        plans: list[tuple[str, LaunchPlan]] = []
        try:
            for engine_id in dict.fromkeys(str(item) for item in engine_ids):
                run_id = (
                    f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
                    f"{engine_id}-{uuid.uuid4().hex[:6]}"
                )
                plans.append((run_id, build_launch_plan(engine_id, payload, run_id)))
        except Exception:
            for _, plan in plans:
                if plan.config_path is not None:
                    plan.config_path.unlink(missing_ok=True)
            raise

        created: list[RunRecord] = []
        for run_id, plan in plans:
            record = RunRecord(
                id=run_id,
                engine_id=plan.engine.id,
                engine_name=plan.engine.name,
                mode=str(payload.get("mode", "single")),
                task_id=str(payload.get("task_id", "")),
                output_dir=str(plan.output_dir),
            )
            with self._lock:
                self._runs[run_id] = record
                self._persist(record)
            thread = threading.Thread(
                target=self._execute,
                args=(record, plan),
                name=f"run-{run_id}",
                daemon=True,
            )
            thread.start()
            created.append(record)
        return [record.public_dict() for record in created]

    def _append_log(self, record: RunRecord, line: str) -> None:
        with self._lock:
            clean_line = line.rstrip("\r\n")
            record.log_sequence += 1
            record.log_lines.append(clean_line)
            if self._database is not None:
                self._database.append_run_log(record.id, record.log_sequence, clean_line)
            if len(record.log_lines) > 8000:
                del record.log_lines[:1000]
                if self._database is not None:
                    self._database.trim_run_logs(record.id, record.log_sequence - 7999)

    def _execute(self, record: RunRecord, plan: LaunchPlan) -> None:
        try:
            with self._lock:
                record.status = "running"
                record.started_at = _now()
                self._persist(record)
            self._append_log(record, f"[client] 启动 {record.engine_name}")
            self._append_log(record, f"[client] 工作目录：{plan.cwd}")
            self._append_log(record, f"[client] 输出目录：{plan.output_dir}")

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                plan.command,
                cwd=plan.cwd,
                env=plan.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            with self._lock:
                record.process = process
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(record, line)
            exit_code = process.wait()
            with self._lock:
                record.exit_code = exit_code
                if record.stop_requested:
                    record.status = "stopped"
                else:
                    record.status = "succeeded" if exit_code == 0 else "failed"
                record.ended_at = _now()
                self._persist(record)
            self._append_log(record, f"[client] 进程结束，退出码 {exit_code}")
        except Exception as exc:  # noqa: BLE001 - background boundary
            with self._lock:
                record.status = "failed"
                record.error = str(exc)
                record.ended_at = _now()
                self._persist(record)
            self._append_log(record, "[client] 启动失败：" + str(exc))
            self._append_log(record, traceback.format_exc())
        finally:
            with self._lock:
                record.process = None
            if plan.config_path is not None:
                try:
                    plan.config_path.unlink(missing_ok=True)
                except OSError as exc:
                    self._append_log(record, f"[client] 临时配置清理失败：{exc}")

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(self._runs.values(), key=lambda item: item.created_at, reverse=True)
            return [record.public_dict() for record in records]

    def get(self, run_id: str, include_log: bool = False) -> dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return record.public_dict(include_log=include_log)

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            process = record.process
            if process is None or record.status != "running":
                return record.public_dict()
            record.stop_requested = True
            self._persist(record)
            self._append_log(record, "[client] 正在请求停止…")
            process.terminate()
            return record.public_dict()

    def preview(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            output_dir = Path(record.output_dir)
        files = sorted(output_dir.rglob("prediction.csv")) if output_dir.is_dir() else []
        previews: list[dict[str, Any]] = []
        for path in files[:40]:
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.reader(handle))
                previews.append(
                    {
                        "task_id": path.parent.name,
                        "path": str(path),
                        "columns": rows[0] if rows else [],
                        "rows": rows[1:21] if len(rows) > 1 else [],
                        "row_count_previewed": max(len(rows) - 1, 0),
                    }
                )
            except (OSError, UnicodeError, csv.Error) as exc:
                previews.append({"task_id": path.parent.name, "path": str(path), "error": str(exc)})
        return {"run_id": run_id, "count": len(files), "previews": previews}

    def trace(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            output_dir = Path(record.output_dir)
            process_log = list(record.log_lines)

        roots = [
            output_dir,
            RUNTIME_ROOT / "logs" / run_id,
        ]
        artifact_paths: list[Path] = []
        for root in roots:
            if root.is_dir():
                artifact_paths.extend(
                    path
                    for path in root.rglob("*.json")
                    if path.name in {"trace.json", "summary.json", "submission_manifest.json"}
                    or "trace" in path.name.lower()
                )

        artifacts: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for path in sorted(artifact_paths):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
                artifacts.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "kind": "json",
                        "content": content,
                    }
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                artifacts.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "kind": "error",
                        "content": str(exc),
                    }
                )

        trace_db = RUNTIME_ROOT / "traces" / f"{run_id}.db"
        if trace_db.is_file():
            artifacts.append(self._read_trace_database(trace_db))

        return {
            "run_id": run_id,
            "status": record.status,
            "process_log": process_log,
            "artifacts": artifacts,
        }

    def _read_trace_database(self, path: Path) -> dict[str, Any]:
        tables: dict[str, Any] = {}
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                names = [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                ]
                for name in names:
                    quoted = name.replace('"', '""')
                    rows = connection.execute(f'SELECT * FROM "{quoted}"').fetchall()
                    tables[name] = [self._json_safe(dict(row)) for row in rows]
            finally:
                connection.close()
            return {
                "name": path.name,
                "path": str(path),
                "kind": "sqlite",
                "content": tables,
            }
        except (OSError, sqlite3.Error) as exc:
            return {
                "name": path.name,
                "path": str(path),
                "kind": "error",
                "content": str(exc),
            }

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
