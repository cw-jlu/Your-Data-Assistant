"""跨进程文件令牌桶。

被 `rate_limit.RateLimitedAdapter` 使用，独立于"装饰器/重试"逻辑：

- **状态**：每个 Key 一份 JSON 状态文件 + 一份 `filelock` 锁，路径由调用方指定
  （通常是 `RateLimitConfig.state_dir/bucket_<sha1(key)[:10]>.{json,lock}`）。
- **刷新**：按壁钟线性 `tokens = min(cap, tokens + elapsed * cap / 60)`，确保多
  子进程 / 多线程共享配额而不超限。
- **耐用性**：状态文件不存在 / 损坏时按满桶重建；写盘走临时文件 + `os.replace`
  原子替换，并发读者只会看到旧值或新值。

**部署约束**：状态文件和锁文件必须落在本地文件系统（ext4/tmpfs），NFS 下 `flock`
语义未定义。
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock


@dataclass(slots=True)
class _BucketState:
    """桶的内存表示；序列化为 state 文件里的 JSON。

    - rpm / tpm: 当前剩余令牌数（float 便于线性刷新）
    - last: 上次刷新的壁钟（time.time()）；跨进程用壁钟而不是 monotonic，因为子进程的
      monotonic 基准与父进程不一致
    """

    rpm: float
    tpm: float
    last: float


class FileTokenBucket:
    """基于 filelock + JSON 文件的跨进程令牌桶。

    单次 acquire/refund 的临界区内耗约为 "读 1KB + 写 1KB + fsync"，在本地 SSD 上通常 <1ms。
    即便 16 workers 挤一把锁，P99 也只是增加个位 ms；不会成为瓶颈。

    状态文件不存在时视为满桶（首次启动/新 run_id）；文件损坏时按满桶重建，避免
    "上次 crash 留下半截 JSON" 拖垮整次运行。
    """

    def __init__(
        self,
        *,
        state_path: Path,
        lock_path: Path,
        rpm_capacity: int,
        tpm_capacity: int,
    ) -> None:
        if rpm_capacity <= 0 or tpm_capacity <= 0:
            raise ValueError("rpm_capacity and tpm_capacity must be positive.")
        self._state_path = state_path
        # filelock 接受 str 路径
        self._lock = FileLock(str(lock_path))
        self._rpm_cap = int(rpm_capacity)
        self._tpm_cap = int(tpm_capacity)
        # 父目录可能还不存在——acquire/refund 都会走这个路径，提前建好
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    # -- 私有工具 -----------------------------------------------------------------

    def _load_or_init(self) -> _BucketState:
        """读磁盘状态；不存在或损坏时按满桶初始化。"""
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                return _BucketState(
                    rpm=float(raw.get("rpm", self._rpm_cap)),
                    tpm=float(raw.get("tpm", self._tpm_cap)),
                    last=float(raw.get("last", time.time())),
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                # 损坏的状态文件当满桶处理；下面的 _save 会覆盖
                pass
        return _BucketState(rpm=float(self._rpm_cap), tpm=float(self._tpm_cap), last=time.time())

    def _save(self, state: _BucketState) -> None:
        # 写临时文件 + os.replace 实现原子替换：并发读者只会看到旧值或新值，
        # 永远不会读到撕裂的半截 JSON。tempfile.mkstemp 用 O_CREAT|O_EXCL 拿到唯一文件名，
        # 防止外部预放置 symlink 把写入引向 state_dir 之外。
        payload = {"rpm": state.rpm, "tpm": state.tpm, "last": state.last}
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self._state_path.name}.", suffix=".tmp", dir=self._state_path.parent
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp_name, self._state_path)
        except BaseException:
            # rename 失败时清理临时文件，避免目录里留垃圾
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise

    def _refill(self, state: _BucketState) -> _BucketState:
        """按壁钟线性补充令牌；负 elapsed（时钟回拨）视作 0。"""
        now = time.time()
        elapsed = max(0.0, now - state.last)
        state.rpm = min(float(self._rpm_cap), state.rpm + elapsed * self._rpm_cap / 60.0)
        state.tpm = min(float(self._tpm_cap), state.tpm + elapsed * self._tpm_cap / 60.0)
        state.last = now
        return state

    # -- 公共 API -----------------------------------------------------------------

    def acquire(self, *, need_rpm: int, need_tpm: int) -> float:
        """尝试消费 `(need_rpm, need_tpm)`；成功返 0.0，失败返应等待秒数。

        失败时不消费、仅把刷新后的状态落盘（让其他等待者也能读到最新时间戳）。调用方
        收到 >0 的返回值时应 `time.sleep(wait + jitter)` 后再试。
        """
        with self._lock:
            state = self._refill(self._load_or_init())
            if state.rpm >= need_rpm and state.tpm >= need_tpm:
                state.rpm -= need_rpm
                state.tpm -= need_tpm
                self._save(state)
                return 0.0
            # 不够：计算还差多少令牌、转成秒数（桶的刷新速率是 cap/60 令牌/秒）
            wait_r = (need_rpm - state.rpm) * 60.0 / self._rpm_cap if need_rpm > 0 else 0.0
            wait_t = (need_tpm - state.tpm) * 60.0 / self._tpm_cap if need_tpm > 0 else 0.0
            self._save(state)
            return max(0.0, wait_r, wait_t)

    def refund(self, *, rpm: int, tpm: int) -> None:
        """把预扣过但实际未消耗的令牌退回桶里；封顶在 capacity，不会溢出。"""
        if rpm <= 0 and tpm <= 0:
            return
        with self._lock:
            state = self._refill(self._load_or_init())
            state.rpm = min(float(self._rpm_cap), state.rpm + max(0, rpm))
            state.tpm = min(float(self._tpm_cap), state.tpm + max(0, tpm))
            self._save(state)
