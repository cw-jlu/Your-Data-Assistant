"""`FileTokenBucket` 行为测试。

关键断言：
- 初始满桶；首次 acquire 从满桶扣减
- 超容量请求 → 返等待秒数，桶不消费
- 按壁钟线性刷新：等 0.5s → 补回 cap/120 个令牌
- refund 封顶在 cap，不会溢出
- 同一对 state/lock 文件上的两个实例能互相看到对方的消费（跨 "进程" 语义）
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from agents.llm.rate_limit import FileTokenBucket


def _make_bucket(tmp_path: Path, *, rpm: int = 600, tpm: int = 1000) -> FileTokenBucket:
    return FileTokenBucket(
        state_path=tmp_path / "bucket.json",
        lock_path=tmp_path / "bucket.lock",
        rpm_capacity=rpm,
        tpm_capacity=tpm,
    )


def test_initial_consume_within_capacity(tmp_path: Path) -> None:
    bucket = _make_bucket(tmp_path, rpm=600, tpm=1000)
    wait = bucket.acquire(need_rpm=1, need_tpm=100)
    assert wait == 0.0
    # 状态文件应落盘，且数值反映消费后状态
    state = json.loads((tmp_path / "bucket.json").read_text())
    assert state["rpm"] == pytest.approx(599.0, abs=0.01)
    assert state["tpm"] == pytest.approx(900.0, abs=0.01)


def test_over_capacity_returns_wait_without_consuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 固定时钟：避免测试飘
    monkeypatch.setattr("agents.llm.rate_limit.time.time", lambda: 1000.0)
    bucket = _make_bucket(tmp_path, rpm=60, tpm=60)
    # 先把桶里的令牌花光
    assert bucket.acquire(need_rpm=60, need_tpm=60) == 0.0
    # 再请求 30 个——60 RPM / min = 1 RPM/s，等待 30s
    wait = bucket.acquire(need_rpm=30, need_tpm=0)
    assert wait == pytest.approx(30.0, abs=0.5)
    # 未消费：状态里 RPM 仍为 ~0
    state = json.loads((tmp_path / "bucket.json").read_text())
    assert state["rpm"] <= 1.0


def test_linear_refill_math(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 模拟时钟跨过 0.5 秒：60 RPM/min 应补 0.5 个；60 TPM/min 同理
    now = [1000.0]
    monkeypatch.setattr("agents.llm.rate_limit.time.time", lambda: now[0])
    bucket = _make_bucket(tmp_path, rpm=60, tpm=120)
    # 花光
    assert bucket.acquire(need_rpm=60, need_tpm=120) == 0.0
    # 推进时钟 0.5s：应补 0.5 RPM + 1 TPM
    now[0] += 0.5
    # 请 0.4 RPM / 0.9 TPM 都应成功
    wait = bucket.acquire(need_rpm=0, need_tpm=0)  # 触发一次 refill + save
    assert wait == 0.0
    state = json.loads((tmp_path / "bucket.json").read_text())
    assert state["rpm"] == pytest.approx(0.5, abs=0.05)
    assert state["tpm"] == pytest.approx(1.0, abs=0.05)


def test_refund_caps_at_capacity(tmp_path: Path) -> None:
    bucket = _make_bucket(tmp_path, rpm=100, tpm=1000)
    # 初始满桶；refund 不应让数值超过 cap
    bucket.refund(rpm=50, tpm=500)
    state = json.loads((tmp_path / "bucket.json").read_text())
    assert state["rpm"] <= 100.0
    assert state["tpm"] <= 1000.0


def test_refund_restores_consumed_tokens(tmp_path: Path) -> None:
    bucket = _make_bucket(tmp_path, rpm=100, tpm=1000)
    bucket.acquire(need_rpm=10, need_tpm=300)
    bucket.refund(rpm=0, tpm=200)
    state = json.loads((tmp_path / "bucket.json").read_text())
    # 退 200 后剩余 TPM 应 ~= 700 + 200 = 900（忽略微小刷新）
    assert state["tpm"] >= 890.0


def test_two_instances_share_state_via_file(tmp_path: Path) -> None:
    """两个 FileTokenBucket 实例指向同一对 state/lock —— 模拟跨进程协作。

    实例 A 消费后，实例 B 能从磁盘读到同一状态（filelock 保证串行化）。
    """
    kwargs = {
        "state_path": tmp_path / "bucket.json",
        "lock_path": tmp_path / "bucket.lock",
        "rpm_capacity": 100,
        "tpm_capacity": 1000,
    }
    a = FileTokenBucket(**kwargs)  # type: ignore[arg-type]
    b = FileTokenBucket(**kwargs)  # type: ignore[arg-type]

    a.acquire(need_rpm=40, need_tpm=400)
    # B 发起一个大请求：桶剩 ~60 RPM / ~600 TPM，60 RPM 以内应成功
    wait = b.acquire(need_rpm=60, need_tpm=500)
    assert wait == 0.0
    # 再请求已超——应等
    wait2 = b.acquire(need_rpm=50, need_tpm=0)
    assert wait2 > 0


def test_corrupted_state_file_treated_as_full(tmp_path: Path) -> None:
    """state 文件损坏时按满桶重建——避免一次 crash 拖垮整次运行。"""
    state_path = tmp_path / "bucket.json"
    state_path.write_text("{ not valid json")
    bucket = FileTokenBucket(
        state_path=state_path,
        lock_path=tmp_path / "bucket.lock",
        rpm_capacity=100,
        tpm_capacity=1000,
    )
    wait = bucket.acquire(need_rpm=50, need_tpm=500)
    assert wait == 0.0


def test_zero_need_returns_zero_wait(tmp_path: Path) -> None:
    """need_rpm=need_tpm=0 相当于 "只刷新桶状态"，永远立即成功。"""
    bucket = _make_bucket(tmp_path, rpm=60, tpm=60)
    # 先花光
    bucket.acquire(need_rpm=60, need_tpm=60)
    wait = bucket.acquire(need_rpm=0, need_tpm=0)
    assert wait == 0.0


def test_constructor_rejects_non_positive_capacity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        FileTokenBucket(
            state_path=tmp_path / "bucket.json",
            lock_path=tmp_path / "bucket.lock",
            rpm_capacity=0,
            tpm_capacity=100,
        )


def test_concurrent_reads_observe_only_valid_json(tmp_path: Path) -> None:
    """`_save` 原子性：并发读者绝不能读到截断或半写的字节。

    `_save` 必须用 "写临时文件 + os.replace" 保证并发读者要么读到旧的完整
    JSON，要么读到新的完整 JSON。多个 writer 线程持续 acquire/refund（间接
    触发 _save），一个 reader 线程绕过锁直接读状态文件并 `json.loads`，断言
    所有非空读取都是合法 JSON。旧实现（直接 write_text）会被本测试稳定捕获
    ——open(W) 截断与后续 write 之间存在撕裂窗口。
    """
    bucket = FileTokenBucket(
        state_path=tmp_path / "bucket.json",
        lock_path=tmp_path / "bucket.lock",
        rpm_capacity=600,
        tpm_capacity=10_000,
    )
    # 触发首次落盘，让 reader 不会一直读到空文件
    bucket.acquire(need_rpm=1, need_tpm=10)

    state_path = tmp_path / "bucket.json"
    stop = threading.Event()
    parse_failures: list[str] = []
    successful_reads = 0
    reads_lock = threading.Lock()

    def writer() -> None:
        # 反复 refund + acquire：每次都触发一次 _save，制造高频写
        while not stop.is_set():
            bucket.refund(rpm=1, tpm=10)
            bucket.acquire(need_rpm=1, need_tpm=10)

    def reader() -> None:
        nonlocal successful_reads
        while not stop.is_set():
            try:
                raw = state_path.read_text()
            except FileNotFoundError:
                # os.replace 期间窗口理论上不存在（rename 是原子的），
                # 但 read_text 有可能在文件刚被替换的瞬间命中——容忍
                continue
            if not raw:
                # 撕裂的旧实现会出现这种情况（write_text 先 truncate 后写）
                parse_failures.append("empty file observed")
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_failures.append(f"json error: {exc!r} on raw={raw!r}")
                continue
            # schema 也得对（防止以后 _save 写错字段）
            if not {"rpm", "tpm", "last"}.issubset(payload):
                parse_failures.append(f"missing keys in {payload!r}")
                continue
            with reads_lock:
                successful_reads += 1

    writers = [threading.Thread(target=writer) for _ in range(4)]
    readers = [threading.Thread(target=reader) for _ in range(2)]
    for t in writers + readers:
        t.start()
    # 短而密的窗口：足够触发上千次写/读，又不至于让测试跑太慢
    time.sleep(0.5)
    stop.set()
    for t in writers + readers:
        t.join(timeout=5.0)
        assert not t.is_alive(), "thread failed to exit"

    assert parse_failures == [], f"observed torn writes: {parse_failures[:5]}"
    # 没读到任何东西意味着调度太偏，测试本身失效——给个下限断言
    assert successful_reads > 50, f"only {successful_reads} reads—too sparse to be meaningful"
