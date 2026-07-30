"""Deterministic public-task split into train / holdout id files.

Holdout is the only honest signal we have for prompt iteration without
burning leaderboard submissions. Keep it strictly out of any prompt-tuning
loop.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset

HASH_BUCKETS = 100
DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_SALT = b"dabench-2026-v1"
TRAIN_FILE = "train_ids.txt"
HOLDOUT_FILE = "holdout_ids.txt"


@dataclass(frozen=True, slots=True)
class TaskSplit:
    train_ids: list[str]
    holdout_ids: list[str]

    @property
    def total(self) -> int:
        return len(self.train_ids) + len(self.holdout_ids)


def _bucket(task_id: str, salt: bytes = DEFAULT_SALT) -> int:
    digest = hashlib.blake2b(salt + task_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HASH_BUCKETS


def split_task_ids(
    task_ids: list[str],
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    salt: bytes = DEFAULT_SALT,
) -> TaskSplit:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1).")
    threshold = int(round(holdout_fraction * HASH_BUCKETS))
    train: list[str] = []
    holdout: list[str] = []
    for task_id in task_ids:
        if _bucket(task_id, salt) < threshold:
            holdout.append(task_id)
        else:
            train.append(task_id)
    return TaskSplit(train_ids=sorted(train), holdout_ids=sorted(holdout))


def write_split(split: TaskSplit, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / TRAIN_FILE
    holdout_path = output_dir / HOLDOUT_FILE
    train_path.write_text("\n".join(split.train_ids) + ("\n" if split.train_ids else ""))
    holdout_path.write_text("\n".join(split.holdout_ids) + ("\n" if split.holdout_ids else ""))
    return train_path, holdout_path


def load_split(split_dir: Path) -> TaskSplit:
    train_path = split_dir / TRAIN_FILE
    holdout_path = split_dir / HOLDOUT_FILE
    train = _read_ids(train_path)
    holdout = _read_ids(holdout_path)
    return TaskSplit(train_ids=train, holdout_ids=holdout)


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_split_for_dataset(
    dataset_root: Path,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> TaskSplit:
    dataset = DABenchPublicDataset(dataset_root)
    if not dataset.exists:
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    return split_task_ids(dataset.list_task_ids(), holdout_fraction=holdout_fraction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/holdout id files.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    args = parser.parse_args()

    split = build_split_for_dataset(args.dataset_root, holdout_fraction=args.holdout_fraction)
    train_path, holdout_path = write_split(split, args.output_dir)
    print(
        f"Wrote {len(split.train_ids)} train ids → {train_path}\n"
        f"Wrote {len(split.holdout_ids)} holdout ids → {holdout_path}\n"
        f"Holdout fraction: {len(split.holdout_ids) / split.total:.1%} (target {args.holdout_fraction:.1%})"
    )


if __name__ == "__main__":
    main()
