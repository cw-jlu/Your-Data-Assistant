from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv
load_dotenv(REPO / ".env")

from kobushi_core.benchmark.dataset import DABenchPublicDataset
from kobushi_core.model import OpenAIModelAdapter
from experiments.exp_144_modality.prefix_cache import with_prefix_cache_header
from experiments.exp_144_modality.video_relevance import assess_keyframe_relevance


INPUT_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"


def make_model(task_id: str) -> OpenAIModelAdapter:
    headers = with_prefix_cache_header(
        {
            "CF-Access-Client-Id": os.environ.get("CF_ACCESS_CLIENT_ID", ""),
            "CF-Access-Client-Secret": os.environ.get("CF_ACCESS_CLIENT_SECRET", ""),
        },
        task_id,
    )
    return OpenAIModelAdapter(
        model="qwen3.5-35b-a3b",
        api_base=os.environ.get("AGENT_API_BASE", ""),
        api_key=os.environ.get("AGENT_API_KEY", ""),
        temperature=0.2,
        extra_headers=headers,
        enable_thinking=False,
        max_tokens=2048,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--keyframe-root", required=True)
    ap.add_argument("--out", default="artifacts/runs/video_relevance_poc")
    ap.add_argument("--max-images", type=int, default=18)
    args = ap.parse_args()

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    keyframe_root = Path(args.keyframe_root)
    if not keyframe_root.is_absolute():
        keyframe_root = REPO / keyframe_root

    ds = DABenchPublicDataset(root_dir=INPUT_ROOT)
    results = []
    for tid in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        task = ds.get_task(tid)
        frames = sorted((keyframe_root / tid / "keyframes").glob("*.jpg"))
        result = assess_keyframe_relevance(
            task=task,
            keyframes=frames,
            model=make_model(tid),
            max_images=args.max_images,
        )
        result = {"task_id": tid, "keyframes": [str(p) for p in frames[: args.max_images]], **result}
        results.append(result)
        (out_dir / f"{tid}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"{tid}: type={result.get('key_info_type')} "
            f"sql_role={result.get('sql_role_hint')} "
            f"has={result.get('video_has_key_info')} "
            f"reason={result.get('reason')}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
