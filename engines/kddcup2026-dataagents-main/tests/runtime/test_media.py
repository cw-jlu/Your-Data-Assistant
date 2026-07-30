"""media.py：视频发现与 video agent 专用内容构造。"""

from __future__ import annotations

import base64
from pathlib import Path

from agents.runtime.media import build_video_user_content, find_videos


def test_find_videos_recurses_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.mp4").write_bytes(b"\x00")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.mov").write_bytes(b"\x00")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")

    videos = find_videos(tmp_path)

    assert [str(v.relative_to(tmp_path)) for v in videos] == ["b.mp4", "sub/a.mov"]


def test_build_video_user_content_attaches_data_uri(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    payload = b"\x00\x01\x02"
    video.write_bytes(payload)
    blocks = build_video_user_content("HELLO BRIEF", video)
    assert blocks[0] == {"type": "text", "text": "HELLO BRIEF"}
    url = blocks[1]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.standard_b64encode(payload).decode("ascii") in url
