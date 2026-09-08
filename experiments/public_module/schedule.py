from __future__ import annotations

from typing import Any


def frame_pair_start_indices(record_count: int, frame_config: dict[str, Any]) -> list[int]:
    start = int(frame_config["start_index"])
    end = record_count - int(frame_config["end_margin"]) - int(frame_config["temporal_gap_frames"])
    stride = int(frame_config["start_stride"])
    if end <= start:
        raise ValueError("序列长度不足以生成冻结帧对")
    return list(range(start, end, stride))
