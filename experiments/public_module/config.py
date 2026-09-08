from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """冻结配置无效。"""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = config_path.read_bytes()
    manifest_path = Path(__file__).resolve().parents[2] / 'evidence/source_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = next(item['sha256'] for item in manifest['files']
                    if item['path'] == 'experiments/contracts/public_module_config_v1.5.json')
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ConfigurationError('配置文件不匹配公开冻结版本；科学配置变更需要新版本。')
    config = json.loads(raw.decode("utf-8"))
    errors = validate_config(config)
    if errors:
        raise ConfigurationError("；".join(errors))
    config["_config_path"] = str(config_path.resolve())
    config["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("protocol_version") != "v1.5":
        errors.append("protocol_version 必须为 v1.5")
    if config.get("provenance") != "formal_public_module_evidence":
        errors.append("正式来源标记不正确")

    dataset = config.get("dataset", {})
    sequences = dataset.get("sequences", [])
    if len(sequences) != 6 or len(set(sequences)) != 6:
        errors.append("EuRoC 序列必须是六条互异序列")
    if dataset.get("origin") != "real_world_public_dataset":
        errors.append("数据来源类型不正确")

    pairs = config.get("frame_pairs", {})
    for key in ("start_index", "end_margin", "start_stride", "temporal_gap_frames"):
        if not isinstance(pairs.get(key), int) or pairs[key] <= 0:
            errors.append(f"{key} 必须为正整数")

    stereo = config.get("stereo", {})
    disparities = stereo.get("num_disparities")
    if not isinstance(disparities, int) or disparities <= 0 or disparities % 16:
        errors.append("num_disparities 必须是 16 的正整数倍")
    if stereo.get("block_size", 0) % 2 != 1:
        errors.append("StereoSGBM block_size 必须为奇数")

    split = config.get("split", {})
    budgets = split.get("fit_budgets", [])
    holdout = split.get("holdout_tracks")
    if budgets != [16, 48]:
        errors.append("拟合观测量必须冻结为 [16, 48]")
    if not isinstance(holdout, int) or holdout <= 0:
        errors.append("预留观测量必须为正整数")
    if not split.get("nested_fit_sets"):
        errors.append("两档拟合集必须嵌套")

    calibration = config.get("calibration", {})
    expected_methods = [
        "Raw",
        "Global-DoF",
        "DPR-Exact-Block",
        "DUC-K16",
        "Permuted-K16",
    ]
    if calibration.get("methods") != expected_methods:
        errors.append("比较方法或顺序与冻结方案不一致")
    if calibration.get("k") != 16:
        errors.append("K 必须保持为 16")
    if calibration.get("chi_square_df") != 2:
        errors.append("二维残差必须使用自由度 2")

    statistics = config.get("statistics", {})
    if statistics.get("independent_unit") != "sequence":
        errors.append("独立统计单位必须为 sequence")
    if statistics.get("primary_fit_budget") != 16:
        errors.append("主要观测量必须为 16")
    if statistics.get("minimum_valid_pairs_per_sequence") != 80:
        errors.append("每序列最小有效帧对数必须为 80")
    return errors
