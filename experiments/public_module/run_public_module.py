from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.public_module.config import load_config
    from experiments.public_module.core import (
        compute_scale_estimates,
        deterministic_permutation,
        evaluate_holdout,
        project_points,
        reprojection_residual,
        solve_relative_pose,
    )
    from experiments.public_module.euroc import (
        StereoRectifier,
        extract_tracks,
        frame_pair_start_indices,
        locate_sequence,
        read_image_index,
    )
else:
    from .config import load_config
    from .core import (
        compute_scale_estimates,
        deterministic_permutation,
        evaluate_holdout,
        project_points,
        reprojection_residual,
        solve_relative_pose,
    )
    from .euroc import (
        StereoRectifier,
        extract_tracks,
        frame_pair_start_indices,
        locate_sequence,
        read_image_index,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_row(
    sequence: str,
    pair_id: str,
    start_index: int,
    start_timestamp_ns: int,
    target_timestamp_ns: int,
    fit_budget: int,
    method: str,
    provenance: str,
    track_counts: tuple[int, int, int],
) -> dict[str, Any]:
    detected, disparity_valid, tracked = track_counts
    return {
        "sequence": sequence,
        "pair_id": pair_id,
        "start_index": start_index,
        "start_timestamp_ns": start_timestamp_ns,
        "target_timestamp_ns": target_timestamp_ns,
        "fit_budget": fit_budget,
        "method": method,
        "provenance": provenance,
        "detected_count": detected,
        "valid_disparity_count": disparity_valid,
        "tracked_count": tracked,
    }


def append_failure_rows(
    rows: list[dict[str, Any]],
    methods: list[str],
    common: dict[str, Any],
    reason: str,
) -> None:
    for method in methods:
        row = dict(common)
        row.update({"method": method, "valid": False, "failure_reason": reason})
        rows.append(row)


def report_progress(sequence: str, ordinal: int, total: int) -> None:
    if ordinal % 20 == 0 or ordinal == total:
        print(f"PROGRESS {sequence} {ordinal}/{total}", flush=True)


def run_sequence(
    config: dict[str, Any],
    dataset_root: Path,
    sequence: str,
    output_dir: Path,
    profile: str,
    max_pairs: int | None,
) -> int:
    if profile not in ('tiny', 'formal'):
        raise ValueError('profile必须为tiny或formal')
    if (profile == 'tiny' and (type(max_pairs) is not int or max_pairs <= 0)) or (profile == 'formal' and max_pairs is not None):
        raise ValueError('tiny需要正整数max_pairs；formal禁止max_pairs')
    if sequence not in config["dataset"]["sequences"]:
        raise ValueError(f"序列不在冻结清单中：{sequence}")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")

    started = datetime.now(timezone.utc).isoformat()
    sequence_root = locate_sequence(dataset_root, sequence)
    records = read_image_index(sequence_root)
    rectifier = StereoRectifier.from_sequence(sequence_root, config["stereo"])
    indices = frame_pair_start_indices(len(records), config["frame_pairs"])
    if profile == "tiny":
        if max_pairs is None or max_pairs <= 0:
            raise ValueError("tiny 模式必须给出正的 --max-pairs")
        indices = indices[:max_pairs]
        provenance = "dev_tiny_real_sample"
    else:
        if max_pairs is not None:
            raise ValueError("formal 模式禁止使用 --max-pairs")
        provenance = str(config["provenance"])

    methods = list(config["calibration"]["methods"])
    output_dir.mkdir(parents=True)
    holdout_count = int(config["split"]["holdout_tracks"])
    fit_budgets = [int(value) for value in config["split"]["fit_budgets"]]
    required_tracks = holdout_count + max(fit_budgets)
    master_seed = int(config["split"]["master_seed"])
    cv2.setNumThreads(int(config["solver"]["opencv_threads"]))
    cv2.setRNGSeed(master_seed % (2**31 - 1))
    rows: list[dict[str, Any]] = []

    for ordinal, start_index in enumerate(indices, start=1):
        start_record = records[start_index]
        target_record = records[start_index + int(config["frame_pairs"]["temporal_gap_frames"])]
        pair_id = f"{sequence}:{start_record.timestamp_ns}:{target_record.timestamp_ns}"
        try:
            tracks = extract_tracks(
                sequence_root,
                records,
                start_index,
                int(config["frame_pairs"]["temporal_gap_frames"]),
                rectifier,
                config["features"],
                config["stereo"],
            )
            track_counts = (
                tracks.detected_count,
                tracks.valid_disparity_count,
                tracks.tracked_count,
            )
        except Exception as exc:
            track_counts = (0, 0, 0)
            for budget in fit_budgets:
                common = base_row(
                    sequence,
                    pair_id,
                    start_index,
                    start_record.timestamp_ns,
                    target_record.timestamp_ns,
                    budget,
                    "",
                    provenance,
                    track_counts,
                )
                append_failure_rows(rows, methods, common, f"PREPROCESSING_ERROR:{type(exc).__name__}")
            report_progress(sequence, ordinal, len(indices))
            continue

        if tracks.tracked_count < required_tracks:
            for budget in fit_budgets:
                common = base_row(
                    sequence,
                    pair_id,
                    start_index,
                    start_record.timestamp_ns,
                    target_record.timestamp_ns,
                    budget,
                    "",
                    provenance,
                    track_counts,
                )
                append_failure_rows(rows, methods, common, "INSUFFICIENT_TRACKS")
            report_progress(sequence, ordinal, len(indices))
            continue

        order = deterministic_permutation(
            tracks.tracked_count,
            master_seed,
            "public_module_split",
            sequence,
            start_record.timestamp_ns,
        )
        holdout_indices = order[:holdout_count]
        fit_pool = order[holdout_count : holdout_count + max(fit_budgets)]
        holdout_points_3d = tracks.points_3d[holdout_indices]
        holdout_points_2d = tracks.target_points_2d[holdout_indices]

        for budget in fit_budgets:
            fit_indices = fit_pool[:budget]
            fit_points_3d = tracks.points_3d[fit_indices]
            fit_points_2d = tracks.target_points_2d[fit_indices]
            common = base_row(
                sequence,
                pair_id,
                start_index,
                start_record.timestamp_ns,
                target_record.timestamp_ns,
                budget,
                "",
                provenance,
                track_counts,
            )
            solution = solve_relative_pose(
                fit_points_3d,
                fit_points_2d,
                holdout_points_3d,
                rectifier.rectified_camera,
                config["solver"],
            )
            common.update(
                {
                    "solver_success": solution.success,
                    "solver_reason": solution.reason,
                    "solver_nfev": solution.nfev,
                    "jacobian_rank": solution.rank,
                    "stationarity_eta": solution.eta,
                    "fit_min_depth_m": solution.fit_min_depth_m,
                    "holdout_min_depth_m": solution.holdout_min_depth_m,
                }
            )
            if not solution.success:
                append_failure_rows(rows, methods, common, solution.reason)
                continue
            assert solution.parameters is not None
            assert solution.residual is not None
            assert solution.jacobian is not None
            holdout_residual = reprojection_residual(
                solution.parameters,
                holdout_points_3d,
                holdout_points_2d,
                rectifier.rectified_camera,
            )
            _, holdout_depth = project_points(
                solution.parameters,
                holdout_points_3d,
                rectifier.rectified_camera,
            )
            common.update(
                {
                    "fit_rms_px": float(np.sqrt(np.mean(solution.residual * solution.residual))),
                    "holdout_rms_px": float(np.sqrt(np.mean(holdout_residual * holdout_residual))),
                    "holdout_positive_depth": bool(np.all(holdout_depth > 0.0)),
                }
            )
            try:
                scales = compute_scale_estimates(
                    solution.residual,
                    solution.jacobian,
                    int(config["calibration"]["k"]),
                    float(config["calibration"]["epsilon"]),
                    master_seed,
                    (sequence, start_record.timestamp_ns, budget),
                )
                method_rows = []
                for method in methods:
                    metrics = evaluate_holdout(
                        holdout_residual,
                        scales.scales[method],
                        float(config["calibration"]["coverage_probability"]),
                    )
                    row = dict(common)
                    row.update(
                        {
                            "method": method,
                            "valid": True,
                            "failure_reason": "",
                            "estimated_scale": scales.scales[method],
                            "calibration_time_ms": scales.timings_ms[method],
                            **metrics,
                            **scales.diagnostics,
                        }
                    )
                    method_rows.append(row)
                # 同一帧对/拟合集的五种方法全部完成后再写入，异常时不会混入半组有效行。
                rows.extend(method_rows)
            except Exception as exc:
                append_failure_rows(rows, methods, common, f"CALIBRATION_ERROR:{type(exc).__name__}")

        report_progress(sequence, ordinal, len(indices))

    fieldnames = sorted({key for row in rows for key in row})
    result_csv = output_dir / "pair_method_results.csv"
    with result_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    status_rows = [row for row in rows if row["method"] == methods[0]]
    failure_counts = Counter(str(row.get("failure_reason", "")) for row in status_rows if not row.get("valid"))
    summary = {
        "protocol_version": config["protocol_version"],
        "profile": profile,
        "provenance": provenance,
        "sequence": sequence,
        "sequence_root": str(sequence_root.resolve()),
        "candidate_pair_count": len(indices),
        "valid_pair_budget_count": int(sum(bool(row.get("valid")) for row in status_rows)),
        "valid_pairs_by_budget": {
            str(budget): int(
                sum(bool(row.get("valid")) and int(row["fit_budget"]) == budget for row in status_rows)
            )
            for budget in fit_budgets
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config["_config_sha256"],
        "result_csv_sha256": sha256_file(result_csv),
        "execution_status": "COMPLETED",
        "evidence_eligibility": "INDEPENDENT_REPRODUCTION_REQUIRES_REVIEW" if profile == "formal" else "SMOKE_ONLY",
        "replaces_archived_paper_results": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "cam0_sensor_sha256": sha256_file(sequence_root / "mav0" / "cam0" / "sensor.yaml"),
        "cam1_sensor_sha256": sha256_file(sequence_root / "mav0" / "cam1" / "sensor.yaml"),
        "cam0_index_sha256": sha256_file(sequence_root / "mav0" / "cam0" / "data.csv"),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 EuRoC 局部残差校准实验")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("tiny", "formal"), default="formal")
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    return run_sequence(
        config,
        args.dataset_root,
        args.sequence,
        args.output,
        args.profile,
        args.max_pairs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
