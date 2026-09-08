from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.public_module.config import load_config
    from experiments.public_module.result_validation import validate_sequence_rows
else:
    from .config import load_config
    from .result_validation import validate_sequence_rows


METRICS = (
    "ce95",
    "absolute_log_scale_ratio",
    "holdout_nll",
    "estimated_scale",
    "coverage95",
    "calibration_time_ms",
    "random_trace_median_relative_error",
    "random_trace_p95_relative_error",
    "random_trace_mae",
    "random_gain_median_relative_error",
    "random_gain_p95_relative_error",
    "k16_vs_exact_scale_relative_error",
    "projection_basis_time_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def read_sequence_rows(input_root: Path, sequence: str) -> tuple[list[dict[str, str]], Path]:
    candidates = sorted(input_root.rglob(f"{sequence}/pair_method_results.csv"))
    if len(candidates) != 1:
        raise ValueError(f"{sequence} 的正式结果文件不是唯一值：{candidates}")
    path = candidates[0]
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{sequence} 结果为空")
    return rows, path


def median(values: list[float]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(np.quantile(np.asarray(values, dtype=float), 0.5, method="linear"))


def paired_bootstrap_interval(differences: np.ndarray, repeats: int, seed: int) -> list[float]:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences) or not np.all(np.isfinite(differences)):
        raise ValueError("bootstrap需要非空、有限的一维序列差值")
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("bootstrap次数必须为正整数")
    rng = np.random.default_rng(seed)
    count = len(differences)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = differences[rng.integers(0, count, size=count)]
        draws[index] = np.quantile(sample, 0.5, method="linear")
    return [
        float(np.quantile(draws, 0.025, method="linear")),
        float(np.quantile(draws, 0.975, method="linear")),
    ]


def available_interval(values, repeats, seed):
    # 缺失整个序列时不静默缩小独立统计单位；结果保持不可估计。
    return None if any(value is None for value in values) else paired_bootstrap_interval(values, repeats, seed)


def metric_difference(corrected, raw):
    return None if corrected is None or raw is None else float(corrected) - float(raw)


def aggregate(config: dict[str, Any], input_root: Path, output_dir: Path) -> int:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}")
    sequences = list(config["dataset"]["sequences"])
    methods = list(config["calibration"]["methods"])
    budgets = [int(value) for value in config["split"]["fit_budgets"]]
    all_rows: list[dict[str, str]] = []
    source_hashes: dict[str, str] = {}
    for sequence in sequences:
        rows, path = read_sequence_rows(input_root, sequence)
        if any(row.get("provenance") != config["provenance"] for row in rows):
            raise ValueError(f"{sequence} 含有非正式来源行")
        validate_sequence_rows(rows, path, sequence, config, METRICS)
        all_rows.extend(rows)
        source_hashes[str(path.resolve())] = sha256_file(path)

    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["sequence"], int(row["fit_budget"]), row["method"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    failure_table: list[dict[str, Any]] = []
    for sequence in sequences:
        for budget in budgets:
            valid_pair_sets: dict[str, set[str]] = {}
            for method in methods:
                method_rows = grouped[(sequence, budget, method)]
                valid_pair_sets[method] = {
                    row["pair_id"] for row in method_rows if parse_bool(row.get("valid", "false"))
                }
            if len({frozenset(value) for value in valid_pair_sets.values()}) != 1:
                raise ValueError(f"共同有效集合不一致：{sequence}, N={budget}")
            common_valid = valid_pair_sets[methods[0]]
            for method in methods:
                method_rows = grouped[(sequence, budget, method)]
                valid_rows = [row for row in method_rows if row["pair_id"] in common_valid]
                result: dict[str, Any] = {
                    "sequence": sequence,
                    "fit_budget": budget,
                    "method": method,
                    "candidate_pairs": len(method_rows),
                    "valid_pairs": len(valid_rows),
                }
                for metric in METRICS:
                    result[f"median_{metric}"] = median(
                        [float(row[metric]) for row in valid_rows if row.get(metric, "") != ""]
                    )
                summary_rows.append(result)
                failures = Counter(
                    row.get("failure_reason", "UNKNOWN")
                    for row in method_rows
                    if not parse_bool(row.get("valid", "false"))
                )
                for reason, count in sorted(failures.items()):
                    failure_table.append(
                        {
                            "sequence": sequence,
                            "fit_budget": budget,
                            "method": method,
                            "failure_reason": reason,
                            "count": count,
                        }
                    )

    output_dir.mkdir(parents=True)
    summary_csv = output_dir / "sequence_method_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    failure_csv = output_dir / "failure_counts.csv"
    with failure_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["sequence", "fit_budget", "method", "failure_reason", "count"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_table)

    lookup = {
        (row["sequence"], int(row["fit_budget"]), row["method"]): row for row in summary_rows
    }
    primary_budget = int(config["statistics"]["primary_fit_budget"])
    minimum_pairs = int(config["statistics"]["minimum_valid_pairs_per_sequence"])
    ce_differences: list[float] = []
    log_differences: list[float] = []
    nll_differences: list[float] = []
    ce_wins = 0
    log_wins = 0
    minimum_complete = True
    sequence_effects: list[dict[str, Any]] = []
    for sequence in sequences:
        raw = lookup[(sequence, primary_budget, "Raw")]
        duc = lookup[(sequence, primary_budget, "DUC-K16")]
        ce_difference = metric_difference(duc["median_ce95"], raw["median_ce95"])
        log_difference = metric_difference(duc["median_absolute_log_scale_ratio"], raw["median_absolute_log_scale_ratio"])
        nll_difference = metric_difference(duc["median_holdout_nll"], raw["median_holdout_nll"])
        ce_differences.append(ce_difference)
        log_differences.append(log_difference)
        nll_differences.append(nll_difference)
        ce_wins += int(ce_difference is not None and ce_difference < 0.0)
        log_wins += int(log_difference is not None and log_difference < 0.0)
        minimum_complete &= int(raw["valid_pairs"]) >= minimum_pairs
        sequence_effects.append(
            {
                "sequence": sequence,
                "ce95_difference": ce_difference,
                "absolute_log_scale_ratio_difference": log_difference,
                "holdout_nll_difference": nll_difference,
                "valid_pairs": int(raw["valid_pairs"]),
            }
        )

    ce_median, log_median, nll_median = [median(v) for v in (ce_differences, log_differences, nll_differences)]
    repeats = int(config["statistics"]["cluster_bootstrap_repeats"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    required_wins = int(config["statistics"]["positive_gate_min_sequence_wins"])
    gate_checks = {
        "ce95_sequence_wins": ce_wins >= required_wins,
        "ce95_equal_sequence_median_difference_negative": ce_median is not None and ce_median < 0.0,
        "scale_log_error_sequence_wins": log_wins >= required_wins,
        "scale_log_error_equal_sequence_median_difference_negative": log_median is not None and log_median < 0.0,
        "holdout_nll_equal_sequence_median_not_worse": nll_median is not None and nll_median <= 0.0,
        "minimum_valid_pairs_all_sequences": bool(minimum_complete),
    }
    gate = {
        "protocol_version": config["protocol_version"],
        "status": "PASS" if all(gate_checks.values()) else "FAIL_FINAL",
        "primary_fit_budget": primary_budget,
        "checks": gate_checks,
        "sequence_effects": sequence_effects,
        "equal_sequence_median_differences": {
            "ce95": median(ce_differences),
            "absolute_log_scale_ratio": median(log_differences),
            "holdout_nll": median(nll_differences),
        },
        "descriptive_cluster_bootstrap_95_percentile_intervals": {
            "ce95": available_interval(ce_differences, repeats, bootstrap_seed),
            "absolute_log_scale_ratio": available_interval(log_differences, repeats, bootstrap_seed),
            "holdout_nll": available_interval(nll_differences, repeats, bootstrap_seed),
        },
        "source_hashes": source_hashes,
        "sequence_summary_sha256": sha256_file(summary_csv),
        "failure_counts_sha256": sha256_file(failure_csv),
        "config_sha256": config["_config_sha256"],
    }
    (output_dir / "gate_review.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="聚合 EuRoC 模块校准结果")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return aggregate(load_config(args.config), args.input_root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
