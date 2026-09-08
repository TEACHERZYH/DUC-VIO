"""Create compact manuscript tables from the reviewed v1.6 aggregate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.public_module.result_validation import load_reviewed_aggregate


METHOD_ORDER = ["Raw", "Global-DoF", "DPR-Exact-Block", "DUC-K16", "Permuted-K16"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_tables(gate_path: Path, summary_path: Path, output_dir: Path) -> None:
    gate, summary_rows = load_reviewed_aggregate(gate_path, summary_path)
    if gate.get("protocol_version") != "v1.6" or gate.get("status") != "PASS":
        raise ValueError("tables require a reviewed PASS v1.6 gate")

    sequence_rows = [
        {
            "序列": effect["sequence"],
            "主规模有效帧对（汇总行为总数）": int(effect["valid_pairs"]),
            "ΔCE95": round(float(effect["ce95_difference"]), 4),
            "Δ绝对对数尺度比": round(float(effect["absolute_log_scale_ratio_difference"]), 4),
            "Δ留出集NLL": round(float(effect["holdout_nll_difference"]), 4),
        }
        for effect in gate["sequence_effects"]
    ]
    sequence_rows.append(
        {
            "序列": "五序列等权汇总",
            "主规模有效帧对（汇总行为总数）": sum(
                int(effect["valid_pairs"]) for effect in gate["sequence_effects"]
            ),
            "ΔCE95": round(float(gate["equal_sequence_median_differences"]["ce95"]), 4),
            "Δ绝对对数尺度比": round(
                float(gate["equal_sequence_median_differences"]["absolute_log_scale_ratio"]), 4
            ),
            "Δ留出集NLL": round(
                float(gate["equal_sequence_median_differences"]["holdout_nll"]), 4
            ),
        }
    )
    write_rows(output_dir / "Table_Public_Module_Sequence_Effects_v1_6.csv", sequence_rows)

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        if int(row["fit_budget"]) != 16:
            continue
        for source, target in (
            ("median_ce95", "CE95"),
            ("median_absolute_log_scale_ratio", "绝对对数尺度比"),
            ("median_holdout_nll", "留出集NLL"),
            ("median_calibration_time_ms", "增量时间_ms"),
        ):
            values[row["method"]][target].append(float(row[source]))
    method_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_rows.append(
            {
                "方法": method,
                "CE95_五序列中位数": round(float(np.median(values[method]["CE95"])), 4),
                "绝对对数尺度比_五序列中位数": round(
                    float(np.median(values[method]["绝对对数尺度比"])), 4
                ),
                "留出集NLL_五序列中位数": round(
                    float(np.median(values[method]["留出集NLL"])), 4
                ),
                "增量时间_ms_五序列中位数": round(
                    float(np.median(values[method]["增量时间_ms"])), 4
                ),
            }
        )
    write_rows(output_dir / "Table_Public_Module_Method_Summary_v1_6.csv", method_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    make_tables(args.gate, args.summary, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
