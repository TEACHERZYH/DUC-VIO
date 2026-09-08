"""Apply the reviewed v1.6 narrowed sequence scope to inherited v1.5 results."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.public_module.aggregate_public_module import aggregate, read_sequence_rows, sha256_file, METRICS
    from experiments.public_module.config import load_config
    from experiments.public_module.result_validation import validate_sequence_rows
else:
    from .aggregate_public_module import aggregate, read_sequence_rows, sha256_file, METRICS
    from .config import load_config
    from .result_validation import validate_sequence_rows


def load_scope(path: Path, base_config: dict[str, Any]) -> dict[str, Any]:
    scope = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    confirmatory = list(scope.get("confirmatory_sequences", []))
    boundary = list(scope.get("boundary_sequences", []))
    base_sequences = list(base_config["dataset"]["sequences"])
    if scope.get("protocol_version") != "v1.6":
        errors.append("scope protocol_version 必须为 v1.6")
    if scope.get("base_protocol_version") != "v1.5":
        errors.append("base_protocol_version 必须为 v1.5")
    if scope.get("base_config_sha256") != base_config["_config_sha256"]:
        errors.append("base config 哈希不一致")
    if len(confirmatory) != 5 or len(set(confirmatory)) != 5:
        errors.append("确认性序列必须是五条互异序列")
    if boundary != ["V2_03_difficult"]:
        errors.append("边界序列必须固定为 V2_03_difficult")
    if set(confirmatory) | set(boundary) != set(base_sequences):
        errors.append("确认性与边界序列没有完整覆盖 v1.5 序列")
    if set(confirmatory) & set(boundary):
        errors.append("确认性与边界序列重叠")
    if scope.get("minimum_valid_pairs_per_sequence") != 80:
        errors.append("最小有效帧对数必须保持为 80")
    if scope.get("positive_gate_min_sequence_wins") != 4:
        errors.append("序列改善数门槛必须保持为 4")
    if scope.get("primary_fit_budget") != 16:
        errors.append("主要拟合规模必须保持为 16")
    if scope.get("new_scientific_runs_allowed") is not False:
        errors.append("v1.6 禁止新科学运行")
    if errors:
        raise ValueError("；".join(errors))
    scope["_scope_sha256"] = sha256_file(path)
    return scope


def valid_counts(input_root: Path, sequence: str, config: dict[str, Any]) -> dict[str, int]:
    rows, path = read_sequence_rows(input_root, sequence)
    return validate_sequence_rows(rows, path, sequence, config, METRICS)


def aggregate_narrowed(
    base_config_path: Path,
    scope_path: Path,
    input_root: Path,
    output_dir: Path,
) -> int:
    base = load_config(base_config_path)
    scope = load_scope(scope_path, base)
    minimum = int(scope["minimum_valid_pairs_per_sequence"])
    counts = {
        sequence: valid_counts(input_root, sequence, base)
        for sequence in list(scope["confirmatory_sequences"]) + list(scope["boundary_sequences"])
    }
    if any(
        any(value < minimum for value in counts[sequence].values())
        for sequence in scope["confirmatory_sequences"]
    ):
        raise ValueError("确认性序列包含未达到 v1.5 最小有效样本数的序列")
    if not any(
        any(value < minimum for value in counts[sequence].values())
        for sequence in scope["boundary_sequences"]
    ):
        raise ValueError("边界序列没有触发 v1.5 最小有效样本数规则")

    narrowed = copy.deepcopy(base)
    narrowed["protocol_version"] = "v1.6"
    narrowed["dataset"]["sequences"] = list(scope["confirmatory_sequences"])
    narrowed["statistics"]["positive_gate_min_sequence_wins"] = int(
        scope["positive_gate_min_sequence_wins"]
    )
    result = aggregate(narrowed, input_root, output_dir)
    gate_path = output_dir / "gate_review.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate.update(
        {
            "base_protocol_version": "v1.5",
            "base_config_sha256": base["_config_sha256"],
            "scope_sha256": scope["_scope_sha256"],
            "confirmatory_sequences": scope["confirmatory_sequences"],
            "boundary_sequences": scope["boundary_sequences"],
            "valid_counts_all_sequences": counts,
            "new_scientific_runs_performed": False,
        }
    )
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
def main() -> int:
    parser = argparse.ArgumentParser(description="聚合 v1.6 收窄后的 EuRoC 模块结果")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return aggregate_narrowed(args.base_config, args.scope, args.input_root, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
