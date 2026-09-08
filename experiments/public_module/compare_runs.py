"""Compare the scientific fields of two public-module result tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TIMING_FIELDS = {"calibration_time_ms", "projection_basis_time_ms"}
KEY_FIELDS = ("sequence", "pair_id", "fit_budget", "method")


def _load(path: Path) -> tuple[list[str], dict[tuple[str, ...], dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(KEY_FIELDS) <= set(reader.fieldnames):
            raise ValueError("比较表缺少身份列或含重复列")
        fields = [field for field in reader.fieldnames if field not in TIMING_FIELDS]
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        for row in reader:
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in rows:
                raise ValueError(f"duplicate result key in {path}: {key}")
            rows[key] = row
    if not rows: raise ValueError("不能把两个空结果表判断为复现一致")
    return fields, rows


def _numeric_equal(left: str, right: str, rtol: float, atol: float) -> bool:
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def compare(
    left: Path,
    right: Path,
    rtol: float,
    atol: float,
    exclude_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("比较容差必须为有限非负数")
    left_fields, left_rows = _load(left)
    right_fields, right_rows = _load(right)
    excluded = TIMING_FIELDS | set(exclude_fields)
    common_fields = sorted((set(left_fields) & set(right_fields)) - excluded)
    field_mismatch = sorted((set(left_fields) ^ set(right_fields)) - excluded)
    key_mismatch = sorted(set(left_rows) ^ set(right_rows))
    mismatches: list[dict[str, object]] = []

    for key in sorted(set(left_rows) & set(right_rows)):
        for field in common_fields:
            a = left_rows[key][field]
            b = right_rows[key][field]
            if a == b or _numeric_equal(a, b, rtol=rtol, atol=atol):
                continue
            mismatches.append({"key": key, "field": field, "left": a, "right": b})
            if len(mismatches) >= 20:
                break
        if len(mismatches) >= 20:
            break

    passed = not field_mismatch and not key_mismatch and not mismatches
    return {
        "status": "PASS" if passed else "FAIL",
        "left": str(left),
        "right": str(right),
        "rtol": rtol,
        "atol": atol,
        "excluded_fields": sorted(excluded),
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "field_mismatch": field_mismatch,
        "key_mismatch_count": len(key_mismatch),
        "value_mismatch_count_capped": len(mismatches),
        "value_mismatch_examples": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--exclude-field", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare(
        args.left,
        args.right,
        rtol=args.rtol,
        atol=args.atol,
        exclude_fields=tuple(args.exclude_field),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
