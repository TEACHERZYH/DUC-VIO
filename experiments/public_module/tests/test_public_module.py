from __future__ import annotations

import csv
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.public_module.aggregate_public_module import aggregate
from experiments.public_module.aggregate_public_module_v1_6 import load_scope
from experiments.public_module.compare_runs import compare
from experiments.public_module.config import load_config, validate_config
from experiments.public_module.core import (
    compute_scale_estimates,
    derived_seed,
    evaluate_holdout,
    project_points,
    solve_relative_pose,
)
from experiments.public_module.euroc import read_camera_calibration
from experiments.public_module.schedule import frame_pair_start_indices


CONFIG_PATH = REPO_ROOT / "experiments" / "contracts" / "public_module_config_v1.5.json"
SCOPE_PATH = REPO_ROOT / "experiments" / "contracts" / "public_module_scope_v1.6.json"


class ConfigurationTests(unittest.TestCase):
    def test_frozen_configuration(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(config["split"]["fit_budgets"], [16, 48])
        self.assertEqual(config["calibration"]["k"], 16)
        self.assertEqual(config["statistics"]["independent_unit"], "sequence")

    def test_configuration_fails_closed(self) -> None:
        config = load_config(CONFIG_PATH)
        config = {key: value for key, value in config.items() if not key.startswith("_")}
        altered = deepcopy(config)
        altered["split"]["fit_budgets"] = [12, 48]
        self.assertTrue(validate_config(altered))

    def test_frozen_frame_schedule(self) -> None:
        config = load_config(CONFIG_PATH)
        indices = frame_pair_start_indices(1000, config["frame_pairs"])
        self.assertEqual(indices[:3], [100, 110, 120])
        self.assertEqual(indices[-1], 890)

    def test_standard_euroc_yaml_without_opencv_header(self) -> None:
        content = """sensor_type: camera
T_BS:
  cols: 4
  rows: 4
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
resolution: [752, 480]
intrinsics: [458.0, 457.0, 367.0, 248.0]
distortion_coefficients: [-0.28, 0.07, 0.0, 0.0]
"""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "sensor.yaml"
            path.write_text(content, encoding="utf-8")
            calibration = read_camera_calibration(path)
        self.assertEqual(calibration.resolution, (752, 480))
        self.assertEqual(calibration.body_from_sensor.shape, (4, 4))

    def test_v1_6_scope_only_narrows_the_v1_5_sequence_set(self) -> None:
        config = load_config(CONFIG_PATH)
        scope = load_scope(SCOPE_PATH, config)
        self.assertEqual(len(scope["confirmatory_sequences"]), 5)
        self.assertEqual(scope["boundary_sequences"], ["V2_03_difficult"])
        self.assertFalse(scope["new_scientific_runs_allowed"])


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH)
        rng = np.random.default_rng(401)
        points = rng.uniform([-1.5, -1.0, 3.0], [1.5, 1.0, 8.0], size=(96, 3))
        camera = np.asarray(
            [[460.0, 0.0, 376.0], [0.0, 460.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        parameters = np.asarray([0.02, -0.01, 0.015, 0.06, -0.02, 0.03])
        observations, _ = project_points(parameters, points, camera)
        observations += rng.normal(0.0, 0.8, size=observations.shape)
        cls.points = points
        cls.camera = camera
        cls.observations = observations

    def test_seed_domains_are_deterministic_and_separate(self) -> None:
        self.assertEqual(derived_seed(9, "a", 1), derived_seed(9, "a", 1))
        self.assertNotEqual(derived_seed(9, "a", 1), derived_seed(9, "b", 1))

    def test_pose_solution_and_scale_formulas(self) -> None:
        solution = solve_relative_pose(
            self.points[:48],
            self.observations[:48],
            self.points[48:72],
            self.camera,
            self.config["solver"],
        )
        self.assertTrue(solution.success, solution.reason)
        assert solution.residual is not None
        assert solution.jacobian is not None
        result = compute_scale_estimates(
            solution.residual,
            solution.jacobian,
            16,
            1e-8,
            91531,
            ("synthetic_test", 1, 48),
        )
        total_dimension, parameter_dimension = solution.jacobian.shape
        self.assertAlmostEqual(
            result.scales["Global-DoF"],
            result.scales["Raw"] * total_dimension / (total_dimension - parameter_dimension),
            places=12,
        )
        self.assertAlmostEqual(
            float(result.diagnostics["exact_trace_sum"]), parameter_dimension, places=8
        )
        repeated = compute_scale_estimates(
            solution.residual,
            solution.jacobian,
            16,
            1e-8,
            91531,
            ("synthetic_test", 1, 48),
        )
        self.assertEqual(result.scales, repeated.scales)

    def test_holdout_metric_definition(self) -> None:
        residual = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
        metrics = evaluate_holdout(residual, scale=1.0)
        self.assertEqual(metrics["coverage95"], 1.0)
        self.assertAlmostEqual(metrics["ce95"], 0.05)
        self.assertAlmostEqual(metrics["holdout_scale"], 2.5 / 6.0)

    def test_projection_matches_opencv(self) -> None:
        parameters = np.asarray([0.01, -0.02, 0.03, 0.1, -0.04, 0.02])
        ours, _ = project_points(parameters, self.points[:8], self.camera)
        opencv, _ = cv2.projectPoints(
            self.points[:8],
            parameters[:3],
            parameters[3:],
            self.camera,
            np.zeros(4),
        )
        self.assertTrue(np.allclose(ours, opencv.reshape(-1, 2), atol=1e-10))


class AggregationTests(unittest.TestCase):
    def test_sequence_level_gate_and_common_sets(self) -> None:
        config = load_config(CONFIG_PATH)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            methods = config["calibration"]["methods"]
            for sequence in config["dataset"]["sequences"]:
                directory = input_root / sequence
                directory.mkdir(parents=True)
                path = directory / "pair_method_results.csv"
                fieldnames = [
                    "sequence",
                    "pair_id",
                    "fit_budget",
                    "method",
                    "provenance",
                    "valid",
                    "failure_reason",
                    *METRIC_FIELDS,
                ]
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    for budget in (16, 48):
                        for index in range(80):
                            for method in methods:
                                improvement = 0.6 if method == "DUC-K16" else 1.0
                                writer.writerow(
                                    {
                                        "sequence": sequence,
                                        "pair_id": f"{sequence}:{index}",
                                        "fit_budget": budget,
                                        "method": method,
                                        "provenance": config["provenance"],
                                        "valid": True,
                                        "failure_reason": "",
                                        "ce95": 0.1 * improvement,
                                        "absolute_log_scale_ratio": 0.2 * improvement,
                                        "holdout_nll": 3.0 * improvement,
                                        "estimated_scale": 1.0,
                                        "coverage95": 0.95 - 0.1 * improvement,
                                        "calibration_time_ms": 0.1,
                                        "random_trace_median_relative_error": 0.02,
                                        "random_trace_p95_relative_error": 0.08,
                                        "random_trace_mae": 0.01,
                                        "random_gain_median_relative_error": 0.01,
                                        "random_gain_p95_relative_error": 0.02,
                                        "k16_vs_exact_scale_relative_error": 0.01,
                                        "projection_basis_time_ms": 0.1,
                                    }
                                )
                from experiments.public_module.result_validation import file_hash
                (directory / "summary.json").write_text(json.dumps({
                    "protocol_version": "v1.5", "profile": "formal",
                    "sequence": sequence, "provenance": config["provenance"],
                    "config_sha256": config["_config_sha256"],
                    "candidate_pair_count": 80, "result_csv_sha256": file_hash(path),
                }), encoding="utf-8")
            output = root / "aggregate"
            aggregate(config, input_root, output)
            gate = json.loads((output / "gate_review.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["status"], "PASS")
            self.assertTrue(all(gate["checks"].values()))


class RunComparisonTests(unittest.TestCase):
    def test_timing_is_excluded_but_scientific_values_are_checked(self) -> None:
        fieldnames = [
            "sequence",
            "pair_id",
            "fit_budget",
            "method",
            "ce95",
            "solver_nfev",
            "calibration_time_ms",
            "projection_basis_time_ms",
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.csv"
            right = root / "right.csv"
            for path, ce95, timing in ((left, 0.1, 1.0), (right, 0.1, 9.0)):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sequence": "V1_01_easy",
                            "pair_id": "pair-1",
                            "fit_budget": 16,
                            "method": "DUC-K16",
                            "ce95": ce95,
                            "solver_nfev": int(timing),
                            "calibration_time_ms": timing,
                            "projection_basis_time_ms": timing,
                        }
                    )
            self.assertEqual(compare(left, right, rtol=0.0, atol=0.0)["status"], "FAIL")
            self.assertEqual(
                compare(
                    left,
                    right,
                    rtol=0.0,
                    atol=0.0,
                    exclude_fields=("solver_nfev",),
                )["status"],
                "PASS",
            )

            with right.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["ce95"] = "0.2"
            with right.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(
                compare(
                    left,
                    right,
                    rtol=0.0,
                    atol=0.0,
                    exclude_fields=("solver_nfev",),
                )["status"],
                "FAIL",
            )


METRIC_FIELDS = [
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
]


if __name__ == "__main__":
    unittest.main()
