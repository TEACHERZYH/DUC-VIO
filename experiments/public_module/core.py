from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2


def derived_seed(master_seed: int, *parts: object) -> int:
    payload = json.dumps(
        [int(master_seed), *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def deterministic_permutation(length: int, master_seed: int, *parts: object) -> np.ndarray:
    rng = np.random.default_rng(derived_seed(master_seed, *parts))
    return rng.permutation(length)


def project_points(parameters: np.ndarray, points_3d: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(parameters[:3], dtype=np.float64))
    translation = np.asarray(parameters[3:6], dtype=np.float64).reshape(3)
    transformed = (rotation @ np.asarray(points_3d, dtype=np.float64).T).T + translation
    depth = transformed[:, 2]
    predicted = np.empty((len(transformed), 2), dtype=np.float64)
    predicted[:, 0] = camera[0, 0] * transformed[:, 0] / depth + camera[0, 2]
    predicted[:, 1] = camera[1, 1] * transformed[:, 1] / depth + camera[1, 2]
    return predicted, depth


def reprojection_residual(
    parameters: np.ndarray,
    points_3d: np.ndarray,
    observations_2d: np.ndarray,
    camera: np.ndarray,
) -> np.ndarray:
    predicted, _ = project_points(parameters, points_3d, camera)
    return (np.asarray(observations_2d, dtype=np.float64) - predicted).reshape(-1)


def stationarity_eta(jacobian: np.ndarray, residual: np.ndarray) -> float:
    gradient = jacobian.T @ residual
    residual_norm = float(np.linalg.norm(residual))
    column_norms = np.linalg.norm(jacobian, axis=0)
    floor = float(np.sqrt(np.finfo(float).eps))
    if residual_norm == 0.0 and np.linalg.norm(gradient) == 0.0:
        return 0.0
    denominators = np.maximum(column_norms, floor) * max(residual_norm, floor)
    return float(np.max(np.abs(gradient) / denominators))


@dataclass(frozen=True)
class PoseSolution:
    success: bool
    reason: str
    parameters: np.ndarray | None
    residual: np.ndarray | None
    jacobian: np.ndarray | None
    rank: int
    eta: float | None
    nfev: int
    fit_min_depth_m: float | None
    holdout_min_depth_m: float | None


def solve_relative_pose(
    fit_points_3d: np.ndarray,
    fit_observations_2d: np.ndarray,
    holdout_points_3d: np.ndarray,
    camera: np.ndarray,
    solver_config: dict[str, Any],
) -> PoseSolution:
    fit_points = np.asarray(fit_points_3d, dtype=np.float64)
    fit_observations = np.asarray(fit_observations_2d, dtype=np.float64)
    if fit_points.ndim != 2 or fit_points.shape[1] != 3 or len(fit_points) < 6:
        return PoseSolution(False, "INVALID_FIT_SHAPE", None, None, None, 0, None, 0, None, None)
    if fit_observations.shape != (len(fit_points), 2):
        return PoseSolution(False, "INVALID_OBSERVATION_SHAPE", None, None, None, 0, None, 0, None, None)

    ok, rvec, tvec = cv2.solvePnP(
        fit_points,
        fit_observations,
        camera,
        np.zeros(4, dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return PoseSolution(False, "EPNP_FAILED", None, None, None, 0, None, 0, None, None)
    initial = np.concatenate([rvec.reshape(3), tvec.reshape(3)]).astype(np.float64)
    if not np.all(np.isfinite(initial)):
        return PoseSolution(False, "EPNP_NONFINITE", None, None, None, 0, None, 0, None, None)

    result = least_squares(
        reprojection_residual,
        initial,
        args=(fit_points, fit_observations, camera),
        method="trf",
        loss="linear",
        jac="3-point",
        max_nfev=int(solver_config["max_nfev"]),
        ftol=float(solver_config["ftol"]),
        xtol=float(solver_config["xtol"]),
        gtol=float(solver_config["gtol"]),
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    residual = reprojection_residual(parameters, fit_points, fit_observations, camera)
    jacobian = np.asarray(result.jac, dtype=np.float64)
    if not (
        np.all(np.isfinite(parameters))
        and np.all(np.isfinite(residual))
        and np.all(np.isfinite(jacobian))
    ):
        return PoseSolution(False, "NONFINITE_SOLUTION", None, None, None, 0, None, int(result.nfev), None, None)

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    eta = stationarity_eta(jacobian, residual)
    _, fit_depth = project_points(parameters, fit_points, camera)
    _, holdout_depth = project_points(parameters, np.asarray(holdout_points_3d, dtype=np.float64), camera)
    fit_min_depth = float(np.min(fit_depth))
    holdout_min_depth = float(np.min(holdout_depth))

    if not result.success:
        reason = "LEAST_SQUARES_FAILED"
    elif rank != int(solver_config["required_rank"]):
        reason = "JACOBIAN_RANK_DEFICIENT"
    elif eta > float(solver_config["stationarity_eta_max"]):
        reason = "STATIONARITY_FAILED"
    elif fit_min_depth <= 0.0 or holdout_min_depth <= 0.0:
        reason = "NONPOSITIVE_TARGET_DEPTH"
    else:
        reason = "OK"
    return PoseSolution(
        reason == "OK",
        reason,
        parameters,
        residual,
        jacobian,
        rank,
        eta,
        int(result.nfev),
        fit_min_depth,
        holdout_min_depth,
    )


@dataclass(frozen=True)
class ScaleResult:
    scales: dict[str, float]
    timings_ms: dict[str, float]
    diagnostics: dict[str, float | int]


def compute_scale_estimates(
    residual: np.ndarray,
    jacobian: np.ndarray,
    k: int,
    epsilon: float,
    master_seed: int,
    seed_parts: tuple[object, ...],
) -> ScaleResult:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    total_dimension, parameter_dimension = jacobian.shape
    if residual.size != total_dimension or total_dimension % 2:
        raise ValueError("残差和 Jacobian 维度无效")
    if total_dimension <= parameter_dimension:
        raise ValueError("Global-DoF 分母非正")

    start = perf_counter_ns()
    raw_scale = float(np.mean(residual * residual))
    raw_ms = (perf_counter_ns() - start) / 1e6

    start = perf_counter_ns()
    global_dof = float(np.dot(residual, residual) / (total_dimension - parameter_dimension))
    dof_ms = (perf_counter_ns() - start) / 1e6

    start = perf_counter_ns()
    q_design, r_design = np.linalg.qr(jacobian, mode="reduced")
    qr_ms = (perf_counter_ns() - start) / 1e6
    diagonal = np.abs(np.diag(r_design))
    tolerance = np.finfo(float).eps * max(jacobian.shape) * diagonal.max()
    if int(np.sum(diagonal > tolerance)) != parameter_dimension:
        raise ValueError("Jacobian 不满列秩")
    blocks = [slice(i, i + 2) for i in range(0, total_dimension, 2)]

    start = perf_counter_ns()
    exact_blocks = [q_design[block] @ q_design[block].T for block in blocks]
    exact_traces = np.asarray([np.trace(block) for block in exact_blocks], dtype=np.float64)
    exact_energy = 0.0
    for factor_slice, leverage in zip(blocks, exact_blocks, strict=True):
        symmetric = 0.5 * (leverage + leverage.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        if np.min(eigenvalues) < -1e-10 or np.max(eigenvalues) > 1.0 + 1e-10:
            raise ValueError("精确块杠杆超出理论范围")
        eigenvalues = np.clip(eigenvalues, 0.0, 1.0 - epsilon)
        inverse = (eigenvectors * (1.0 / (1.0 - eigenvalues))) @ eigenvectors.T
        factor_residual = residual[factor_slice]
        exact_energy += float(factor_residual @ inverse @ factor_residual)
    exact_scale = exact_energy / total_dimension
    exact_ms = (perf_counter_ns() - start) / 1e6

    start = perf_counter_ns()
    trace_rng = np.random.default_rng(
        derived_seed(master_seed, "public_module_trace", *seed_parts)
    )
    probes = trace_rng.choice(np.asarray([-1.0, 1.0]), size=(total_dimension, k))
    projected = q_design @ (q_design.T @ probes)
    random_traces_raw = np.asarray(
        [np.sum(probes[block] * projected[block]) / k for block in blocks],
        dtype=np.float64,
    )
    random_traces = np.clip(random_traces_raw, 0.0, 2.0 * (1.0 - epsilon))
    k16_energy = 0.0
    for factor_slice, trace in zip(blocks, random_traces, strict=True):
        mean_leverage = float(trace / 2.0)
        k16_energy += float(np.dot(residual[factor_slice], residual[factor_slice]) / (1.0 - mean_leverage))
    k16_scale = k16_energy / total_dimension
    k16_ms = (perf_counter_ns() - start) / 1e6

    start = perf_counter_ns()
    permutation_rng = np.random.default_rng(
        derived_seed(master_seed, "public_module_permutation", *seed_parts)
    )
    permuted_traces = random_traces[permutation_rng.permutation(len(random_traces))]
    permuted_energy = 0.0
    for factor_slice, trace in zip(blocks, permuted_traces, strict=True):
        mean_leverage = float(trace / 2.0)
        permuted_energy += float(np.dot(residual[factor_slice], residual[factor_slice]) / (1.0 - mean_leverage))
    permuted_scale = permuted_energy / total_dimension
    permuted_ms = (perf_counter_ns() - start) / 1e6

    denominator = np.maximum(np.abs(exact_traces), 1e-12)
    relative_trace_error = np.abs(random_traces - exact_traces) / denominator
    exact_mean_leverage = np.clip(exact_traces / 2.0, 0.0, 1.0 - epsilon)
    random_mean_leverage = random_traces / 2.0
    exact_gain = 1.0 / np.sqrt(1.0 - exact_mean_leverage)
    random_gain = 1.0 / np.sqrt(1.0 - random_mean_leverage)
    relative_gain_error = np.abs(random_gain / exact_gain - 1.0)
    scales = {
        "Raw": raw_scale,
        "Global-DoF": global_dof,
        "DPR-Exact-Block": float(exact_scale),
        "DUC-K16": float(k16_scale),
        "Permuted-K16": float(permuted_scale),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in scales.values()):
        raise ValueError("尺度估计必须为有限正数")
    return ScaleResult(
        scales=scales,
        timings_ms={
            "Raw": raw_ms,
            "Global-DoF": dof_ms,
            "DPR-Exact-Block": exact_ms,
            "DUC-K16": k16_ms,
            "Permuted-K16": permuted_ms,
        },
        diagnostics={
            "projection_basis_time_ms": qr_ms,
            "exact_trace_sum": float(np.sum(exact_traces)),
            "random_trace_raw_negative_count": int(np.sum(random_traces_raw < 0.0)),
            "random_trace_raw_upper_count": int(np.sum(random_traces_raw > 2.0)),
            "random_trace_mae": float(np.mean(np.abs(random_traces - exact_traces))),
            "random_trace_median_relative_error": float(np.median(relative_trace_error)),
            "random_trace_p95_relative_error": float(np.quantile(relative_trace_error, 0.95, method="linear")),
            "random_gain_median_relative_error": float(np.median(relative_gain_error)),
            "random_gain_p95_relative_error": float(np.quantile(relative_gain_error, 0.95, method="linear")),
            "k16_vs_exact_scale_relative_error": abs(float(k16_scale / exact_scale - 1.0)),
        },
    )


def evaluate_holdout(
    holdout_residual: np.ndarray,
    scale: float,
    coverage_probability: float = 0.95,
) -> dict[str, float]:
    values = np.asarray(holdout_residual, dtype=np.float64).reshape(-1, 2)
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("预留残差无效")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("尺度必须为有限正数")
    squared_norm = np.sum(values * values, axis=1)
    threshold = float(chi2.ppf(coverage_probability, df=2)) * scale
    coverage = float(np.mean(squared_norm <= threshold))
    holdout_scale = float(np.mean(values * values))
    if holdout_scale <= 0.0:
        raise ValueError("预留方差必须为正")
    negative_log_likelihood = float(
        np.mean(np.log(2.0 * np.pi * scale) + squared_norm / (2.0 * scale))
    )
    return {
        "coverage95": coverage,
        "ce95": abs(coverage - coverage_probability),
        "holdout_scale": holdout_scale,
        "absolute_log_scale_ratio": abs(float(np.log(scale / holdout_scale))),
        "holdout_nll": negative_log_likelihood,
    }
