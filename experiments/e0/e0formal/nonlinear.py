from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import sqrt
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Optional, Union
import unicodedata

import numpy as np

from .authorization import FormalProcessSession
from .contract import FormalExecutionLocked, FrozenContract


class NonlinearProtocolError(RuntimeError):
    """Raised when the frozen nonlinear protocol cannot be followed exactly."""


class NonlinearInfrastructureError(RuntimeError):
    """依赖或运行环境不可用；不得记作科学数值失败。"""


@dataclass
class _PoseSolve:
    valid: bool
    parameters: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    failure_reasons: list[str]
    success: bool
    status: int
    message: str
    optimality: float
    normalized_stationarity: float
    stationarity_raw_gradient: tuple[float, ...]
    stationarity_column_norms: tuple[float, ...]
    stationarity_residual_norm: float
    cost: float
    nfev: int
    njev: Optional[int]
    rank: int
    rank_tolerance: float
    minimum_final_depth: float
    validity_checks: dict[str, bool]


def _as_config(config_or_contract: Any) -> Mapping[str, Any]:
    if hasattr(config_or_contract, "config"):
        config_or_contract = config_or_contract.config
    if not isinstance(config_or_contract, Mapping):
        raise TypeError("config must be a mapping or expose a mapping-valued .config attribute")
    return config_or_contract


def _formal_contract(config_or_contract: Any) -> Mapping[str, Any]:
    config = _as_config(config_or_contract)
    formal = config.get("formal_contract", config)
    if not isinstance(formal, Mapping):
        raise NonlinearProtocolError("formal_contract is missing or is not an object")
    return formal


def _normalise_json(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalise_json(item) for item in value]
    if isinstance(value, tuple):
        return [_normalise_json(item) for item in value]
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", str(key)): _normalise_json(item)
            for key, item in value.items()
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        _normalise_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def _derived_seed(
    formal: Mapping[str, Any], component: str, graph_id: int, subcomponent_id: str = "none"
) -> int:
    randomness = formal["randomness"]
    if component not in randomness["named_component_streams"]:
        raise NonlinearProtocolError(f"unregistered random component: {component}")
    specification = randomness["derived_seed"]
    payload_spec = specification["payload"]
    prefix = list(payload_spec["namespace_prefix"])
    setting_id = int(payload_spec["nonlinear_setting_id"])
    payload = prefix + [component, "nonlinear", setting_id, int(graph_id), subcomponent_id]
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    prefix_bytes = int(specification["digest_prefix_bytes"])
    return int.from_bytes(
        digest[:prefix_bytes],
        byteorder=str(specification["byte_order"]),
        signed=bool(specification["signed"]),
    )


def _rng_for(formal: Mapping[str, Any], component: str, graph_id: int) -> np.random.Generator:
    seed = _derived_seed(formal, component, graph_id)
    return np.random.Generator(np.random.PCG64(seed))


def _pose_vector(specification: Mapping[str, Any], *, initial: bool) -> np.ndarray:
    prefix = "initial" if initial else "true"
    rotation = np.deg2rad(np.asarray(specification[f"{prefix}_rotation_vector_degrees"], dtype=np.float64))
    translation = np.asarray(specification[f"{prefix}_translation_m"], dtype=np.float64)
    if rotation.shape != (3,) or translation.shape != (3,):
        raise NonlinearProtocolError("pose rotation and translation must each contain three coordinates")
    return np.concatenate([rotation, translation])


def _rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("rotation vector must contain three coordinates")
    theta_squared = float(vector @ vector)
    theta = float(np.sqrt(theta_squared))
    x, y, z = vector
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    if theta < 1e-8:
        coefficient_a = 1.0 - theta_squared / 6.0 + theta_squared * theta_squared / 120.0
        coefficient_b = 0.5 - theta_squared / 24.0 + theta_squared * theta_squared / 720.0
    else:
        coefficient_a = float(np.sin(theta) / theta)
        coefficient_b = float((1.0 - np.cos(theta)) / theta_squared)
    return np.eye(3, dtype=np.float64) + coefficient_a * skew + coefficient_b * (skew @ skew)


def project_points(
    world_points: np.ndarray,
    pose: np.ndarray,
    camera: Mapping[str, Any],
) -> np.ndarray:
    """Project known points using X_c = Exp(phi^) X_w + t."""
    points = np.asarray(world_points, dtype=np.float64)
    parameters = np.asarray(pose, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape (N, 3)")
    if parameters.shape != (6,):
        raise ValueError("pose must contain global rotvec then translation")
    rotation = _rotation_matrix(parameters[:3])
    camera_points = points @ rotation.T + parameters[3:][None, :]
    depth = camera_points[:, 2]
    if np.any(~np.isfinite(camera_points)) or np.any(np.abs(depth) <= np.finfo(np.float64).tiny):
        raise FloatingPointError("projection produced a non-finite or zero-depth point")
    pixels = np.empty((points.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = float(camera["fx_px"]) * camera_points[:, 0] / depth + float(camera["cx_px"])
    pixels[:, 1] = float(camera["fy_px"]) * camera_points[:, 1] / depth + float(camera["cy_px"])
    return pixels


def residual_vector(
    pose: np.ndarray,
    world_points: np.ndarray,
    observed_pixels: np.ndarray,
    camera: Mapping[str, Any],
) -> np.ndarray:
    """Return factor-id ordered (u, v) residuals with observed-predicted sign."""
    observed = np.asarray(observed_pixels, dtype=np.float64)
    predicted = project_points(world_points, pose, camera)
    if observed.shape != predicted.shape:
        raise ValueError("observed_pixels must match the projected point array")
    return (observed - predicted).reshape(-1)


def central_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    rotation_step: float,
    translation_step: float,
) -> np.ndarray:
    """Differentiate by direct addition in the six global pose coordinates."""
    x = np.asarray(parameters, dtype=np.float64)
    if x.shape != (6,):
        raise ValueError("the nonlinear pose has exactly six coordinates")
    base = np.asarray(function(x), dtype=np.float64)
    jacobian = np.empty((base.size, 6), dtype=np.float64)
    for coordinate in range(6):
        step = float(rotation_step if coordinate < 3 else translation_step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("central-difference steps must be positive and finite")
        plus = x.copy()
        minus = x.copy()
        plus[coordinate] += step
        minus[coordinate] -= step
        jacobian[:, coordinate] = (
            np.asarray(function(plus), dtype=np.float64)
            - np.asarray(function(minus), dtype=np.float64)
        ) / (2.0 * step)
    return jacobian


def _jacobian_rank(jacobian: np.ndarray) -> tuple[int, float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        return 0, float("nan")
    tolerance = (
        np.finfo(np.float64).eps
        * max(jacobian.shape)
        * float(singular_values[0])
    )
    return int(np.count_nonzero(singular_values > tolerance)), tolerance


def _column_scaled_stationarity(
    residual: np.ndarray, jacobian: np.ndarray
) -> tuple[float, tuple[float, ...], tuple[float, ...], float]:
    if (
        residual.ndim != 1
        or jacobian.ndim != 2
        or jacobian.shape[0] != residual.size
        or jacobian.shape[1] == 0
    ):
        raise NonlinearProtocolError("归一化驻点量的残差或Jacobian形状无效。")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
        return float("nan"), tuple(), tuple(), float("nan")
    floor = float(np.sqrt(np.finfo(np.float64).eps))
    raw_residual_norm = float(np.linalg.norm(residual))
    raw_column_norms = np.asarray(np.linalg.norm(jacobian, axis=0), dtype=np.float64)
    gradient = np.asarray(jacobian.T @ residual, dtype=np.float64)
    if (
        not np.isfinite(raw_residual_norm)
        or not np.all(np.isfinite(raw_column_norms))
        or not np.all(np.isfinite(gradient))
    ):
        return (
            float("nan"),
            tuple(float(value) for value in gradient),
            tuple(float(value) for value in raw_column_norms),
            raw_residual_norm,
        )
    if raw_residual_norm == 0.0 and np.all(gradient == 0.0):
        return (
            0.0,
            tuple(float(value) for value in gradient),
            tuple(float(value) for value in raw_column_norms),
            0.0,
        )
    denominators = np.maximum(raw_column_norms, floor) * max(raw_residual_norm, floor)
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        return (
            float("nan"),
            tuple(float(value) for value in gradient),
            tuple(float(value) for value in raw_column_norms),
            raw_residual_norm,
        )
    eta = float(np.max(np.abs(gradient) / denominators))
    return (
        eta,
        tuple(float(value) for value in gradient),
        tuple(float(value) for value in raw_column_norms),
        raw_residual_norm,
    )


def _load_least_squares() -> Callable[..., Any]:
    try:
        from scipy.optimize import least_squares
    except (ImportError, OSError) as error:
        raise NonlinearInfrastructureError(
            "SciPy optimize 不可用或其二进制依赖无法加载。"
        ) from error
    return least_squares


def _solve_pose(
    points: np.ndarray,
    observations: np.ndarray,
    initial: np.ndarray,
    nonlinear: Mapping[str, Any],
    *,
    least_squares_solver: Optional[Callable[..., Any]] = None,
) -> _PoseSolve:
    camera = nonlinear["camera"]
    solver = nonlinear["solver"]
    jacobian_spec = solver["jacobian"]

    def function(parameters: np.ndarray) -> np.ndarray:
        return residual_vector(parameters, points, observations, camera)

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        return central_difference_jacobian(
            function,
            parameters,
            float(jacobian_spec["rotation_absolute_step_rad"]),
            float(jacobian_spec["translation_absolute_step_m"]),
        )

    if least_squares_solver is None:
        least_squares_solver = _load_least_squares()

    try:
        result = least_squares_solver(
            function,
            np.asarray(initial, dtype=np.float64),
            jac=jacobian,
            method="trf",
            loss="linear",
            tr_solver="exact",
            x_scale="jac",
            ftol=float(solver["ftol"]),
            xtol=float(solver["xtol"]),
            gtol=float(solver["gtol"]),
            max_nfev=int(solver["max_nfev"]),
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        # 正式有效性检查必须在返回端点重新计算，不能依赖求解器缓存的
        # fun/jac。这样驻点量、秩检查和后续DPR使用完全相同的残差与
        # 固定绝对步长中心差分Jacobian。
        residual = np.asarray(function(parameters), dtype=np.float64)
        final_jacobian = np.asarray(jacobian(parameters), dtype=np.float64)
        rank, rank_tolerance = _jacobian_rank(final_jacobian)
        camera_points = (
            points @ _rotation_matrix(parameters[:3]).T
            + parameters[3:][None, :]
        )
        solver_success_pass = bool(result.success)
        validity = solver["validity"]
        (
            normalized_stationarity,
            stationarity_raw_gradient,
            stationarity_column_norms,
            stationarity_residual_norm,
        ) = _column_scaled_stationarity(
            residual, final_jacobian
        )
        normalized_rule = validity.get("normalized_stationarity")
        uses_normalized_stationarity = isinstance(normalized_rule, Mapping) or (
            "maximum_normalized_stationarity" in validity
        )
        if uses_normalized_stationarity:
            maximum_eta = (
                normalized_rule["maximum_eta"]
                if isinstance(normalized_rule, Mapping)
                else validity["maximum_normalized_stationarity"]
            )
            optimality_pass = bool(
                np.isfinite(normalized_stationarity)
                and normalized_stationarity
                <= float(maximum_eta)
            )
            optimality_check_name = "maximum_normalized_stationarity_pass"
            optimality_failure_reason = (
                "normalized_stationarity_above_frozen_maximum"
            )
        else:
            optimality_pass = bool(
                np.isfinite(float(result.optimality))
                and float(result.optimality)
                <= float(validity["maximum_optimality"])
            )
            optimality_check_name = "maximum_optimality_pass"
            optimality_failure_reason = "optimality_above_frozen_maximum"
        rank_pass = rank == int(solver["validity"]["required_jacobian_rank"])
        finite_pass = bool(
            np.all(np.isfinite(parameters))
            and np.all(np.isfinite(residual))
            and np.all(np.isfinite(final_jacobian))
        )
        positive_depth_pass = bool(
            np.all(np.isfinite(camera_points))
            and np.all(camera_points[:, 2] > 0.0)
        )
        minimum_final_depth = (
            float(np.min(camera_points[:, 2]))
            if camera_points.shape[0] and np.all(np.isfinite(camera_points[:, 2]))
            else float("nan")
        )
        validity_checks = {
            "solver_success_required_pass": solver_success_pass,
            optimality_check_name: optimality_pass,
            "required_jacobian_rank_pass": rank_pass,
            "finite_solution_residual_and_jacobian_pass": finite_pass,
            "all_final_depths_positive_pass": positive_depth_pass,
        }
        reasons: list[str] = []
        if not solver_success_pass:
            reasons.append("solver_not_successful")
        if not optimality_pass:
            reasons.append(optimality_failure_reason)
        if not rank_pass:
            reasons.append("jacobian_rank_not_six")
        if not finite_pass:
            reasons.append("nonfinite_solution_residual_or_jacobian")
        if not positive_depth_pass:
            reasons.append("nonpositive_final_depth")
        return _PoseSolve(
            valid=not reasons,
            parameters=parameters,
            residual=residual,
            jacobian=final_jacobian,
            failure_reasons=reasons,
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            optimality=float(result.optimality),
            normalized_stationarity=normalized_stationarity,
            stationarity_raw_gradient=stationarity_raw_gradient,
            stationarity_column_norms=stationarity_column_norms,
            stationarity_residual_norm=stationarity_residual_norm,
            cost=float(result.cost),
            nfev=int(result.nfev),
            njev=None if result.njev is None else int(result.njev),
            rank=rank,
            rank_tolerance=float(rank_tolerance),
            minimum_final_depth=minimum_final_depth,
            validity_checks=validity_checks,
        )
    except (
        NonlinearProtocolError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
    ) as error:
        return _PoseSolve(
            valid=False,
            parameters=np.full(6, np.nan, dtype=np.float64),
            residual=np.array([], dtype=np.float64),
            jacobian=np.empty((0, 6), dtype=np.float64),
            failure_reasons=[f"solver_exception:{type(error).__name__}"],
            success=False,
            status=-1,
            message=str(error),
            optimality=float("nan"),
            normalized_stationarity=float("nan"),
            stationarity_raw_gradient=tuple(),
            stationarity_column_norms=tuple(),
            stationarity_residual_norm=float("nan"),
            cost=float("nan"),
            nfev=0,
            njev=None,
            rank=0,
            rank_tolerance=float("nan"),
            minimum_final_depth=float("nan"),
            validity_checks={
                "solver_success_required_pass": False,
                "maximum_optimality_pass": False,
                "maximum_normalized_stationarity_pass": False,
                "required_jacobian_rank_pass": False,
                "finite_solution_residual_and_jacobian_pass": False,
                "all_final_depths_positive_pass": False,
            },
        )


def _generate_landmarks(
    formal: Mapping[str, Any], graph_id: int, nonlinear: Mapping[str, Any]
) -> np.ndarray:
    specification = nonlinear["landmarks"]
    coordinates = specification["coordinates_m"]
    acceptance = specification["acceptance"]
    count = int(specification["count_per_graph"])
    maximum_draws = int(acceptance["maximum_candidate_draws"])
    rng = _rng_for(formal, "nonlinear_landmarks", graph_id)
    low = np.array(
        [coordinates[axis]["minimum"] for axis in ("X", "Y", "Z")],
        dtype=np.float64,
    )
    high = np.array(
        [coordinates[axis]["maximum"] for axis in ("X", "Y", "Z")],
        dtype=np.float64,
    )
    candidates = low[None, :] + rng.random((maximum_draws, 3)) * (high - low)[None, :]
    true_pose = _pose_vector(nonlinear["pose"], initial=False)
    pixels = project_points(candidates, true_pose, nonlinear["camera"])
    rotation = _rotation_matrix(true_pose[:3])
    depths = (candidates @ rotation.T + true_pose[3:][None, :])[:, 2]
    margin = float(acceptance["minimum_image_border_margin_px"])
    camera = nonlinear["camera"]
    accepted = (
        (depths >= float(acceptance["minimum_true_camera_depth_m"]))
        & (pixels[:, 0] >= margin)
        & (pixels[:, 0] <= float(camera["image_width_px"]) - margin)
        & (pixels[:, 1] >= margin)
        & (pixels[:, 1] <= float(camera["image_height_px"]) - margin)
        & np.all(np.isfinite(pixels), axis=1)
    )
    selected = candidates[np.flatnonzero(accepted)[:count]]
    if selected.shape[0] != count:
        raise NonlinearProtocolError("frozen landmark rejection sampling did not accept 30 points")
    return selected


def select_nonlinear_lofo_factors(graph_id: int, count: int = 8, total: int = 30) -> list[int]:
    if count < 0 or count > total:
        raise ValueError("LOFO selection count must lie between zero and the factor count")
    ranked: list[tuple[bytes, int]] = []
    for factor_id in range(total):
        payload = ["DUC-VIO", "E0", "v1.4", "nonlinear_lofo_factor", int(graph_id), factor_id]
        ranked.append((hashlib.sha256(_canonical_json_bytes(payload)).digest(), factor_id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [factor_id for _, factor_id in ranked[:count]]


def _full_block_dpr(
    residual: np.ndarray,
    jacobian: np.ndarray,
    formal: Mapping[str, Any],
) -> tuple[np.ndarray, list[np.ndarray], int]:
    numerics = formal["numerics"]
    leverage_spec = numerics["full_block_leverage"]
    epsilon = float(numerics["spectral_clip_epsilon"])
    hessian = jacobian.T @ jacobian
    if not np.all(np.isfinite(hessian)):
        raise FloatingPointError("statistical Hessian is non-finite")
    leverage_blocks: list[np.ndarray] = []
    corrected = np.empty_like(residual, dtype=np.float64)
    clipped_count = 0
    gross = leverage_spec["gross_eigenvalue_valid_range"]
    for factor_id in range(residual.size // 2):
        factor_slice = slice(2 * factor_id, 2 * factor_id + 2)
        block_jacobian = jacobian[factor_slice]
        leverage = block_jacobian @ np.linalg.solve(hessian, block_jacobian.T)
        symmetric = 0.5 * (leverage + leverage.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        if np.any(eigenvalues < float(gross["minimum"])) or np.any(
            eigenvalues > float(gross["maximum"])
        ):
            raise FloatingPointError("block leverage eigenvalue is outside the frozen gross range")
        clipped = np.clip(eigenvalues, 0.0, 1.0 - epsilon)
        clipped_count += int(np.count_nonzero(clipped != eigenvalues))
        inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(1.0 - clipped))) @ eigenvectors.T
        corrected[factor_slice] = inverse_sqrt @ residual[factor_slice]
        leverage_blocks.append(symmetric)
    if not np.all(np.isfinite(corrected)):
        raise FloatingPointError("full-block DPR produced a non-finite residual")
    return corrected, leverage_blocks, clipped_count


def _endpoint(residual: np.ndarray, holdout: np.ndarray, probability: float) -> dict[str, float]:
    scale = float(np.sum(np.square(residual)) / residual.size)
    # Chi-square with two degrees of freedom has CDF 1-exp(-x/2).
    threshold = float(-2.0 * np.log1p(-float(probability))) * scale
    events = np.sum(np.square(holdout), axis=1) <= threshold
    coverage = float(np.mean(events))
    return {
        "scale": scale,
        "scale_error": abs(scale - 1.0),
        "coverage95": coverage,
        "ce95": abs(coverage - 0.95),
    }


def _json_number(
    value: Optional[Union[float, int]]
) -> Optional[Union[float, int]]:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def run_nonlinear_graph(
    config: Any,
    graph_id: int,
    *,
    holdout_samples: Optional[int] = None,
    lofo_factors: Optional[int] = None,
    process_session: Optional[FormalProcessSession] = None,
) -> dict[str, Any]:
    """Run one graph; frozen scale requires an authorized FrozenContract."""
    formal = _formal_contract(config)
    nonlinear = formal["nonlinear"]
    expected_graphs = int(nonlinear["graphs"])
    if type(graph_id) is not int:
        raise ValueError("graph_id 必须是规范JSON整数。")
    if graph_id < 0 or graph_id >= expected_graphs:
        raise ValueError(f"graph_id must be in [0, {expected_graphs - 1}]")
    frozen_holdout = int(nonlinear["noise_and_holdout"]["holdout"]["samples_per_graph"])
    frozen_lofo = int(nonlinear["lofo_factors_per_graph"])
    if holdout_samples is not None and type(holdout_samples) is not int:
        raise ValueError("holdout_samples 必须是JSON整数，禁止静默截断。")
    if lofo_factors is not None and type(lofo_factors) is not int:
        raise ValueError("lofo_factors 必须是JSON整数，禁止静默截断。")
    actual_holdout = frozen_holdout if holdout_samples is None else holdout_samples
    actual_lofo = frozen_lofo if lofo_factors is None else lofo_factors
    if actual_holdout <= 0:
        raise ValueError("holdout_samples must be positive")
    if actual_lofo < 0 or actual_lofo > int(nonlinear["landmarks"]["count_per_graph"]):
        raise ValueError("lofo_factors is outside the frozen factor range")
    frozen_scale = actual_holdout == frozen_holdout and actual_lofo == frozen_lofo
    actually_reduced = bool(
        actual_holdout <= frozen_holdout
        and actual_lofo <= frozen_lofo
        and not frozen_scale
    )
    if not frozen_scale and not actually_reduced:
        raise ValueError(
            "dev_toy overrides may only reduce the frozen holdout or LOFO scale"
        )
    if frozen_scale:
        if not isinstance(config, FrozenContract):
            raise FormalExecutionLocked(
                "frozen-scale nonlinear execution requires a validated FrozenContract"
            )
    development_override = actually_reduced
    provenance = "dev_toy" if development_override else "designed_synthetic_e0"
    stable_id = "v1.4:E0:N:g{:03d}".format(graph_id)
    if frozen_scale:
        if not isinstance(process_session, FormalProcessSession):
            raise FormalExecutionLocked(
                "正式非线性worker必须使用本进程完成预热后的FormalProcessSession。"
            )
        process_session.verify(config, stable_id)

    base: dict[str, Any] = {
        "protocol_version": "v1.4",
        "experiment_id": "E0_nonlinear",
        "stable_id": stable_id,
        "graph_id": int(graph_id),
        "provenance": provenance,
        "dataset_origin": "designed_synthetic",
        "run_status": "DEV_TOY" if development_override else "FORMAL_CANDIDATE_STAGE_REVIEW_REQUIRED",
        "formal_evidence_candidate": not development_override,
        "evidence_eligibility": (
            "NOT_ELIGIBLE_DEVELOPMENT"
            if development_override
            else "PENDING_STAGE_REVIEW"
        ),
        "science_claim_eligible": False,
        "stage_review_required": True,
        "development_overrides": {
            "holdout_samples": actual_holdout if actual_holdout != frozen_holdout else None,
            "lofo_factors": actual_lofo if actual_lofo != frozen_lofo else None,
        },
        "holdout_samples": actual_holdout,
        "lofo_factors_requested": actual_lofo,
        "raw_valid": False,
        "dpr_valid": False,
        "common_valid": False,
        "raw_failure_reasons": [],
        "dpr_failure_reasons": [],
        "lofo_valid": False,
        "lofo_failure_reasons": [],
        "selected_lofo_factor_ids": [],
        "lofo_factor_rows": [],
        "formal_execution_unlock": False,
        "seeds": {
            component: _derived_seed(formal, component, graph_id)
            for component in (
                "nonlinear_landmarks",
                "nonlinear_training_noise",
                "nonlinear_holdout",
            )
        },
    }

    try:
        points = _generate_landmarks(formal, graph_id, nonlinear)
        true_pose = _pose_vector(nonlinear["pose"], initial=False)
        initial_pose = _pose_vector(nonlinear["pose"], initial=True)
        truth = project_points(points, true_pose, nonlinear["camera"])
        noise_spec = nonlinear["noise_and_holdout"]["training_noise"]
        covariance = np.asarray(noise_spec["covariance_px2"], dtype=np.float64)
        mean = np.asarray(noise_spec["mean_px"], dtype=np.float64)
        training_rng = _rng_for(formal, "nonlinear_training_noise", graph_id)
        training_noise = training_rng.multivariate_normal(mean, covariance, size=points.shape[0])
        observations = truth + training_noise
        holdout_rng = _rng_for(formal, "nonlinear_holdout", graph_id)
        holdout = holdout_rng.multivariate_normal(mean, covariance, size=actual_holdout)
    except (
        NonlinearProtocolError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
    ) as error:
        reason = f"graph_construction_exception:{type(error).__name__}"
        base["raw_failure_reasons"] = [reason]
        base["dpr_failure_reasons"] = [reason]
        base["lofo_failure_reasons"] = [reason]
        return base

    # 依赖加载是进程准备，不属于冻结的基础求解器计时边界。
    least_squares_solver = _load_least_squares()
    solver_start = perf_counter_ns()
    full_solve = _solve_pose(
        points,
        observations,
        initial_pose,
        nonlinear,
        least_squares_solver=least_squares_solver,
    )
    solver_elapsed = perf_counter_ns() - solver_start
    base["landmark_count"] = int(points.shape[0])
    base["state_dimension"] = 6
    base["residual_dimension"] = int(points.shape[0] * 2)
    base["solver"] = {
        "success": full_solve.success,
        "status": full_solve.status,
        "message": full_solve.message,
        "termination_reason": full_solve.message,
        "optimality": _json_number(full_solve.optimality),
        "normalized_stationarity": _json_number(
            full_solve.normalized_stationarity
        ),
        "stationarity_raw_gradient": [
            _json_number(value) for value in full_solve.stationarity_raw_gradient
        ],
        "stationarity_column_norms": [
            _json_number(value) for value in full_solve.stationarity_column_norms
        ],
        "stationarity_residual_norm": _json_number(
            full_solve.stationarity_residual_norm
        ),
        "cost": _json_number(full_solve.cost),
        "nfev": full_solve.nfev,
        "njev": full_solve.njev,
        "jacobian_rank": full_solve.rank,
        "rank_tolerance": _json_number(full_solve.rank_tolerance),
        "minimum_final_depth_m": _json_number(full_solve.minimum_final_depth),
        "validity_checks": dict(full_solve.validity_checks),
        "parameters": [
            _json_number(value) for value in full_solve.parameters.tolist()
        ],
    }
    base["timing_ns"] = {
        "base_full_graph_solver": int(solver_elapsed),
        "postfit_dpr_only": None,
        "lofo_resolves": None,
    }
    if not full_solve.valid:
        base["raw_failure_reasons"] = list(full_solve.failure_reasons)
        base["dpr_failure_reasons"] = list(full_solve.failure_reasons)
        base["lofo_failure_reasons"] = ["full_graph_invalid"]
        return base

    raw_endpoint = _endpoint(
        full_solve.residual,
        holdout,
        float(nonlinear["noise_and_holdout"]["holdout"]["coverage_quantile_probability"]),
    )
    raw_valid = all(np.isfinite(value) for value in raw_endpoint.values())
    base.update(
        {f"raw_{key}": _json_number(value) for key, value in raw_endpoint.items()}
    )
    base["raw_valid"] = bool(raw_valid)
    if not raw_valid:
        base["raw_failure_reasons"] = ["raw_endpoint_nonfinite"]

    dpr_start = perf_counter_ns()
    leverage_blocks: list[np.ndarray] = []
    try:
        dpr_residual, leverage_blocks, clipped_count = _full_block_dpr(
            full_solve.residual, full_solve.jacobian, formal
        )
        base["timing_ns"]["postfit_dpr_only"] = int(
            perf_counter_ns() - dpr_start
        )
        dpr_endpoint = _endpoint(
            dpr_residual,
            holdout,
            float(nonlinear["noise_and_holdout"]["holdout"]["coverage_quantile_probability"]),
        )
        dpr_endpoint_finite = all(
            np.isfinite(value) for value in dpr_endpoint.values()
        )
        dpr_valid = raw_valid and dpr_endpoint_finite
        base.update(
            {
                f"dpr_{key}": _json_number(value)
                for key, value in dpr_endpoint.items()
            }
        )
        base["dpr_clipped_eigenvalue_count"] = int(clipped_count)
        if not dpr_valid:
            base["dpr_failure_reasons"] = []
            if not raw_valid:
                base["dpr_failure_reasons"].append("raw_endpoint_invalid")
            if not dpr_endpoint_finite:
                base["dpr_failure_reasons"].append("dpr_endpoint_nonfinite")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        dpr_valid = False
        dpr_residual = np.array([], dtype=np.float64)
        base["dpr_failure_reasons"] = [f"dpr_exception:{type(error).__name__}"]
        if base["timing_ns"]["postfit_dpr_only"] is None:
            base["timing_ns"]["postfit_dpr_only"] = int(
                perf_counter_ns() - dpr_start
            )
    base["dpr_valid"] = bool(dpr_valid)
    base["common_valid"] = bool(raw_valid and dpr_valid)

    selected_factors = select_nonlinear_lofo_factors(
        graph_id, actual_lofo, int(nonlinear["landmarks"]["count_per_graph"])
    )
    base["selected_lofo_factor_ids"] = selected_factors
    lofo_rows: list[dict[str, Any]] = []
    lofo_reasons: list[str] = []
    lofo_solver_elapsed_ns = 0
    if not leverage_blocks:
        lofo_reasons.append("full_graph_leverage_unavailable")
    else:
        for factor_id in selected_factors:
            mask = np.ones(points.shape[0], dtype=bool)
            mask[factor_id] = False
            lofo_solve_start = perf_counter_ns()
            deleted = _solve_pose(
                points[mask],
                observations[mask],
                full_solve.parameters,
                nonlinear,
                least_squares_solver=least_squares_solver,
            )
            lofo_solver_elapsed_ns += perf_counter_ns() - lofo_solve_start
            row: dict[str, Any] = {
                "graph_id": int(graph_id),
                "factor_id": int(factor_id),
                "valid": False,
                "failure_reasons": list(deleted.failure_reasons),
                "solver": {
                    "success": deleted.success,
                    "status": deleted.status,
                    "termination_reason": deleted.message,
                    "optimality": _json_number(deleted.optimality),
                    "normalized_stationarity": _json_number(
                        deleted.normalized_stationarity
                    ),
                    "stationarity_raw_gradient": [
                        _json_number(value)
                        for value in deleted.stationarity_raw_gradient
                    ],
                    "stationarity_column_norms": [
                        _json_number(value)
                        for value in deleted.stationarity_column_norms
                    ],
                    "stationarity_residual_norm": _json_number(
                        deleted.stationarity_residual_norm
                    ),
                    "nfev": deleted.nfev,
                    "njev": deleted.njev,
                    "jacobian_rank": deleted.rank,
                    "rank_tolerance": _json_number(deleted.rank_tolerance),
                    "minimum_final_depth_m": _json_number(
                        deleted.minimum_final_depth
                    ),
                    "validity_checks": dict(deleted.validity_checks),
                },
                "raw_error": None,
                "press_error": None,
            }
            if deleted.valid:
                try:
                    exact_deleted = (
                        observations[factor_id]
                        - project_points(
                            points[[factor_id]],
                            deleted.parameters,
                            nonlinear["camera"],
                        )[0]
                    )
                except (
                    NonlinearProtocolError,
                    ValueError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                    OverflowError,
                ) as error:
                    row["failure_reasons"] = [
                        f"deleted_factor_projection_exception:{type(error).__name__}"
                    ]
                else:
                    factor_slice = slice(2 * factor_id, 2 * factor_id + 2)
                    raw_factor = full_solve.residual[factor_slice]
                    try:
                        press = np.linalg.solve(
                            np.eye(2, dtype=np.float64)
                            - leverage_blocks[factor_id],
                            raw_factor,
                        )
                        raw_error = float(
                            np.linalg.norm(raw_factor - exact_deleted) / sqrt(2.0)
                        )
                        press_error = float(
                            np.linalg.norm(press - exact_deleted) / sqrt(2.0)
                        )
                        valid = bool(
                            np.isfinite(raw_error) and np.isfinite(press_error)
                        )
                        row.update(
                            {
                                "valid": valid,
                                "failure_reasons": (
                                    [] if valid else ["nonfinite_lofo_error"]
                                ),
                                "raw_error": raw_error if valid else None,
                                "press_error": press_error if valid else None,
                            }
                        )
                    except (
                        ValueError,
                        FloatingPointError,
                        np.linalg.LinAlgError,
                    ) as error:
                        row["failure_reasons"] = [
                            f"press_exception:{type(error).__name__}"
                        ]
            if not row["valid"]:
                lofo_reasons.append(f"factor_{factor_id}_invalid")
            lofo_rows.append(row)
    base["timing_ns"]["lofo_resolves"] = int(lofo_solver_elapsed_ns)
    lofo_valid = actual_lofo > 0 and len(lofo_rows) == actual_lofo and not lofo_reasons
    base["lofo_factor_rows"] = lofo_rows
    base["lofo_valid"] = bool(lofo_valid)
    base["lofo_failure_reasons"] = lofo_reasons
    if lofo_valid:
        raw_errors = np.asarray([row["raw_error"] for row in lofo_rows], dtype=np.float64)
        press_errors = np.asarray([row["press_error"] for row in lofo_rows], dtype=np.float64)
        raw_median = float(np.median(raw_errors))
        press_median = float(np.median(press_errors))
        base["lofo_raw_graph_median_error"] = raw_median
        base["lofo_press_graph_median_error"] = press_median
        base["lofo_press_strictly_better"] = bool(press_median < raw_median)
    else:
        base["lofo_raw_graph_median_error"] = None
        base["lofo_press_graph_median_error"] = None
        base["lofo_press_strictly_better"] = False
    return base
