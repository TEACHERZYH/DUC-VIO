from __future__ import annotations

"""E0 v1.4 formal linear mechanism experiment.

The public entry point is :func:`run_linear_graph`.  One call evaluates one
frozen graph identity and returns a JSON-serialisable graph pack.  Development
overrides are deliberately labelled ``dev_toy`` and cannot enter scientific
aggregation.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from time import perf_counter_ns
from typing import Any, Mapping, Optional, Sequence, Union
import unicodedata

import numpy as np

from .identity import FormalInstanceID
from .contract import FormalExecutionLocked, FrozenContract
from .authorization import BatchAuthorization, FormalProcessSession
from .randomness import derive_seed as _formal_derive_seed
from .randomness import rng_for as _formal_rng_for


PROTOCOL_VERSION = "v1.4"
FORMAL_PROVENANCE = "designed_synthetic_e0"
DEV_PROVENANCE = "dev_toy"

# Frozen 0.95 chi-square quantiles for the only linear/sequential block
# dimensions in v1.4.  Keeping these constants here avoids adding a SciPy
# runtime dependency on the Python 3.9 remote environment.
_CHI_SQUARE_95 = {
    2: 5.991464547107979,
    6: 12.591587243743977,
    15: 24.995790139728616,
}


class LinearNumericalFailure(RuntimeError):
    """A recorded numerical failure for one frozen graph identity."""


@dataclass(frozen=True)
class LinearSetting:
    setting_id: int
    block_dimension: int
    covariance_multiplier: float
    mean_scalar_leverage: float
    state_dimension: int

    def to_dict(self) -> dict[str, Union[int, float]]:
        return {
            "setting_id": self.setting_id,
            "block_dimension": self.block_dimension,
            "covariance_multiplier": self.covariance_multiplier,
            "mean_scalar_leverage": self.mean_scalar_leverage,
            "state_dimension": self.state_dimension,
        }


@dataclass(frozen=True)
class _LeverageBlock:
    matrix: np.ndarray
    trace: float
    inverse_sqrt_identity_minus: np.ndarray
    clipped_eigenvalue_count: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float


def _config_mapping(config_or_contract: Any) -> Mapping[str, Any]:
    candidate = getattr(config_or_contract, "config", config_or_contract)
    if not isinstance(candidate, Mapping):
        raise TypeError("config_or_contract 必须提供配置对象。")
    return candidate


def _contract(config: Any) -> Mapping[str, Any]:
    config_mapping = _config_mapping(config)
    contract = config_mapping.get("formal_contract", config_mapping)
    if not isinstance(contract, Mapping):
        raise TypeError("formal_contract 必须是字典。")
    return contract


def enumerate_linear_settings(config: Mapping[str, Any]) -> list[LinearSetting]:
    """Enumerate the 27 frozen settings; the last axis varies fastest."""

    linear = _contract(config)["linear"]
    settings: list[LinearSetting] = []
    setting_id = 0
    state_map = linear["state_dimension_by_mean_leverage"]
    for block_dimension in linear["block_dimensions"]:
        for covariance_multiplier in linear["true_noise_multipliers"]:
            for mean_leverage in linear["mean_scalar_leverages"]:
                key = str(float(mean_leverage))
                if key not in state_map:
                    key = str(mean_leverage)
                settings.append(
                    LinearSetting(
                        setting_id=setting_id,
                        block_dimension=int(block_dimension),
                        covariance_multiplier=float(covariance_multiplier),
                        mean_scalar_leverage=float(mean_leverage),
                        state_dimension=int(state_map[key]),
                    )
                )
                setting_id += 1
    if len(settings) != int(linear["expected_settings"]):
        raise ValueError("线性设置数与冻结配置不一致。")
    return settings


def _require_formal_source_authorized(
    config_or_contract: Any,
    process_session: Optional[FormalProcessSession],
    stable_id: str,
) -> None:
    if not isinstance(config_or_contract, FrozenContract):
        raise FormalExecutionLocked(
            "正式 E0 入口必须使用已验证的 FrozenContract；"
            "裸 config 只允许显式的 dev_toy 缩减检查。"
        )
    if not isinstance(process_session, FormalProcessSession):
        raise FormalExecutionLocked(
            "正式线性worker必须使用本进程完成预热后的FormalProcessSession。"
        )
    process_session.verify(config_or_contract, stable_id)


def _canonical_json_bytes(payload: Sequence[Any]) -> bytes:
    text = json.dumps(
        list(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return unicodedata.normalize("NFC", text).encode("utf-8")


def _derived_seed(
    config: Mapping[str, Any],
    component: str,
    instance_kind: str,
    setting_id: int,
    instance_id: int,
    subcomponent_id: str = "none",
) -> int:
    identity = FormalInstanceID(instance_kind, int(setting_id), int(instance_id))
    return int(
        _formal_derive_seed(config, component, identity, str(subcomponent_id))
    )


def _rng(
    config: Mapping[str, Any],
    component: str,
    instance_kind: str,
    setting_id: int,
    instance_id: int,
    subcomponent_id: str = "none",
) -> np.random.Generator:
    identity = FormalInstanceID(instance_kind, int(setting_id), int(instance_id))
    return _formal_rng_for(config, component, identity, str(subcomponent_id))


def _block_slices(total_dimension: int, block_dimension: int) -> tuple[slice, ...]:
    if total_dimension <= 0 or block_dimension <= 0:
        raise ValueError("残差维度和块维度必须为正整数。")
    if total_dimension % block_dimension:
        raise ValueError("总残差维度必须能被块维度整除。")
    return tuple(
        slice(start, start + block_dimension)
        for start in range(0, total_dimension, block_dimension)
    )


def _reduced_qr_design(
    config: Mapping[str, Any], setting: LinearSetting, graph_id: int
) -> tuple[np.ndarray, int, float]:
    total_dimension = int(_contract(config)["linear"]["total_residual_dimension"])
    generator = _rng(
        config, "linear_design", "linear", setting.setting_id, graph_id
    )
    gaussian = generator.standard_normal(
        (total_dimension, setting.state_dimension), dtype=np.float64
    )
    q_design, r_factor = np.linalg.qr(gaussian, mode="reduced")

    # A positive R diagonal removes an irrelevant QR sign ambiguity.  The
    # projector, residuals and leverage blocks are unchanged by this choice.
    diagonal = np.diag(r_factor)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q_design = np.asarray(q_design * signs, dtype=np.float64)
    absolute_diagonal = np.abs(diagonal)
    sigma_proxy = float(np.max(absolute_diagonal)) if absolute_diagonal.size else 0.0
    rank_tolerance = (
        np.finfo(np.float64).eps
        * max(gaussian.shape)
        * sigma_proxy
    )
    rank = int(np.count_nonzero(absolute_diagonal > rank_tolerance))
    if rank != setting.state_dimension:
        raise LinearNumericalFailure(
            f"降维 QR 设计秩不足: rank={rank}, required={setting.state_dimension}"
        )
    return q_design, rank, float(rank_tolerance)


def _repair_trace_at_machine_boundary(trace: float, block_dimension: int) -> float:
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1, block_dimension)
    if -tolerance <= trace < 0.0:
        return 0.0
    if block_dimension < trace <= block_dimension + tolerance:
        return float(block_dimension)
    return float(trace)


def _build_leverage_blocks(
    q_design: np.ndarray,
    blocks: Sequence[slice],
    numerics: Mapping[str, Any],
) -> tuple[list[_LeverageBlock], dict[str, Union[float, int]]]:
    epsilon = float(numerics["spectral_clip_epsilon"])
    leverage_rule = numerics["full_block_leverage"]
    gross = leverage_rule["gross_eigenvalue_valid_range"]
    gross_minimum = float(gross["minimum"])
    gross_maximum = float(gross["maximum"])
    result: list[_LeverageBlock] = []
    all_minimum = np.inf
    all_maximum = -np.inf
    total_clipped = 0

    for factor_slice in blocks:
        rows = q_design[factor_slice]
        raw = rows @ rows.T
        symmetric = np.asarray(0.5 * (raw + raw.T), dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        all_minimum = min(all_minimum, minimum)
        all_maximum = max(all_maximum, maximum)
        if minimum < gross_minimum or maximum > gross_maximum:
            raise LinearNumericalFailure(
                "块杠杆矩阵特征值超出冻结数值容差: "
                f"[{minimum:.6e}, {maximum:.6e}]"
            )
        clipped = np.clip(eigenvalues, 0.0, 1.0 - epsilon)
        clipped_count = int(np.count_nonzero(clipped != eigenvalues))
        inverse_sqrt = (
            eigenvectors * (1.0 / np.sqrt(1.0 - clipped))
        ) @ eigenvectors.T
        trace = _repair_trace_at_machine_boundary(
            float(np.trace(symmetric)), symmetric.shape[0]
        )
        result.append(
            _LeverageBlock(
                matrix=symmetric,
                trace=trace,
                inverse_sqrt_identity_minus=np.asarray(inverse_sqrt, dtype=np.float64),
                clipped_eigenvalue_count=clipped_count,
                minimum_eigenvalue=minimum,
                maximum_eigenvalue=maximum,
            )
        )
        total_clipped += clipped_count

    return result, {
        "minimum_raw_eigenvalue": float(all_minimum),
        "maximum_raw_eigenvalue": float(all_maximum),
        "clipped_eigenvalue_count": int(total_clipped),
    }


def _full_block_dpr(
    residual: np.ndarray,
    blocks: Sequence[slice],
    leverage_blocks: Sequence[_LeverageBlock],
    *,
    permutation: Optional[Sequence[int]] = None,
) -> np.ndarray:
    corrected = np.empty_like(residual, dtype=np.float64)
    indices: Sequence[int] = (
        tuple(range(len(blocks))) if permutation is None else permutation
    )
    if len(indices) != len(blocks) or sorted(int(i) for i in indices) != list(
        range(len(blocks))
    ):
        raise ValueError("杠杆块置换必须是完整排列。")
    for target_slice, source_index in zip(blocks, indices):
        transform = leverage_blocks[int(source_index)].inverse_sqrt_identity_minus
        corrected[target_slice] = transform @ residual[target_slice]
    return corrected


def _scalar_trace_dpr(
    residual: np.ndarray,
    blocks: Sequence[slice],
    traces: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if traces.shape != (len(blocks),):
        raise ValueError("标量迹数量与因子块数不一致。")
    corrected = np.empty_like(residual, dtype=np.float64)
    block_dimension = blocks[0].stop - blocks[0].start
    clipped = np.clip(traces, 0.0, block_dimension * (1.0 - epsilon))
    denominators = np.sqrt(1.0 - clipped / block_dimension)
    for factor_slice, denominator in zip(blocks, denominators):
        corrected[factor_slice] = residual[factor_slice] / denominator
    return corrected


def _exact_scalar_trace_dpr(
    residual: np.ndarray,
    blocks: Sequence[slice],
    traces: np.ndarray,
) -> np.ndarray:
    """Apply exact scalar DPR without the random-trace deployment clip."""

    if traces.shape != (len(blocks),):
        raise ValueError("标量迹数量与因子块数不一致。")
    corrected = np.empty_like(residual, dtype=np.float64)
    block_dimension = blocks[0].stop - blocks[0].start
    repaired = np.asarray(
        [
            _repair_trace_at_machine_boundary(float(value), block_dimension)
            for value in traces
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(repaired)):
        raise LinearNumericalFailure("精确标量块迹含非有限值。")
    if np.any(repaired < 0.0) or np.any(repaired > block_dimension):
        raise LinearNumericalFailure("精确标量块迹超出理论区间[0,d]。")
    if np.any(repaired == block_dimension):
        raise LinearNumericalFailure("精确标量块迹到达奇异边界h=1。")
    denominators = np.sqrt(1.0 - repaired / block_dimension)
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        raise LinearNumericalFailure("精确标量DPR分母无效。")
    for factor_slice, denominator in zip(blocks, denominators):
        corrected[factor_slice] = residual[factor_slice] / denominator
    return corrected


def _exact_block_traces_from_design(
    q_design: np.ndarray, blocks: Sequence[slice]
) -> np.ndarray:
    """Compute exact block traces without constructing full leverage blocks."""

    return np.asarray(
        [
            _repair_trace_at_machine_boundary(
                float(np.sum(np.square(q_design[factor_slice]), dtype=np.float64)),
                factor_slice.stop - factor_slice.start,
            )
            for factor_slice in blocks
        ],
        dtype=np.float64,
    )


def _hutchinson_block_traces(
    q_design: np.ndarray,
    blocks: Sequence[slice],
    probes: np.ndarray,
    candidates: Sequence[int],
    epsilon: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if probes.shape != (q_design.shape[0], max(candidates)):
        raise ValueError("随机探针形状与冻结配置不一致。")
    projected = q_design @ (q_design.T @ probes)
    block_dimension = blocks[0].stop - blocks[0].start
    raw: dict[int, np.ndarray] = {}
    clipped: dict[int, np.ndarray] = {}
    for candidate in candidates:
        k = int(candidate)
        estimates = np.asarray(
            [
                np.sum(
                    probes[factor_slice, :k] * projected[factor_slice, :k],
                    dtype=np.float64,
                )
                / k
                for factor_slice in blocks
            ],
            dtype=np.float64,
        )
        raw[k] = estimates
        clipped[k] = np.clip(
            estimates, 0.0, block_dimension * (1.0 - epsilon)
        )
    return raw, clipped


def _nested_hutchinson_corrections(
    q_design: np.ndarray,
    blocks: Sequence[slice],
    probe_generator: np.random.Generator,
    candidates: Sequence[int],
    epsilon: float,
    residual: np.ndarray,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[str, int],
]:
    """按列增量生成共享探针，并记录K=4/8/16的独立累计耗时。"""

    ordered = tuple(int(value) for value in candidates)
    if not ordered or ordered != tuple(sorted(ordered)) or ordered[0] <= 0:
        raise ValueError("嵌套探针候选不符合冻结配置。")
    timing_start_ns = perf_counter_ns()
    probes = np.empty((q_design.shape[0], max(ordered)), dtype=np.float64)
    projected = np.empty_like(probes, dtype=np.float64)
    block_dimension = blocks[0].stop - blocks[0].start
    raw: dict[int, np.ndarray] = {}
    clipped: dict[int, np.ndarray] = {}
    corrected: dict[int, np.ndarray] = {}
    checkpoints: dict[str, int] = {}
    previous = 0
    for candidate in ordered:
        increment = candidate - previous
        probe_bits = probe_generator.integers(
            0,
            2,
            size=(increment, q_design.shape[0]),
            dtype=np.int8,
        )
        probes[:, previous:candidate] = np.asarray(
            2 * probe_bits.T - 1, dtype=np.float64
        )
        projected[:, previous:candidate] = q_design @ (
            q_design.T @ probes[:, previous:candidate]
        )
        estimates = np.asarray(
            [
                np.sum(
                    probes[factor_slice, :candidate]
                    * projected[factor_slice, :candidate],
                    dtype=np.float64,
                )
                / candidate
                for factor_slice in blocks
            ],
            dtype=np.float64,
        )
        raw[candidate] = estimates
        clipped[candidate] = np.clip(
            estimates, 0.0, block_dimension * (1.0 - epsilon)
        )
        corrected[candidate] = _scalar_trace_dpr(
            residual, blocks, clipped[candidate], epsilon
        )
        checkpoints[str(candidate)] = int(perf_counter_ns() - timing_start_ns)
        previous = candidate
    if any(
        checkpoints[str(left)] > checkpoints[str(right)]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise LinearNumericalFailure("嵌套K累计计时未保持单调。")
    return raw, clipped, corrected, checkpoints


def _runtime_environment() -> dict[str, Any]:
    try:
        scipy_version: Optional[str] = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = None
    build = getattr(np.__config__, "CONFIG", {})
    blas = build.get("Build Dependencies", {}).get("blas", {}) if isinstance(build, dict) else {}
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "blas_name": blas.get("name") if isinstance(blas, dict) else None,
        "blas_version": blas.get("version") if isinstance(blas, dict) else None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def _method_metrics(
    residual: np.ndarray,
    true_covariance: float,
    holdout_squared_norms: np.ndarray,
    chi_square_95: float,
    total_dimension: int,
) -> dict[str, Union[float, int]]:
    scale = float(np.sum(np.square(residual), dtype=np.float64) / total_dimension)
    if not np.isfinite(scale) or scale < 0.0:
        raise LinearNumericalFailure("尺度估计出现非有限值或负值。")
    coverage_events = holdout_squared_norms <= chi_square_95 * scale
    coverage = float(np.mean(coverage_events, dtype=np.float64))
    return {
        "scale_estimate": scale,
        "scale_relative_error": float(abs(scale / true_covariance - 1.0)),
        "coverage95": coverage,
        "ce95": float(abs(coverage - 0.95)),
        "coverage_success_count": int(np.count_nonzero(coverage_events)),
        "coverage_event_count": int(holdout_squared_norms.size),
    }


def _chi_square_95(block_dimension: int) -> float:
    try:
        return _CHI_SQUARE_95[int(block_dimension)]
    except KeyError as error:
        raise ValueError(
            f"冻结配置未定义块维度 {block_dimension} 的卡方 0.95 分位数。"
        ) from error


def _json_float_or_none(value: float) -> Optional[float]:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _hash_rank(payload: Sequence[Any]) -> bytes:
    return hashlib.sha256(_canonical_json_bytes(payload)).digest()


def selected_linear_graph_ids(
    config: Mapping[str, Any], setting_id: int
) -> tuple[int, ...]:
    """Return the 30 frozen LOFO graph IDs for one setting."""

    contract = _contract(config)
    linear = contract["linear"]
    lofo = contract["linear_lofo"]
    if setting_id < 0 or setting_id >= int(linear["expected_settings"]):
        raise IndexError("线性 setting_id 超出冻结范围。")
    candidates = range(int(linear["repeats_per_setting"]))
    ranked = sorted(
        candidates,
        key=lambda graph_id: (
            _hash_rank(
                [
                    "DUC-VIO",
                    "E0",
                    PROTOCOL_VERSION,
                    "linear_lofo_graph",
                    int(setting_id),
                    int(graph_id),
                ]
            ),
            graph_id,
        ),
    )
    return tuple(ranked[: int(lofo["graphs_per_setting"])])


def _selected_lofo_factor_ids(
    config: Mapping[str, Any],
    setting: LinearSetting,
    graph_id: int,
    exact_traces: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    lofo = _contract(config)["linear_lofo"]
    strata_count = int(lofo["leverage_strata"])
    per_stratum = int(lofo["factors_per_stratum"])
    factor_ids = np.arange(exact_traces.size, dtype=np.int64)
    mean_leverage = exact_traces / setting.block_dimension
    # np.lexsort uses the last key as primary; factor ID is the tie breaker.
    ranked = factor_ids[np.lexsort((factor_ids, mean_leverage))]
    strata = np.array_split(ranked, strata_count)
    selected: list[tuple[int, int]] = []
    for stratum_id, stratum in enumerate(strata):
        if stratum.size < per_stratum:
            raise LinearNumericalFailure(
                "LOFO 杠杆分层内的因子数少于冻结抽样数。"
            )
        ordered = sorted(
            (int(factor_id) for factor_id in stratum),
            key=lambda factor_id: (
                _hash_rank(
                    [
                        "DUC-VIO",
                        "E0",
                        PROTOCOL_VERSION,
                        "linear_lofo_factor",
                        setting.setting_id,
                        graph_id,
                        stratum_id,
                        factor_id,
                    ]
                ),
                factor_id,
            ),
        )
        selected.extend((stratum_id, factor_id) for factor_id in ordered[:per_stratum])
    if len(selected) != int(lofo["factors_per_graph"]):
        raise LinearNumericalFailure("LOFO 因子抽样数与冻结配置不一致。")
    return tuple(selected)


def _svd_least_squares_full_rank(
    design: np.ndarray, observations: np.ndarray, required_rank: int
) -> tuple[np.ndarray, int, float]:
    if design.dtype != np.float64 or observations.dtype != np.float64:
        raise TypeError("LOFO SVD 重求解必须使用 float64。")
    left, singular_values, right_transpose = np.linalg.svd(
        design, full_matrices=False
    )
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = (
        np.finfo(np.float64).eps * max(design.shape) * sigma_max
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != required_rank:
        raise LinearNumericalFailure(
            f"LOFO 删除求解秩不足: rank={rank}, required={required_rank}"
        )
    projected = left.T @ observations
    estimate = right_transpose.T @ (projected / singular_values)
    if not np.all(np.isfinite(estimate)):
        raise LinearNumericalFailure("LOFO SVD 解含非有限值。")
    return np.asarray(estimate, dtype=np.float64), rank, float(tolerance)


def _run_linear_lofo(
    config: Mapping[str, Any],
    setting: LinearSetting,
    graph_id: int,
    q_design: np.ndarray,
    observations: np.ndarray,
    raw_residual: np.ndarray,
    blocks: Sequence[slice],
    leverage_blocks: Sequence[_LeverageBlock],
) -> dict[str, Any]:
    selected_graphs = selected_linear_graph_ids(config, setting.setting_id)
    if graph_id not in selected_graphs:
        return {
            "selected_graph": False,
            "status": "NOT_SELECTED",
            "valid": True,
            "factor_results": [],
        }

    exact_traces = np.asarray(
        [block.trace for block in leverage_blocks], dtype=np.float64
    )
    selected_factors = _selected_lofo_factor_ids(
        config, setting, graph_id, exact_traces
    )
    factor_results: list[dict[str, Any]] = []
    all_valid = True
    total_dimension = q_design.shape[0]
    all_rows = np.ones(total_dimension, dtype=bool)

    for stratum_id, factor_id in selected_factors:
        factor_slice = blocks[factor_id]
        keep = all_rows.copy()
        keep[factor_slice] = False
        factor_record: dict[str, Any] = {
            "factor_id": factor_id,
            "stratum_id": stratum_id,
            "valid": False,
        }
        try:
            estimate, rank, tolerance = _svd_least_squares_full_rank(
                np.asarray(q_design[keep], dtype=np.float64),
                np.asarray(observations[keep], dtype=np.float64),
                setting.state_dimension,
            )
            deleted_residual = observations[factor_slice] - q_design[factor_slice] @ estimate
            raw_prediction = raw_residual[factor_slice]
            identity_minus = (
                np.eye(setting.block_dimension, dtype=np.float64)
                - leverage_blocks[factor_id].matrix
            )
            press_prediction = np.linalg.solve(identity_minus, raw_prediction)
            if not (
                np.all(np.isfinite(deleted_residual))
                and np.all(np.isfinite(press_prediction))
            ):
                raise LinearNumericalFailure("LOFO 预测残差含非有限值。")
            normalizer = np.sqrt(setting.block_dimension)
            raw_rmse = float(
                np.linalg.norm(raw_prediction - deleted_residual) / normalizer
            )
            press_rmse = float(
                np.linalg.norm(press_prediction - deleted_residual) / normalizer
            )
            factor_record.update(
                {
                    "valid": True,
                    "rank": rank,
                    "rank_tolerance": tolerance,
                    "raw_vector_rmse": raw_rmse,
                    "press_vector_rmse": press_rmse,
                    "press_identity_max_abs_error": float(
                        np.max(np.abs(press_prediction - deleted_residual))
                    ),
                }
            )
        except (LinearNumericalFailure, np.linalg.LinAlgError) as error:
            all_valid = False
            factor_record["failure_type"] = type(error).__name__
            factor_record["failure_message"] = str(error)
        factor_results.append(factor_record)

    result: dict[str, Any] = {
        "selected_graph": True,
        "status": "PASS" if all_valid else "NUMERICAL_FAILURE",
        "valid": all_valid,
        "selected_factor_count": len(selected_factors),
        "factor_results": factor_results,
    }
    if all_valid:
        result["graph_median_raw_vector_rmse"] = float(
            np.median([row["raw_vector_rmse"] for row in factor_results])
        )
        result["graph_median_press_vector_rmse"] = float(
            np.median([row["press_vector_rmse"] for row in factor_results])
        )
    return result


def run_linear_graph(
    config: Any,
    setting_id: int,
    graph_id: int,
    *,
    holdout_samples: Optional[int] = None,
    lofo_enabled: bool = True,
    process_session: Optional[FormalProcessSession] = None,
) -> dict[str, Any]:
    """Evaluate one frozen linear graph and return its atomic graph pack.

    ``holdout_samples`` and ``lofo_enabled`` exist only for local development.
    Any value that differs from the frozen protocol is fail-soft labelled as
    ``dev_toy`` and is never claim eligible.
    """

    contract = _contract(config)
    linear = contract["linear"]
    numerics = contract["numerics"]
    trace_rule = contract["trace_approximation"]
    settings = enumerate_linear_settings(config)
    if type(setting_id) is not int or type(graph_id) is not int:
        raise TypeError("setting_id 和 graph_id 必须是 JSON 整数。")
    if setting_id < 0 or setting_id >= len(settings):
        raise IndexError("线性 setting_id 超出冻结范围。")
    repeats = int(linear["repeats_per_setting"])
    if graph_id < 0 or graph_id >= repeats:
        raise IndexError("线性 graph_id 超出冻结范围。")
    setting = settings[setting_id]

    frozen_holdout = int(linear["holdout"]["samples_per_graph"])
    if holdout_samples is not None and type(holdout_samples) is not int:
        raise TypeError("holdout_samples 必须是整数或 None。")
    if type(lofo_enabled) is not bool:
        raise TypeError("lofo_enabled 必须是布尔值。")
    requested_holdout = frozen_holdout if holdout_samples is None else holdout_samples
    if requested_holdout <= 0:
        raise ValueError("holdout_samples 必须为正整数。")
    development_override = (
        requested_holdout != frozen_holdout or not bool(lofo_enabled)
    )
    stable_id = "v1.4:E0:L:s{:02d}:r{:03d}".format(setting_id, graph_id)
    if not development_override:
        _require_formal_source_authorized(config, process_session, stable_id)
    provenance = DEV_PROVENANCE if development_override else FORMAL_PROVENANCE
    evidence_eligibility = (
        "NOT_ELIGIBLE_DEVELOPMENT"
        if development_override
        else "PENDING_STAGE_REVIEW"
    )
    total_dimension = int(linear["total_residual_dimension"])
    epsilon = float(numerics["spectral_clip_epsilon"])
    candidates = tuple(int(value) for value in trace_rule["candidates"])
    if candidates != (4, 8, 16):
        raise ValueError("v1.4 随机迹候选必须为 (4, 8, 16)。")
    seed_components = (
        "linear_design",
        "linear_training_noise",
        "linear_holdout",
        "linear_negative_control_permutation",
        "linear_rademacher_probes",
    )
    derived_seeds = {
        component: _derived_seed(
            config, component, "linear", setting_id, graph_id
        )
        for component in seed_components
    }
    record: dict[str, Any] = {
        "schema_version": "e0.linear.graph_pack.v1",
        "protocol_version": PROTOCOL_VERSION,
        "stable_id": stable_id,
        "provenance": provenance,
        "dataset_origin": "designed_synthetic",
        "evidence_eligibility": evidence_eligibility,
        "science_claim_eligible": False,
        "stage_review_required": True,
        "status": "RUNNING",
        "main_status": "NOT_EVALUATED",
        "setting": setting.to_dict(),
        "graph_id": int(graph_id),
        "master_seed": int(contract["randomness"]["master_seed"]),
        "derived_seeds": derived_seeds,
        "development_overrides": {
            "holdout_samples": (
                requested_holdout if requested_holdout != frozen_holdout else None
            ),
            "lofo_disabled": not bool(lofo_enabled),
        },
    }

    try:
        q_design, design_rank, design_rank_tolerance = _reduced_qr_design(
            config, setting, graph_id
        )
        noise_generator = _rng(
            config,
            "linear_training_noise",
            "linear",
            setting_id,
            graph_id,
        )
        training_noise = (
            np.sqrt(setting.covariance_multiplier)
            * noise_generator.standard_normal(total_dimension, dtype=np.float64)
        )
        observations = np.asarray(training_noise, dtype=np.float64)
        raw_residual = observations - q_design @ (q_design.T @ observations)
        if not np.all(np.isfinite(raw_residual)):
            raise LinearNumericalFailure("原始优化后残差含非有限值。")

        blocks = _block_slices(total_dimension, setting.block_dimension)
        full_block_start = perf_counter_ns()
        leverage_blocks, spectral_diagnostics = _build_leverage_blocks(
            q_design, blocks, numerics
        )
        exact_traces = np.asarray(
            [block.trace for block in leverage_blocks], dtype=np.float64
        )
        full_block_residual = _full_block_dpr(
            raw_residual, blocks, leverage_blocks
        )
        full_block_elapsed_ns = perf_counter_ns() - full_block_start

        exact_scalar_start = perf_counter_ns()
        timed_exact_traces = _exact_block_traces_from_design(q_design, blocks)
        exact_scalar_residual = _exact_scalar_trace_dpr(
            raw_residual, blocks, timed_exact_traces
        )
        exact_scalar_elapsed_ns = perf_counter_ns() - exact_scalar_start

        probe_generator = _rng(
            config,
            "linear_rademacher_probes",
            "linear",
            setting_id,
            graph_id,
        )
        random_raw, random_clipped, random_residuals, timing_checkpoints = (
            _nested_hutchinson_corrections(
                q_design,
                blocks,
                probe_generator,
                candidates,
                epsilon,
                raw_residual,
            )
        )

        permutation_generator = _rng(
            config,
            "linear_negative_control_permutation",
            "linear",
            setting_id,
            graph_id,
        )
        negative_permutation = np.asarray(
            permutation_generator.permutation(len(blocks)), dtype=np.int64
        )
        negative_residual = _full_block_dpr(
            raw_residual,
            blocks,
            leverage_blocks,
            permutation=negative_permutation,
        )
        holdout_generator = _rng(
            config, "linear_holdout", "linear", setting_id, graph_id
        )
        holdout = (
            np.sqrt(setting.covariance_multiplier)
            * holdout_generator.standard_normal(
                (requested_holdout, setting.block_dimension), dtype=np.float64
            )
        )
        holdout_squared_norms = np.sum(
            np.square(holdout), axis=1, dtype=np.float64
        )
        chi_square_95 = _chi_square_95(setting.block_dimension)
        residuals = {
            "raw_postfit": raw_residual,
            "full_block_dpr": full_block_residual,
            "exact_scalar_block_trace_dpr": exact_scalar_residual,
            "permuted_full_block_negative_control": negative_residual,
        }
        methods: dict[str, Any] = {
            name: _method_metrics(
                residual,
                setting.covariance_multiplier,
                holdout_squared_norms,
                chi_square_95,
                total_dimension,
            )
            for name, residual in residuals.items()
        }

        trace_candidates: dict[str, Any] = {}
        for candidate in candidates:
            raw_estimate = random_raw[candidate]
            clipped_estimate = random_clipped[candidate]
            finite = bool(
                np.all(np.isfinite(raw_estimate))
                and np.all(np.isfinite(clipped_estimate))
                and candidate in random_residuals
            )
            candidate_record: dict[str, Any] = {
                "candidate_k": candidate,
                "eligible_for_k_selection": finite,
                "status": "PASS" if finite else "NONFINITE_INELIGIBLE",
                "raw_unclipped_traces": [
                    _json_float_or_none(value) for value in raw_estimate
                ],
                "deployment_clipped_traces": [
                    _json_float_or_none(value) for value in clipped_estimate
                ],
            }
            if finite:
                relative_errors = np.abs(clipped_estimate - exact_traces) / np.maximum(
                    np.abs(exact_traces), 1.0e-12
                )
                candidate_record["graph_median_trace_relative_error"] = float(
                    np.median(relative_errors)
                )
                exact_mean_leverage = np.clip(
                    exact_traces / setting.block_dimension,
                    0.0,
                    1.0 - epsilon,
                )
                approximate_mean_leverage = np.clip(
                    clipped_estimate / setting.block_dimension,
                    0.0,
                    1.0 - epsilon,
                )
                deployment_gain_errors = np.abs(
                    np.sqrt(
                        (1.0 - exact_mean_leverage)
                        / (1.0 - approximate_mean_leverage)
                    )
                    - 1.0
                )
                candidate_record[
                    "graph_median_deployment_gain_relative_error"
                ] = float(np.median(deployment_gain_errors))
                methods["random_trace_k{}_dpr".format(candidate)] = _method_metrics(
                    random_residuals[candidate],
                    setting.covariance_multiplier,
                    holdout_squared_norms,
                    chi_square_95,
                    total_dimension,
                )
            trace_candidates[str(candidate)] = candidate_record

        required_main = (
            "raw_postfit",
            "full_block_dpr",
            "permuted_full_block_negative_control",
        )
        if not all(
            all(np.isfinite(float(value)) for value in methods[name].values())
            for name in required_main
        ):
            raise LinearNumericalFailure("线性主端点出现非有限值。")

        lofo = (
            _run_linear_lofo(
                config,
                setting,
                graph_id,
                q_design,
                observations,
                raw_residual,
                blocks,
                leverage_blocks,
            )
            if lofo_enabled
            else {
                "selected_graph": graph_id
                in selected_linear_graph_ids(config, setting_id),
                "status": "DISABLED_DEV_ONLY",
                "valid": False,
                "factor_results": [],
            }
        )

        block_diagnostics: list[dict[str, Any]] = []
        for factor_id, leverage in enumerate(leverage_blocks):
            block_row: dict[str, Any] = {
                "factor_id": factor_id,
                "exact_unregularized_trace": float(leverage.trace),
                "exact_mean_scalar_leverage": float(
                    leverage.trace / setting.block_dimension
                ),
                "minimum_raw_eigenvalue": leverage.minimum_eigenvalue,
                "maximum_raw_eigenvalue": leverage.maximum_eigenvalue,
                "clipped_eigenvalue_count": leverage.clipped_eigenvalue_count,
            }
            for candidate in candidates:
                block_row["k{}_raw_unclipped_trace".format(candidate)] = (
                    _json_float_or_none(random_raw[candidate][factor_id])
                )
                block_row["k{}_deployment_clipped_trace".format(candidate)] = (
                    _json_float_or_none(random_clipped[candidate][factor_id])
                )
            block_diagnostics.append(block_row)

        lofo_failed = bool(
            lofo_enabled and lofo["selected_graph"] and not lofo["valid"]
        )
        record.update(
            {
                "status": "LOFO_NUMERICAL_FAILURE" if lofo_failed else "PASS",
                "main_status": "PASS",
                "design": {
                    "rank": design_rank,
                    "required_rank": setting.state_dimension,
                    "rank_tolerance": design_rank_tolerance,
                    "true_state": "all_zeros",
                },
                "factor_block_count": len(blocks),
                "actual_mean_scalar_leverage": float(
                    np.sum(exact_traces, dtype=np.float64) / total_dimension
                ),
                "exact_trace_sum": float(
                    np.sum(exact_traces, dtype=np.float64)
                ),
                "spectral_diagnostics": spectral_diagnostics,
                "negative_control_permutation": [
                    int(value) for value in negative_permutation
                ],
                "holdout_samples": requested_holdout,
                "chi_square_95": chi_square_95,
                "methods": methods,
                "trace_candidates": trace_candidates,
                "block_diagnostics": block_diagnostics,
                "lofo": lofo,
                "timing": {
                    "clock": "time.perf_counter_ns",
                    "full_block_dpr_elapsed_ns": int(full_block_elapsed_ns),
                    "exact_scalar_block_trace_dpr_elapsed_ns": int(
                        exact_scalar_elapsed_ns
                    ),
                    "nested_k_cumulative_elapsed_ns": timing_checkpoints,
                    "nested_k_shared_probe_columns": 16,
                    "nested_k_probe_generation": "column_major_incremental_4_8_16",
                    "excludes_graph_generation_base_solve_holdout_lofo_and_io": True,
                },
                "runtime_environment": _runtime_environment(),
            }
        )
    except (LinearNumericalFailure, np.linalg.LinAlgError, FloatingPointError) as error:
        record.update(
            {
                "status": "NUMERICAL_FAILURE",
                "main_status": "NUMERICAL_FAILURE",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
    return record


def run_linear_timing_warmup(
    config_or_contract: Any,
    *,
    instances_per_structure: Optional[int] = None,
    batch_authorization: Optional[BatchAuthorization] = None,
) -> dict[str, Any]:
    """执行独立计时预热；结果不进入样本数、K选择或计时汇总。"""

    config = _config_mapping(config_or_contract)
    formal = _contract(config)
    timing = formal["timing"]
    frozen_count = int(
        timing["warmup"][
            "instances_per_unique_block_dimension_and_state_dimension"
        ]
    )
    if instances_per_structure is not None and type(instances_per_structure) is not int:
        raise ValueError("instances_per_structure 必须是JSON整数。")
    actual_count = (
        frozen_count
        if instances_per_structure is None
        else instances_per_structure
    )
    if actual_count <= 0 or actual_count > frozen_count:
        raise ValueError("计时预热次数必须在冻结上限内。")
    formal_scale = actual_count == frozen_count
    if formal_scale:
        if not isinstance(config_or_contract, FrozenContract) or not isinstance(
            batch_authorization, BatchAuthorization
        ):
            raise FormalExecutionLocked(
                "正式计时预热需要FrozenContract与BatchAuthorization。"
            )
        batch_authorization.verify(config_or_contract)
    elif isinstance(config_or_contract, FrozenContract):
        raise ValueError("正式合同不得缩减计时预热次数。")

    settings = enumerate_linear_settings(config)
    representatives: dict[tuple[int, int], LinearSetting] = {}
    for setting in settings:
        representatives.setdefault(
            (setting.block_dimension, setting.state_dimension), setting
        )
    expected_structures = {
        (int(block), int(state))
        for block in formal["linear"]["block_dimensions"]
        for state in formal["linear"]["state_dimension_by_mean_leverage"].values()
    }
    if set(representatives) != expected_structures:
        raise LinearNumericalFailure("计时预热结构集合与冻结配置不一致。")

    total_dimension = int(formal["linear"]["total_residual_dimension"])
    epsilon = float(formal["numerics"]["spectral_clip_epsilon"])
    candidates = tuple(int(value) for value in formal["trace_approximation"]["candidates"])
    completed = 0
    for (_, _), setting in sorted(representatives.items()):
        blocks = _block_slices(total_dimension, setting.block_dimension)
        for warmup_index in range(actual_count):
            identity = FormalInstanceID("linear", setting.setting_id, warmup_index)
            design_rng = _formal_rng_for(
                config, "timing_warmup", identity, "design"
            )
            gaussian = design_rng.standard_normal(
                (total_dimension, setting.state_dimension), dtype=np.float64
            )
            q_design, r_factor = np.linalg.qr(gaussian, mode="reduced")
            signs = np.where(np.diag(r_factor) < 0.0, -1.0, 1.0)
            q_design = np.asarray(q_design * signs, dtype=np.float64)
            residual_rng = _formal_rng_for(
                config, "timing_warmup", identity, "residual"
            )
            noise = residual_rng.standard_normal(total_dimension, dtype=np.float64)
            residual = noise - q_design @ (q_design.T @ noise)
            leverage, _ = _build_leverage_blocks(
                q_design, blocks, formal["numerics"]
            )
            _full_block_dpr(residual, blocks, leverage)
            exact_traces = _exact_block_traces_from_design(q_design, blocks)
            _exact_scalar_trace_dpr(residual, blocks, exact_traces)
            probe_rng = _formal_rng_for(
                config, "timing_warmup", identity, "probes"
            )
            _nested_hutchinson_corrections(
                q_design,
                blocks,
                probe_rng,
                candidates,
                epsilon,
                residual,
            )
            completed += 1
    return {
        "protocol_version": PROTOCOL_VERSION,
        "record_type": "E0_LINEAR_TIMING_WARMUP_RECEIPT",
        "provenance": FORMAL_PROVENANCE if formal_scale else DEV_PROVENANCE,
        "random_stream": "timing_warmup",
        "unique_block_dimension_state_dimension_count": len(representatives),
        "instances_per_structure": actual_count,
        "completed_warmup_count": completed,
        "included_in_scientific_sample_counts": False,
        "included_in_timing_summary": False,
        "may_select_k": False,
        "evidence_eligibility": "NOT_ELIGIBLE_DEVELOPMENT",
        "science_claim_eligible": False,
        "runtime_environment": _runtime_environment(),
    }


__all__ = [
    "LinearNumericalFailure",
    "LinearSetting",
    "enumerate_linear_settings",
    "run_linear_graph",
    "run_linear_timing_warmup",
    "selected_linear_graph_ids",
]
