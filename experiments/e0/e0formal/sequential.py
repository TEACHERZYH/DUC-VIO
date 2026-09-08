from __future__ import annotations

"""E0 v1.4 formal sequential covariance-step experiment."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np

from .authorization import FormalProcessSession
from .linear import (
    DEV_PROVENANCE,
    FORMAL_PROVENANCE,
    PROTOCOL_VERSION,
    LinearNumericalFailure,
    _block_slices,
    _build_leverage_blocks,
    _contract,
    _derived_seed,
    _exact_scalar_trace_dpr,
    _require_formal_source_authorized,
    _rng,
)


class SequentialNumericalFailure(RuntimeError):
    """A recorded numerical failure for one frozen sequential identity."""


@dataclass(frozen=True)
class SequentialSetting:
    setting_id: int
    block_dimension: int
    covariance_step_multiplier: float
    contamination_rate: float

    def to_dict(self) -> dict[str, Union[int, float]]:
        return {
            "setting_id": self.setting_id,
            "block_dimension": self.block_dimension,
            "covariance_step_multiplier": self.covariance_step_multiplier,
            "contamination_rate": self.contamination_rate,
        }


def enumerate_sequential_settings(
    config: Mapping[str, Any]
) -> list[SequentialSetting]:
    """Enumerate the eight frozen settings; contamination varies fastest."""

    sequential = _contract(config)["sequential"]
    settings: list[SequentialSetting] = []
    setting_id = 0
    for block_dimension in sequential["block_dimensions"]:
        for step_multiplier in sequential["step_multipliers"]:
            for contamination_rate in sequential["contamination_rates"]:
                settings.append(
                    SequentialSetting(
                        setting_id=setting_id,
                        block_dimension=int(block_dimension),
                        covariance_step_multiplier=float(step_multiplier),
                        contamination_rate=float(contamination_rate),
                    )
                )
                setting_id += 1
    if len(settings) != int(sequential["expected_settings"]):
        raise ValueError("顺序设置数与冻结配置不一致。")
    return settings


class DTCTracker:
    """Frozen dual-time-scale log-covariance tracker used by E0."""

    def __init__(
        self,
        dtc: Mapping[str, Any],
        initialization: Mapping[str, Any],
        scale_bounds: Optional[tuple[float, float]] = None,
    ) -> None:
        if initialization["hidden_truth_warmup_allowed"]:
            raise ValueError("E0 禁止使用真值预热 DTC。")
        if initialization["initial_mode"] != "slow":
            raise ValueError("DTC 冻结初始模式必须为 slow。")
        self.slow_log_scale = float(initialization["slow_log_scale"])
        self.fast_log_scale = float(initialization["fast_log_scale"])
        self.output_log_scale = float(initialization["output_log_scale"])
        self.g_plus = float(initialization["g_plus"])
        self.g_minus = float(initialization["g_minus"])
        self.fast_mode = False
        self.fast_hold_count = int(initialization["fast_hold_count"])
        self.stable_count = int(initialization["stable_count"])
        self.dtc = dtc
        configured_update_rule = str(
            dtc.get(
                "update_rule",
                dtc.get("valid_observation_update", "legacy_mean_reverting_log_ema"),
            )
        )
        if configured_update_rule in {
            "standard_log_ema",
            "standard_log_domain_exponential_moving_average",
        } or configured_update_rule.startswith("u_j_t=clip((1-alpha_j)"):
            self.update_rule = "standard_log_ema"
        elif configured_update_rule == "legacy_mean_reverting_log_ema":
            self.update_rule = configured_update_rule
        else:
            raise ValueError("DTC有效观测更新规则不受支持。")
        self.configured_update_rule = configured_update_rule
        self.invalid_update_rule = str(
            dtc.get(
                "invalid_observation_update",
                "hold_previous_slow_fast_output_cusum_mode_fast_hold_and_stable_count_states",
            )
        )
        if self.invalid_update_rule != (
            "hold_previous_slow_fast_output_cusum_mode_fast_hold_and_stable_count_states"
        ):
            raise ValueError("DTC无效观测更新规则不受支持。")
        if scale_bounds is None:
            self.log_scale_bounds = None
        else:
            lower, upper = (float(scale_bounds[0]), float(scale_bounds[1]))
            if not 0.0 < lower < upper:
                raise ValueError("DTC尺度边界必须满足0<lower<upper。")
            self.log_scale_bounds = (float(np.log(lower)), float(np.log(upper)))

    def _clip_log_scale(self, value: float) -> float:
        if self.log_scale_bounds is None:
            return float(value)
        return float(np.clip(value, *self.log_scale_bounds))

    @property
    def covariance_scale(self) -> float:
        value = float(np.exp(self.output_log_scale))
        if not np.isfinite(value) or value <= 0.0:
            raise SequentialNumericalFailure("DTC 协方差尺度非有限或不为正。")
        return value

    def update(
        self,
        normalized_block_energies: Optional[np.ndarray],
        *,
        observation_valid: bool = True,
    ) -> dict[str, Any]:
        if type(observation_valid) is not bool:
            raise TypeError("observation_valid 必须是布尔值。")
        if not observation_valid:
            return {
                "covariance_scale": self.covariance_scale,
                "q": None,
                "instantaneous_log_scale": None,
                "deviation": None,
                "trigger": False,
                "mode": "fast" if self.fast_mode else "slow",
                "slow_log_scale": float(self.slow_log_scale),
                "fast_log_scale": float(self.fast_log_scale),
                "output_log_scale": float(self.output_log_scale),
                "g_plus": float(self.g_plus),
                "g_minus": float(self.g_minus),
                "fast_hold_count": int(self.fast_hold_count),
                "stable_count": int(self.stable_count),
                "valid_observation_update": self.update_rule,
                "configured_update_rule": self.configured_update_rule,
                "observation_status": "HELD_INVALID",
            }
        if not isinstance(normalized_block_energies, np.ndarray):
            raise TypeError("有效DTC观测必须是NumPy数组。")
        if normalized_block_energies.ndim != 1 or normalized_block_energies.size == 0:
            raise ValueError("DTC 必须接收非空的块能量向量。")
        if not np.all(np.isfinite(normalized_block_energies)):
            raise SequentialNumericalFailure("DTC 归一化块能量含非有限值。")
        clip_low, clip_high = self.dtc["energy_clip"]
        q_value = float(
            np.mean(
                np.clip(
                    normalized_block_energies,
                    float(clip_low),
                    float(clip_high),
                ),
                dtype=np.float64,
            )
        )
        instantaneous_log_scale = self.output_log_scale + float(np.log(q_value))
        deviation = instantaneous_log_scale - self.slow_log_scale
        drift = float(self.dtc["cusum_drift"])
        self.g_plus = max(0.0, self.g_plus + deviation - drift)
        self.g_minus = max(0.0, self.g_minus - deviation - drift)
        triggered = bool(
            max(self.g_plus, self.g_minus) > float(self.dtc["cusum_threshold"])
        )

        alpha_slow = float(self.dtc["alpha_slow"])
        alpha_fast = float(self.dtc["alpha_fast"])
        slow_update = (
            (1.0 - alpha_slow) * self.slow_log_scale
            + alpha_slow * instantaneous_log_scale
        )
        fast_update = (
            (1.0 - alpha_fast) * self.fast_log_scale
            + alpha_fast * instantaneous_log_scale
        )
        if self.update_rule == "legacy_mean_reverting_log_ema":
            beta = float(self.dtc["beta"])
            slow_update *= 1.0 - beta
            fast_update *= 1.0 - beta
        self.slow_log_scale = self._clip_log_scale(slow_update)
        self.fast_log_scale = self._clip_log_scale(fast_update)

        if triggered:
            self.fast_mode = True
            self.fast_hold_count = max(
                self.fast_hold_count, int(self.dtc["minimum_fast_hold"])
            )
            self.g_plus = 0.0
            self.g_minus = 0.0
        if abs(deviation) < float(self.dtc["slow_return_abs_deviation"]):
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.fast_mode:
            self.output_log_scale = self.fast_log_scale
            if self.fast_hold_count > 0:
                self.fast_hold_count -= 1
            elif self.stable_count >= int(self.dtc["slow_return_stable_steps"]):
                self.fast_mode = False
                self.output_log_scale = self.slow_log_scale
        else:
            self.output_log_scale = self.slow_log_scale

        scale = self.covariance_scale
        return {
            "covariance_scale": scale,
            "q": q_value,
            "instantaneous_log_scale": float(instantaneous_log_scale),
            "deviation": float(deviation),
            "trigger": triggered,
            "mode": "fast" if self.fast_mode else "slow",
            "slow_log_scale": float(self.slow_log_scale),
            "fast_log_scale": float(self.fast_log_scale),
            "output_log_scale": float(self.output_log_scale),
            "g_plus": float(self.g_plus),
            "g_minus": float(self.g_minus),
            "fast_hold_count": int(self.fast_hold_count),
            "stable_count": int(self.stable_count),
            "valid_observation_update": self.update_rule,
            "configured_update_rule": self.configured_update_rule,
            "observation_status": "UPDATED_VALID",
        }


def _sequential_design(
    config: Mapping[str, Any], setting: SequentialSetting, instance_id: int
) -> tuple[np.ndarray, int, float]:
    sequential = _contract(config)["sequential"]
    total_dimension = int(sequential["total_residual_dimension"])
    state_dimension = int(sequential["mean_leverage"]["state_dimension"])
    generator = _rng(
        config,
        "sequential_design",
        "sequential",
        setting.setting_id,
        instance_id,
    )
    gaussian = generator.standard_normal(
        (total_dimension, state_dimension), dtype=np.float64
    )
    q_design, r_factor = np.linalg.qr(gaussian, mode="reduced")
    diagonal = np.diag(r_factor)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    q_design = np.asarray(q_design * signs, dtype=np.float64)
    absolute_diagonal = np.abs(diagonal)
    sigma_proxy = float(np.max(absolute_diagonal)) if absolute_diagonal.size else 0.0
    tolerance = (
        np.finfo(np.float64).eps * max(gaussian.shape) * sigma_proxy
    )
    rank = int(np.count_nonzero(absolute_diagonal > tolerance))
    if rank != state_dimension:
        raise SequentialNumericalFailure(
            f"顺序设计秩不足: rank={rank}, required={state_dimension}"
        )
    return q_design, rank, float(tolerance)


def run_sequential_instance(
    config: Any,
    setting_id: int,
    instance_id: int,
    *,
    length: Optional[int] = None,
    process_session: Optional[FormalProcessSession] = None,
) -> dict[str, Any]:
    """Evaluate one frozen sequential instance.

    A shorter ``length`` is permitted only for module testing.  Such output is
    labelled ``dev_toy`` and is not eligible for scientific claims.
    """

    contract = _contract(config)
    sequential = contract["sequential"]
    settings = enumerate_sequential_settings(config)
    if type(setting_id) is not int or type(instance_id) is not int:
        raise TypeError("setting_id 和 instance_id 必须是 JSON 整数。")
    if setting_id < 0 or setting_id >= len(settings):
        raise IndexError("顺序 setting_id 超出冻结范围。")
    repeats = int(sequential["repeats_per_setting"])
    if instance_id < 0 or instance_id >= repeats:
        raise IndexError("顺序 instance_id 超出冻结范围。")
    setting = settings[setting_id]

    frozen_length = int(sequential["length"])
    if length is not None and type(length) is not int:
        raise TypeError("length 必须是整数或 None。")
    requested_length = frozen_length if length is None else length
    if requested_length <= 0 or requested_length > frozen_length:
        raise ValueError("开发 length 必须在 1 到冻结序列长度之间。")
    development_override = requested_length != frozen_length
    stable_id = "v1.4:E0:S:s{:02d}:r{:03d}".format(setting_id, instance_id)
    if not development_override:
        _require_formal_source_authorized(config, process_session, stable_id)
    provenance = DEV_PROVENANCE if development_override else FORMAL_PROVENANCE
    evidence_eligibility = (
        "NOT_ELIGIBLE_DEVELOPMENT"
        if development_override
        else "PENDING_STAGE_REVIEW"
    )
    seed_components = (
        "sequential_design",
        "sequential_gaussian_noise",
        "sequential_contamination_mask",
        "sequential_contamination_values",
    )
    record: dict[str, Any] = {
        "schema_version": "e0.sequential.instance_pack.v1",
        "protocol_version": PROTOCOL_VERSION,
        "stable_id": stable_id,
        "provenance": provenance,
        "dataset_origin": "designed_synthetic",
        "evidence_eligibility": evidence_eligibility,
        "science_claim_eligible": False,
        "stage_review_required": True,
        "status": "RUNNING",
        "setting": setting.to_dict(),
        "instance_id": int(instance_id),
        "master_seed": int(contract["randomness"]["master_seed"]),
        "derived_seeds": {
            component: _derived_seed(
                config, component, "sequential", setting_id, instance_id
            )
            for component in seed_components
        },
        "development_overrides": {
            "length": requested_length if development_override else None
        },
    }

    try:
        q_design, design_rank, design_rank_tolerance = _sequential_design(
            config, setting, instance_id
        )
        total_dimension = int(sequential["total_residual_dimension"])
        state_dimension = int(sequential["mean_leverage"]["state_dimension"])
        change_index = int(sequential["change_index_zero_based"])
        blocks = _block_slices(total_dimension, setting.block_dimension)
        leverage_blocks, spectral_diagnostics = _build_leverage_blocks(
            q_design, blocks, contract["numerics"]
        )
        exact_traces = np.asarray(
            [block.trace for block in leverage_blocks], dtype=np.float64
        )
        gaussian_generator = _rng(
            config,
            "sequential_gaussian_noise",
            "sequential",
            setting_id,
            instance_id,
        )
        mask_generator = _rng(
            config,
            "sequential_contamination_mask",
            "sequential",
            setting_id,
            instance_id,
        )
        contamination_generator = _rng(
            config,
            "sequential_contamination_values",
            "sequential",
            setting_id,
            instance_id,
        )
        factor_count = len(blocks)
        gaussian_blocks = gaussian_generator.standard_normal(
            (requested_length, factor_count, setting.block_dimension),
            dtype=np.float64,
        )
        contamination_masks = (
            mask_generator.random((requested_length, factor_count))
            < setting.contamination_rate
        )
        student_t_blocks = np.asarray(
            contamination_generator.standard_t(
                df=5.0,
                size=(requested_length, factor_count, setting.block_dimension),
            ),
            dtype=np.float64,
        )
        student_t_blocks *= float(
            sequential["contamination"][
                "student_t_variance_normalization_factor"
            ]
        )

        bounds_by_dimension = sequential["dtc"].get(
            "covariance_scale_bounds_by_block_dimension",
            sequential.get("scale_bounds_by_block_dimension"),
        )
        if bounds_by_dimension is None:
            scale_bounds = None
        else:
            raw_bounds = bounds_by_dimension.get(str(setting.block_dimension))
            if (
                not isinstance(raw_bounds, (list, tuple))
                or len(raw_bounds) != 2
            ):
                raise ValueError("顺序DTC缺少当前块维度的冻结尺度边界。")
            scale_bounds = (float(raw_bounds[0]), float(raw_bounds[1]))
        trackers = {
            "raw_postfit": DTCTracker(
                sequential["dtc"], sequential["initialization"], scale_bounds
            ),
            "exact_scalar_block_trace_dpr": DTCTracker(
                sequential["dtc"], sequential["initialization"], scale_bounds
            ),
        }
        trigger_times: dict[str, list[int]] = {
            method: [] for method in trackers
        }
        scale_histories: dict[str, list[float]] = {
            method: [] for method in trackers
        }
        time_series: list[dict[str, Any]] = []

        for time_index in range(requested_length):
            true_covariance = (
                1.0
                if time_index < change_index
                else setting.covariance_step_multiplier
            )
            mask = contamination_masks[time_index]
            standardized = np.where(
                mask[:, None],
                student_t_blocks[time_index],
                gaussian_blocks[time_index],
            )
            physical_noise = np.asarray(
                np.sqrt(true_covariance) * standardized.reshape(total_dimension),
                dtype=np.float64,
            )
            raw_residual = physical_noise - q_design @ (q_design.T @ physical_noise)
            scalar_residual = _exact_scalar_trace_dpr(
                raw_residual, blocks, exact_traces
            )
            method_residuals = {
                "raw_postfit": raw_residual,
                "exact_scalar_block_trace_dpr": scalar_residual,
            }
            time_record: dict[str, Any] = {
                "time_index": time_index,
                "true_covariance_multiplier": float(true_covariance),
                "contaminated_block_count": int(np.count_nonzero(mask)),
                "contaminated_factor_ids": [
                    int(value) for value in np.flatnonzero(mask)
                ],
                "methods": {},
            }
            for method, residual in method_residuals.items():
                current_scale = trackers[method].covariance_scale
                block_energies = np.asarray(
                    [
                        np.mean(
                            np.square(residual[factor_slice]),
                            dtype=np.float64,
                        )
                        / current_scale
                        for factor_slice in blocks
                    ],
                    dtype=np.float64,
                )
                update = trackers[method].update(block_energies)
                if update["trigger"]:
                    trigger_times[method].append(time_index)
                scale_histories[method].append(float(update["covariance_scale"]))
                time_record["methods"][method] = update
            time_series.append(time_record)

        evaluation = sequential["detection_evaluation"]
        stable_interval = evaluation["stable_scale_error_interval"]
        stable_start = int(stable_interval["start_inclusive"])
        stable_end = int(stable_interval["end_inclusive"])
        method_summaries: dict[str, Any] = {}
        for method in trackers:
            postchange_triggers = [
                value for value in trigger_times[method] if value >= change_index
            ]
            if postchange_triggers:
                first_postchange_trigger: Optional[int] = postchange_triggers[0]
                miss_indicator = 0
                assigned_delay = first_postchange_trigger - change_index
            else:
                first_postchange_trigger = None
                miss_indicator = int(evaluation["miss_rule"]["miss_indicator"])
                assigned_delay = int(evaluation["miss_rule"]["assigned_delay"])

            available_stable_indices = [
                value
                for value in range(stable_start, stable_end + 1)
                if value < requested_length
            ]
            if available_stable_indices:
                stable_errors = [
                    abs(
                        scale_histories[method][value]
                        / setting.covariance_step_multiplier
                        - 1.0
                    )
                    for value in available_stable_indices
                ]
                stable_error: Optional[float] = float(np.median(stable_errors))
            else:
                stable_error = None
            false_alarm_start = int(
                evaluation["prechange_false_alarm_interval"]["start_inclusive"]
            )
            false_alarm_end = int(
                evaluation["prechange_false_alarm_interval"]["end_exclusive"]
            )
            boundary_interval = evaluation["boundary_hit_interval"]
            boundary_start = int(boundary_interval["start_inclusive"])
            boundary_end = int(boundary_interval["end_inclusive"])
            boundary_indices = [
                value
                for value in range(boundary_start, boundary_end + 1)
                if value < requested_length
            ]
            if scale_bounds is None:
                boundary_summary: dict[str, Any] = {
                    "enabled": False,
                    "lower": None,
                    "upper": None,
                    "interval_start_inclusive": boundary_start,
                    "interval_end_inclusive": boundary_end,
                    "denominator": len(boundary_indices),
                    "lower_hit_count": 0,
                    "upper_hit_count": 0,
                    "any_boundary_hit_count": 0,
                    "any_boundary_hit_rate": 0.0,
                }
            else:
                scale_values = np.asarray(
                    [scale_histories[method][value] for value in boundary_indices],
                    dtype=np.float64,
                )
                lower_bound, upper_bound = scale_bounds
                relative_tolerance = float(boundary_interval["relative_tolerance"])
                lower_hits = scale_values <= lower_bound * (1.0 + relative_tolerance)
                upper_hits = scale_values >= upper_bound * (1.0 - relative_tolerance)
                any_hits = np.logical_or(lower_hits, upper_hits)
                boundary_summary = {
                    "enabled": True,
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                    "interval_start_inclusive": boundary_start,
                    "interval_end_inclusive": boundary_end,
                    "denominator": len(boundary_indices),
                    "lower_hit_count": int(np.count_nonzero(lower_hits)),
                    "upper_hit_count": int(np.count_nonzero(upper_hits)),
                    "any_boundary_hit_count": int(np.count_nonzero(any_hits)),
                    "any_boundary_hit_rate": float(
                        np.mean(any_hits, dtype=np.float64)
                        if any_hits.size
                        else 0.0
                    ),
                }
            method_summaries[method] = {
                "first_postchange_trigger": first_postchange_trigger,
                "detection_delay": int(assigned_delay),
                "miss_indicator": int(miss_indicator),
                "prechange_false_alarm_count": int(
                    sum(
                        false_alarm_start <= value < false_alarm_end
                        for value in trigger_times[method]
                    )
                ),
                "all_trigger_times": [int(value) for value in trigger_times[method]],
                "stable_covariance_scale_relative_error": stable_error,
                "stable_evaluation_sample_count": len(available_stable_indices),
                "final_covariance_scale": float(scale_histories[method][-1]),
                "scale_boundary": boundary_summary,
            }

        if not all(
            np.all(np.isfinite(np.asarray(values, dtype=np.float64)))
            for values in scale_histories.values()
        ):
            raise SequentialNumericalFailure("DTC 尺度时序含非有限值。")

        record.update(
            {
                "status": "PASS",
                "length": requested_length,
                "change_index_zero_based": change_index,
                "design": {
                    "rank": design_rank,
                    "required_rank": state_dimension,
                    "rank_tolerance": design_rank_tolerance,
                    "fixed_within_instance": True,
                },
                "factor_block_count": factor_count,
                "exact_block_traces": [float(value) for value in exact_traces],
                "exact_trace_sum": float(
                    np.sum(exact_traces, dtype=np.float64)
                ),
                "actual_mean_scalar_leverage": float(
                    np.sum(exact_traces, dtype=np.float64) / total_dimension
                ),
                "spectral_diagnostics": spectral_diagnostics,
                "method_summaries": method_summaries,
                "time_series": time_series,
                "interpretation": "descriptive_dtc_response_only",
                "eligible_for_k_selection": False,
            }
        )
    except (
        SequentialNumericalFailure,
        LinearNumericalFailure,
        np.linalg.LinAlgError,
        FloatingPointError,
    ) as error:
        record.update(
            {
                "status": "NUMERICAL_FAILURE",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
    return record


__all__ = [
    "DTCTracker",
    "SequentialNumericalFailure",
    "SequentialSetting",
    "enumerate_sequential_settings",
    "run_sequential_instance",
]
