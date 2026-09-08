from __future__ import annotations

from typing import Any, Iterable, Union

import numpy as np


class StatisticsProtocolError(ValueError):
    """Raised rather than silently dropping or repairing invalid formal values."""


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise StatisticsProtocolError(f"{name} must be a non-empty one-dimensional sample")
    if not np.all(np.isfinite(vector)):
        raise StatisticsProtocolError(f"{name} contains a non-finite value")
    return vector


def type7_quantile(values: Iterable[float], probability: float) -> float:
    """Hyndman-Fan Type 7: h=(n-1)p with linear interpolation."""
    sample = np.sort(_finite_vector(values, "values"))
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise StatisticsProtocolError("probability must be a JSON number")
    p = float(probability)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise StatisticsProtocolError("probability must be finite and lie in [0, 1]")
    position = (sample.size - 1) * p
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(sample[lower])
    weight = position - lower
    return float((1.0 - weight) * sample[lower] + weight * sample[upper])


def median_and_iqr(values: Iterable[float]) -> dict[str, Union[float, int]]:
    sample = _finite_vector(values, "values")
    q1 = type7_quantile(sample, 0.25)
    median = type7_quantile(sample, 0.5)
    q3 = type7_quantile(sample, 0.75)
    return {
        "count": int(sample.size),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def relative_median_drop(raw_errors: Iterable[float], dpr_errors: Iterable[float]) -> dict[str, Any]:
    raw = _finite_vector(raw_errors, "raw_errors")
    dpr = _finite_vector(dpr_errors, "dpr_errors")
    if raw.size != dpr.size:
        raise StatisticsProtocolError("paired raw and DPR errors must have equal length")
    raw_median = type7_quantile(raw, 0.5)
    dpr_median = type7_quantile(dpr, 0.5)
    if raw_median <= 0.0:
        return {
            "valid": False,
            "reason": "zero_or_negative_raw_median_error",
            "raw_median": raw_median,
            "dpr_median": dpr_median,
            "relative_drop": None,
        }
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_drop = float(
            1.0 - np.float64(dpr_median) / np.float64(raw_median)
        )
    if not np.isfinite(relative_drop):
        return {
            "valid": False,
            "reason": "nonfinite_relative_drop",
            "raw_median": raw_median,
            "dpr_median": dpr_median,
            "relative_drop": None,
        }
    return {
        "valid": True,
        "reason": None,
        "raw_median": raw_median,
        "dpr_median": dpr_median,
        "relative_drop": relative_drop,
    }


def paired_bootstrap_difference_of_medians(
    raw_values: Iterable[float],
    dpr_values: Iterable[float],
    *,
    repeats: int = 10_000,
    seed: int = 91_027,
    upper_probability: float = 0.95,
) -> dict[str, Any]:
    """Paired graph bootstrap; every call resets an independent PCG64 generator."""
    raw = _finite_vector(raw_values, "raw_values")
    dpr = _finite_vector(dpr_values, "dpr_values")
    if raw.size != dpr.size:
        raise StatisticsProtocolError("paired bootstrap inputs must have equal length")
    count = int(raw.size)
    if type(repeats) is not int or repeats <= 0:
        raise StatisticsProtocolError("bootstrap repeats must be a positive JSON integer")
    if type(seed) is not int or seed < 0:
        raise StatisticsProtocolError("bootstrap seed must be a non-negative JSON integer")
    if isinstance(upper_probability, bool) or not isinstance(
        upper_probability, (int, float)
    ):
        raise StatisticsProtocolError("upper_probability must be a JSON number")
    upper_p = float(upper_probability)
    if not np.isfinite(upper_p) or upper_p < 0.0 or upper_p > 1.0:
        raise StatisticsProtocolError(
            "upper_probability must be finite and lie in [0, 1]"
        )
    repetitions = repeats
    generator = np.random.Generator(np.random.PCG64(seed))
    statistics = np.empty(repetitions, dtype=np.float64)
    for repeat_index in range(repetitions):
        indices = generator.integers(0, count, size=count)
        statistics[repeat_index] = float(
            np.median(dpr[indices]) - np.median(raw[indices])
        )
    if not np.all(np.isfinite(statistics)):
        raise StatisticsProtocolError("paired bootstrap generated a non-finite statistic")
    return {
        "resampling_unit": "paired_graph",
        "statistic": "median_dpr_minus_median_raw",
        "sample_count": count,
        "repeats": repetitions,
        "bit_generator": "PCG64",
        "seed": int(seed),
        "quantile_interpolation": "Hyndman-Fan_Type_7",
        "upper_probability": upper_p,
        "observed_difference": float(np.median(dpr) - np.median(raw)),
        "upper_bound": type7_quantile(statistics, upper_p),
    }
