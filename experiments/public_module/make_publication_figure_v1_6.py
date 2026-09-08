"""Create the Applied Sciences public-module result figure from reviewed data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.public_module.result_validation import load_reviewed_aggregate


PALETTE = {
    "duc": "#0072B2",
    "exact": "#E69F00",
    "neutral": "#6B7280",
    "light": "#D1D5DB",
    "zero": "#4B5563",
}
SEQUENCE_LABELS = {
    "V1_01_easy": "V1-01",
    "V1_02_medium": "V1-02",
    "V1_03_difficult": "V1-03",
    "V2_01_easy": "V2-01",
    "V2_02_medium": "V2-02",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_source_data(
    path: Path,
    gate: dict[str, object],
    summary_rows: list[dict[str, str]],
) -> None:
    records: list[dict[str, object]] = []
    intervals = gate["descriptive_cluster_bootstrap_95_percentile_intervals"]
    metrics = (
        ("a", "ce95_difference", "ce95"),
        ("b", "absolute_log_scale_ratio_difference", "absolute_log_scale_ratio"),
        ("c", "holdout_nll_difference", "holdout_nll"),
    )
    for panel, effect_key, interval_key in metrics:
        interval = intervals[interval_key]
        for effect in gate["sequence_effects"]:
            records.append(
                {
                    "panel": panel,
                    "sequence": effect["sequence"],
                    "method": "DUC-K16 minus Raw",
                    "value": effect[effect_key],
                    "median": gate["equal_sequence_median_differences"][interval_key],
                    "interval_low": interval[0],
                    "interval_high": interval[1],
                    "n_definition": "independent sequence",
                    "n": 5,
                }
            )
    timing_lookup = {
        (row["sequence"], row["method"]): float(row["median_calibration_time_ms"])
        for row in summary_rows
        if int(row["fit_budget"]) == 16
        and row["method"] in {"DPR-Exact-Block", "DUC-K16"}
    }
    for sequence in SEQUENCE_LABELS:
        for method in ("DPR-Exact-Block", "DUC-K16"):
            records.append(
                {
                    "panel": "d",
                    "sequence": sequence,
                    "method": method,
                    "value": timing_lookup[(sequence, method)],
                    "median": "",
                    "interval_low": "",
                    "interval_high": "",
                    "n_definition": "sequence-level median time",
                    "n": 5,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def paired_difference_panel(
    ax: plt.Axes,
    effects: list[dict[str, object]],
    effect_key: str,
    median_value: float,
    interval: list[float],
    ylabel: str,
) -> None:
    x = np.arange(len(effects), dtype=float)
    values = np.asarray([float(effect[effect_key]) for effect in effects])
    ax.axhline(0.0, color=PALETTE["zero"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(x, values, s=28, color=PALETTE["duc"], edgecolor="white", linewidth=0.6, zorder=3)
    median_x = len(effects) + 0.35
    lower = median_value - float(interval[0])
    upper = float(interval[1]) - median_value
    ax.errorbar(
        median_x,
        median_value,
        yerr=np.asarray([[lower], [upper]]),
        fmt="D",
        markersize=4.5,
        color="#111827",
        ecolor="#111827",
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    labels = [SEQUENCE_LABELS[str(effect["sequence"])] for effect in effects] + ["Median"]
    ax.set_xticks([*x, median_x], labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    margin = max(0.01, float(np.ptp(np.r_[values, interval])) * 0.25)
    ax.set_ylim(min(values.min(), interval[0]) - margin, max(0.0, values.max(), interval[1]) + margin)
    ax.text(
        0.02,
        0.04,
        "Lower favors DUC-K16",
        transform=ax.transAxes,
        color=PALETTE["neutral"],
        fontsize=6.2,
        va="bottom",
    )


def timing_panel(ax: plt.Axes, summary_rows: list[dict[str, str]]) -> None:
    methods = ("DPR-Exact-Block", "DUC-K16")
    method_labels = ("Exact block", "DUC-K16")
    colors = (PALETTE["exact"], PALETTE["duc"])
    lookup = {
        (row["sequence"], row["method"]): float(row["median_calibration_time_ms"])
        for row in summary_rows
        if int(row["fit_budget"]) == 16 and row["method"] in methods
    }
    values = np.asarray([[lookup[(sequence, method)] for method in methods] for sequence in SEQUENCE_LABELS])
    for row in values:
        ax.plot([0, 1], row, color=PALETTE["light"], linewidth=0.9, zorder=1)
    for index, color in enumerate(colors):
        ax.scatter(
            np.full(values.shape[0], index),
            values[:, index],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            index,
            np.median(values[:, index]),
            marker="_",
            s=220,
            linewidth=2.0,
            color="#111827",
            zorder=4,
        )
    exact_median, duc_median = np.median(values, axis=0)
    reduction = 100.0 * (1.0 - duc_median / exact_median)
    ax.set_xticks([0, 1], method_labels)
    ax.set_ylabel("Incremental time (ms)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        0.95,
        f"{reduction:.1f}% lower median",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#111827",
        fontsize=7,
    )
    ax.text(
        0.02,
        0.04,
        "Each line: one sequence\nQR basis reported separately",
        transform=ax.transAxes,
        color=PALETTE["neutral"],
        fontsize=6.2,
        va="bottom",
    )


def make_figure(gate_path: Path, summary_path: Path, output_stem: Path) -> None:
    gate, summary_rows = load_reviewed_aggregate(gate_path, summary_path)
    if gate.get("protocol_version") != "v1.6" or gate.get("status") != "PASS":
        raise ValueError("figure requires a reviewed PASS v1.6 gate")
    write_source_data(output_stem.with_name(output_stem.name + "_source_data.csv"), gate, summary_rows)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 112 / 25.4), constrained_layout=True)
    effects = list(gate["sequence_effects"])
    interval_map = gate["descriptive_cluster_bootstrap_95_percentile_intervals"]
    median_map = gate["equal_sequence_median_differences"]
    paired_difference_panel(
        axes[0, 0], effects, "ce95_difference", float(median_map["ce95"]), interval_map["ce95"], "Δ CE95"
    )
    paired_difference_panel(
        axes[0, 1],
        effects,
        "absolute_log_scale_ratio_difference",
        float(median_map["absolute_log_scale_ratio"]),
        interval_map["absolute_log_scale_ratio"],
        "Δ absolute log scale ratio",
    )
    paired_difference_panel(
        axes[1, 0],
        effects,
        "holdout_nll_difference",
        float(median_map["holdout_nll"]),
        interval_map["holdout_nll"],
        "Δ holdout NLL",
    )
    timing_panel(axes[1, 1], summary_rows)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.16, 1.07, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    make_figure(args.gate, args.summary, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
