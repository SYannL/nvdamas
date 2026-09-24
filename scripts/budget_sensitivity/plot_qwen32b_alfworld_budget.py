#!/usr/bin/env python3
"""Plot Seen/Unseen success against the maximum ALFWorld interaction budget."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/nvdamas-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {"ICL": "#5b6770", "MemCo": "#9f6262"}
MARKERS = {"ICL": "s", "MemCo": "o"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render(rows: list[dict[str, str]], output_dir: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIX", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=True)
    for panel_index, (axis, split) in enumerate(zip(axes, ("Seen", "Unseen"))):
        for method in ("ICL", "MemCo"):
            selected = sorted(
                (row for row in rows if row["method"] == method and row["split"] == split),
                key=lambda row: int(row["budget"]),
            )
            axis.plot(
                [int(row["budget"]) for row in selected],
                [float(row["success_rate_pct"]) for row in selected],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=5.4,
                markeredgecolor="white",
                markeredgewidth=0.75,
                linewidth=2.0,
                label=method,
                zorder=3,
            )
        axis.set_xlabel("Maximum interaction budget")
        axis.set_xticks([10, 30, 50])
        axis.grid(axis="both", color="#dedede", linewidth=0.65, linestyle="--", zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.5,
            -0.22,
            f"({chr(ord('a') + panel_index)}) {split}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9.0,
            fontweight="normal",
            clip_on=False,
        )
    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.4)
    fig.subplots_adjust(bottom=0.24)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "qwen32b_alfworld_max_budget_sensitivity"
    for suffix in ("pdf", "svg", "png"):
        kwargs: dict[str, float] = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == "png":
            kwargs["dpi"] = dpi
        fig.savefig(stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    render(load_rows(args.summary.resolve()), args.output_dir.resolve(), args.dpi)


if __name__ == "__main__":
    main()
