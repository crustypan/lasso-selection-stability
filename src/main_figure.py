"""Render the formal figure from its condition table, with explicit event denominators."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from common import ROOT
from summarize import COLORS, wilson


METHODS = ("lasso", "elastic")
RHO = [0, .5, .9, .99]
RHO_LABELS = ["0", "0.5", "0.9", "0.99"]


def event_panel(ax, frame, denominator, show_rho):
    for method, offset in zip(METHODS, [-.16, .16]):
        data = frame[frame.method == method].sort_values("rho_signal")
        for i, row in enumerate(data.itertuples()):
            hits, total = int(row.both_n), int(getattr(row, denominator))
            y = i + offset
            label = f"{hits}/{total}" if total else "0/0 (n.e.)"
            ax.text(1.035, y, label, transform=ax.get_yaxis_transform(), va="center",
                    color=COLORS[method], fontsize=9, clip_on=False)
            if total:
                rate = hits / total
                lo, hi = wilson(hits, total)
                ax.hlines(y, lo, hi, color=COLORS[method], linewidth=1.7)
                ax.plot(rate, y, "o" if method == "lasso" else "s",
                        color=COLORS[method], markersize=5)
    ax.set_xlim(-.025, 1.025)
    ax.set_ylim(3.44, -.44)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks(range(4), RHO_LABELS)
    ax.tick_params(axis="y", labelleft=show_rho, length=0, pad=8)
    ax.tick_params(axis="x", length=0, labelsize=9)
    ax.grid(axis="x", color="#d9dfe6", lw=.7)
    for y in [.5, 1.5, 2.5]:
        ax.axhline(y, color="#eef1f4", lw=.65, zorder=0)
    ax.text(1.035, 1.025, "Count", transform=ax.transAxes, color="#536170", fontsize=8)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)


def diagnostics(ax, frame, metric):
    for method in (*METHODS, "ridge"):
        z = frame[frame.method == method].sort_values("rho_signal")
        if metric == "rmse":
            ax.plot(range(4), z.rmse_q50, "o-", lw=1.6, ms=4, color=COLORS[method])
        elif method in METHODS:
            ax.plot(range(4), z.j_var_mean, "o-", lw=1.6, ms=4, color=COLORS[method])
            ax.plot(range(4), z.j_group_mean, "s--", lw=1.6, ms=3.5, color=COLORS[method])
    if metric == "rmse":
        ax.set_ylim(.95, 1.8)
        ax.set_yticks([1, 1.25, 1.5, 1.75])
        ax.axhline(1, ls=":", color="#64748b", lw=1)
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
    ax.set_xticks(range(4), RHO_LABELS)
    ax.set_xlabel("Within-block correlation", fontsize=8)
    ax.tick_params(labelsize=8, length=0)
    ax.grid(axis="y", color="#dee3e8", lw=.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c5cdd5")


def plot_formal(frame, destination):
    equal = frame[frame.rho_signal == frame.rho_null]
    for n in (80, 240):
        for method in (*METHODS, "ridge"):
            subset = equal[(equal.n == n) & (equal.method == method)]
            if sorted(subset.rho_signal) != RHO:
                raise ValueError(f"Missing or duplicate formal conditions for {n}, {method}")
    fig = plt.figure(figsize=(11.6, 8.4), facecolor="white")
    fig.text(.085, .965, "Similar prediction errors and low selection overlap",
             fontsize=18, weight="bold", color="#192736")
    fig.text(.085, .93, "Close error: population RMSE gap ≤ 0.05     •     Low overlap: variable Jaccard ≤ 0.5",
             fontsize=10, color="#48586a")
    handles = [Line2D([], [], color=COLORS[m], marker=k, linestyle="none",
                      label=t, markersize=6) for m, k, t in
               [("lasso", "o", "Lasso"), ("elastic", "s", "Elastic Net"), ("ridge", "o", "Ridge (RMSE only)")]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.076, .919),
               frameon=False, ncol=3, handletextpad=.15, columnspacing=1.4, fontsize=9)
    for left, title, subtitle in [(.085, "Joint proportion", "Both events / all valid pairs"),
                                  (.57, "Conditional proportion", "Both events / pairs with close error")]:
        fig.text(left, .849, title, fontsize=14, weight="bold")
        fig.text(left, .824, subtitle, fontsize=10, color="#48586a")
    for n, bottom in [(80, .60), (240, .37)]:
        part = equal[equal.n == n]
        for col, (left, denom) in enumerate([(.085, "total"), (.57, "close_n")]):
            ax = fig.add_axes([left, bottom, .324, .182])
            event_panel(ax, part, denom, show_rho=True)
            if col == 0:
                ax.set_ylabel(f"n = {n}\nCorrelation", fontsize=9, labelpad=8)
    fig.text(.085, .308, "Prediction error and selection stability", fontsize=13, weight="bold")
    fig.text(.085, .283, "RMSE: dotted line = oracle.     Jaccard: solid = variable, dashed = group.",
             fontsize=9, color="#48586a")
    for n, left in [(80, .085), (240, .565)]:
        for metric, dx, title in [("rmse", 0, "Median RMSE"), ("jaccard", .230, "Mean Jaccard")]:
            ax = fig.add_axes([left+dx, .086, .17, .14])
            diagnostics(ax, equal[equal.n == n], metric)
            ax.set_title(f"n = {n}  ·  {title}", fontsize=9, loc="left", pad=6)
    fig.text(.085, .014, "Intervals are 95% Wilson intervals. Counts show the numerator and denominator of each estimate.",
             fontsize=8.5, color="#48586a")
    fig.savefig(destination, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    default = ROOT / "results/formal/main"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, default=default / "conditions.csv")
    parser.add_argument("--output", type=Path, default=default / "simulation.png")
    args = parser.parse_args()
    plot_formal(pd.read_csv(args.conditions), args.output)
