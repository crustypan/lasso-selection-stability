"""Exploratory sensitivity and covariance diagnostics from original-scale fits."""
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from common import ROOT
from summarize import wilson


SOURCES = {
    "formal": ROOT / "results/formal/main",
    "exploratory": ROOT / "results/exploratory/rho099",
}
OUT = ROOT / "results/exploratory/interpretation"
GAPS = (0.02, 0.05, 0.10)
OVERLAPS = (0.25, 0.50, 0.75)


def paired_observations():
    rows = []
    settings = json.loads((ROOT / "config/frozen.json").read_text())
    width = settings["width"]
    truth = settings["truth_zero_based"]
    signal_blocks = [t // width for t in truth]
    for source, folder in SOURCES.items():
        archive = folder / "fits.jsonl.gz"
        meta = json.loads((folder / "archive.json").read_text())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != meta["sha256"]:
            raise ValueError(f"Archive digest mismatch in {folder}")
        expected = pd.read_csv(folder / "pairs.csv").set_index(["condition", "pair", "method"])
        start = len(rows)
        with gzip.open(archive, "rt") as stream:
            for line in stream:
                pair = json.loads(line)
                c, i = pair["condition"], pair["pair"]
                rs, rn = pair["rho_signal"], pair["rho_null"]
                rho = np.full(settings["groups"], float(rn))
                rho[signal_blocks] = rs
                for method in ("lasso", "elastic"):
                    a, b = [s["fits"][method] for s in pair["samples"]]
                    wa, wb = (np.asarray(m["coef"]) for m in (a, b))
                    sa, sb = wa != 0, wb != 0
                    union = (sa | sb).sum()
                    j = (sa & sb).sum() / union if union else 1.0
                    delta = (wa - wb).reshape(-1, width)
                    centers = delta.mean(axis=1)
                    level = width * centers**2
                    contrast = ((delta - centers[:, None])**2).sum(axis=1)
                    level_risk = (1 + (width - 1) * rho) * level
                    contrast_risk = (1 - rho) * contrast
                    squared_disagreement = (a["intercept"] - b["intercept"])**2 + (level_risk + contrast_risk).sum()
                    ref = expected.loc[(c, i, method)]
                    np.testing.assert_allclose(np.sqrt(squared_disagreement), ref.disagreement, atol=1e-8, rtol=0)
                    np.testing.assert_allclose(j, ref.j_var, atol=1e-9, rtol=0)

                    left = sa.reshape(-1, width)[signal_blocks]
                    right = sb.reshape(-1, width)[signal_blocks]
                    both_hit = left.any(axis=1) & right.any(axis=1)
                    disjoint = both_hit & ~(left & right).any(axis=1)
                    one_true = np.count_nonzero(disjoint & (sa[truth] ^ sb[truth]))
                    neither_true = np.count_nonzero(disjoint & ~(sa[truth] | sb[truth]))
                    if one_true + neither_true != ref.disjoint or both_hit.sum() != ref.jointly_hit:
                        raise ValueError(f"Signal-block tally mismatch for {source}, {c}, {i}, {method}")
                    rows.append(dict(source=source, condition=c, n=pair["n"], rho_signal=rs, rho_null=rn,
                        pair=i, method=method, gap=abs(a["rmse"] - b["rmse"]), jaccard=j,
                        signal_level=level[signal_blocks].sum(), signal_contrast=contrast[signal_blocks].sum(),
                        risk_level=level_risk[signal_blocks].sum(), risk_contrast=contrast_risk[signal_blocks].sum(),
                        jointly_hit=int(both_hit.sum()), swap_one_true=one_true, swap_neither_true=neither_true))
        if len(rows) - start != 2 * meta["rows"] or len(expected) != 2 * meta["rows"]:
            raise ValueError(f"Unexpected pair count in {source}")
    return pd.DataFrame(rows)


def summarize(frame):
    thresholds, geometry = [], []
    keys = ["source", "condition", "n", "rho_signal", "rho_null", "method"]
    for values, group in frame.groupby(keys):
        info = dict(zip(keys, values))
        for gap in GAPS:
            close = group.gap <= gap
            for cutoff in OVERLAPS:
                both = int((close & (group.jaccard <= cutoff)).sum())
                nclose, n = int(close.sum()), len(group)
                jl, jh = wilson(both, n)
                cl, ch = wilson(both, nclose)
                thresholds.append(dict(**info, gap_limit=gap, j_limit=cutoff, valid=n, close=nclose,
                    joint_n=both, joint=both/n, joint_lo=jl, joint_hi=jh,
                    conditional=both/nclose if nclose else np.nan, conditional_lo=cl, conditional_hi=ch))
        sums = group[["signal_level","signal_contrast","risk_level","risk_contrast",
                      "jointly_hit","swap_one_true","swap_neither_true"]].sum()
        coef_distance = sums.signal_level + sums.signal_contrast
        prediction_distance = sums.risk_level + sums.risk_contrast
        geometry.append(dict(**info, valid=len(group), jointly_hit=int(sums.jointly_hit),
            swap_one_true=int(sums.swap_one_true), swap_neither_true=int(sums.swap_neither_true),
            mean_signal_coef_distance_sq=coef_distance/len(group),
            mean_signal_prediction_distance_sq=prediction_distance/len(group),
            contrast_share_coef=sums.signal_contrast/coef_distance if coef_distance else np.nan,
            contrast_share_prediction=sums.risk_contrast/prediction_distance if prediction_distance else np.nan))
    thresholds = pd.DataFrame(thresholds)
    for source, folder in SOURCES.items():
        old = pd.read_csv(folder / "thresholds.csv").sort_values(["condition","method","threshold"])
        new = thresholds[(thresholds.source == source) & (thresholds.j_limit == .5)]
        new = new.sort_values(["condition","method","gap_limit"])
        np.testing.assert_array_equal(old[["total","close_n","both_n"]].to_numpy(), new[["valid","close","joint_n"]].to_numpy())
    return thresholds, pd.DataFrame(geometry)


def plot_overlap(frame):
    selected = frame[(frame.method == "lasso") & (frame.source == "formal") &
                     (frame.rho_signal == frame.rho_null) & (frame.gap <= .05)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    colors = ["#243f54", "#498781", "#b66120", "#873957"]
    styles = ["-", "--", "-.", ":"]
    for ax, (n, data) in zip(axes, selected.groupby("n")):
        for (rho, cell), color, style in zip(data.groupby("rho_signal"), colors, styles):
            values, counts = np.unique(cell.jaccard, return_counts=True)
            ax.step(np.r_[0,values,1], np.r_[0,counts.cumsum()/len(cell),1], where="post",
                    color=color, linestyle=style, lw=1.8, label=f"ρ = {rho:g}   ({len(cell)} pairs)")
        ax.axvline(.5, linestyle="--", color="#7a8288", lw=1, zorder=0)
        ax.set(xlim=(0,1), ylim=(-.02,1.025), xlabel="Variable Jaccard cutoff", title=f"Training sample size {n}")
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1"])
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.17)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Share at or below cutoff")
    fig.suptitle("Lasso overlap among pairs with close prediction error", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "overlap_distribution.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    observations = paired_observations()
    sensitivities, geometry = summarize(observations)
    sensitivities.to_csv(OUT / "threshold_grid.csv", index=False, float_format="%.10g")
    geometry.to_csv(OUT / "signal_blocks.csv", index=False, float_format="%.10g")
    plot_overlap(observations)
    print(f"Checked {len(observations)} sparse-method pair records and wrote {len(sensitivities)} threshold cells.")
