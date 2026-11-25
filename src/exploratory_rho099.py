"""Run, rebuild or verify the two exploratory high-correlation controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, atomic_json, seed
from package_records import package, restore
from simulation import evaluate_pair
from summarize import sim_tables, save

SPEC = json.loads((ROOT / "config/exploratory-rho099.json").read_text())
FORMAL = ROOT / "results/formal/main"
OUT = ROOT / "results/exploratory/rho099"
NAMES = {0: "Independent", 10: "Signal blocks only", 11: "Null blocks only", 3: "All blocks"}
ORDER = list(NAMES)


def fit(task):
    condition, pair, records = task
    identity = condition["id"]
    design = tuple(condition[k] for k in ("n", "rho_signal", "rho_null"))
    try:
        evaluate_pair(identity, pair, SPEC["root_seed"], records, design=design, check=(pair == 0))
    except RuntimeError as error:
        if "path failed convergence after retry" not in str(error):
            raise
        return dict(condition=identity, pair=pair, error=str(error))


def fit_all(folder, jobs):
    records = folder / "records"
    records.mkdir(parents=True)
    tasks = [(condition, i, records) for condition in SPEC["conditions"]
             for i in range(SPEC["pairs_per_condition"])]
    with ProcessPoolExecutor(jobs) as workers:
        failures = [error for error in workers.map(fit, tasks) if error is not None]
    atomic_json(folder / "run.json", dict(label=SPEC["label"], planned=len(tasks),
                valid=len(tasks)-len(failures), failures=failures, root_seed=SPEC["root_seed"],
                frozen_sha256=SPEC["frozen_sha256"]))
    package("rho099", base=folder.parent)


def audit(folder):
    """Independently score slopes with block sums and recompute counts from supports."""
    rows = pd.read_csv(folder / "conditions.csv").set_index(["condition","method"])
    tallies = {}
    seen_seeds = set()
    original_seeds = set()
    with gzip.open(FORMAL / "fits.jsonl.gz", "rt") as stream:
        for line in stream:
            for sample in json.loads(line)["samples"]:
                original_seeds.update([sample["data_seed"], sample["fold_seed"]])
    max_error = 0.
    truth = np.array([2, 7, 12])
    for path in sorted((folder / "records").glob("*.json")):
        pair = json.loads(path.read_text())
        idx = pair["condition"]
        rho = np.array([pair["rho_signal"]] * 3 + [pair["rho_null"]] * 21)
        for side, sample in enumerate(pair["samples"]):
            for k, field in enumerate(("data_seed","fold_seed")):
                value = sample[field]
                assert value == seed(SPEC["root_seed"], idx, pair["pair"], side, k)
                assert value not in seen_seeds and value not in original_seeds
                seen_seeds.add(value)
        for method in ("lasso","elastic","ridge"):
            risks, supports = [], []
            for sample in pair["samples"]:
                model = sample["fits"][method]
                slopes = np.array(model["coef"])
                supports.append(set(slopes.nonzero()[0]))
                slopes[truth] -= 1
                blocks = slopes.reshape(24, 5)
                energy = (1-rho) * np.sum(blocks**2,axis=1) + rho * np.sum(blocks,axis=1)**2
                risk = np.sqrt(1 + model["intercept"]**2 + energy.sum())
                risks.append(risk)
                max_error = max(max_error, abs(risk-model["rmse"]))
            if method == "ridge":
                continue
            a, b = supports
            low = bool(a|b) and 2*len(a&b) <= len(a|b)
            close = abs(risks[0]-risks[1]) <= .05
            hits,swaps = 0,0
            for g in range(3):
                block = set(range(5*g,5*g+5))
                if a&block and b&block:
                    hits += 1
                    swaps += int(not (a&b&block))
            t = tallies.setdefault((idx,method),np.zeros(6,dtype=int))
            t += [1, close, low, close and low, hits, swaps]
    assert max_error < 1e-10
    columns = ["total","close_n","low_n","both_n","joint_group_hits","complete_swaps"]
    for key,t in tallies.items():
        np.testing.assert_array_equal(t, rows.loc[key, columns].to_numpy(dtype=int))
    atomic_json(folder / "audit.json", dict(unique_data_and_fold_seeds=len(seen_seeds),
                overlapping_formal_seeds=0, checked_counts=columns, max_rmse_error=max_error))


def four_cells(folder):
    reference = pd.read_csv(FORMAL / "conditions.csv")
    extra = pd.read_csv(folder / "conditions.csv")
    cells = pd.concat([reference[reference.condition.isin([0,3])], extra], ignore_index=True)
    cells = cells[cells.method.isin(["lasso","elastic"])].copy()
    cells.insert(0, "cell", cells.condition.map(NAMES))
    cells.insert(1, "source", cells.condition.map(lambda c: "exploratory" if c in (10,11) else "formal reference"))
    cols = ["cell","source","method","condition","rho_signal","rho_null","total","close_n","low_n","both_n",
            "joint","joint_lo","joint_hi","conditional","cond_lo","cond_hi","joint_group_hits","complete_swaps",
            "complete_swap","swap_lo","swap_hi","signal_diff_mean","null_diff_mean","rmse_mean","j_var_mean",
            "j_group_mean","size_mean","recall_mean","false_vars_mean"]
    cells["order"] = cells.condition.map({c:i for i,c in enumerate(ORDER)})
    cells=cells.sort_values(["order","method"])[cols]
    save(cells, folder, "four_cells")
    return cells


def contrasts(folder):
    formal = pd.read_csv(FORMAL / "pairs.csv")
    added = pd.read_csv(folder / "pairs.csv")
    pairs = pd.concat([formal[formal.condition.isin([0,3])], added], ignore_index=True)
    pairs = pairs[pairs.method=="lasso"]
    rng = np.random.default_rng(SPEC["bootstrap_seed"])
    rates, boot = {}, {}
    for c in ORDER:
        v = pairs.loc[pairs.condition==c, ["disjoint","jointly_hit"]].to_numpy()
        numer, denom = v.sum(axis=0)
        rates[c] = numer / denom
        resampled = v[rng.integers(len(v),size=(SPEC["bootstrap_draws"], len(v)))].sum(axis=1)
        boot[c] = np.divide(resampled[:,0],resampled[:,1],out=np.full(len(resampled),np.nan),
                            where=resampled[:,1]>0)
    rows = []
    for other in [3,11]:
        diff = boot[10]-boot[other]
        lo, hi = np.nanquantile(diff,[.025,.975])
        rows.append(dict(first=NAMES[10], second=NAMES[other], difference=rates[10]-rates[other],
                         lower=lo, upper=hi, bootstrap_draws=int(np.isfinite(diff).sum())))
    save(pd.DataFrame(rows),folder,"swap_contrasts")


def mechanism_plot(cells,folder):
    subset=cells[cells.method=="lasso"].set_index("condition").loc[ORDER]
    fig, (left,right) = plt.subplots(1,2,figsize=(10.4,3.35))
    fig.subplots_adjust(left=.205,right=.97,top=.72,bottom=.19,wspace=.43)
    fig.text(.025,.91,"Where Lasso selections differ at high correlation",fontsize=15,weight="bold")
    fig.text(.025,.82,"Exploratory comparison  ·  n = 80  ·  High within-block correlation = 0.99",fontsize=10)
    y=np.arange(4)
    for pos, (_,r) in enumerate(subset.iterrows()):
        color="#20639b" if r.source=="exploratory" else "#697583"
        left.hlines(pos,r.swap_lo,r.swap_hi,color=color,lw=2)
        left.plot(r.complete_swap,pos,"o",color=color)
        left.text(1.02,pos,f"{int(r.complete_swaps)}/{int(r.joint_group_hits)}",
                  transform=left.get_yaxis_transform(),va="center",fontsize=9,color=color)
    left.set_xlim(-.015,max(.52,subset.swap_hi.max()+.03))
    left.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
    left.set_yticks(y,subset.cell)
    left.set_xlabel("Complete substitutions / jointly hit blocks",fontsize=9)
    right.barh(y,subset.signal_diff_mean,color="#337d86",height=.45,label="Signal blocks")
    right.barh(y,subset.null_diff_mean,left=subset.signal_diff_mean,color="#c9d0d9",height=.45,label="Null blocks")
    right.legend(loc="upper right",bbox_to_anchor=(1.02,1.24),frameon=False,ncol=2,fontsize=8)
    right.set_yticks(y,[""]*4)
    right.set_xlabel("Mean number of differing variables",fontsize=9)
    for ax in (left,right):
        ax.set_ylim(3.54,-.54)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x",alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(length=0,labelsize=9)
    fig.savefig(folder/"mechanism.png",dpi=165)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=2)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild",action="store_true",help="Recreate summaries from the archived fits")
    mode.add_argument("--verify-refit",action="store_true",help="Refit all pairs and compare every output")
    args=parser.parse_args()
    frozen=hashlib.sha256((ROOT/"config/frozen.json").read_bytes()).hexdigest()
    if frozen!=SPEC["frozen_sha256"]:
        raise ValueError("The formal configuration differs from the exploratory reference")
    (ROOT/"work").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"work") as temp:
        folder=Path(temp)/"rho099"
        folder.mkdir()
        if args.rebuild:
            for name in ["archive.json","fits.jsonl.gz","run.json"]:
                shutil.copy2(OUT/name,folder/name)
            restore("rho099",base=folder.parent)
        else:
            fit_all(folder,args.jobs)
        sim_tables(folder/"records",folder)
        audit(folder)
        cells=four_cells(folder)
        contrasts(folder)
        mechanism_plot(cells,folder)
        OUT.mkdir(parents=True,exist_ok=True)
        files=sorted(p for p in folder.iterdir() if p.is_file())
        for path in files:
            if args.verify_refit:
                if path.read_bytes()!=(OUT/path.name).read_bytes():
                    raise ValueError(f"Refit differs for {path.name}")
            else:
                os.replace(path,OUT/path.name)
        print(f"{'Verified' if args.verify_refit else 'Generated'} {len(files)} study files")


if __name__=="__main__":
    main()
