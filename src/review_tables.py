"""Reader-facing tables from the frozen-condition and credit summaries."""
import numpy as np
import pandas as pd

from common import ROOT
from summarize import save

OUT = ROOT / "results/formal"
METHOD = {"lasso":"Lasso","elastic":"Elastic Net","ridge":"Ridge"}


def condition_table():
    root = OUT/"main"
    df = pd.read_csv(root/"conditions.csv")
    df["order"] = df.method.map({"lasso":0,"elastic":1,"ridge":2})
    rows = []
    for r in df.sort_values(["condition","order"]).itertuples():
        if r.method == "ridge":
            continue
        rows.append([r.n, f"{r.rho_signal:g} / {r.rho_null:g}", METHOD[r.method], f"{r.rmse_mean:.3f}",
            f"{r.j_var_mean:.3f} / {r.j_group_mean:.3f}",
            f"{int(r.close_n)} / {int(r.low_n)}",
            f"{int(r.both_n)}/{int(r.total)} ({100*r.joint_lo:.1f}–{100*r.joint_hi:.1f}%)",
            f"{int(r.both_n)}/{int(r.close_n)}", f"{r.size_mean:.1f}",
            f"{r.recall_mean:.3f}",f"{r.false_vars_mean:.1f}"])
    columns=["n","ρ signal / null","Method","Mean RMSE","Mean J, variable / group",
             "Close / low counts","Joint count (95% CI)","Low among close","Mean size","Recall","False inclusions"]
    lines=["# Formal simulation by condition","",
        "The target is 200 independent sample pairs in each condition. The denominator in the joint-count column gives the number of valid complete pairs. RMSE, support size, recall and false inclusions are averaged across the two training samples in each pair. Close means an absolute population RMSE gap ≤ 0.05. Low means variable Jaccard ≤ 0.5. The joint count is shown with its 95% Wilson interval for the probability.","",
        "| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    lines.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
    lines+=["","The last four rows are the two additional mechanism controls at n = 80. The [full condition table](main/conditions.csv) also includes the Ridge baseline, double-empty rates, both marginal probabilities, continuous distributions, common signal-group hits and complete swaps. See [threshold sensitivities](main/thresholds.csv) for all event counts and intervals.",""]
    (OUT/"condition_summary.md").write_text("\n".join(lines))


def mechanism_table():
    rows=[]
    for stage in ("main","fixed","replication","ar"):
        pairs=pd.read_csv(OUT/stage/"pairs.csv")
        for keys,pp in pairs.groupby(["condition","n","rho_signal","rho_null","method"]):
            rec=dict(zip(["condition","n","rho_signal","rho_null","method"],keys))
            common=int(pp.jointly_hit.sum())
            total=3*len(pp)
            state={k:int(pp[k].sum()) for k in ["neither","one","disjoint","partial","identical","identical_true"]}
            assert sum(state[k] for k in ["neither","one","disjoint","partial","identical"])==total
            rows.append(dict(stage=stage,**rec,**state,pairs=len(pp),block_pairs=total,
                jointly_hit=common,identical_only_companions=state["identical"]-state["identical_true"],
                swap_given_joint_hits=state["disjoint"]/common if common else np.nan,
                signal_diff_mean=pp.signal_diff.mean(),null_diff_mean=pp.null_diff.mean(),
                all_pair_prediction_disagreement_median=pp.disagreement.median(),
                close_pair_prediction_disagreement_median=pp.loc[pp.gap<=.05,"disagreement"].median()))
    save(pd.DataFrame(rows),OUT,"mechanisms")


def credit_table():
    rows=[]
    for stage in ("credit","credit_raw"):
        folder=OUT/stage
        fits=pd.read_csv(folder/"credit_fits.csv")
        pairs=pd.read_csv(folder/"credit_pairs.csv")
        events=pd.read_csv(folder/"credit_events.csv")
        for method,ff in fits.groupby("method"):
            row=dict(encoding=stage, method=method, fits=len(ff),mean_log_loss=ff.loss.mean(),
                     mean_brier=ff.brier.mean(),mean_size=ff["size"].mean())
            if method!="baseline":
                pp=pairs[pairs.method==method]
                e=events[(events.method==method)&(events.multiplier==1)].iloc[0]
                row.update(mean_variable_j=pp.j_var.mean(),mean_group_j=pp.j_group.mean(),
                           median_gap=pp.gap.median(),gap_q25=pp.gap.quantile(.25),gap_q75=pp.gap.quantile(.75),
                           close_n=int(e.close),both_n=int(e.joint),total=int(e.total),
                           double_empty=float(pp.double_empty.mean()),boundary=int(ff.boundary.eq(True).sum()))
            rows.append(row)
    save(pd.DataFrame(rows),OUT,"credit_models")


if __name__=="__main__":
    condition_table()
    mechanism_table()
    credit_table()
