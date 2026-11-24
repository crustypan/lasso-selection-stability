"""Aggregate frozen-design results and compare the prespecified conditions."""
import json

import numpy as np
import pandas as pd

from common import ROOT
from formal import STAGES,load_config
from summarize import sim_tables,sim_plot,credit_tables,save,wilson


def endpoint_table(p):
    rows=[]
    for (n,m),z in p[p.rho_signal==p.rho_null].groupby(["n","method"]):
        parts=[z[z.rho_signal==r] for r in [0,.9]]
        if any(v.empty for v in parts): continue
        counts=[((v.gap<=.05)&(v.j_var<=.5)).sum() for v in parts]
        ns=[len(v) for v in parts]
        p0,p1=[x/n for x,n in zip(counts,ns)]
        (l0,h0),(l1,h1)=[wilson(x,n) for x,n in zip(counts,ns)]
        diff=p1-p0
        rows.append(dict(n=n,method=m,low_count=counts[0],low_denominator=ns[0],
                         high_count=counts[1],high_denominator=ns[1],difference=diff,
                         lower=diff-np.hypot(p1-l1,h0-p0),upper=diff+np.hypot(h1-p1,p0-l0)))
    return pd.DataFrame(rows)


if __name__=="__main__":
    cfg=load_config()
    root=ROOT/"results/formal"
    for stage in STAGES:
        out=root/stage
        status=json.loads((out/"run.json").read_text())
        if status["valid"]==0: raise ValueError(f"No valid pairs for {stage}")
        if stage.startswith("credit"):
            credit_tables(out/"records",out,"Formal" if stage=="credit" else "Raw status")
            f=pd.read_csv(out/"credit_fits.csv")
            save(f.groupby("method")[["loss","brier","size"]].agg(["mean","median","std"]).reset_index(),
                 out,"credit_fit_summary")
        else:
            s=sim_tables(out/"records",out)
            p=pd.read_csv(out/"pairs.csv")
            table=endpoint_table(p)
            if not table.empty: save(table,out,"endpoint_contrasts")
            if stage=="main": sim_plot(s,out,"Formal")
