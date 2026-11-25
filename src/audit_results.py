"""Recompute published metrics from saved coefficients and the held-out credit outcomes."""
import json

import numpy as np
import pandas as pd

from common import ROOT, atomic_json
from credit_data import TARGET, workbook, split, design
from scipy.special import expit
from verify_formal import FORMAL, fit_records
from formal import load_config


def audit_simulation(stage):
    tab=pd.read_csv(FORMAL/stage/"pairs.csv").set_index(["condition","pair","method"])
    fit_tab=pd.read_csv(FORMAL/stage/"fits.csv").set_index(["condition","pair","side","method"])
    primary={}
    count=0
    max_risk_error=0.
    for record in fit_records(stage):
        c,i=record["condition"],record["pair"]
        correlations=np.array([record["rho_signal"]]*3+[record["rho_null"]]*21)
        def energy(delta):
            blocks=delta.reshape(24,5)
            if "ar_parameter" in record:
                return float(np.square(blocks).sum()+sum(2*record["ar_parameter"]**lag*
                            np.sum(blocks[:,lag:]*blocks[:,:-lag]) for lag in range(1,5)))
            return float(((1-correlations)*np.square(blocks).sum(axis=1)+
                          correlations*np.square(blocks.sum(axis=1))).sum())
        for method in ("lasso","elastic","ridge"):
            models=[s["fits"][method] for s in record["samples"]]
            coefs=[np.array(m["coef"]) for m in models]
            supports=[set(np.flatnonzero(b)) for b in coefs]
            risks=[]
            for side,(m,b,s) in enumerate(zip(models,coefs,supports)):
                delta=b.copy()
                delta[[2,7,12]]-=1
                risk=(1+m["intercept"]**2+energy(delta))**.5
                risks.append(risk)
                max_risk_error=max(max_risk_error,abs(risk-m["rmse"]))
                row=fit_tab.loc[(c,i,side,method)]
                assert row["size"]==len(s) and row.false_vars==len(s-{2,7,12})
                assert abs(row.rmse-risk)<1e-8
            if method=="ridge": continue
            a,b=supports
            j=len(a&b)/len(a|b) if a|b else 1.
            ga,gb=({v//5 for v in support} for support in (a,b))
            gj=len(ga&gb)/len(ga|gb) if ga|gb else 1.
            pp=tab.loc[(c,i,method)]
            assert abs(pp.j_var-j)<1e-9 and abs(pp.j_group-gj)<1e-9
            disagreement=((models[0]["intercept"]-models[1]["intercept"])**2+
                          energy(coefs[0]-coefs[1]))**.5
            assert abs(pp.disagreement-disagreement)<1e-8
            key=(c,method)
            primary[key]=primary.get(key,0)+int(abs(risks[0]-risks[1])<=.05 and j<=.5)
        count+=1
    summary=pd.read_csv(FORMAL/stage/"conditions.csv").set_index(["condition","method"])
    assert all(summary.loc[k,"both_n"]==v for k,v in primary.items())
    assert max_risk_error<1e-10
    return dict(pairs=count,checked_primary_counts=len(primary),max_absolute_rmse_error=max_risk_error)


def audit_credit(stage,df,test):
    x=design(df,raw_status=(stage=="credit_raw"))[test]
    y=df[TARGET].to_numpy()[test]
    count=0
    max_error=0.
    for record in fit_records(stage):
        for sample in record["samples"]:
            for method,m in sample["fits"].items():
                if method=="baseline":
                    prob=np.full(len(y),m["prevalence"])
                    loss=-np.where(y,np.log(prob),np.log1p(-prob)).mean()
                else:
                    eta=x@np.array(m["coef"])+m["intercept"]
                    prob=expit(eta)
                    loss=(np.logaddexp(0,eta)-eta*y).mean()
                brier=np.mean((prob-y)**2)
                max_error=max(max_error,abs(loss-m["loss"]),abs(brier-m["brier"]))
                count+=1
    assert max_error<1e-10
    return dict(scored_models=count,held_out_customers=len(test),max_absolute_score_error=max_error)


if __name__=="__main__":
    stages={stage:audit_simulation(stage) for stage in ("main","fixed","replication","ar")}
    df,h=workbook()
    assert h["xls_sha256"]==load_config()["xls_sha256"]
    dev,test,_,_=split(df)
    assert not set(df.ID.to_numpy()[dev]) & set(df.ID.to_numpy()[test])
    for stage in ("credit","credit_raw"):
        stages[stage]=audit_credit(stage,df,test)
    atomic_json(FORMAL/"metric_audit.json",stages)
    print(json.dumps(stages,indent=2))
