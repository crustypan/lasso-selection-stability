"""Build aggregate tables and figures from saved pairs, with pair-level uncertainty."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CFG, OUT, ROOT, atomic_json, jaccard, seed
from simulation import CONDITIONS, TRUTH, covariance
from credit_data import NAMES, MONTHS

METHODS=["lasso","elastic"]
COLORS={"lasso":"#20639b","elastic":"#b6561b","ridge":"#727984"}
TRUES=set(np.flatnonzero(TRUTH))
SIG=set(range(15))


def wilson(x,n):
    if n==0:
        return np.nan,np.nan
    z=1.95996398454
    p=x/n
    adj=1+z*z/n
    center=(p+z*z/(2*n))/adj
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/adj
    return max(0.,center-radius),min(1.,center+radius)


def groups(s,width):
    return {int(i)//width for i in s}


def overlap(k,a,b):
    ga,gb=groups(a,k),groups(b,k)
    return dict(j_var=jaccard(a,b),j_group=jaccard(ga,gb),double_empty=int(not a and not b))


def save(df,out,name):
    df.to_csv(out/(name+".csv"),index=False,float_format="%.10g")


def sim_tables(folder,out):
    out.mkdir(parents=True,exist_ok=True)
    fits,pairs,audits,frequency=[],[],[],[]
    for file in sorted(folder.glob("*.json")):
        r=json.loads(file.read_text())
        c,n,rs,rn,i=[r[k] for k in ["condition","n","rho_signal","rho_null","pair"]]
        cov=covariance(rs,rn) if "ar_parameter" not in r else np.kron(np.eye(24),
                        r["ar_parameter"]**np.abs(np.subtract.outer(np.arange(5),np.arange(5))))
        base=dict(condition=c,n=n,rho_signal=rs,rho_null=rn,pair=i)
        for method in r["samples"][0]["fits"]:
            ff=[s["fits"][method] for s in r["samples"]]
            supports=[set(np.flatnonzero(f["coef"])) for f in ff]
            for side,(f,s) in enumerate(zip(ff,supports)):
                fits.append(dict(**base,side=side,method=method,rmse=f["rmse"],alpha=f["alpha"],
                    boundary=f.get("boundary",False),max_iterations=f["max_iterations"],retries=f["retries"],
                    size=len(s),true_hits=len(s&TRUES),recall=len(s&TRUES)/3,
                    companions=len((s&SIG)-TRUES),null_vars=len(s-SIG),false_vars=len(s-TRUES),
                    signal_groups=len(groups(s&SIG,5)),null_groups=len(groups(s-SIG,5))))
                if method in METHODS:
                    frequency.append([c,method,np.isin(np.arange(120),list(s)).astype(int)])
                if "audit" in f:
                    audit=dict(f["audit"])
                    audit["test_n"]=audit.pop("n")
                    audits.append(dict(**base,side=side,method=method,**audit))
            if method not in METHODS:
                continue
            a,b=supports
            d=np.asarray(ff[0]["coef"])-ff[1]["coef"]
            dp=np.sqrt((ff[0]["intercept"]-ff[1]["intercept"])**2+d@cov@d)
            counts=dict(neither=0,one=0,disjoint=0,partial=0,identical=0,identical_true=0)
            for g in range(3):
                members=set(range(5*g,5*g+5))
                u,v=a&members,b&members
                if not u and not v: status="neither"
                elif not u or not v: status="one"
                elif not u&v: status="disjoint"
                elif u!=v: status="partial"
                else:
                    status="identical"
                    counts["identical_true"]+=int(bool(u&TRUES))
                counts[status]+=1
            pairs.append(dict(**base,method=method,**overlap(5,a,b),**counts,
                jointly_hit=counts["disjoint"]+counts["partial"]+counts["identical"],
                gap=abs(ff[0]["rmse"]-ff[1]["rmse"]),disagreement=dp,
                signal_diff=len((a^b)&SIG),null_diff=len((a^b)-SIG)))
    f,p=pd.DataFrame(fits),pd.DataFrame(pairs)
    save(f,out,"fits"); save(p,out,"pairs")
    if audits: save(pd.DataFrame(audits),out,"analytic_checks")
    freq=[]
    for c in sorted(f.condition.unique()):
        for m in METHODS:
            subset=[z for c0,m0,z in frequency if c0==c and m0==m]
            for j,v in enumerate(np.mean(subset,axis=0)):
                freq.append(dict(condition=c,method=m,variable=j+1,signal=j in TRUES,
                                 group=j//5+1,frequency=v))
    save(pd.DataFrame(freq),out,"variable_frequencies")
    summary=[]; sens=[]; contrasts=[]
    rng=np.random.default_rng(35097)
    for (c,name),ff in f.groupby(["condition","method"]):
        rr=ff.iloc[0]
        row={k:rr[k] for k in ["condition","n","rho_signal","rho_null","method"]}
        for col in ["rmse","size","recall","false_vars","companions","null_vars","signal_groups","null_groups"]:
            row[col+"_mean"]=ff[col].mean()
        row["rmse_q25"],row["rmse_q50"],row["rmse_q75"]=ff.rmse.quantile([.25,.5,.75])
        row.update(fits=len(ff),boundary_fits=int(ff.boundary.sum()),solver_retries=int(ff.retries.sum()))
        if name in METHODS:
            pp=p[(p.condition==c)&(p.method==name)]
            for col in ["j_var","j_group","gap","disagreement","null_diff","signal_diff"]:
                row[col+"_mean"]=pp[col].mean()
                row[col+"_q25"],row[col+"_q50"],row[col+"_q75"]=pp[col].quantile([.25,.5,.75])
            joint_hits=pp.jointly_hit.sum()
            row.update(pairs=len(pp),double_empty=pp.double_empty.mean(),common_hit=joint_hits/(3*len(pp)),
                       complete_swap=pp.disjoint.sum()/joint_hits if joint_hits else np.nan,
                       joint_group_hits=joint_hits,complete_swaps=pp.disjoint.sum())
            block=pp[["jointly_hit","disjoint"]].to_numpy()
            block_boot=block[rng.integers(len(pp),size=(3000,len(pp)))].sum(axis=1)
            good=block_boot[:,0]>0
            row["swap_lo"],row["swap_hi"]=np.quantile(block_boot[good,1]/block_boot[good,0],[.025,.975]) if good.any() else (np.nan,np.nan)
            for t in CFG["thresholds"]:
                close=pp.gap<=t
                low=pp.j_var<=.5
                both=(close&low).sum()
                lo,hi=wilson(both,len(pp))
                cl,ch=wilson(both,close.sum())
                rec=dict(condition=c,method=name,threshold=t,total=len(pp),close_n=close.sum(),low_n=low.sum(),
                         both_n=both,joint=both/len(pp),joint_lo=lo,joint_hi=hi,
                         conditional=both/close.sum() if close.any() else np.nan,cond_lo=cl,cond_hi=ch)
                for key,count in [("close",close.sum()),("low",low.sum())]:
                    rec[key]=count/len(pp)
                    rec[key+"_lo"],rec[key+"_hi"]=wilson(count,len(pp))
                sens.append(rec)
                if t==.05: row.update({k:v for k,v in rec.items() if k not in ("method","threshold","condition")})
        summary.append(row)
    for c,pp in p.groupby("condition"):
        ff=f[f.condition==c]
        by=ff.pivot(index=["pair","side"],columns="method",values=["rmse","size","recall","false_vars"])
        good=((abs(by["size"].elastic-by["size"].lasso)<=2)&
              (abs(by["rmse"].elastic-by["rmse"].lasso)<=.05)).groupby(level="pair").all()
        for label,ids in [("all",good.index),("matched",good[good].index)]:
            tab=pp.pivot(index="pair",columns="method",values=["j_var","j_group","gap"]).reindex(ids)
            diffs={k:(tab[k].elastic-tab[k].lasso).values for k in ["j_var","j_group"]}
            for k in ["size","rmse","recall","false_vars"]:
                v=(by[k].elastic-by[k].lasso).groupby(level="pair").mean()
                diffs[k]=v.reindex(ids).to_numpy()
            for k,v in diffs.items():
                boot=v[rng.integers(len(v),size=(3000,len(v)))].mean(axis=1) if len(v) else []
                # One observed pair supplies a point difference but no resampling uncertainty.
                lo,hi=np.quantile(boot,[.025,.975]) if len(v)>1 else (np.nan,np.nan)
                contrasts.append(dict(condition=c,subset=label,pairs=len(v),outcome=k,
                                      mean_elastic_minus_lasso=np.mean(v) if len(v) else np.nan,lo=lo,hi=hi))
    s=pd.DataFrame(summary)
    save(s,out,"conditions"); save(pd.DataFrame(sens),out,"thresholds")
    save(pd.DataFrame(contrasts),out,"method_differences")
    return s


def sim_plot(s,out,label):
    fig,ax=plt.subplots(3,2,figsize=(10,8),sharex=True,sharey="row",layout="constrained")
    for j,n in enumerate([80,240]):
        for method in [*METHODS,"ridge"]:
            z=s[(s.n==n)&(s.rho_signal==s.rho_null)&(s.method==method)].sort_values("rho_signal")
            if z.empty: continue
            x=np.arange(len(z))
            color=COLORS[method]
            ax[0,j].plot(x,z.rmse_q50,"o-",color=color,label="Elastic Net" if method=="elastic" else method.title())
            if method=="ridge":continue
            ax[1,j].plot(x,z.j_var_mean,"o-",color=color)
            ax[1,j].plot(x,z.j_group_mean,"s--",color=color)
            off=-.055 if method=="lasso" else .055
            ax[2,j].errorbar(x+off,z.joint,yerr=[np.maximum(0,z.joint-z.joint_lo),np.maximum(0,z.joint_hi-z.joint)],
                             color=color,fmt="o-",capsize=3)
        ax[0,j].set_title(f"Training sample size {n}")
        ax[0,j].axhline(1,ls=":",color="#64748b")
        ax[2,j].set_xticks(range(4),["0","0.5","0.9","0.99"])
        ax[2,j].set_xlabel("Within-block correlation")
    ax[0,0].set_ylabel("Population RMSE (median)")
    ax[1,0].set_ylabel("Jaccard (mean)")
    ax[2,0].set_ylabel("Close RMSE and low Jaccard\n(proportion, 95% Wilson interval)")
    ax[1,0].set_ylim(-.02,1.02); ax[2,0].set_ylim(-.02,1.02)
    ax[0,0].legend(frameon=False,ncol=3,fontsize=8)
    ax[1,0].plot([],[],"o-",color="gray",label="Variable")
    ax[1,0].plot([],[],"s--",color="gray",label="Group")
    ax[1,0].legend(frameon=False,fontsize=8)
    for a in ax.flat:
        a.grid(axis="y",alpha=.18); a.spines[["top","right"]].set_visible(False)
    fig.suptitle(label+" | Prediction and selection across independent samples",fontsize=14)
    fig.savefig(out/"simulation.png",dpi=165);plt.close(fig)


def credit_tables(folder,out,label):
    ff=[];pp=[];select=[]; blocks=[]
    for path in sorted(folder.glob("*.json")):
        r=json.loads(path.read_text()); i=r["pair"]
        for m in [*METHODS,"baseline"]:
            fits=[side["fits"][m] for side in r["samples"]]
            sets=[]
            for side,f in enumerate(fits):
                s=set(np.flatnonzero(f.get("coef",[])))
                ff.append(dict(pair=i,side=side,method=m,size=len(s),**{k:v for k,v in f.items() if k not in ["coef","intercept"]}))
                if m in METHODS:
                    select.append([m,np.isin(np.arange(18),list(s)).astype(int)])
                    sets.append(s)
            if m in METHODS:
                a,b=sets
                pp.append(dict(pair=i,method=m,**overlap(6,a,b),gap=abs(fits[0]["loss"]-fits[1]["loss"])))
                for g in range(3):
                    u,v=[s&set(range(6*g,6*g+6)) for s in sets]
                    blocks.append(dict(method=m,pair=i,group=["status","bill","repayment"][g],
                                       hits=int(bool(u))+int(bool(v)),common=int(bool(u and v)),
                                       changed=int(bool(u and v) and u!=v)))
    f,p,b=pd.DataFrame(ff),pd.DataFrame(pp),pd.DataFrame(blocks)
    save(f,out,"credit_fits");save(p,out,"credit_pairs");save(b,out,"credit_groups")
    wide=f.pivot(index=["pair","side"],columns="method",values="loss")
    gain=float((wide.baseline-wide.lasso).median())
    if label=="Pilot":
        threshold=max(.001,.05*gain)
    else:
        threshold=json.loads((ROOT/"config/frozen.json").read_text())["credit_threshold"]
    atomic_json(out/"credit_scale.json",dict(median_gain=gain,threshold=threshold,pairs=len(wide)//2))
    cs=[]; freq=[]
    for m in METHODS:
        v=np.mean([x for k,x in select if k==m],axis=0)
        rows=b[b.method==m].groupby("group").sum(numeric_only=True)
        for j,rate in enumerate(v):
            g=["status","bill","repayment"][j//6];row=rows.loc[g]
            freq.append(dict(method=m,group=g,month=MONTHS[j%6],field=NAMES[j],monthly_rate=rate,
                group_rate=row.hits/(2*len(p[p.method==m])),common_group_pairs=row.common,
                changed_given_common=row.changed/row.common if row.common else np.nan))
        for mul in [.5,1,2]:
            pp0=p[p.method==m];sel=pp0.gap<=mul*threshold
            cs.append(dict(method=m,multiplier=mul,total=len(pp0),close=sel.sum(),
                           low=(pp0.j_var<=.5).sum(),joint=(sel&(pp0.j_var<=.5)).sum()))
    fr=pd.DataFrame(freq)
    save(fr,out,"credit_months");save(pd.DataFrame(cs),out,"credit_events")
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
    for ax,m in zip(axs,METHODS):
        matrix=[]
        for g in ["status","bill","repayment"]:
            z=fr[(fr.method==m)&(fr.group==g)]
            matrix.append([*z.monthly_rate,z.group_rate.iloc[0]])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap="Blues",aspect="auto")
        for i,z in enumerate(matrix):
            for j,v in enumerate(z):
                percent=f"{100*v:.1f}".rstrip("0").rstrip(".")+"%"
                ax.text(j,i,percent,ha="center",va="center",color="white" if v>.56 else "black",fontsize=8)
        ax.set_xticks(range(7),MONTHS+["Any"]);ax.set_yticks(range(3),["Status","Bill","Repayment"])
        ax.set_title("Elastic Net" if m=="elastic" else "L1 logistic")
    fig.suptitle(label+" | Selection frequency across credit subsamples")
    fig.savefig(out/"credit.png",dpi=165);plt.close(fig)


if __name__=="__main__":
    out=OUT
    s=sim_tables(out/"simulation",out);sim_plot(s,out,"Pilot")
    credit_tables(out/"credit",out,"Pilot")
