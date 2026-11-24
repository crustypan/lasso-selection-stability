"""Paired logistic pilot scored on a reserved development validation set."""
import time
from concurrent.futures import ProcessPoolExecutor
import argparse

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import CFG, OUT, atomic_json, grid, scaler, seed
from credit_data import workbook, split, design, TARGET


def logistic_path(x,y,alphas,mix):
    """Mean binary log loss plus alpha*(mix*L1 + (1-mix)*L2^2/2).
    Split each slope into nonnegative positive and negative parts for L-BFGS-B.
    Check the first-order conditions in the original coefficient space.
    """
    p=x.shape[1]
    init=np.zeros(2*p+1)
    init[0]=np.log(y.mean()/(1-y.mean()))
    bounds=[(None,None)]+[(0,None)]*(2*p)
    fits=[]
    for alpha in alphas:
        l1,l2=alpha*mix,alpha*(1-mix)

        def objective(theta):
            w=theta[1:p+1]-theta[p+1:]
            eta=theta[0]+x@w
            residual=(expit(eta)-y)/len(y)
            grad=x.T@residual+l2*w
            loss=np.mean(np.logaddexp(0,(1-2*y)*eta))+l1*theta[1:].sum()+l2*(w@w)/2
            return loss,np.r_[residual.sum(),grad+l1,l1-grad]

        for retry,ftol in enumerate([1e-13,1e-15]):
            opt=minimize(objective,init,method="L-BFGS-B",jac=True,bounds=bounds,
                         options=dict(ftol=ftol,gtol=1e-8,maxiter=4000,maxls=80,maxcor=25))
            w=opt.x[1:p+1]-opt.x[p+1:]
            init=np.r_[opt.x[0],np.maximum(w,0),np.maximum(-w,0)]
            residual=(expit(opt.x[0]+x@w)-y)/len(y)
            g=x.T@residual+l2*w
            kkt=np.where(w!=0,abs(g+l1*np.sign(w)),np.maximum(abs(g)-l1,0))
            violation=max(abs(residual.sum()),kkt.max())
            if opt.success and violation <= 5e-6:
                break
        else:
            raise RuntimeError(f"Logistic optimization failed: KKT={violation}, {opt.message}")
        fits.append((float(init[0]),w.copy(),float(violation),int(opt.nit),retry))
    return fits


def metrics(eta,y):
    return dict(loss=float(np.mean(np.logaddexp(0,(1-2*y)*eta))),
                brier=float(np.mean((expit(eta)-y)**2)))


def train(x,y,xv,yv,fold_seed):
    alphas=grid("credit_grid")
    scores={name:np.zeros(len(alphas)) for name in CFG["l1_ratios"]}
    counts={name:dict(max_kkt=0.,max_iterations=0,retries=0) for name in scores}

    def track(name,fit):
        counts[name]["max_kkt"]=max(counts[name]["max_kkt"],fit[2])
        counts[name]["max_iterations"]=max(counts[name]["max_iterations"],fit[3])
        counts[name]["retries"]+=fit[4]

    for tr,va in StratifiedKFold(5,shuffle=True,random_state=fold_seed).split(x,y):
        m,s=scaler(x[tr])
        for name,mix in CFG["l1_ratios"].items():
            path=logistic_path((x[tr]-m)/s,y[tr],alphas,mix)
            for j,fit in enumerate(path):
                track(name,fit)
                a,w=fit[:2]
                scores[name][j]+=metrics(a+((x[va]-m)/s)@w,y[va])["loss"]*len(va)
    m,s=scaler(x)
    result={}
    for name,mix in CFG["l1_ratios"].items():
        idx=int(np.argmin(scores[name]))
        fit,=logistic_path((x-m)/s,y,alphas[idx:idx+1],mix)
        track(name,fit)
        a,w=fit[:2]
        coef=w/s
        intercept=float(a-m@coef)
        eta=intercept+xv@coef
        np.testing.assert_allclose(eta,a+((xv-m)/s)@w,atol=1e-10)
        result[name]=dict(**metrics(eta,yv),**counts[name],coef=coef.tolist(),intercept=intercept,
                          alpha=float(alphas[idx]),boundary=idx in (0,len(alphas)-1),
                          cv_loss=float(scores[name][idx]/len(y)))
    prevalence=y.mean()
    result["baseline"]=dict(**metrics(np.full(len(yv),np.log(prevalence/(1-prevalence))),yv),
                            prevalence=float(prevalence))
    return result


def evaluate_pair(i,root_seed,folder,x,y,pool,validation):
    sampling=seed(root_seed,i,0)
    left,remainder=train_test_split(pool,train_size=CFG["credit_each"],stratify=y[pool],random_state=sampling)
    right,_=train_test_split(remainder,train_size=CFG["credit_each"],stratify=y[remainder],
                             random_state=seed(root_seed,i,1))
    assert set(left).isdisjoint(right) and set(np.r_[left,right]).isdisjoint(validation)
    samples=[]
    for side,indices in enumerate([left,right]):
        foldseed=seed(root_seed,i,2+side)
        fits=train(x[indices],y[indices],x[validation],y[validation],foldseed)
        samples.append(dict(fold_seed=foldseed,defaults=int(y[indices].sum()),fits=fits))
    atomic_json(folder/f"p{i:03d}.json",dict(pair=i,sampling_seeds=[sampling,seed(root_seed,i,1)],samples=samples))
    return i


def run_pair(i):
    return evaluate_pair(i,CFG["credit_pilot_seed"],OUT/"credit",X,Y,POOL,VAL)


def load_worker():
    global X,Y,POOL,VAL
    df,_=workbook()
    _,_,POOL,VAL=split(df)
    X,Y=design(df),df[TARGET].to_numpy()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--jobs",type=int,default=2)
    args=parser.parse_args()
    start=time.monotonic()
    with ProcessPoolExecutor(args.jobs,initializer=load_worker) as pool:
        for i in pool.map(run_pair,range(CFG["pairs"])):
            print("Completed credit pair",i+1,flush=True)
    atomic_json(OUT/"credit_runtime.json",dict(workers=args.jobs,pairs=CFG["pairs"],seconds=time.monotonic()-start))
