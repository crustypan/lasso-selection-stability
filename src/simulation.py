"""Paired Gaussian simulation. Run from the repository root with one BLAS thread."""
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import enet_path
from sklearn.model_selection import KFold

from common import CFG, OUT, ROOT, atomic_json, grid, scaler, seed

P = 120
TRUTH = np.zeros(P)
TRUTH[CFG["truth_zero_based"]] = 1
CONDITIONS = [(n, r, r) for n in CFG["n"] for r in CFG["rho"]] + [(80, .9, 0), (80, 0, .9)]


def covariance(rs, rn):
    blocks = [np.eye(5) * (1-r) + np.ones((5, 5)) * r for r in [rs]*3 + [rn]*21]
    return np.kron(np.eye(24), np.ones((5, 5))) * np.tile(np.vstack(blocks), (1, 24))


def generate(rng, n, factor):
    x = rng.standard_normal((n, P)) @ factor.T
    return x, x @ TRUTH + rng.standard_normal(n)


def coef_path(x, y, name, penalties):
    """Coefficients on centered, standardized inputs. y is centered."""
    if name == "ridge":
        u, d, vt = np.linalg.svd(x, full_matrices=False)
        filters = d[:, None] / (d[:, None]**2 + len(y)*penalties)
        return vt.T @ ((u.T @ y)[:, None]*filters), 0, 0
    for retry, limit in enumerate([12000, 60000]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            _, coefficients, _, iterations = enet_path(
                x, y, l1_ratio=CFG["l1_ratios"][name], alphas=penalties,
                precompute=False, max_iter=limit, tol=1e-7, return_n_iter=True)
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            return coefficients, max(iterations), retry
    raise RuntimeError(f"{name} path failed convergence after retry")


def fit_methods(x, y, fold_seed):
    methods = {"lasso":grid("linear_grid"), "elastic":grid("linear_grid"), "ridge":grid("ridge_grid")}
    losses = {k:np.zeros(len(g)) for k,g in methods.items()}
    retries = dict.fromkeys(methods, 0)
    iters = dict.fromkeys(methods, 0)
    for train, val in KFold(5, shuffle=True, random_state=fold_seed).split(x):
        mean, sd = scaler(x[train])
        xt, xv = (x[train]-mean)/sd, (x[val]-mean)/sd
        center = y[train].mean()
        for k,g in methods.items():
            co, it, retr = coef_path(xt, y[train]-center, k, g)
            losses[k] += np.square(y[val,None]-center-xv@co).sum(axis=0)
            retries[k] += retr
            iters[k] = max(iters[k], it)
    mean, sd = scaler(x)
    fits = {}
    for k,g in methods.items():
        idx = int(np.argmin(losses[k]))
        alpha = g[idx:idx+1]
        co, it, retr = coef_path((x-mean)/sd, y-y.mean(), k, alpha)
        coef = co[:,0]/sd
        intercept = float(y.mean()-mean@coef)
        np.testing.assert_allclose(x@coef+intercept, ((x-mean)/sd)@co[:,0]+y.mean(), atol=1e-10)
        fits[k] = dict(alpha=float(alpha[0]), boundary=idx in (0,len(g)-1),
                       cv_mse=float(losses[k][idx]/len(y)), intercept=intercept, coef=coef.tolist(),
                       max_iterations=max(it,iters[k]), retries=retries[k]+retr)
    return fits


def audit(fits, factor, key):
    sums = {name:np.zeros(2) for name in fits}
    rng = np.random.default_rng(key)
    n_test = CFG["large_test"]
    for start in range(0,n_test,5000):
        x,y = generate(rng, min(5000,n_test-start), factor)
        for name,f in fits.items():
            e2 = np.square(y-f["intercept"]-x@f["coef"])
            sums[name] += [e2.sum(), e2@e2]
    for name,(s,s2) in sums.items():
        mse = s/n_test
        se = np.sqrt((s2-s*s/n_test)/(n_test-1)/n_test)
        expected = fits[name]["rmse"]**2
        fits[name]["audit"] = dict(n=n_test, seed=key, mse=mse, analytic_mse=expected,
                                   se=se, z=(mse-expected)/se)


def fixed_fits(x,y,penalties):
    mean,sd=scaler(x)
    z=(x-mean)/sd
    fits={}
    for name,alpha in penalties.items():
        co,it,retry=coef_path(z,y-y.mean(),name,np.array([alpha]))
        slopes=co[:,0]/sd
        intercept=float(y.mean()-mean@slopes)
        np.testing.assert_allclose(x@slopes+intercept,z@co[:,0]+y.mean(),atol=1e-10)
        fits[name]=dict(alpha=alpha,coef=slopes.tolist(),intercept=intercept,max_iterations=it,retries=retry)
    return fits


def evaluate_pair(c,i,root_seed,folder,*,fixed=None,ar_parameter=None,check=False):
    n,rs,rn = CONDITIONS[c]
    cov = covariance(rs,rn)
    if ar_parameter is not None:
        distance=np.abs(np.subtract.outer(np.arange(5),np.arange(5)))
        block=ar_parameter**distance
        np.testing.assert_allclose(block[np.triu_indices(5,1)].mean(),rs)
        cov=np.kron(np.eye(24),block)
    np.testing.assert_allclose(TRUTH@cov@TRUTH,3)
    factor = np.linalg.cholesky(cov)
    path = folder/f"c{c:02d}-p{i:03d}.json"
    pair = dict(condition=c,n=n,rho_signal=rs,rho_null=rn,pair=i,samples=[])
    if ar_parameter is not None:
        pair["ar_parameter"]=ar_parameter
    start = time.monotonic()
    for side in range(2):
        data_seed, fold_seed, audit_seed = [seed(root_seed,c,i,side,k) for k in range(3)]
        x,y = generate(np.random.default_rng(data_seed),n,factor)
        fits = fit_methods(x,y,fold_seed) if fixed is None else fixed_fits(x,y,fixed)
        for f in fits.values():
            delta = np.asarray(f["coef"])-TRUTH
            f["rmse"] = float(np.sqrt(1+f["intercept"]**2+delta@cov@delta))
        if check:
            audit(fits,factor,audit_seed)
        pair["samples"].append(dict(data_seed=data_seed,fold_seed=fold_seed,fits=fits))
    pair["seconds"] = round(time.monotonic()-start,3)
    atomic_json(path,pair)
    return c,i


def run_pair(task):
    c,i=task
    return evaluate_pair(c,i,CFG["seed"],OUT/"simulation",check=(i==0 and c<8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",type=int,default=2)
    args = parser.parse_args()
    tasks = [(c,i) for c in range(len(CONDITIONS)) for i in range(CFG["pairs"])]
    started = time.monotonic()
    with ProcessPoolExecutor(args.jobs) as pool:
        for completed,_ in enumerate(pool.map(run_pair,tasks),1):
            if completed % 20 == 0:
                print(f"Completed {completed}/{len(tasks)} pairs",flush=True)
    atomic_json(OUT/"simulation_runtime.json",dict(pairs=len(tasks),workers=args.jobs,seconds=time.monotonic()-started))
