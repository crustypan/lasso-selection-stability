"""Exploratory QA of the largest original MSE check using independent Gaussian factors."""
import json

import numpy as np
import pandas as pd

from common import ROOT, atomic_json
from verify_formal import fit_records

if __name__ == "__main__":
    folder = ROOT/"results/formal/main"
    checks = pd.read_csv(folder/"analytic_checks.csv")
    row = checks.loc[checks.z.abs().idxmax()]
    record = next(r for r in fit_records("main") if r["condition"]==row.condition and r["pair"]==row.pair)
    fit = record["samples"][int(row.side)]["fits"][row.method]
    truth = np.zeros(120)
    truth[[2,7,12]] = 1
    rng = np.random.default_rng(8811350)
    squared_sum = fourth_sum = 0.
    n = 1_000_000
    rho = float(row.rho_signal)
    assert rho == row.rho_null
    for _ in range(n//5000):
        shared = rng.normal(size=(5000,24,1))
        specific = rng.normal(size=(5000,24,5))
        x = (rho**.5*shared+(1-rho)**.5*specific).reshape(-1,120)
        error = x@(truth-fit["coef"])+rng.normal(size=len(x))-fit["intercept"]
        squared = error**2
        squared_sum += squared.sum()
        fourth_sum += squared@squared
    empirical = squared_sum/n
    se = ((fourth_sum-squared_sum**2/n)/(n-1)/n)**.5
    analytic = fit["rmse"]**2
    rec = dict(condition=int(row.condition), pair=int(row.pair), side=int(row.side),method=row.method,
               original_z=row.z, new_seed=8811350, test_n=n, empirical_mse=empirical,
               analytic_mse=analytic,mcse=se,new_z=(empirical-analytic)/se)
    atomic_json(ROOT/"results/formal/exploratory_mse_check.json",rec)
    print(json.dumps(rec,indent=2))
