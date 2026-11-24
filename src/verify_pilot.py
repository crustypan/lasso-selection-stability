"""Replay a high-correlation pair and a credit pair and compare saved fits."""
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import credit
import simulation
from common import CFG, OUT, ROOT, atomic_json


def replay():
    with TemporaryDirectory(dir=ROOT/"work") as tmp:
        where=Path(tmp)
        c,i=3,0
        simulation.evaluate_pair(c,i,CFG["seed"],where,check=True)
        before=json.loads((OUT/"simulation"/"c03-p000.json").read_text())
        after=json.loads((where/"c03-p000.json").read_text())
        assert before["samples"]==after["samples"]
        credit.load_worker()
        credit.evaluate_pair(0,CFG["credit_pilot_seed"],where,
                             credit.X,credit.Y,credit.POOL,credit.VAL)
        assert json.loads((where/"p000.json").read_text())==json.loads((OUT/"credit/p000.json").read_text())
    rng=np.random.default_rng(25681)
    x=rng.normal(size=(1200,8)); prob=expit(.22+x[:,0]-.4*x[:,1])
    y=rng.binomial(1,prob)
    errors={}
    for mix in [1,.5]:
        penalty=.03
        a,w,*_=credit.logistic_path(x,y,[penalty],mix)[0]
        model=LogisticRegression(solver="saga",C=1/(len(y)*penalty),penalty="elasticnet",
                                 l1_ratio=mix,random_state=345,tol=1e-11,max_iter=15000).fit(x,y)
        error=np.max(abs((a+x@w)-model.decision_function(x)))
        assert error<1e-5
        errors[str(mix)]=float(error)
    atomic_json(OUT/"replay_check.json",dict(simulation_pair="c03-p000",
            credit_pair="p000",simulation_sample_records_exact=True,credit_record_exact=True,
            independent_logistic_solver_max_logit_diff=errors))
    print("Pilot replays matched and independent logistic solver cross-check passed.")


if __name__=="__main__":
    replay()
