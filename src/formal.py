"""Run the frozen design. Each result is saved atomically, one pair at a time."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import credit
import simulation
from common import ROOT, atomic_json
from credit_data import workbook, split, design, TARGET

STAGES=("main","fixed","replication","ar","credit","credit_raw")


def load_config():
    path=ROOT/"config/frozen.json"
    if not path.exists():
        raise SystemExit("Formal settings await Junyang's review. See config/formal-candidate.json.")
    return json.loads(path.read_text())


def setup(stage,cfg,folder):
    global STAGE, CONFIG, FOLDER, CREDIT_DATA
    STAGE,CONFIG,FOLDER=stage,cfg,folder
    if stage in ("credit","credit_raw"):
        df,hashes=workbook()
        if hashes["xls_sha256"]!=cfg["xls_sha256"]:
            raise ValueError("Workbook hash differs from the frozen data version")
        dev,test,_,_=split(df)
        CREDIT_DATA=(design(df,raw_status=stage=="credit_raw"),df[TARGET].to_numpy(),dev,test)


def execute(task):
    c,i=task
    try:
        if STAGE.startswith("credit"):
            credit.evaluate_pair(i,CONFIG["credit_formal_seed"],FOLDER,*CREDIT_DATA)
        else:
            key=CONFIG["formal_seed"]
            options={}
            if STAGE=="replication":
                key=CONFIG["independent_seed"]
            elif STAGE=="ar":
                key=CONFIG["ar_seed"]
                options["ar_parameter"]=brentq(lambda a:sum((5-h)*a**h for h in range(1,5))/10-.9,0,1)
            elif STAGE=="fixed":
                options["fixed"]=CONFIG["pilot_fixed_alphas"][str(c)]
            simulation.evaluate_pair(c,i,key,FOLDER,check=(STAGE=="main" and i==0 and c<8),**options)
        return c,i,None
    except (ValueError, RuntimeError, AssertionError, FloatingPointError) as exc:
        return c,i,f"{type(exc).__name__}: {exc}"


def tasks_for(stage,cfg):
    if stage.startswith("credit"):
        return [(0,i) for i in range(cfg["credit_pairs"])]
    if stage=="replication":
        return [(c,i) for c in cfg["replication_conditions"] for i in range(cfg["replication_pairs"])]
    if stage=="ar":
        return [(2,i) for i in range(cfg["ar_pairs"])]
    return [(c,i) for c in range(len(simulation.CONDITIONS)) for i in range(cfg["pairs"])]


def run(stage,jobs):
    cfg=load_config()
    version=hashlib.sha256((ROOT/"config/frozen.json").read_bytes()).hexdigest()
    out=ROOT/"results/formal"/stage
    folder=out/"records"
    folder.mkdir(parents=True,exist_ok=True)
    stamp=out/"configuration.json"
    if stamp.exists() and json.loads(stamp.read_text())["sha256"]!=version:
        raise ValueError("This output directory already contains a different configuration")
    atomic_json(stamp,dict(sha256=version))
    tasks=tasks_for(stage,cfg)
    def destination(task):
        c,i=task
        return folder/(f"p{i:03d}.json" if stage.startswith("credit") else f"c{c:02d}-p{i:03d}.json")
    remaining=[t for t in tasks if not destination(t).is_file()]
    ledger_path=out/"run.json"
    if ledger_path.is_file():
        previous=json.loads(ledger_path.read_text())
        recorded_failures={(f["condition"],f["pair"]) for f in previous["failures"]}
        if previous["valid"]==len(tasks)-len(remaining) and set(remaining)==recorded_failures:
            print(f"{stage}: existing run accounts for every planned pair",flush=True)
            return
    failures=[]
    start=time.monotonic()
    with ProcessPoolExecutor(jobs,initializer=setup,initargs=(stage,cfg,folder)) as pool:
        for j,(c,i,err) in enumerate(pool.map(execute,remaining),1):
            if err: failures.append(dict(condition=c,pair=i,error=err))
            if j%50==0: print(f"{stage}: {j}/{len(remaining)}",flush=True)
    atomic_json(out/"run.json",dict(stage=stage,planned=len(tasks),previous=len(tasks)-len(remaining),
                attempted=len(remaining),valid=sum(destination(t).exists() for t in tasks),
                failures=failures,workers=jobs,elapsed_seconds=time.monotonic()-start))
    if failures: print("Failed pairs recorded in",out/"run.json",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=(*STAGES,"all"),default="all")
    parser.add_argument("--jobs",type=int,default=2)
    args=parser.parse_args()
    for stage in (STAGES if args.stage=="all" else [args.stage]):
        run(stage,args.jobs)
