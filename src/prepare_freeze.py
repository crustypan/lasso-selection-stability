"""Write the proposed settings from the completed pilot for human review."""
import json

import pandas as pd

from common import CFG, OUT, ROOT, atomic_json


def candidate():
    values=[]
    for path in sorted((OUT/"simulation").glob("*.json")):
        record=json.loads(path.read_text())
        for sample in record["samples"]:
            for method,fit in sample["fits"].items():
                values.append(dict(condition=record["condition"],method=method,alpha=fit["alpha"]))
    f=pd.DataFrame(values)
    counts=f.groupby(["condition","method"]).size()
    if len(counts)!=30 or (counts!=2*CFG["pairs"]).any():
        raise ValueError("Incomplete pilot condition or method")
    medians={str(c):dict(z.groupby("method").alpha.median()) for c,z in f.groupby("condition")}
    credit=json.loads((OUT/"credit_scale.json").read_text())
    if credit["pairs"]!=CFG["pairs"]:
        raise ValueError("Incomplete credit pilot")
    audit=json.loads((OUT/"credit_data_audit.json").read_text())
    return dict(**{k:v for k,v in CFG.items() if k not in ("phase","pairs")},
                phase="formal",pairs=CFG["formal_target"],credit_pairs=CFG["credit_formal_target"],
                replication_pairs=200,ar_pairs=200,replication_conditions=[0,2],
                credit_threshold=credit["threshold"],credit_gain=credit["median_gain"],
                pilot_fixed_alphas=medians,xls_sha256=audit["xls_sha256"],
                ar_seed=25112504,bootstrap_draws=3000)


if __name__=="__main__":
    atomic_json(ROOT/"config/formal-candidate.json",candidate())
