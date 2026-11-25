"""Replay selected formal fits in a temporary directory and audit shared inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from common import ROOT, atomic_json
from formal import STAGES, execute, load_config, setup

FORMAL = ROOT / "results/formal"


def fit_records(stage):
    folder = FORMAL / stage
    meta = json.loads((folder / "archive.json").read_text())
    file = folder / meta["file"]
    if hashlib.sha256(file.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{stage}: archive checksum mismatch")
    with gzip.open(file, "rt") as stream:
        return [json.loads(line) for line in stream]


def check_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            check_tree(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        if a and all(isinstance(x, (int, float)) for x in a):
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=1e-9)
        else:
            for x, y in zip(a, b):
                check_tree(x, y)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=1e-9)
    else:
        assert a == b


def identity(r):
    return r.get("condition", 0), r["pair"]


def inputs(r):
    if "sampling_seeds" in r:
        return (*r["sampling_seeds"], *(s["fold_seed"] for s in r["samples"]))
    return tuple((s["data_seed"], s["fold_seed"]) for s in r["samples"])


def verify(whole_condition=False, jobs=2):
    cfg = load_config()
    config_hash = hashlib.sha256((ROOT/"config/frozen.json").read_bytes()).hexdigest()
    previous = {}
    status = {}
    (ROOT/"work").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"work") as tmp:
        folder = Path(tmp)
        for stage in STAGES:
            records = fit_records(stage)
            ledger = json.loads((FORMAL/stage/"run.json").read_text())
            assert len({identity(r) for r in records}) == len(records) == ledger["valid"]
            assert len(records) + len(ledger["failures"]) == ledger["planned"]
            assert json.loads((FORMAL/stage/"configuration.json").read_text())["sha256"] == config_hash
            ids = {identity(r): inputs(r) for r in records}
            if stage in ("main", "credit"):
                previous[stage] = ids
            if stage in ("fixed", "credit_raw"):
                reference = previous["main" if stage=="fixed" else "credit"]
                assert all(ids[k]==reference[k] for k in ids.keys() & reference.keys())
            if stage == "replication":
                assert all(ids[k]!=previous["main"][k] for k in ids.keys() & previous["main"].keys())
            if whole_condition and stage in ("main","replication"):
                chosen = [r for r in records if r["condition"] == 2]
            else:
                chosen = [r for r in records if r["pair"] == 0 and
                          (stage.startswith("credit") or r["condition"] == 2)]
            assert chosen, f"Missing preselected replay records: {stage}"
            with ProcessPoolExecutor(min(jobs,len(chosen)), initializer=setup,
                                     initargs=(stage,cfg,folder)) as workers:
                for _,_,err in workers.map(execute,map(identity,chosen)):
                    if err: raise RuntimeError(err)
            for expected in chosen:
                c,i=identity(expected)
                path = folder/(f"p{i:03d}.json" if stage.startswith("credit") else f"c{c:02d}-p{i:03d}.json")
                actual = json.loads(path.read_text())
                actual.pop("seconds", None)
                check_tree(expected, actual)
                for a,b in zip(expected["samples"],actual["samples"]):
                    for method in ("lasso","elastic"):
                        np.testing.assert_array_equal(np.flatnonzero(a["fits"][method]["coef"]),
                                                      np.flatnonzero(b["fits"][method]["coef"]))
            status[stage] = dict(archive_records=len(records), replayed=len(chosen),
                                 plan=ledger["planned"], failures=len(ledger["failures"]))
            print(stage,status[stage],flush=True)
    dest = ROOT/"work"/("full_replay.json" if whole_condition else "formal_replay.json")
    atomic_json(dest, dict(frozen_sha256=config_hash, stages=status))


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--whole-condition",action="store_true",
                    help="Replay all available n=80, rho=0.9 main and independent replication pairs.")
    ap.add_argument("--jobs",type=int,default=2)
    args=ap.parse_args()
    verify(args.whole_condition,args.jobs)
