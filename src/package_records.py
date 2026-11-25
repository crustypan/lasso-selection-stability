"""Publish compact, lossless fit records. No customer rows or identifiers are included."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from common import ROOT, atomic_json
from formal import STAGES


def package(stage):
    out = ROOT / "results/formal" / stage
    run = json.loads((out / "run.json").read_text())
    paths = sorted((out / "records").glob("*.json"))
    if len(paths) != run["valid"] or run["valid"] + len(run["failures"]) != run["planned"]:
        raise ValueError(f"{stage}: run ledger and saved records differ")
    archive = out / "fits.jsonl.gz"
    temp = out / "fits.jsonl.gz.tmp"
    with temp.open("wb") as sink, gzip.GzipFile(filename="", fileobj=sink, mode="wb", mtime=0) as compressed:
        for path in paths:
            record = json.loads(path.read_text())
            # Wall time belongs to the execution ledger and varies across replays.
            record.pop("seconds", None)
            compressed.write((json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode())
    os.replace(temp, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    atomic_json(out / "archive.json", dict(file=archive.name, rows=len(paths), sha256=digest))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    stage = ap.parse_args().stage
    for name in STAGES if stage == "all" else (stage,):
        package(name)
