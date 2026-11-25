"""Publish compact, lossless fit records. No customer rows or identifiers are included."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from common import ROOT, atomic_json
from formal import STAGES


def package(stage, *, base=ROOT / "results/formal"):
    out = base / stage
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


def restore(stage):
    out=ROOT / "results/formal" / stage
    meta=json.loads((out/"archive.json").read_text())
    file=out/"fits.jsonl.gz"
    if hashlib.sha256(file.read_bytes()).hexdigest()!=meta["sha256"]:
        raise ValueError(f"{stage}: archive checksum mismatch")
    seen=set()
    with gzip.open(file,"rt") as stream:
        for line in stream:
            rec=json.loads(line)
            i=int(rec["pair"])
            name=f"p{i:03d}.json" if stage.startswith("credit") else f"c{int(rec['condition']):02d}-p{i:03d}.json"
            if name in seen:
                raise ValueError(f"Duplicate record: {name}")
            seen.add(name)
            path=out/"records"/name
            if path.exists():
                old=json.loads(path.read_text())
                old.pop("seconds",None)
                if old!=rec:
                    raise ValueError(f"Existing fit differs from the archive: {path}")
            else:
                atomic_json(path,rec)
    if len(seen)!=meta["rows"]:
        raise ValueError(f"{stage}: archive count mismatch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    ap.add_argument("--restore",action="store_true",help="Unpack published records for rebuilding tables and plots.")
    args=ap.parse_args()
    for name in STAGES if args.stage=="all" else (args.stage,):
        (restore if args.restore else package)(name)
