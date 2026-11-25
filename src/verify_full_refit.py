"""Compare a fresh fit of the frozen study with the original outputs.

The reference root is this repository. The argument is an independently
refitted checkout containing both pilot results and packaged formal records.
Print a machine-readable record after every comparison succeeds.
"""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("main", "fixed", "replication", "ar", "credit", "credit_raw")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_equal(left, right, label):
    if left != right:
        raise ValueError(f"Different {label}")


def check_pilot(reference, repeat):
    counts = {}
    for name in ("simulation", "credit"):
        original = sorted((reference / name).glob("*.json"))
        new_names = sorted(p.name for p in (repeat / name).glob("*.json"))
        require_equal([p.name for p in original], new_names, f"pilot {name} pair IDs")
        if not original:
            raise ValueError(f"No {name} pilot pairs")
        for path in original:
            old, new = read(path), read(repeat / name / path.name)
            for record in (old, new):
                record.pop("seconds", None)
            require_equal(old, new, str(path))
        counts[name] = len(original)
    audit = read(reference / "credit_data_audit.json")
    fresh = read(repeat / "credit_data_audit.json")
    for record in (audit, fresh):
        record.pop("retrieved_utc", None)
    require_equal(audit, fresh, "workbook checksums, mappings and data audit")
    return counts, audit["xls_sha256"]


def check_stage(reference, repeat):
    old, new = read(reference / "run.json"), read(repeat / "run.json")
    if new["previous"] != 0 or new["attempted"] != new["planned"]:
        raise ValueError(f"{repeat.name} included preexisting fits")
    for key in ("planned", "valid", "failures"):
        require_equal(old[key], new[key], f"{repeat.name} {key}")
    meta = read(reference / "archive.json")
    require_equal(read(repeat / "archive.json"), meta, f"{repeat.name} archive manifest")
    for folder in (reference, repeat):
        require_equal(digest(folder / meta["file"]), meta["sha256"], str(folder))
    require_equal(meta["rows"], new["valid"], f"{repeat.name} archived valid pairs")
    return dict(attempted=new["attempted"], valid=new["valid"], failures=new["failures"],
                fit_archive_sha256=meta["sha256"])


def compare_outputs(reference, repeat):
    files = sorted(p.relative_to(reference) for p in reference.rglob("*")
                   if p.suffix in {".csv", ".md", ".png"})
    files += [Path("metric_audit.json"), Path("exploratory_mse_check.json")]
    for stage in ("credit", "credit_raw"):
        files.append(Path(stage) / "credit_scale.json")
    for path in files:
        require_equal(digest(reference / path), digest(repeat / path), str(path))
    return [str(path) for path in files]


def proxy_example(archive, truth):
    """Illustration: first close-RMSE pair with an identical proxy-only block."""
    with gzip.open(archive, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if (record["n"], record["rho_signal"], record["rho_null"]) != (80, .99, .99):
                continue
            left, right = [s["fits"]["lasso"] for s in record["samples"]]
            gap = abs(left["rmse"] - right["rmse"])
            if gap > .05:
                continue
            for true_index in truth:
                block = true_index // 5
                members = range(5 * block, 5 * block + 5)
                a, b = [{j for j in members if fit["coef"][j] != 0} for fit in (left, right)]
                if a and a == b and true_index not in a:
                    return dict(condition=record["condition"], pair_zero_based=record["pair"],
                                block_one_based=block + 1, true_variable_one_based=true_index + 1,
                                shared_selected_one_based=sorted(j + 1 for j in a),
                                rmse=[left["rmse"], right["rmse"]], rmse_gap=gap)
    raise ValueError("Illustrative pair missing")


def verify(repeat):
    if repeat.resolve() == ROOT:
        raise ValueError("A separate checkout is required")
    ref, new = (base / "results" / "formal" for base in (ROOT, repeat))
    frozen = digest(ROOT / "config/frozen.json")
    for path in ("config/frozen.json", "config/formal-candidate.json"):
        require_equal(frozen, digest(repeat / path), f"repeat {path}")
    counts, xls_sha = check_pilot(ROOT / "results/pilot", repeat / "results/pilot")
    stages = {stage: check_stage(ref / stage, new / stage) for stage in STAGES}
    for stage in STAGES:
        require_equal(read(new / stage / "configuration.json")["sha256"], frozen,
                      f"{stage} freeze checksum")
    outputs = compare_outputs(ref, new)
    commit = subprocess.check_output(["git", "-C", str(repeat), "rev-parse", "HEAD"], text=True).strip()
    truth = read(ROOT / "config/frozen.json")["truth_zero_based"]
    return dict(refit_code_commit=commit, frozen_sha256=frozen, workbook_sha256=xls_sha,
                pilot_pairs=counts, stages=stages, byte_identical_output_count=len(outputs),
                byte_identical_outputs=outputs,
                illustrative_pair=proxy_example(new / "main/fits.jsonl.gz", truth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refit_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.refit_directory.resolve()), indent=2))
