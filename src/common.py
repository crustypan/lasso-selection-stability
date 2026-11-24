"""Shared deterministic settings and project-local output helpers."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config/pilot.json").read_text())
OUT = ROOT / "results/pilot"
OUT.mkdir(parents=True, exist_ok=True)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def seed(*keys):
    return int(np.random.SeedSequence(keys).generate_state(1)[0])


def grid(key):
    settings = CFG[key]
    return np.geomspace(settings["max"], settings["min"], settings["count"])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def scaler(x):
    scale = x.std(axis=0)
    return x.mean(axis=0), np.where(scale == 0, 1, scale)
