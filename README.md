# How Stable Is Lasso Feature Selection When Prediction Errors Are Similar?

A paired-sample simulation separates predictive risk, variable stability and recovery of a known signal. An application to UCI credit default compares selected months across customer subsamples. The protocol is in [docs/protocol.md](docs/protocol.md). The initial settings are in [config/pilot.json](config/pilot.json).

## Current milestone

The pilot is complete. Both figures and aggregate tables are available below. Formal results require a configuration freeze after Junyang reviews the pilot.

## Data

I-Cheng Yeh. Default of Credit Card Clients. UCI Machine Learning Repository. DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0. Development and test observations remain in the research directory on the lab server. Only source code, configurations, audit summaries and aggregate results are shared in this repository.

## Reproduce the pilot

Use Python 3.11 (the recorded run used 3.11.4) from the repository root. Install the pinned numerical and workbook packages in an isolated environment. The simulation and credit commands replace the corresponding deterministic pilot outputs.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/mpl"
.venv/bin/python src/credit_data.py
.venv/bin/python src/simulation.py --jobs 2
.venv/bin/python src/credit.py --jobs 2
.venv/bin/python src/summarize.py
```

The [pilot review](docs/pilot.md) reports the simulated and development-validation outcomes separately. See the [condition table](results/pilot/conditions.csv), [threshold counts and intervals](results/pilot/thresholds.csv), [matched method contrasts](results/pilot/method_differences.csv), [analytic risk checks](results/pilot/analytic_checks.csv), [credit data audit](results/pilot/credit_data_audit.json), and [monthly frequencies](results/pilot/credit_months.csv). Individual fitting seeds and coefficients are recorded in the per-pair JSON outputs. The validation criterion for the custom logistic optimizer is maximum first-order residual at most 5e-6 across the intercept and slopes. Raw and per-customer UCI data are excluded from version control.
