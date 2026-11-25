# Reproduce the study

Use Python 3.11 from the repository root. The recorded run used Python 3.11.4. Numerical packages, workbook reader and plotting versions are pinned. The sources derive seeds from the stage, condition, pair, sample and purpose. The formal run used eight worker processes with one BLAS thread per worker. Two workers are sufficient for the commands below.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/mpl"
.venv/bin/python src/credit_data.py
```

**Rebuild published tables and figures from the archived coefficients.** Each stage includes a compressed JSON-lines file containing seeds, original-scale coefficients, intercepts and solver diagnostics. The restore command verifies SHA-256 and unpacks records under that stage.

```sh
.venv/bin/python src/package_records.py --restore
.venv/bin/python src/formal_tables.py
.venv/bin/python src/review_tables.py
.venv/bin/python src/audit_results.py
.venv/bin/python src/verify_formal.py
```

**Refit the entire study in a fresh clone.** Start after environment setup and data acquisition, before unpacking the formal archives. Stage ledgers support resuming interrupted work. Full-precision pilot medians and development-validation log loss determine the candidate settings. The committed frozen settings govern formal evaluation.

```sh
.venv/bin/python src/simulation.py --jobs 2
.venv/bin/python src/credit.py --jobs 2
.venv/bin/python src/summarize.py
.venv/bin/python src/verify_pilot.py
.venv/bin/python src/prepare_freeze.py
.venv/bin/python src/formal.py --stage all --jobs 2
.venv/bin/python src/formal_tables.py
.venv/bin/python src/review_tables.py
.venv/bin/python src/package_records.py
.venv/bin/python src/audit_results.py
.venv/bin/python src/verify_formal.py --whole-condition --jobs 2
.venv/bin/python src/check_audit_outlier.py
```

At the archived full-refit checkpoint, the full refit reproduced the 200 simulation and 20 credit pilot pairs, all 4,799 valid formal stage-pair records, and the same failed pair. All six fit archives and 50 rebuilt result files matched byte for byte.

The [completed verification](review.md) covers the signal variance, both panels of the main comparison, the three robustness checks and an observed pair where both fits selected the same proxy. The [full-refit check](../src/verify_full_refit.py) compares a separate, freshly refitted checkout to the reference repository at the same reporting version. Earlier audit manifests refer to the file versions in their commit. From the reference repository, pass the path to that checkout after running the commands above.

```sh
.venv/bin/python src/verify_full_refit.py /path/to/refitted-checkout
```

## Reproduce the revised figure and exploratory controls

From the repository root, use the same Python environment and single-threaded BLAS settings as above. The first command plots the formal results with joint and conditional denominators side by side. The second reconstructs all exploratory tables and the mechanism figure from the separate fit archive. The third refits all 400 exploratory pairs and checks the generated files against that archive and its results.

```sh
.venv/bin/python src/main_figure.py
.venv/bin/python src/exploratory_rho099.py --rebuild
.venv/bin/python src/exploratory_rho099.py --verify-refit --jobs 2
```

To regenerate the exploratory archive as well, run `.venv/bin/python src/exploratory_rho099.py --jobs 2`. The [run ledger](../results/exploratory/rho099/run.json) identifies all four failed signal-only pairs. The [audit](../results/exploratory/rho099/audit.json) checks seeds, population RMSE and event counts from the saved coefficients.


## Reproduce the post-formal interpretation audit

Use the published fit archives and the same pinned Python environment. The first command computes the two-threshold sensitivity grid, classifies complete substitutions by truth and decomposes coefficient disagreement. It also renders the overlap distribution. The second reconstructs the credit splits and performs an evaluation-label permutation on development data. Run the data acquisition step above first.

```sh
.venv/bin/python src/interpretation_checks.py
.venv/bin/python src/verify_design.py
```

The fitted models and primary thresholds are archived. The current reporting code leaves bootstrap limits empty for any matched subgroup with fewer than two eligible pairs. That correction affects the lower and upper limits of six outcomes for one matched main-condition pair and leaves the six point differences intact. The earlier full-refit and exploratory-replay records remain evidence for the corresponding analysis versions.
