# How Stable Is Lasso Feature Selection When Prediction Errors Are Similar?

How often do two independent training samples yield models with similar prediction errors and different selected variables? A Gaussian simulation measures predictive risk, selection stability and recovery of known signals. A credit-default application tests the same diagnostics with 18 observed monthly predictors.

## Findings

- **Most of the low overlap at independence came from noise variables.** At training size 80 and zero correlation, the Lasso supports differed on an average of 19.28 null-block variables and 2.04 signal-block variables. At correlation 0.99, complete within-block substitution occurred in 244 of 600 jointly detected signal blocks.
- **Sample size changed the headline result.** The frequency of close population RMSE and low variable overlap rose from 43.5% to 53.5% between correlations 0 and 0.9 at n = 80. At n = 240 it changed from 97.0% to 94.0%.
- **Elastic Net selected larger models.** At n = 80 and correlation 0.99 its mean variable Jaccard was 0.420 versus 0.126 for Lasso. Mean sizes were 26.82 and 11.35. Only one pair across the ten conditions met the prespecified size/risk matching rule.
- **Monthly selection varied on real credit data.** L1 logistic regression selected the repayment-status family in all 200 training fits but changed its selected status months in 85/100 pairs. Mean test log loss was 0.4442 versus 0.5284 for the training-prevalence baseline.

The exploratory follow-up restricted within-block correlation of 0.99 to either the three signal blocks or the 21 null blocks at training size 80. Complete Lasso substitutions occurred in **252/588** jointly detected signal blocks with signal-only correlation and **0/600** with null-only correlation. The formal all-blocks reference gave **244/600**. [Exploratory figure](results/exploratory/rho099/mechanism.png) · [Four-cell comparison](results/exploratory/rho099/four_cells.csv)

[Short report](docs/report.md) · [Condition table](results/formal/condition_summary.md) · [Credit case](results/formal/credit_models.csv) · [Completed verification](docs/review.md) · [Application CV wording](docs/cv-drafts.md)

![Prediction and selection across independently trained models](results/formal/main/simulation.png)

**Figure 1.** Each dot and interval estimates a proportion across independent training pairs (95% Wilson interval). The joint column divides the number of close-error, low-overlap pairs by all valid pairs. The conditional column divides the same numerator by the number with close error. The count beside each dot gives its exact numerator and denominator. Rows separate training samples of size 80 and 240. The bottom panels give median population RMSE and mean variable and group Jaccard.

## Study record

| Milestone | Evidence |
|---|---|
| Design and pilot | [Protocol](docs/protocol.md), [pilot review](docs/pilot.md), [two-variable derivation](docs/mechanism.md) |
| Freeze before formal fitting | [Settings](config/frozen.json), [freeze commit](https://github.com/crustypan/lasso-selection-stability/commit/38c96dca752798d113634467eb8249596a33424a) |
| Primary and mechanism results | [Complete-pair ledger](results/formal/main/run.json), [conditional probabilities and sensitivities](results/formal/main/thresholds.csv), [block-level explanations](results/formal/mechanisms.csv), [method comparisons](results/formal/main/method_differences.csv) |
| Robustness | [Independent seeds](results/formal/replication/endpoint_contrasts.csv), [pilot-fixed penalties](results/formal/fixed/conditions.csv), [AR(1) blocks](results/formal/ar/conditions.csv) |
| Exploratory 0.99 controls | [Plan](docs/exploratory-plan.md), [run and failures](results/exploratory/rho099/run.json), [pair-bootstrap contrasts](results/exploratory/rho099/swap_contrasts.csv) |
| Application | [Credit figure](results/formal/credit/credit.png), [monthly frequencies](results/formal/credit/credit_months.csv), [raw-encoding comparison](results/formal/credit_models.csv) |
| Verification | [Independent metric calculations](results/formal/metric_audit.json), [full refit from seeds](results/formal/full_refit.json), [initial 404 pair replays](results/formal/replay.json), [exploratory follow-up to a test-set discrepancy](results/formal/exploratory_mse_check.json) |

The primary simulation targeted 2,000 pairs across eight main conditions and two controls. All 1,600 main-condition pairs completed. One fit failed in the signal-only correlation control, giving 199/200 valid pairs in that condition and 1,999/2,000 overall. Fixed penalties used the same training samples and folds. Independent replicates used new seeds. The three prespecified robustness stages completed 2,000, 400 and 200 pairs, respectively. Each credit encoding completed 100 pairs.

## Reproduce

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

The full refit reproduced the 200 simulation and 20 credit pilot pairs, all 4,799 valid formal stage-pair records, and the same failed pair. All six fit archives and 50 rebuilt result files matched byte for byte.

The [completed verification](docs/review.md) covers the signal variance, both panels of the main comparison, the three robustness checks and an observed pair where both fits selected the same proxy. The [full-refit check](src/verify_full_refit.py) compares a separate, freshly refitted checkout to the archived study. From the reference repository, pass the path to that checkout after running the commands above.

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

To regenerate the exploratory archive as well, run `.venv/bin/python src/exploratory_rho099.py --jobs 2`. The [run ledger](results/exploratory/rho099/run.json) identifies all four failed signal-only pairs. The [audit](results/exploratory/rho099/audit.json) checks seeds, population RMSE and event counts from the saved coefficients.

## Credit data and license

I-Cheng Yeh (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The program downloads the original UCI archive and verifies the workbook against its frozen SHA-256 before formal fitting. Derived inputs retain the six monthly bills, repayments and repayment-status fields. The primary status encoding floors nonpositive codes at zero. The raw-code sensitivity uses the same clients, folds and test set. ID is used for customer-level checks and removed from model inputs. Customer records and ID splits are stored in the server project and excluded from Git. [Data audit, version, month mapping and special-code counts](results/pilot/credit_data_audit.json)
