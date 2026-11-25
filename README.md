# How Stable Is Lasso Feature Selection When Prediction Errors Are Similar?

How often do two independent training samples yield models with similar prediction errors and different selected variables? A Gaussian simulation measures predictive risk, selection stability and recovery of known signals. A credit-default application tests the same diagnostics with 18 observed monthly predictors.

## Findings

- **Most of the low overlap at independence came from noise variables.** At training size 80 and zero correlation, the Lasso supports differed on an average of 19.28 null-block variables and 2.04 signal-block variables. At correlation 0.99, complete within-block substitution occurred in 244 of 600 jointly detected signal blocks.
- **Sample size changed the headline result.** The frequency of close population RMSE and low variable overlap rose from 43.5% to 53.5% between correlations 0 and 0.9 at n = 80. At n = 240 it changed from 97.0% to 94.0%.
- **Elastic Net selected larger models.** At n = 80 and correlation 0.99 its mean variable Jaccard was 0.420 versus 0.126 for Lasso. Mean sizes were 26.82 and 11.35. Only one pair across the ten conditions met the prespecified size/risk matching rule.
- **Monthly selection varied on real credit data.** L1 logistic regression selected the repayment-status family in all 200 training fits but changed its selected status months in 85/100 pairs. Mean test log loss was 0.4442 versus 0.5284 for the training-prevalence baseline.

[Short report](docs/report.md) · [Condition table](results/formal/condition_summary.md) · [Credit case](results/formal/credit_models.csv) · [Human review](docs/review.md)

![Prediction and selection across independently trained models](results/formal/main/simulation.png)

**Figure 1.** Each column corresponds to a training size. The upper row shows median population RMSE with oracle RMSE at one. The middle row gives mean variable and group Jaccard. The lower row shows the probability of an RMSE gap at most 0.05 and variable Jaccard at most 0.5, with a 95% Wilson interval. The horizontal positions represent four ordered correlation settings.

## Study record

| Milestone | Evidence |
|---|---|
| Design and pilot | [Protocol](docs/protocol.md), [pilot review](docs/pilot.md), [two-variable derivation](docs/mechanism.md) |
| Freeze before formal fitting | [Settings](config/frozen.json), [freeze commit](https://github.com/crustypan/lasso-selection-stability/commit/38c96dca752798d113634467eb8249596a33424a) |
| Primary and mechanism results | [Complete-pair ledger](results/formal/main/run.json), [conditional probabilities and sensitivities](results/formal/main/thresholds.csv), [block-level explanations](results/formal/mechanisms.csv), [method comparisons](results/formal/main/method_differences.csv) |
| Robustness | [Independent seeds](results/formal/replication/endpoint_contrasts.csv), [pilot-fixed penalties](results/formal/fixed/conditions.csv), [AR(1) blocks](results/formal/ar/conditions.csv) |
| Application | [Credit figure](results/formal/credit/credit.png), [monthly frequencies](results/formal/credit/credit_months.csv), [raw-encoding comparison](results/formal/credit_models.csv) |
| Verification | [Independent metric calculations](results/formal/metric_audit.json), [404 pair replays](results/formal/replay.json), [exploratory follow-up to a test-set discrepancy](results/formal/exploratory_mse_check.json) |

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

## Credit data and license

I-Cheng Yeh (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The program downloads the original UCI archive and verifies the workbook against its frozen SHA-256 before formal fitting. Derived inputs retain the six monthly bills, repayments and repayment-status fields. The primary status encoding floors nonpositive codes at zero. The raw-code sensitivity uses the same clients, folds and test set. ID is used for customer-level checks and removed from model inputs. Customer records and ID splits are stored in the server project and excluded from Git. [Data audit, version, month mapping and special-code counts](results/pilot/credit_data_audit.json)
