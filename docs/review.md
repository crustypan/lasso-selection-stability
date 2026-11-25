# Completed research verification

Junyang selected the question and authorized the frozen design. At his request, Codex completed the design check, figure review, robustness reruns, interpretation and final computational verification.

## Design and Figure 1

Each of the three unit signal coefficients multiplies a standardized variable from a different independent block. Their variances add to 3 for every within-block correlation. The oracle population RMSE is 1. At training size 80, the two extra controls change correlation only in the signal blocks or only in the null blocks. Their Lasso joint-event counts were 74/199 and 95/200, respectively, compared with 87/200 at independence and 107/200 when all blocks had correlation 0.9. The signal-only denominator records one failed Lasso path. [Frozen design](../config/frozen.json) · [Condition table](../results/formal/condition_summary.md)

We checked the figure against the underlying counts. At training size 80, 87/88 close-risk Lasso pairs at independence and 107/107 at correlation 0.9 had low variable overlap. The joint proportion thus rises from 43.5% to 53.5% mostly because more pairs satisfied the risk criterion. At size 240 the analogous joint frequencies were 97.0% and 94.0%. At size 80 and correlation 0.99, both fits found every signal block, yet whole-model group Jaccard averaged only 0.368 as different null groups entered. Variable Jaccard averaged 0.126. The figure and the condition table answer different parts of the question about recovery. [Figure 1](../results/formal/main/simulation.png) · [Conditional counts](../results/formal/main/thresholds.csv) · [Mechanism counts](../results/formal/mechanisms.csv)

## What the robustness checks establish

| Check | Lasso result at training size 80 | Interpretation |
|---|---|---|
| Independent seeds at correlations 0 and 0.9 | 84/200 and 101/200 joint events. Difference +8.5 percentage points, 95% interval −1.3 to +18.0. Original difference +10.0 points, +0.2 to +19.5. | The direction repeated. The independent estimate is also compatible with a small negative difference. |
| Pilot-fixed penalties | 96/200 and 118/200 joint events at 0 and 0.9. At 0.99, complete swaps occurred in 236/600 jointly detected signal blocks. | Frequent near-collinear substitutions persisted after fixing the penalty in advance. |
| AR(1) blocks, mean correlation 0.9 | Equicorrelated and AR(1) blocks both gave 107/200 joint events. Complete swaps rose from 5/600 to 23/600. | The same binary event frequency concealed different within-block behavior. |

The method-size comparison has limited support. Only one of 1,999 valid pairs met the prespecified cross-method RMSE and size matching rule on both training samples. This leaves the advantage of Elastic Net at comparable sparsity unresolved. [Independent seeds](../results/formal/replication/endpoint_contrasts.csv) · [Fixed penalties](../results/formal/fixed/conditions.csv) · [AR(1) results](../results/formal/ar/conditions.csv) · [Method differences](../results/formal/main/method_differences.csv)

## Why close prediction errors can coexist with the same wrong selection

In formal pair 19 (zero-based), the population RMSEs were 1.09977 and 1.07735. Each model selected only variable 9 in the second signal block. The true signal in this block is variable 8. Variables are numbered from 1. The RMSE gap measures how similarly the models perform on average. Their shared choice of variable 9 measures agreement. Comparing that choice to the known coefficient of variable 8 reveals the shared error. This pair also had an RMS difference of 0.448 between predictions for the same new subject. A small difference between average risks can conceal appreciable differences in individual predictions.

This worked example was selected after the run for illustration. Across all 600 jointly detected signal blocks in this condition, 54 had identical memberships, including four identical proxy-only sets. In the credit study, observed defaults support a check of probability forecasts. They do not identify a true generating set of months. All 200 L1 fits selected repayment status, but the chosen status months changed in 85/100 pairs. [Worked example and simulation results](report.md) · [Credit figure](../results/formal/credit/credit.png)

## Computational reproduction

A fresh checkout of the original analysis commit was created inside the isolated lab project. It downloaded the official UCI workbook, repeated both pilots, regenerated the candidate freeze, and independently refitted all six formal stages from seeds. The repeated formal fits used the already approved configuration. No archived record was restored into a fitting path. Seed-exact reruns verify computational reproducibility. The independent-seed stage measures variation across fresh training samples.

The 200 simulation pilot pairs and 20 credit pilot pairs reproduced their saved fit records, and the recalibrated configuration matched the freeze byte for byte. All 4,800 formal stage-pair fits were attempted from scratch. The original 4,799 valid records and the single failed pair were reproduced. The table counts stage evaluations, so fixed-penalty and alternate-encoding reruns of the same training samples each contribute a stage count.

| Stage | Attempted pairs | Valid pairs | Comparison with original |
|---|---:|---:|---|
| Main simulation and controls | 2,000 | 1,999 | Same complete fits and same convergence failure at condition 8, pair 192 |
| Fixed penalties | 2,000 | 2,000 | Same complete fits |
| Independent seeds | 400 | 400 | Same complete fits |
| AR(1) blocks | 200 | 200 | Same complete fits |
| Credit, main encoding | 100 | 100 | Same complete fits |
| Credit, raw status encoding | 100 | 100 | Same complete fits |

All six compressed fit archives matched byte for byte. Another 50 output files matched byte for byte, including the regenerated tables, figures, threshold records and metric audits. The independent implementation recomputed population RMSE to within 4.45 × 10⁻¹⁶ and credit scores to within 1.12 × 10⁻¹⁶. The exploratory Monte Carlo check of the largest original analytic-MSE discrepancy also reproduced its estimate of −1.28 Monte Carlo standard errors. [Full-refit record](../results/formal/full_refit.json) · [Independent metric audit](../results/formal/metric_audit.json) · [Analysis and reproduction commands](../README.md#reproduce)

## Exploratory extension

The revised formal figure displays the unconditional joint rate and the conditional rate side by side. All 16 method-by-condition numerators and both denominator series were checked against the formal pair table. Re-rendering from the formal condition table reproduced the revised PNG byte for byte.

The two post-formal 0.99 controls each attempted 200 fresh training pairs. The 196 common valid signal-only pairs yielded 252 complete Lasso swaps across 588 jointly detected signal blocks. The 200 null-only pairs yielded zero across 600. A second execution of the 400 attempted pairs reproduced all 14 result files byte for byte, including the ledger for four convergence failures. The coefficient-based audit agreed with the primary event and swap counts. Its maximum absolute population RMSE discrepancy was 2.23 × 10⁻¹⁶ or less. [Exploratory section](report.md#exploratory-controls-at-correlation-099) · [Verification record](../results/exploratory/rho099/verification.json)
