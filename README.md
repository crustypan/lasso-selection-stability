# How Stable Is Lasso Feature Selection When Prediction Errors Are Similar?

**Question.** When two independently trained models have similar average prediction errors, how much do their variable lists overlap? When their lists differ, do the models exchange correlated measurements of the same signal, or pick different irrelevant variables?

**Design.** Draw two independent training samples from a population with 120 predictors in 24 groups. Exactly three predictors carry signal, one in each of three groups. Fit Lasso and Elastic Net to each sample with separate five-fold tuning. Vary correlation within groups while keeping the signal variance at 3 and the noise variance at 1. We can then calculate exact population prediction error and check every selection against the known nonzero coefficients.

**Finding.** Similar errors often accompanied very different lists, even with independent predictors. High correlation changed *which variables differed*. At training size 80 and within-group correlation 0.99, Lasso identified all three signal groups in both fits, yet the fits selected entirely different members in **244 of 600** jointly detected signal groups. In **42** of those 244 substitutions, both fits missed the true member.

## What does “when errors are similar” mean?

The table compares two probabilities for Lasso at training size 80. Both use the same numerator. RMSE is root mean squared prediction error. “Close” means the two population RMSEs differ by at most 0.05 noise standard deviations. “Low overlap” means that at most half of the selected variables in the union are common to both fits.

| Within-group correlation | Both events, out of all independent pairs | Low overlap, out of close-error pairs |
|---|---:|---:|
| 0 | 87/200 (43.5%) | 87/88 (98.9%) |
| 0.90 | 107/200 (53.5%) | 107/107 (100%) |
| 0.99 | 114/200 (57.0%) | 114/114 (100%) |

From 0 to 0.90, the joint rate rose mainly because more pairs had close prediction errors. At training size 240, the joint rate was already 97.0% at zero correlation and was 94.0% at 0.90. In a post-formal sensitivity check at size 80, tightening the overlap cutoff to 0.25 changed the joint counts to **76/200 and 75/200**. The data support a change in the *source* of disagreement more clearly than a general rise in its frequency. [Conditional distributions and denominators](docs/report.md#how-sensitive-is-low-overlap-to-its-definition)

![Joint and conditional proportions side by side, with exact counts for each method, training size and correlation setting](results/formal/main/simulation.png)

*Figure 1.* Each point is a proportion of independent training pairs with a 95% Wilson interval. Rows separate the two training sizes. The lower panels show absolute prediction error and mean overlap at variable and group level. For Lasso at size 80, mean RMSE was 1.134 at independence and 1.097 at correlation 0.99, against an oracle value of 1. [Complete condition table, including the two original controls](results/formal/condition_summary.md)

## What explains the disagreement?

In the exploratory follow-up, **only signal groups** had correlation 0.99 in one condition and **only null groups** had that correlation in the other. Lasso completely changed its within-group members in **252/588** jointly detected signal groups in the first condition and **0/600** in the second. Yet 111/112 close-error pairs in the null-only condition still had low overall overlap. Four of 200 signal-only pairs failed the solver rule, leaving 196 valid pairs and 588 jointly detected signal groups. [Exploratory comparison and uncertainty](docs/report.md#exploratory-controls-at-correlation-099)

![Complete signal-group substitutions and the number of differences coming from signal versus null groups](results/exploratory/rho099/mechanism.png)

*Figure 2.* Blue marks are the two post-formal controls. Gray marks are the formal references. The counts on the left use signal groups jointly detected by both fits. The bars count differing variables across all 24 groups.

**Elastic Net traded sparsity for stability.** At training size 80 and correlation 0.99, mean variable Jaccard increased from 0.126 (Lasso) to 0.420 (Elastic Net), mean support size from 11.35 to 26.82, and mean false inclusions from 9.50 to 23.84. Only one pair met the rule for comparable size and predictive risk, so the study cannot identify an advantage at equal sparsity.

**The credit case has an observable outcome and an unknown true support.** On 6,000 held-out customers, L1 logistic regression achieved mean log loss 0.4442 versus 0.5284 for the training-prevalence baseline. Every fit used repayment status, but the chosen months differed in 85/100 training pairs. These comparisons describe this customer pool and test set. [Credit results, figure and status-coding sensitivity](docs/report.md#credit-default-on-held-out-customers)

## Research decisions and evidence

Pan Junyang selected the question, approved the design and formal freeze, and requested the two targeted 0.99 controls and separate denominators. Codex implemented the models, ran the experiments, checked the results and drafted the report.

- [English research report](docs/report.md) — theory, simulation estimates, uncertainty, robustness and the credit application.
- [Design and discussion audit](docs/research-audit.md) — how each design choice answers the question, which alternative explanations survived, and the statistical and computational skills involved.
- [Original protocol](docs/protocol.md) · [Pilot review](docs/pilot.md) · [Frozen settings](config/frozen.json) · [Post-formal interpretation checks](docs/interpretation-checks.md).
- [Derivation and fitted-coefficient check](docs/mechanism.md) · [Computational verification](docs/review.md) · [Application CV wording](docs/cv-drafts.md).

## Reproduce

The pinned Python environment, commands from download through figures, archive-only rebuild and full refit are documented in the [reproduction guide](docs/reproduce.md). Credit data are from I-Cheng Yeh’s *Default of Credit Card Clients*, UCI Machine Learning Repository, DOI 10.24432/C55S3H, licensed under CC BY 4.0. The repository contains aggregate results and model fits. The original workbook and customer IDs are kept in the lab project. [UCI dataset](https://doi.org/10.24432/C55S3H) · [License](https://creativecommons.org/licenses/by/4.0/) · [Data audit and month mapping](results/pilot/credit_data_audit.json)
