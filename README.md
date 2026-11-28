# How Stable Is Lasso Feature Selection When Prediction Errors Are Similar?

**Introduction.** Lasso is a regression method that can set some coefficients to zero, leaving a shorter list of variables in the model. Imagine using monthly repayment records to predict credit default. If two months contain similar information, a model trained on one sample might keep the first month, while a model trained on another sample might keep the second. Both could have nearly the same average prediction error. This raises a question for anyone interpreting the selected variables: would the same variables be selected if the model were trained again on new data?

**Question.** When two Lasso fits have similar average prediction errors, how much do their lists of selected variables overlap? When the lists differ, are the models using different variables from the same group of correlated predictors, or are they picking variables from groups that contain no underlying signal?

**Method.** We simulated 120 candidate predictors in 24 groups of five. By construction, only three predictors directly contributed to the outcome, one in each of three groups. For example, suppose only X₁ directly affects the outcome in a group. A fit that selects the highly correlated X₂ can still use information about X₁. It has detected the group without recovering X₁ itself. We varied how strongly predictors within a group were correlated while keeping the strength of the signal and noise fixed. In each repetition, we drew two independent training samples and tuned Lasso and Elastic Net on each sample using five-fold cross-validation. We then compared prediction errors, the individual variables selected, and which of the three known signal groups each fit detected.

**Key Findings.** With 80 observations per training sample, we found the following.

1. **Similar prediction errors can hide very different variable lists.** Even when all predictors were independent, **87 of the 88 Lasso pairs with similar prediction errors** shared at most half of the variables selected by either fit. In this setting, similar average error gave little indication that the same variables would be selected after retraining.
2. **The two fits can agree on the group and pick different variables within it.** When the correlation between any two predictors in the same group was **0.99**, the five variables in each group tended to move almost in lockstep. In all 200 pairs, both fits selected at least one variable from each of the three signal groups. Yet in **244 of the 600 group comparisons**, the two fits selected entirely different members of the same group. The 600 comparisons come from checking the three signal groups in each of the 200 pairs. This shows how Lasso can repeatedly find the group containing the signal while changing which variables it uses to represent it.
3. **Finding the right group can still leave the true predictor out.** In **42 of those 244 comparisons**, neither fit selected the variable that actually contributed to the simulated outcome. Both fits had selected correlated stand-ins, so agreement at the group level gave an incomplete picture of whether Lasso had recovered the variables used to generate the outcome.

---

## How do we define similar prediction errors in this study?

For the simulation, we call two fits “close” when their population root mean squared errors (RMSEs) differ by at most 0.05 times the noise standard deviation. The noise standard deviation is 1, so a gap of 0.05 meets the cutoff. We chose this tolerance before the pilot and also specified tighter (0.02) and looser (0.10) cutoffs to see how much the answer depends on that choice. We call variable overlap “low” when at most half the variables selected by either fit appear in both (Jaccard ≤ 0.50). [Original protocol](docs/protocol.md)

Start with the first row. Of 200 Lasso pairs trained with 80 observations per sample, 88 had close prediction errors. Of those 88 pairs, 87 had low variable overlap. The joint proportion (87/200) measures how often both events occurred across all pairs. The conditional proportion (87/88) answers the title question by looking only at pairs whose prediction errors were already close.

| Within-group correlation | Close errors and low overlap, out of all pairs | Low overlap, out of close-error pairs |
|---|---:|---:|
| 0 | 87/200 (43.5%) | 87/88 (98.9%) |
| 0.90 | 107/200 (53.5%) | 107/107 (100%) |
| 0.99 | 114/200 (57.0%) | 114/114 (100%) |

At training size 80, raising the correlation from 0 to 0.90 increased the number of close-error pairs from 88 to 107. Low overlap was already common among close-error pairs at zero correlation, so the rise from 87/200 to 107/200 mainly reflects more pairs meeting the error cutoff. At size 240, the joint rate fell from 97.0% to 94.0% over the same range. At size 80, an exploratory check with a stricter overlap cutoff of 0.25 gave 76/200 at zero correlation and 75/200 at 0.90. The increase in the joint rate therefore depends on the sample size and on where we draw the line for low overlap. [Conditional distributions and denominators](docs/report.md#how-sensitive-is-low-overlap-to-its-definition)

**How to read Figure 1.** Start with the pink Lasso points in the upper-right panel. At correlation 0, the count 87/88 is the first row of the table above. The panel to its left shows the same 87 cases out of all 200 pairs.

![Joint and conditional proportions with exact counts, plus median prediction error and mean selection overlap](results/formal/main/simulation.png)

*Figure 1.* In the top four panels, points show proportions of training pairs and horizontal bars show 95% Wilson intervals. The upper row uses 80 observations per training sample and the next uses 240. Within-group correlation is on the vertical axis. The bottom four panels put correlation on the horizontal axis. They show median population RMSE (the dotted line at 1 is the error of the true regression function) and mean Jaccard overlap (solid lines compare variables and dashed lines compare groups). [Complete condition table, including the two original controls](results/formal/condition_summary.md)

## What explains the disagreement?

Figure 1 shows how often the variable lists disagreed. To understand where those differences came from, we ran two follow-up simulations. We set within-group correlation to 0.99 either in the three groups containing a true predictor (signal groups) or in the other 21 groups (null groups).

- **High correlation only in signal groups.** In **252 of 588** signal-group comparisons where both Lasso fits selected at least one variable from the group, the selected members had no overlap. Strong correlation within the signal groups was enough to produce frequent substitutions in this simulation. There were 588 comparisons because 196 of the 200 training pairs passed the convergence rule and each valid pair contributed three signal groups.
- **High correlation only in null groups.** There were **0 of 600** complete substitutions within signal groups, yet **111 of 112** pairs with close errors still had low overall variable overlap. In this setting, the two lists could differ substantially without a complete swap in any signal group. The right panel shows that more of the differing variables came from null groups than from signal groups.

[Exploratory comparison and uncertainty](docs/report.md#exploratory-controls-at-correlation-099)

![Lasso comparisons with strong correlation in signal groups, null groups, all groups or no groups](results/exploratory/rho099/mechanism.png)

*Figure 2.* Each row is one correlation setting. The left panel includes only signal groups where both fits picked at least one variable. It shows how often their choices had no overlap. The right panel shows the mean number of variables selected by exactly one fit, split by whether they came from a signal group or a null group. Blue points identify the two follow-up runs and gray points identify the original reference conditions.

**Elastic Net selected more of the same variables, and more variables overall.** At training size 80 and correlation 0.99, mean variable Jaccard rose from 0.126 for Lasso to 0.420 for Elastic Net. The average list grew from 11.35 to 26.82 variables, including 9.50 and 23.84 false selections, respectively. The higher overlap came with a longer list and more variables whose coefficients were zero in the simulation. Across all 1,999 valid pairs in the main simulation and its original controls, only one met the rule for comparing methods at similar size and predictive risk. One qualifying pair is too little evidence to judge whether Elastic Net would have higher overlap at comparable model size and prediction error.

**What happened to the repayment months in the opening example?** On 6,000 held-out customers, L1 logistic regression achieved a mean log loss of 0.4442. Predicting the training sample’s default rate for everyone gave 0.5284. Lower log loss means the fitted model’s default probabilities performed better on the held-out customers than that baseline. Every L1 fit used repayment status, and the selected months differed in 85 of 100 training pairs. Repayment status was consistently used as a category even though the specific months varied. The customer records let us compare prediction and selection stability. We check recovery against known true predictors in the simulation. [Pilot calibration](docs/pilot.md) · [Credit results and coding sensitivity](docs/report.md#credit-default-on-held-out-customers)

## Why we designed the study this way

**Why start with a simulation?** Lasso’s zero coefficients give us a list to interpret. On real credit records, we can see whether a month appears on that list, but we cannot check whether it belongs there. In the simulation we know the answer. Putting each true predictor in a separate, independent group keeps the signal variance at 3 as within-group correlation changes. This lets us make a stand-in more informative without making the overall signal stronger. [Tibshirani (1996), on Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) · [Design and derivation](docs/mechanism.md)

**Why retrain both methods?** Selection stability concerns how a variable list changes with the training data. We therefore drew fresh samples and tuned each fit from scratch. Within a sample, Lasso and Elastic Net used the same folds. Zou and Hastie showed that Elastic Net tends to keep correlated predictors together. That gives a reason to compare the two methods and to count how many extra variables that agreement brings in. [Nogueira, Sechidis and Brown (2018), on selection stability](https://www.jmlr.org/papers/v18/17-514.html) · [Zou and Hastie (2005), on the grouping effect](https://doi.org/10.1111/j.1467-9868.2005.00503.x)

**Why add two more controls?** In the original runs, complete substitutions jumped from 5 of 600 signal-group comparisons at correlation 0.90 to 244 of 600 at 0.99. Since correlation had changed in all 24 groups at once, I requested two follow-up simulations that changed correlation in one type of group at a time. This let us test whether strong correlation in the signal groups was sufficient to reproduce frequent substitutions. We report it as exploratory because the original results prompted the decision. [Follow-up plan](docs/exploratory-plan.md) · [Control results](docs/report.md#exploratory-controls-at-correlation-099)

## Report and supporting material

- [English research report](docs/report.md) — theory, simulation estimates, uncertainty, robustness and the credit application.
- [Design and discussion audit](docs/research-audit.md) — how each design choice answers the question, which alternative explanations survived, and the statistical and computational skills involved.
- [Original protocol](docs/protocol.md) · [Pilot review](docs/pilot.md) · [Frozen settings](config/frozen.json) · [Post-formal interpretation checks](docs/interpretation-checks.md).
- [Derivation and fitted-coefficient check](docs/mechanism.md) · [Computational verification](docs/review.md) · [Application CV wording](docs/cv-drafts.md).

## Reproduce

The pinned Python environment, commands from download through figures, archive-only rebuild and full refit are documented in the [reproduction guide](docs/reproduce.md). Credit data are from I-Cheng Yeh’s *Default of Credit Card Clients*, UCI Machine Learning Repository, DOI 10.24432/C55S3H, licensed under CC BY 4.0. The repository contains aggregate results and model fits. The original workbook and customer IDs are kept in the lab project. [UCI dataset](https://doi.org/10.24432/C55S3H) · [License](https://creativecommons.org/licenses/by/4.0/) · [Data audit and month mapping](results/pilot/credit_data_audit.json)
