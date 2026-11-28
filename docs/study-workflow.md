# Study workflow: Lasso selection stability

This study asks whether two models trained on separate samples can have similar prediction errors while selecting different variables. It also asks where the differences come from. A simulation gives us known true predictors, and a credit default application lets us see how selection changes across training samples drawn from an observed customer dataset.

| Stage | Input | Main output |
| --- | --- | --- |
| [1. Design the simulation](#stage-1-design-the-simulation) | A specified Gaussian population | Known signals and correlation settings |
| [2. Acquire and prepare the credit data](#stage-2-acquire-and-prepare-the-credit-data) | Official UCI workbook | Audited predictors and customer splits |
| [3. Fit models and define the measurements](#stage-3-fit-models-and-define-the-measurements) | Training pairs from either data source | Selected variables, prediction errors and overlap measures |
| [4. Run the pilots and freeze the analysis](#stage-4-run-the-pilots-and-freeze-the-analysis) | Pilot simulations and credit development data | Diagnostics, calibrated credit tolerance and formal settings |
| [5. Run the formal simulation](#stage-5-run-the-formal-simulation) | Fresh simulated pairs and frozen settings | Main results and Figure 1 |
| [6. Check robustness](#stage-6-check-robustness) | Fresh simulation streams or formal samples with fixed penalties | Evidence on sensitivity to sampling, tuning and correlation structure |
| [7. Evaluate credit default prediction](#stage-7-evaluate-credit-default-prediction) | Formal customer draws and the fixed test set | Test losses, selected months and coding sensitivity |
| [8. Run the exploratory controls](#stage-8-run-the-exploratory-controls) | New simulation pairs at selective correlation 0.99 | Evidence about where substitutions occur and Figure 2 |
| [9. Diagnose the formal and exploratory results](#stage-9-diagnose-the-formal-and-exploratory-results) | Archived coefficients and pair records | Threshold sensitivity, coefficient geometry and true-variable recovery |
| [10. Verify and reproduce the study](#stage-10-verify-and-reproduce-the-study) | Seeds, source code, workbook and archived outputs | Rebuilt results, fresh refits and independent score checks |

## Stage 1. Design the simulation

**Input.** Generate observations from a population of 120 standardized Gaussian predictors in 24 independent groups of five. Within any group, all pairwise correlations are equal in the main design. Different groups are independent. The true response is the sum of predictors 3, 8 and 13 (one-based numbering) plus independent Gaussian noise with standard deviation 1. Each true predictor is the middle member of its own group and has coefficient 1. The other 117 coefficients are zero.

**What we did.** Vary within-group correlation over 0, 0.50, 0.90 and 0.99 at training sizes 80 and 240. These combinations give eight main conditions. Add two conditions at size 80: correlation 0.90 only in the three signal groups, and correlation 0.90 only in the other 21 groups. In every condition, draw two independent training samples per repetition.

**What this gives us.** Signal variance stays at 3 as correlation changes because the three generating predictors belong to independent groups. Prediction by the true regression function has population RMSE 1, and prediction by the population mean has RMSE 2. This design allows a zero-coefficient variable to be predictive through its correlation with a true predictor. For example, selecting variable 9 can use information about variable 8. It counts as finding the right group, while exact recovery requires selecting variable 8. [Protocol](protocol.md) · [Frozen design](../config/frozen.json) · [Simulation code](../src/simulation.py)

## Stage 2. Acquire and prepare the credit data

**Input.** I-Cheng Yeh’s *Default of Credit Card Clients*, UCI dataset 350. The Taiwan customer workbook has 30,000 rows, and the target is default payment in the next month. The project uses 18 of the 23 available predictors, covering six months of records from April to September 2005.

| Predictor family | Workbook columns | Role in the study |
| --- | --- | --- |
| Repayment status | PAY_0, PAY_2 through PAY_6 | Test whether the months chosen within a predictor family stay the same |
| Billed amount | BILL_AMT1 through BILL_AMT6 | Compare the selection of monthly bills and check their correlations |
| Repayment amount | PAY_AMT1 through PAY_AMT6 | Track selection of payment amounts by month and as a family |

**What we did.** Download the official ZIP, hash it and its workbook, verify customer IDs and target counts, map each selected field to its month, and audit repayment-status codes and missing entries. The main encoding maps nonpositive status values to zero and retains positive values as ordered numeric grades. The alternate encoding keeps the original numeric values. Both encode the same customers and monthly fields. The audit records the unusual values and the reporting code retains both encodings for comparison.

Split by customer into a stratified development pool of 24,000 and a held-out test set of 6,000. Split development into 19,200 customers available for pilot training and 4,800 for pilot validation. Formal training draws use the full development pool. The 6,000 test customers are held out for formal scoring.

**Checks and outputs.** The workbook had 30,000 unique IDs, 6,636 defaults (22.12%), and no missing entries in the modelling fields. The audit records the source URL, hashes, counts and full month mapping. The original workbook and customer IDs stay outside the public result tables. [UCI dataset and license](https://doi.org/10.24432/C55S3H) · [Data preparation code](../src/credit_data.py) · [Workbook and split audit](../results/pilot/credit_data_audit.json)

## Stage 3. Fit models and define the measurements

### Fit each training sample

For linear simulations, fit Lasso, Elastic Net with L1 mixing fraction 0.5, and Ridge as a prediction baseline. Within each sample, the methods share five shuffled cross-validation folds. Estimate standardization on each fold’s training subset, select the penalty with the lowest validation mean squared error, then refit on the entire training sample. Repeat the tuning process independently for the second sample. The frozen grids contain 24 penalties for the sparse linear methods and 25 for Ridge.

For credit, draw two disjoint stratified training samples of 3,000 customers per pair. Customers may recur across different pairs. Fit L1 logistic regression and Elastic Net logistic regression with stratified five-fold cross-validation and a 16-point penalty grid. Select the penalty by validation log loss. Compare their probability forecasts with a baseline that predicts the training sample’s default rate for every customer. [Linear fitting code](../src/simulation.py) · [Credit fitting code](../src/credit.py) · [Penalty grids](../config/frozen.json)

### Score predictions, selections and recovery

| Measurement | How it is computed | What it answers |
| --- | --- | --- |
| Population RMSE | Transform fitted intercept and coefficients back to the population scale and use the known covariance to compute expected squared error, including noise | How accurate a simulation fit is on average over new observations |
| Prediction disagreement | Compare the predictions from two fitted models on the same new inputs and aggregate their squared difference over the population | How much the two models can disagree about individual predictions |
| Variable Jaccard | Count variables selected by both fits and divide by the number selected by either fit | How similar the two variable lists are |
| Group Jaccard | Apply the same calculation to groups in which a fit selected at least one variable | How similar the lists are when any member counts as finding its group |
| True-variable recall and false inclusions | Compare the nonzero fitted coefficients with the three known generating coefficients | How well the fitted list recovers the simulation truth |
| Complete substitution | Among signal groups detected by both fits, count cases where the two selected sets within the group have empty intersection | How often both fits find the same signal group through entirely different members |
| Credit log loss and Brier score | Score predicted default probabilities against observed outcomes on the common evaluation set | How well probability forecasts perform on the evaluated customers |

In the simulation, a pair has **close errors** when its two population RMSEs differ by at most 0.05. A pair has **low overlap** when variable Jaccard is at most 0.50. If two lists are both empty, their Jaccard is defined as 1. None of the main sparse-model pairs fell into that case. Both the joint and conditional proportions count pairs that meet both criteria. The joint proportion divides by all valid pairs, and the conditional proportion divides by pairs that already meet the error criterion. Pair-level proportions use Wilson intervals. For bootstrap summaries of group outcomes, each resampled training pair carries its groups together. [Definitions, analytic risk and uncertainty](report.md#design-and-estimands)

## Stage 4. Run the pilots and freeze the analysis

**Simulation pilot.** Run 20 independent pairs per condition across all ten conditions. This yields 200 pairs, 400 training samples and 1,200 final linear fits. For Lasso at size 80, 7/20, 10/20 and 11/20 pairs met both criteria at correlations 0, 0.90 and 0.99. At size 240 the corresponding counts were 20/20, 19/20 and 20/20. The large-sample counts showed that the binary summary could be near 100%, motivating a closer look at continuous overlap and error. At size 80, complete substitutions rose from 1/60 at 0.90 to 33/60 at 0.99, motivating a separate group-level analysis.

**Credit pilot.** Fit 20 pairs using the pilot training pool and score them on the 4,800 development validation customers. Mean validation log loss was 0.528486 for the prevalence baseline, 0.450399 for L1 and 0.450042 for Elastic Net. Set the proposed tolerance for close credit losses to 5% of the median gain from baseline to L1 (0.07853256), yielding 0.0039266281. This calibrates the tolerance against a measured gain in probability prediction before evaluating the final test set.

**Freeze.** Build the candidate configuration from the pilot records, including condition-specific median penalties for the fixed-penalty check and the workbook SHA-256. Junyang approved the frozen configuration before formal fitting. The main simulation kept its original 0.05 error and 0.50 Jaccard cutoffs, with additional error-gap checks at 0.02 and 0.10. The formal plan specified 200 pairs per simulation condition and 100 credit pairs per encoding. [Pilot review](pilot.md) · [Pilot diagnostics and metrics](../results/pilot/) · [Freeze preparation](../src/prepare_freeze.py) · [Approved settings](../config/frozen.json)

## Stage 5. Run the formal simulation

**Input and execution.** Generate 200 independent pairs for each main and control condition with the formal seed. Train both samples in each pair, store coefficients, penalties and solver diagnostics, and write the run ledger. One Lasso path failed in the 0.90 signal-only control after the retry, leaving 199 common valid pairs in that condition and 1,999 valid pairs across the ten conditions.

The table gives Lasso counts for the eight main conditions. The numerator is the number of pairs satisfying both criteria. The two columns for each sample size use different denominators.

| Within-group correlation | Size 80, all valid pairs | Size 80, close-error pairs | Size 240, all valid pairs | Size 240, close-error pairs |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 87/200 | 87/88 | 194/200 | 194/195 |
| 0.50 | 88/200 | 88/88 | 190/200 | 190/190 |
| 0.90 | 107/200 | 107/107 | 188/200 | 188/189 |
| 0.99 | 114/200 | 114/114 | 196/200 | 196/196 |

**Interpretation.** At size 80, the joint proportion rose from 43.5% at independence to 53.5% at correlation 0.90 mainly because more pairs had close errors. Among pairs already within the cutoff, low overlap was common in both conditions. At size 240, the joint proportion moved from 97% to 94% over the same contrast. The primary contrast depends on the sample size.

At size 80 and correlation 0.99, both fits detected all three signal groups in every pair. The fits chose entirely different members in **244 of 600** comparisons, compared with **5 of 600** at 0.90. At size 240 and correlation 0.99, the count was **65 of 600**. These comparisons track a change in what gets selected *within* a signal group. Whole-model group Jaccard can still be modest even if both fits detect every signal group, because it also includes selected null groups.

At size 80 and correlation 0.99, Elastic Net reached mean variable Jaccard 0.420 versus 0.126 for Lasso. It also selected 26.82 variables per fit versus 11.35, with 23.84 versus 9.50 false inclusions on average. Only one of 1,999 pairs satisfied the rule for comparable cross-method size and prediction risk in both samples. This leaves the overlap advantage at comparable sparsity unresolved.

**Outputs.** The formal summary contains pair-level and fit-level records, condition summaries, sensitivity to the error cutoff, selection frequencies and Figure 1. Figure 1’s top panels show proportions, while its bottom panels show median population RMSE and mean Jaccard. [Formal runner](../src/formal.py) · [Main ledger](../results/formal/main/run.json) · [Condition table](../results/formal/condition_summary.md) · [Group results](../results/formal/mechanisms.csv) · [Figure 1](../results/formal/main/simulation.png)

## Stage 6. Check robustness

| Check and input | What we changed | Result and interpretation |
| --- | --- | --- |
| Independent seeds, 400 fresh pairs at size 80 | Repeat the 0 versus 0.90 comparison with 200 pairs per condition | Joint Lasso counts were 84/200 and 101/200. The estimated difference was +8.5 percentage points with a 95% interval of −1.3 to +18.0. The direction repeated, with uncertainty spanning zero. |
| Fixed penalties, 2,000 condition evaluations | Reuse formal simulation training draws and fix each method’s penalty at its condition-specific pilot median | At size 80, joint counts were 96/200 and 118/200 for 0 and 0.90. At 0.99, 236/600 signal-group comparisons had complete substitutions. Substitution persisted when tuning was held fixed within a condition. |
| AR(1) correlation, 200 pairs at size 80 | Use fresh draws and let correlation decay with distance within a group while keeping its mean at 0.90. Adjacent correlation was approximately 0.948 | Both the AR(1) and equal-correlation designs yielded 107/200 joint events, while complete substitutions were 23/600 and 5/600, respectively. The aggregate count can agree while the within-group selection patterns differ. |

[Independent-seed results](../results/formal/replication/endpoint_contrasts.csv) · [Fixed-penalty results](../results/formal/fixed/conditions.csv) · [AR(1) results](../results/formal/ar/conditions.csv)

## Stage 7. Evaluate credit default prediction

**Input and execution.** Draw 100 pairs of stratified 3,000-customer training samples from the 24,000-customer development pool. Reuse the same draws across the two encodings and evaluate all fits on the same 6,000 test customers. Compare each fit’s selected months and selected predictor families. Use the frozen credit tolerance of 0.0039266281 to flag close test log losses.

| Main encoding | Mean test log loss | Mean Brier score | Mean variable / family Jaccard |
| --- | ---: | ---: | ---: |
| Training default rate | 0.52838 | 0.17225 | — |
| L1 logistic | 0.44424 | 0.13800 | 0.589 / 0.813 |
| Elastic Net logistic | 0.44398 | 0.13797 | 0.632 / 0.843 |

**Interpretation.** Both fitted models scored better than the baseline on this test set. All 200 L1 fits selected at least one repayment-status month, but which months they selected changed in **85/100 pairs**. After restricting to pairs with close log loss, **19/87** L1 pairs and **9/86** Elastic Net pairs had low variable overlap. The month-change count covers all pairs and uses a different definition of disagreement.

With raw numeric status codes, mean L1 log loss rose to **0.46653** and family Jaccard rose to **0.977**. Billed amounts entered 98% of those L1 fits, versus 59% under the main coding. Nearly always including a family can raise agreement at the family level even when probability forecasts deteriorate. The common test set and repeated use of development customers make the comparisons descriptive for this historical dataset. The observed outcomes let us score predictions. Exact recovery is assessed in the simulation, where the generating variables are known. [Credit report](report.md#credit-default-on-held-out-customers) · [Model and coding results](../results/formal/credit_models.csv) · [Credit figure](../results/formal/credit/credit.png)

## Stage 8. Run the exploratory controls

**Input and reason for the analysis.** The main simulation at 0.99 raised correlation in all 24 groups at once. The formal 244/600 result prompted two additional size-80 runs to see where the strong correlation needed to be. The follow-up plan specified the two new conditions before fitting them. Each condition attempted 200 fresh training pairs under the original fitting and scoring rules.

| Groups with within-group correlation 0.99 | Source | Complete substitutions among jointly detected signal groups | Low overlap among close-error pairs |
| --- | --- | ---: | ---: |
| None | Formal reference | 0/600 | 87/88 |
| Signal groups only | Exploratory | 252/588 | 89/89 |
| Null groups only | Exploratory | 0/600 | 111/112 |
| All groups | Formal reference | 244/600 | 114/114 |

Four signal-only pairs failed the convergence rule, so 196 valid pairs contributed three jointly detected groups each. Strong correlation confined to the signal groups was sufficient for frequent complete substitutions in this simulation. In the null-only condition, overall low overlap remained common even with zero observed complete substitutions inside signal groups. The whole-list difference therefore needs to be interpreted alongside where the differing variables came from. The exploratory difference between the signal-only and all-correlated substitution rates was 2.2 percentage points, with a pair-bootstrap interval of −3.6 to +7.9. The comparison leaves room for a difference in either direction.

**Outputs.** Figure 2 puts the substitution rate beside the average number of variables appearing in exactly one fit, split by signal or null group. The result files also retain the exact denominators, bootstrap contrasts and solver failures. [Follow-up plan](exploratory-plan.md) · [Exploratory configuration](../config/exploratory-rho099.json) · [Four-condition table](../results/exploratory/rho099/four_cells.csv) · [Figure 2](../results/exploratory/rho099/mechanism.png)

## Stage 9. Diagnose the formal and exploratory results

These checks use archived results and coefficients. They were added after the main analysis.

1. **Vary the definitions of close error and low overlap.** Cross the three RMSE-gap cutoffs with Jaccard cutoffs of 0.25, 0.50 and 0.75. At size 80, the Lasso joint count changed from 87/200 versus 107/200 at correlations 0 and 0.90 under the 0.50 overlap cutoff to 76/200 versus 75/200 under the 0.25 cutoff. The direction of the joint-count comparison can depend on the chosen cutoff. [Sensitivity grid](../results/exploratory/interpretation/threshold_grid.csv)
2. **Check which fit found the true variable during a swap.** Of the 244 complete substitutions in the size-80 formal condition at 0.99, exactly one fit picked the generating variable in 202 and both missed it in 42. Thus two fits can detect the right group and both select correlated substitutes. [Group and truth counts](../results/exploratory/interpretation/signal_blocks.csv)
3. **Explain why coefficients can change more than predictions.** For unit slope and standardized variables correlated at 0.99, swapping one variable for another increases population MSE by 0.02 in the simplified calculation. The recorded fits show the same geometry: at size 80 and correlation 0.99, shifts that redistributed Lasso coefficients within signal groups accounted for 99.2% of pooled squared coefficient differences in those groups, but 19.2% of their covariance-weighted squared prediction disagreement. These percentages describe the signal-group components. [Derivation and coefficient checks](mechanism.md)
4. **Inspect a concrete shared substitute.** In archived formal pair 19, the two Lasso fits had population RMSEs of 1.09977 and 1.07735. In the second signal group both chose only variable 9, while the true member was variable 8. In that same condition, four of the 54 identical nonempty signal-group selections contained only substitutes. Agreement of the fitted lists and recovery of the generating variable can give different answers. [Worked example](report.md#correlation-changes-the-source-of-disagreement)
5. **Check the comparison at comparable size and risk.** The only eligible pair provides a descriptive comparison. Resampling a single pair cannot estimate variability across pairs, so the corresponding bootstrap limits are omitted. [Interpretation plan](interpretation-checks.md) · [Design audit](research-audit.md)

## Stage 10. Verify and reproduce the study

| Verification route | Inputs and operation | Recorded result |
| --- | --- | --- |
| Rebuild from archived coefficients | Restore the compressed records, check hashes, recompute scores and regenerate tables and figures | The archive supplies original-scale coefficients, intercepts, seeds and solver diagnostics for independent scoring |
| Full refit | Start from a fresh checkout, download the UCI workbook, rerun the pilots and use the approved configuration to refit all six formal stages from seeds | Reproduced 200 simulation pilot pairs, 20 credit pilot pairs, 4,799 valid formal evaluations and the one formal failure |
| Compare published artifacts | Compare the rerun archives and rebuilt outputs at the recorded analysis version | All six formal fit archives and 50 rebuilt output files matched byte for byte at that checkpoint |
| Check analytic prediction error | Compare analytic MSE to independent simulated test data. For the largest original discrepancy, run an exploratory check using a different generator and one million observations | The original check was 3.73 Monte Carlo standard errors from the analytic value, and the exploratory rerun was −1.28 |
| Repeat the exploratory controls | Refit the 400 attempted follow-up pairs and compare with the archive and ledger | Reproduced the four failures as well as the recorded exploratory outputs |

The 4,799 formal count is a count of stage evaluations. Some stages deliberately reuse training draws for paired comparisons. Verification against the same seeds checks whether the computation can be reproduced. The independent-seed experiment in Stage 6 checks how the statistical result varies with fresh training samples.

To run the study, follow the repository’s [reproduction guide](reproduce.md). It gives the pinned environment, workbook retrieval command, archive-based rebuild, full refit and exploratory verification commands. [Completed verification](review.md) · [Full refit record](../results/formal/full_refit.json) · [Independent metric audit](../results/formal/metric_audit.json) · [Exploratory verification](../results/exploratory/rho099/verification.json)
