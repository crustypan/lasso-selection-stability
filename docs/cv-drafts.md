# CV wording for Junyang's review

The role recorded here is selection of the research question and approval of the study design. The assistant implemented and ran the code. Personal replication and interpretation can be added after Junyang completes them.

## Statistics

- Selected the research question and approved a paired-sample simulation design with 120 predictors in 24 blocks, two training sizes, and independently tuned Lasso and Elastic Net models across 10 conditions.
- Quantified complete Lasso proxy substitution in 5/600 jointly detected signal blocks at correlation 0.90 and 244/600 at 0.99 (n = 80), while the larger-sample joint instability rate stayed near its ceiling.

## Alternative emphases

- **DSML.** Compared predictive risk with feature reproducibility under resampling. At n = 80 and correlation 0.99, Elastic Net raised mean variable Jaccard from 0.126 to 0.420 and mean model size from 11.35 to 26.82.
- **Mathematics.** Connected the excess prediction risk from a correlated proxy, 2b²(1 − ρ), to exact covariance-based population RMSE and observed support changes under equicorrelated and AR(1) block designs.
- **Finance / FinTech.** Evaluated credit-default probabilities on 6,000 held-out customers. L1 logistic regression achieved mean log loss 0.444 versus 0.528 for prevalence, while selected repayment-status months varied in 85 of 100 training pairs.

[Formal report](report.md) · [Review and replication instructions](review.md)
