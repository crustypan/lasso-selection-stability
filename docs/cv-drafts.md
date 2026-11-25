# Application CV wording

Junyang selected the research question and approved the design and formal configuration. Codex wrote the implementation, ran the experiments, and checked the results. The lines below distinguish Junyang’s design decisions from the findings of the study.

## Statistics

- Selected the research question and approved a paired-sample study of Lasso and Elastic Net with 120 predictors in 24 blocks, two training sizes, and separate five-fold tuning across 10 simulation conditions.
- At training size 80, complete Lasso proxy substitution occurred in 5/600 jointly detected signal blocks at correlation 0.90 and 244/600 at 0.99. At training size 240 the latter frequency fell to 65/600.

## Alternative emphases

- **DSML.** The study benchmarked predictive performance and feature reproducibility across independently tuned models. At training size 80 and correlation 0.99, Elastic Net raised mean variable Jaccard from 0.126 to 0.420 while mean model size grew from 11.35 to 26.82.
- **Mathematics.** The analysis linked the excess prediction risk of a correlated proxy, 2b²(1 − ρ), to exact population RMSE. With group mean correlation fixed at 0.9, complete Lasso substitutions at training size 80 occurred in 5/600 equicorrelated and 23/600 AR(1) signal blocks jointly detected by both fits.
- **Finance / FinTech.** On 6,000 held-out credit-card clients, L1 logistic regression achieved mean log loss 0.444 versus 0.528 for the prevalence baseline. Repayment status entered all 200 fits, while the selected months differed in 85/100 training pairs.

[Formal report](report.md) · [Completed verification](review.md)
