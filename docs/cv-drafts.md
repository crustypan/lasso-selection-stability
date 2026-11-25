# Application CV wording

Junyang selected the research question, approved the design and formal configuration, and commissioned the two exploratory controls and separate joint and conditional denominators. Codex wrote the implementation, ran the experiments, and checked the results. The lines below distinguish Junyang’s design decisions from the findings of the study.

## Statistics

- Selected the question and approved a paired-sample design for Lasso and Elastic Net with 120 predictors in 24 blocks, then commissioned two high-correlation controls to distinguish signal-group substitutions from null-variable selections.
- At training size 80, the follow-up found complete Lasso substitutions in 252/588 jointly detected signal groups with signal-only correlation of 0.99 and 0/600 with null-only correlation, despite low overlap in 111/112 close-error pairs in the latter condition.

## Alternative emphases

- **DSML.** The study benchmarked predictive performance and feature reproducibility across independently tuned models. At training size 80 and correlation 0.99, Elastic Net raised mean variable Jaccard from 0.126 to 0.420 while mean model size grew from 11.35 to 26.82.
- **Mathematics.** The analysis linked the excess prediction risk of a correlated proxy, 2b²(1 − ρ), to exact population RMSE. With group mean correlation fixed at 0.9, complete Lasso substitutions at training size 80 occurred in 5/600 equicorrelated and 23/600 AR(1) signal blocks jointly detected by both fits.
- **Finance / FinTech.** On 6,000 held-out credit-card clients, L1 logistic regression achieved mean log loss 0.444 versus 0.528 for the prevalence baseline. Repayment status entered all 200 fits, while the selected months differed in 85/100 training pairs.

## Emphasis for an MSc Statistics application

The NUS MSc Statistics curriculum lists Design of Experiments, Applied Regression Analysis and Statistical Foundations of Data Science. For this project, the strongest examples are the controlled placement of correlation, the distinction between two probability denominators and the decision to limit a claim when the evidence changes. The held-out credit case provides a short applied example. [NUS programme structure](https://www.stat.nus.edu.sg/education/graduate/msc-in-statistics/prospective-students/) · [Evidence behind the methods](research-audit.md)

**Short research account.** I chose to study whether a reproducible prediction error implies a reproducible variable list. In this code-assisted study, I approved a design with known generating variables. After the formal run showed frequent substitutions at correlation 0.99, I requested separate signal-only and null-only controls. The null-only control produced low overlap in 111 of 112 pairs with close errors, yet never produced a complete substitution within a signal group. This is the distinction I would retain in reporting the result.

[Formal report](report.md) · [Completed verification](review.md)
