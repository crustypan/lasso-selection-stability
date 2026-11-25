# Prediction risk and support recovery

Let the two standardized predictors have unit variance and correlation ρ. If the conditional mean is bX₁ and the independent noise has variance σ², then the true predictor has risk σ². Replacing it with bX₂ gives the residual ε + b(X₁ − X₂). The cross term vanishes by independence and centering. Therefore the increase in population mean squared error is

```text
b² Var(X₁ − X₂) = b² [Var(X₁) + Var(X₂) − 2 Cov(X₁,X₂)]
                = 2b²(1 − ρ).
```

For b = 1 and ρ = 0.99 the excess MSE is 0.02. With noise SD one, total RMSE rises from 1 to √1.02, about 1.010. This comparison holds the slope fixed during a complete substitution. Finite-sample Lasso also changes the slope, the intercept and other selected terms. A small substitution cost by itself supplies no probability that a fitted algorithm will make that substitution.

In the simulation the three active variables belong to independent blocks, so the signal variance is 1 + 1 + 1 = 3 at every within-block correlation. For a fitted intercept a and slope error δ, population MSE is 1 + a² + δᵀΣδ. Pointwise prediction disagreement between two fits on a new random subject has root mean square √[(a₁ − a₂)² + (β̂₁ − β̂₂)ᵀΣ(β̂₁ − β̂₂)]. Models with the same risk can disagree substantially on subjects. The study retains both quantities.

A complete substitution is counted only when both fits hit a signal block and their selected within-block members are disjoint. Enlarging a support while retaining its original members is a separate recorded outcome. Selection frequencies, false inclusions and symmetric differences from signal and null blocks supply additional checks against a misleading interpretation of low Jaccard.

## Why coefficient differences can have a small predictive cost

For a block of five standardized variables with common correlation $\rho$, the covariance is $\Sigma_g=(1-\rho)I+\rho\mathbf 1\mathbf 1^T$. Let $d_g$ be the difference in original-scale coefficients between two fits, $\bar d_g$ its mean, and $c_g=d_g-\bar d_g\mathbf 1$. The constant direction and the within-block contrasts are orthogonal. Multiplying by the covariance gives

$$
d_g^T\Sigma_g d_g
= (1+4\rho)\,5\bar d_g^2 + (1-\rho)\,\lVert c_g\rVert^2.
$$

Adding this quantity over independent blocks and including the squared intercept difference gives the mean squared difference between the two predictions on the same subject. Moving a coefficient of size $b$ from one member to another keeps the block sum fixed. Its entire coefficient change lies in $c_g$, with squared length $2b^2$. This recovers the two-variable substitution cost above. At $\rho=0.99$, both eigenvalues are positive, but equal-length changes along the common and contrast directions have predictive costs in the ratio $4.96/0.01=496$.

**Check against fitted coefficients (exploratory).** Pool squared distances over all 200 formal Lasso pairs at training size 80, using just their three signal blocks. At correlation 0.99, within-block contrasts accounted for **99.2% of squared coefficient distance** and **19.2% of the signal-block contribution to squared prediction disagreement**. The first ratio uses ordinary Euclidean distance. The second uses covariance-weighted distance. Both are ratios of sums, and neither is a fraction of total response variance. At zero correlation, the two shares coincided at 79.7%, as the identity covariance predicts. Numerical reconstruction using every block and the intercept agreed with archived prediction disagreement for all 4,790 sparse-method pair records in the main and selective-correlation analyses. [Coefficient decomposition](../results/exploratory/interpretation/signal_blocks.csv) · [Implementation](../src/interpretation_checks.py)

## Sources

- Zou and Hastie (2005), [Regularization and Variable Selection Via the Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x).
- Morris, White and Crowther (2019), [Using simulation studies to evaluate statistical methods](https://doi.org/10.1002/sim.8086).
- Yeh (2009), [Default of Credit Card Clients](https://doi.org/10.24432/C55S3H), UCI Machine Learning Repository, CC BY 4.0.
