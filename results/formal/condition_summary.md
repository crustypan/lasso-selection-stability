# Formal simulation by condition

The target is 200 independent sample pairs in each condition. The denominator in the joint-count column gives the number of valid complete pairs. RMSE, support size, recall and false inclusions are averaged across the two training samples in each pair. Close means an absolute population RMSE gap ≤ 0.05. Low means variable Jaccard ≤ 0.5. The joint count is shown with its 95% Wilson interval for the probability.

| n | ρ signal / null | Method | Mean RMSE | Mean J, variable / group | Close / low counts | Joint count (95% CI) | Low among close | Mean size | Recall | False inclusions |
|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 0 / 0 | Lasso | 1.134 | 0.181 / 0.378 | 88 / 199 | 87/200 (36.8–50.4%) | 87/88 | 14.9 | 1.000 | 11.9 |
| 80 | 0 / 0 | Elastic Net | 1.218 | 0.170 / 0.534 | 69 / 200 | 69/200 (28.3–41.3%) | 69/69 | 26.2 | 1.000 | 23.2 |
| 80 | 0.5 / 0.5 | Lasso | 1.129 | 0.186 / 0.379 | 88 / 200 | 88/200 (37.3–50.9%) | 88/88 | 14.2 | 1.000 | 11.2 |
| 80 | 0.5 / 0.5 | Elastic Net | 1.207 | 0.177 / 0.491 | 66 / 200 | 66/200 (26.9–39.8%) | 66/66 | 25.2 | 1.000 | 22.2 |
| 80 | 0.9 / 0.9 | Lasso | 1.122 | 0.207 / 0.366 | 107 / 200 | 107/200 (46.6–60.3%) | 107/107 | 13.2 | 0.989 | 10.2 |
| 80 | 0.9 / 0.9 | Elastic Net | 1.164 | 0.269 / 0.389 | 104 / 197 | 102/200 (44.1–57.8%) | 102/104 | 23.5 | 1.000 | 20.5 |
| 80 | 0.99 / 0.99 | Lasso | 1.097 | 0.126 / 0.368 | 114 / 200 | 114/200 (50.1–63.7%) | 114/114 | 11.3 | 0.618 | 9.5 |
| 80 | 0.99 / 0.99 | Elastic Net | 1.104 | 0.420 / 0.368 | 112 / 164 | 97/200 (41.7–55.4%) | 97/112 | 26.8 | 0.993 | 23.8 |
| 240 | 0 / 0 | Lasso | 1.037 | 0.203 / 0.369 | 195 / 199 | 194/200 (93.6–98.6%) | 194/195 | 12.5 | 1.000 | 9.5 |
| 240 | 0 / 0 | Elastic Net | 1.066 | 0.161 / 0.516 | 184 / 200 | 184/200 (87.4–95.0%) | 184/184 | 24.9 | 1.000 | 21.9 |
| 240 | 0.5 / 0.5 | Lasso | 1.038 | 0.190 / 0.362 | 190 / 200 | 190/200 (91.0–97.3%) | 190/190 | 13.3 | 1.000 | 10.3 |
| 240 | 0.5 / 0.5 | Elastic Net | 1.066 | 0.172 / 0.520 | 181 / 200 | 181/200 (85.6–93.8%) | 181/181 | 26.1 | 1.000 | 23.1 |
| 240 | 0.9 / 0.9 | Lasso | 1.037 | 0.213 / 0.371 | 189 / 199 | 188/200 (89.8–96.5%) | 188/189 | 13.2 | 1.000 | 10.2 |
| 240 | 0.9 / 0.9 | Elastic Net | 1.063 | 0.257 / 0.459 | 190 / 200 | 190/200 (91.0–97.3%) | 190/190 | 25.6 | 1.000 | 22.6 |
| 240 | 0.99 / 0.99 | Lasso | 1.033 | 0.196 / 0.377 | 196 / 200 | 196/200 (95.0–99.2%) | 196/196 | 12.2 | 0.894 | 9.6 |
| 240 | 0.99 / 0.99 | Elastic Net | 1.039 | 0.446 / 0.376 | 196 / 146 | 143/200 (64.9–77.3%) | 143/196 | 25.4 | 1.000 | 22.4 |
| 80 | 0.9 / 0 | Lasso | 1.140 | 0.184 / 0.382 | 74 / 199 | 74/199 (30.8–44.1%) | 74/74 | 16.3 | 0.992 | 13.3 |
| 80 | 0.9 / 0 | Elastic Net | 1.195 | 0.261 / 0.481 | 87 / 197 | 86/199 (36.5–50.2%) | 86/87 | 28.8 | 1.000 | 25.8 |
| 80 | 0 / 0.9 | Lasso | 1.109 | 0.219 / 0.357 | 95 / 198 | 95/200 (40.7–54.4%) | 95/95 | 11.6 | 1.000 | 8.6 |
| 80 | 0 / 0.9 | Elastic Net | 1.163 | 0.163 / 0.425 | 83 / 200 | 83/200 (34.9–48.4%) | 83/83 | 21.0 | 1.000 | 18.0 |

The last four rows are the two additional mechanism controls at n = 80. The [full condition table](main/conditions.csv) also includes the Ridge baseline, double-empty rates, both marginal probabilities, continuous distributions, common signal-group hits and complete swaps. See [threshold sensitivities](main/thresholds.csv) for all event counts and intervals.
