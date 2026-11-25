# Junyang's research review

The frozen configuration, model fits, figures, formal report and computational checks are complete. Junyang selected the question and authorized the frozen experiment. Personal review of the formal evidence remains open.

1. **Check the design.** From the three nonzero coefficients and block independence, derive the signal variance as 3 at every correlation. Identify the two extra n = 80 conditions where only the signal blocks or only the null blocks are correlated. [Protocol](protocol.md) · [Frozen configuration](../config/frozen.json)
2. **Read the main figure alongside its counts.** Inspect the lower row for both sample sizes. Use the condition table to write down the high-minus-low correlation contrast at n = 80 and at n = 240. Check the numerators for low overlap conditional on close risk. [Main figure](../results/formal/main/simulation.png) · [Condition table](../results/formal/condition_summary.md)
3. **Reproduce a robustness result.** In the repository root on the lab server, run the commands below. They read the frozen settings and published archives and refit in temporary directories. Compare the independent-seed endpoint estimate and its interval with the main estimate. The full option repeats all 200 pairs at n = 80 and correlation 0.9 in both runs, plus four specified pairs from the other stages.
4. **Explain the distinction in your own words.** Why can two predictors with close population RMSE both select an incorrect proxy? Use the 54 identical memberships and the four identical proxy-only memberships at n = 80, correlation 0.99 to explain which result establishes correctness. Then explain what the credit labels let you check. [Mechanism counts](../results/formal/mechanisms.csv)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/mpl"
.venv/bin/python src/verify_formal.py --whole-condition --jobs 2
cat results/formal/main/endpoint_contrasts.csv
cat results/formal/replication/endpoint_contrasts.csv
```

The assistant verified the original six stages, independently rescored the held-out credit predictions, and reran 404 selected formal pair records. Junyang's own execution, explanation and public-release decision are reserved for his review.
