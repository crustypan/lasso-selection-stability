"""Audit the design invariants and the credit evaluation boundary on the lab data."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import credit
from common import CFG, ROOT, atomic_json, seed
from credit_data import TARGET, design, split, workbook
from simulation import CONDITIONS, TRUTH, covariance
from verify_formal import fit_records


def check_credit(frozen):
    df, hashes = workbook()
    assert hashes["xls_sha256"] == frozen["xls_sha256"]
    y = df[TARGET].to_numpy()
    dev, test, pilot, validation = split(df)
    assert df.ID.is_unique and set(dev).isdisjoint(test)
    assert set(pilot).isdisjoint(validation) and set(pilot) | set(validation) == set(dev)
    rows = fit_records("credit")
    raw = {r["pair"]: r for r in fit_records("credit_raw")}
    for record in rows:
        i = record["pair"]
        draw_seeds = [seed(frozen["credit_formal_seed"], i, k) for k in (0, 1)]
        assert record["sampling_seeds"] == raw[i]["sampling_seeds"] == draw_seeds
        pool = dev
        members = []
        for side, draw_seed in enumerate(draw_seeds):
            selected, pool = train_test_split(pool, train_size=frozen["credit_each"], stratify=y[pool],
                                              random_state=draw_seed)
            expected_seed = seed(frozen["credit_formal_seed"], i, 2 + side)
            for fit_record in (record, raw[i]):
                assert fit_record["samples"][side]["fold_seed"] == expected_seed
                assert fit_record["samples"][side]["defaults"] == y[selected].sum()
            members.append(set(selected))
        assert members[0].isdisjoint(members[1])
        assert not (members[0] | members[1]) & set(test)

    # Use pilot development data to exercise the same training entry point.
    selected, _ = train_test_split(pilot, train_size=frozen["credit_each"], stratify=y[pilot],
                                   random_state=seed(frozen["credit_pilot_seed"], 0, 0))
    fold_seed = seed(frozen["credit_pilot_seed"], 0, 2)
    x = design(df)
    original = credit.train(x[selected], y[selected], x[validation], y[validation], fold_seed)
    permuted = np.random.default_rng(35091).permutation(y[validation])
    changed = credit.train(x[selected], y[selected], x[validation], permuted, fold_seed)
    for method in ("lasso", "elastic"):
        for field in ("alpha", "coef", "intercept", "cv_loss"):
            np.testing.assert_array_equal(original[method][field], changed[method][field])
        assert original[method]["loss"] != changed[method]["loss"]
    return dict(customers=len(y), development=len(dev), held_out=len(test),
                paired_customer_draws_checked=len(rows), encodings_checked=2,
                pilot_validation_labels_permuted=len(validation),
                invariant_fit_fields=["alpha","coef","intercept","cv_loss"],
                changed_validation_log_losses=2)


def check_design():
    frozen = json.loads((ROOT / "config/frozen.json").read_text())
    for field in ("truth_zero_based","sigma","groups","width","n","rho","l1_ratios","linear_grid",
                  "ridge_grid","credit_grid","folds","credit_each","credit_split_seed","credit_validation_seed",
                  "credit_test_fraction","credit_validation_fraction","thresholds","low_jaccard"):
        assert frozen[field] == CFG[field], field
    assert (TRUTH @ TRUTH, len(TRUTH), frozen["folds"]) == (3, 120, 5)
    controls = json.loads((ROOT / "config/exploratory-rho099.json").read_text())["conditions"]
    designs = list(CONDITIONS) + [(x["n"],x["rho_signal"],x["rho_null"]) for x in controls]
    for _, rs, rn in designs:
        matrix = covariance(rs, rn)
        np.testing.assert_allclose(TRUTH @ matrix @ TRUTH, 3)
        assert np.linalg.eigvalsh(matrix)[0] > 0
        np.testing.assert_allclose(np.diag(matrix), 1)
    matched = pd.read_csv(ROOT/"results/formal/main/method_differences.csv").query("subset == 'matched'")
    assert matched.loc[matched.pairs < 2, ["lo","hi"]].isna().all().all()
    accepted = matched[matched.outcome == "j_var"]
    return dict(simulation_design_cells=len(designs),signal_variance=3,noise_variance=1,
                strictly_positive_covariance_eigenvalues=True,
                eligible_size_and_risk_matched_pairs=int(accepted.pairs.sum()),
                matched_conditions_with_estimable_intervals=int((accepted.pairs>=2).sum()),
                credit=check_credit(frozen))


if __name__ == "__main__":
    result = check_design()
    atomic_json(ROOT/"results/exploratory/interpretation/design_audit.json", result)
    print(json.dumps(result,indent=2))
