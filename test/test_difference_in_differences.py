import numpy as np
import pandas as pd

import causal_inference.difference_in_differences as did


def generated_staggered_treatment_data(treatment_effect=3):
    # Parameters
    n_units = 1000
    n_pre = 5
    n_post = 5
    random_seed = 42

    np.random.seed(random_seed)

    # Create unit and time grids
    units = np.arange(n_units)
    times = np.arange(-n_pre, n_post + 1)

    # Create cartesian product of units and time
    df = pd.DataFrame([(u, t) for u in units for t in times], columns=["unit", "time"])

    # Assign treatment randomly to half the units
    treated_units = np.random.choice(units, size=n_units // 2, replace=False)

    # Assign a random treatment start period for each treated unit (between -n_pre and n_post)
    treatment_start_dict = {u: np.random.randint(-n_pre + 2, n_post) for u in treated_units}

    # Compute treated indicator and post-treatment periods
    df["treated"] = 0
    df["treatment_start"] = df["unit"].map(treatment_start_dict)
    df["treated"] = ((df["unit"].isin(treated_units)) & (df["time"] >= df["treatment_start"])).astype(int)

    # Calculate baseline trend (can customize this)
    df["baseline"] = 10 + df["unit"] * 0.1 + df["time"] * 0.2

    # Apply treatment effect only for treated units and post-treatment periods
    df["treatment_effect"] = np.where((df["treated"] == 1), treatment_effect, 0)

    # Add noise
    df["noise"] = np.random.normal(0, 1, size=len(df))

    # Final outcome
    df["outcome"] = df["baseline"] + df["treatment_effect"] + df["noise"]

    # Drop intermediate columns if desired
    df = df[["unit", "time", "treated", "outcome"]]

    return df


def test_staggered_treatment_effect_close_to_truth():
    treatment_effect = 3
    data = generated_staggered_treatment_data(treatment_effect=treatment_effect)
    model = did.Staggered(data=data, unit_col="unit", time_col="time", value_col="outcome", treatment="treated", covariates=None, cov_type="HC3").fit()
    # Only check post-treatment event times (event_time >= 0)
    effects = model.model_effects["outcome"]
    post_effects = effects[effects.index >= 0]["estimate"]
    assert np.allclose(post_effects, treatment_effect, atol=0.5)
    model.plot()
