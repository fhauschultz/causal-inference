import numpy as np
import pandas as pd

from causal_inference import BinaryDiD, EventStudy
from causal_inference.difference_in_differences import conf_int_robust


def test_estimates():
    # generate data
    test_data, true_effect_size = generate_test_data()
    # fit model
    did_model = EventStudy(test_data, ["id"], "time", "treatment", ["y"], confidence_level=0.1).fit()
    treatment_effect = did_model.model_effects["y"]["treatment_effect"].values
    assert np.all(np.isclose(true_effect_size["effect"].values, treatment_effect, atol=0.1)), "Treatment effect estimates do not match"


def test_conf_int_robust():
    test_data, _ = generate_test_data()
    # fit model
    model = EventStudy(test_data, ["id"], "time", "treatment", ["y"], confidence_level=0.1).fit()
    model = model.models["y"]
    non_robust_from_new_function = conf_int_robust(model, 0.05, cov_type="nonrobust")["lower"].values
    built_in_robust = model.conf_int()[0].values
    assert np.all(np.isclose(non_robust_from_new_function, built_in_robust)), "Non-robust confidence intervals do not match built in version"


def generate_test_data(seed=42):
    effect_size = 3
    periods = 15
    treatment_start = 10
    n = 10000
    noise_level = 1.0
    np.random.seed(seed)

    # Generate IDs for individuals
    ids = np.arange(n)

    # Generate treatment assignment (50/50 split)
    treatment = np.random.choice([0, 1], size=n)

    # Create time periods
    time = np.tile(np.arange(periods), n)

    # Expand data to have multiple rows per individual (one per time period)
    data = pd.DataFrame({"id": np.repeat(ids, periods), "time": time, "treatment": np.repeat(treatment, periods)})

    data["treated"] = data["treatment"] * data["time"].apply(lambda x: 1 if x >= treatment_start else 0)

    # Baseline outcome
    baseline = np.random.normal(10, 2, n)
    data["baseline"] = np.repeat(baseline, periods)

    # Time-varying treatment effect (after specified treatment start period)
    data["effect"] = np.where((data["time"] >= treatment_start) & (data["treatment"] == 1), -(data["time"] - treatment_start + 1) / 5 + effect_size, 0)

    # Outcome generation
    data["y"] = (
        data["baseline"]
        + data["effect"]  # Time-varying DiD effect
        + np.random.normal(0, noise_level, size=len(data))  # Noise
    )

    true_effect_size = data[data.treatment == 1][["time", "effect"]].drop_duplicates(subset=["time"])
    true_effect_size["periods_since_treatment"] = true_effect_size["time"] - treatment_start

    return data, true_effect_size


def generate_test_data_2():
    # Parameters
    n_units = 100
    n_pre = 5
    n_post = 5
    treatment_effect = 3.0
    random_seed = 42

    np.random.seed(random_seed)

    # Create unit and time grids
    units = np.arange(n_units)
    times = np.arange(-n_pre, n_post + 1)

    # Create cartesian product of units and time
    df = pd.DataFrame([(u, t) for u in units for t in times], columns=["unit", "time"])

    # Assign treatment randomly to half the units
    treated_units = np.random.choice(units, size=n_units // 2, replace=False)
    df["treated"] = df["unit"].isin(treated_units).astype(int)

    # Calculate baseline trend (can customize this)
    df["baseline"] = 10 + df["unit"] * 0.1 + df["time"] * 0.2

    # Apply treatment effect only for treated units and post-treatment periods
    df["treatment_effect"] = np.where((df["treated"] == 1) & (df["time"] >= 0), treatment_effect, 0)

    # Add noise
    df["noise"] = np.random.normal(0, 1, size=len(df))

    # Final outcome
    df["outcome"] = df["baseline"] + df["treatment_effect"] + df["noise"]

    # Drop intermediate columns if desired
    df = df[["unit", "time", "treated", "outcome"]]

    return df


def estimate_did(df, pre_period=-1, post_period=1, outcome_col="outcome"):
    """
    Estimate the Difference-in-Differences (DiD) treatment effect using groupby and pivot.

    Parameters:
    - df: pandas DataFrame containing columns 'treated', 'time', and the outcome variable
    - pre_period: time period before treatment (e.g., -1)
    - post_period: time period after treatment (e.g., 1)
    - outcome_col: name of the outcome column (default: 'outcome')

    Returns:
    - did_estimate: float, estimated treatment effect
    - pivot: DataFrame showing average outcomes and differences
    """
    grouped = df.groupby(["treated", "time"])[outcome_col].mean().reset_index()

    pivot = grouped.pivot(index="time", columns="treated", values=outcome_col)
    pivot.columns = ["control", "treated"]  # 0 = control, 1 = treated
    pivot["diff"] = pivot["treated"] - pivot["control"]

    # Estimate DiD: (Treated_Post - Control_Post) - (Treated_Pre - Control_Pre)
    did_estimate = pivot.loc[pivot.index >= 0, "diff"].mean() - pivot.loc[pivot.index < 0, "diff"].mean()

    return did_estimate, pivot


def test_event_study_class():
    df = generate_test_data_2()
    experiment_eval = EventStudy(df, ["unit"], "time", "outcome", "treated")
    experiment_eval.fit(significance_level=0.1)

    assert np.all(np.isclose(experiment_eval.model_effects["outcome"]["treatment_effect"].values, estimate_did(df)[1]["diff"].values))


def test_bdid_class():
    df = generate_test_data_2()
    bdid = BinaryDiD(df, "treated", "time", "outcome", "treated")

    bdid.fit(significance_level=0.1)
    print(bdid.model_effects["outcome"].loc[5]["treatment_effect_cumulative"])
    print(estimate_did(df)[0])
    assert np.isclose(bdid.model_effects["outcome"].loc[5]["treatment_effect_cumulative"], estimate_did(df)[0])
