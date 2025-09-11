import re

import linearmodels as lm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from causal_inference.utils import BaseCausalInference


class StaggeredDID(BaseCausalInference):
    def fit(self, alpha=0.1, cov_type="clustered", method="twfe", event_time_window=None):
        self.models = {}
        self.model_effects = {}
        if method == "twfe":
            self.did_model, self.df, table = estimate_staggered_did(self.data, self.unit_col, self.time_col, self.value_col, "treatment_start", covariates=self.covariates, alpha=alpha, cov_type=cov_type)
            self.model_effects[self.value_col] = table
            self.models[self.value_col] = self.did_model
        elif method == "sun_abraham":
            res = estimate_staggered_did_sa(self.data, self.unit_col, self.time_col, self.value_col, "treatment_start", covariates=self.covariates, reference_period=-1, event_time_window=event_time_window, alpha=alpha, cov_type=cov_type)
            # Store aggregated SA event-study (safe visual)
            self.model_effects[self.value_col] = res["agg_effects"]
            self.models[self.value_col] = res["fit"]
            self._sa_details = res  # keep cohort-level results if you want a cohort plot
        else:
            raise ValueError("Unknown method. Use 'twfe' or 'sun_abraham'.")
        self.model_fitted = True
        return self

    def plot(
        self,
        figsize=None,
        matplotlib_style="ggplot",
        matplotlib_theme_color="navy",
    ):
        self.matplotlib_style = matplotlib_style
        self.matplotlib_theme_color = matplotlib_theme_color
        plt.style.use(self.matplotlib_style)
        title = "Estimated Effect"
        if not self.model_fitted:
            raise ValueError("Model must be fitted before plotting effects.")

        if figsize is None:
            figsize = (10, 6)
        effects = self.model_effects[self.value_col]
        plt.figure(figsize=figsize)
        plt.plot(effects.index, effects["estimate"], label="Estimate", color=self.matplotlib_theme_color, lw=3)
        plt.fill_between(effects.index, effects["ci_lower"], effects["ci_upper"], color=self.matplotlib_theme_color, alpha=0.2, label="Error Band")
        plt.axhline(0, color="black", linestyle="--", lw=2)
        plt.axvline(0, color="black", linestyle="-.", lw=2)
        plt.xlabel("Time Since Treatment Start")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()


def conf_int(model, alpha=0.05):
    """
    Compute confidence intervals using specified covariance type, including heteroscedasticity-robust standard errors.

    Parameters:
    - model: A fitted statsmodels regression model (e.g., OLS).
    - alpha: Significance level for the confidence interval (default is 0.05 for 95% CI).
    - : The covariance type (e.g., 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3').
    - cov_kwds: Additional arguments for robust covariance calculation.

    Returns:
    - DataFrame containing the lower and upper bounds of the confidence intervals.
    """

    t_value = norm.ppf(1 - alpha / 2)

    # Extract parameter estimates and standard errors.
    params = model.params
    res = pd.concat([model.params, model.std_errors], axis=1)
    # Scale standard error to 95% CI
    res["ci"] = res["std_error"] * t_value

    # Compute confidence intervals
    ci_lower = params - res["ci"]
    ci_upper = params + res["ci"]

    return pd.DataFrame({"lower": ci_lower, "upper": ci_upper})


def extract_staggered_treatment_effect(model, alpha=0.05, reference_period=-1, cov_type="nonrobust", **cov_kwds):
    """
    Extract estimated treatment effects and confidence intervals for staggered DiD.
    Returns a DataFrame with event-time effects and confidence bands.
    Allows toggling robust covariance type via cov_type.
    """

    # Only keep coefficients for event_time dummies (dynamic treatment effects)
    effect_params = {k: v for k, v in model.params.items() if k.startswith("event_time_")}
    # Compute robust confidence intervals
    ci = conf_int(model, alpha=alpha)
    effect_ci = {k: ci.loc[k] for k in effect_params.keys() if k in ci.index}

    # Extract event-time numbers
    def extract_number(s):
        match = re.search(r"event_time_(-?\d+)", s)
        return int(match.group(1)) if match else None

    estimates = {extract_number(k): v for k, v in effect_params.items()}
    lower = {extract_number(k): effect_ci[k]["lower"] for k in effect_params.keys() if k in effect_ci}
    upper = {extract_number(k): effect_ci[k]["upper"] for k in effect_params.keys() if k in effect_ci}

    # Build DataFrame
    df = pd.DataFrame({"estimate": pd.Series(estimates), "ci_lower": pd.Series(lower), "ci_upper": pd.Series(upper)})
    df.index.name = "event_time"
    df.loc[reference_period, ["estimate", "ci_lower", "ci_upper"]] = [0, 0, 0]

    return df.sort_index()


def estimate_staggered_did(data, unit_col, time_col, outcome_col, treatment_start_col, covariates=None, reference_period=-1.0, alpha=0.1, cov_type="clustered"):
    """
    Estimate staggered Difference-in-Differences (DiD) using statsmodels OLS.
    Each treated unit can have a different treatment start period.

    Args:
        data: DataFrame with panel data.
        unit_col: Column name for unit identifier.
        time_col: Column name for time variable.
        outcome_col: Column name for outcome variable.
        treatment_start_col: Column name for treatment start period (NaN for never-treated).
        covariates: List of additional covariate column names (optional).

    Returns:
        model: Fitted statsmodels OLS model.
        df: DataFrame with periods since treatment and treatment indicator.
    """

    df = data.copy()
    # Calculate periods since treatment start (NaN for never-treated)
    df["periods_since_treatment"] = df[time_col] - df[treatment_start_col]
    df["periods_since_treatment"] = df["periods_since_treatment"].fillna(0)

    # Optionally, create event-time dummies for dynamic effects
    event_time_dummies = pd.get_dummies(df["periods_since_treatment"].fillna(0), prefix="event_time", dtype=float)
    # Only keep dummies for post-treatment periods (e.g., 0, 1, 2, ...)
    post_event_dummies = event_time_dummies.loc[:, event_time_dummies.columns.str.contains("event_time_")].drop(columns=f"event_time_{reference_period}")

    # Build regression matrix
    X = df[[unit_col, time_col]]
    X = X.join(post_event_dummies)

    if covariates:
        X = X.join(df[covariates].astype(float))
    y = df[[unit_col, time_col, outcome_col]].set_index([unit_col, time_col])
    y[outcome_col] = y[outcome_col].astype(float)
    X = X.set_index([unit_col, time_col])

    model = lm.PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    estimates = extract_staggered_treatment_effect(fit, alpha=alpha, reference_period=reference_period)
    return model, df, estimates


def _make_event_time(s, g):
    # s: time, g: cohort start time
    return s - g


def estimate_staggered_did_sa(
    data,
    unit_col,
    time_col,
    outcome_col,
    treatment_start_col,
    covariates=None,
    reference_period=-1,
    event_time_window=None,  # e.g., (-5, 5); None = use full support
    alpha=0.1,
    cov_type="clustered",
):
    """
    Sun & Abraham (2020) interaction-weighted / stacked event-study estimator.

    - For each cohort g, keep (i) treated units in g and (ii) NOT-YET-TREATED (and never-treated) units at each t.
    - Construct cohort-specific event-time dummies 1{G_i=g}*1{t-g = l}, omit l=reference_period.
    - Stack cohorts into one dataset and run TWFE on the stack with unit and time FE.
    - Aggregate cohort-specific coefficients at each l using cohort-size weights.

    Returns
    -------
    results : dict with keys
        - 'fit': linearmodels fit result on stacked sample
        - 'cohort_effects': DataFrame of beta_{l,g} with CIs
        - 'agg_effects': DataFrame of interaction-weighted beta_l with CIs
        - 'stack': stacked dataframe used in regression
    """
    df = data.copy()

    # Identify cohorts (treated units with a finite start time)
    cohorts = df[[unit_col, treatment_start_col]].drop_duplicates().dropna(subset=[treatment_start_col])
    cohorts = cohorts.rename(columns={treatment_start_col: "cohort_start"})

    # Precompute panel size per cohort for weights
    cohort_sizes = cohorts.groupby("cohort_start")[unit_col].nunique().sort_index()

    # Build stacked dataset
    stacks = []
    for g, g_units in cohorts.groupby("cohort_start"):
        # Sample: cohort g units + NOT-YET-TREATED (including never-treated) at each t
        # Not-yet-treated at time t means: (t < cohort_start_i) or never-treated
        tmp = df.copy()
        # Mark whether unit belongs to cohort g
        tmp["is_cohort_g"] = tmp[unit_col].isin(set(g_units[unit_col]))
        # Unit's own treatment start
        tmp["_g_i"] = tmp[treatment_start_col]
        # Not-yet-treated indicator at (i,t): True if never-treated OR t < g_i
        tmp["not_yet_treated"] = tmp["_g_i"].isna() | (tmp[time_col] < tmp["_g_i"])

        # Keep: (cohort g units) OR (not-yet-treated at t)
        tmp = tmp.loc[tmp["is_cohort_g"] | tmp["not_yet_treated"]].copy()

        # Event time relative to cohort g
        tmp["event_time"] = tmp[time_col] - g

        # Optional window trimming
        if event_time_window is not None:
            lo, hi = event_time_window
            tmp = tmp.loc[(tmp["event_time"] >= lo) & (tmp["event_time"] <= hi)]

        # Create cohort-specific event-time dummies (prefix includes g)
        et = tmp["event_time"].astype(int)
        dums = pd.get_dummies(et, prefix=f"et_g{g}", dtype=float)

        # Drop reference period l = reference_period
        ref_col = f"et_g{g}_{int(reference_period)}"
        if ref_col in dums.columns:
            dums = dums.drop(columns=ref_col)

        # Assemble design
        X_parts = [dums]
        if covariates:
            X_parts.append(tmp[covariates].astype(float))

        Xg = pd.concat(X_parts, axis=1)
        yg = tmp[[unit_col, time_col, outcome_col]].copy()
        # Create stacked unit id (i,g) to allow unit FE per stack row group
        yg["stack_unit"] = yg[unit_col].astype(str) + f"__g{g}"
        Xg["stack_unit"] = yg["stack_unit"]
        Xg[time_col] = yg[time_col]

        # Keep meta to later retrieve cohort g, event_time, etc.
        tmp = tmp[[unit_col, time_col, "event_time"]].copy()
        tmp["stack_unit"] = yg["stack_unit"]

        Xg.set_index(["stack_unit", time_col], inplace=True)
        yg.set_index(["stack_unit", time_col], inplace=True)

        stacks.append((g, Xg, yg[[outcome_col]], cohort_sizes.loc[g]))

    # Concatenate across cohorts
    X_list, y_list, w_list = [], [], []
    for g, Xg, yg, wg in stacks:
        X_list.append(Xg)
        y_list.append(yg)
        w_list.append((g, wg))
    X_stack = pd.concat(X_list, axis=0).sort_index()
    y_stack = pd.concat(y_list, axis=0).sort_index()
    weights = dict(w_list)  # cohort-size weights for aggregation

    # Fit PanelOLS on stacked sample with unit and time FE; cluster by original unit if available
    # Here we can only cluster at the stacked unit level; if you want original-unit clustering,
    # add original unit id to the index and pass cluster_entity=True (approximate).
    model = lm.PanelOLS(y_stack, X_stack, entity_effects=True, time_effects=True)
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    # Extract cohort-specific coefficients
    params = fit.params.copy()
    ses = fit.std_errors.copy()
    # Parse names like "et_g{g}_{l}"
    rows = []
    for name, val in params.items():
        if not name.startswith("et_g"):
            continue
        # name pattern: et_g{g}_{l}
        try:
            g_str, l_str = name.split("_")[1:]  # ["g{g}", "{l}"]
            g_val = int(g_str[1:])  # drop 'g'
            l_val = int(l_str)
        except Exception:
            continue
        rows.append((g_val, l_val, val, ses.loc[name]))
    if not rows:
        raise ValueError("No cohort-specific event-time coefficients were constructed. Check your inputs.")

    coef_df = pd.DataFrame(rows, columns=["cohort_start", "event_time", "estimate", "std_error"])
    # Add CIs for cohort-specific betas
    z = norm.ppf(1 - alpha / 2)
    coef_df["ci_lower"] = coef_df["estimate"] - z * coef_df["std_error"]
    coef_df["ci_upper"] = coef_df["estimate"] + z * coef_df["std_error"]

    # Interaction-weighted aggregation across cohorts at each event time
    # weights w_g proportional to cohort size; drop cohort/event cells not present
    w_series = pd.Series(weights, name="w")
    coef_df = coef_df.merge(w_series, left_on="cohort_start", right_index=True, how="left")
    # Normalize weights within each event_time using available cohorts
    coef_df["w_norm"] = coef_df.groupby("event_time")["w"].transform(lambda s: s / s.sum())

    agg = coef_df.assign(weighted_est=lambda d: d["w_norm"] * d["estimate"]).groupby("event_time", as_index=True).agg(estimate=("weighted_est", "sum")).sort_index()

    # Conservative SEs for the weighted averages (delta method with independence assumption).
    # For more exact inference, run a single regression that directly constructs the
    # l-specific weighted index, or bootstrap. This keeps things simple and transparent.
    agg_se = coef_df.assign(weighted_var=lambda d: (d["w_norm"] ** 2) * (d["std_error"] ** 2)).groupby("event_time")["weighted_var"].sum().pow(0.5)
    agg["std_error"] = agg_se
    agg["ci_lower"] = agg["estimate"] - z * agg["std_error"]
    agg["ci_upper"] = agg["estimate"] + z * agg["std_error"]

    # Insert reference period at zero (omitted by construction)
    if reference_period not in agg.index:
        agg.loc[reference_period, ["estimate", "std_error", "ci_lower", "ci_upper"]] = [0.0, 0.0, 0.0, 0.0]
    agg = agg.sort_index()

    return {
        "fit": fit,
        "cohort_effects": coef_df.set_index(["cohort_start", "event_time"]).sort_index(),
        "agg_effects": agg,
        "stack": {"X": X_stack, "y": y_stack},
    }
