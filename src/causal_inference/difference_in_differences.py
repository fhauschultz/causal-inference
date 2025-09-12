import re

import linearmodels as lm
import matplotlib.pyplot as plt
import numpy as np
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

    model = lm.PanelOLS(y, X, entity_effects=True, time_effects=True)
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
    treatment_start_col,  # e.g., "_nfd"
    covariates=None,  # optional list of time-varying controls
    reference_period=-1,  # omitted event time
    event_time_window=None,  # e.g., (-5, 5) or None
    alpha=0.10,
    cov_type="clustered",
):
    """
    Minimal Sun & Abraham (2020) stacked event-study with optional covariates.
    - For each cohort g: keep cohort-g and not-yet-treated units.
    - Build cohort-specific event-time dummies, drop the reference period.
    - Stack and estimate PanelOLS with unit & time FE.
    - Aggregate across cohorts using cohort-size weights.
    """
    if covariates is None:
        covariates = []

    df = data.copy()

    # --- cohorts
    cohorts = df[[unit_col, treatment_start_col]].drop_duplicates().dropna(subset=[treatment_start_col]).rename(columns={treatment_start_col: "cohort_start"})
    if cohorts.empty:
        raise ValueError("No treated cohorts found.")

    cohort_sizes = cohorts.groupby("cohort_start")[unit_col].nunique().sort_index()

    stacks = []
    for cohort_start, g_units in cohorts.groupby("cohort_start"):
        cohort_start = int(cohort_start)
        gset = set(g_units[unit_col])

        tmp = df.copy()
        tmp["is_g"] = tmp[unit_col].isin(gset)
        tmp["not_yet_or_never_treated"] = tmp[treatment_start_col].isna() | (tmp[time_col] < tmp[treatment_start_col])
        # then keep cohort-g units or not-yet-treated (plus never-treated) rows
        tmp = tmp.loc[tmp["is_g"] | tmp["not_yet_or_never_treated"]].copy()

        # event-time relative to cohort g
        tmp["event_time"] = (tmp[time_col] - cohort_start).astype(int)
        if event_time_window is not None:
            lo, hi = event_time_window
            tmp = tmp.loc[(tmp["event_time"] >= lo) & (tmp["event_time"] <= hi)]

        # cohort-specific dummies
        d = pd.get_dummies(tmp["event_time"], prefix=f"et_g{cohort_start}", dtype=float)
        ref = f"et_g{cohort_start}_{int(reference_period)}"
        if ref in d.columns:
            d = d.drop(columns=ref)
        if d.shape[1] == 0:
            continue  # nothing estimable for this cohort

        # build X (dummies + optional covariates) and y
        X_parts = [d]
        if covariates:
            cov = tmp[covariates].apply(pd.to_numeric, errors="coerce").astype(float)
            # very light NA policy: drop covariates that are (almost) constant or all-NA
            cov = cov.loc[:, cov.notna().any(axis=0)]
            X_parts.append(cov)

        Xg = pd.concat(X_parts, axis=1)
        yg = tmp[[unit_col, time_col, outcome_col]].copy()

        # stacked unit id for entity FE
        yg["stack_unit"] = yg[unit_col].astype(str) + f"__g{cohort_start}"
        Xg["stack_unit"] = yg["stack_unit"]
        Xg[time_col] = yg[time_col]

        Xg.set_index(["stack_unit", time_col], inplace=True)
        yg.set_index(["stack_unit", time_col], inplace=True)

        # numeric
        Xg = Xg.apply(pd.to_numeric, errors="coerce").astype(float)
        yg[outcome_col] = pd.to_numeric(yg[outcome_col], errors="coerce").astype(float)

        stacks.append((cohort_start, Xg, yg[[outcome_col]], int(cohort_sizes.loc[cohort_start])))

    if not stacks:
        raise ValueError("No estimable cohorts after building event-time dummies.")
    # stack
    X = pd.concat([Xg for _, Xg, _, _ in stacks], axis=0).sort_index()
    y = pd.concat([yg for _, _, yg, _ in stacks], axis=0).sort_index()
    weights = {g: w for g, _, _, w in stacks}

    # --- minimal cleaning ---
    mask = y[outcome_col].notna()
    X, y = X.loc[mask], y.loc[mask]

    et_cols = [c for c in X.columns if c.startswith("et_g")]
    X[et_cols] = X[et_cols].fillna(0.0)

    if X.empty:
        raise ValueError("Design empty after dropping rows with NA outcome/event-time dummies.")

    # simple within-entity mean imputation for covariates (if any remain)
    cov_cols = [c for c in X.columns if c not in et_cols]
    if cov_cols:
        grp_mean = X[cov_cols].groupby(level=0).transform("mean")
        X[cov_cols] = X[cov_cols].fillna(grp_mean).fillna(0.0)

    # drop all-zero / all-one / globally-constant columns
    X = X.loc[:, (X != 0).any(axis=0)]
    X = X.loc[:, ~(X == 1).all(axis=0)]
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] == 0:
        raise ValueError("No regressors left after basic cleaning.")

    # drop regressors with no within-entity variation (absorbed by entity FE)
    demeaned = X.groupby(level=0).transform(lambda c: c - c.groupby(level=0).transform("mean"))
    keep = demeaned.var().replace(0, np.nan).notna()
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        raise ValueError("All regressors absorbed by entity fixed effects.")

    # fit
    model = lm.PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    # extract cohort-specific betas
    params, ses = fit.params, fit.std_errors
    rows = []
    for name, val in params.items():
        if not name.startswith("et_g"):
            continue
        try:
            parts = name.split("_")  # ["et","g{g}","{l}"]
            g_val = int(parts[1][1:])
            l_val = int(parts[2])
        except Exception:
            continue
        rows.append((g_val, l_val, float(val), float(ses.get(name, np.nan))))
    if not rows:
        raise ValueError("No cohort-specific event-time coefficients found.")

    coef = pd.DataFrame(rows, columns=["cohort_start", "event_time", "estimate", "std_error"]).sort_values(["cohort_start", "event_time"])
    z = norm.ppf(1 - alpha / 2)
    coef["ci_lower"] = coef["estimate"] - z * coef["std_error"]
    coef["ci_upper"] = coef["estimate"] + z * coef["std_error"]

    # interaction-weighted aggregation (weights by cohort size)
    w = pd.Series(weights, name="w")
    coef = coef.merge(w, left_on="cohort_start", right_index=True, how="left")
    coef["w_norm"] = coef.groupby("event_time")["w"].transform(lambda s: s / s.sum())

    agg = coef.assign(weighted_est=lambda d: d["w_norm"] * d["estimate"]).groupby("event_time")["weighted_est"].sum().to_frame("estimate").sort_index()
    agg_se = coef.assign(weighted_var=lambda d: (d["w_norm"] ** 2) * (d["std_error"] ** 2)).groupby("event_time")["weighted_var"].sum().pow(0.5)
    agg["std_error"] = agg_se
    agg["ci_lower"] = agg["estimate"] - z * agg["std_error"]
    agg["ci_upper"] = agg["estimate"] + z * agg["std_error"]

    # add reference period at zero (for plotting)
    if reference_period not in agg.index:
        agg.loc[int(reference_period), ["estimate", "std_error", "ci_lower", "ci_upper"]] = [0.0, 0.0, 0.0, 0.0]
        agg = agg.sort_index()

    return {"fit": fit, "cohort_effects": coef.set_index(["cohort_start", "event_time"]).sort_index(), "agg_effects": agg}
