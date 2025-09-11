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
    treatment_start_col,  # <-- pass "_nfd" in your call
    covariates=None,
    reference_period=-1,
    event_time_window=None,  # e.g. (-5, 5)
    alpha=0.10,
    cov_type="clustered",
    min_covariate_non_missing_frac=0.6,  # drop covariate if too sparse
):
    import linearmodels as lm
    import pandas as pd
    from scipy.stats import norm

    if covariates is None:
        covariates = []

    df = data.copy()

    # --- cohorts
    cohorts = df[[unit_col, treatment_start_col]].drop_duplicates().dropna(subset=[treatment_start_col]).rename(columns={treatment_start_col: "cohort_start"})
    if cohorts.empty:
        raise ValueError("No treated cohorts (treatment_start_col all NaN).")

    cohort_sizes = cohorts.groupby("cohort_start")[unit_col].nunique().sort_index()

    stacks = []
    for g, g_units in cohorts.groupby("cohort_start"):
        g = int(g)
        g_units_set = set(g_units[unit_col])

        tmp = df.copy()
        tmp["is_cohort_g"] = tmp[unit_col].isin(g_units_set)
        tmp["_g_i"] = tmp[treatment_start_col]
        tmp["not_yet_treated"] = tmp["_g_i"].isna() | (tmp[time_col] < tmp["_g_i"])

        # keep cohort g or not-yet-treated
        tmp = tmp.loc[tmp["is_cohort_g"] | tmp["not_yet_treated"]].copy()

        # event time
        tmp["event_time"] = (tmp[time_col] - g).astype(int)
        if event_time_window is not None:
            lo, hi = event_time_window
            tmp = tmp.loc[(tmp["event_time"] >= lo) & (tmp["event_time"] <= hi)]

        # cohort-specific dummies
        dums = pd.get_dummies(tmp["event_time"], prefix=f"et_g{g}", dtype=float)
        ref_col = f"et_g{g}_{int(reference_period)}"
        if ref_col in dums.columns:
            dums = dums.drop(columns=ref_col)
        if dums.shape[1] == 0:
            continue

        # optional covariates (numeric)
        X_parts = [dums]
        if covariates:
            cov = tmp[covariates].apply(pd.to_numeric, errors="coerce").astype(float)

            # keep only covariates with enough non-missing mass
            good_covs = [c for c in cov.columns if cov[c].notna().mean() >= min_covariate_non_missing_frac]
            cov = cov[good_covs]
            # attach but we will not force-drop rows because of their NAs
            X_parts.append(cov)

        Xg = pd.concat(X_parts, axis=1)

        yg = tmp[[unit_col, time_col, outcome_col]].copy()
        yg["stack_unit"] = yg[unit_col].astype(str) + f"__g{g}"

        # index
        Xg["stack_unit"] = yg["stack_unit"]
        Xg[time_col] = yg[time_col]
        Xg.set_index(["stack_unit", time_col], inplace=True)
        yg.set_index(["stack_unit", time_col], inplace=True)

        # numeric
        Xg = Xg.apply(pd.to_numeric, errors="coerce").astype(float)
        yg[outcome_col] = pd.to_numeric(yg[outcome_col], errors="coerce").astype(float)

        stacks.append((g, Xg, yg[[outcome_col]], int(cohort_sizes.loc[g])))

    if not stacks:
        raise ValueError("After preprocessing, no cohorts have estimable event-time dummies.")

    # concat
    X_stack = pd.concat([Xg for _, Xg, _, _ in stacks], axis=0).sort_index()
    y_stack = pd.concat([yg for _, _, yg, _ in stacks], axis=0).sort_index()
    weights = {g: w for g, _, _, w in stacks}

    # ---------- NA handling ----------
    # Required fields: y and ALL event-time dummies. Covariates are optional.
    # Identify event-time columns
    et_cols = [c for c in X_stack.columns if c.startswith("et_g")]
    cov_cols = [c for c in X_stack.columns if c not in et_cols]

    # rows must have non-missing y and all event-time dummies (they’re 0/1 already)
    mask = y_stack[outcome_col].notna()
    if et_cols:
        mask &= X_stack[et_cols].notna().all(axis=1)

    # For covariates: keep them, but do NOT force-drop rows; we’ll drop covariate columns
    # that are missing on a given row after this mask via column-wise pruning below.
    X_stack = X_stack.loc[mask]
    y_stack = y_stack.loc[mask]

    # If after this we have no data, bail early with a helpful message
    if X_stack.empty:
        raise ValueError("Design empty after keeping only rows with non-missing outcome and event-time dummies. Check that outcome and event-time dummies are not entirely missing.")

    # Drop covariate columns that are still mostly missing on the kept rows
    if cov_cols:
        cov_keep = []
        for c in cov_cols:
            frac_non_missing = X_stack[c].notna().mean()
            if frac_non_missing >= min_covariate_non_missing_frac:
                cov_keep.append(c)
        # keep et_cols + kept covariates
        X_stack = X_stack[et_cols + cov_keep]
        # For the remaining covariates, fill any residual NAs with within-entity mean (then 0 if all NA)
        if cov_keep:
            for c in cov_keep:
                # within-entity mean imputation (stack_unit level)
                grp_mean = X_stack[c].groupby(level=0).transform("mean")
                X_stack[c] = X_stack[c].fillna(grp_mean)
                X_stack[c] = X_stack[c].fillna(0.0)

    # ---------- Exog screening ----------
    # drop all-NaN columns
    X_stack = X_stack.dropna(axis=1, how="all")
    # drop all-zero & all-one
    nonzero_cols = X_stack.columns[(X_stack != 0).any(axis=0)]
    X_stack = X_stack[nonzero_cols]
    all_one_cols = X_stack.columns[(X_stack == 1).all(axis=0)]
    if len(all_one_cols):
        X_stack = X_stack.drop(columns=all_one_cols)
    # drop globally-constant
    const_cols = X_stack.columns[X_stack.nunique(dropna=False) == 1]
    if len(const_cols):
        X_stack = X_stack.drop(columns=const_cols)
    # drop duplicates
    if X_stack.shape[1] > 1:
        X_stack = X_stack.T.drop_duplicates().T
    if X_stack.shape[1] == 0:
        raise ValueError("No regressors left after removing constant/duplicate columns.")

    # keep only columns with within-entity variation
    demeaned = X_stack.groupby(level=0).transform(lambda c: c - c.groupby(level=0).transform("mean"))
    keep_cols = demeaned.var().replace(0, np.nan).notna()
    X_stack = X_stack.loc[:, keep_cols]
    if X_stack.shape[1] == 0:
        raise ValueError("All regressors absorbed by entity fixed effects (no within-entity variation).")

    # final safety
    all_one_cols = X_stack.columns[(X_stack == 1).all(axis=0)]
    if len(all_one_cols):
        X_stack = X_stack.drop(columns=all_one_cols)
    nonzero_cols = X_stack.columns[(X_stack != 0).any(axis=0)]
    X_stack = X_stack[nonzero_cols]
    if X_stack.shape[1] == 0:
        raise ValueError("No regressors remain after final screening.")

    # ---------- fit ----------
    model = lm.PanelOLS(y_stack, X_stack, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    # ---------- extract & aggregate ----------
    params = fit.params.copy()
    ses = fit.std_errors.copy()
    rows = []
    for name, val in params.items():
        if not name.startswith("et_g"):
            continue
        parts = name.split("_")  # ["et","g{g}","{l}"]
        try:
            g_val = int(parts[1][1:])
            l_val = int(parts[2])
        except Exception:
            continue
        rows.append((g_val, l_val, float(val), float(ses.get(name, np.nan))))
    if not rows:
        raise ValueError("No cohort-specific event-time coefficients found in fit results.")

    coef_df = pd.DataFrame(rows, columns=["cohort_start", "event_time", "estimate", "std_error"]).sort_values(["cohort_start", "event_time"])
    z = norm.ppf(1 - alpha / 2)
    coef_df["ci_lower"] = coef_df["estimate"] - z * coef_df["std_error"]
    coef_df["ci_upper"] = coef_df["estimate"] + z * coef_df["std_error"]

    w_series = pd.Series({g: w for g, _, _, w in stacks}, name="w")
    coef_df = coef_df.merge(w_series, left_on="cohort_start", right_index=True, how="left")
    coef_df["w_norm"] = coef_df.groupby("event_time")["w"].transform(lambda s: s / s.sum())

    agg = coef_df.assign(weighted_est=lambda d: d["w_norm"] * d["estimate"]).groupby("event_time")["weighted_est"].sum().to_frame("estimate").sort_index()
    agg_se = coef_df.assign(weighted_var=lambda d: (d["w_norm"] ** 2) * (d["std_error"] ** 2)).groupby("event_time")["weighted_var"].sum().pow(0.5)
    agg["std_error"] = agg_se
    agg["ci_lower"] = agg["estimate"] - z * agg["std_error"]
    agg["ci_upper"] = agg["estimate"] + z * agg["std_error"]
    if reference_period not in agg.index:
        agg.loc[int(reference_period), ["estimate", "std_error", "ci_lower", "ci_upper"]] = [0.0, 0.0, 0.0, 0.0]
        agg = agg.sort_index()

    return {
        "fit": fit,
        "cohort_effects": coef_df.set_index(["cohort_start", "event_time"]).sort_index(),
        "agg_effects": agg,
        "stack": {"X": X_stack, "y": y_stack},
    }
