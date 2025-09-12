import re
import warnings

import linearmodels as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
            res = estimate_staggered_did_sa(self.data, self.unit_col, self.time_col, self.value_col, "treatment_start", covariates=self.covariates, alpha=alpha, cov_type=cov_type)
            # Store aggregated SA event-study (safe visual)
            self.model_effects[self.value_col] = res[2]
            self.models[self.value_col] = res[0]
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


def estimate_staggered_did_sa(
    data,
    unit_col,
    time_col,
    outcome_col,
    treatment_start_col,
    covariates=None,
    reference_period=-1.0,
    alpha=0.1,
    cov_type="clustered",
):
    """
    Sun & Abraham (2021) interaction-weighted (IW) event study for staggered adoption.
    - No binning
    - Observation-level weights with support restriction (cohort must appear at both ℓ and the anchor)
    - Explicit two-way within (unit & time demeaning) before OLS

    Returns:
        model: dict-like handle {'params','cov','pvalues'}
        df:    input data + helper cols (_Ei, _ell, _Ei_excluded)
        estimates: DataFrame indexed by event-time ℓ with ['estimate','ci_lower','ci_upper'],
                   including the anchor period as 0,0,0.
    """

    df = data.copy()
    # ---------- cohorts & exclusions ----------
    Ei_series = pd.to_numeric(df[treatment_start_col], errors="coerce")  # NaN => never-treated
    has_never = Ei_series.isna().any()
    Ei_helper = Ei_series.fillna(np.inf)  # helper only
    df["_Ei"] = Ei_series

    t_min = df[time_col].min()
    t_max = df[time_col].max()

    finite_E = Ei_helper[np.isfinite(Ei_helper)]
    earliest_treat = finite_E.min() if not finite_E.empty else np.nan
    has_always = np.isfinite(earliest_treat) and (earliest_treat == t_min)

    all_cohorts = np.sort(Ei_helper.unique())
    finite_cohorts = [int(c) for c in all_cohorts if np.isfinite(c)]

    excluded = set()
    trim_tmax = None
    if has_never:
        excluded.add(np.inf)
    else:
        if len(finite_cohorts) == 0:
            raise ValueError("No treated cohorts found.")
        latest = int(np.max(finite_cohorts))
        excluded.add(latest)
        trim_tmax = latest - 1

    if has_always and np.isfinite(earliest_treat):
        excluded.add(int(earliest_treat))

    if trim_tmax is not None:
        df = df.loc[df[time_col] <= trim_tmax].copy()
        t_max = df[time_col].max()

    # ---------- event time ----------
    df["_ell"] = df[time_col] - df["_Ei"]
    df["_ell"] = df["_ell"].where(df["_Ei"].notna(), np.nan)
    req_ref = int(reference_period)

    cohorts_use = [e for e in finite_cohorts if e not in excluded]
    if not cohorts_use:
        raise ValueError("No cohorts remain after exclusions.")

    ell_support = df.loc[df["_Ei"].isin(cohorts_use), "_ell"].dropna().astype(int).unique()
    ell_support = np.sort(ell_support)

    # ---------- validate/choose effective reference ----------
    obs_counts = df.loc[df["_Ei"].isin(cohorts_use), [unit_col, "_Ei", "_ell"]].dropna(subset=["_Ei", "_ell"]).assign(_Ei=lambda d: d["_Ei"].astype(int), _ell=lambda d: d["_ell"].astype(int)).groupby(["_Ei", "_ell"])[unit_col].size().rename("n_obs").reset_index()

    def total_obs_at_l(l):
        tmp = obs_counts.loc[obs_counts["_ell"] == int(l), "n_obs"]
        return int(tmp.sum()) if not tmp.empty else 0

    if total_obs_at_l(req_ref) == 0:
        pre_ls = [int(l) for l in ell_support if l < 0]
        if not pre_ls:
            raise ValueError(f"No observations at requested reference_period ℓ={req_ref}, and no pre-treatment periods available to anchor.")
        eff_ref = max(pre_ls, key=total_obs_at_l)
        warnings.warn(f"reference_period={req_ref} has no support; using ℓ={eff_ref} as the anchor.")
    else:
        eff_ref = req_ref

    # ---------- build interactions (omit eff_ref) ----------
    ell_for_dummies = [int(l) for l in ell_support if int(l) != int(eff_ref)]

    cohort_key = df["_Ei"].astype("Int64")
    cohort_df = pd.get_dummies(cohort_key, prefix="cohort").astype(float)
    cohort_cols = [f"cohort_{e}" for e in cohorts_use]
    cohort_df = cohort_df.reindex(columns=cohort_cols, fill_value=0.0)

    event_key = df["_ell"].astype("Int64")
    event_df = pd.get_dummies(event_key, prefix="ell").astype(float)
    event_cols = [f"ell_{l}" for l in ell_for_dummies if f"ell_{l}" in event_df.columns]
    event_df = event_df.reindex(columns=event_cols, fill_value=0.0)

    inter_cols = []
    inter_mat = []
    for e in cohorts_use:
        e_col = f"cohort_{e}"
        for l in ell_for_dummies:
            l_col = f"ell_{l}"
            if l_col not in event_df.columns:
                continue
            name = f"int_e{e}_l{l}"
            inter_cols.append((e, l, name))
            inter_mat.append(cohort_df[e_col].values * event_df[l_col].values)

    if not inter_mat:
        raise ValueError("No cohort×event interactions after exclusions/reference choice.")

    X_int = pd.DataFrame(
        np.column_stack(inter_mat),
        columns=[n for (_, _, n) in inter_cols],
        index=df.index,
    )

    X = X_int.copy()
    if covariates:
        X = pd.concat([X, df[covariates].astype(float)], axis=1)

    # ---------- set panel index ----------
    y = df[[unit_col, time_col, outcome_col]].set_index([unit_col, time_col]).astype(float)[outcome_col]
    X = X.set_index([df[unit_col], df[time_col]])

    # ---------- prune degenerate columns (pandas 2.x safe) ----------
    if X.shape[1] > 0:
        X = X.loc[:, X.notna().any(axis=0)]
        if X.shape[1] > 0:
            X = X.loc[:, (X != 0).any(axis=0)]

    # ---------- explicit two-way within transform ----------
    # y~
    y_df = y.to_frame("y")
    y_df["unit_mean"] = y_df.groupby(level=0)["y"].transform("mean")
    y_df["time_mean"] = y_df.groupby(level=1)["y"].transform("mean")
    grand_y = y_df["y"].mean()
    y_within = y_df["y"] - y_df["unit_mean"] - y_df["time_mean"] + grand_y

    # X~  (column-wise grand means)
    if X.shape[1] > 0:
        X_unit_mean = X.groupby(level=0).transform("mean")
        X_time_mean = X.groupby(level=1).transform("mean")
        X_grand = X.mean()  # Series: per-column grand mean
        X_within = X - X_unit_mean - X_time_mean + X_grand
        keep = X_within.abs().sum(axis=0) > 0
        X_within = X_within.loc[:, keep]
    else:
        X_within = X

    # ---------- OLS on within-transformed data (no intercept) ----------

    ols = sm.OLS(y_within.values, X_within.values)  # no constant
    fit_res = ols.fit(cov_type="cluster", cov_kwds={"groups": y_within.index.get_level_values(0)})

    params = pd.Series(fit_res.params, index=X_within.columns)
    cov = pd.DataFrame(fit_res.cov_params(), index=X_within.columns, columns=X_within.columns)
    pvals = pd.Series(fit_res.pvalues, index=X_within.columns)

    z = norm.ppf(1 - alpha / 2.0)

    # ---------- obs-level weights with support restriction ----------
    ref_presence = set(obs_counts.loc[obs_counts["_ell"] == int(eff_ref), "_Ei"].tolist())

    rows = []
    for l in np.sort(ell_support):
        l = int(l)
        if l == int(eff_ref):
            rows.append({"event_l": l, "estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0})
            continue

        at_l = obs_counts.loc[obs_counts["_ell"] == l]
        cohorts_at_l = set(at_l["_Ei"].tolist())
        eligible = sorted(list(cohorts_at_l & ref_presence))
        if not eligible:
            rows.append({"event_l": l, "estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue

        n_by_e = {int(r["_Ei"]): int(r["n_obs"]) for _, r in at_l.iterrows() if int(r["_Ei"]) in eligible}
        total = float(sum(n_by_e.values()))
        if total <= 0:
            rows.append({"event_l": l, "estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue
        w_e = {e: n_by_e[e] / total for e in eligible}

        names_l, weights_l = [], []
        for e in eligible:
            name = f"int_e{int(e)}_l{int(l)}"
            if name in params.index:
                names_l.append(name)
                weights_l.append(w_e[e])

        if not names_l:
            rows.append({"event_l": l, "estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue

        weights_l = np.asarray(weights_l, dtype=float)
        weights_l = weights_l / weights_l.sum()

        beta_l = params.loc[names_l].values
        est_l = float(weights_l @ beta_l)

        cov_ll = cov.loc[names_l, names_l].values
        var_l = float(weights_l @ cov_ll @ weights_l)
        var_l = max(var_l, 0.0)
        se_l = np.sqrt(var_l)

        rows.append({"event_l": l, "estimate": est_l, "ci_lower": est_l - z * se_l, "ci_upper": est_l + z * se_l})

    estimates = pd.DataFrame(rows).set_index("event_l").sort_index()

    model = {"params": params, "cov": cov, "pvalues": pvals}
    df["_Ei_excluded"] = Ei_helper.isin(excluded).astype(int)

    return model, df, estimates
