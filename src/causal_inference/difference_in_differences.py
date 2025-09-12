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
    Sun & Abraham (2021) interaction-weighted (IW) DiD for staggered adoption.

    Returns:
        model: PanelOLS model handle
        df:    input data plus helper cols (_Ei, _ell, _Ei_excluded)
        estimates: DataFrame indexed by event-time (ℓ), with columns:
                   'estimate', 'ci_lower', 'ci_upper'
                   where each row is the IW-aggregated CATTe,ℓ across cohorts.
    """
    import pandas as pd
    from scipy.stats import norm

    # Import PanelOLS
    try:
        import linearmodels as lm

        PanelOLS = lm.PanelOLS
    except Exception:
        from linearmodels.panel import PanelOLS

    df = data.copy()

    # ---------- Cohorts (Ei) & panel info ----------
    Ei_series = pd.to_numeric(df[treatment_start_col], errors="coerce")  # NaN => never-treated
    has_never = Ei_series.isna().any()

    # Helper with inf for logic (never used in dummies)
    Ei_helper = Ei_series.fillna(np.inf)
    df["_Ei"] = Ei_series  # keep NA for never-treated (avoids inf→Int64 casting issues)

    t_min = df[time_col].min()
    t_max = df[time_col].max()

    finite_E = Ei_helper[np.isfinite(Ei_helper)]
    earliest_treat = finite_E.min() if not finite_E.empty else np.nan
    has_always = np.isfinite(earliest_treat) and (earliest_treat == t_min)

    all_cohorts_helper = np.sort(Ei_helper.unique())
    finite_cohorts = [int(c) for c in all_cohorts_helper if np.isfinite(c)]

    # ---------- Exclusion / trimming per SA ----------
    excluded_cohorts = set()
    trim_tmax = None

    if has_never:
        excluded_cohorts.add(np.inf)  # exclude never-treated cohort from interactions
    else:
        if len(finite_cohorts) == 0:
            raise ValueError("No treated cohorts found.")
        latest = int(np.max(finite_cohorts))
        excluded_cohorts.add(latest)
        trim_tmax = latest - 1  # use t <= max{Ei} - 1

    if has_always and np.isfinite(earliest_treat):
        excluded_cohorts.add(int(earliest_treat))

    if trim_tmax is not None:
        df = df.loc[df[time_col] <= trim_tmax].copy()
        t_max = df[time_col].max()

    # ---------- Event time ℓ = t - Ei (NaN for never-treated) ----------
    df["_ell"] = df[time_col] - df["_Ei"]
    df["_ell"] = df["_ell"].where(df["_Ei"].notna(), np.nan)

    ref_l = int(reference_period)

    # Cohorts to interact (finite, not excluded)
    cohorts_for_interaction = [e for e in finite_cohorts if e not in excluded_cohorts]
    if len(cohorts_for_interaction) == 0:
        raise ValueError("No cohorts remain after exclusions.")

    # Event-time support among included cohorts
    ell_support = df.loc[df["_Ei"].isin(cohorts_for_interaction), "_ell"].dropna().astype(int).unique()
    ell_support = np.sort(ell_support)
    ell_for_dummies = [int(l) for l in ell_support if l != ref_l]

    # ---------- Dummies ----------
    cohort_key = df["_Ei"].astype("Int64")  # allows NA
    cohort_df = pd.get_dummies(cohort_key, prefix="cohort").astype(float)
    cohort_cols = [f"cohort_{e}" for e in cohorts_for_interaction]
    cohort_df = cohort_df.reindex(columns=cohort_cols, fill_value=0.0)

    event_key = df["_ell"].astype("Int64")
    event_df = pd.get_dummies(event_key, prefix="ell").astype(float)
    event_cols = [f"ell_{l}" for l in ell_for_dummies if f"ell_{l}" in event_df.columns]
    event_df = event_df.reindex(columns=event_cols, fill_value=0.0)

    # ---------- Interactions: 1{Ei=e} * 1{ℓ=l} ----------
    inter_cols = []
    inter_mat = []
    for e in cohorts_for_interaction:
        e_col = f"cohort_{e}"
        for l in ell_for_dummies:
            l_col = f"ell_{l}"
            if l_col not in event_df.columns:
                continue
            name = f"int_e{e}_l{l}"
            inter_cols.append((e, l, name))
            inter_mat.append(cohort_df[e_col].values * event_df[l_col].values)

    if not inter_mat:
        raise ValueError("No cohort×event-time interactions after exclusions / reference_period choice.")

    X_int = pd.DataFrame(
        np.column_stack(inter_mat),
        columns=[n for (_, _, n) in inter_cols],
        index=df.index,
    )

    # Additive covariates
    X = X_int.copy()
    if covariates:
        X = pd.concat([X, df[covariates].astype(float)], axis=1)

    # Outcome and panel index
    y = df[[unit_col, time_col, outcome_col]].set_index([unit_col, time_col]).astype(float)[outcome_col]
    X = X.set_index([df[unit_col], df[time_col]])

    # Prune degenerate columns
    if X.shape[1] > 0:
        non_na = X.columns[X.notna().any(axis=0)]
        X = X[non_na]
        if X.shape[1] > 0:
            nonzero = X.columns[(X != 0).any(axis=0)]
            X = X[nonzero]

    # ---------- Fit TWFE (handles collinearity) ----------
    model = PanelOLS(
        y,
        X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,  # drop regressors absorbed by FE
        check_rank=False,  # allow estimation even if design isn't full rank
    )
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    # ---------- Extract parameters & covariance ----------
    params = fit.params
    cov = fit.cov
    z = norm.ppf(1 - alpha / 2.0)

    # ---------- Step 2: weights = cohort shares among eligible cohorts ----------
    # Compute cohort sizes using unique units per cohort
    unit_cohort = df[[unit_col, "_Ei"]].drop_duplicates(subset=[unit_col]).dropna(subset=["_Ei"])
    unit_cohort["_Ei"] = unit_cohort["_Ei"].astype(int)
    n_units_by_cohort = unit_cohort.groupby("_Ei")[unit_col].nunique().to_dict()

    t0_shift = int(t_min)
    T_for_weights = int(t_max - t0_shift)

    def eligible_cohorts_for_l(l: int):
        a = -l
        b = T_for_weights - l
        lower = a + t0_shift
        upper = b + t0_shift
        return [e for e in cohorts_for_interaction if (e >= lower and e <= upper)]

    # ---------- Build per-ℓ IW estimates + CIs ----------
    rows = []
    for l in sorted(ell_for_dummies):
        elig = eligible_cohorts_for_l(l)
        if len(elig) == 0:
            rows.append({"event_l": l, "estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue

        # Raw weights by cohort sample shares among eligible cohorts
        counts = np.array([n_units_by_cohort.get(int(e), 0) for e in elig], dtype=float)
        if counts.sum() <= 0:
            # fallback to equal weights if counts missing
            counts = np.ones(len(elig), dtype=float)
        w_raw = counts / counts.sum()

        # Keep only parameters actually estimated (not dropped/absorbed)
        names_l = []
        w_kept = []
        for e, w in zip(elig, w_raw, strict=False):
            name = f"int_e{int(e)}_l{int(l)}"
            if name in params.index:
                names_l.append(name)
                w_kept.append(w)

        if len(names_l) == 0:
            rows.append({"event_l": l, "estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue

        # Renormalize weights over kept parameters
        w_kept = np.asarray(w_kept, dtype=float)
        w_kept = w_kept / w_kept.sum()

        betas_l = params.loc[names_l].values
        est_l = float(np.dot(w_kept, betas_l))

        # Delta-method variance: w' Cov w
        cov_ll = cov.loc[names_l, names_l].values
        var_l = float(w_kept @ cov_ll @ w_kept)
        var_l = max(var_l, 0.0)  # guard against tiny negative due to numerics
        se_l = np.sqrt(var_l)
        ci_low = est_l - z * se_l
        ci_high = est_l + z * se_l

        rows.append({"event_l": l, "estimate": est_l, "ci_lower": ci_low, "ci_upper": ci_high})

    estimates = pd.DataFrame(rows).set_index("event_l").sort_index()

    # Helper flag column
    df["_Ei_excluded"] = Ei_helper.isin(excluded_cohorts).astype(int)

    return model, df, estimates
