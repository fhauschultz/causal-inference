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
    SAME signature / return as original: (model, df, estimates)

    Args:
        data: DataFrame with panel data.
        unit_col: Column name for unit identifier.
        time_col: Column name for (integer-like) time.
        outcome_col: Column name for outcome variable.
        treatment_start_col: First-treatment period Ei (NaN for never-treated).
        covariates: Optional list of additive covariates (not interacted).
        reference_period: event-time (e.g., -1) omitted as baseline.
        alpha: two-sided CI level (used for δ_{e,ℓ} CIs).
        cov_type: PanelOLS covariance type, e.g. "clustered" (clustered at entity).

    Returns:
        model: PanelOLS model object (handle).
        df: DataFrame with helper columns (_Ei, _ell, _Ei_excluded).
        estimates: dict with:
            - delta_table: DataFrame of δ_{e,ℓ} with SEs/CIs
            - weights: Pr{Ei=e | Ei∈[-ℓ, T-ℓ]} per ℓ
            - iw_by_l: Series of ∑_e δ_{e,ℓ} * weight_{e|ℓ}
            - iw_g: float, average IW over post periods (ℓ ≥ 0)
            - g: list of event-times used for iw_g
            - agg_effects: dict with 'per_event' DataFrame (event_l, iw) and 'overall' float
            - meta: bookkeeping info
            - fit: fitted results object
    """
    import pandas as pd
    from scipy.stats import norm

    # Import PanelOLS either via linearmodels or the submodule
    try:
        import linearmodels as lm

        PanelOLS = lm.PanelOLS
    except Exception:
        from linearmodels.panel import PanelOLS

    df = data.copy()

    # ---------- Cohorts (Ei) and panel info ----------
    # Keep never-treated as NA (NOT inf) in the column we will dummy-code
    Ei_series = pd.to_numeric(df[treatment_start_col], errors="coerce")
    has_never = Ei_series.isna().any()

    # Helper with inf only for logic (never in dummies)
    Ei_helper = Ei_series.fillna(np.inf)
    df["_Ei"] = Ei_series  # finite ints/floats or NA only

    t_min = df[time_col].min()
    t_max = df[time_col].max()

    finite_E = Ei_helper[np.isfinite(Ei_helper)]
    earliest_treat = finite_E.min() if not finite_E.empty else np.nan
    has_always = np.isfinite(earliest_treat) and (earliest_treat == t_min)

    all_cohorts_helper = np.sort(Ei_helper.unique())
    finite_cohorts = [int(c) for c in all_cohorts_helper if np.isfinite(c)]

    # ---------- Exclusion / trimming per Sun–Abraham ----------
    excluded_cohorts = set()
    trim_tmax = None

    if has_never:
        excluded_cohorts.add(np.inf)  # conceptually exclude ∞ cohort from interactions
    else:
        if len(finite_cohorts) == 0:
            raise ValueError("No treated cohorts found.")
        latest = int(np.max(finite_cohorts))
        excluded_cohorts.add(latest)
        trim_tmax = latest - 1  # restrict to t <= max{Ei}-1

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

    # ---------- Build 1{Ei=e} and 1{ℓ=ℓ} ----------
    # Use Pandas nullable Int64 for keys (allows NA, but we avoided inf entirely)
    cohort_key = df["_Ei"].astype("Int64")
    cohort_df = pd.get_dummies(cohort_key, prefix="cohort").astype(float)
    cohort_cols = [f"cohort_{e}" for e in cohorts_for_interaction]
    cohort_df = cohort_df.reindex(columns=cohort_cols, fill_value=0.0)

    event_key = df["_ell"].astype("Int64")
    event_df = pd.get_dummies(event_key, prefix="ell").astype(float)
    event_cols = [f"ell_{l}" for l in ell_for_dummies if f"ell_{l}" in event_df.columns]
    event_df = event_df.reindex(columns=event_cols, fill_value=0.0)

    # Interactions: 1{Ei=e} * 1{ℓ=l}
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

    # Additive (non-interacted) covariates
    X = X_int.copy()
    if covariates:
        X = pd.concat([X, df[covariates].astype(float)], axis=1)

    # Outcome and index
    y = df[[unit_col, time_col, outcome_col]].set_index([unit_col, time_col]).astype(float)[outcome_col]
    X = X.set_index([df[unit_col], df[time_col]])

    # ---------- Prune degenerate columns (helps rank) ----------
    # Drop all-NA or all-zero columns
    if X.shape[1] > 0:
        non_na = X.columns[X.notna().any(axis=0)]
        X = X[non_na]
        if X.shape[1] > 0:
            nonzero = X.columns[(X != 0).any(axis=0)]
            X = X[nonzero]

    # ---------- Fit TWFE with safeguards against collinearity ----------
    model = PanelOLS(
        y,
        X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,  # drop regressors absorbed by FE
        check_rank=False,  # allow estimation even if design isn't full rank
    )
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    # ---------- Step 1: δ_{e,ℓ} with SEs/CIs ----------
    params = fit.params
    bse = fit.std_errors
    pvals = fit.pvalues

    z = norm.ppf(1 - alpha / 2.0)
    rows = []
    for e_val, l_val, name in inter_cols:
        if name in params.index:
            est = float(params[name])
            se = float(bse[name])
            rows.append(
                {
                    "cohort_e": int(e_val),
                    "event_l": int(l_val),
                    "param": name,
                    "delta_hat": est,
                    "se": se,
                    "ci_low": est - z * se,
                    "ci_high": est + z * se,
                    "pvalue": float(pvals[name]),
                }
            )
    delta_table = pd.DataFrame(rows).sort_values(["event_l", "cohort_e"]).reset_index(drop=True)

    # ---------- Step 2: Weights Pr{Ei=e | Ei∈[-ℓ, T-ℓ]} ----------
    t0_shift = int(t_min)
    T_for_weights = int(t_max - t0_shift)

    def eligible_cohorts_for_l(l):
        a = -l
        b = T_for_weights - l
        lower = a + t0_shift
        upper = b + t0_shift
        return [e for e in cohorts_for_interaction if (e >= lower and e <= upper)]

    weight_rows = []
    for l in sorted(set(int(x) for x in ell_for_dummies)):
        elig = eligible_cohorts_for_l(l)
        if len(elig) == 0:
            continue
        # Equal mass over eligible cohorts (replace with cohort-size shares if desired)
        w = {e: 1.0 / len(elig) for e in elig}
        for e, w_e in w.items():
            weight_rows.append({"event_l": int(l), "cohort_e": int(e), "weight": float(w_e)})
    weights = pd.DataFrame(weight_rows)
    if weights.empty:
        weights = pd.DataFrame(
            [(int(l), int(e), 1.0 / max(1, len(cohorts_for_interaction))) for l in ell_for_dummies for e in cohorts_for_interaction],
            columns=["event_l", "cohort_e", "weight"],
        )

    # ---------- Step 3: IW aggregation ----------
    merged = pd.merge(delta_table, weights, on=["event_l", "cohort_e"], how="inner")
    iw_by_l = merged.groupby("event_l").apply(lambda d: float(np.sum(d["delta_hat"] * d["weight"])))
    iw_by_l = iw_by_l.sort_index()

    # Default g: all post-treatment event times observed (ℓ ≥ 0)
    g = [int(l) for l in iw_by_l.index if l >= 0]
    iw_g = float(np.mean([iw_by_l[l] for l in g])) if len(g) > 0 else np.nan

    # Helper flags on df
    df["_Ei_excluded"] = Ei_helper.isin(excluded_cohorts).astype(int)

    # For convenience with downstream plotting, also provide a minimal "agg_effects"
    agg_effects = {
        "per_event": pd.DataFrame({"event_l": list(iw_by_l.index), "iw": list(iw_by_l.values)}),
        "overall": iw_g,
    }

    estimates = {
        "delta_table": delta_table,
        "weights": weights.sort_values(["event_l", "cohort_e"]),
        "iw_by_l": iw_by_l,
        "iw_g": iw_g,
        "g": g,
        "agg_effects": agg_effects,  # lightweight, plot-friendly
        "meta": {
            "excluded_cohorts": sorted(
                [int(c) if np.isfinite(c) else np.inf for c in excluded_cohorts],
                key=lambda x: (x is not np.inf, x),
            ),
            "time_trimmed_at": int(trim_tmax) if trim_tmax is not None else None,
            "reference_period": int(reference_period),
            "covariates": list(covariates) if covariates else [],
            "notes": ("Never-treated kept as NA for dummies (no inf) and excluded from interactions; if no never-treated, latest cohort excluded and time trimmed to t ≤ max{Ei}-1; always-treated excluded if present. PanelOLS uses drop_absorbed=True and check_rank=False."),
        },
        "fit": fit,
    }

    return model, df, estimates
