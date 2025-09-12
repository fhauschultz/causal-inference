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
            res = estimate_staggered_did_sa(self.data, self.unit_col, self.time_col, self.value_col, "treatment_start", covariates=self.covariates, event_time_window=event_time_window, alpha=alpha, cov_type=cov_type)
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
    cohort_col,  # treatment start period per unit; NaN => never-treated
    covariates=None,
    ref_p=None,  # default: [-1, .F]  (.F = first relative period)
    ref_c="never",  # "never" => never-treated cohorts as references
    event_time_window=None,  # optional (lo, hi) on relative periods
    alpha=0.05,
    cov_type="clustered",
    use_weights=False,  # if you pass weights in df["_w"], they’ll be used by aggregation
    weight_col="_w",
):
    """
    Python analogue of fixest::sunab():
      - Builds a single (relative period x cohort) interaction design.
      - Drops ref.p (default [-1, .F]) and ref.c = never-treated.
      - Excludes always-treated (treated before sample begins).
      - Fits TWFE with unit/time FE on the full interaction design.
      - Aggregates to ATT by relative period using fixest-style column shares.

    Returns:
      {
        "fit": PanelOLSResults,
        "cohort_period_coefs": DataFrame with all (l, cohort) CATTs,
        "agg_effects": DataFrame ATT_l with fixest-style aggregation
      }
    """
    if covariates is None:
        covariates = []

    df = data.copy()

    # 1) Compute relative period (RP) = time - cohort
    gi = df[cohort_col]
    df["_rp"] = (df[time_col] - gi).astype("Int64")  # Int64 to allow NA

    # Identify sample min/max calendar time for always-treated detection
    t_min = df[time_col].min()

    # 2) Tag cohorts
    # never-treated: cohort is NaN
    is_never = gi.isna()
    # always-treated: treated before sample starts (cohort < t_min)
    # (these have only non-negative RPs in-sample or no pre-periods at all)
    is_always = gi.notna() & (gi < t_min)

    # 3) Optional window on RP (like binning/trimming of relative periods)
    if event_time_window is not None:
        lo, hi = event_time_window
        in_win = df["_rp"].between(lo, hi, inclusive="both")
        # rows outside window remain in sample, they just won’t create dummies
    else:
        in_win = df["_rp"].notna()

    # 4) Reference periods: default [-1, .F]
    # .F = first relative period in the data (min over observed _rp)
    rp_min = int(pd.to_numeric(df["_rp"], errors="coerce").min()) if df["_rp"].notna().any() else -1
    if ref_p is None:
        ref_p = [-1, rp_min]
    ref_p = [int(x) for x in ref_p]

    # 5) Build factor variables for RP and cohort (treated cohorts only)
    # Treated cohorts are those with non-missing gi and NOT always-treated
    treated_mask = gi.notna() & ~is_always
    treated_cohorts = pd.Series(gi.where(treated_mask), name="cohort_fac")
    # never-treated cohorts act as references => no cohort dummies for them

    # RP dummies only for rows inside window and where RP is observed
    rp_for_dummies = df.loc[in_win & df["_rp"].notna(), "_rp"].astype(int)

    # cohort factor values aligned on the same rows:
    cohort_for_dummies = treated_cohorts.loc[rp_for_dummies.index]

    # Drop rows where cohort is NaN (never-treated) from the interaction construction
    valid_rows = cohort_for_dummies.notna()
    rp_vals = rp_for_dummies[valid_rows]
    cohort_vals = cohort_for_dummies[valid_rows].astype(int)

    # 6) Create full interaction matrix: i(RP, cohort)
    # RP as one-hot (excluding ref.p later), cohort as one-hot (excluding ref.c = never)
    rp_dum = pd.get_dummies(rp_vals, prefix="rp", dtype=float)  # rp_-3, rp_-2, rp_0, ...
    # drop reference periods
    for r in set(ref_p):
        col = f"rp_{r}"
        if col in rp_dum.columns:
            rp_dum = rp_dum.drop(columns=col)

    # cohort dummies for treated cohorts only
    coh_dum = pd.get_dummies(cohort_vals, prefix="coh", dtype=float)  # coh_2003, coh_2004, ...
    # (ref.c = never-treated => already excluded since we only created coh dummies on treated rows)

    # Kronecker-style interactions: for each rp col and each coh col
    # Build on valid_rows index then reindex to full df
    inter_list = []
    for rp_col in rp_dum.columns:
        for coh_col in coh_dum.columns:
            name = f"{rp_col}::cohort::{coh_col.split('_', 1)[1]}"
            inter_col = (rp_dum[rp_col] * coh_dum[coh_col]).rename(name)
            inter_list.append(inter_col)

    if not inter_list:
        raise ValueError("No interaction columns created. Check your ref.p or the window—it may be dropping everything.")

    M = pd.concat(inter_list, axis=1)  # design on 'valid_rows' index
    # Reindex M to full df, fill missing with 0 (rows never contribute to these interactions)
    M = M.reindex(df.index).fillna(0.0)

    # 7) Build the full X: interactions + covariates (controls)
    X_parts = [M]
    if covariates:
        cov = df[covariates].apply(pd.to_numeric, errors="coerce").astype(float)
        X_parts.append(cov)

    X = pd.concat(X_parts, axis=1)
    y = df[[unit_col, time_col, outcome_col]].copy()
    if outcome_col not in y:
        y[outcome_col] = df[outcome_col]

    # Always-treated => drop (NA outcome so PanelOLS removes them)
    y.loc[is_always, outcome_col] = np.nan

    # Set panel index
    y = y.set_index([unit_col, time_col])
    X.index = y.index  # align

    # Drop rows with missing outcome
    keep = y[outcome_col].notna()
    X, y = X.loc[keep], y.loc[keep]

    # Drop all-zero / constant columns
    X = X.loc[:, (X != 0).any(axis=0)]
    X = X.loc[:, X.nunique(dropna=False) > 1]

    # 8) Fit TWFE with unit & time FE
    model = lm.PanelOLS(y[outcome_col], X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    fit = model.fit(cov_type=cov_type, cluster_entity=True)

    # 9) Extract CATTs (all (rp, cohort) coefficients)
    params, ses = fit.params, fit.std_errors
    rows = []
    for name, val in params.items():
        # names like: "rp_-3::cohort::2004"
        if "::cohort::" not in name:
            continue
        # parse RP and cohort
        try:
            rp_str, coh_str = name.split("::cohort::")
            l = int(rp_str.split("_", 1)[1])
            g = int(coh_str)
        except Exception:
            continue
        rows.append((l, g, float(val), float(ses.get(name, np.nan)), name))
    if not rows:
        raise ValueError("No (relative period × cohort) coefficients found.")

    catt = pd.DataFrame(rows, columns=["event_time", "cohort_start", "estimate", "std_error", "name"]).sort_values(["event_time", "cohort_start"])

    # 10) Fixest-style aggregation by relative period using model-matrix shares
    # shares = colSums(sign(mm[, v_names])) / sum(colSums(...))
    # build mm only on kept rows, for the interaction columns
    mm = X[catt["name"].tolist()].copy()
    if use_weights and (weight_col in df.columns):
        w = pd.to_numeric(df.loc[mm.index.get_level_values(0), weight_col], errors="coerce").fillna(1.0).values
        # align weights to mm rows (by entity index); if not provided, fallback to 1
        # note: in most cases you won’t pass weights; this branch is optional
    else:
        w = np.ones(len(mm), dtype=float)

    agg_rows = []
    z = norm.ppf(1 - alpha / 2)
    for l, grp in catt.groupby("event_time"):
        cols = grp["name"].tolist()
        sub = mm[cols]

        # column shares (like fixest: using sign of entries; entries are 0/1 anyway)
        col_sums = (np.abs(sub.values) * w[:, None]).sum(axis=0) if use_weights else (np.abs(sub.values)).sum(axis=0)
        if col_sums.sum() == 0:
            continue
        shares = col_sums / col_sums.sum()

        # aggregate coef
        beta = np.dot(shares, grp["estimate"].values)

        # aggregate variance: s'V s over those columns
        V = fit.cov.loc[cols, cols].values
        var = shares @ V @ shares
        se = float(np.sqrt(max(var, 0.0)))

        agg_rows.append((int(l), float(beta), se, float(beta - z * se), float(beta + z * se)))

    agg = pd.DataFrame(agg_rows, columns=["event_time", "estimate", "std_error", "ci_lower", "ci_upper"]).sort_values("event_time").set_index("event_time")

    # ensure the reference appears at 0 for plotting continuity
    for r in ref_p:
        if r not in agg.index:
            agg.loc[int(r), ["estimate", "std_error", "ci_lower", "ci_upper"]] = [0.0, 0.0, 0.0, 0.0]
    agg = agg.sort_index()

    return {
        "fit": fit,
        "cohort_period_coefs": catt.set_index(["event_time", "cohort_start"]).sort_index(),
        "agg_effects": agg,
    }
