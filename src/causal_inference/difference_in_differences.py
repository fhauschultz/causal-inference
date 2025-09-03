import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

from causal_inference.utils import BaseCausalInference


class StaggeredDID(BaseCausalInference):
    def fit(self, significance_level=0.1):
        self.models = {}
        self.model_effects = {}
        self.significance_level = significance_level
        model, df = estimate_staggered_did(self.data, self.unit_col, self.time_col, self.value_col, "treatment_start", covariates=self.covariates)
        table = extract_staggered_treatment_effect(model, alpha=significance_level, cov_type=self.cov_type)
        self.model_effects[self.value_col] = table
        self.models[self.value_col] = self.model
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


def estimate_staggered_did(data, unit_col, time_col, outcome_col, treatment_start_col, covariates=None):
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
    # Indicator for post-treatment period (0 if never treated or pre-treatment)
    df["treated"] = (df["periods_since_treatment"] >= 0) & (~df[treatment_start_col].isna())
    df["treated"] = df["treated"].astype(int)

    # Optionally, create event-time dummies for dynamic effects
    event_time_dummies = pd.get_dummies(df["periods_since_treatment"], prefix="event_time", dtype=float, drop_first=True)
    # Only keep dummies for post-treatment periods (e.g., 0, 1, 2, ...)
    post_event_dummies = event_time_dummies.loc[:, event_time_dummies.columns.str.contains("event_time_")]

    # Build regression matrix
    X = pd.get_dummies(df[unit_col], prefix="unit", drop_first=True, dtype=float)
    X = X.join(pd.get_dummies(df[time_col], prefix="time", drop_first=True, dtype=float))
    X = X.join(post_event_dummies)
    if covariates:
        X = X.join(df[covariates].astype(float))
    X = sm.add_constant(X)
    y = df[outcome_col].astype(float)

    model = sm.OLS(y, X).fit()
    return model, df


def conf_int_robust(model, alpha=0.05, cov_type="nonrobust", **cov_kwds):
    """
    Compute confidence intervals using specified covariance type, including heteroscedasticity-robust standard errors.

    Parameters:
    - model: A fitted statsmodels regression model (e.g., OLS).
    - alpha: Significance level for the confidence interval (default is 0.05 for 95% CI).
    - cov_type: The covariance type (e.g., 'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3').
    - cov_kwds: Additional arguments for robust covariance calculation.

    Returns:
    - DataFrame containing the lower and upper bounds of the confidence intervals.
    """
    # Handle non-robust covariance explicitly
    if cov_type == "nonrobust":
        robust_cov = model.cov_params()
    else:
        robust_cov = model.get_robustcov_results(cov_type=cov_type, **cov_kwds).cov_params()

    # Extract parameter estimates and standard errors.
    params = model.params
    robust_se = np.sqrt(np.diag(robust_cov))

    # Compute critical t-value based on alpha and degrees of freedom
    df_resid = model.df_resid
    t_value = t.ppf(1 - alpha / 2, df_resid)

    # Compute confidence intervals
    ci_lower = params - t_value * robust_se
    ci_upper = params + t_value * robust_se

    return pd.DataFrame({"lower": ci_lower, "upper": ci_upper})


def extract_staggered_treatment_effect(model, alpha=0.05, cov_type="nonrobust", **cov_kwds):
    """
    Extract estimated treatment effects and confidence intervals for staggered DiD.
    Returns a DataFrame with event-time effects and confidence bands.
    Allows toggling robust covariance type via cov_type.
    """

    # Only keep coefficients for event_time dummies (dynamic treatment effects)
    effect_params = {k: v for k, v in model.params.items() if k.startswith("event_time_")}
    # Compute robust confidence intervals
    ci = conf_int_robust(model, alpha=alpha, cov_type=cov_type, **cov_kwds)
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
    return df.sort_index()
