import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


class BaseDifferenceInDifferences:
    def __init__(self, data, group_cols, time_col, treatment_col, value_vars, experiment_start_date, experiment_end_date=None, covariates=None, confidence_level=0.1, cov_type="HC3", sklearn_model=None):
        self.data = data
        self.group_cols = group_cols
        self.time_col = time_col
        self.treatment_col = treatment_col
        self.value_vars = value_vars
        self.confidence_level = confidence_level
        self.cov_type = cov_type
        self.sklearn_model = sklearn_model
        self.covariates = covariates

        if self.sklearn_model is not None and hasattr(self.sklearn_model, "fit_intercept"):
            self.sklearn_model.fit_intercept = False

        start_date_is_int = check_date_format_consistency(data, time_col, experiment_start_date, experiment_end_date)

        if start_date_is_int:
            self.data["days_since_experiment_start"] = self.data[time_col] - experiment_start_date
            self.experiment_duration_days = experiment_end_date - experiment_start_date if experiment_end_date else None
        else:
            start_date = pd.to_datetime(experiment_start_date)
            end_date = pd.to_datetime(experiment_end_date) if experiment_end_date else self.data[time_col].max()
            self.data["days_since_experiment_start"] = (pd.to_datetime(self.data[time_col]) - start_date).dt.days
            self.experiment_duration_days = (end_date - start_date).days if end_date else None

        plt.style.use("classic")


class EventStudy(BaseDifferenceInDifferences):
    def fit(self):
        self.models = {}
        self.model_effects = {}
        for var in self.value_vars:
            x, y = event_study_data(self.data, "days_since_experiment_start", self.treatment_col, var, self.covariates)
            if not self.sklearn_model:
                self.model = sm.OLS(y, x).fit()
                table = extract_all_coefficients_statsmodels(self.model, cov_type=self.cov_type, alpha=self.confidence_level)
            else:
                self.model = self.sklearn_model.fit(x, y)
                table = extract_coefficients_sklearn(self.model)

            self.model_effects[var] = add_treatment_group(table)
            self.models[var] = self.model
        return self

    def plot_treatment_control(self, variables=None, linewidth=2.5):
        variables = _set_variables(variables, self.value_vars)
        fig, ax = _make_fig(variables)
        fig.suptitle("Treatment and Control Development", fontsize=14)
        confidence_interval_size = int(np.round((1 - self.confidence_level) * 100, 0))
        for i, variable in enumerate(variables):
            plotdata = self.model_effects[variable]
            ax[i].axvline(0, label="Experiment Start", c="black", lw=linewidth)
            ax[i].axvline(self.experiment_duration_days, label="Experiment End", c="black", linestyle="--", lw=linewidth)
            ax[i].plot(plotdata["treatment_group"], label="Treatment Group", lw=linewidth)
            ax[i].plot(plotdata["control_group"], label="Control Group", lw=linewidth)
            if "treatment_group_lower_bound" in plotdata.columns:
                ax[i].fill_between(plotdata.index, plotdata["treatment_group_lower_bound"], plotdata["treatment_group_upper_bound"], alpha=0.2, label=f"{confidence_interval_size}% Confidence Interval")
            if "control_group_lower_bound" in plotdata.columns:
                ax[i].fill_between(plotdata.index, plotdata["control_group_lower_bound"], plotdata["control_group_upper_bound"], alpha=0.2, label=f"{confidence_interval_size}% Confidence Interval")

            ax[i].set_xlabel("Periods Since Experiment")
            ax[i].set_ylabel(variable)
            if i == 0:
                fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=2)
        return fig

    def plot_treatment_effects(self, variables=None, linewidth=2.5):
        variables = _set_variables(variables, self.value_vars)
        fig, ax = _make_fig(variables)
        confidence_interval_size = int(np.round((1 - self.confidence_level) * 100, 0))
        fig.suptitle("Estimated Period-Specific Treatment Effects", fontsize=14)
        for i, variable in enumerate(variables):
            plotdata = self.model_effects[variable]
            ax[i].axvline(0, label="Experiment Start", c="black", lw=linewidth)
            ax[i].axvline(self.experiment_duration_days, label="Experiment End", c="black", linestyle="--", lw=linewidth)
            ax[i].plot(plotdata["treatment_effect"], label="Treatment Effect", lw=linewidth)
            if "treatment_effect_lower_bound" in plotdata.columns:
                ax[i].fill_between(plotdata.index, plotdata["treatment_effect_lower_bound"], plotdata["treatment_effect_upper_bound"], alpha=0.2, label=f"{confidence_interval_size}% Confidence Interval")
            ax[i].axhline(0, c="gray", linestyle="--", lw=linewidth)
            ax[i].set_xlabel("Periods Since Experiment")

            ax[i].set_ylabel(variable)
            if i == 0:
                fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=False, ncol=2)
        return fig


class BinaryDiD(BaseDifferenceInDifferences):
    def fit(self):
        self.models = {}
        self.model_effects = {}
        for var in self.value_vars:
            table, model = cumulative_treatment_effects(
                self.data, self.group_cols, "days_since_experiment_start", self.treatment_col, treatment_start_date=0, treatment_end_date=self.experiment_duration_days, covariates=self.covariates, outcome_col=var, alpha=self.confidence_level, cov_type=self.cov_type
            )
            self.models[var] = model
            self.model_effects[var] = table
        return self

    def plot_treatment_effect(self, variables=None, linewidth=2.5):
        variables = _set_variables(variables, self.value_vars)
        fig, ax = _make_fig(variables)
        fig.suptitle("Estimated Two-way Fixed Effects Treatment Effect", fontsize=14)
        confidence_interval_size = int(np.round((1 - self.confidence_level) * 100, 0))
        for i, variable in enumerate(variables):
            ax[i].axvline(0, label="Experiment Start", c="black", lw=linewidth)
            ax[i].axvline(self.experiment_duration_days, label="Experiment End", c="black", lw=linewidth, linestyle="--")
            ax[i].axhline(0, c="gray", lw=linewidth, linestyle="--", alpha=0.5)
            ax[i].plot(self.model_effects[variable].treatment_effect_cumulative, lw=linewidth)
            ax[i].fill_between(self.model_effects[variable].index, self.model_effects[variable].treatment_effect_cumulative_lower_bound, self.model_effects[variable].treatment_effect_cumulative_upper_bound, alpha=0.2, label=f"{confidence_interval_size}% Confidence Interval")
            ax[i].set_xlabel("Periods Since Experiment")
            ax[i].set_ylabel(variable)
            if i == 0:
                fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.01), fancybox=True, shadow=False, ncol=2)
        return fig


def _set_variables(variables, value_vars):
    if isinstance(variables, str):
        variables = [variables]
    if not variables:
        variables = value_vars
    return variables


def _make_fig(variables):
    subplot_h, subplot_w = 4, 4
    n_vars = len(variables)
    fig, ax = plt.subplots(len(variables), 1, figsize=(subplot_w, n_vars * subplot_h), gridspec_kw={"hspace": 0.2})
    if len(variables) == 1:
        ax = [ax]  # Ensures it's always iterable
    return fig, ax


def package_data_fixed_effects(data, treatment_col, time_col, outcome_col=None, covariates=None):
    cov_data = None
    if covariates:
        cov_data = data[covariates]
    treatment_dummy = pd.get_dummies(data[treatment_col], prefix="treatment", drop_first=True)
    time_dummies = pd.get_dummies(data[time_col], prefix="time")
    post_treatment_interaction = pd.get_dummies(data["post_treatment_interaction"], prefix="time_treatment_effect", drop_first=True)
    x = pd.concat([time_dummies, treatment_dummy, post_treatment_interaction, cov_data], axis=1)
    if outcome_col:
        y = data[outcome_col]
    return x, y


def bdid_did_data(data, unit_col, time_col, treatment_col, treatment_start_date, treatment_end_date, covariates=None, outcome_col=None):
    y = None
    # Create time dummies and identify post-treatment periods
    data["post_treatment_interaction"] = (data[time_col] <= treatment_end_date).astype(int) * (data[time_col] >= treatment_start_date).astype(int) * data[treatment_col]
    x, y = package_data_fixed_effects(data, treatment_col, time_col, outcome_col, covariates)
    return x, y


def event_study_data(data, time_col, treatment_col, outcome_col, covariates=None):
    # Create time dummies and identify post-treatment periods
    time_dummies = pd.get_dummies(data[time_col], prefix="time")
    post_treatment_interaction = time_dummies.mul(data[treatment_col], axis=0)

    # Rename post_treatment_interaction columns
    post_treatment_interaction.columns = [col.replace("time", "time_treatment_effect") for col in post_treatment_interaction.columns]
    cov_data = None

    if covariates:
        cov_data = data[covariates]

    x = pd.concat([time_dummies, post_treatment_interaction, cov_data], axis=1)
    if outcome_col:
        y = data[outcome_col]
    return x, y


def cumulative_treatment_effects(data, unit_col, time_col, treatment_col, treatment_start_date, treatment_end_date, covariates=None, outcome_col=None, alpha=0.05, cov_type="nonrobust"):
    cumulative_effects = []
    max_period = data.days_since_experiment_start.max() + 1
    print("Estimating cumulative treatment effects for periods 1 to", max_period, "...")
    for period in np.arange(1, max_period):
        estimation_data = data[data.days_since_experiment_start <= period].copy(deep=False)
        x, y = bdid_did_data(estimation_data, unit_col, time_col, treatment_col, treatment_start_date, treatment_end_date, outcome_col=outcome_col, covariates=covariates)
        model = sm.OLS(y, x).fit()
        treatment_effect_data = extract_all_coefficients_statsmodels(model, alpha=alpha, cov_type=cov_type).dropna()[["treatment_effect", "treatment_effect_lower_bound", "treatment_effect_upper_bound"]]
        treatment_effect_data["days_since_experiment_start"] = period
        cumulative_effects.append(treatment_effect_data)

    cumulative_effects = pd.concat(cumulative_effects).set_index("days_since_experiment_start")
    cumulative_effects = cumulative_effects.rename(columns={"treatment_effect": "treatment_effect_cumulative", "treatment_effect_lower_bound": "treatment_effect_cumulative_lower_bound", "treatment_effect_upper_bound": "treatment_effect_cumulative_upper_bound"})
    return cumulative_effects, model


def add_treatment_group(data):
    data["treatment_group"] = data["control_group"] + data["treatment_effect"]
    data["treatment_group_lower_bound"] = data["control_group"] + data["treatment_effect_lower_bound"]
    data["treatment_group_upper_bound"] = data["control_group"] + data["treatment_effect_upper_bound"]
    return data


# Function to extract coefficients


def extract_coefficients_statsmodels(coefficients):
    out_names = ["treatment_effect", "control_group"]
    time_coefficients = coefficients.filter(like="time_")  # .dropna()
    treatment_group = time_coefficients.loc[time_coefficients.index.str.contains("treatment_effect")]
    control_group = time_coefficients.loc[~time_coefficients.index.str.contains("treatment_effect")]
    treatment_group.index = [item.replace("_treatment_effect", "") for item in treatment_group.index]
    df = pd.concat([treatment_group, control_group], axis=1, keys=out_names, join="outer")
    df.index = extract_numbers(df.index)
    return df


def extract_all_coefficients_statsmodels(model, alpha=0.05, cov_type="nonrobust"):
    confidence_bands = conf_int_robust(model, alpha=alpha, cov_type=cov_type)
    estimates = extract_coefficients_statsmodels(model.params)
    upper_confidence_bands = extract_coefficients_statsmodels(confidence_bands["upper"])
    lower_confidence_bands = extract_coefficients_statsmodels(confidence_bands["lower"])

    upper_confidence_bands.columns = [a + "_upper_bound" for a in upper_confidence_bands.columns]
    lower_confidence_bands.columns = [a + "_lower_bound" for a in lower_confidence_bands.columns]

    return (estimates.join(upper_confidence_bands, how="left").join(lower_confidence_bands, how="left")).sort_index()


# Function to extract coefficients from sklearn
def extract_coefficients_sklearn(model):
    feature_names = model.feature_names_in_
    coef_series = pd.Series(model.coef_, index=feature_names)
    time_coefficients = coef_series.filter(like="time_").dropna()
    treatment_group = time_coefficients.loc[time_coefficients.index.str.contains("treatment_effect")]
    control_group = time_coefficients.loc[~time_coefficients.index.str.contains("treatment_effect")]
    treatment_group.index = [item.replace("_treatment_effect", "") for item in treatment_group.index]
    df = pd.concat([treatment_group, control_group], axis=1, keys=["treatment_group", "control_group"], join="outer")
    df.index = extract_numbers(df.index)
    return df.dropna()


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


def extract_numbers(index_values):
    return [int(re.search(r"[-\d]+$", item).group()) for item in index_values]


def check_date_format_consistency(data, time_col, experiment_start_date, experiment_end_date):
    """Ensure all date-related inputs are either all integers or all datetime-like values."""
    time_col_is_int = pd.api.types.is_integer_dtype(data[time_col])
    start_date_is_int = isinstance(experiment_start_date, int)
    end_date_is_int = isinstance(experiment_end_date, int) if experiment_end_date is not None else start_date_is_int

    if not ((time_col_is_int and start_date_is_int and end_date_is_int) or (not time_col_is_int and not start_date_is_int and not end_date_is_int)):
        raise ValueError("Mismatch in date formats: Ensure `time_col`, `experiment_start_date`, and `experiment_end_date` are either all integers or all datetime values.")

    return start_date_is_int  # Return format flag for downstream logic
