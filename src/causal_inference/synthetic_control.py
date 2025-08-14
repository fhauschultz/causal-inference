import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_datetime64_any_dtype
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from causal_inference.utils import BaseCausalInference

# Set the style for matplotlib
plt.style.use("https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle")


def get_inference_data(data, time_col, unit_col, value_col, treated_unit):
    """
    Extract inference data for the full period.

    Returns:
    --------
        tuple: (x_outcome_treated, x_outcome_donor, donor_units)
    """
    full_data_pivot = data.pivot(index=time_col, columns=unit_col, values=value_col)
    treated_values_full = full_data_pivot[treated_unit].dropna()
    donor_units = [unit for unit in full_data_pivot.columns if unit != treated_unit]
    donor_values_full = full_data_pivot[donor_units]

    # Align time periods for full outcomes
    common_time_full = treated_values_full.index.intersection(donor_values_full.index)
    x_outcome_treated = treated_values_full.loc[common_time_full].values.reshape(1, -1)
    x_outcome_donor = donor_values_full.loc[common_time_full].values

    return x_outcome_treated, x_outcome_donor, donor_units


def get_training_data(data, time_col, unit_col, value_col, treated_unit, experiment_date, training_end_date=None, covariates=None):
    """
    Extract training data for the pre-experiment period, including covariates.

    Returns:
    --------
        tuple: (x_train_treated, x_train_donor)
    """
    pre_treatment_data = data[data[time_col] < experiment_date]
    if training_end_date:
        pre_treatment_data = pre_treatment_data[pre_treatment_data[time_col] < training_end_date]

    # Pivot data for training
    data_pivot = pre_treatment_data.pivot(index=time_col, columns=unit_col, values=value_col)
    treated_values = data_pivot[treated_unit].dropna()
    donor_units = [unit for unit in data_pivot.columns if unit != treated_unit]
    donor_values = data_pivot[donor_units]

    # Align time periods for training data
    common_time = treated_values.index.intersection(donor_values.index)
    x_train_treated = [treated_values.loc[common_time].values]
    x_train_donor = [donor_values.loc[common_time].values]

    # Handle covariates if provided
    if covariates:
        for covariate in covariates:
            covar_data = pre_treatment_data.pivot(index=time_col, columns=unit_col, values=covariate)
            treated_covar = covar_data[treated_unit].dropna()
            donor_covar = covar_data[donor_units]

            # Align time periods
            common_time_covar = treated_covar.index.intersection(donor_covar.index)
            x_train_treated.append(treated_covar.loc[common_time_covar].values)
            x_train_donor.append(donor_covar.loc[common_time_covar].values)

    # Concatenate data after looping through covariates
    x_train_treated = np.concatenate(x_train_treated)
    x_train_donor = np.concatenate(x_train_donor)

    return x_train_treated, x_train_donor


def normalize_treatment(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment,  # str column OR dict {unit_id: treat_time}
) -> pd.Series:
    """
    Return a Series mapping *treated* units -> adoption time,
    in the same dtype/format as df[time_col].
    Units that are never treated are omitted from the index.
    If `treatment` is a column name, it should be a binary indicator (1/True for treated).
    """

    # Coercion helper
    def coerce(values):
        if is_datetime64_any_dtype(df[time_col].dtype):
            return pd.to_datetime(values, errors="coerce")
        else:
            return pd.to_numeric(values, errors="coerce")

    if isinstance(treatment, str):
        # Find first treated period for each unit
        treated_rows = df[df[treatment].astype(bool)]
        first_treat = treated_rows.groupby(unit_col)[time_col].min()
        first_treat.index = first_treat.index.astype(str)
        first_treat = coerce(first_treat)
        return first_treat.dropna()

    if isinstance(treatment, dict):
        s = pd.Series({str(k): v for k, v in treatment.items()})
        s = coerce(s)
        return s.dropna()

    raise TypeError("`treatment` must be a column name (str) or a dict {unit_id: treat_time}.")


def prepare_data(data, time_col, unit_col, value_col, treated_unit, experiment_date, training_end_date=None, covariates=None):
    """
    Prepare data for fitting the model by combining training and inference data extraction.

    Returns:
    --------
        tuple: (x_train_treated, x_train_donor, x_outcome_treated, x_outcome_donor, donor_units)
    """
    x_train_treated, x_train_donor = get_training_data(data, time_col, unit_col, value_col, treated_unit, experiment_date, training_end_date, covariates)
    x_outcome_treated, x_outcome_donor, donor_units = get_inference_data(data, time_col, unit_col, value_col, treated_unit)

    return x_train_treated, x_train_donor, x_outcome_treated, x_outcome_donor, donor_units


def filter_donor_units(df, treatment_unit, unit_col):
    """
    Filter a DataFrame to retain only donor units (and the treated unit) that have the same number
    of observations as the treatment unit.

    Parameters:
    -----------
        df: pd.DataFrame
            DataFrame with columns 'UNITS', 'TIME', and 'VAR'.
        treatment_unit: str
            Identifier for the treatment unit.
        unit_col: str
            Column name for the units.

    Returns:
    --------
        pd.DataFrame: Filtered DataFrame containing only donor units with the
        same number of observations as the treatment unit.
    """
    # Count observations for each unit
    unit_counts = df.groupby(unit_col).size()

    # Get the number of observations for the treatment unit
    treatment_count = unit_counts[treatment_unit]

    # Identify donor units with matching counts
    matching_units = unit_counts[unit_counts >= treatment_count].index

    # Filter the DataFrame
    filtered_df = df[df[unit_col].isin(matching_units)]

    return filtered_df


class ClassicModelFitter(BaseEstimator, RegressorMixin):
    """
    Classic synthetic control method using constrained optimization.
    Implements scikit-learn's estimator interface.
    """

    def __init__(self):
        self.coef_ = None

    def fit(self, x, y):
        def loss(w):
            return np.linalg.norm(y - x @ w)

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(x.shape[1])]

        w_init = np.ones(x.shape[1]) / x.shape[1]
        result = minimize(loss, w_init, constraints=cons, bounds=bounds)
        self.coef_ = result.x
        return self

    def predict(self, x):
        return x @ self.coef_


def collect_series_to_dataframe(series_list):
    df = pd.DataFrame({s.name: s for s in series_list})
    return df


def get_good_fit_units(effects, index_threshold, tolerance_pct):
    """
    Identify units with good fit based on their effects before a given threshold date.

    Parameters:
    -----------
    effects : pd.DataFrame
        DataFrame containing the effects with dates as the index.
    index_threshold : datetime-like
        The threshold date to separate the data.
    tolerance_pct : float
        The percentage tolerance to determine the threshold for good fit.

    Returns:
    --------
    pd.Index
        Index of columns (units) that have a good fit based on the specified tolerance.
    """
    abs_effects = effects.abs()
    before_date_data = abs_effects[abs_effects.index < index_threshold]
    means_before_date = before_date_data.mean()
    threshold = means_before_date.quantile(1 - tolerance_pct / 100)
    filtered_df = effects.loc[:, means_before_date < threshold]
    return filtered_df.columns


def prune_units_for_se_computation(placebo_effects, experiment_date, tolerance_pct):
    good_fit_units = get_good_fit_units(placebo_effects, index_threshold=experiment_date, tolerance_pct=tolerance_pct)
    return placebo_effects[good_fit_units]


class SyntheticControl(BaseCausalInference):
    def __init__(
        self,
        data,  # tidy panel
        unit_col,  # str | list[str]
        time_col,  # str
        value_col,  # str
        treatment,  # column name | dict | Series | list[(unit, time)] | (unit, time)
        *,
        training_end_date=None,
        covariates=None,
        model=None,
    ):
        self.covariates = covariates
        self.time_col = time_col
        self.unit_col = unit_col
        self.value_col = value_col
        self.treatment = normalize_treatment(data, unit_col=unit_col, time_col=time_col, treatment=treatment)
        self.donors = set(data[unit_col].unique()) - set(self.treatment.index)
        self.cols = [unit_col, time_col, value_col] + (covariates or [])

        self.data = data[self.cols].copy(deep=False)
        self.training_end_date = training_end_date
        self.se_computed = False
        self.se = None
        self.model = model if model is not None else ClassicModelFitter()
        self.results = None
        self.sc_fitted = False

    def _fit_model(self, treated_unit, experiment_date, training_end_date=None):
        """
        Fit the model to the data.

        Parameters:
        -----------
            treated_unit: str
                Identifier for the treated unit.
            experiment_date: str or pd.Timestamp
                Date of the experiment.
            training_end_date: str or pd.Timestamp, optional
                End date for the training period. Defaults to None.

        Returns:
        --------
            tuple: (synthetic_control, weights)
        """
        (x_train_treated, x_train_donor, x_outcome_treated, x_outcome_donor, donor_units) = prepare_data(self.data, self.time_col, self.unit_col, self.value_col, treated_unit, experiment_date, training_end_date, self.covariates)
        self.model.fit(x_train_donor, x_train_treated)
        weights = None

        # Extract weights - handle different model types
        if hasattr(self.model, "coef_"):
            weights = self.model.coef_
        elif hasattr(self.model, "feature_importances_"):
            weights = self.model.feature_importances_
            # Normalize to sum to 1 if needed
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
        else:
            print("Warning: Model must have coef_ or feature_importances_ attribute to extract weights")

        synthetic_control = self.model.predict(x_outcome_donor)

        weights_df = pd.DataFrame(weights, index=donor_units, columns=["weight"]).sort_values("weight", ascending=False)
        return synthetic_control, weights_df

    def _get_treated_values(self, treated_unit):
        """
        Get values for the treated unit.

        Parameters:
        -----------
            treated_unit: str
                Identifier for the treated unit.

        Returns:
        --------
            pd.Series: Values for the treated unit.
        """
        outcome_data = self.data[[self.unit_col, self.time_col, self.value_col]]
        return outcome_data[outcome_data[self.unit_col] == treated_unit].pivot(index=self.time_col, columns=self.unit_col, values=self.value_col).iloc[:, 0]

    def _get_results(self, treated_unit=None):
        """
        Get the results of the synthetic control method.

        Parameters:
        -----------
            treated_unit: str, optional
                Identifier for the treated unit. Defaults to None.

        Returns:
        --------
            pd.DataFrame: DataFrame containing the results.
        """
        if not treated_unit:
            treated_unit = self.treated_unit
        impact = pd.DataFrame()
        impact["Treated"] = self._get_treated_values(treated_unit)
        impact["Synthetic Control"] = self.synthetic_controls[treated_unit]
        impact["Effect"] = impact["Treated"] - impact["Synthetic Control"]
        if self.se_computed:
            impact = impact.join(self.se, how="left")
        return impact

    def _calculate_standard_errors(self, experiment_date, significance_level, prune_data):
        """
        Calculate standard errors for the synthetic control method.

        Parameters:
        -----------
            experiment_date: str or pd.Timestamp
                Date of the experiment.

        Returns:
        --------
            pd.DataFrame: DataFrame containing the standard errors.
        """
        tolerance_pre_treatment_pruning_pct = 10
        if not significance_level:
            significance_level = 5
        placebo_effects = []
        for donor in self.donors:
            synthetic_control, _ = self._fit_model(donor, experiment_date)
            placebo_impact = self._get_treated_values(donor) - synthetic_control
            placebo_effects.append(placebo_impact)

        placebo_effects = collect_series_to_dataframe(placebo_effects)

        if prune_data:
            placebo_effects = prune_units_for_se_computation(placebo_effects, experiment_date, tolerance_pre_treatment_pruning_pct)

        placebo_effects_np = np.array(placebo_effects)
        upper_bound = np.percentile(placebo_effects_np, 100 - significance_level / 2, axis=1)
        lower_bound = np.percentile(placebo_effects_np, significance_level / 2, axis=1)
        self.se = pd.DataFrame({"Upper Bound": upper_bound, "Lower Bound": lower_bound}, index=placebo_effects.index)

        self.se_computed = True
        self.placebo_effects = placebo_effects
        return self.se

    def fit(self, calculate_se=True, significance_level=None, prune_data_for_se_computation=True):
        results = {}
        self.synthetic_controls = {}
        self.synthetic_weights = {}
        for treated_unit in self.treatment.index:
            experiment_date = self.treatment[treated_unit]
            training_end_date = self.training_end_date if self.training_end_date else experiment_date

            # Prepare data and fit the model
            self.synthetic_controls[treated_unit], self.synthetic_weights[treated_unit] = self._fit_model(treated_unit, experiment_date, training_end_date)

            if calculate_se and self.treatment.index.size == 1:
                self._calculate_standard_errors(experiment_date, significance_level, prune_data_for_se_computation)

            # Store results
            results[treated_unit] = self._get_results(treated_unit)

        self.results = self.aggregate_results(results)
        self.sc_fitted = True

    def aggregate_results(self, results):
        """
        Aggregate results by averaging the 'Effect', 'Synthetic Control', and 'Treated' across treated units,
        normalizing time to 'periods since treatment' if needed.
        Returns a DataFrame with columns: 'period', 'average_effect', 'average_synth', 'average_treated'.
        """
        dfs = []
        multi_treated = len(results) > 1

        for treated_unit, df in results.items():
            df = df.copy()
            if multi_treated:
                treat_time = self.treatment[treated_unit]
                df["Period"] = df.index - treat_time
            else:
                df["Period"] = df.index
                return df
            dfs.append(df.reset_index(drop=True))
        combined = pd.concat(dfs, ignore_index=True)
        avg = (
            combined.groupby("Period")
            .agg(
                {
                    "Treated": "mean",
                    "Synthetic Control": "mean",
                    "Effect": "mean",
                }
            )
            .reset_index()
        )
        return avg

    def summary(self):
        """
        Get a summary of the results.

        Returns:
        --------
            pd.Series: Summary of the results.
        """
        impact = self._get_results()
        return impact[impact.index > self.experiment_date].mean()

    def get_experiment_date(self):
        experiment_date = self.treatment.iloc[0] if len(self.treatment) == 1 else 0
        return experiment_date

    def plot_histogram(self):
        """
        Plot a histogram of the placebo effects.
        """
        plt.hist(self.placebo_effects.mean(1))
        plt.axvline(self.summary()["Effect"], c="black", label="Treatment Effect")
        plt.legend()
        plt.show()

    def plot(self, calculate_se=True, significance_level=None, prune_data_for_se_computation=True):
        """
        Plot the treated unit and synthetic control.

        """
        if not self.sc_fitted:
            self.fit(calculate_se=calculate_se, significance_level=significance_level, prune_data_for_se_computation=prune_data_for_se_computation)
        impact = self.results

        plt.figure(figsize=(10, 6))
        plt.title(self.value_col)
        plt.plot(impact["Period"], impact["Treated"], label="Treated Unit", linestyle="-", color="blue", lw=3, alpha=0.8)
        plt.plot(impact["Period"], impact["Synthetic Control"], label="Synthetic Control", linestyle="--", color="black", lw=3, alpha=0.8)
        if self.se_computed:
            plt.fill_between(impact["Period"], impact["Synthetic Control"] + impact["Lower Bound"], impact["Synthetic Control"] + impact["Upper Bound"], color="grey", alpha=0.3, label="Confidence Band")
        plt.axvline(x=self.get_experiment_date(), color="gray", linestyle=":", label="Experiment Date", lw=2)

        if self.training_end_date:
            plt.axvline(x=self.training_end_date, color="magenta", linestyle="--", label="Training End Date", alpha=0.5, lw=2)

        plt.xticks(rotation=90)
        plt.legend()
        sns.despine()
        plt.show()

    def plot_effect(self):
        """
        Plot the effect of the treatment.
        """
        if not self.sc_fitted:
            print("Synthetic control has not been computed. Please run the fit method first.")
            return
        impact = self.results

        plt.figure(figsize=(10, 6))
        plt.plot(impact["Period"], impact["Effect"], label="Treated Unit", linestyle="-", color="blue", lw=3, alpha=0.8)
        if self.se_computed:
            plt.fill_between(impact["Period"], impact["Lower Bound"], impact["Upper Bound"], color="grey", alpha=0.3, label="Confidence Band")
        plt.axvline(x=self.get_experiment_date(), color="gray", linestyle=":", label="Experiment Date", lw=2)

        if self.training_end_date:
            plt.axvline(x=self.training_end_date, color="magenta", linestyle="--", label="Training End Date", alpha=0.5, lw=2)

        plt.ylabel("Effect")
        plt.title(self.value_col)
        plt.xticks(rotation=90)
        plt.axhline(0, c="gray", alpha=0.2)
        plt.legend()
        sns.despine()
        plt.show()
