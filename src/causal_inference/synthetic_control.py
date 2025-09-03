import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from causal_inference.utils import BaseCausalInference


class SyntheticControl(BaseCausalInference):
    def _fit_standard_synthetic_control(self, treated_unit, experiment_date, training_end_date=None):
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

    def _calculate_standard_errors(self, significance_level, prune_data):
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

        def get_combinations(A, N):
            combs = [list(c) for c in itertools.combinations(A, N)]
            return combs

        tolerance_pre_treatment_pruning_pct = 10
        if not significance_level:
            significance_level = 5
        placebo_effects = []
        combs = get_combinations(self.donors, N=len(self.treatment))
        treatment_start = self.treatment["treatment_start"].min()
        i = 0
        for comb in combs:
            i += 1
            placebo_treatment = pd.DataFrame({self.unit_col: comb, "treatment_start": [treatment_start] * len(comb)})
            placebo_results = self._fit_model(placebo_treatment, self.unit_col)
            placebo_effects.append(placebo_results[["Period", "Effect"]].rename(columns={"Effect": str(i)}).set_index("Period"))

        placebo_effects = pd.concat(placebo_effects, axis=1)
        if prune_data:
            placebo_effects = prune_units_for_se_computation(placebo_effects, treatment_start, tolerance_pre_treatment_pruning_pct)

        placebo_effects_np = np.array(placebo_effects)
        upper_bound = np.percentile(placebo_effects_np, 100 - significance_level / 2, axis=1)
        lower_bound = np.percentile(placebo_effects_np, significance_level / 2, axis=1)
        mean = np.mean(placebo_effects_np, axis=1)
        self.se = pd.DataFrame({"Upper Bound": upper_bound - mean, "Lower Bound": lower_bound - mean}, index=placebo_effects.index).reset_index()
        self.se_computed = True
        self.placebo_effects = placebo_effects
        return self.se

    def _fit_model(self, treatment, unit_col):
        synthetic_controls = {}
        synthetic_weights = {}
        results_pd = pd.DataFrame()
        n_treated = len(treatment)
        for treated_unit in treatment[unit_col].unique():
            experiment_date = treatment[treatment[unit_col] == treated_unit]["treatment_start"].iloc[0]
            training_end_date = self.training_end_date if self.training_end_date else experiment_date

            # Prepare data and fit the model
            synthetic_controls[treated_unit], synthetic_weights[treated_unit] = self._fit_standard_synthetic_control(treated_unit, experiment_date, training_end_date)

            impact = pd.DataFrame()
            impact["Treated"] = self._get_treated_values(treated_unit)
            impact["Synthetic Control"] = synthetic_controls[treated_unit]
            impact["Effect"] = impact["Treated"] - impact["Synthetic Control"]

            if n_treated > 1:
                impact["Period"] = impact.index - experiment_date
            else:
                impact["Period"] = impact.index

            # Store results
            results_pd = pd.concat([results_pd, impact], axis=0)

        avg = (
            results_pd.groupby("Period")
            .agg(
                Treated=("Treated", "mean"),
                Synthetic_Control=("Synthetic Control", "mean"),
                Effect=("Effect", "mean"),
                Count=("Effect", "count"),  # or Treated/any col, same result if no NaNs
            )
            .reset_index()
            .rename(columns={"Synthetic_Control": "Synthetic Control"})
        )
        avg = avg[avg["Count"] >= n_treated][["Period", "Treated", "Synthetic Control", "Effect"]]

        return avg

    def fit(self, calculate_se=True, significance_level=None, prune_data_for_se_computation=True):
        self.model = self.model if self.model is not None else ClassicModelFitter()
        self.results = self._fit_model(self.treatment, unit_col=self.unit_col)
        if calculate_se:
            self._calculate_standard_errors(significance_level, prune_data_for_se_computation)
            self.results = self.results.merge(self.se, on="Period", how="left")
            self.results["Upper Bound"] = self.results["Upper Bound"] + self.results["Treated"]
            self.results["Lower Bound"] = self.results["Lower Bound"] + self.results["Treated"]
        self.model_fitted = True

    def get_experiment_date(self):
        """
        Returns the experiment date for the treated unit(s) as a native Python type.
        If there is only one treated unit, returns its treatment start date.
        Otherwise, returns 0.
        """
        if len(self.treatment) == 1:
            # Get the treatment start date for the single treated unit
            experiment_date = self.treatment["treatment_start"].iloc[0]
            # Convert numpy types to native Python types for compatibility with plotting
            if hasattr(experiment_date, "item"):
                experiment_date = experiment_date.item()
            return experiment_date
        else:
            # Multiple treated units or no treatment info
            return 0

    def plot(
        self,
        layout="row",
        figsize=None,
        matplotlib_style="ggplot",
        matplotlib_theme_color="navy",
    ):
        """
        Create a single figure with subplots for:
        1) Treated vs Synthetic Control
        2) Effect
        3) Placebo-effects Histogram

        Parameters
        ----------
        layout : {"row","col"}, optional
            Arrange subplots horizontally ("row") or vertically ("col").
        figsize : tuple, optional
            Figure size passed to plt.subplots(). If None, picks a good default.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray of matplotlib.axes.Axes (length 3)
        """
        self.matplotlib_theme_color = matplotlib_theme_color
        self.matplotlib_style = matplotlib_style

        if layout not in {"row", "col"}:
            layout = "row"

        if figsize is None:
            figsize = (14, 8) if layout == "row" else (6, 14)

        nrows, ncols = (1, 3) if layout == "row" else (3, 1)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.ravel()

        # 1) Treated vs Synthetic Control
        self.plot_treatment_control(ax=axes[0])

        # 2) Effect
        self.plot_effect(ax=axes[1])

        # 3) Placebo-effects Histogram
        self.plot_histogram(ax=axes[2])

        plt.tight_layout()
        return fig, axes

    def plot_histogram(self, ax=None):
        """
        Plot a histogram of the placebo effects on the provided axis.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        post_mask = self.results["Period"] > self.get_experiment_date()
        treatment_effects = self.results.loc[post_mask].mean(numeric_only=True)

        placebo_post = self.placebo_effects.loc[self.placebo_effects.index > self.get_experiment_date()]
        placebo_means = placebo_post.mean(axis=0)

        ax.hist(placebo_means, alpha=0.2, color=self.matplotlib_theme_color, label="Placebo Means")
        ax.axvline(
            treatment_effects["Effect"],
            color=self.matplotlib_theme_color,
            label="Treatment Effect",
            linewidth=2,
        )
        ax.set_title("Placebo Effects (Post-Period) vs Treatment Effect")
        ax.legend()
        sns.despine(ax=ax)
        return ax

    def plot_treatment_control(
        self,
        ax=None,
        calculate_se=True,
        significance_level=None,
        prune_data_for_se_computation=True,
    ):
        """
        Plot the treated unit and synthetic control on the provided axis.
        """
        if not self.model_fitted:
            self.fit(
                calculate_se=calculate_se,
                significance_level=significance_level,
                prune_data_for_se_computation=prune_data_for_se_computation,
            )

        impact = self.results

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.set_title(self.value_col)
        ax.plot(
            impact["Period"],
            impact["Treated"],
            label="Treated Unit",
            linestyle="-",
            color=self.matplotlib_theme_color,
            lw=3,
            alpha=0.8,
        )
        ax.plot(
            impact["Period"],
            impact["Synthetic Control"],
            label="Synthetic Control",
            linestyle="--",
            color="black",
            lw=3,
            alpha=0.8,
        )
        if self.se_computed:
            ax.fill_between(
                impact["Period"],
                impact["Lower Bound"],
                impact["Upper Bound"],
                color=self.matplotlib_theme_color,
                alpha=0.2,
                label="Confidence Band",
            )

        ax.axvline(x=self.get_experiment_date(), color="gray", linestyle=":", label="Experiment Date", lw=2)
        if getattr(self, "training_end_date", None):
            ax.axvline(
                x=self.training_end_date,
                color="magenta",
                linestyle="--",
                label="Training End Date",
                alpha=0.5,
                lw=2,
            )

        for lbl in ax.get_xticklabels():
            lbl.set_rotation(90)
        ax.legend()
        sns.despine(ax=ax)
        return ax

    def plot_effect(self, ax=None):
        """
        Plot the effect of the treatment on the provided axis.
        """
        if not self.model_fitted:
            print("Synthetic control has not been computed. Please run the fit method first.")
            return None

        impact = self.results

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            impact["Period"],
            impact["Effect"],
            label="Treated Unit",
            linestyle="-",
            color=self.matplotlib_theme_color,
            lw=3,
            alpha=0.8,
        )

        if self.se_computed:
            lower = impact["Lower Bound"] - impact["Treated"] + impact["Effect"]
            upper = impact["Upper Bound"] - impact["Treated"] + impact["Effect"]
            ax.fill_between(
                impact["Period"],
                lower,
                upper,
                color=self.matplotlib_theme_color,
                alpha=0.2,
                label="Confidence Band",
            )

        ax.axvline(x=self.get_experiment_date(), color="gray", linestyle=":", label="Experiment Date", lw=2)
        if getattr(self, "training_end_date", None):
            ax.axvline(
                x=self.training_end_date,
                color="magenta",
                linestyle="--",
                label="Training End Date",
                alpha=0.5,
                lw=2,
            )

        ax.set_title(self.value_col + " - Effect")
        ax.axhline(0, c="gray", alpha=0.2)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(90)
        ax.legend()
        sns.despine(ax=ax)
        return ax


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
