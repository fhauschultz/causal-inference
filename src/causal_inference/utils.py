import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


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
        first_treat = first_treat.dropna().reset_index()
        first_treat.columns = [unit_col, "treatment_start"]
        return first_treat

    if isinstance(treatment, dict):
        s = pd.Series({str(k): v for k, v in treatment.items()})
        s = coerce(s)
        s = s.dropna().reset_index()
        s.columns = [unit_col, "treatment_start"]
        return s

    raise TypeError("`treatment` must be a column name (str) or a dict {unit_id: treat_time}.")


def check_date_format_consistency(data, time_col):
    """Ensure all date-related inputs are either all integers or all datetime-like values."""
    time_col_is_int = pd.api.types.is_integer_dtype(data[time_col])
    experiment_start_date_is_int = isinstance(data["treatment_start"], int)

    if not ((time_col_is_int and experiment_start_date_is_int) or (not time_col_is_int and not experiment_start_date_is_int)):
        raise ValueError("Mismatch in date formats: Ensure `time_col`, `experiment_start_date`, and `experiment_end_date` are either all integers or all datetime values.")

    return experiment_start_date_is_int  # Return format flag for downstream logic


def add_days_since_experiment_start(data, time_col):
    start_date_is_int = check_date_format_consistency(data, time_col)

    if start_date_is_int:
        data["days_since_experiment_start"] = data[time_col] - data["treatment_start"]
    else:
        start_date = pd.to_datetime(data["treatment_start"])
        data["days_since_experiment_start"] = (pd.to_datetime(data[time_col]) - start_date).dt.days

    return data


class BaseCausalInference:
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
        self.donors = set(data[unit_col].unique()) - set(self.treatment[unit_col])
        data = data.merge(self.treatment, on=unit_col, how="left")
        self.data = add_days_since_experiment_start(data, time_col=self.time_col)
        self.cols = [unit_col, time_col, value_col, "days_since_treatment_start"] + (covariates or [])
        self.data = self.data[self.cols].copy(deep=False)

        self.training_end_date = training_end_date
        self.se_computed = False
        self.se = None
        self.results = None
        self.model_fitted = False
        self.model = model

        if self.sklearn_model is not None and hasattr(self.sklearn_model, "fit_intercept"):
            self.sklearn_model.fit_intercept = False

        plt.style.use("classic")

    def fit(self, calculate_se=False, significance_level=0.05, prune_data_for_se_computation=True): ...
