import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def validate_panel_data(data, unit_col, time_col, value_col, covariates):
    """Validate the common input schema for causal inference models."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must contain at least one observation.")
    if not data.columns.is_unique:
        raise ValueError("data must not contain duplicate column names.")
    if not all(isinstance(column, str) and column for column in (unit_col, time_col, value_col)):
        raise TypeError("unit_col, time_col, and value_col must be non-empty column names.")
    if len({unit_col, time_col, value_col}) != 3:
        raise ValueError("unit_col, time_col, and value_col must identify different columns.")
    if covariates is not None and (not isinstance(covariates, list) or not all(isinstance(column, str) and column for column in covariates)):
        raise TypeError("covariates must be a list of column names or None.")
    if covariates and (len(set(covariates)) != len(covariates) or set(covariates) & {unit_col, time_col, value_col}):
        raise ValueError("covariates must be distinct from each other and from unit, time, and outcome columns.")
    required = [unit_col, time_col, value_col] + (covariates or [])
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}.")
    if "treatment_start" in data.columns:
        raise ValueError("data must not contain the reserved column 'treatment_start'.")
    if data[[unit_col, time_col]].isna().any().any():
        raise ValueError("unit and time columns must not contain missing values.")
    if data.duplicated([unit_col, time_col]).any():
        raise ValueError("data must contain at most one observation per unit and time.")
    if not (pd.api.types.is_numeric_dtype(data[time_col]) or is_datetime64_any_dtype(data[time_col])):
        raise TypeError(f"time column '{time_col}' must contain numeric or datetime values.")
    if pd.api.types.is_numeric_dtype(data[time_col]) and not np.isfinite(data[time_col].to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError(f"time column '{time_col}' must contain finite values.")
    for column in [value_col] + (covariates or []):
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(f"column '{column}' must contain numeric values.")
        if not np.isfinite(data[column].to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"column '{column}' must not contain missing or non-finite values.")


def validate_treatment(data, unit_col, time_col, value_col, covariates, treatment):
    """Validate the treatment column or unit-to-start mapping."""
    if isinstance(treatment, str):
        if treatment not in data.columns:
            raise ValueError(f"data is missing treatment column '{treatment}'.")
        if treatment in {unit_col, time_col, value_col, *(covariates or [])}:
            raise ValueError("treatment column must be distinct from unit, time, outcome, and covariate columns.")
        if data[treatment].isna().any() or not data[treatment].isin([0, 1, False, True]).all():
            raise ValueError(f"treatment column '{treatment}' must contain only 0/1 or boolean values, without missing values.")
        if not data[treatment].any():
            raise ValueError(f"treatment column '{treatment}' must identify at least one treated unit.")
        ordered = data.sort_values([unit_col, time_col])
        if (ordered[treatment].astype(int).groupby(ordered[unit_col]).diff() < 0).any():
            raise ValueError(f"treatment column '{treatment}' must remain 1 after a unit is treated.")
    elif isinstance(treatment, dict):
        if not treatment:
            raise ValueError("treatment mapping must identify at least one treated unit.")
        unknown = [unit for unit in treatment if unit not in set(data[unit_col])]
        if unknown:
            raise ValueError(f"treatment mapping contains units not present in data: {unknown}.")
        starts = pd.Series(list(treatment.values()))
        if is_datetime64_any_dtype(data[time_col]):
            parsed = pd.to_datetime(starts, errors="coerce")
        else:
            parsed = pd.to_numeric(starts, errors="coerce")
        if parsed.isna().any() or (not is_datetime64_any_dtype(data[time_col]) and not np.isfinite(parsed.to_numpy(dtype=float)).all()):
            raise ValueError("treatment mapping must contain valid start times matching the time column.")
    else:
        raise TypeError("treatment must be a binary column name or a mapping of unit IDs to start times.")


def normalize_treatment(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment,  # str column OR dict {unit_id: treat_time}
) -> pd.DataFrame:
    """
    Return a DataFrame mapping *treated* units -> adoption time,
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
        # Keep the dtype of unit_col
        first_treat = first_treat.dropna().reset_index()
        first_treat.columns = [unit_col, "treatment_start"]
        first_treat["treatment_start"] = coerce(first_treat["treatment_start"])
        first_treat[unit_col] = first_treat[unit_col].astype(df[unit_col].dtype)
        return first_treat

    if isinstance(treatment, dict):
        s = pd.Series(treatment)
        s = s.dropna().reset_index()
        s.columns = [unit_col, "treatment_start"]
        s["treatment_start"] = coerce(s["treatment_start"])
        s[unit_col] = s[unit_col].astype(df[unit_col].dtype)
        return s

    raise TypeError("`treatment` must be a column name (str) or a dict {unit_id: treat_time}.")


def check_date_format_consistency(data, time_col):
    """Ensure all date-related inputs are either all integers or all datetime-like values."""
    # Use pandas nullable integer type check
    time_col_is_int = pd.api.types.is_integer_dtype(data[time_col].dropna())
    experiment_start_date_is_int = pd.api.types.is_integer_dtype(data["treatment_start"].dropna()) or pd.api.types.is_float_dtype(data["treatment_start"].dropna())

    if not ((time_col_is_int and experiment_start_date_is_int) or (not time_col_is_int and not experiment_start_date_is_int)):
        raise ValueError("Mismatch in date formats: Ensure `time_col`, `experiment_start_date`, and `experiment_end_date` are either all integers or all datetime values.")

    return experiment_start_date_is_int  # Return format flag for downstream logic


def add_days_since_treatment_start(data, time_col):
    start_date_is_int = check_date_format_consistency(data, time_col)

    if start_date_is_int:
        data["days_since_treatment_start"] = np.max(data[time_col] - data["treatment_start"], 0).astype(int)
    else:
        start_date = pd.to_datetime(data["treatment_start"])
        data["days_since_treatment_start"] = np.max((pd.to_datetime(data[time_col]) - start_date).dt.days, 0).fillna(0).astype(int)
    data["days_since_treatment_start"] = data["days_since_treatment_start"].fillna(0)
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
        validate_panel_data(data, unit_col, time_col, value_col, covariates)
        validate_treatment(data, unit_col, time_col, value_col, covariates, treatment)
        self.covariates = covariates
        self.time_col = time_col
        self.unit_col = unit_col
        self.value_col = value_col
        self.treatment = normalize_treatment(data, unit_col=unit_col, time_col=time_col, treatment=treatment)

        self.donors = set(data[unit_col].unique()) - set(self.treatment[unit_col])
        data = data.merge(self.treatment, on=unit_col, how="left")
        self.data = data
        self.cols = [unit_col, time_col, self.value_col, "treatment_start"] + (covariates or [])
        self.data = self.data[self.cols].copy(deep=False)

        self.training_end_date = training_end_date
        self.se_computed = False
        self.se = None
        self.results = None
        self.model_fitted = False
        self.model = model

        if self.model is not None and hasattr(self.model, "fit_intercept"):
            self.model.fit_intercept = False

    def fit(self, calculate_se=False, significance_level=0.05, prune_data_for_se_computation=True): ...
