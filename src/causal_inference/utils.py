def check_date_format_consistency(data, time_col, experiment_start_date, experiment_end_date):
    """Ensure all date-related inputs are either all integers or all datetime-like values."""
    time_col_is_int = pd.api.types.is_integer_dtype(data[time_col])
    start_date_is_int = isinstance(experiment_start_date, int)
    end_date_is_int = isinstance(experiment_end_date, int) if experiment_end_date is not None else start_date_is_int

    if not ((time_col_is_int and start_date_is_int and end_date_is_int) or (not time_col_is_int and not start_date_is_int and not end_date_is_int)):
        raise ValueError("Mismatch in date formats: Ensure `time_col`, `experiment_start_date`, and `experiment_end_date` are either all integers or all datetime values.")

    return start_date_is_int  # Return format flag for downstream logic


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
        if self.sklearn_model is not None and hasattr(self.sklearn_model, "fit_intercept"):
            self.sklearn_model.fit_intercept = False

        start_date_is_int = check_date_format_consistency(data, time_col, experiment_start_date, experiment_end_date)

    def fit(self, calculate_se=False, significance_level=0.05, prune_data_for_se_computation=True): ...
