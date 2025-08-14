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
    ): ...

    def fit(self, calculate_se=False, significance_level=0.05, prune_data_for_se_computation=True): ...
