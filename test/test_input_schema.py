import pandas as pd
import pytest

from causal_inference import StaggeredDID, SyntheticControl


@pytest.fixture
def panel():
    return pd.DataFrame({"unit": ["A", "A", "B", "B"], "time": [0, 1, 0, 1], "outcome": [1.0, 2.0, 1.5, 1.6], "treated": [0, 1, 0, 0]})


@pytest.mark.parametrize("model_class", [StaggeredDID, SyntheticControl])
@pytest.mark.parametrize(
    ("change", "exception", "message"),
    [
        (lambda data: None, TypeError, "data must be a pandas DataFrame"),
        (lambda data: data.drop(columns="outcome"), ValueError, "missing required columns.*outcome"),
        (lambda data: pd.concat([data, data.iloc[[0]]]), ValueError, "at most one observation per unit and time"),
        (lambda data: data.assign(outcome="bad"), TypeError, "column 'outcome' must contain numeric values"),
        (lambda data: data.assign(time="bad"), ValueError, "at most one observation per unit and time"),
    ],
)
def test_invalid_panel_schema(model_class, panel, change, exception, message):
    with pytest.raises(exception, match=message):
        model_class(data=change(panel), unit_col="unit", time_col="time", value_col="outcome", treatment="treated")


@pytest.mark.parametrize("model_class", [StaggeredDID, SyntheticControl])
@pytest.mark.parametrize(
    ("change", "treatment", "exception", "message"),
    [
        (lambda data: data, "missing", ValueError, "missing treatment column 'missing'"),
        (lambda data: data.assign(treated=[0, 2, 0, 0]), "treated", ValueError, "must contain only 0/1"),
        (lambda data: data.assign(treated=[1, 0, 0, 0]), "treated", ValueError, "must remain 1"),
        (lambda data: data.assign(treated=0), "treated", ValueError, "at least one treated unit"),
        (lambda data: data, {}, ValueError, "at least one treated unit"),
        (lambda data: data, {"C": 1}, ValueError, "units not present in data"),
        (lambda data: data, {"A": "invalid"}, ValueError, "valid start times"),
        (lambda data: data, ["A"], TypeError, "treatment must be"),
        (lambda data: data.assign(outcome=[1, 2, float("nan"), 4]), "treated", ValueError, "outcome.*missing or non-finite"),
    ],
)
def test_invalid_treatment_schema(model_class, panel, change, treatment, exception, message):
    with pytest.raises(exception, match=message):
        model_class(data=change(panel), unit_col="unit", time_col="time", value_col="outcome", treatment=treatment)


def test_synthetic_control_requires_untreated_donor(panel):
    with pytest.raises(ValueError, match="at least one untreated donor"):
        SyntheticControl(data=panel, unit_col="unit", time_col="time", value_col="outcome", treatment={"A": 1, "B": 1})


def test_staggered_did_requires_numeric_time(panel):
    panel["time"] = pd.to_datetime(panel["time"], unit="D")
    with pytest.raises(TypeError, match="StaggeredDID requires a numeric time column"):
        StaggeredDID(data=panel, unit_col="unit", time_col="time", value_col="outcome", treatment="treated")
