import pandas as pd
import pytest

from app.analysis import AnalysisConfig


@pytest.fixture
def analysis_config():
    return AnalysisConfig(
        treatment_col="Treatment",
        outcome_col="outcome",
        segment_col="segment",
        state_col="state",
        threshold=0.5,
        confounder_cols=["x1"],
        missing_type="zero",
        outcome_levels=[1, 2, 3],
        score_values=[1.0, 2.0, 3.0],
        reverse_score_max={},
        exclude_cols=[],
        exclude_conditions=[],
    )


@pytest.fixture
def analysis_data():
    return pd.DataFrame({
        "outcome": [1, 2, 3, 1],
        "segment": [1, 1, 2, 2],
        "state": [1.0, 2.0, 3.0, 4.0],
        "x1": [10.0, 20.0, 30.0, 40.0],
    })