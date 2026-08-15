import pandas as pd
import pytest

from app.analysis import AnalysisConfig
from app.dantic import QuantileTreatment

@pytest.fixture
def analysis_config():
    return AnalysisConfig(
        treatment=QuantileTreatment(
            mode="quantile",
            source_column="state",
            quantile=0.5,
            treated_when="ge"
        ),
        treatment_col="Treatment",
        outcome_col="outcome",
        segment_col="segment",
        confounder_cols=["x1"],
        missing_type="zero",
        outcome_levels=[1, 2, 3],
        score_values=[1.0, 2.0, 3.0]
    )


@pytest.fixture
def analysis_data():
    return pd.DataFrame({
        "outcome": [1, 2, 3, 1],
        "segment": [1, 1, 2, 2],
        "state": [1.0, 2.0, 3.0, 4.0],
        "x1": [10.0, 20.0, 30.0, 40.0],
    })