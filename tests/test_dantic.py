import pytest
from pydantic import ValidationError

from app.dantic import AnalysisRequest


def test_analysis_request_accepts_distinct_column_roles(
    config,
):
    request = AnalysisRequest.model_validate(
        config,
    )

    assert request.outcome.column == "Q4_1"
    assert request.segment.column == "Q7_4"


@pytest.mark.parametrize(
    "overlapping_column",
    [
        "Q4_1",  # outcome
        "Q7_4",  # segment
        "Q2_9",  # treatment source
    ],
)
def test_analysis_request_rejects_covariate_role_overlap(
    config,
    overlapping_column,
):
    config["covariates"]["columns"] = [
        "x1",
        overlapping_column,
    ]

    with pytest.raises(
        ValidationError,
        match="covariatesには",
    ):
        AnalysisRequest.model_validate(
            config,
        )


@pytest.mark.parametrize(
    (
        "section",
        "field",
        "overlapping_column",
    ),
    [
        ("segment", "column", "Q4_1"),
        ("treatment", "source_column", "Q4_1"),
        ("treatment", "source_column", "Q7_4"),
    ],
)
def test_analysis_request_rejects_main_role_overlap(
    config,
    section,
    field,
    overlapping_column,
):
    config[section][field] = overlapping_column

    with pytest.raises(
        ValidationError,
        match="異なる列",
    ):
        AnalysisRequest.model_validate(
            config,
        )