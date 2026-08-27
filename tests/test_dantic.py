import pytest
from pydantic import ValidationError

from app.dantic import AnalysisRequest


def test_analysis_request_accepts_distinct_column_roles(config):
    request = AnalysisRequest.model_validate(config)

    assert request.outcome.column == "Q4_1"
    assert request.segment.column == "Q7_4"


@pytest.mark.parametrize(
    "overlapping_column",
    [
        "Q4_1",  # outcome
        "Q7_4",  # segment
        "Q2_9"   # treatment source
    ]
)
def test_analysis_request_rejects_covariate_role_overlap(config, overlapping_column):
    config["covariates"]["columns"] = [
        "x1",
        overlapping_column
    ]

    with pytest.raises(ValidationError, match="covariatesには"):
        AnalysisRequest.model_validate(config)


@pytest.mark.parametrize(
    (
        "section",
        "field",
        "overlapping_column"
    )
    [
        ("segment", "column", "Q4_1"),
        ("treatment", "source_column", "Q4_1"),
        ("treatment", "source_column", "Q7_4")
    ]
)
def test_analysis_request_rejects_main_role_overlap(config, section, field, overlapping_column):
    config[section][field] = overlapping_column

    with pytest.raises(ValidationError, match="異なる列"):
        AnalysisRequest.model_validate(config)
        
def test_analysis_request_rejects_binary_treatment_overlap(config):
    config["treatment"] = {
        "mode": "binary_column",
        "column": "Q4_1",
        "treated_value": 1,
        "control_value": 0
    }

    with pytest.raises(ValidationError, match="異なる列"):
        AnalysisRequest.model_validate(config)
        
def test_analysis_request_accepts_fill_strategy(config):
    config["missing"] = {
        "strategy": "fill",
        "fill_values": {
            "x1": -1.0
        }
    }

    request = AnalysisRequest.model_validate(config)

    assert request.missing.strategy == "fill"
    assert request.missing.fill_values == {
        "x1": -1.0
    }
    
def test_analysis_request_rejects_fill_without_values(config):
    config["missing"] = {
        "strategy": "fill"
    }

    with pytest.raises(ValidationError, match="fill_valuesを指定"):
        AnalysisRequest.model_validate(config)
        
@pytest.mark.parametrize(
    "strategy",
    [
        "drop",
        "zero",
    ],
)
def test_analysis_request_rejects_fill_values_for_non_fill(config, strategy):
    config["missing"] = {
        "strategy": strategy,
        "fill_values": {
            "x1": 0
        }
    }

    with pytest.raises(ValidationError, match="fillの場合だけ"):
        AnalysisRequest.model_validate(config)
        
def test_analysis_request_rejects_unknown_missing_strategy(config):
    config["missing"] = {
        "strategy": "mean"
    }

    with pytest.raises(ValidationError) as exc_info:
        AnalysisRequest.model_validate(config)

    assert any(
        error["type"] == "literal_error"
        and error["loc"][-1] == "strategy"
        for error in exc_info.value.errors()
    )
    
def test_analysis_request_rejects_equal_binary_values(config):
    config["treatment"] = {
        "mode": "binary_column",
        "column": "treatment",
        "treated_value": "yes",
        "control_value": "yes"
    }

    with pytest.raises(ValidationError,match="treated_valueとcontrol_value"):
        AnalysisRequest.model_validate(config)
        
@pytest.mark.parametrize(
    "quantile",
    [
        0,
        1,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf")
    ])

def test_analysis_request_rejects_invalid_quantile(config, quantile):
    config["treatment"]["quantile"] = quantile

    with pytest.raises(ValidationError) as exc_info:
        AnalysisRequest.model_validate(config)

    assert any(
        error["loc"][-1] == "quantile"
        for error in exc_info.value.errors())
    
def test_analysis_request_rejects_unknown_treated_when(config):
    config["treatment"]["treated_when"] = "equal"

    with pytest.raises(ValidationError) as exc_info:
        AnalysisRequest.model_validate(config)

    assert any(
        error["type"] == "literal_error"
        and error["loc"][-1] == "treated_when"
        for error in exc_info.value.errors()
    )
    
def test_analysis_request_rejects_unknown_treatment_mode(config):
    config["treatment"] = {
        "mode": "unsupported",
        "column": "treatment"
    }

    with pytest.raises(ValidationError) as exc_info:
        AnalysisRequest.model_validate(config)

    assert any(
        error["type"] == "union_tag_invalid"
        for error in exc_info.value.errors()
    )
    
def test_analysis_request_rejects_extra_treatment_field(config):
    config["treatment"]["unexpected"] = "value"

    with pytest.raises(ValidationError) as exc_info:
        AnalysisRequest.model_validate(config)

    assert any(
        error["type"] == "extra_forbidden"
        and error["loc"][-1] == "unexpected"
        for error in exc_info.value.errors()
    )