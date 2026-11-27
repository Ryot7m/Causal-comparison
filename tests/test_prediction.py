from math import isfinite
import pandas as pd
import pytest
from predict.prediction import train_prediction_model, predict_with_model
pytestmark = pytest.mark.no_database

def make_prediction_data() -> pd.DataFrame:
    return pd.DataFrame({
        "age": list(range(30)),
        "region": ["east", "west", "north"] * 10,
        "recommendation": ["low", "middle", "high"] * 10
    })

def test_train_and_predict_with_saved_model(tmp_path):
    data = make_prediction_data()
    features = ["age", "region"]
    model_path = tmp_path / "model.joblib"

    result = train_prediction_model(
        data,
        target_column="recommendation",
        feature_columns=features,
        categorical_columns=["region"],
        target_scores={"low": 1.0, "middle": 2.0, "high": 3.0},
        random_state=123,
        model_path=str(model_path)
    )

    predictions = predict_with_model(
        data[features].iloc[:3],
        model_path=str(model_path),
        feature_columns=features
    )

    assert model_path.exists()
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["rmse"] >= 0
    assert {row["feature"] for row in result["feature_importance"]} == set(features)
    assert len(predictions) == 3
    assert all(isfinite(value) for value in predictions)
    
def test_train_rejects_target_value_without_score_mapping(tmp_path):
    data = make_prediction_data()
    data.loc[0, "recommendation"] = "unknown"

    with pytest.raises(ValueError, match="target_scoresにない"):
        train_prediction_model(
            data,
            target_column="recommendation",
            feature_columns=["age", "region"],
            categorical_columns=["region"],
            target_scores={"low": 1.0, "middle": 2.0, "high": 3.0},
            model_path=str(tmp_path / "should-not-exist.joblib"),
        )


def test_predict_rejects_missing_feature_column(tmp_path):
    with pytest.raises(ValueError, match="必要な特徴量"):
        predict_with_model(
            pd.DataFrame({"age": [35]}),
            model_path=str(tmp_path / "unused.joblib"),
            feature_columns=["age", "region"],
        )