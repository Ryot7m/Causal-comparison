import pandas as pd
from predict.prediction import train_prediction_model, predict_with_model


def test_train_and_predict_with_saved_model(tmp_path):
    data = pd.DataFrame({
        "age": list(range(30)),
        "region": ["east", "west"] * 15,
        "recommendation": ["low", "middle", "high"] * 10,
    })
    features = ["age", "region"]
    model_path = tmp_path / "model.joblib"

    result = train_prediction_model(
        data,
        target_column="recommendation",
        feature_columns=features,
        categorical_columns=["region"],
        target_scores={"low": 1.0, "middle": 2.0, "high": 3.0},
        model_path=str(model_path),
    )

    predictions = predict_with_model(
        data[features].iloc[:3],
        model_path=str(model_path),
        feature_columns=features,
    )

    assert model_path.exists()
    assert len(predictions) == 3
    assert len(result["feature_importance"]) == len(features)