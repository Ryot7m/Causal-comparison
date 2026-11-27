from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def fit_prediction_model(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    categorical_columns: list[str] | None = None,
    target_scores: dict | None = None,
    random_state: int = 123) -> dict:
    categorical_columns = categorical_columns or []

    if not feature_columns:
        raise ValueError("feature_columnsを1列以上指定してください。")
    if target_column in feature_columns:
        raise ValueError("目的変数を特徴量に含めることはできません。")
    if not set(categorical_columns).issubset(feature_columns):
        raise ValueError("categorical_columnsはfeature_columnsの部分集合にしてください。")

    required = set(feature_columns) | {target_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {sorted(missing)}")

    X = data[feature_columns].copy()
    raw_y = data[target_column]

    if raw_y.isna().any():
        raise ValueError("目的変数に欠損があります。")

    # 順序カテゴリを数値スコアにする場合の例:
    # {"low": 1.0, "middle": 2.0, "high": 3.0}
    if target_scores is not None:
        y = raw_y.map(target_scores)
        if y.isna().any():
            unknown = raw_y[y.isna()].unique().tolist()
            raise ValueError(f"target_scoresにない値があります: {unknown}")
    else:
        y = pd.to_numeric(raw_y, errors="raise")

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    transformers = []

    if numeric_columns:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    preprocess = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # MAEがどれだけ増えるかを特徴量単位に計算
    importance = permutation_importance(model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=10, random_state=random_state, n_jobs=1)

    feature_importance = pd.DataFrame({
        "feature": feature_columns,
        "mae_increase_mean": importance.importances_mean,
        "mae_increase_std": importance.importances_std,
    }).sort_values("mae_increase_mean", ascending=False)

    return {
        "model": model,  # 別途ファイル保存する。DBのJSONには入れない
        "metrics": {
            "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r2": float(r2_score(y_test, predicted)),
        },
        "feature_importance": feature_importance.to_dict(orient="records")
    }
    
def train_prediction_model(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    model_path: str,
    target_scores: dict | None = None,
    random_state: int = 123,
) -> dict:
    if not feature_columns:
        raise ValueError("特徴量を1列以上指定してください。")
    if target_column in feature_columns:
        raise ValueError("目的変数を特徴量に含めることはできません。")
    if not set(categorical_columns).issubset(feature_columns):
        raise ValueError("カテゴリ列は特徴量の中から指定してください。")

    required = set(feature_columns) | {target_column}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {sorted(missing)}")

    X = data[feature_columns].copy()
    raw_y = data[target_column]

    if raw_y.isna().any():
        raise ValueError("目的変数に欠損があります。")

    # 例: {"low": 1.0, "middle": 2.0, "high": 3.0}
    if target_scores is not None:
        y = raw_y.map(target_scores)
        if y.isna().any():
            raise ValueError("target_scoresにない目的変数の値があります。")
    else:
        y = pd.to_numeric(raw_y, errors="raise")

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    preprocess = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    # 独立した行を想定したランダム分割。
    # 時系列や同一ユーザーの複数行がある場合は分割方法を変える。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )

    importance_rows = [
        {
            "feature": feature,
            "mae_increase_mean": float(mean),
            "mae_increase_std": float(std),
        }
        for feature, mean, std in zip(
            feature_columns,
            importance.importances_mean,
            importance.importances_std,
        )
    ]
    importance_rows.sort(
        key=lambda row: row["mae_increase_mean"],
        reverse=True,
    )

    artifact = Path(model_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact)

    return {
        "metrics": {
            "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        },
        "feature_importance": importance_rows,
    }


def predict_with_model(
    data: pd.DataFrame,
    *,
    model_path: str,
    feature_columns: list[str],
) -> list[float]:
    missing = set(feature_columns) - set(data.columns)
    if missing:
        raise ValueError(f"CSVに必要な特徴量がありません: {sorted(missing)}")

    model = joblib.load(model_path)
    predicted = model.predict(data[feature_columns])
    return [float(value) for value in predicted]