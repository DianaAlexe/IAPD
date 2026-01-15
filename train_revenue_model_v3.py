from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

DATA_PATH = PROC_DIR / "banner_dataset_full.csv"
MODEL_PATH = MODELS_DIR / "revenue_model_v3.joblib"
FEATS_PATH = MODELS_DIR / "model_features_v3.txt"


DROP_NON_FEATURES = {
    "banner",
    "banner_characters",
    "top_views_character",
    "top_meta_character",
    "dominant_character",
    "dominant_reason",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in DROP_NON_FEATURES:
            continue
        if c == "revenue":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "revenue" not in df.columns:
        raise ValueError(f"{DATA_PATH} trebuie să aibă coloana 'revenue'.")

    df = df.dropna(subset=["revenue"]).copy()
    y = df["revenue"].astype(float).values

    y_log = np.log1p(y)

    feature_cols = pick_feature_cols(df)
    if not feature_cols:
        raise ValueError("Nu am găsit nicio coloană numerică de features în dataset.")

    X = df[feature_cols].copy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=3,
        max_iter=800,
        l2_regularization=0.2,
        min_samples_leaf=6,
        random_state=42,
        early_stopping=True,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    maes, rmses = [], []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y_log[train_idx], y_log[test_idx]

        pipe.fit(Xtr, ytr)
        pred_log = pipe.predict(Xte)
        pred = np.expm1(pred_log)
        true = np.expm1(yte)

        maes.append(mean_absolute_error(true, pred))
        rmses.append(rmse(true, pred))

    print(f"CV MAE (USD):  mean={np.mean(maes):,.0f}  std={np.std(maes):,.0f}")
    print(f"CV RMSE (USD): mean={np.mean(rmses):,.0f}  std={np.std(rmses):,.0f}")

    pipe.fit(X, y_log)

    payload = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "target": "log1p(revenue)",
        "data_path": str(DATA_PATH),
    }
    joblib.dump(payload, MODEL_PATH)

    with open(FEATS_PATH, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved features: {FEATS_PATH} ({len(feature_cols)} cols)")


if __name__ == "__main__":
    main()
