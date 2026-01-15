from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

DATASET = PROC_DIR / "banner_dataset_compact_v2.csv"
MODEL_OUT = MODELS_DIR / "revenue_model_v2.joblib"
FEATS_OUT = MODELS_DIR / "model_features_v2.txt"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)

    df = df.dropna(subset=["revenue"]).copy()

    feature_cols = [
        "avg_views_best",
        "top_views_avg",
        "meta_relevance_mean",
        "top_meta_score",
        "n_characters",
        "is_rerun",
    ]

    df["meta_x_views"] = df["top_meta_score"] * np.log1p(df["top_views_avg"])
    df["meta_x_rerun"] = df["top_meta_score"] * df["is_rerun"]
    feature_cols += ["meta_x_views", "meta_x_rerun"]

    X = df[feature_cols].copy()
    y = np.log1p(df["revenue"].astype(float))

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-3, 2, 50),
            cv=5,
            random_state=42,
            max_iter=50000,
        ))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scorers = {
        "mae": make_scorer(lambda yt, yp: mean_absolute_error(np.expm1(yt), np.expm1(yp)), greater_is_better=False),
        "rmse": make_scorer(lambda yt, yp: -rmse(np.expm1(yt), np.expm1(yp)), greater_is_better=False),
    }

    scores = cross_validate(pipe, X, y, cv=cv, scoring=scorers, return_train_score=False)

    mae = -scores["test_mae"]
    rmse_vals = -scores["test_rmse"]

    print(f"CV MAE (USD):  mean={mae.mean():,.0f}  std={mae.std():,.0f}")
    print(f"CV RMSE (USD): mean={rmse_vals.mean():,.0f}  std={rmse_vals.std():,.0f}")

    pipe.fit(X, y)

    payload = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "train_rows": int(len(df)),
        "dataset": str(DATASET),
    }
    joblib.dump(payload, MODEL_OUT)

    FEATS_OUT.write_text("\n".join(feature_cols), encoding="utf-8")

    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved features: {FEATS_OUT} ({len(feature_cols)} cols)")

if __name__ == "__main__":
    main()
