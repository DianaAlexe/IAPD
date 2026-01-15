from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

DATA = Path("data/processed/banner_dataset_full.csv")
OUT = Path("models/revenue_model_v5_base.joblib")

DECAY = 0.35  

def is_rerun_banner(banner: str) -> int:
    b = str(banner or "")
    return 1 if re.search(r"\(rerun\)", b, flags=re.I) else 0

def main():
    df = pd.read_csv(DATA)

    if "revenue" not in df.columns:
        raise ValueError("banner_dataset.csv trebuie să aibă coloana 'revenue'")
    if "banner" not in df.columns:
        raise ValueError("banner_dataset.csv trebuie să aibă coloana 'banner'")

    df["is_rerun"] = df["banner"].map(is_rerun_banner).astype(int)
    df["rerun_weight"] = np.where(df["is_rerun"] == 1, DECAY, 1.0)

    y = (df["revenue"].astype(float) / df["rerun_weight"]).clip(lower=0.0)
    y_log = np.log1p(y)

    X = df.drop(columns=["revenue"])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    drop_cols = {"is_rerun", "rerun_weight"}
    feature_cols = [c for c in num_cols if c not in drop_cols]

    X = X[feature_cols].copy()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.05, l1_ratio=0.15, random_state=42)),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y_log, cv=cv,
        scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error"),
        n_jobs=1
    )

    mae = -scores["test_neg_mean_absolute_error"]
    rmse = -scores["test_neg_root_mean_squared_error"]

    print(f"CV MAE (log space):  mean={mae.mean():.4f}  std={mae.std():.4f}")
    print(f"CV RMSE (log space): mean={rmse.mean():.4f}  std={rmse.std():.4f}")

    pipe.fit(X, y_log)

    payload = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "decay": float(DECAY),
        "target": "log1p(revenue / rerun_weight)"
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT)
    print(f"Saved: {OUT}  (features={len(feature_cols)})")

if __name__ == "__main__":
    main()
