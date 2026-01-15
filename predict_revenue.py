from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

CHAR_FEATURES = PROC_DIR / "character_features.csv"

MODEL_PATH = MODELS_DIR / "revenue_model.joblib"

BANNER_FULL_CANDIDATES = [
    PROC_DIR / "banner_dataset_full.csv",
    PROC_DIR / "banner__datasetr_compact_v2.csv",
    PROC_DIR / "banner_dataset.csv",
]

MIN_RERUN_FACTOR = 0.35

DUO_META_COMBINE = "mean"  

ENABLE_RAW_SOFT_CLAMP = False
RAW_CLAMP_Q_LOW = 0.05
RAW_CLAMP_Q_HIGH = 0.95


ALIASES = {
    "arataki itto": "itto",
    "itto": "itto",
    "yumemizuki mizuki": "mizuki",
    "yumemizuki": "mizuki",
    "mizuki": "mizuki",
    "raiden shogun": "raiden",
    "shogun": "raiden",
    "shougun": "raiden",
    "shogun raiden": "raiden",
    "childe": "tartaglia",
    "tartaglia": "tartaglia",
}

def _norm(x: object) -> str:
    if x is None or x is Ellipsis:
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonical_name(name: str) -> str:
    n = _norm(name)
    return ALIASES.get(n, n)

def parse_banner_characters(s: str) -> List[str]:
    t = _norm(s)
    if not t:
        return []
    t = t.replace("&", ",").replace("+", ",").replace("/", ",")
    t = t.replace(" and ", ",")
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return [canonical_name(p) for p in parts if p]

def is_rerun_banner_text(banner: str) -> int:
    b = _norm(banner)
    return int(("rerun" in b) or ("re-run" in b) or ("(rerun)" in b))

def safe_float(x: object) -> float:
    try:
        if x is None or x is Ellipsis:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def aggregate_features_for_chars(cf: pd.DataFrame, chars: List[str]) -> Dict[str, float]:
    chars_norm = [canonical_name(c) for c in chars]
    sub = cf[cf["character_norm"].isin(chars_norm)].copy()

    out: Dict[str, float] = {}
    num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()

    for c in num_cols:
        out[f"{c}__sum"] = float(sub[c].sum(skipna=True)) if not sub.empty else np.nan
        out[f"{c}__mean"] = float(sub[c].mean(skipna=True)) if not sub.empty else np.nan
        out[f"{c}__max"] = float(sub[c].max(skipna=True)) if not sub.empty else np.nan

    out["n_characters"] = float(len(set(chars_norm)))
    return out


def load_banner_history() -> Tuple[pd.DataFrame | None, str]:
    for p in BANNER_FULL_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df, str(p)
            except Exception:
                continue
    return None, "not_found"

def compute_prev_revenue_cap(
    bd_full: pd.DataFrame,
    chars: List[str],
) -> Tuple[float, str]:
    """
    Prefer non-rerun history; else any history.
    """
    chars_norm = [canonical_name(c) for c in chars]

    if "revenue" not in bd_full.columns:
        return np.nan, "no_revenue_col"

    banner_chars_col = "banner_characters" if "banner_characters" in bd_full.columns else None
    banner_col = "banner" if "banner" in bd_full.columns else None
    is_rerun_col = "is_rerun" if "is_rerun" in bd_full.columns else None

    bd = bd_full.copy()
    bd["revenue"] = pd.to_numeric(bd["revenue"], errors="coerce")

    def row_chars(row) -> List[str]:
        if banner_chars_col is not None:
            return parse_banner_characters(row.get(banner_chars_col, ""))
        if banner_col is not None:
            return parse_banner_characters(row.get(banner_col, ""))
        return []

    def match_row(row) -> bool:
        parts = row_chars(row)
        return any(p in chars_norm for p in parts)

    matched = bd[bd.apply(match_row, axis=1)].copy()
    matched = matched[matched["revenue"].notna()]

    if matched.empty:
        return np.nan, "no_history"

    if is_rerun_col is not None:
        matched["is_rerun"] = pd.to_numeric(matched[is_rerun_col], errors="coerce").fillna(0).astype(int)
    elif banner_col is not None:
        matched["is_rerun"] = matched[banner_col].astype(str).map(is_rerun_banner_text).astype(int)
    else:
        matched["is_rerun"] = 0

    non = matched[matched["is_rerun"] == 0]
    if not non.empty:
        return float(non["revenue"].max()), "cap_from_non_rerun"
    return float(matched["revenue"].max()), "cap_from_any_history"

def get_char_meta(cf: pd.DataFrame, char: str) -> float:
    cn = canonical_name(char)
    sub = cf[cf["character_norm"] == cn]
    if sub.empty:
        return np.nan

    for c in ["relevance_mean", "relevance_mean_adj", "relevance_max"]:
        if c in sub.columns:
            return safe_float(sub.iloc[0][c])
    return np.nan

def meta_effective_for_banner(meta_list: List[float]) -> float:
    metas = [m for m in meta_list if np.isfinite(m)]
    if not metas:
        return np.nan
    if len(metas) == 1:
        return metas[0]
    if DUO_META_COMBINE == "max":
        return max(metas)
    return float(np.mean(metas))

def rerun_factor_from_meta(meta_eff: float) -> float:
    """
    Linear scale:
      meta=5 -> 1.0
      meta=0 -> MIN_RERUN_FACTOR
    """
    if not np.isfinite(meta_eff):
        return MIN_RERUN_FACTOR
    meta_eff = float(np.clip(meta_eff, 0.0, 5.0))
    return float(MIN_RERUN_FACTOR + (1.0 - MIN_RERUN_FACTOR) * (meta_eff / 5.0))

def apply_rerun_cap_strict_meta(
    pred: float,
    prev_cap: float,
    meta_list: List[float],
) -> Tuple[float, str, float, float]:
    """
    FINAL <= prev_cap * factor(meta)
    Equality (factor=1) only when meta==5.
    Never exceed prev cap (or scaled cap).
    """
    if not np.isfinite(prev_cap):
        return pred, "no_cap_no_history", np.nan, np.nan

    meta_eff = meta_effective_for_banner(meta_list)
    factor = rerun_factor_from_meta(meta_eff)
    cap_value = prev_cap * factor

    return min(pred, cap_value), "cap_strict_meta", meta_eff, cap_value

def soft_clamp_raw(pred_raw: float, bd_full: pd.DataFrame | None) -> Tuple[float, str]:
    if not ENABLE_RAW_SOFT_CLAMP:
        return pred_raw, "raw_no_clamp"
    if bd_full is None or "revenue" not in bd_full.columns:
        return pred_raw, "raw_no_clamp_no_history"

    rev = pd.to_numeric(bd_full["revenue"], errors="coerce").dropna()
    if rev.empty:
        return pred_raw, "raw_no_clamp_empty_history"

    lo = float(rev.quantile(RAW_CLAMP_Q_LOW))
    hi = float(rev.quantile(RAW_CLAMP_Q_HIGH))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pred_raw, "raw_no_clamp_bad_quantiles"

    clamped = float(np.clip(pred_raw, lo, hi))
    if clamped != pred_raw:
        return clamped, f"raw_clamped_to_q{RAW_CLAMP_Q_LOW:.2f}-{RAW_CLAMP_Q_HIGH:.2f}"
    return pred_raw, "raw_in_range"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="Apply rerun caps/business rules")
    parser.add_argument("--print-raw", action="store_true", help="Print RAW prediction too")
    parser.add_argument("characters", nargs="+", help='Character names, e.g. "Kazuha" "Itto"')
    args = parser.parse_args()

    chars_input = args.characters
    chars_norm = [canonical_name(c) for c in chars_input]

    # Load character features
    cf = pd.read_csv(CHAR_FEATURES)
    if "character" not in cf.columns:
        raise ValueError("character_features.csv trebuie să aibă coloana 'character'.")
    cf["character_norm"] = cf["character"].map(canonical_name)

    # Load model
    payload = joblib.load(MODEL_PATH)
    pipe = payload["pipeline"]
    feature_cols = payload["feature_cols"]

    feats = aggregate_features_for_chars(cf, chars_input)
    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in feature_cols}])

    pred_log = float(pipe.predict(X)[0])
    pred_raw = float(np.expm1(pred_log))

    bd_full, history_path = load_banner_history()
    pred_raw2, raw_clamp_reason = soft_clamp_raw(pred_raw, bd_full)

    meta_list = [get_char_meta(cf, c) for c in chars_input]

    pred_final = pred_raw2
    cap_reason = "no_cap"
    prev_cap = np.nan
    cap_source = "n/a"
    meta_eff = np.nan
    cap_value = np.nan

    if args.rerun:
        if bd_full is not None:
            prev_cap, cap_source = compute_prev_revenue_cap(bd_full, chars_input)

        pred_final, cap_reason, meta_eff, cap_value = apply_rerun_cap_strict_meta(
            pred=pred_raw2,
            prev_cap=prev_cap,
            meta_list=meta_list,
        )

    # Print
    print("Characters:", ", ".join(chars_input))
    print("Characters (canonical):", ", ".join(chars_norm))
    print("Rerun:", "YES" if args.rerun else "NO")

    for ch, m in zip(chars_input, meta_list):
        ms = f"{m:.2f}" if np.isfinite(m) else "NA"
        print(f" - {ch}: meta_relevance_mean={ms}")

    if args.rerun:
        if np.isfinite(prev_cap):
            print(f"Prev cap (USD): {prev_cap:,.0f}")
        else:
            print("Prev cap (USD): NA")
        print("Prev cap source:", cap_source)
        print("History file:", history_path if bd_full is not None else "not_found")

        me = f"{meta_eff:.2f}" if np.isfinite(meta_eff) else "NA"
        print("meta_effective:", me)
        if np.isfinite(cap_value):
            print(f"Scaled cap (USD): {cap_value:,.0f}")

    if args.print_raw:
        print(f"Predicted revenue RAW (USD): {pred_raw:,.0f}")
        if pred_raw2 != pred_raw:
            print(f"Predicted revenue RAW(clamped) (USD): {pred_raw2:,.0f}")
        print("RAW clamp reason:", raw_clamp_reason)

    print(f"Predicted revenue FINAL (USD): {pred_final:,.0f}")
    print("Cap reason:", cap_reason)

if __name__ == "__main__":
    main()
