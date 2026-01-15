from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib


PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

CHAR_FEATURES = PROC_DIR / "character_features.csv"
HISTORY_FULL = PROC_DIR / "banner_dataset.csv"
MODEL_PATH = MODELS_DIR / "revenue_model_v3.joblib"


ALIASES = {
    "arataki itto": "itto",
    "itto": "itto",
    "yumemizuki mizuki": "mizuki",
    "yumemizuki": "mizuki",
    "mizuki": "mizuki",
    "raiden shogun": "raiden",
    "shogun": "raiden",
    "shougun": "raiden",
    "childe": "tartaglia",
    "tartaglia": "tartaglia",
    "yoimia": "yoimiya",
    "yoymia": "yoimiya",
    "arlechino": "arlecchino",
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

def split_csv_list(s: str) -> List[str]:
    s = _norm(s)
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [canonical_name(p) for p in parts if p]


def load_character_features() -> pd.DataFrame:
    cf = pd.read_csv(CHAR_FEATURES)

    if "character" not in cf.columns:
        raise ValueError("character_features.csv trebuie să aibă coloana 'character'.")

    cf["character_norm"] = cf["character"].map(canonical_name)

    # Ensure standard columns exist
    if "views_best" not in cf.columns:
        # fallback if older format exists
        # try to infer
        for cand in ["views_best", "views_best_max", "views_best__max"]:
            if cand in cf.columns:
                cf["views_best"] = pd.to_numeric(cf[cand], errors="coerce")
                break
        if "views_best" not in cf.columns:
            cf["views_best"] = np.nan

    if "meta_relevance_mean" not in cf.columns:
        for cand in ["meta_relevance_mean", "relevance_mean", "relevance_mean__max"]:
            if cand in cf.columns:
                cf["meta_relevance_mean"] = pd.to_numeric(cf[cand], errors="coerce")
                break
        if "meta_relevance_mean" not in cf.columns:
            cf["meta_relevance_mean"] = np.nan

    cf["views_best"] = pd.to_numeric(cf["views_best"], errors="coerce")
    cf["meta_relevance_mean"] = pd.to_numeric(cf["meta_relevance_mean"], errors="coerce")

    return cf


def load_history_full() -> pd.DataFrame:
    if not HISTORY_FULL.exists():
        raise FileNotFoundError(
            f"Lipsește {HISTORY_FULL}. Rulează: python -m processing.merge_dataset_full"
        )
    h = pd.read_csv(HISTORY_FULL)

    if "banner_characters" not in h.columns:
        raise ValueError("banner_dataset_full.csv trebuie să aibă coloana 'banner_characters'.")
    if "revenue" not in h.columns:
        raise ValueError("banner_dataset_full.csv trebuie să aibă coloana 'revenue'.")

    h["revenue"] = pd.to_numeric(h["revenue"], errors="coerce")
    h["banner_characters_norm"] = h["banner_characters"].astype(str).apply(split_csv_list)

    # best effort flag if it exists
    if "is_rerun_banner" not in h.columns:
        h["is_rerun_banner"] = 0
    else:
        h["is_rerun_banner"] = pd.to_numeric(h["is_rerun_banner"], errors="coerce").fillna(0).astype(int)

    return h


def aggregate_features_for_chars(cf: pd.DataFrame, chars_norm: List[str]) -> Dict[str, float]:
    chars_norm_unique = list(dict.fromkeys(chars_norm))
    sub = cf[cf["character_norm"].isin(chars_norm_unique)].copy()

    out: Dict[str, float] = {}

    num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()

    for drop in ["character_norm"]:
        if drop in num_cols:
            num_cols.remove(drop)

    for c in num_cols:
        out[f"{c}__sum"] = float(sub[c].sum(skipna=True))
        out[f"{c}__mean"] = float(sub[c].mean(skipna=True))
        out[f"{c}__max"] = float(sub[c].max(skipna=True))

    out["n_characters"] = float(len(chars_norm_unique))

    views = sub["views_best"].astype(float).values if "views_best" in sub.columns else np.array([], dtype=float)
    metas = sub["meta_relevance_mean"].astype(float).values if "meta_relevance_mean" in sub.columns else np.array([], dtype=float)

    out["views_best_mean_banner"] = float(np.nanmean(views)) if views.size else np.nan
    out["top_views_value"] = float(np.nanmax(views)) if views.size and np.isfinite(np.nanmax(views)) else np.nan

    out["meta_relevance_mean_banner"] = float(np.nanmean(metas)) if metas.size else np.nan
    out["top_meta_value"] = float(np.nanmax(metas)) if metas.size and np.isfinite(np.nanmax(metas)) else np.nan

    out["is_rerun_banner"] = 0.0

    return out

def prev_max_revenue_for_char(history: pd.DataFrame, char_norm: str) -> Optional[float]:
    mask = history["banner_characters_norm"].apply(lambda lst: char_norm in lst)
    sub = history.loc[mask, "revenue"]
    sub = sub.dropna()
    if len(sub) == 0:
        return None
    return float(sub.max())

def detect_rerun_flags(
    history: pd.DataFrame,
    chars_norm: List[str],
    force_rerun: List[str],
    force_debut: List[str],
) -> Dict[str, bool]:
    fr = {canonical_name(x) for x in force_rerun}
    fd = {canonical_name(x) for x in force_debut}

    flags: Dict[str, bool] = {}
    for c in chars_norm:
        if c in fd:
            flags[c] = False
            continue
        if c in fr:
            flags[c] = True
            continue
        flags[c] = (prev_max_revenue_for_char(history, c) is not None)

    return flags

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = x - np.nanmax(x)
    ex = np.exp(x)
    s = np.nansum(ex)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x) / max(1, len(x))
    return ex / s

def compute_character_shares(
    views: List[float],
    metas: List[float],
    meta_weight: float = 0.70,
    views_weight: float = 0.30,
) -> np.ndarray:
    v = np.array(views, dtype=float)
    m = np.array(metas, dtype=float)

    m_norm = np.clip(m / 5.0, 0.0, 1.0)
    m_norm = np.nan_to_num(m_norm, nan=0.0)

    v = np.nan_to_num(v, nan=0.0)
    v_log = np.log1p(v)
    if len(v_log) > 0 and np.max(v_log) > np.min(v_log):
        v_norm = (v_log - np.min(v_log)) / (np.max(v_log) - np.min(v_log))
    else:
        v_norm = np.zeros_like(v_log)

    score = meta_weight * m_norm + views_weight * v_norm
    return softmax(score)


def rerun_multiplier_from_meta(meta: float) -> float:
    if not np.isfinite(meta):
        meta = 2.5
    meta = float(np.clip(meta, 0.0, 5.0))

    # meta=5 -> 1.0
    # meta=0 -> 0.55
    return 0.55 + 0.45 * (meta / 5.0)

def partner_boost(meta: float, partner_best: float) -> float:
    if not np.isfinite(meta):
        meta = 2.5
    if not np.isfinite(partner_best):
        return 0.0

    meta = float(np.clip(meta, 0.0, 5.0))
    partner_best = float(np.clip(partner_best, 0.0, 5.0))

    if partner_best < 4.5:
        return 0.0

    boost = 0.10 * (min(meta, partner_best) / 5.0)
    return float(np.clip(boost, 0.0, 0.10))


def apply_per_character_caps(
    raw_total: float,
    chars: List[str],
    rerun_flags: Dict[str, bool],
    prev_caps: Dict[str, Optional[float]],
    views: Dict[str, float],
    metas: Dict[str, float],
    meta_weight_for_share: float = 0.70,
    views_weight_for_share: float = 0.30,
    debug: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, object]]]:
  
    c_list = chars[:]
    v_list = [views.get(c, np.nan) for c in c_list]
    m_list = [metas.get(c, np.nan) for c in c_list]

    shares = compute_character_shares(v_list, m_list, meta_weight=meta_weight_for_share, views_weight=views_weight_for_share)
    raw_contrib = {c: float(raw_total) * float(shares[i]) for i, c in enumerate(c_list)}

    m_vals = [metas.get(c, np.nan) for c in c_list]
    partner_best_by_char: Dict[str, float] = {}
    for c in c_list:
        others = [x for x in c_list if x != c]
        if not others:
            partner_best_by_char[c] = float("nan")
        else:
            partner_best_by_char[c] = float(np.nanmax([metas.get(o, np.nan) for o in others]))

    final_contrib: Dict[str, float] = {}
    debug_info: Dict[str, Dict[str, object]] = {}

    for c in c_list:
        c_raw = raw_contrib[c]
        is_rerun = bool(rerun_flags.get(c, False))
        prev = prev_caps.get(c, None)

        info = {
            "raw_contrib": c_raw,
            "is_rerun": is_rerun,
            "prev_cap": prev,
            "meta": metas.get(c, np.nan),
            "views": views.get(c, np.nan),
            "share": float(raw_contrib[c] / raw_total) if raw_total > 0 else float("nan"),
        }

        if (not is_rerun) or (prev is None) or (not np.isfinite(prev)):
            final_contrib[c] = c_raw
            info["cap_applied"] = False
            info["cap_reason"] = "no_cap_or_debut_or_no_history"
            debug_info[c] = info
            continue

        mult = rerun_multiplier_from_meta(float(metas.get(c, np.nan)))
        pb = partner_boost(float(metas.get(c, np.nan)), float(partner_best_by_char.get(c, np.nan)))
        allowed = float(prev) * mult * (1.0 + pb)

        meta_c = float(metas.get(c, np.nan)) if np.isfinite(metas.get(c, np.nan)) else 2.5
        meta_p = float(partner_best_by_char.get(c, np.nan)) if np.isfinite(partner_best_by_char.get(c, np.nan)) else 0.0
        if meta_c >= 4.8 and meta_p >= 4.8:
            allowed = max(allowed, float(prev) * 1.00)  # can equal
            allowed = min(allowed, float(prev) * 1.05)  # slight exceed
        else:
            allowed = min(allowed, float(prev) * 1.00)  # cannot exceed prev

        capped = min(c_raw, allowed)

        final_contrib[c] = float(capped)
        info["cap_applied"] = True
        info["multiplier"] = mult
        info["partner_best_meta"] = partner_best_by_char.get(c, np.nan)
        info["partner_boost"] = pb
        info["allowed_cap"] = allowed
        info["cap_reason"] = "rerun_cap_by_meta_and_partner"
        debug_info[c] = info

    final_total = float(sum(final_contrib.values()))
    return final_total, final_contrib, debug_info


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("characters", nargs="+", help='Ex: "Mavuika" "Arlecchino"')
    ap.add_argument("--debug", action="store_true", help="Print debug details")
    ap.add_argument("--force-rerun", nargs="*", default=[], help="Force rerun for specific characters")
    ap.add_argument("--force-debut", nargs="*", default=[], help="Force debut for specific characters (no rerun cap)")
    args = ap.parse_args(argv)

    chars_in = args.characters
    chars_norm = [canonical_name(c) for c in chars_in]
    chars_norm_unique = list(dict.fromkeys(chars_norm))

    cf = load_character_features()
    history = load_history_full()

    views_map: Dict[str, float] = {}
    meta_map: Dict[str, float] = {}

    for c in chars_norm_unique:
        row = cf[cf["character_norm"] == c]
        if len(row) == 0:
            views_map[c] = float("nan")
            meta_map[c] = float("nan")
        else:
            views_map[c] = float(row["views_best"].iloc[0]) if pd.notna(row["views_best"].iloc[0]) else float("nan")
            meta_map[c] = float(row["meta_relevance_mean"].iloc[0]) if pd.notna(row["meta_relevance_mean"].iloc[0]) else float("nan")

    rerun_flags = detect_rerun_flags(
        history=history,
        chars_norm=chars_norm_unique,
        force_rerun=args.force_rerun,
        force_debut=args.force_debut,
    )

    prev_caps: Dict[str, Optional[float]] = {}
    for c in chars_norm_unique:
        prev_caps[c] = prev_max_revenue_for_char(history, c)

    payload = joblib.load(MODEL_PATH)
    pipe = payload["pipeline"]
    feature_cols: List[str] = payload["feature_cols"]

    feats = aggregate_features_for_chars(cf, chars_norm_unique)

    feats["is_rerun_banner"] = float(any(rerun_flags.values()))

    X = pd.DataFrame([{c: feats.get(c, np.nan) for c in feature_cols}])

    pred_log = float(pipe.predict(X)[0])
    raw_pred = float(np.expm1(pred_log))
    raw_pred = max(0.0, raw_pred)

    final_pred, contrib_final, dbg = apply_per_character_caps(
        raw_total=raw_pred,
        chars=chars_norm_unique,
        rerun_flags=rerun_flags,
        prev_caps=prev_caps,
        views=views_map,
        metas=meta_map,
        debug=args.debug,
    )

    print("Characters:", ", ".join(chars_in))
    print("Characters (canonical):", ", ".join(chars_norm_unique))
    print("Predicted revenue RAW (USD):", f"{raw_pred:,.0f}")
    print("Predicted revenue FINAL (USD):", f"{final_pred:,.0f}")

    if args.debug:
        print("\n--- Per-character details ---")
        for c in chars_norm_unique:
            meta = meta_map.get(c, np.nan)
            views = views_map.get(c, np.nan)
            rr = rerun_flags.get(c, False)
            prev = prev_caps.get(c, None)

            v_str = f"{views:,.0f}" if np.isfinite(views) else "NaN"
            m_str = f"{meta:.2f}" if np.isfinite(meta) else "NaN"
            prev_str = f"{prev:,.0f}" if (prev is not None and np.isfinite(prev)) else "None"

            info = dbg.get(c, {})
            share = info.get("share", float("nan"))
            raw_c = info.get("raw_contrib", float("nan"))
            fin_c = contrib_final.get(c, float("nan"))

            print(f"\n[{c}]")
            print(f"  views_best={v_str}  meta_relevance_mean={m_str}")
            print(f"  rerun={rr}  prev_max_revenue={prev_str}")
            print(f"  share={share:.3f}  raw_contrib={raw_c:,.0f}  final_contrib={fin_c:,.0f}")

            if info.get("cap_applied"):
                print(f"  cap_reason={info.get('cap_reason')}")
                print(f"  multiplier={info.get('multiplier'):.3f}")
                print(f"  partner_best_meta={info.get('partner_best_meta')}")
                print(f"  partner_boost={info.get('partner_boost')}")
                print(f"  allowed_cap={float(info.get('allowed_cap')):,.0f}")
            else:
                print(f"  cap_reason={info.get('cap_reason')}")

        print("\n--- Feature sanity (selected) ---")
        for k in ["n_characters", "is_rerun_banner", "views_best_mean_banner", "top_views_value",
                  "meta_relevance_mean_banner", "top_meta_value"]:
            if k in feats:
                v = feats[k]
                if isinstance(v, float) and np.isfinite(v):
                    if "views" in k or "top_views" in k:
                        print(f"{k}: {v:,.0f}")
                    else:
                        print(f"{k}: {v:.3f}")
                else:
                    print(f"{k}: {v}")

if __name__ == "__main__":
    main()
