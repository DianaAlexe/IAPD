from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

YOUTUBE_METRICS = RAW_DIR / "youtube_metrics.csv"
BANNER_REVENUE = RAW_DIR / "banner_revenue.csv"
RELEVANCE_ROLE = RAW_DIR / "genshin_relevance_role.csv"

CHAR_OUT = PROC_DIR / "character_features.csv"
BANNER_FULL_OUT = PROC_DIR / "banner_dataset_full.csv"
BANNER_SLIM_OUT = PROC_DIR / "banner_dataset.csv" 



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


def is_rerun_banner(banner: str) -> int:
    b = _norm(banner)
    return int(("rerun" in b) or ("re-run" in b) or ("(rerun)" in b))


def split_banner_into_characters(banner: str) -> List[str]:
    """
    Supports:
      "Albedo & Eula"
      "Shenhe / Xiao"
      "Venti (Rerun)"
      "Ayato + Venti"
    """
    b = _norm(banner)
    b = re.sub(r"\s*\([^)]*\)\s*", " ", b).strip()
    b = b.replace(" and ", " & ")
    parts = re.split(r"\s*&\s*|\s*/\s*|\s*\+\s*", b)
    parts = [canonical_name(p.strip()) for p in parts if p and p.strip()]
    return parts

def _lower_cols(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc not in out:
            out[lc] = c
    return out


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    mapping = _lower_cols(df)
    for cand in candidates:
        if cand in mapping:
            return mapping[cand]
    return None


def detect_banner_and_revenue_columns(br: pd.DataFrame) -> Tuple[str, str]:
    banner_candidates = [
        "banner", "banners", "banner_name", "bannername",
        "name", "title", "character", "characters",
        "unit", "units"
    ]
    revenue_candidates = [
        "revenue", "rev", "sales", "earning", "earnings", "income",
        "gross", "amount", "usd", "value"
    ]

    banner_col = pick_column(br, banner_candidates)
    revenue_col = pick_column(br, revenue_candidates)

    if banner_col and revenue_col:
        return banner_col, revenue_col

    lc_map = _lower_cols(br)
    if not banner_col:
        for lc, orig in lc_map.items():
            if "banner" in lc or "name" in lc or "character" in lc or "title" in lc:
                banner_col = orig
                break
    if not revenue_col:
        for lc, orig in lc_map.items():
            if "revenue" in lc or "sale" in lc or "earning" in lc or "income" in lc:
                revenue_col = orig
                break

    if not banner_col or not revenue_col:
        raise ValueError(
            "Nu pot detecta coloanele din banner_revenue.csv.\n"
            f"Coloane găsite: {list(br.columns)}\n"
            "Am nevoie de ceva gen banner/name/character și revenue/sales."
        )
    return banner_col, revenue_col

def build_character_features(youtube_path: Path, relevance_path: Path) -> pd.DataFrame:
    ym = pd.read_csv(youtube_path)
    rel = pd.read_csv(relevance_path)

    ym["character"] = ym["character"].map(canonical_name)

    numeric = ["views", "likes", "comments", "like_ratio", "engagement_score"]
    for c in numeric:
        ym[c] = pd.to_numeric(ym[c], errors="coerce")

    pivot = ym.pivot_table(
        index="character",
        columns="video_type",
        values=numeric,
        aggfunc="max"
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    def best_of(row, base: str):
        a = row.get(f"{base}_demo", np.nan)
        b = row.get(f"{base}_teaser", np.nan)
        return np.nanmax([a, b])

    for base in numeric:
        pivot[f"{base}_best"] = pivot.apply(lambda r: best_of(r, base), axis=1)

    rel = rel.copy()
    if "name" in rel.columns:
        rel["character"] = rel["name"].map(canonical_name)
    elif "character" in rel.columns:
        rel["character"] = rel["character"].map(canonical_name)
    else:
        raise ValueError("genshin_relevance_role.csv trebuie să aibă coloana 'name' sau 'character'.")

    rel_score_col = "relevance_mean_adj" if "relevance_mean_adj" in rel.columns else "relevance_mean"
    if rel_score_col not in rel.columns:
        raise ValueError("genshin_relevance_role.csv trebuie să aibă 'relevance_mean_adj' sau 'relevance_mean'.")

    rel[rel_score_col] = pd.to_numeric(rel[rel_score_col], errors="coerce")

    role_map = {
        "main dps": "relevance_main_dps",
        "sub-dps": "relevance_sub-dps",
        "sub dps": "relevance_sub-dps",
        "support": "relevance_support",
    }
    rel["role_norm"] = rel["role"].map(_norm)
    rel["role_key"] = rel["role_norm"].map(role_map)

    rel_role = rel.dropna(subset=["role_key"]).pivot_table(
        index="character",
        columns="role_key",
        values=rel_score_col,
        aggfunc="max"
    ).reset_index()

    for col in ["relevance_main_dps", "relevance_sub-dps", "relevance_support"]:
        if col not in rel_role.columns:
            rel_role[col] = np.nan

    rel_role["relevance_max"] = rel_role[["relevance_main_dps", "relevance_sub-dps", "relevance_support"]].max(axis=1, skipna=True)
    rel_role["relevance_mean"] = rel_role[["relevance_main_dps", "relevance_sub-dps", "relevance_support"]].mean(axis=1, skipna=True)

    cf = pivot.merge(rel_role, on="character", how="outer")
    return cf


def aggregate_for_banner(char_rows: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    num_cols = char_rows.select_dtypes(include=[np.number]).columns.tolist()

    for c in num_cols:
        out[f"{c}__sum"] = float(char_rows[c].sum(skipna=True))
        out[f"{c}__mean"] = float(char_rows[c].mean(skipna=True))
        out[f"{c}__max"] = float(char_rows[c].max(skipna=True))

    out["n_characters"] = float(char_rows["character_norm"].nunique())
    return out


def safe_argmax(series: pd.Series) -> Optional[str]:
    if series is None or len(series) == 0:
        return None
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return None
    return series.loc[s.idxmax()]


def build_banner_dataset_full_and_slim(
    banner_revenue_path: Path,
    character_features: pd.DataFrame,
    meta_weight: float = 0.70,
    views_weight: float = 0.30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    br0 = pd.read_csv(banner_revenue_path)
    banner_col, revenue_col = detect_banner_and_revenue_columns(br0)

    br = br0.rename(columns={banner_col: "banner", revenue_col: "revenue"}).copy()
    br["banner"] = br["banner"].astype(str)
    br["revenue"] = pd.to_numeric(br["revenue"], errors="coerce")
    br["is_rerun"] = br["banner"].map(is_rerun_banner)

    cf = character_features.copy()
    if "character" not in cf.columns:
        raise ValueError("character_features.csv trebuie să aibă coloana 'character'.")
    cf["character_norm"] = cf["character"].map(canonical_name)

    rows_full = []

    for _, r in br.iterrows():
        banner = r["banner"]
        revenue = r["revenue"]
        rerun = int(r["is_rerun"])

        chars = split_banner_into_characters(banner)
        sub = cf[cf["character_norm"].isin(chars)].copy()

        feats = aggregate_for_banner(sub) if not sub.empty else {"n_characters": float(len(chars))}

        top_views_character = None
        top_views_value = np.nan
        if not sub.empty and "views_best" in sub.columns:
            tmp = sub[["character_norm", "views_best"]].copy()
            tmp["views_best"] = pd.to_numeric(tmp["views_best"], errors="coerce")
            if tmp["views_best"].notna().any():
                idx = tmp["views_best"].idxmax()
                top_views_character = str(tmp.loc[idx, "character_norm"])
                top_views_value = float(tmp.loc[idx, "views_best"])

        top_meta_character = None
        top_meta_value = np.nan
        if not sub.empty and "relevance_mean" in sub.columns:
            tmp = sub[["character_norm", "relevance_mean"]].copy()
            tmp["relevance_mean"] = pd.to_numeric(tmp["relevance_mean"], errors="coerce")
            if tmp["relevance_mean"].notna().any():
                idx = tmp["relevance_mean"].idxmax()
                top_meta_character = str(tmp.loc[idx, "character_norm"])
                top_meta_value = float(tmp.loc[idx, "relevance_mean"])

        avg_views = float(pd.to_numeric(sub["views_best"], errors="coerce").mean()) if (not sub.empty and "views_best" in sub.columns) else np.nan
        meta_relevance_avg = float(pd.to_numeric(sub["relevance_mean"], errors="coerce").mean()) if (not sub.empty and "relevance_mean" in sub.columns) else np.nan
        rows_full.append({
            **feats,
            "banner": banner,
            "banner_characters": ", ".join(chars),
            "revenue": float(revenue) if pd.notna(revenue) else np.nan,
            "is_rerun": rerun,

            "avg_views": avg_views,
            "top_views_character": top_views_character,
            "top_views_value": top_views_value,

            "meta_relevance_avg": meta_relevance_avg,
            "top_meta_character": top_meta_character,
            "top_meta_value": top_meta_value,
        })

    bd_full = pd.DataFrame(rows_full)

    vcol = "top_views_value"
    mcol = "top_meta_value"

    vmin = float(pd.to_numeric(bd_full[vcol], errors="coerce").min(skipna=True)) if vcol in bd_full.columns else 0.0
    vmax = float(pd.to_numeric(bd_full[vcol], errors="coerce").max(skipna=True)) if vcol in bd_full.columns else 1.0
    mmin = float(pd.to_numeric(bd_full[mcol], errors="coerce").min(skipna=True)) if mcol in bd_full.columns else 0.0
    mmax = float(pd.to_numeric(bd_full[mcol], errors="coerce").max(skipna=True)) if mcol in bd_full.columns else 1.0

    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax == vmin: vmax = vmin + 1.0
    if not np.isfinite(mmin): mmin = 0.0
    if not np.isfinite(mmax) or mmax == mmin: mmax = mmin + 1.0

    bd_full["top_views_norm"] = (pd.to_numeric(bd_full[vcol], errors="coerce") - vmin) / (vmax - vmin)
    bd_full["top_meta_norm"] = (pd.to_numeric(bd_full[mcol], errors="coerce") - mmin) / (mmax - mmin)

    bd_full["views_strength"] = float(views_weight) * bd_full["top_views_norm"]
    bd_full["meta_strength"] = float(meta_weight) * bd_full["top_meta_norm"]

    bd_full["dominant_character"] = np.where(
        bd_full["meta_strength"] >= bd_full["views_strength"],
        bd_full["top_meta_character"],
        bd_full["top_views_character"],
    )
    bd_full["dominant_reason"] = np.where(
        bd_full["meta_strength"] >= bd_full["views_strength"],
        "meta",
        "views",
    )

    slim_cols = [
        "banner",
        "banner_characters",
        "is_rerun",
        "avg_views",
        "top_views_character",
        "revenue",
        "meta_relevance_avg",
        "top_meta_character",
        "dominant_character",
        "dominant_reason",
    ]
    for c in slim_cols:
        if c not in bd_full.columns:
            bd_full[c] = np.nan

    bd_slim = bd_full[slim_cols].copy()

    return bd_full, bd_slim


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    cf = build_character_features(YOUTUBE_METRICS, RELEVANCE_ROLE)
    cf.to_csv(CHAR_OUT, index=False, encoding="utf-8")
    print(f"Saved: {CHAR_OUT} ({len(cf)} rows)")

    bd_full, bd_slim = build_banner_dataset_full_and_slim(BANNER_REVENUE, cf)

    bd_full.to_csv(BANNER_FULL_OUT, index=False, encoding="utf-8")
    print(f"Saved: {BANNER_FULL_OUT} ({len(bd_full)} rows)")

    bd_slim.to_csv(BANNER_SLIM_OUT, index=False, encoding="utf-8")
    print(f"Saved: {BANNER_SLIM_OUT} ({len(bd_slim)} rows)")

    if "n_characters" in bd_full.columns:
        dual = bd_full[bd_full["banner"].astype(str).str.contains(r"&|/|\+", regex=True)]
        bad = dual[dual["n_characters"] < 2]
        if len(bad) > 0:
            print("\nWARNING: Unele bannere multi-character au n_characters < 2 (alias lipsă sau nume diferit):")
            print(bad[["banner", "banner_characters", "n_characters"]].head(30).to_string(index=False))

    missing_rev = bd_full[bd_full["revenue"].isna()] if "revenue" in bd_full.columns else pd.DataFrame()
    if len(missing_rev) > 0:
        print(f"\nWARNING: {len(missing_rev)} rânduri au revenue NaN (format diferit/valori lipsă).")


if __name__ == "__main__":
    main()
