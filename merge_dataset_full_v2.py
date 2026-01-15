from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

YOUTUBE_METRICS = RAW_DIR / "youtube_metrics.csv"
RELEVANCE_ROLE = RAW_DIR / "genshin_relevance_role.csv"
BANNER_REVENUE = RAW_DIR / "banner_revenue.csv"

CHAR_OUT = PROC_DIR / "character_features.csv"
BANNER_OUT = PROC_DIR / "banner_dataset_full.csv"

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

def _lower_cols(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc not in out:
            out[lc] = c
    return out


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    m = _lower_cols(df)
    for cand in candidates:
        if cand in m:
            return m[cand]
    return None


def detect_banner_and_revenue_columns(br: pd.DataFrame) -> Tuple[str, str]:
    banner_candidates = [
        "banner", "banner_name", "bannername", "name", "title",
        "character", "characters", "unit", "units"
    ]
    revenue_candidates = [
        "revenue", "rev", "sales", "earning", "earnings", "income",
        "gross", "amount", "usd", "value"
    ]

    banner_col = pick_column(br, banner_candidates)
    revenue_col = pick_column(br, revenue_candidates)

    if banner_col and revenue_col:
        return banner_col, revenue_col

    lc = _lower_cols(br)
    if not banner_col:
        for k, orig in lc.items():
            if "banner" in k or "name" in k or "character" in k or "title" in k:
                banner_col = orig
                break
    if not revenue_col:
        for k, orig in lc.items():
            if "revenue" in k or "sale" in k or "earning" in k or "income" in k:
                revenue_col = orig
                break

    if not banner_col or not revenue_col:
        raise ValueError(
            "Nu pot detecta coloanele banner/revenue în banner_revenue.csv.\n"
            f"Cols={list(br.columns)}"
        )
    return banner_col, revenue_col

def is_rerun_banner(banner: str) -> int:
    b = _norm(banner)
    return int(("rerun" in b) or ("re-run" in b) or ("(rerun)" in b))


def split_banner_into_characters(banner: str) -> List[str]:
    """
    Supports:
      "Albedo & Eula"
      "Shenhe / Xiao"
      "Ayato + Venti"
      "Venti (Rerun)"
      "Arlecchino & Clorinde(Rerun)"  -> clorinde
    """
    b = _norm(banner)
    b = re.sub(r"\s*\([^)]*\)\s*", " ", b).strip()
    b = b.replace(" and ", " & ")
    parts = re.split(r"\s*&\s*|\s*/\s*|\s*\+\s*|,\s*", b)
    parts = [canonical_name(p.strip()) for p in parts if p and p.strip()]
    parts = [p for p in parts if p]
    return parts

def build_character_features(youtube_path: Path, relevance_path: Path) -> pd.DataFrame:
    ym = pd.read_csv(youtube_path)
    rel = pd.read_csv(relevance_path)

    if "character" not in ym.columns:
        raise ValueError(f"youtube_metrics.csv trebuie să aibă coloana 'character'. Cols={list(ym.columns)}")
    if "video_type" not in ym.columns:
        raise ValueError(f"youtube_metrics.csv trebuie să aibă coloana 'video_type'. Cols={list(ym.columns)}")

    ym["character"] = ym["character"].map(canonical_name)

    numeric = ["views", "likes", "comments", "like_ratio", "engagement_score"]
    for c in numeric:
        if c not in ym.columns:
            ym[c] = np.nan
        ym[c] = pd.to_numeric(ym[c], errors="coerce")

    pivot = ym.pivot_table(
        index="character",
        columns="video_type",
        values=numeric,
        aggfunc="max",
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    def best_of(row, base: str):
        a = row.get(f"{base}_demo", np.nan)
        b = row.get(f"{base}_teaser", np.nan)
        return float(np.nanmax([a, b])) if np.isfinite(np.nanmax([a, b])) else np.nan

    for base in numeric:
        pivot[f"{base}_best"] = pivot.apply(lambda r: best_of(r, base), axis=1)

    rel = rel.copy()
    if "name" in rel.columns:
        rel["character"] = rel["name"].map(canonical_name)
    elif "character" in rel.columns:
        rel["character"] = rel["character"].map(canonical_name)
    else:
        raise ValueError("genshin_relevance_role.csv trebuie să aibă 'name' sau 'character'.")

    if "role" not in rel.columns:
        raise ValueError("genshin_relevance_role.csv trebuie să aibă coloana 'role'.")

    score_col = "relevance_mean_adj" if "relevance_mean_adj" in rel.columns else "relevance_mean"
    if score_col not in rel.columns:
        raise ValueError("genshin_relevance_role.csv trebuie să aibă 'relevance_mean_adj' sau 'relevance_mean'.")

    rel[score_col] = pd.to_numeric(rel[score_col], errors="coerce")

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
        values=score_col,
        aggfunc="max",
    ).reset_index()

    for col in ["relevance_main_dps", "relevance_sub-dps", "relevance_support"]:
        if col not in rel_role.columns:
            rel_role[col] = np.nan

    rel_role["meta_relevance_max"] = rel_role[["relevance_main_dps", "relevance_sub-dps", "relevance_support"]].max(axis=1, skipna=True)
    rel_role["meta_relevance_mean"] = rel_role[["relevance_main_dps", "relevance_sub-dps", "relevance_support"]].mean(axis=1, skipna=True)

    cf = pivot.merge(rel_role, on="character", how="outer")

    if "views_best" not in cf.columns:
        cf["views_best"] = np.nan
    if "meta_relevance_mean" not in cf.columns:
        cf["meta_relevance_mean"] = np.nan

    return cf

def aggregate_for_banner(char_rows: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    num_cols = char_rows.select_dtypes(include=[np.number]).columns.tolist()

    for c in num_cols:
        out[f"{c}__sum"] = float(char_rows[c].sum(skipna=True))
        out[f"{c}__mean"] = float(char_rows[c].mean(skipna=True))
        out[f"{c}__max"] = float(char_rows[c].max(skipna=True))

    out["n_characters"] = float(char_rows["character"].nunique())
    return out


def build_banner_dataset_full(banner_revenue_path: Path, character_features: pd.DataFrame) -> pd.DataFrame:
    br = pd.read_csv(banner_revenue_path)
    banner_col, revenue_col = detect_banner_and_revenue_columns(br)

    br = br.rename(columns={banner_col: "banner", revenue_col: "revenue"}).copy()
    br["banner"] = br["banner"].astype(str)
    br["revenue"] = pd.to_numeric(br["revenue"], errors="coerce")

    cf = character_features.copy()
    cf["character_norm"] = cf["character"].map(canonical_name)

    if "views_best" not in cf.columns:
        cf["views_best"] = np.nan
    if "meta_relevance_mean" not in cf.columns:
        cf["meta_relevance_mean"] = np.nan

    rows: List[Dict[str, float | str | int]] = []

    for _, r in br.iterrows():
        banner = str(r["banner"])
        revenue = r["revenue"]

        chars = split_banner_into_characters(banner)
        sub = cf[cf["character_norm"].isin(chars)].copy()

        banner_chars = ", ".join(chars)

        views_map = {}
        meta_map = {}
        for c in chars:
            rr = sub[sub["character_norm"] == c]
            if len(rr) == 0:
                views_map[c] = np.nan
                meta_map[c] = np.nan
            else:
                views_map[c] = float(rr["views_best"].iloc[0]) if pd.notna(rr["views_best"].iloc[0]) else np.nan
                meta_map[c] = float(rr["meta_relevance_mean"].iloc[0]) if pd.notna(rr["meta_relevance_mean"].iloc[0]) else np.nan

        views_vals = np.array([views_map[c] for c in chars], dtype=float) if chars else np.array([], dtype=float)
        meta_vals = np.array([meta_map[c] for c in chars], dtype=float) if chars else np.array([], dtype=float)

        views_mean = float(np.nanmean(views_vals)) if views_vals.size else np.nan
        meta_mean = float(np.nanmean(meta_vals)) if meta_vals.size else np.nan

        top_views_character = None
        top_views_value = np.nan
        if chars and np.isfinite(np.nanmax(views_vals)):
            idx = int(np.nanargmax(views_vals))
            top_views_character = chars[idx]
            top_views_value = float(views_vals[idx])

        top_meta_character = None
        top_meta_value = np.nan
        if chars and np.isfinite(np.nanmax(meta_vals)):
            idx = int(np.nanargmax(meta_vals))
            top_meta_character = chars[idx]
            top_meta_value = float(meta_vals[idx])

        if chars and np.isfinite(top_views_value):
            vmin = float(np.nanmin(views_vals))
            vmax = float(np.nanmax(views_vals))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                top_views_norm = (top_views_value - vmin) / (vmax - vmin)
            else:
                top_views_norm = 0.0
        else:
            top_views_norm = 0.0

        META_WEIGHT = 0.70
        VIEWS_WEIGHT = 0.30

        meta_strength = META_WEIGHT * (top_meta_value if np.isfinite(top_meta_value) else 0.0)
        views_strength = VIEWS_WEIGHT * top_views_norm

        dominant_reason = "meta" if meta_strength >= views_strength else "views"
        dominant_character = (top_meta_character if dominant_reason == "meta" else top_views_character) or ""

        feats = aggregate_for_banner(sub) if not sub.empty else {"n_characters": float(len(chars))}

        feats.update({
            "banner": banner,
            "banner_characters": banner_chars,
            "revenue": float(revenue) if pd.notna(revenue) else np.nan,
            "is_rerun_banner": int(is_rerun_banner(banner)),
            "views_best_mean_banner": views_mean,
            "top_views_character": top_views_character or "",
            "top_views_value": float(top_views_value) if np.isfinite(top_views_value) else np.nan,
            "meta_relevance_mean_banner": meta_mean,
            "top_meta_character": top_meta_character or "",
            "top_meta_value": float(top_meta_value) if np.isfinite(top_meta_value) else np.nan,
            "dominant_character": dominant_character,
            "dominant_reason": dominant_reason,
        })

        rows.append(feats)

    out = pd.DataFrame(rows)

    preferred = [
        "banner",
        "banner_characters",
        "n_characters",
        "is_rerun_banner",
        "views_best_mean_banner",
        "top_views_character",
        "top_views_value",
        "revenue",
        "meta_relevance_mean_banner",
        "top_meta_character",
        "top_meta_value",
        "dominant_character",
        "dominant_reason",
    ]
    keep = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in keep]
    out = out[keep + rest]

    return out


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    cf = build_character_features(YOUTUBE_METRICS, RELEVANCE_ROLE)
    cf.to_csv(CHAR_OUT, index=False, encoding="utf-8")
    print(f"Saved: {CHAR_OUT} ({len(cf)} rows)")

    bd = build_banner_dataset_full(BANNER_REVENUE, cf)
    bd.to_csv(BANNER_OUT, index=False, encoding="utf-8")
    print(f"Saved: {BANNER_OUT} ({len(bd)} rows)")

    missing_rev = bd[bd["revenue"].isna()]
    if len(missing_rev) > 0:
        print(f"WARNING: {len(missing_rev)} rânduri au revenue NaN.")

    multi = bd[bd["banner"].astype(str).str.contains(r"&|/|\+", regex=True)]
    if "n_characters" in multi.columns:
        bad = multi[multi["n_characters"] < 2]
        if len(bad) > 0:
            print("\nWARNING: bannere multi-character cu n_characters < 2 (alias lipsă / parsing):")
            print(bad[["banner", "banner_characters", "n_characters"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
