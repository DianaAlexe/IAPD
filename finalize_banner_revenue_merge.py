from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

REAL_CSV = Path("data/processed/revenue_per_character.csv")
APPROX_CSV = Path("data/processed/revenue_per_character_approx.csv")

OUT_FINAL = Path("data/processed/revenue_per_character_final.csv")
OUT_AUDIT = Path("data/processed/revenue_per_character_final_audit.csv")

MAX_REASONABLE_DAYS = 30
MIN_DAYS = 1


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _days_inclusive(start: pd.Series, end: pd.Series) -> pd.Series:
    sd = pd.to_datetime(start, errors="coerce")
    ed = pd.to_datetime(end, errors="coerce")
    return (ed - sd).dt.days + 1


def _normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    if "character_name" in df.columns:
        df["character_name"] = (
            df["character_name"]
            .astype(str)
            .str.replace(r"\s*&\s*", " & ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


def _ensure_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    for col in ["character_name", "element", "start_date", "end_date", "revenue_total", "days_observed"]:
        if col not in df.columns:
            df[col] = np.nan

    df = _normalize_names(df)
    df["start_date"] = _to_date(df["start_date"])
    df["end_date"] = _to_date(df["end_date"])

    df["revenue_total"] = pd.to_numeric(df["revenue_total"], errors="coerce")
    df["days_observed"] = pd.to_numeric(df["days_observed"], errors="coerce")

    computed_days = _days_inclusive(df["start_date"], df["end_date"])
    bad_days = df["days_observed"].isna() | (df["days_observed"] <= 0) | (df["days_observed"] != computed_days)
    df.loc[bad_days, "days_observed"] = computed_days[bad_days]

    if "source" not in df.columns:
        df["source"] = kind
    else:
        df["source"] = df["source"].fillna(kind)

    if "crawled_at" not in df.columns:
        df["crawled_at"] = pd.Timestamp.utcnow().isoformat()
    else:
        df["crawled_at"] = df["crawled_at"].fillna(pd.Timestamp.utcnow().isoformat())

    return df


def split_long_banners(approx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dacă un banner din approx e prea lung (ex: 52 zile), cel mai probabil a înghițit bannere între timp.
    Îl spargem folosind următorul start_date pentru același character_name din datasetul de approx (sortat cronologic).
    """
    df = approx_df.copy()
    df = df.sort_values(["start_date", "end_date", "character_name"]).reset_index(drop=True)
    next_start = pd.to_datetime(df["start_date"]).shift(-1)
    df["_next_start"] = next_start.dt.date

    out_rows = []
    for _, row in df.iterrows():
        sd = row["start_date"]
        ed = row["end_date"]
        days = row["days_observed"]

        if pd.isna(sd) or pd.isna(ed) or pd.isna(days):
            out_rows.append(row.drop(labels=["_next_start"]))
            continue

        if days <= MAX_REASONABLE_DAYS:
            out_rows.append(row.drop(labels=["_next_start"]))
            continue

        ns = row["_next_start"]
        if ns and sd < ns <= ed:
            first_end = (pd.to_datetime(ns) - pd.Timedelta(days=1)).date()
            first_days = (pd.to_datetime(first_end) - pd.to_datetime(sd)).days + 1

            rev = row["revenue_total"]
            if pd.isna(rev) or rev < 0:
                rev_first = rev
            else:
                rev_first = float(rev) * (first_days / float(days))

            r1 = row.copy()
            r1["end_date"] = first_end
            r1["days_observed"] = first_days
            r1["revenue_total"] = rev_first
            r1["source"] = "approx_split_long_banner"

            out_rows.append(r1.drop(labels=["_next_start"]))
        else:
            r = row.copy()
            r["source"] = "approx_long_banner_unresolved"
            out_rows.append(r.drop(labels=["_next_start"]))

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["start_date", "end_date", "character_name"]).reset_index(drop=True)
    return out


def merge_real_with_approx(real_df: pd.DataFrame, approx_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge pe cheie: (character_name, start_date, end_date).
    Regula:
      - dacă există REAL -> păstrezi REAL
      - dacă lipsește REAL -> iei APPROX
    """
    key = ["character_name", "start_date", "end_date"]

    real_df = real_df.copy()
    approx_df = approx_df.copy()

    real_df["_kind"] = "REAL"
    approx_df["_kind"] = "APPROX"

    merged = real_df.merge(
        approx_df,
        on=key,
        how="outer",
        suffixes=("_real", "_approx"),
        indicator=True,
    )

    def pick(col: str):
        real_col = f"{col}_real"
        approx_col = f"{col}_approx"
        if real_col in merged.columns and approx_col in merged.columns:
            return merged[real_col].combine_first(merged[approx_col])
        if real_col in merged.columns:
            return merged[real_col]
        if approx_col in merged.columns:
            return merged[approx_col]
        return np.nan

    final = pd.DataFrame({
        "character_name": merged["character_name"],
        "element": pick("element"),
        "start_date": merged["start_date"],
        "end_date": merged["end_date"],
        "revenue_total": pick("revenue_total"),
        "days_observed": pick("days_observed"),
        "source": pick("source"),
        "crawled_at": pick("crawled_at"),
    })

    audit = pd.DataFrame({
        "character_name": merged["character_name"],
        "start_date": merged["start_date"],
        "end_date": merged["end_date"],
        "merge_flag": merged["_merge"],
        "revenue_real": merged.get("revenue_total_real", np.nan),
        "revenue_approx": merged.get("revenue_total_approx", np.nan),
        "picked_source": final["source"],
    })

    final = final.dropna(subset=["character_name", "start_date", "end_date"], how="any")

    final["days_observed"] = pd.to_numeric(final["days_observed"], errors="coerce")
    final = final[(final["days_observed"].isna()) | (final["days_observed"] >= MIN_DAYS)]

    final = final.sort_values(["start_date", "end_date", "character_name"]).reset_index(drop=True)
    audit = audit.sort_values(["start_date", "end_date", "character_name"]).reset_index(drop=True)

    return final, audit


def main():
    if not REAL_CSV.exists():
        raise FileNotFoundError(f"Missing REAL CSV: {REAL_CSV}")

    if not APPROX_CSV.exists():
        raise FileNotFoundError(
            f"Missing APPROX CSV: {APPROX_CSV}\n"
            f"Tip: pune fișierul de aproximări aici sau modifică variabila APPROX_CSV."
        )

    real = pd.read_csv(REAL_CSV)
    approx = pd.read_csv(APPROX_CSV)

    real = _ensure_columns(real, "real_flourish")
    approx = _ensure_columns(approx, "approx_ennead_alloc")

    approx2 = split_long_banners(approx)

    final, audit = merge_real_with_approx(real, approx2)

    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_FINAL, index=False)
    audit.to_csv(OUT_AUDIT, index=False)

    print(f"Saved FINAL:  {OUT_FINAL}  (rows={len(final)})")
    print(f"Saved AUDIT:  {OUT_AUDIT}  (rows={len(audit)})")

    print("\nCounts by merge_flag:")
    print(audit["merge_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
