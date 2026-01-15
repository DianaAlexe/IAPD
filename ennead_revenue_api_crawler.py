from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

from extraction.utils import ensure_dir

API_BASE = "https://api.ennead.cc"
WEB_ORIGIN = "https://revenue.ennead.cc"

URL_TOKEN = f"{API_BASE}/api/auth/token"
URL_CONFIG = f"{API_BASE}/api/config"
URL_REVENUE = f"{API_BASE}/api/revenue"
URL_SOURCES = f"{API_BASE}/api/sources"

RAW_DIR = Path("data/raw/ennead_api")
OUT_DIR = Path("data/processed")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

def polite_sleep():
    time.sleep(0.35)

def get_client_token() -> str:
    return str(uuid.uuid4())

def token_headers(client_token: str, request_path: str) -> dict[str, str]:
    return {
        "User-Agent": UA,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": WEB_ORIGIN,
        "Referer": f"{WEB_ORIGIN}/revenue",
        "X-Client-Token": client_token,
        "X-Request-Path": request_path,
    }

def authed_headers(client_token: str, request_path: str, signature: str, timestamp: str) -> dict[str, str]:
    return {
        "User-Agent": UA,
        "Accept": "application/json",
        "Origin": WEB_ORIGIN,
        "Referer": f"{WEB_ORIGIN}/revenue",
        "X-Client-Token": client_token,
        "X-Request-Path": request_path,
        "X-Signature": signature,
        "X-Timestamp": timestamp,
    }

def fetch_token(session: requests.Session, client_token: str, request_path: str) -> dict[str, str]:
    polite_sleep()
    h = token_headers(client_token, request_path)

    r = session.post(URL_TOKEN, headers=h, json={}, timeout=30)
    r.raise_for_status()
    j = r.json()

    if "signature" not in j or "timestamp" not in j:
        raise RuntimeError(f"Unexpected token response: {j}")

    return {"signature": str(j["signature"]), "timestamp": str(j["timestamp"])}

def fetch_authed_json(session: requests.Session, url: str, client_token: str, request_path: str) -> Any:
    tok = fetch_token(session, client_token, request_path)
    h = authed_headers(client_token, request_path, tok["signature"], tok["timestamp"])

    polite_sleep()
    r = session.get(url, headers=h, timeout=60)
    r.raise_for_status()
    return r.json()

def normalize_month_key(k: str) -> str:
    k = str(k).strip()
    if "-" not in k:
        return k
    a, b = k.split("-", 1)
    if len(a) == 2 and len(b) == 4:
        return f"{b}-{a}"
    return k

def flatten_revenue_payload(payload: dict) -> pd.DataFrame:
    games = payload.get("games", [])
    rows: list[dict] = []

    for g in games:
        gid = g.get("id")
        monthly = g.get("monthly_data", {}) or {}

        for month_key, md in monthly.items():
            if not isinstance(md, dict):
                continue

            ios_rev = md.get("ios_revenue")
            and_rev = md.get("android_revenue")
            ios_dl = md.get("ios_downloads")
            and_dl = md.get("android_downloads")

            total_rev = md.get("revenue")
            if total_rev is None:
                try:
                    total_rev = (float(ios_rev) if ios_rev is not None else 0.0) + (float(and_rev) if and_rev is not None else 0.0)
                except Exception:
                    total_rev = None

            total_dl = md.get("downloads")
            if total_dl is None:
                try:
                    total_dl = (float(ios_dl) if ios_dl is not None else 0.0) + (float(and_dl) if and_dl is not None else 0.0)
                except Exception:
                    total_dl = None

            rows.append({
                "game_id": gid,
                "month_key_raw": month_key,
                "month": normalize_month_key(month_key),

                "ios_revenue_usd": ios_rev,
                "android_revenue_usd": and_rev,
                "total_revenue_usd": total_rev,

                "ios_downloads": ios_dl,
                "android_downloads": and_dl,
                "total_downloads": total_dl,

                "trend": md.get("trend"),
            })

    df = pd.DataFrame(rows)

    for c in [
        "ios_revenue_usd","android_revenue_usd","total_revenue_usd",
        "ios_downloads","android_downloads","total_downloads","trend"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["month_date"] = pd.to_datetime(df["month"] + "-01", errors="coerce")
    df = df.sort_values(["game_id", "month_date"])
    return df

def main():
    ensure_dir(str(RAW_DIR))
    ensure_dir(str(OUT_DIR))

    s = requests.Session()

    client_token = get_client_token()
    (RAW_DIR / "client_token.txt").write_text(client_token, encoding="utf-8")

    try:
        sources = fetch_authed_json(s, URL_SOURCES, client_token, "/sources")
        (RAW_DIR / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
        print(f"Saved sources: {RAW_DIR / 'sources.json'}")
    except Exception as e:
        print(f"WARNING: could not fetch /sources ({e})")

    config = fetch_authed_json(s, URL_CONFIG, client_token, "/config")
    (RAW_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved config: {RAW_DIR / 'config.json'}")

    payload = fetch_authed_json(s, URL_REVENUE, client_token, "/revenue")
    (RAW_DIR / "revenue.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved revenue payload: {RAW_DIR / 'revenue.json'}")

    df_all = flatten_revenue_payload(payload)
    out_all = OUT_DIR / "ennead_monthly_revenue_all_games.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8")
    print(f"Saved ALL games monthly revenue: {out_all} (rows={len(df_all)})")

    df_g = df_all[df_all["game_id"].astype(str).str.lower().str.contains("genshin")].copy()
    out_g = OUT_DIR / "ennead_monthly_revenue_genshin.csv"
    df_g.to_csv(out_g, index=False, encoding="utf-8")
    print(f"Saved Genshin monthly revenue: {out_g} (rows={len(df_g)})")

    if df_g["month_date"].notna().any():
        print(f"Genshin coverage: {df_g['month_date'].min().date()} -> {df_g['month_date'].max().date()}")

if __name__ == "__main__":
    main()
