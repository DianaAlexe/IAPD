import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

GAME8_URL = "https://game8.co/games/Genshin-Impact/archives/297465"
GENSHIN_GG_URL = "https://genshin.gg/tier-list/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tierlist-crawler/1.0; +https://example.com/bot)"
}

TIER_SCORE_GAME8 = {"SS": 5, "S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
TIER_SCORE_GG = {"SS": 5, "S": 4, "A": 3, "B": 2, "C": 1, "D": 0}

ROLE_NORMALIZE = {
    "Main DPS": "Main DPS",
    "DPS": "Main DPS",
    "Sub DPS": "Sub-DPS",
    "Sub-DPS": "Sub-DPS",
    "Support": "Support"
}

def fetch_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def parse_game8() -> pd.DataFrame:
    soup = fetch_soup(GAME8_URL)

    main_text = soup.find(string=re.compile(r"\bMain Tier List\b", re.I))
    if not main_text:
        raise RuntimeError("Could not find 'Main Tier List' marker on Game8 page.")

    container = main_text.find_parent()
    table = container.find_next("table")
    if not table:
        raise RuntimeError("Could not find tier list table after 'Main Tier List' on Game8 page.")

    rows = table.find_all("tr")
    if len(rows) < 2:
        raise RuntimeError("Game8 tier table seems empty/unexpected.")

    data = []
    for tr in rows[1:]:
        th = tr.find("th")
        tds = tr.find_all("td")
        if not th or len(tds) < 3:
            continue

        tier_img = th.find("img")
        if not tier_img:
            continue

        tier_alt = (tier_img.get("alt") or "").strip() 
        m_tier = re.match(r"^(SS|S|A|B|C|D)\s+Tier$", tier_alt)
        if not m_tier:
            continue
        tier = m_tier.group(1)

        col_roles = ["Main DPS", "Sub-DPS", "Support"]
        for idx, role in enumerate(col_roles):
            imgs = tds[idx].find_all("img")
            for im in imgs:
                alt = (im.get("alt") or "").strip()
                m = re.match(r"^Genshin\s*-\s*(.+?)\s+(DPS|Sub-DPS|Support)\s+Rank$", alt)
                if not m:
                    continue
                name = m.group(1).strip()
                role_from_alt = m.group(2).strip()
                role_norm = ROLE_NORMALIZE.get(role_from_alt, role)

                data.append({
                    "source": "game8",
                    "name": name,
                    "role": role_norm,
                    "tier": tier,
                    "score": TIER_SCORE_GAME8.get(tier, None),
                    "url": GAME8_URL
                })

    df = pd.DataFrame(data).drop_duplicates(subset=["source", "name", "role"])
    return df

def parse_genshin_gg() -> pd.DataFrame:
    soup = fetch_soup(GENSHIN_GG_URL)

    dropzone = soup.select_one(".tierlist-dropzone")
    if not dropzone:
        raise RuntimeError("Could not find .tierlist-dropzone on genshin.gg")

    data = []
    for row in dropzone.select(".dropzone-row"):
        tier_el = row.select_one(".dropzone-title")
        if not tier_el:
            continue
        tier = tier_el.get_text(strip=True).upper()  

        for a in row.select("a.tierlist-portrait"):
            name_el = a.select_one(".tierlist-name")
            role_el = a.select_one(".tierlist-role")
            if not name_el or not role_el:
                continue

            name = name_el.get_text(strip=True)
            role_raw = role_el.get_text(strip=True)
            role_norm = ROLE_NORMALIZE.get(role_raw, role_raw)

            href = a.get("href") or ""
            full_url = urljoin(GENSHIN_GG_URL, href)

            data.append({
                "source": "genshin.gg",
                "name": name,
                "role": role_norm,
                "tier": tier,
                "score": TIER_SCORE_GG.get(tier, None),
                "url": full_url
            })

    df = pd.DataFrame(data).drop_duplicates(subset=["source", "name", "role"])
    return df

def build_relevance_table() -> pd.DataFrame:
    df1 = parse_game8()
    time.sleep(1.0)
    df2 = parse_genshin_gg()

    all_df = pd.concat([df1, df2], ignore_index=True)

    pivot = all_df.pivot_table(
        index=["name", "role"],
        columns="source",
        values="score",
        aggfunc="max"
    ).reset_index()

    score_cols = [c for c in pivot.columns if c in ("game8", "genshin.gg")]
    pivot["relevance_mean"] = pivot[score_cols].mean(axis=1, skipna=True)

    tier_pivot = all_df.pivot_table(
        index=["name", "role"],
        columns="source",
        values="tier",
        aggfunc="first"
    ).reset_index()

    out = pivot.merge(tier_pivot, on=["name", "role"], suffixes=("_score", "_tier"))

    out = out.sort_values(["relevance_mean", "name"], ascending=[False, True]).reset_index(drop=True)
    return out

if __name__ == "__main__":
    df = build_relevance_table()
    print(df.head(30))
    df.to_csv("genshin_relevance_mean.csv", index=False, encoding="utf-8")
    print("\nSaved: genshin_relevance_mean.csv")
