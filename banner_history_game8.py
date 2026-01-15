from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

GAME8_URL = "https://game8.co/games/Genshin-Impact/archives/297500"

RAW_DIR = Path("data/raw/game8")
OUT_DIR = Path("data/processed")

RAW_HTML_PATH = RAW_DIR / "game8_wish_banner_history.html"
SUSPICIOUS_LOG = RAW_DIR / "suspicious_alt_names.txt"
OUT_CSV_PATH = OUT_DIR / "banner_history_game8.csv"

TIMEOUT = 30

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def polite_get(url: str, session: Optional[requests.Session] = None) -> requests.Response:
    s = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    }
    r = s.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def parse_date_yyyy_mm_dd(s: str) -> Optional[str]:
    s = clean_ws(s)
    if not s:
        return None

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.search(r"\b(\d{4})/(\d{2})/(\d{2})\b", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    try:
        dt = datetime.strptime(s.replace(",", ""), "%b %d %Y") 
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        dt = datetime.strptime(s.replace(",", ""), "%B %d %Y")  
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    return None

def extract_featured_from_block(block: BeautifulSoup, banner_name: str) -> Tuple[List[str], List[str]]:
    def clean_name(name: str) -> str:
        name = clean_ws(name)
        return name

    def looks_like_character(name: str) -> bool:
        if not name or len(name) < 2:
            return False
        if re.fullmatch(r"\d+", name):
            return False
        return True

    five: List[str] = []
    four: List[str] = []

    bucket = None 
    five_triggers = ("5-star", "5★", "5 star", "5-star character", "5-star characters", "5★ character", "5★ characters")
    four_triggers = ("4-star", "4★", "4 star", "4-star character", "4-star characters", "4★ character", "4★ characters")

    suspicious = set()

    for node in block.descendants:
        if getattr(node, "name", None) in {"h2", "h3", "h4", "h5", "strong", "b", "p", "span", "div", "li"}:
            txt = clean_ws(node.get_text(" ", strip=True))
            low = txt.lower()
            if any(t in low for t in five_triggers):
                bucket = "5"
            elif any(t in low for t in four_triggers):
                bucket = "4"

        if getattr(node, "name", None) == "img":
            alt = clean_ws(node.get("alt") or "")
            if not alt.lower().startswith("genshin - "):
                continue

            name = clean_name(alt.split("-", 1)[1])

            if banner_name and name.lower() == banner_name.lower():
                continue

            if not looks_like_character(name):
                suspicious.add(name)
                continue

            if bucket == "5":
                five.append(name)
            elif bucket == "4":
                four.append(name)
            else:
                five.append(name)

    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    five = dedup(five)
    four = dedup(four)

    if not four and len(five) >= 3:
        all_names = five[:]
        five = all_names[:2] if len(all_names) >= 5 else all_names[:1]
        four = all_names[len(five):]

    if suspicious:
        append_lines(SUSPICIOUS_LOG, sorted(suspicious))

    return five, four

@dataclass
class BannerRow:
    version: str
    phase: str
    banner_name: str
    start_date: str
    end_date: str
    featured_5: List[str]
    featured_4: List[str]
    source_url: str
    crawled_at: str


def guess_version_phase(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = clean_ws(text)

    m = re.search(r"(?:Version|Ver\.?)\s*([0-9]+\.[0-9]+)\s*(?:Phase|PHASE)\s*([0-9]+)", t, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)

    m = re.search(r"\b([0-9]+\.[0-9]+)\s*(?:Phase|PHASE)\s*([0-9]+)\b", t, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)

    return None, None


def extract_date_range_near(block_text: str) -> Tuple[Optional[str], Optional[str]]:
    t = clean_ws(block_text)

    m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})\s*(?:-|–|~|to)\s*(\d{4}[-/]\d{2}[-/]\d{2})", t)
    if m:
        return parse_date_yyyy_mm_dd(m.group(1)), parse_date_yyyy_mm_dd(m.group(2))

    m = re.search(
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\s*(?:-|–|~|to)\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
        t
    )
    if m:
        return parse_date_yyyy_mm_dd(m.group(1)), parse_date_yyyy_mm_dd(m.group(2))

    return None, None


def parse_game8_character_banners(html: str, source_url: str) -> List[BannerRow]:
    soup = BeautifulSoup(html, "lxml")

    main = soup.select_one("article") or soup.body or soup

    text_all = main.get_text("\n", strip=True)
    if "Wish Banner History" not in text_all and "Banner" not in text_all:
        pass
    headings = main.find_all(["h2", "h3", "h4"])
    sections: List[Tuple[str, str, Any]] = []  # (version, phase, start_node)

    for h in headings:
        v, ph = guess_version_phase(h.get_text(" ", strip=True))
        if v and ph:
            sections.append((v, ph, h))

    if not sections:
        sections = [("unknown", "unknown", main)]

    rows: List[BannerRow] = []
    crawled_at = now_utc_iso()

    def collect_until_next(start_node, stop_nodes_set) -> List[Any]:
        out = []
        for sib in start_node.next_siblings:
            if sib in stop_nodes_set:
                break
            if isinstance(sib, str) and not sib.strip():
                continue
            out.append(sib)
        return out

    stop_nodes = set([sec[2] for sec in sections[1:]]) if len(sections) > 1 else set()

    for idx, (version, phase, start_node) in enumerate(sections):
        if idx < len(sections) - 1:
            stop_set = {sections[idx + 1][2]}
        else:
            stop_set = set()

        nodes = []
        if hasattr(start_node, "next_siblings"):
            nodes = collect_until_next(start_node, stop_set)
        else:
            nodes = [start_node]

        section_html = "\n".join(str(n) for n in nodes if n is not None)
        section_soup = BeautifulSoup(section_html, "lxml")

        candidate_blocks = []
        for blk in section_soup.find_all(["div", "li", "section", "table", "tbody", "tr"]):
            imgs = blk.find_all("img")
            if any((img.get("alt") or "").lower().startswith("genshin - ") for img in imgs):
                if clean_ws(blk.get_text(" ", strip=True)):
                    candidate_blocks.append(blk)

        minimal = []
        for b in candidate_blocks:
            if any((b is not other and b in other.find_all(True)) for other in candidate_blocks):
                continue
            minimal.append(b)
        if minimal:
            candidate_blocks = minimal

        if not candidate_blocks:
            candidate_blocks = [section_soup]

        for blk in candidate_blocks:
            blk_text = clean_ws(blk.get_text("\n", strip=True))
            if not blk_text:
                continue

            lines = [clean_ws(x) for x in blk_text.split("\n") if clean_ws(x)]
            banner_name = None

            for ln in lines[:8]:
                low = ln.lower()
                if "phase" in low and "version" in low:
                    continue
                if re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", ln):
                    continue
                if 3 <= len(ln) <= 60:
                    banner_name = ln
                    break

            if not banner_name:
                banner_name = f"unknown_banner_{version}_{phase}"

            start_date, end_date = extract_date_range_near(blk_text)

            if not start_date or not end_date:
                raw = str(blk)
                sd, ed = extract_date_range_near(raw)
                start_date = start_date or sd
                end_date = end_date or ed

            if not start_date or not end_date:
                continue

            featured_5, featured_4 = extract_featured_from_block(blk, banner_name=banner_name)

            if not featured_5 and not featured_4:
                continue

            rows.append(
                BannerRow(
                    version=version,
                    phase=phase,
                    banner_name=banner_name,
                    start_date=start_date,
                    end_date=end_date,
                    featured_5=featured_5,
                    featured_4=featured_4,
                    source_url=source_url,
                    crawled_at=crawled_at,
                )
            )

    uniq = {}
    for r in rows:
        key = (r.version, r.phase, r.banner_name, r.start_date, r.end_date)
        if key not in uniq:
            uniq[key] = r
    return list(uniq.values())

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    resp = polite_get(GAME8_URL, session=s)
    html = resp.text

    save_text(RAW_HTML_PATH, html)
    print(f"Saved HTML: {RAW_HTML_PATH}")

    rows = parse_game8_character_banners(html, source_url=GAME8_URL)

    if not rows:
        raise RuntimeError(
            "Nu am reușit să parsez niciun banner din Game8. "
            "Verifică data/raw/game8/game8_wish_banner_history.html (poate site structure changed)."
        )

    df = pd.DataFrame([{
        "version": r.version,
        "phase": r.phase,
        "banner_name": r.banner_name,
        "start_date": r.start_date,
        "end_date": r.end_date,
        "featured_5": r.featured_5,
        "featured_4": r.featured_4,
        "source_url": r.source_url,
        "crawled_at": r.crawled_at,
    } for r in rows])

    df["start_date_dt"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.sort_values(["start_date_dt", "version", "phase", "banner_name"], ascending=[True, True, True, True])
    df = df.drop(columns=["start_date_dt"])

    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"Saved parsed banners: {OUT_CSV_PATH}  ({len(df)} rows)")

    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
