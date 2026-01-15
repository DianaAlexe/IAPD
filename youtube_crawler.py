from __future__ import annotations

import os
import re
import json
import math
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from yt_dlp import YoutubeDL

GENSHIN_OFFICIAL_CHANNEL_URL = os.getenv(
    "GENSHIN_OFFICIAL_CHANNEL_URL",
    "https://www.youtube.com/@GenshinImpact"
)

VIDEO_TYPES = [
    ("teaser", "Character Teaser"),
    ("demo", "Character Demo"),
]

TEASER_PATTERNS = [
    "character teaser",
    "overture teaser",
    "teaser",
]

DEMO_PATTERNS_STRICT = [
    "character demo",
]
INDEX_CACHE_PATH = Path("data/raw/youtube_official_index_flat.json")

OUT_CSV = Path("data/raw/youtube_metrics.csv")

PER_VIDEO_SLEEP_SEC = 0.05

SOCKET_TIMEOUT = float(os.getenv("YT_DLP_SOCKET_TIMEOUT", "20"))

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_int(x) -> int:
    try:
        if x is None:
            return 0
        if isinstance(x, bool):
            return int(x)
        return int(float(x))
    except Exception:
        return 0


def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(s: object) -> str:
    """Safe normalize to lowercase string."""
    if s is None:
        return ""
    if s is Ellipsis:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    return s.lower().strip()


def title_has_character(title: str, character: str) -> bool:
    return norm(character) in norm(title)


def title_matches_any(title: str, patterns: List[str]) -> bool:
    t = norm(title)
    return any(p in t for p in patterns)

ALIASES = {
    "arataki itto": "itto",
    "itto": "itto",

    "yumemizuki mizuki": "mizuki",
    "mizuki": "mizuki",

    "raiden shogun": "raiden",
    "shogun": "raiden",
    "shougun": "raiden",   

    "childe": "tartaglia",
    "tartaglia": "tartaglia",

    "traveler (anemo)": "traveler (anemo)",
    "traveler (geo)": "traveler (geo)",
    "traveler (electro)": "traveler (electro)",
    "traveler (dendro)": "traveler (dendro)",
    "traveler (hydro)": "traveler (hydro)",
    "traveler (pyro)": "traveler (pyro)",
}

def canonical_character(name: str) -> str:
    n = norm(name)
    n = n.replace("’", "'")
    n = re.sub(r"\s+", " ", n).strip()
    return ALIASES.get(n, n)

def _ydl_opts_flat() -> dict:
    """
    Flat index: fast, does not request detailed formats.
    """
    return {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "socket_timeout": SOCKET_TIMEOUT,
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "retries": 3,
        "fragment_retries": 3,
        "noplaylist": False,
        "concurrent_fragment_downloads": 1,
    }


def _ydl_opts_videoinfo() -> dict:
    """
    Detailed per-video info: we use it to get view/like/comment counts.
    """
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": SOCKET_TIMEOUT,
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "retries": 3,
        "fragment_retries": 3,
        "noplaylist": True,
        "concurrent_fragment_downloads": 1,
    }


def build_flat_index(channel_url: str) -> List[dict]:
    """
    Returns a list of flat entries for the channel uploads.
    """
    with YoutubeDL(_ydl_opts_flat()) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = []
    if isinstance(info, dict):
        if "entries" in info and info["entries"]:
            for e in info["entries"]:
                if not e:
                    continue
                if isinstance(e, dict) and "entries" in e and e["entries"]:
                    entries.extend([x for x in e["entries"] if x])
                else:
                    entries.append(e)
    return entries


def save_index(entries: List[dict], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(path: Path) -> Optional[List[dict]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_official_index(channel_url: str, force_rebuild: bool = False) -> List[dict]:
    """
    Use cached flat index if available; otherwise build it.
    """
    if not force_rebuild:
        cached = load_index(INDEX_CACHE_PATH)
        if cached:
            return cached

    print("Building flat index from channel (fast)...")
    entries = build_flat_index(channel_url)
    print(f"Flat index entries: {len(entries)}")
    save_index(entries, INDEX_CACHE_PATH)
    return entries


def extract_video_id(entry: dict) -> Optional[str]:
    """
    Flat entries can provide id, url, or webpage_url.
    """
    if not entry:
        return None
    vid = entry.get("id")
    if vid:
        return str(vid)
    url = entry.get("url") or entry.get("webpage_url") or ""
    m = re.search(r"v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    if isinstance(url, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,}", url.strip()):
        return url.strip()
    return None


@dataclass
class VideoMeta:
    video_id: str
    title: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int


def fetch_video_meta(video_id: str) -> Optional[VideoMeta]:
    """
    Fetch per-video statistics using yt-dlp.
    """
    with YoutubeDL(_ydl_opts_videoinfo()) as ydl:
        vinfo = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    if not vinfo:
        return None

    title = vinfo.get("title") or ""
    upload_date = vinfo.get("upload_date")  
    if upload_date and len(upload_date) == 8:
        published_at = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    else:
        published_at = vinfo.get("timestamp")
        published_at = str(published_at) if published_at else ""

    return VideoMeta(
        video_id=video_id,
        title=title,
        published_at=published_at,
        view_count=safe_int(vinfo.get("view_count")),
        like_count=safe_int(vinfo.get("like_count")),
        comment_count=safe_int(vinfo.get("comment_count")),
    )


def pick_best_from_index(index_entries: List[dict], character: str, vtype: str) -> Optional[dict]:
    """
    Pick the best candidate from flat index by title rules.
    Returns the flat entry dict (needs meta fetch later).
    """
    if not index_entries:
        return None

    ch = canonical_character(character)

    candidates = [e for e in index_entries if title_has_character(e.get("title", ""), ch)]
    if not candidates:
        candidates = [e for e in index_entries if title_has_character(e.get("title", ""), character)]
    if not candidates:
        return None

    if vtype == "teaser":
        strict = [e for e in candidates if title_matches_any(e.get("title", ""), TEASER_PATTERNS)]
        if strict:
            return strict[0]  
        return candidates[0]

    if vtype == "demo":
        strict = [e for e in candidates if title_matches_any(e.get("title", ""), DEMO_PATTERNS_STRICT)]
        if strict:
            return strict[0]
        return None

    return candidates[0]


def crawl_youtube_for_characters(
    characters: List[str],
    out_csv: Path = OUT_CSV,
    channel_url: str = GENSHIN_OFFICIAL_CHANNEL_URL,
    force_reindex: bool = False,
) -> pd.DataFrame:
    """
    Produces rows per character per video_type (teaser/demo).
    """
    ensure_dir(out_csv.parent)

    index_entries = get_official_index(channel_url, force_rebuild=force_reindex)

    usable = []
    for e in index_entries:
        if not isinstance(e, dict):
            continue
        if not e.get("title"):
            continue
        vid = extract_video_id(e)
        if not vid:
            continue
        usable.append({**e, "video_id": vid})

    rows = []
    for ch in characters:
        ch_canon = canonical_character(ch)

        for vtype, keyword in VIDEO_TYPES:
            chosen = pick_best_from_index(usable, ch_canon, vtype)

            if not chosen:
                rows.append({
                    "character": ch_canon,
                    "video_type": vtype,
                    "query": f'{keyword} "{ch_canon}"',
                    "video_id": None,
                    "title": None,
                    "published_at": None,
                    "views": None,
                    "likes": None,
                    "comments": None,
                    "like_ratio": None,
                    "engagement_score": None,
                    "crawled_at": utc_now_iso(),
                })
                continue

            vid = chosen["video_id"]

            meta = fetch_video_meta(vid)
            time.sleep(PER_VIDEO_SLEEP_SEC)

            if not meta:
                rows.append({
                    "character": ch_canon,
                    "video_type": vtype,
                    "query": f'{keyword} "{ch_canon}"',
                    "video_id": vid,
                    "title": chosen.get("title"),
                    "published_at": None,
                    "views": None,
                    "likes": None,
                    "comments": None,
                    "like_ratio": None,
                    "engagement_score": None,
                    "crawled_at": utc_now_iso(),
                })
                continue

            views = meta.view_count
            likes = meta.like_count
            comments = meta.comment_count
            like_ratio = (likes / views) if views else 0.0
            engagement = ((likes + comments) / views) if views else 0.0

            rows.append({
                "character": ch_canon,
                "video_type": vtype,
                "query": f'{keyword} "{ch_canon}"',
                "video_id": meta.video_id,
                "title": meta.title,
                "published_at": meta.published_at,
                "views": views,
                "likes": likes,
                "comments": comments,
                "like_ratio": like_ratio,
                "engagement_score": engagement,
                "crawled_at": utc_now_iso(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved: {out_csv} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    characters = [
        "Albedo","Alhaitham","Aloy","Arataki Itto","Arlecchino","Ayaka","Ayato","Baizhu",
        "Chasca","Childe","Chiori","Citlali","Clorinde","Columbina","Cyno","Dehya","Diluc",
        "Durin","Eula","Flins","Furina","Ganyu","Hu Tao","Iansan","Ineffa","Jean","Kazuha",
        "Keqing","Klee","Kokomi","Lauma","Lan Yan","Lyney","Mavuika","Mizuki","Mona",
        "Mualani","Nahida","Navia","Nefer","Neuvillette","Nilou","Qiqi","Raiden","Shenhe",
        "Skirk","Sigewinne","Tartaglia","Tighnari","Varesa","Venti","Wanderer","Wriothesley",
        "Xianyun","Xiao","Xilonen","Yae Miko","Yelan","Yoimiya","Zhongli"
    ]
    crawl_youtube_for_characters(characters)
