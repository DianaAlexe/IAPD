from __future__ import annotations

import json
import re
from pathlib import Path

from extraction.config import USER_AGENT
from extraction.utils import ensure_dir, polite_get

OUT_DIR = Path("data/raw/flourish")
DATA_DIR = OUT_DIR / "data"
REPORT = OUT_DIR / "flourish_data_report.txt"

VIS_IDS = ["7340150", "7347246"]

ENDPOINT_TEMPLATES = [
    "https://public.flourish.studio/visualisation/{vid}/visualisation.json",
    "https://public.flourish.studio/visualisation/{vid}/visualisation",
    "https://public.flourish.studio/visualisation/{vid}/embed", 
]

URL_ANY = re.compile(r"https?://[^\s\"\'<>]+", re.IGNORECASE)

def try_fetch(url: str, headers: dict) -> tuple[bool, str]:
    try:
        resp = polite_get(url, headers=headers, timeout=30, sleep_range=(0.2, 0.8))
        return True, resp.text
    except Exception as e:
        return False, str(e)

def main():
    ensure_dir(str(DATA_DIR))
    headers = {"User-Agent": USER_AGENT}

    lines = []
    lines.append("Flourish data extraction report\n")

    for vid in VIS_IDS:
        lines.append(f"\n=== VISUALISATION {vid} ===")
        for tmpl in ENDPOINT_TEMPLATES:
            url = tmpl.format(vid=vid)
            ok, text = try_fetch(url, headers)
            if ok:
                out = DATA_DIR / f"{vid}__{Path(url).name or 'visualisation'}.txt"
                out.write_text(text, encoding="utf-8", errors="ignore")
                lines.append(f"OK  {url}  -> {out.name} (len={len(text)})")

                try:
                    obj = json.loads(text)
                    out_json = DATA_DIR / f"{vid}__{Path(url).name or 'visualisation'}.json"
                    out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
                    lines.append(f"    Parsed JSON -> {out_json.name}")

                    found_urls = set()
                    def walk(x):
                        if isinstance(x, dict):
                            for k, v in x.items():
                                if isinstance(v, str) and "flourish" in v and ("data" in v or "sheet" in v or "csv" in v or "json" in v):
                                    found_urls.add(v)
                                walk(v)
                        elif isinstance(x, list):
                            for it in x:
                                walk(it)
                        elif isinstance(x, str):
                            for m in URL_ANY.finditer(x):
                                found_urls.add(m.group(0))

                    walk(obj)

                    if found_urls:
                        url_list = DATA_DIR / f"{vid}__discovered_urls.txt"
                        url_list.write_text("\n".join(sorted(found_urls)), encoding="utf-8")
                        lines.append(f"    Discovered URLs -> {url_list.name} ({len(found_urls)})")

                except Exception:
                    pass
            else:
                lines.append(f"FAIL {url}  err={text}")

        embed_file = OUT_DIR / f"visualisation_{vid}_embed.html"
        if embed_file.exists():
            embed_html = embed_file.read_text(encoding="utf-8", errors="ignore")
            urls = set(m.group(0) for m in URL_ANY.finditer(embed_html) if "flourish" in m.group(0).lower())
            if urls:
                out_urls = DATA_DIR / f"{vid}__urls_in_embed.txt"
                out_urls.write_text("\n".join(sorted(urls)), encoding="utf-8")
                lines.append(f"Embed URL scan -> {out_urls.name} ({len(urls)} urls)")
            else:
                lines.append("Embed URL scan -> no flourish urls found in embed HTML")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {REPORT}")
    print(f"Data files saved in: {DATA_DIR}")

if __name__ == "__main__":
    main()
