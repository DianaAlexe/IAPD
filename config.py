import os

# YouTube Data API v3 key (set in env: YT_API_KEY)
YOUTUBE_API_KEY = os.getenv("YT_API_KEY", "")

# Official channel(s) (poți adăuga și alte canale locale)
GENSHIN_OFFICIAL_CHANNEL_ID = "UCiS882YPwZt1NfaM0gR0D9Q"  # Genshin Impact (verify if needed)

# User-Agent pentru scraping
USER_AGENT = "Mozilla/5.0 (compatible; DatasetCrawler/1.0; +https://example.com)"

# Output paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
