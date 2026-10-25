"""TMDB API client: fetches posters / ratings / links for movies by title."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

# Your key file may be called ".env" or "_env" - load whichever exists.
_here = Path(__file__).parent
for name in (".env", "_env"):
    if (_here / name).exists():
        load_dotenv(_here / name)
        break

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMG_URL = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER = "https://placehold.co/342x513/1f2430/9aa3b2?text=No+Poster"

_session = requests.Session()


def fetch_movie_info(title: str) -> dict:
    """Search TMDB for a title. Always returns a dict (never raises)."""
    info = {"poster": PLACEHOLDER, "tmdb_url": None, "year": ""}
    if not API_KEY:
        return info
    try:
        r = _session.get(
            f"{BASE_URL}/search/movie",
            params={"api_key": API_KEY, "query": title, "include_adult": "false"},
            timeout=6,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            m = results[0]
            if m.get("poster_path"):
                info["poster"] = IMG_URL + m["poster_path"]
            info["tmdb_url"] = f"https://www.themoviedb.org/movie/{m['id']}"
            info["year"] = (m.get("release_date") or "")[:4]
    except requests.RequestException:
        pass
    return info


def fetch_many(titles: list[str]) -> list[dict]:
    """Fetch several titles in parallel (keeps the page fast)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(fetch_movie_info, titles))
