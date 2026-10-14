import csv
import re
import time
from pathlib import Path

import requests

# --- CONFIGURATION ---------------------------------------------------------
OUT_DIR = Path("mushroom_dataset/not_mushroom")
IMAGES_PER_TERM = 15        # cap per search term so the folder stays varied
THUMB_WIDTH = 640           # downloaded width in pixels (plenty for 224x224 training)
SLEEP_SECONDS = 1.0

SEARCH_TERMS = [
    # "forest floor",
    # "leaf litter",
    # "dry leaves on ground",
    # "soil",
    # "dirt ground",
    # "mud",
    # "mulch",
    # "gravel path",
    # "rainforest floor",
    # "grass lawn close-up",
    # "tree roots ground",
    # "forest trail",
    "desk", "kitchen utensils", "household objects", "living room",
    "office supplies", "bedroom", "bookshelf", "table setting",
]

# Skip anything whose title/description/categories hint at fungi.
EXCLUDE = re.compile(
    r"mushroom|fung|toadstool|agaric|amanita|bolete|puffball|polypore|"
    r"mycel|boletus|russula|chanterelle|morel",
    re.IGNORECASE,
)
# ---------------------------------------------------------------------------

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MycoVision-dataset-builder/1.0 (student project; python-requests)"}


def search_files(term: str, offset: int):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,                       # File: namespace
        "gsrsearch": f"{term} filetype:bitmap",
        "gsrlimit": 50,
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": THUMB_WIDTH,
    }
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {}).values()
    next_offset = data.get("continue", {}).get("gsroffset")
    return list(pages), next_offset


def meta(info: dict, key: str) -> str:
    return info.get("extmetadata", {}).get(key, {}).get("value", "") or ""


def license_ok(license_name: str) -> bool:
    name = license_name.lower()
    return name.startswith("cc") or "public domain" in name or name == "pd"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    credits_path = OUT_DIR / "_credits_commons.csv"
    new_credits_file = not credits_path.exists()
    total_new = 0

    with open(credits_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        if new_credits_file:
            writer.writerow(["file", "title", "artist", "license", "search_term"])

        for term in SEARCH_TERMS:
            got = 0
            offset = 0
            print(f"Searching: {term}")

            while got < IMAGES_PER_TERM:
                try:
                    pages, next_offset = search_files(term, offset)
                except requests.RequestException as exc:
                    print(f"  search failed: {exc}")
                    break
                if not pages:
                    break

                for page in pages:
                    if got >= IMAGES_PER_TERM:
                        break
                    info = (page.get("imageinfo") or [{}])[0]
                    if info.get("mime") not in ("image/jpeg", "image/png"):
                        continue

                    title = page.get("title", "")
                    haystack = " ".join([
                        title,
                        meta(info, "ImageDescription"),
                        meta(info, "Categories"),
                    ])
                    if EXCLUDE.search(haystack):
                        continue
                    license_name = meta(info, "LicenseShortName")
                    if not license_ok(license_name):
                        continue

                    url = info.get("thumburl") or info.get("url")
                    if not url:
                        continue
                    ext = ".png" if info["mime"] == "image/png" else ".jpg"
                    dest = OUT_DIR / f"commons_{page['pageid']}{ext}"
                    if dest.exists():
                        continue

                    try:
                        img = requests.get(url, headers=HEADERS, timeout=30)
                        img.raise_for_status()
                    except requests.RequestException:
                        continue

                    dest.write_bytes(img.content)
                    artist = re.sub(r"<[^>]+>", "", meta(info, "Artist")).strip()
                    writer.writerow([dest.name, title, artist, license_name, term])
                    got += 1
                    total_new += 1
                    time.sleep(0.3)

                if next_offset is None:
                    break
                offset = next_offset
                time.sleep(SLEEP_SECONDS)

            print(f"  kept {got} for '{term}'")

    print(f"Done. {total_new} new images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()