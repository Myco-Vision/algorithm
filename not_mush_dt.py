"""
not_mush_dt.py — Non-Mushroom & Close-Up Hard-Negative Dataset Builder
======================================================================
Populates `mushroom_dataset/not_mushroom/` up to TARGET_TOTAL_IMAGES (1,250)
to match the 1,250 edible and 1,250 poisonous mushroom counts (1:1:1 balance).

Includes 3 targeted hard-negative groups in addition to existing indoor scenes:
  1. Close-Up Fruits & Food (durian look-alikes):
     durian fruit, jackfruit, pineapple, coconut, potato, garlic bulb, cauliflower, bread loaf
  2. Close-Up Tech & Handheld Objects (computer mouse look-alikes):
     computer mouse, computer keyboard, smartphone, coffee mug, water bottle,
     human hand close up, shoes, eyeglasses
  3. Natural Foraging Backgrounds & Plant Look-Alikes:
     forest floor, leaf litter, tree bark close up, mossy rock, dead wood log,
     pine cone, fern leaves, wildflower
"""

import argparse
import csv
import re
import time
from pathlib import Path

import requests

# --- CONFIGURATION ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "mushroom_dataset" / "not_mushroom"
TARGET_TOTAL_IMAGES = 1250
IMAGES_PER_TERM = 36        # ~35-36 per new term across 24 terms -> +841 images (409 + 841 = 1,250)
THUMB_WIDTH = 640           # downloaded width in pixels (plenty for 224x224 training)
SLEEP_SECONDS = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SEARCH_TERMS = [
    # Group 1: Close-Up Fruits, Vegetables & Round/Spiky Foods (Durian look-alikes)
    "durian fruit",
    "jackfruit",
    "pineapple",
    "coconut",
    "potato",
    "garlic bulb",
    "cauliflower",
    "bread loaf",

    # Group 2: Close-Up Tech, Desk & Handheld Objects (Computer mouse look-alikes)
    "computer mouse",
    "computer keyboard",
    "smartphone",
    "coffee mug",
    "water bottle",
    "human hand",
    "shoes",
    "eyeglasses",

    # Group 3: Natural Foraging Backgrounds & Outdoor Look-Alikes
    "forest floor",
    "leaf litter",
    "tree bark",
    "mossy rock",
    "dead wood log",
    "pine cone",
    "fern leaves",
    "wildflower",

    # Group 4: Fallback everyday indoor/outdoor objects if top-up needed to reach 1,250
    "desk",
    "kitchen utensils",
    "household objects",
    "living room",
    "office supplies",
    "soil ground",
    "tree roots ground",
    "grass lawn",
]

# Skip anything whose title/description/categories hint at fungi.
EXCLUDE = re.compile(
    r"mushroom|fung|toadstool|agaric|amanita|bolete|puffball|polypore|"
    r"mycel|boletus|russula|chanterelle|morel|ganoderma|trametes|pleurotus|auricularia",
    re.IGNORECASE,
)
# ---------------------------------------------------------------------------

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MycoVision-dataset-builder/1.0 (student project; python-requests)"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


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
    parser = argparse.ArgumentParser(description="Download balanced hard-negative non-mushroom images.")
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL_IMAGES, help="Stop when folder reaches this total image count (default: 1250)")
    parser.add_argument("--per-term", type=int, default=IMAGES_PER_TERM, help="Max new images per search term (default: 36)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_total = count_images(OUT_DIR)
    print(f"Starting not_mushroom count: {current_total} | Target total: {args.target_total}")

    if current_total >= args.target_total:
        print(f"Already at or above target total ({current_total}/{args.target_total}).")
        return

    credits_path = OUT_DIR / "_credits_commons.csv"
    new_credits_file = not credits_path.exists()
    total_new = 0

    with open(credits_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        if new_credits_file:
            writer.writerow(["file", "title", "artist", "license", "search_term"])

        for term in SEARCH_TERMS:
            if current_total >= args.target_total:
                break

            got = 0
            offset = 0
            print(f"Searching: '{term}' (current total: {current_total}/{args.target_total})")

            while got < args.per_term and current_total < args.target_total:
                try:
                    pages, next_offset = search_files(term, offset)
                except requests.RequestException as exc:
                    print(f"  search failed: {exc}")
                    break
                if not pages:
                    break

                for page in pages:
                    if got >= args.per_term or current_total >= args.target_total:
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
                    current_total += 1
                    time.sleep(0.15)

                if next_offset is None:
                    break
                offset = next_offset
                time.sleep(SLEEP_SECONDS)

            print(f"  kept {got} for '{term}' (folder total: {current_total}/{args.target_total})")

    print(f"Done. {total_new} new images saved to {OUT_DIR} (final total: {count_images(OUT_DIR)})")


if __name__ == "__main__":
    main()
