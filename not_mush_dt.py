"""
not_mush_dt.py — 125-Object × 10-Variant Non-Mushroom Dataset Builder (1,250 Images)
====================================================================================
Builds a strictly balanced `mushroom_dataset/not_mushroom/` dataset containing:
  - 125 distinct non-mushroom object categories
  - 10 different variant images per object category
  - Total: 125 × 10 = 1,250 not_mushroom images (matching 1,250 edible & 1,250 poisonous)

Each image is saved as:
    mushroom_dataset/not_mushroom/<object_slug>_<01..10>_<pageid>.jpg
so every object has 10 distinct visual variants and the folder can be audited or resumed
deterministically.
"""

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# --- CONFIGURATION ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "mushroom_dataset" / "not_mushroom"
IMAGES_PER_OBJECT = 10      # 10 different variants per object
THUMB_WIDTH = 640           # downloaded width in pixels (plenty for 224x224 training)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Exactly 125 distinct non-mushroom objects across 5 real-world categories (25 × 5 = 125)
OBJECTS_125 = [
    # 1. Furniture, Household & Room Objects (25)
    "chair",
    "wooden table",
    "sofa couch",
    "bed furniture",
    "bookshelf",
    "office desk",
    "wardrobe closet",
    "wooden cabinet",
    "table lamp",
    "wall mirror",
    "wall clock",
    "pillow cushion",
    "window curtain",
    "woven carpet",
    "wooden door",
    "glass window",
    "staircase",
    "wooden stool",
    "park bench",
    "chest of drawers",
    "chandelier",
    "trash can",
    "laundry basket",
    "ceramic vase",
    "picture frame",

    # 2. Electronics, Desk Supplies & Everyday Tools (25)
    "computer mouse",
    "computer keyboard",
    "laptop computer",
    "smartphone",
    "computer monitor",
    "headphones",
    "digital camera",
    "television set",
    "remote control",
    "flashlight",
    " wristwatch",
    "ballpoint pen",
    "paper notebook",
    "scissors",
    "office stapler",
    "claw hammer",
    "screwdriver",
    "metal wrench",
    "paintbrush",
    "pocket calculator",
    "microphone",
    "audio speaker",
    "desktop printer",
    "electric fan",
    " padlock",

    # 3. Kitchen, Containers, Clothing & Personal Items (25)
    "coffee mug",
    "water bottle",
    "drinking glass",
    "ceramic plate",
    "soup bowl",
    "metal spoon",
    "dining fork",
    "kitchen knife",
    "frying pan",
    "cooking pot",
    "tea kettle",
    "teapot",
    "wooden cutting board",
    "backpack",
    "leather handbag",
    "leather wallet",
    "umbrella",
    "eyeglasses",
    "baseball cap",
    "leather shoe",
    "running sneaker",
    "winter jacket",
    "knit glove",
    "metal key",
    "metal coin",

    # 4. Fruits, Vegetables, Food & Culinary Items (25)
    "durian fruit",
    "jackfruit",
    "pineapple fruit",
    "coconut fruit",
    "red apple",
    "banana fruit",
    "orange citrus",
    "mango fruit",
    "strawberry fruit",
    "watermelon",
    "grape cluster",
    "yellow lemon",
    "avocado fruit",
    "papaya fruit",
    "red tomato",
    "potato tuber",
    "orange carrot",
    "onion bulb",
    "garlic bulb",
    "green broccoli",
    "cauliflower",
    "green cabbage",
    "bell pepper",
    "bread loaf",
    "chicken egg",

    # 5. Nature, Animals, Plants, Outdoor & Vehicles (25)
    "tree bark",
    "green leaf",
    "autumn dry leaves",
    "pine cone",
    "hibiscus flower",
    "red rose flower",
    "sunflower",
    "fern plant",
    "cactus plant",
    "green grass lawn",
    "granite rock",
    "river pebbles",
    "seashell",
    "bird feather",
    "sparrow bird",
    "domestic cat",
    "domestic dog",
    "monarch butterfly",
    "goldfish",
    "human hand",
    "bicycle",
    "passenger car",
    "motorcycle",
    "wooden boat",
    "brick house",
]

assert len(OBJECTS_125) == 125, f"Expected 125 objects, got {len(OBJECTS_125)}"

# Skip anything whose title/description/categories hint at fungi.
EXCLUDE = re.compile(
    r"mushroom|fung|toadstool|agaric|amanita|bolete|puffball|polypore|"
    r"mycel|boletus|russula|chanterelle|morel|ganoderma|trametes|pleurotus|auricularia",
    re.IGNORECASE,
)

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MycoVision-dataset-builder/1.0 (student project; python-requests)"}


def slugify(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.strip().lower()).strip("_")


def search_files(session: requests.Session, term: str, offset: int):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,                       # File: namespace
        "gsrsearch": f"{term} filetype:bitmap",
        "gsrlimit": 45,
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": THUMB_WIDTH,
    }
    for attempt in range(6):
        try:
            resp = session.get(API_URL, params=params, headers=HEADERS, timeout=25)
            if resp.status_code == 429:
                time.sleep(2.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {}).values()
            next_offset = data.get("continue", {}).get("gsroffset")
            return list(pages), next_offset
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return [], None


def meta(info: dict, key: str) -> str:
    return info.get("extmetadata", {}).get(key, {}).get("value", "") or ""


def license_ok(license_name: str) -> bool:
    name = license_name.lower()
    return name.startswith("cc") or "public domain" in name or name == "pd"


def existing_variants_for_slug(slug: str) -> list[Path]:
    if not OUT_DIR.exists():
        return []
    return sorted(
        f for f in OUT_DIR.glob(f"{slug}_*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and f.stat().st_size > 0
    )


def download_object_variants(idx: int, raw_term: str, per_object: int) -> tuple[str, int, list[list[str]]]:
    term = raw_term.strip()
    slug = f"{idx:03d}_{slugify(term)}"
    existing = existing_variants_for_slug(slug)

    # If more than per_object exist, trim excess
    if len(existing) > per_object:
        for extra in existing[per_object:]:
            try:
                extra.unlink()
            except Exception:
                pass
        existing = existing[:per_object]

    got = len(existing)
    if got >= per_object:
        return (term, got, [])

    session = requests.Session()
    seen_page_ids = set()
    seen_title_prefixes = set()
    rows = []
    offset = 0

    for f in existing:
        parts = f.stem.split("_")
        if parts and parts[-1].isdigit():
            seen_page_ids.add(int(parts[-1]))

    while got < per_object:
        pages, next_offset = search_files(session, term, offset)
        if not pages:
            break

        # Sort by pageid for deterministic ordering
        pages.sort(key=lambda p: p.get("pageid", 0))

        for page in pages:
            if got >= per_object:
                break
            pageid = page.get("pageid")
            if not pageid or pageid in seen_page_ids:
                continue

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

            # Encourage visual variety: avoid burst-shot duplicates with identical title prefix
            clean_title = re.sub(r"^File:", "", title, flags=re.IGNORECASE).strip().lower()
            title_prefix = re.sub(r"[\d_\-\.\s]+$", "", Path(clean_title).stem)[:18]
            if title_prefix and title_prefix in seen_title_prefixes and next_offset is not None:
                continue

            url = info.get("thumburl") or info.get("url")
            if not url:
                continue

            ext = ".png" if info["mime"] == "image/png" else ".jpg"
            variant_num = got + 1
            dest = OUT_DIR / f"{slug}_{variant_num:02d}_{pageid}{ext}"

            try:
                img = session.get(url, headers=HEADERS, timeout=25)
                if img.status_code != 200 or not img.content:
                    continue
                dest.write_bytes(img.content)
            except requests.RequestException:
                continue

            seen_page_ids.add(pageid)
            if title_prefix:
                seen_title_prefixes.add(title_prefix)
            artist = re.sub(r"<[^>]+>", "", meta(info, "Artist")).strip()
            rows.append([dest.name, title, artist, license_name, term])
            got += 1
            time.sleep(0.08)

        if next_offset is None:
            break
        offset = next_offset
        time.sleep(0.25)

    return (term, got, rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build 125-object × 10-variant not_mushroom dataset (1,250 total images)."
    )
    parser.add_argument("--per-object", type=int, default=IMAGES_PER_OBJECT, help="Variants per object (default: 10)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers (default: 4)")
    parser.add_argument("--clean-legacy", action="store_true", help="Remove old unorganized commons_*.jpg files first")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean_legacy:
        valid_prefixes = tuple(
            f"{idx:03d}_{slugify(term)}_"
            for idx, term in enumerate(OBJECTS_125, start=1)
        )
        removed = 0
        for f in OUT_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                if not f.name.startswith(valid_prefixes):
                    f.unlink()
                    removed += 1
        print(f"Cleaned {removed} legacy unorganized image(s).")

    credits_path = OUT_DIR / "_credits_commons.csv"
    new_credits_file = not credits_path.exists() or args.clean_legacy

    print(
        f"Building not_mushroom dataset: {len(OBJECTS_125)} distinct objects × "
        f"{args.per_object} variants = {len(OBJECTS_125) * args.per_object} target images..."
    )

    all_rows = []
    completed_objects = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_object_variants, idx, term, args.per_object): (idx, term)
            for idx, term in enumerate(OBJECTS_125, start=1)
        }
        for future in as_completed(futures):
            idx, term = futures[future]
            try:
                obj_name, count, rows = future.result()
                all_rows.extend(rows)
                completed_objects += 1
                print(
                    f"[{completed_objects:3d}/{len(OBJECTS_125)}] Object #{idx:03d} '{obj_name}': "
                    f"{count}/{args.per_object} variant(s) ready",
                    flush=True,
                )
            except Exception as exc:
                print(f"[ERROR] Object #{idx:03d} '{term}' failed: {exc}", flush=True)

    mode = "w" if new_credits_file else "a"
    with open(credits_path, mode, newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        if new_credits_file:
            writer.writerow(["file", "title", "artist", "license", "search_term"])
        writer.writerows(all_rows)

    total_imgs = sum(
        1 for f in OUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    print("=" * 60)
    print(f"Done! Total images in {OUT_DIR}: {total_imgs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
