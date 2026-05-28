"""
iNaturalist Mindanao Fungi Image Downloader (Expanded)
======================================================
Wider search than the Davao-only version:
  - Covers all of Mindanao (place_id=8225)
  - Includes needs_id grade (not just research grade)
  - Falls back to Davao City (place_id=29052) as a second pass
  - Skips already-downloaded images (safe to re-run)

Usage:
    python download_mindanao_fungi.py

Output:
    mindanao_fungi_dataset/
        images/
            edible/          <- move images here after labeling
            poisonous/       <- move images here after labeling
            unlabeled/       <- all raw downloads land here first
                Phlebopus_beniensis/
                    obs_12345_1.jpg
        species_list.csv
        download_log.txt
"""

import csv
import sys
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

# ── CONFIG ────────────────────────────────────────────────────────────────────

SEARCH_TARGETS = [
    {"label": "Mindanao",   "place_id": 8225,  "quality": "research"},
    {"label": "Mindanao",   "place_id": 8225,  "quality": "needs_id"},
    {"label": "Davao City", "place_id": 29052, "quality": "research"},
    {"label": "Davao City", "place_id": 29052, "quality": "needs_id"},
]

TAXON_ID            = 47170      # Fungi kingdom
PER_PAGE            = 200        # Max allowed by iNaturalist API
SLEEP_BETWEEN       = 1.2        # Seconds between API page calls
IMG_SLEEP           = 0.4        # Seconds between image downloads
MAX_IMAGES_PER_OBS  = 3          # Up to 3 photos per observation
OUTPUT_DIR          = Path("mindanao_fungi_dataset")

API_BASE = "https://api.inaturalist.org/v1"
HEADERS  = {"User-Agent": "MindanaoFungiResearch/1.0 (academic project)"}

# ── SETUP ─────────────────────────────────────────────────────────────────────

for sub in ["images/unlabeled", "images/edible", "images/poisonous"]:
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

stream_handler = logging.StreamHandler(sys.stdout)
try:
    stream_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    )
except Exception:
    pass

file_handler = logging.FileHandler(OUTPUT_DIR / "download_log.txt", encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[file_handler, stream_handler]
)
log = logging.getLogger(__name__)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-")


def best_image_url(photo: dict) -> str:
    return photo.get("url", "").replace("/square.", "/medium.")


def fetch_page(place_id: int, quality: str, page: int) -> dict:
    resp = requests.get(
        f"{API_BASE}/observations",
        params={
            "place_id":      place_id,
            "taxon_id":      TAXON_ID,
            "quality_grade": quality,
            "per_page":      PER_PAGE,
            "page":          page,
            "order":         "desc",
            "order_by":      "created_at",
            "photos":        "true",
        },
        headers=HEADERS,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def download_image(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        time.sleep(IMG_SLEEP)
        return "downloaded"
    except Exception as e:
        log.warning(f"    FAILED {url}: {e}")
        return "failed"


def count_images() -> int:
    return sum(1 for f in (OUTPUT_DIR / "images").rglob("*") if f.is_file())

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("iNaturalist Mindanao Fungi Downloader (Expanded)")
    log.info("=" * 60)

    seen_obs_ids = set()       # Avoid duplicate observations across passes
    species_stats = {}
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for target in SEARCH_TARGETS:
        label    = target["label"]
        place_id = target["place_id"]
        quality  = target["quality"]

        log.info("")
        log.info(f"--- Searching: {label} | quality={quality} | place_id={place_id} ---")

        page = 1
        total_results = None
        pass_count = 0

        while True:
            log.info(f"  Fetching page {page}...")
            try:
                data = fetch_page(place_id, quality, page)
            except Exception as e:
                log.error(f"  API error: {e}")
                break

            if total_results is None:
                total_results = data.get("total_results", 0)
                log.info(f"  Total available: {total_results}")

            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                obs_id = obs.get("id")
                if obs_id in seen_obs_ids:
                    continue
                seen_obs_ids.add(obs_id)
                pass_count += 1

                taxon = obs.get("taxon")
                if not taxon:
                    continue

                species_name = taxon.get("name", "Unknown")
                common_name  = taxon.get("preferred_common_name", "")
                rank         = taxon.get("rank", "")

                # Species folder under unlabeled/
                species_dir = OUTPUT_DIR / "images" / "unlabeled" / safe_name(species_name)
                species_dir.mkdir(exist_ok=True)

                if species_name not in species_stats:
                    species_stats[species_name] = {
                        "common_name": common_name,
                        "taxon_rank":  rank,
                        "taxon_id":    taxon.get("id"),
                        "obs_count":   0,
                        "img_count":   0,
                        "edibility":   "",
                        "notes":       ""
                    }
                species_stats[species_name]["obs_count"] += 1

                photos = obs.get("photos", [])[:MAX_IMAGES_PER_OBS]
                for i, photo in enumerate(photos, 1):
                    img_url = best_image_url(photo)
                    ext     = Path(urlparse(img_url).path).suffix or ".jpg"
                    dest    = species_dir / f"obs_{obs_id}_{i}{ext}"

                    result = download_image(img_url, dest)
                    if result == "downloaded":
                        species_stats[species_name]["img_count"] += 1
                        total_downloaded += 1
                        log.info(f"  Downloaded: {species_name} | obs {obs_id} img {i}")
                    elif result == "skipped":
                        species_stats[species_name]["img_count"] += 1
                        total_skipped += 1
                    else:
                        total_failed += 1

            log.info(f"  New unique observations this pass: {pass_count}")

            if len(seen_obs_ids) >= (total_results or 0) or pass_count >= (total_results or 0):
                break

            page += 1
            time.sleep(SLEEP_BETWEEN)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    log.info("")
    log.info("Writing species_list.csv...")

    csv_path   = OUTPUT_DIR / "species_list.csv"
    fieldnames = [
        "species_name", "common_name", "taxon_rank", "taxon_id",
        "obs_count", "img_count", "edibility", "notes"
    ]

    sorted_species = sorted(
        species_stats.items(),
        key=lambda x: x[1]["obs_count"],
        reverse=True
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, s in sorted_species:
            writer.writerow({
                "species_name": name,
                "common_name":  s["common_name"],
                "taxon_rank":   s["taxon_rank"],
                "taxon_id":     s["taxon_id"],
                "obs_count":    s["obs_count"],
                "img_count":    s["img_count"],
                "edibility":    s["edibility"],
                "notes":        s["notes"]
            })

    log.info(f"  Saved {len(species_stats)} species to {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("DONE")
    log.info(f"  Unique observations : {len(seen_obs_ids)}")
    log.info(f"  Unique species      : {len(species_stats)}")
    log.info(f"  Images downloaded   : {total_downloaded}")
    log.info(f"  Images skipped      : {total_skipped}")
    log.info(f"  Images failed       : {total_failed}")
    log.info(f"  Total images on disk: {count_images()}")
    log.info(f"  Output folder       : {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)
    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Open species_list.csv")
    log.info("  2. Fill in edibility column: edible / poisonous / unknown")
    log.info("  3. Move images from images/unlabeled/<species>/ into")
    log.info("       images/edible/   or   images/poisonous/")
    log.info("  4. Merge with Kaggle dataset (see README below)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()