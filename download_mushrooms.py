"""
iNaturalist Davao City Fungi Image Downloader
=============================================
Downloads all research-grade fungi observations and images
from Davao City (place_id=29052) using the iNaturalist API.

Organizes images into folders by species name and generates
a CSV with edibility column for manual labeling.

Usage:
    python download_davao_fungi.py

Output structure:
    davao_fungi_dataset/
    ├── images/
    │   ├── Ganoderma_lucidum/
    │   │   ├── obs_12345_1.jpg
    │   │   └── obs_12345_2.jpg
    │   └── Trametes_versicolor/
    │       └── obs_67890_1.jpg
    ├── species_list.csv       ← fill in the 'edibility' column
    └── download_log.txt
"""

import os
import csv
import json
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

# ─── CONFIG ──────────────────────────────────────────────────────────────────

PLACE_ID        = 29052          # Davao City on iNaturalist
TAXON_ID        = 47170          # Fungi kingdom
QUALITY_GRADE   = "research"     # Only verified observations
PER_PAGE        = 200            # Max allowed by API
SLEEP_BETWEEN   = 1.0            # Seconds between API calls (be polite)
IMG_SLEEP       = 0.5            # Seconds between image downloads
OUTPUT_DIR      = Path("davao_fungi_dataset")
MAX_IMAGES_PER_OBS = 3           # Download up to 3 photos per observation

API_BASE = "https://api.inaturalist.org/v1"
HEADERS  = {"User-Agent": "DavaoFungiResearch/1.0 (academic project)"}

# ─── SETUP ───────────────────────────────────────────────────────────────────

(OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "download_log.txt"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    """Convert species name to a safe folder name."""
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-")


def fetch_observations(page: int) -> dict:
    """Fetch one page of fungi observations from Davao City."""
    url = f"{API_BASE}/observations"
    params = {
        "place_id":     PLACE_ID,
        "taxon_id":     TAXON_ID,
        "quality_grade": QUALITY_GRADE,
        "per_page":     PER_PAGE,
        "page":         page,
        "order":        "desc",
        "order_by":     "created_at",
        "photos":       "true",   # Only observations that have photos
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_image(url: str, dest: Path) -> bool:
    """Download a single image. Returns True on success."""
    if dest.exists():
        return True  # Skip already downloaded
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        time.sleep(IMG_SLEEP)
        return True
    except Exception as e:
        log.warning(f"  ✗ Failed to download {url}: {e}")
        return False


def best_image_url(photo: dict) -> str:
    """Get the medium-size image URL from a photo object."""
    url = photo.get("url", "")
    # iNaturalist photo URLs end in /square.jpg — upgrade to medium
    return url.replace("/square.", "/medium.")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("iNaturalist Davao City Fungi Downloader")
    log.info(f"Place ID : {PLACE_ID}  |  Taxon ID : {TAXON_ID}")
    log.info("=" * 60)

    all_observations = []
    species_stats    = {}   # { species_name: { obs_count, image_count } }

    # ── Step 1: Fetch all observation pages ──────────────────────────────────
    log.info("\n[1/3] Fetching observations from iNaturalist API...")
    page = 1
    total_results = None

    while True:
        log.info(f"  Fetching page {page}...")
        try:
            data = fetch_observations(page)
        except Exception as e:
            log.error(f"  API error on page {page}: {e}")
            break

        if total_results is None:
            total_results = data.get("total_results", 0)
            log.info(f"  Total research-grade fungi observations in Davao City: {total_results}")

        results = data.get("results", [])
        if not results:
            break

        all_observations.extend(results)
        log.info(f"  Collected {len(all_observations)}/{total_results} observations")

        if len(all_observations) >= total_results:
            break

        page += 1
        time.sleep(SLEEP_BETWEEN)

    log.info(f"\n  ✓ Total observations fetched: {len(all_observations)}")

    # ── Step 2: Download images, organized by species ─────────────────────────
    log.info("\n[2/3] Downloading images...")

    skipped_no_taxon = 0
    total_images_downloaded = 0

    for obs in all_observations:
        # Get species name
        taxon = obs.get("taxon")
        if not taxon:
            skipped_no_taxon += 1
            continue

        species_name = taxon.get("name", "Unknown")
        common_name  = taxon.get("preferred_common_name", "")
        obs_id       = obs.get("id")
        rank         = taxon.get("rank", "")

        # Create species folder
        folder_name = safe_name(species_name)
        species_dir = OUTPUT_DIR / "images" / folder_name
        species_dir.mkdir(exist_ok=True)

        # Track species stats
        if species_name not in species_stats:
            species_stats[species_name] = {
                "common_name": common_name,
                "taxon_rank":  rank,
                "taxon_id":    taxon.get("id"),
                "obs_count":   0,
                "img_count":   0,
                "edibility":   "",   # To be filled manually
                "notes":       ""
            }
        species_stats[species_name]["obs_count"] += 1

        # Download photos
        photos = obs.get("photos", [])[:MAX_IMAGES_PER_OBS]
        for i, photo in enumerate(photos, 1):
            img_url = best_image_url(photo)
            ext = Path(urlparse(img_url).path).suffix or ".jpg"
            dest = species_dir / f"obs_{obs_id}_{i}{ext}"

            success = download_image(img_url, dest)
            if success:
                species_stats[species_name]["img_count"] += 1
                total_images_downloaded += 1
                log.info(f"  ✓ [{species_name}] obs {obs_id} photo {i}")

    log.info(f"\n  ✓ Total images downloaded: {total_images_downloaded}")
    log.info(f"  ✓ Unique species/taxa: {len(species_stats)}")
    if skipped_no_taxon:
        log.info(f"  ⚠ Skipped {skipped_no_taxon} observations with no taxon info")

    # ── Step 3: Write species CSV ─────────────────────────────────────────────
    log.info("\n[3/3] Writing species_list.csv...")

    csv_path = OUTPUT_DIR / "species_list.csv"
    fieldnames = [
        "species_name",
        "common_name",
        "taxon_rank",
        "taxon_id",
        "obs_count",
        "img_count",
        "edibility",      # ← FILL THIS IN: edible / poisonous / unknown
        "notes"
    ]

    # Sort by observation count descending
    sorted_species = sorted(
        species_stats.items(),
        key=lambda x: x[1]["obs_count"],
        reverse=True
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, stats in sorted_species:
            writer.writerow({
                "species_name": name,
                "common_name":  stats["common_name"],
                "taxon_rank":   stats["taxon_rank"],
                "taxon_id":     stats["taxon_id"],
                "obs_count":    stats["obs_count"],
                "img_count":    stats["img_count"],
                "edibility":    stats["edibility"],
                "notes":        stats["notes"]
            })

    log.info(f"  ✓ Saved to {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"  Observations : {len(all_observations)}")
    log.info(f"  Species/taxa : {len(species_stats)}")
    log.info(f"  Images saved : {total_images_downloaded}")
    log.info(f"  Output folder: {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)
    log.info("\nNEXT STEPS:")
    log.info("  1. Open species_list.csv")
    log.info("  2. Fill in the 'edibility' column for each species:")
    log.info("     edible / poisonous / unknown")
    log.info("  3. Use the labeled CSV + images folder to train your model")
    log.info("=" * 60)


if __name__ == "__main__":
    main()