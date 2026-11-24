"""
iNaturalist Mushroom Image Downloader - Mindanao + Philippines + wider region
=============================================================================
Downloads images for MycoVision's 10 pre-configured mushroom species into
`mushroom_dataset/{edible,poisonous}/<Species_Folder>/` and balances each
species folder to a target image cap (default: 250 images per species =
1,250 edible and 1,250 poisonous).

Pre-configured targets (5 Edible + 5 Poisonous):
  Edible:
    termitomyces             -> Termitomyces (Mamuno)
    auricularia              -> Auricularia (Talingang Daga / Wood Ear)
    oyster                   -> Pleurotus ostreatus (Oyster Mushroom)
    pleurotus_pulmonarius    -> Pleurotus pulmonarius (Phoenix Oyster)
    buna_shimeji             -> Hypsizygus tessellatus (Buna Shimeji)
  Poisonous:
    amanita_virosa           -> Amanita virosa (Destroying Angel)
    leucocoprinus_birnbaumii -> Leucocoprinus birnbaumii (Flowerpot Parasol)
    gymnopilus               -> Gymnopilus (Rustgill)
    ganoderma                -> Ganoderma (Reishi / Artist's Conk)
    trametes                 -> Trametes (Turkey Tail Bracket)

Usage:
    python download_mushrooms.py                           # balance all 10 species to 250 images each
    python download_mushrooms.py --max-images 250          # cap & trim each species to 250 images
    python download_mushrooms.py trametes buna_shimeji     # run specific target(s)
"""

import argparse
import csv
import sys
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

from species_catalog import SPECIES_CATALOG

# -- CONFIG -------------------------------------------------------------------

BASE_DIR            = Path(__file__).resolve().parent
EDIBILITY_ROOT      = BASE_DIR / "mushroom_dataset"
EDIBLE_DIR          = EDIBILITY_ROOT / "edible"
POISONOUS_DIR       = EDIBILITY_ROOT / "poisonous"
DEFAULT_MAX_IMAGES  = 250
IMAGE_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".webp"}

# (label, place name to resolve, fallback place_id or None, cap on new obs per pass or None)
PLACES = [
    ("Mindanao",    "Mindanao",    8225, None),
    ("Philippines", "Philippines", None, None),
    ("Indonesia",   "Indonesia",   None, None),
    ("Malaysia",    "Malaysia",    None, None),
    ("Thailand",    "Thailand",    None, None),
    ("Vietnam",     "Vietnam",     None, None),
    ("Taiwan",      "Taiwan",      None, None),
    ("India",       "India",       None, None),
    ("Worldwide",   None,          None, 400),
]

TARGETS = {
    "termitomyces": {
        "folder":         "Termitomyces",
        "taxa":           SPECIES_CATALOG["Termitomyces"]["taxa"],
        "rank":           SPECIES_CATALOG["Termitomyces"]["rank"],
        "classification": "edible",
        "places":         PLACES,
    },
    "auricularia": {
        "folder":         "Auricularia",
        "taxa":           SPECIES_CATALOG["Auricularia"]["taxa"],
        "rank":           SPECIES_CATALOG["Auricularia"]["rank"],
        "classification": "edible",
        "places":         PLACES,
    },
    "oyster": {
        "folder":         "Pleurotus_ostreatus",
        "taxa":           SPECIES_CATALOG["Pleurotus_ostreatus"]["taxa"],
        "rank":           SPECIES_CATALOG["Pleurotus_ostreatus"]["rank"],
        "classification": "edible",
        "places":         PLACES,
    },
    "pleurotus_pulmonarius": {
        "folder":         "Pleurotus_pulmonarius",
        "taxa":           SPECIES_CATALOG["Pleurotus_pulmonarius"]["taxa"],
        "rank":           SPECIES_CATALOG["Pleurotus_pulmonarius"]["rank"],
        "classification": "edible",
        "places":         PLACES,
    },
    "buna_shimeji": {
        "folder":         "Hypsizygus_tessellatus",
        "taxa":           SPECIES_CATALOG["Hypsizygus_tessellatus"]["taxa"],
        "rank":           SPECIES_CATALOG["Hypsizygus_tessellatus"]["rank"],
        "classification": "edible",
        "places":         PLACES,
    },
    "amanita_virosa": {
        "folder":         "Amanita_virosa",
        "taxa":           SPECIES_CATALOG["Amanita_virosa"]["taxa"],
        "rank":           SPECIES_CATALOG["Amanita_virosa"]["rank"],
        "classification": "poisonous",
        "places":         PLACES,
    },
    "leucocoprinus_birnbaumii": {
        "folder":         "Leucocoprinus_birnbaumii",
        "taxa":           SPECIES_CATALOG["Leucocoprinus_birnbaumii"]["taxa"],
        "rank":           SPECIES_CATALOG["Leucocoprinus_birnbaumii"]["rank"],
        "classification": "poisonous",
        "places":         PLACES,
    },
    "gymnopilus": {
        "folder":         "Gymnopilus",
        "taxa":           SPECIES_CATALOG["Gymnopilus"]["taxa"],
        "rank":           SPECIES_CATALOG["Gymnopilus"]["rank"],
        "classification": "poisonous",
        "places":         PLACES,
    },
    "ganoderma": {
        "folder":         "Ganoderma",
        "taxa":           SPECIES_CATALOG["Ganoderma"]["taxa"],
        "rank":           SPECIES_CATALOG["Ganoderma"]["rank"],
        "classification": "poisonous",
        "places":         PLACES,
    },
    "trametes": {
        "folder":         "Trametes",
        "taxa":           SPECIES_CATALOG["Trametes"]["taxa"],
        "rank":           SPECIES_CATALOG["Trametes"]["rank"],
        "classification": "poisonous",
        "places":         PLACES,
    },
}

QUALITIES          = ["research", "needs_id"]
PER_PAGE           = 200
SLEEP_BETWEEN      = 1.0
IMG_SLEEP          = 0.2
MAX_IMAGES_PER_OBS = 3
OUTPUT_DIR         = BASE_DIR / "mushroom_downloads"

API_BASE = "https://api.inaturalist.org/v1"
HEADERS  = {"User-Agent": "MycoVisionResearch/1.0 (academic project)"}

# -- SETUP --------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    handlers=[file_handler, stream_handler],
)
log = logging.getLogger(__name__)

# -- HELPERS ------------------------------------------------------------------


def best_image_url(photo: dict) -> str:
    return photo.get("url", "").replace("/square.", "/medium.")


def api_get(path: str, params: dict) -> dict:
    resp = requests.get(f"{API_BASE}{path}", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def resolve_taxon_ids(names, rank):
    ids = []
    for name in names:
        try:
            data = api_get("/taxa", {"q": name, "rank": rank, "per_page": 10})
            for t in data.get("results", []):
                if t.get("name", "").lower() == name.lower():
                    log.info(f"Resolved taxon '{name}' ({rank}) -> id {t['id']}")
                    ids.append(str(t["id"]))
                    break
            else:
                log.error(f"No exact match for taxon '{name}' at rank '{rank}'")
        except Exception as e:
            log.error(f"Taxon lookup failed for '{name}': {e}")
    return ",".join(ids)


def resolve_place_id(name: str, fallback):
    try:
        data = api_get("/places/autocomplete", {"q": name, "per_page": 10})
        results = data.get("results", [])
        for p in results:
            if p.get("name", "").lower() == name.lower():
                log.info(f"Resolved place '{name}' -> id {p['id']}")
                return p["id"]
        if results:
            log.info(f"Place '{name}' not exact; using '{results[0].get('name')}' -> id {results[0]['id']}")
            return results[0]["id"]
    except Exception as e:
        log.error(f"Place lookup failed for '{name}': {e}")
    if fallback:
        log.info(f"Using fallback place_id {fallback} for '{name}'")
    return fallback


def fetch_page(place_id, taxon_id: str, quality: str, page: int) -> dict:
    params = {} if place_id is None else {"place_id": place_id}
    return api_get("/observations", {
        **params,
        "taxon_id":      taxon_id,
        "quality_grade": quality,
        "per_page":      PER_PAGE,
        "page":          page,
        "order":         "desc",
        "order_by":      "created_at",
        "photos":        "true",
    })


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
        if dest.exists():
            dest.unlink()
        return "failed"


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(folder: Path) -> int:
    return len(list_images(folder))


def trim_excess_images(folder: Path, max_images: int) -> int:
    """If a species folder exceeds max_images, deterministically remove excess files to enforce balance."""
    imgs = list_images(folder)
    if len(imgs) <= max_images:
        return 0
    excess = imgs[max_images:]
    for f in excess:
        try:
            f.unlink()
        except Exception:
            pass
    return len(excess)


# -- PER-TARGET RUN -----------------------------------------------------------

def run_target(key: str, cfg: dict, max_images: int | None = DEFAULT_MAX_IMAGES, trim_excess: bool = True):
    edibility_dir = EDIBLE_DIR if cfg["classification"] == "edible" else POISONOUS_DIR
    species_dir = edibility_dir / cfg["folder"]
    species_dir.mkdir(parents=True, exist_ok=True)

    if max_images and trim_excess:
        trimmed = trim_excess_images(species_dir, max_images)
        if trimmed > 0:
            log.info(f"[{key}] Trimmed {trimmed} excess image(s) from {species_dir.name} to cap at {max_images}.")

    current_images = count_images(species_dir)

    log.info("")
    log.info("=" * 60)
    log.info(f"TARGET: {key} ({cfg['classification']}) -> {species_dir} (current: {current_images}, target: {max_images})")
    log.info("=" * 60)

    if max_images and current_images >= max_images:
        log.info(f"[{key}] Already at target count ({current_images}/{max_images}). Skipping download.")
        return

    target_out = OUTPUT_DIR / key
    target_out.mkdir(parents=True, exist_ok=True)

    taxon_id = resolve_taxon_ids(cfg["taxa"], cfg["rank"])
    if not taxon_id:
        log.error(f"[{key}] Could not resolve any taxon ids. Skipping target.")
        return
    if len(taxon_id.split(",")) < len(cfg["taxa"]):
        log.warning(f"[{key}] Only some taxa resolved; continuing with those.")

    targets = []
    for label, place_name, fallback, cap in cfg["places"]:
        if place_name is None:
            targets.append((label, None, cap))
            continue
        pid = resolve_place_id(place_name, fallback)
        if pid:
            targets.append((label, pid, cap))
        else:
            log.error(f"Skipping '{label}': no place id")

    seen_obs_ids = set()
    species_stats = {}
    obs_rows = []
    total_downloaded = total_skipped = total_failed = 0

    for label, place_id, cap in targets:
        if max_images and current_images >= max_images:
            log.info(f"  Reached --max-images limit ({max_images}) for {key}")
            break
        for quality in QUALITIES:
            if max_images and current_images >= max_images:
                break
            log.info("")
            log.info(f"--- [{key}] {label} | quality={quality} | place_id={place_id} ---")

            page = 1
            total_results = None
            pass_new = 0

            while True:
                log.info(f"  Fetching page {page}...")
                try:
                    data = fetch_page(place_id, taxon_id, quality, page)
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
                    if cap and pass_new >= cap:
                        break
                    obs_id = obs.get("id")
                    if obs_id in seen_obs_ids:
                        continue
                    seen_obs_ids.add(obs_id)
                    pass_new += 1

                    taxon = obs.get("taxon")
                    if not taxon:
                        continue

                    species_name = taxon.get("name", "Unknown")
                    common_name  = taxon.get("preferred_common_name", "")
                    rank         = taxon.get("rank", "")

                    if species_name not in species_stats:
                        species_stats[species_name] = {
                            "common_name": common_name,
                            "taxon_rank":  rank,
                            "taxon_id":    taxon.get("id"),
                            "obs_count":   0,
                            "img_count":   0,
                        }
                    species_stats[species_name]["obs_count"] += 1

                    obs_rows.append({
                        "obs_id":      obs_id,
                        "species":     species_name,
                        "region":      label,
                        "quality":     obs.get("quality_grade", quality),
                        "location":    obs.get("place_guess", ""),
                        "latlng":      obs.get("location", ""),
                        "observed_on": obs.get("observed_on", ""),
                        "url":         obs.get("uri", ""),
                    })

                    photos = obs.get("photos", [])[:MAX_IMAGES_PER_OBS]
                    for i, photo in enumerate(photos, 1):
                        img_url = best_image_url(photo)
                        ext     = Path(urlparse(img_url).path).suffix or ".jpg"
                        dest    = species_dir / f"obs_{obs_id}_{i}{ext}"

                        result = download_image(img_url, dest)
                        if result == "downloaded":
                            species_stats[species_name]["img_count"] += 1
                            total_downloaded += 1
                            current_images += 1
                            log.info(f"  Downloaded ({current_images}/{max_images or '∞'}): {species_name} | obs {obs_id} img {i}")
                            if max_images and current_images >= max_images:
                                break
                        elif result == "skipped":
                            species_stats[species_name]["img_count"] += 1
                            total_skipped += 1
                        else:
                            total_failed += 1

                    if max_images and current_images >= max_images:
                        break

                if max_images and current_images >= max_images:
                    log.info(f"  Reached --max-images limit ({max_images})")
                    break

                if cap and pass_new >= cap:
                    log.info(f"  Reached cap of {cap} new observations for this pass")
                    break

                if page * PER_PAGE >= (total_results or 0):
                    break

                if page * PER_PAGE >= 10000:
                    log.warning("  Hit the API's 10,000-result limit for this query")
                    break

                page += 1
                time.sleep(SLEEP_BETWEEN)

    # -- Write CSVs -----------------------------------------------------------
    with open(target_out / "species_list.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["species_name", "common_name", "taxon_rank", "taxon_id", "obs_count", "img_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, s in sorted(species_stats.items(), key=lambda x: x[1]["obs_count"], reverse=True):
            writer.writerow({"species_name": name, **s})

    with open(target_out / "observations.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["obs_id", "species", "region", "quality", "location", "latlng", "observed_on", "url"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(obs_rows)

    log.info("")
    log.info(f"[{key}] DONE")
    log.info(f"  Unique observations : {len(seen_obs_ids)}")
    log.info(f"  Unique species      : {len(species_stats)}")
    log.info(f"  Images downloaded   : {total_downloaded}")
    log.info(f"  Images skipped      : {total_skipped}")
    log.info(f"  Images failed       : {total_failed}")
    log.info(f"  Images in folder    : {count_images(species_dir)}")
    log.info(f"  CSVs                : {target_out.resolve()}")


# -- MAIN ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and balance mushroom images from iNaturalist.")
    parser.add_argument("targets", nargs="*", help=f"Target keys from: {list(TARGETS)}")
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help=f"Target cap per species folder (default: {DEFAULT_MAX_IMAGES})",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim existing folders that already exceed --max-images",
    )
    parsed = parser.parse_args()

    requested = [a.lower() for a in parsed.targets] or list(TARGETS)
    unknown = [r for r in requested if r not in TARGETS]
    if unknown:
        log.error(f"Unknown target(s): {unknown}. Choose from: {list(TARGETS)}")
        return

    for key in requested:
        run_target(
            key,
            TARGETS[key],
            max_images=parsed.max_images,
            trim_excess=not parsed.no_trim,
        )

    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Review mushroom_dataset/{edible,poisonous}/<Species>/ folders for bad photos")
    log.info("  2. Train both models: python train.py && python train_species.py")


if __name__ == "__main__":
    main()
