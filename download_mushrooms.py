"""
iNaturalist Mushroom Image Downloader - Mindanao + Philippines + wider region
=============================================================================
Downloads images for MycoVision's pre-configured mushroom species into
`species_dataset/<Species_Folder>/` for training the species identification
model (`train_species.py`).

Pre-configured targets:
  termitomyces             -> Termitomyces (Mamuno)                   -> species_dataset/Termitomyces/
  auricularia              -> Auricularia (Talingang Daga / Wood Ear) -> species_dataset/Auricularia/
  oyster                   -> Pleurotus ostreatus / P. pulmonarius    -> species_dataset/Pleurotus_ostreatus/
  amanita_virosa           -> Amanita virosa (Destroying Angel)       -> species_dataset/Amanita_virosa/
  leucocoprinus_birnbaumii -> Leucocoprinus birnbaumii                -> species_dataset/Leucocoprinus_birnbaumii/
  gymnopilus               -> genus Gymnopilus (Rustgill)             -> species_dataset/Gymnopilus/

- Pass order per target: Mindanao -> Philippines -> nearby countries -> worldwide (capped)
- Deduped by observation id within each target
- Resolves taxon/place IDs from the API at runtime
- Skips already-downloaded images (safe to re-run)

Usage:
    python download_mushrooms.py                           # run all 6 species targets
    python download_mushrooms.py termitomyces auricularia  # run specific target(s)
    python download_mushrooms.py oyster --cap 200          # cap worldwide pass to 200 obs
"""

import csv
import shutil
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
    ("Worldwide",   None,          None, 300),
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
}

QUALITIES          = ["research", "needs_id"]
PER_PAGE           = 200
SLEEP_BETWEEN      = 1.2
IMG_SLEEP          = 0.4
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


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*") if f.is_file())


# -- PER-TARGET RUN -----------------------------------------------------------

def run_target(key: str, cfg: dict):
    edibility_dir = EDIBLE_DIR if cfg["classification"] == "edible" else POISONOUS_DIR
    species_dir = edibility_dir / cfg["folder"]
    species_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("=" * 60)
    log.info(f"TARGET: {key} ({cfg['classification']}) -> {species_dir}")
    log.info("=" * 60)

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
        for quality in QUALITIES:
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
                            log.info(f"  Downloaded: {species_name} | obs {obs_id} img {i}")
                        elif result == "skipped":
                            species_stats[species_name]["img_count"] += 1
                            total_skipped += 1
                        else:
                            total_failed += 1

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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    requested = [a.lower() for a in args] or list(TARGETS)
    unknown = [r for r in requested if r not in TARGETS]
    if unknown:
        log.error(f"Unknown target(s): {unknown}. Choose from: {list(TARGETS)}")
        return

    for key in requested:
        run_target(key, TARGETS[key])

    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Review mushroom_dataset/{edible,poisonous}/<Species>/ folders for bad photos")
    log.info("  2. Train both models: python train.py && python train_species.py")


if __name__ == "__main__":
    main()
