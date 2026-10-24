"""
iNaturalist Mushroom Image Downloader - Mindanao + Philippines + wider region
=============================================================================
Multi-target version of download_oyster_mushrooms.py.

Targets (each goes to its own class folder):
  oyster      -> Pleurotus ostreatus / P. pulmonarius          -> edible/
  gymnopilus  -> genus Gymnopilus (all descendant species)     -> poisonous/

- Pass order per target: Mindanao -> Philippines -> nearby countries -> worldwide (capped)
- Deduped by observation id within each target
- Resolves taxon/place IDs from the API at runtime
- Skips already-downloaded images (safe to re-run)

Usage:
    python download_mushrooms.py                 # run all targets
    python download_mushrooms.py gymnopilus      # run only Gymnopilus
    python download_mushrooms.py oyster          # run only oyster

Output:
    mushroom_downloads/<target>/species_list.csv
    mushroom_downloads/<target>/observations.csv
    mushroom_downloads/download_log.txt
"""

import csv
import sys
import time
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

# -- CONFIG -------------------------------------------------------------------

DATASET_ROOT  = Path(r"C:\Users\Sui\Documents\MycoVision\algorithm\mushroom_dataset")
EDIBLE_DIR    = DATASET_ROOT / "edible"
POISONOUS_DIR = DATASET_ROOT / "poisonous"

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
    ("Worldwide",   None,          None, 1500),
]

TARGETS = {
    # "oyster": {
    #     "taxa":      ["Pleurotus ostreatus", "Pleurotus pulmonarius"],
    #     "rank":      "species",
    #     "class_dir": EDIBLE_DIR,
    #     "places":    PLACES,
    # },
    "gymnopilus": {
        # Genus-level: iNaturalist includes all descendant species in the query.
        # Labelled poisonous on purpose (safety bias): many species are bitter,
        # some contain psilocybin. Folders are split per species for later review.
        "taxa":      ["Gymnopilus"],
        "rank":      "genus",
        "class_dir": POISONOUS_DIR,
        "places":    PLACES,
    },
}

QUALITIES           = ["research", "needs_id"]
PER_PAGE            = 200
SLEEP_BETWEEN       = 1.2
IMG_SLEEP           = 0.4
MAX_IMAGES_PER_OBS  = 3
OUTPUT_DIR          = Path("mushroom_downloads")
PER_SPECIES_FOLDERS = True       # False = drop all images flat into class_dir

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

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-")


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


def count_images(folders) -> int:
    """Count images in this target's own species folders only."""
    return sum(1 for d in folders for f in d.rglob("*") if f.is_file())

# -- PER-TARGET RUN -----------------------------------------------------------

def run_target(key: str, cfg: dict):
    log.info("")
    log.info("=" * 60)
    log.info(f"TARGET: {key}  ->  {cfg['class_dir']}")
    log.info("=" * 60)

    class_dir = cfg["class_dir"]
    class_dir.mkdir(parents=True, exist_ok=True)
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
    species_dirs = set()
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

                    if PER_SPECIES_FOLDERS:
                        species_dir = class_dir / safe_name(species_name)
                        species_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        species_dir = class_dir
                    species_dirs.add(species_dir)

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

                # iNaturalist API won't serve beyond 10,000 results per query
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
    log.info(f"  Images on disk (this target's folders): {count_images(species_dirs)}")
    log.info(f"  CSVs                : {target_out.resolve()}")

# -- MAIN ---------------------------------------------------------------------

def main():
    requested = [a.lower() for a in sys.argv[1:]] or list(TARGETS)
    unknown = [r for r in requested if r not in TARGETS]
    if unknown:
        log.error(f"Unknown target(s): {unknown}. Choose from: {list(TARGETS)}")
        return

    for key in requested:
        run_target(key, TARGETS[key])

    log.info("")
    log.info("NEXT STEPS:")
    log.info("  1. Review the new folders for bad photos (wrong subject, blurry, lookalikes)")
    log.info("  2. Gymnopilus: needs_id observations may be misidentified; filter by")
    log.info("     quality in observations.csv if you want cleaner poisonous labels")
    log.info("  3. Merge into your classifier dataset")


if __name__ == "__main__":
    main()