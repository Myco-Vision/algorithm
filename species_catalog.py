"""
species_catalog.py — Canonical species metadata & dynamic folder resolution
============================================================================
Defines the pre-configured mushroom species (matching SpeciesSeeder.php and
download_mushrooms.py) and resolves metadata for any custom species subfolder
added inside `species_dataset/`.
"""

from pathlib import Path

# Pre-configured species keyed by their subfolder name in `species_dataset/`
SPECIES_CATALOG: dict[str, dict] = {
    "Termitomyces": {
        "scientific_name": "Termitomyces",
        "common_name":     "Mamuno",
        "classification":  "edible",
        "taxa":            ["Termitomyces"],
        "rank":            "genus",
    },
    "Auricularia": {
        "scientific_name": "Auricularia",
        "common_name":     "Talingang Daga",
        "classification":  "edible",
        "taxa":            ["Auricularia"],
        "rank":            "genus",
    },
    "Pleurotus_ostreatus": {
        "scientific_name": "Pleurotus ostreatus",
        "common_name":     "Oyster Mushroom",
        "classification":  "edible",
        "taxa":            ["Pleurotus ostreatus"],
        "rank":            "species",
    },
    "Pleurotus_pulmonarius": {
        "scientific_name": "Pleurotus pulmonarius",
        "common_name":     "Phoenix Oyster",
        "classification":  "edible",
        "taxa":            ["Pleurotus pulmonarius"],
        "rank":            "species",
    },
    "Amanita_virosa": {
        "scientific_name": "Amanita virosa",
        "common_name":     "Destroying Angel",
        "classification":  "poisonous",
        "taxa":            ["Amanita virosa"],
        "rank":            "species",
    },
    "Leucocoprinus_birnbaumii": {
        "scientific_name": "Leucocoprinus birnbaumii",
        "common_name":     "Flowerpot Parasol",
        "classification":  "poisonous",
        "taxa":            ["Leucocoprinus birnbaumii"],
        "rank":            "species",
    },
    "Gymnopilus": {
        "scientific_name": "Gymnopilus",
        "common_name":     "Rustgill",
        "classification":  "poisonous",
        "taxa":            ["Gymnopilus"],
        "rank":            "genus",
    },
}

# Case-insensitive lookup map
_NORMALIZED_LOOKUP = {
    key.lower().replace(" ", "_"): (key, meta)
    for key, meta in SPECIES_CATALOG.items()
}


def get_species_info(folder_name: str) -> dict:
    """
    Resolve metadata for a species folder name in `species_dataset/`.
    If the folder matches a pre-configured species, returns its full record.
    Otherwise, dynamically constructs metadata from the folder name.
    """
    raw = folder_name.strip()
    norm = raw.lower().replace(" ", "_")

    if norm in _NORMALIZED_LOOKUP:
        canonical_key, meta = _NORMALIZED_LOOKUP[norm]
        scientific = meta["scientific_name"]
        common = meta["common_name"]
        return {
            "key":             canonical_key,
            "scientific_name": scientific,
            "common_name":     common,
            "display_name":    scientific,
            "classification":  meta["classification"],
        }

    # Dynamic fallback for any new subfolder added to species_dataset/
    human_name = raw.replace("_", " ").replace("-", " ").strip()
    if human_name:
        human_name = human_name[0].upper() + human_name[1:]
    else:
        human_name = "Unknown Species"

    return {
        "key":             raw,
        "scientific_name": human_name,
        "common_name":     human_name,
        "display_name":    human_name,
        "classification":  None,
    }


def ensure_species_folders(species_dataset_dir: Path) -> list[Path]:
    """Create the default species subfolders inside `species_dataset/` if missing."""
    species_dataset_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for folder_name in SPECIES_CATALOG:
        folder = species_dataset_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        created.append(folder)
    return created
