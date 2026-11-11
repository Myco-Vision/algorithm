import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
from PIL import ImageFile

from species_catalog import get_species_info

ImageFile.LOAD_TRUNCATED_IMAGES = True


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mushroom_image_model.h5"
CLASSES_PATH = BASE_DIR / "models" / "class_indices.npy"

SPECIES_MODEL_PATH = BASE_DIR / "models" / "mushroom_species_model.h5"
SPECIES_CLASSES_PATH = BASE_DIR / "models" / "species_indices.npy"
SPECIES_META_PATH = BASE_DIR / "models" / "species_metadata.json"

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5
SPECIES_THRESHOLD = float(os.environ.get("SPECIES_CONFIDENCE_THRESHOLD", "0.65"))
CALIBRATION_TEMP = float(os.environ.get("CALIBRATION_TEMPERATURE", "1.4"))
CALIBRATION_EPS = float(os.environ.get("CALIBRATION_SMOOTHING", "0.015"))


def calibrate_probabilities(probs: np.ndarray, temperature: float = CALIBRATION_TEMP, eps: float = CALIBRATION_EPS) -> np.ndarray:
    """
    Apply prior smoothing + temperature scaling to soften overconfident 99.99%
    softmax spikes into realistic calibrated probabilities without changing the winner.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    if n <= 1:
        return probs
    smoothed = (1.0 - eps) * probs + (eps / n)
    logits = np.log(np.clip(smoothed, 1e-7, 1.0)) / max(temperature, 1e-3)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


class PredictionError(Exception):
    """Raised when the image cannot be analyzed safely."""


_TF = None


def get_tensorflow():
    global _TF
    if _TF is None:
        import tensorflow as tf
        from absl import logging as absl_logging

        tf.get_logger().setLevel("ERROR")
        absl_logging.set_verbosity(absl_logging.ERROR)
        _TF = tf
    return _TF


_ASSETS = None
_ASSETS_MTIME = None


def load_assets():
    """Load the Stage-1 (edible/poisonous/not_mushroom) model + label map and reuse them."""
    global _ASSETS, _ASSETS_MTIME

    if not MODEL_PATH.exists():
        raise PredictionError(f"Model file not found: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise PredictionError(f"Class map file not found: {CLASSES_PATH}")

    current_mtime = (MODEL_PATH.stat().st_mtime, CLASSES_PATH.stat().st_mtime)
    if _ASSETS is not None and _ASSETS_MTIME == current_mtime:
        return _ASSETS

    tf = get_tensorflow()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_indices = np.load(CLASSES_PATH, allow_pickle=True).item()
    labels_map = {int(value): str(key) for key, value in class_indices.items()}
    _ASSETS = (model, labels_map)
    _ASSETS_MTIME = current_mtime
    return _ASSETS


_SPECIES_ASSETS = None
_SPECIES_ASSETS_MTIME = None


def load_species_assets():
    """
    Load the Stage-2 species classifier model + metadata if available.
    Returns (species_model, species_info_by_index) or None if not trained yet.
    """
    global _SPECIES_ASSETS, _SPECIES_ASSETS_MTIME

    if not SPECIES_MODEL_PATH.exists() or not SPECIES_CLASSES_PATH.exists():
        _SPECIES_ASSETS = None
        _SPECIES_ASSETS_MTIME = None
        return None

    current_mtime = (
        SPECIES_MODEL_PATH.stat().st_mtime,
        SPECIES_CLASSES_PATH.stat().st_mtime,
    )
    if _SPECIES_ASSETS is not None and _SPECIES_ASSETS_MTIME == current_mtime:
        return _SPECIES_ASSETS

    tf = get_tensorflow()
    species_model = tf.keras.models.load_model(SPECIES_MODEL_PATH, compile=False)
    species_indices = np.load(SPECIES_CLASSES_PATH, allow_pickle=True).item()

    saved_meta = {}
    if SPECIES_META_PATH.exists():
        try:
            saved_meta = json.loads(SPECIES_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            saved_meta = {}

    species_info_map: dict[int, dict] = {}
    for folder_name, idx in species_indices.items():
        idx_int = int(idx)
        info = saved_meta.get(str(idx_int)) or get_species_info(str(folder_name))
        species_info_map[idx_int] = info

    _SPECIES_ASSETS = (species_model, species_info_map)
    _SPECIES_ASSETS_MTIME = current_mtime
    return _SPECIES_ASSETS


def preprocess_image(image_path):
    image_path = Path(image_path).expanduser().resolve()

    if not image_path.exists():
        raise PredictionError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise PredictionError(f"Image path is not a file: {image_path}")

    try:
        tf = get_tensorflow()
        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb",
        )
    except Exception as exc:
        raise PredictionError(
            "The uploaded file is not a readable image. Use JPG, PNG, or JPEG."
        ) from exc

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0), image_path


def predict_species(image_batch, edibility_label: str | None = None) -> dict | None:
    """
    Run Stage-2 species identification on a preprocessed mushroom image batch.
    If `edibility_label` ('edible' or 'poisonous') is provided and the trained
    species model includes species of that classification, selects the most
    probable compatible species so the species name and edibility match.
    """
    assets = load_species_assets()
    if assets is None:
        return None

    species_model, species_info_map = assets
    raw_probs = species_model.predict(image_batch, verbose=0)[0]
    probs = calibrate_probabilities(raw_probs)

    candidate_indices = list(range(len(probs)))
    if edibility_label in ("edible", "poisonous"):
        compatible = [
            idx for idx in candidate_indices
            if species_info_map.get(idx, {}).get("classification") in (edibility_label, None)
        ]
        if compatible:
            candidate_indices = compatible

    best_idx = max(candidate_indices, key=lambda idx: float(probs[idx]))
    species_conf = float(probs[best_idx])
    info = species_info_map.get(best_idx) or get_species_info("Unknown")
    is_confident = species_conf >= SPECIES_THRESHOLD

    if is_confident:
        display_name = info["display_name"]
        scientific_name = info["scientific_name"]
        common_name = info["common_name"]
        species_key = info["key"]
    else:
        cls_title = (edibility_label or "mushroom").capitalize()
        display_name = f"Unknown {cls_title} Species"
        scientific_name = None
        common_name = None
        species_key = "unknown_species"

    return {
        "species_key":                species_key,
        "scientific_name":            scientific_name,
        "common_name":                common_name,
        "display_name":               display_name,
        "is_confident_species":       is_confident,
        "closest_species":            info["scientific_name"],
        "species_confidence":         round(species_conf, 4),
        "species_confidence_percent": round(species_conf * 100, 2),
        "species_threshold":          SPECIES_THRESHOLD,
        "species_raw_score":          [round(float(p), 6) for p in probs],
    }


def predict_mushroom(image_path):
    image_batch, resolved_image_path = preprocess_image(image_path)
    model, labels_map = load_assets()

    raw_predictions = model.predict(image_batch, verbose=0)[0]  # shape (3,) — softmax probs
    predictions = calibrate_probabilities(raw_predictions)
    predicted_index = int(np.argmax(predictions))
    stage1_confidence = float(predictions[predicted_index])
    label = labels_map.get(predicted_index, "unknown")
    is_mushroom = label.lower() != "not_mushroom"

    species_result = None
    if is_mushroom:
        species_result = predict_species(image_batch, edibility_label=label.lower())

    # Combine Stage-1 (edibility) and Stage-2 (species) confidence when species model is active
    if is_mushroom and species_result is not None:
        sp_conf = float(species_result["species_confidence"])
        confidence = (0.4 * stage1_confidence) + (0.6 * sp_conf)
    else:
        confidence = stage1_confidence

    if not is_mushroom:
        result_name = "Not a mushroom"
        species_name = "not_mushroom"
        scientific_name = None
        common_name = None
    elif species_result is not None:
        result_name = species_result["display_name"]
        species_name = species_result["scientific_name"] or result_name
        scientific_name = species_result["scientific_name"]
        common_name = species_result["common_name"]
    else:
        result_name = f"Unknown {label.capitalize()} Species"
        species_name = label
        scientific_name = None
        common_name = None

    return {
        "ok":                         True,
        "image":                      str(resolved_image_path),
        "label":                      label,
        "species":                    species_name,
        "result_name":                result_name,
        "scientific_name":            scientific_name,
        "common_name":                common_name,
        "is_poisonous":               label.lower() == "poisonous",
        "is_mushroom":                is_mushroom,
        "confidence":                 round(confidence, 4),
        "confidence_percent":         round(confidence * 100, 2),
        "species_confidence":         species_result["species_confidence"] if species_result else None,
        "species_confidence_percent": species_result["species_confidence_percent"] if species_result else None,
        "raw_score":                  [round(float(p), 6) for p in predictions],
        "species_raw_score":          species_result["species_raw_score"] if species_result else None,
        "threshold":                  THRESHOLD,
        "warning":                    "Do not eat wild mushrooms based only on AI prediction.",
    }


def print_human_result(result):
    label = result["label"].upper()
    confidence = result["confidence_percent"]

    print("=" * 50)
    print(f"CLASSIFICATION : {label} ({confidence:.2f}%)")
    if result.get("is_mushroom"):
        if result.get("scientific_name"):
            common = result.get("common_name")
            common_str = f" ({common})" if common and common != result["scientific_name"] else ""
            sp_conf = result.get("species_confidence_percent")
            sp_conf_str = f" — {sp_conf:.2f}% confidence" if sp_conf is not None else ""
            print(f"SPECIES        : {result['scientific_name']}{common_str}{sp_conf_str}")
        elif result.get("species_confidence_percent") is not None:
            print(
                f"SPECIES        : {result['result_name']} "
                f"(below {SPECIES_THRESHOLD * 100:.0f}% threshold: {result['species_confidence_percent']:.2f}%)"
            )
        else:
            print("SPECIES        : (Species model not trained yet — run `python train_species.py`)")
    print(f"RAW SCORE      : {result['raw_score']}")
    print(result["warning"])
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Predict mushroom edibility (Stage 1) and species name (Stage 2)."
    )
    parser.add_argument("image", nargs="?", default=str(BASE_DIR / "test_mushroom.jpg"))
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    args = parser.parse_args()

    try:
        result = predict_mushroom(args.image)
    except PredictionError as exc:
        error = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(error))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result))
    else:
        print_human_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
