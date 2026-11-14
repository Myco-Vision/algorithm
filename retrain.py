"""
retrain.py — Myco-Vision AI Model Retraining Script
=====================================================
Called by Flask's /retrain endpoint (or manually from CLI).
Accepts a JSON manifest of image paths + labels, copies them into
mushroom_dataset/, then runs MobileNetV2 fine-tuning for 3-class classification
(edible, not_mushroom, poisonous).

Augmentation Pipeline:
    Uses the exact unified field augmentation pipeline from `augmentations.py`:
    - `fill_mode='reflect'` (mirror padding) to eliminate edge-smearing artifacts
    - Random Erasing / Cutout: p=0.5, 1 patch, small 2%-8% area scale, random noise fill
    - Subtle photometric jitter: contrast ±15-20%, subtle saturation ±15%, hue shift clamped to [-0.03, +0.03]
    - Subtle Gaussian blur: ~25% probability, sigma 0.3 to 1.0 (phone autofocus simulation)
    - Geometric transforms: rotation ±40°, width/height shifts ±20%, zoom ±20%, horizontal flip
    - Vertical flip disabled (preserves upward mushroom growth habit)
    - Validation split (20%): clean rescale=1./255 only (zero augmentations)

Usage (CLI):
    python retrain.py --manifest /tmp/retrain_manifest.json

Manifest format:
    [
        {"path": "/absolute/path/to/image.jpg", "label": "edible"},
        {"path": "/absolute/path/to/image2.jpg", "label": "poisonous"},
        {"path": "/absolute/path/to/image3.jpg", "label": "not_mushroom"}
    ]
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure algorithm directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import ImageFile

try:
    from augmentations import field_photometry_and_occlusion
except ImportError:
    from .augmentations import field_photometry_and_occlusion

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "mushroom_dataset"
MODEL_DIR   = BASE_DIR / "models"
LOG_PATH    = BASE_DIR / "retrain_log.json"

EDIBLE_DIR       = DATASET_DIR / "edible"
POISONOUS_DIR    = DATASET_DIR / "poisonous"
NOT_MUSHROOM_DIR = DATASET_DIR / "not_mushroom"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("RETRAIN_BATCH_SIZE", "8"))
EPOCHS     = int(os.environ.get("RETRAIN_EPOCHS", "10"))
VALIDATION_SPLIT = float(os.environ.get("RETRAIN_VALIDATION_SPLIT", "0.2"))

VALID_LABELS = {"edible", "poisonous", "not_mushroom"}


# ── Logging helpers ────────────────────────────────────────────────────────────

def load_log() -> list:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except Exception:
            return []
    return []


def save_log(entries: list):
    LOG_PATH.write_text(json.dumps(entries, indent=2))


def append_log(entry: dict):
    entries = load_log()
    # Replace the active "running" placeholder written by App.py when present
    if entries and isinstance(entries[0], dict) and entries[0].get("status") == "running":
        entries[0] = entry
    else:
        entries.insert(0, entry)      # newest first
    entries = entries[:20]            # keep last 20 runs
    save_log(entries)


# ── Image ingestion ────────────────────────────────────────────────────────────

def ingest_images(manifest: list) -> dict[str, int]:
    """
    Copy new images from the manifest into mushroom_dataset/.
    Returns counts: {"edible": N, "poisonous": M, "not_mushroom": K}
    """
    EDIBLE_DIR.mkdir(parents=True, exist_ok=True)
    POISONOUS_DIR.mkdir(parents=True, exist_ok=True)
    NOT_MUSHROOM_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"edible": 0, "poisonous": 0, "not_mushroom": 0}

    for item in manifest:
        src   = Path(item["path"])
        label = item["label"].lower().strip()

        if label not in VALID_LABELS:
            print(f"[SKIP] Unknown label '{label}' for {src.name}", flush=True)
            continue

        if not src.exists():
            print(f"[SKIP] File not found: {src}", flush=True)
            continue

        if label == "edible":
            dest_dir = EDIBLE_DIR
        elif label == "poisonous":
            dest_dir = POISONOUS_DIR
        else:
            dest_dir = NOT_MUSHROOM_DIR

        dest_path = dest_dir / src.name

        # Avoid overwrite collisions by appending a timestamp suffix
        if dest_path.exists():
            stem   = src.stem
            suffix = src.suffix
            dest_path = dest_dir / f"{stem}_{int(time.time())}{suffix}"

        shutil.copy2(src, dest_path)
        counts[label] += 1
        print(f"[COPY] {label}: {src.name} → {dest_path.name}", flush=True)

    return counts


# ── Training pipeline ──────────────────────────────────────────────────────────

def run_training() -> dict:
    """
    Run MobileNetV2 transfer-learning on mushroom_dataset/ with 3 classes
    (edible, not_mushroom, poisonous).
    Returns a result dict with accuracy and image counts.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models

    # Count images per class for the log (including any species subfolders, excluding non-image files)
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    edible_count       = sum(1 for f in EDIBLE_DIR.rglob("*") if f.is_file() and f.suffix.lower() in image_exts)
    poisonous_count    = sum(1 for f in POISONOUS_DIR.rglob("*") if f.is_file() and f.suffix.lower() in image_exts)
    not_mushroom_count = sum(1 for f in NOT_MUSHROOM_DIR.rglob("*") if f.is_file() and f.suffix.lower() in image_exts)
    total_images       = edible_count + poisonous_count + not_mushroom_count

    print(
        f"\n[INFO] Dataset — edible: {edible_count} | poisonous: {poisonous_count} | "
        f"not_mushroom: {not_mushroom_count} | total: {total_images}",
        flush=True,
    )

    if total_images == 0:
        raise ValueError("No images found in mushroom_dataset/. Aborting.")
    if edible_count == 0 or poisonous_count == 0 or not_mushroom_count == 0:
        raise ValueError("All 3 classes ('edible', 'poisonous', 'not_mushroom') must have at least one image.")

    # ── Data generators (80% augmented training / 20% clean validation) ──────
    augment = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.80, 1.25],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="reflect",
        preprocessing_function=field_photometry_and_occlusion,
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    model = None
    base_model = None
    train_gen = None
    val_gen = None

    try:
        train_gen = augment.flow_from_directory(
            str(DATASET_DIR),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=42,
        )

        val_gen = val_datagen.flow_from_directory(
            str(DATASET_DIR),
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=42,
        )

        class_indices = train_gen.class_indices
        num_classes = len(class_indices)
        print(f"[INFO] Class mapping: {class_indices} ({num_classes} classes)", flush=True)

        if num_classes != 3:
            raise ValueError(f"Expected 3 classes in dataset, but found {num_classes}: {class_indices}")

        # ── Model ────────────────────────────────────────────────────────────────
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False  # freeze pre-trained weights

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # ── Training ─────────────────────────────────────────────────────────────
        has_val = val_gen.samples > 0
        early_stop = callbacks.EarlyStopping(
            monitor="val_accuracy" if has_val else "accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )

        print("[INFO] Starting training loop…", flush=True)
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen if has_val else None,
            callbacks=[early_stop],
            verbose=1,
        )

        final_accuracy     = float(history.history["accuracy"][-1])
        final_val_accuracy = float(history.history.get("val_accuracy", [final_accuracy])[-1])

        # ── Save outputs ─────────────────────────────────────────────────────────
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path   = MODEL_DIR / "mushroom_image_model.h5"
        classes_path = MODEL_DIR / "class_indices.npy"

        model.save(str(model_path))
        np.save(str(classes_path), class_indices)

        print(f"\n[DONE] Model saved → {model_path}", flush=True)

        return {
            "edible_count":       edible_count,
            "poisonous_count":    poisonous_count,
            "not_mushroom_count": not_mushroom_count,
            "total_images":       total_images,
            "train_accuracy":     round(final_accuracy * 100, 2),
            "val_accuracy":       round(final_val_accuracy * 100, 2),
        }
    finally:
        del model, base_model, train_gen, val_gen
        tf.keras.backend.clear_session()
        gc.collect()


# ── Entry point ────────────────────────────────────────────────────────────────

def run_retrain(manifest: list | None = None, started_at: str | None = None) -> dict:
    """Ingest optional manifest images, run training, and record the log entry."""
    run_started = started_at or (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    log_entry   = {"started_at": run_started, "status": "failed", "error": None}

    try:
        new_counts = {"edible": 0, "poisonous": 0, "not_mushroom": 0}
        if manifest:
            print(f"[INFO] Manifest has {len(manifest)} image(s).", flush=True)
            new_counts = ingest_images(manifest)
            print(
                f"[INFO] Added — edible: {new_counts['edible']} | "
                f"poisonous: {new_counts['poisonous']} | "
                f"not_mushroom: {new_counts['not_mushroom']}",
                flush=True,
            )

        result = run_training()

        log_entry.update({
            "status":             "done",
            "finished_at":        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "new_edible":         new_counts["edible"],
            "new_poisonous":      new_counts["poisonous"],
            "new_not_mushroom":   new_counts["not_mushroom"],
            "total_edible":       result["edible_count"],
            "total_poisonous":    result["poisonous_count"],
            "total_not_mushroom": result["not_mushroom_count"],
            "total_images":       result["total_images"],
            "train_accuracy":     result["train_accuracy"],
            "val_accuracy":       result["val_accuracy"],
        })

        append_log(log_entry)

        print("\n" + "=" * 50, flush=True)
        print("✅  Retraining complete!", flush=True)
        print(f"    Train accuracy : {result['train_accuracy']}%", flush=True)
        print(f"    Val   accuracy : {result['val_accuracy']}%", flush=True)
        print(f"    Total images   : {result['total_images']}", flush=True)
        print("=" * 50, flush=True)

        return log_entry

    except Exception as exc:
        log_entry["finished_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        log_entry["error"] = str(exc)
        print(f"\n[ERROR] Retraining failed: {exc}", file=sys.stderr, flush=True)
        append_log(log_entry)
        raise


def main():
    parser = argparse.ArgumentParser(description="Myco-Vision AI Retraining Script")
    parser.add_argument(
        "--manifest",
        required=False,
        help="Path to JSON manifest file listing images to add before retraining.",
    )
    args = parser.parse_args()

    try:
        manifest = None
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())

        run_retrain(manifest=manifest)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
