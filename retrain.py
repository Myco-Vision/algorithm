"""
retrain.py — Myco-Vision AI Model Retraining Script
=====================================================
Called by Flask's /retrain endpoint (or manually from CLI).
Accepts a JSON manifest of image paths + labels, copies them into
mushroom_dataset/, then runs MobileNetV2 fine-tuning.

Usage (CLI):
    python retrain.py --manifest /tmp/retrain_manifest.json

Manifest format:
    [
        {"path": "/absolute/path/to/image.jpg", "label": "edible"},
        {"path": "/absolute/path/to/image2.jpg", "label": "poisonous"},
        ...
    ]
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "mushroom_dataset"
MODEL_DIR   = BASE_DIR / "models"
LOG_PATH    = BASE_DIR / "retrain_log.json"

EDIBLE_DIR    = DATASET_DIR / "edible"
POISONOUS_DIR = DATASET_DIR / "poisonous"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS     = 20

VALID_LABELS = {"edible", "poisonous"}


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
    entries.insert(0, entry)          # newest first
    entries = entries[:20]            # keep last 20 runs
    save_log(entries)


# ── Image ingestion ────────────────────────────────────────────────────────────

def ingest_images(manifest: list) -> dict[str, int]:
    """
    Copy new images from the manifest into mushroom_dataset/.
    Returns counts: {"edible": N, "poisonous": M}
    """
    EDIBLE_DIR.mkdir(parents=True, exist_ok=True)
    POISONOUS_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"edible": 0, "poisonous": 0}

    for item in manifest:
        src   = Path(item["path"])
        label = item["label"].lower().strip()

        if label not in VALID_LABELS:
            print(f"[SKIP] Unknown label '{label}' for {src.name}", flush=True)
            continue

        if not src.exists():
            print(f"[SKIP] File not found: {src}", flush=True)
            continue

        dest_dir  = EDIBLE_DIR if label == "edible" else POISONOUS_DIR
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
    Run MobileNetV2 transfer-learning on mushroom_dataset/.
    Returns a result dict with accuracy and image counts.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Count images per class for the log
    edible_count    = len(list(EDIBLE_DIR.glob("*.*")))
    poisonous_count = len(list(POISONOUS_DIR.glob("*.*")))
    total_images    = edible_count + poisonous_count

    print(f"\n[INFO] Dataset — edible: {edible_count} | poisonous: {poisonous_count} | total: {total_images}", flush=True)

    if total_images == 0:
        raise ValueError("No images found in mushroom_dataset/. Aborting.")
    if edible_count == 0 or poisonous_count == 0:
        raise ValueError("Both 'edible' and 'poisonous' classes must have at least one image.")

    # ── Data generators ──────────────────────────────────────────────────────
    augment = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_gen = augment.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    val_gen = augment.flow_from_directory(
        str(DATASET_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    class_indices = train_gen.class_indices
    print(f"[INFO] Class mapping: {class_indices}", flush=True)

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
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # ── Training ─────────────────────────────────────────────────────────────
    print("[INFO] Starting training loop…", flush=True)
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
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
        "edible_count":     edible_count,
        "poisonous_count":  poisonous_count,
        "total_images":     total_images,
        "train_accuracy":   round(final_accuracy * 100, 2),
        "val_accuracy":     round(final_val_accuracy * 100, 2),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Myco-Vision AI Retraining Script")
    parser.add_argument(
        "--manifest",
        required=False,
        help="Path to JSON manifest file listing images to add before retraining.",
    )
    args = parser.parse_args()

    started_at = datetime.utcnow().isoformat() + "Z"
    log_entry  = {"started_at": started_at, "status": "failed", "error": None}

    try:
        # Optional: ingest new images from manifest
        new_counts = {"edible": 0, "poisonous": 0}
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            print(f"[INFO] Manifest has {len(manifest)} image(s).", flush=True)
            new_counts = ingest_images(manifest)
            print(f"[INFO] Added — edible: {new_counts['edible']} | poisonous: {new_counts['poisonous']}", flush=True)

        # Run the training pipeline
        result = run_training()

        log_entry.update({
            "status":           "done",
            "finished_at":      datetime.utcnow().isoformat() + "Z",
            "new_edible":       new_counts["edible"],
            "new_poisonous":    new_counts["poisonous"],
            "total_edible":     result["edible_count"],
            "total_poisonous":  result["poisonous_count"],
            "total_images":     result["total_images"],
            "train_accuracy":   result["train_accuracy"],
            "val_accuracy":     result["val_accuracy"],
        })

        print("\n" + "=" * 50, flush=True)
        print(f"✅  Retraining complete!", flush=True)
        print(f"    Train accuracy : {result['train_accuracy']}%", flush=True)
        print(f"    Val   accuracy : {result['val_accuracy']}%", flush=True)
        print(f"    Total images   : {result['total_images']}", flush=True)
        print("=" * 50, flush=True)

    except Exception as exc:
        log_entry["error"] = str(exc)
        print(f"\n[ERROR] Retraining failed: {exc}", file=sys.stderr, flush=True)
        append_log(log_entry)
        return 1

    append_log(log_entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
