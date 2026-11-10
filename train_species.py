"""
train_species.py — Mushroom Species Identification Model Trainer
=================================================================
Trains the Stage-2 MobileNetV2 species classifier (`models/mushroom_species_model.h5`)
from subfolders inside `species_dataset/`.

Pre-configured subfolders (created automatically if missing):
  - Termitomyces              (Mamuno — edible)
  - Auricularia               (Talingang Daga / Wood Ear — edible)
  - Pleurotus_ostreatus       (Oyster Mushroom — edible)
  - Pleurotus_pulmonarius     (Phoenix Oyster — edible)
  - Amanita_virosa            (Destroying Angel — poisonous)
  - Leucocoprinus_birnbaumii  (Flowerpot Parasol — poisonous)
  - Gymnopilus                (Rustgill — poisonous)

Any additional subfolder added to `species_dataset/<Species_Name>/` with images
will also be included automatically. Empty subfolders are skipped during training.

Usage:
    python train_species.py
    python train_species.py --epochs 20 --batch-size 32
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

from species_catalog import ensure_species_folders, get_species_info

BASE_DIR            = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "species_dataset"
MODEL_DIR           = BASE_DIR / "models"
SPECIES_MODEL_PATH  = MODEL_DIR / "mushroom_species_model.h5"
SPECIES_INDEX_PATH  = MODEL_DIR / "species_indices.npy"
SPECIES_META_PATH   = MODEL_DIR / "species_metadata.json"

IMAGE_SIZE         = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS     = 20
VALIDATION_SPLIT   = 0.2
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}


def find_populated_species_folders(dataset_dir: Path) -> dict[str, int]:
    """Return a sorted dict of {subfolder_name: image_count} for folders with >=1 image."""
    counts: dict[str, int] = {}
    if not dataset_dir.exists():
        return counts

    for subdir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        imgs = [
            f for f in subdir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if imgs:
            counts[subdir.name] = len(imgs)
    return counts


def train_species_model(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    ensure_species_folders(dataset_dir)

    populated = find_populated_species_folders(dataset_dir)
    all_subdirs = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    empty_folders = [name for name in all_subdirs if name not in populated]

    print("Checking species dataset structure...")
    for folder_name, count in populated.items():
        info = get_species_info(folder_name)
        cls_tag = f" [{info['classification']}]" if info.get("classification") else ""
        print(f"  + {folder_name} ({info['common_name']}){cls_tag}: {count} image(s)")

    if empty_folders:
        print(f"  - Skipping empty species folder(s): {', '.join(empty_folders)}")

    if len(populated) < 2:
        raise ValueError(
            f"Need images in at least 2 species folders inside '{dataset_dir}' to train "
            f"(found {len(populated)} populated folder(s)).\n"
            f"Tip: Run `python download_mushrooms.py` to download species images from iNaturalist, "
            f"or place photos inside the subfolders of '{dataset_dir}'."
        )

    class_names = list(populated.keys())
    num_classes = len(class_names)

    print("\nLoading species data (80% training with augmentations / 20% clean validation)...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        str(dataset_dir),
        classes=class_names,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = val_datagen.flow_from_directory(
        str(dataset_dir),
        classes=class_names,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    class_indices = train_generator.class_indices
    print(f"Species class mapping detected: {class_indices}")
    print(f"Training samples: {train_generator.samples} | Validation samples: {val_generator.samples}")

    print("\nBuilding species model using pre-trained MobileNetV2...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    has_val = val_generator.samples > 0
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy" if has_val else "accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"\nStarting training loop for {num_classes} species (up to {epochs} epochs)...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator if has_val else None,
        callbacks=[early_stop],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(SPECIES_MODEL_PATH))
    np.save(str(SPECIES_INDEX_PATH), class_indices)

    # Save rich metadata keyed by class index so predict.py can resolve scientific/common names
    species_metadata = {}
    for folder_name, idx in class_indices.items():
        species_metadata[str(idx)] = get_species_info(folder_name)

    SPECIES_META_PATH.write_text(json.dumps(species_metadata, indent=2), encoding="utf-8")

    best_acc = float(max(history.history.get("accuracy", [0.0])))
    best_val_acc = float(max(history.history.get("val_accuracy", [best_acc])))

    print("\n" + "=" * 50)
    print("Species Model Training Complete!")
    print(f"  Model saved to     : {SPECIES_MODEL_PATH}")
    print(f"  Indices saved to   : {SPECIES_INDEX_PATH}")
    print(f"  Metadata saved to  : {SPECIES_META_PATH}")
    print(f"  Best Train Accuracy: {best_acc * 100:.2f}%")
    print(f"  Best Val Accuracy  : {best_val_acc * 100:.2f}%")
    print("=" * 50)

    return {
        "model_path":     str(SPECIES_MODEL_PATH),
        "indices_path":   str(SPECIES_INDEX_PATH),
        "metadata_path":  str(SPECIES_META_PATH),
        "class_indices":  class_indices,
        "species_counts": populated,
        "train_accuracy": round(best_acc * 100, 2),
        "val_accuracy":   round(best_val_acc * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the Stage-2 Mushroom Species Identification Model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the species dataset directory (default: species_dataset/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    try:
        train_species_model(
            dataset_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except ValueError as exc:
        print(f"\n[WARNING] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
