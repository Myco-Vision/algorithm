"""
train_species.py — Mushroom Species Identification Model Trainer
=================================================================
Trains the Stage-2 MobileNetV2 species classifier (`models/mushroom_species_model.h5`)
directly from the species subfolders inside `mushroom_dataset/edible/<Species>`
and `mushroom_dataset/poisonous/<Species>` (no duplicate dataset folder needed).

Pre-configured subfolders inside `mushroom_dataset/`:
  - edible/Termitomyces              (Mamuno)
  - edible/Auricularia               (Talingang Daga / Wood Ear)
  - edible/Pleurotus_ostreatus       (Oyster Mushroom)
  - edible/Pleurotus_pulmonarius     (Phoenix Oyster)
  - poisonous/Amanita_virosa            (Destroying Angel)
  - poisonous/Leucocoprinus_birnbaumii  (Flowerpot Parasol)
  - poisonous/Gymnopilus                (Rustgill)

Any additional subfolder added to `mushroom_dataset/edible/<Species_Name>/` or
`mushroom_dataset/poisonous/<Species_Name>/` with images will also be included
automatically. Empty subfolders are skipped during training.

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
from keras.src.legacy.preprocessing.image import BatchFromFilesMixin, Iterator

from species_catalog import ensure_species_folders, get_species_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR            = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "mushroom_dataset"
MODEL_DIR           = BASE_DIR / "models"
SPECIES_MODEL_PATH  = MODEL_DIR / "mushroom_species_model.h5"
SPECIES_INDEX_PATH  = MODEL_DIR / "species_indices.npy"
SPECIES_META_PATH   = MODEL_DIR / "species_metadata.json"

IMAGE_SIZE         = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS     = 20
VALIDATION_SPLIT   = 0.2
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}


class NestedSpeciesIterator(BatchFromFilesMixin, Iterator):
    """
    Keras image iterator that reads species subfolders directly from
    `mushroom_dataset/edible/<Species>` and `mushroom_dataset/poisonous/<Species>`
    without requiring a duplicate top-level directory.
    """

    allowed_class_modes = {"categorical", "binary", "sparse", "input", None}

    def __init__(
        self,
        species_dirs: list[tuple[str, str, Path]],
        image_data_generator,
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        subset=None,
        interpolation="nearest",
        keep_aspect_ratio=False,
        dtype="float32",
    ):
        super().set_processing_attrs(
            image_data_generator,
            target_size,
            color_mode,
            "channels_last",
            None,
            "",
            "png",
            subset,
            interpolation,
            keep_aspect_ratio,
        )
        self.class_mode = class_mode
        self.dtype = dtype

        class_names = [name for name, _, _ in species_dirs]
        self.num_classes = len(class_names)
        self.class_indices = {name: idx for idx, name in enumerate(class_names)}

        filepaths = []
        classes = []

        for species_name, _, folder_path in species_dirs:
            class_idx = self.class_indices[species_name]
            all_files = sorted(
                str(f.resolve())
                for f in folder_path.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if self.split:
                num_files = len(all_files)
                start = int(self.split[0] * num_files)
                stop = int(self.split[1] * num_files)
                selected_files = all_files[start:stop]
            else:
                selected_files = all_files

            filepaths.extend(selected_files)
            classes.extend([class_idx] * len(selected_files))

        self._filepaths = filepaths
        self.filenames = filepaths
        self.samples = len(filepaths)
        self.classes = np.array(classes, dtype="int32")

        print(f"Found {self.samples} images belonging to {self.num_classes} species classes ({subset or 'all'}).")
        super().__init__(self.samples, batch_size, shuffle, seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property
    def sample_weight(self):
        return None


def find_populated_species_folders(dataset_dir: Path) -> tuple[list[tuple[str, str, Path, int]], list[str]]:
    """
    Scan `mushroom_dataset/edible/*` and `mushroom_dataset/poisonous/*` for species subfolders.
    Returns:
      - populated: sorted list of (species_folder_name, parent_classification, Path, image_count)
      - empty_folders: list of empty subfolder names
    """
    populated_map: dict[str, tuple[str, str, Path, int]] = {}
    empty_folders: list[str] = []

    for cls_name in ("edible", "poisonous"):
        cls_dir = dataset_dir / cls_name
        if not cls_dir.exists():
            continue
        for subdir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            imgs = [
                f for f in subdir.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if imgs:
                populated_map[subdir.name] = (subdir.name, cls_name, subdir, len(imgs))
            else:
                empty_folders.append(f"{cls_name}/{subdir.name}")

    populated = [populated_map[k] for k in sorted(populated_map.keys())]
    return populated, empty_folders


def train_species_model(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    ensure_species_folders(dataset_dir)

    populated, empty_folders = find_populated_species_folders(dataset_dir)

    print(f"Checking species folders inside '{dataset_dir}'...")
    for folder_name, cls_name, _, count in populated:
        info = get_species_info(folder_name, parent_classification=cls_name)
        print(f"  + {cls_name}/{folder_name} ({info['common_name']}): {count} image(s)")

    if empty_folders:
        print(f"  - Skipping empty species folder(s): {', '.join(empty_folders)}")

    if len(populated) < 2:
        raise ValueError(
            f"Need images in at least 2 species subfolders inside '{dataset_dir}/edible' or "
            f"'{dataset_dir}/poisonous' to train (found {len(populated)} populated folder(s)).\n"
            f"Tip: Run `python download_mushrooms.py` to download species images from iNaturalist."
        )

    species_dirs = [(name, cls_name, path) for name, cls_name, path, _ in populated]
    num_classes = len(species_dirs)

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

    train_generator = NestedSpeciesIterator(
        species_dirs=species_dirs,
        image_data_generator=train_datagen,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_generator = NestedSpeciesIterator(
        species_dirs=species_dirs,
        image_data_generator=val_datagen,
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
    parent_cls_map = {name: cls_name for name, cls_name, _, _ in populated}
    species_metadata = {}
    for folder_name, idx in class_indices.items():
        species_metadata[str(idx)] = get_species_info(
            folder_name,
            parent_classification=parent_cls_map.get(folder_name),
        )

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
        "species_counts": {name: count for name, _, _, count in populated},
        "train_accuracy": round(best_acc * 100, 2),
        "val_accuracy":   round(best_val_acc * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the Stage-2 Mushroom Species Identification Model from mushroom_dataset/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Path to the mushroom dataset directory (default: mushroom_dataset/).",
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
