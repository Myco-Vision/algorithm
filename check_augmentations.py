"""
check_augmentations.py — Visual Verification of Mushroom Data Augmentation
==========================================================================
Loads 2 sample images from the training dataset (one pale/white, one brightly colored),
passes each through the exact `train` augmentation pipeline 16 times, unnormalizes
the tensors back to viewable RGB range [0, 255], and saves two 4x4 grids:
  - grid_sample_1.png (Pale / white mushroom)
  - grid_sample_2.png (Brightly colored mushroom)

Usage:
    python check_augmentations.py
    python check_augmentations.py --sample1 path/to/pale.jpg --sample2 path/to/bright.jpg
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure algorithm directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure matplotlib and keras use temporary writable caches in sandbox environments
os.environ.setdefault("KERAS_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from augmentations import field_photometry_and_occlusion
except ImportError:
    from .augmentations import field_photometry_and_occlusion

IMG_SIZE = (224, 224)

# Default candidates: one pale/white specimen and one vibrant specimen
DEFAULT_PALE   = BASE_DIR / "mushroom_dataset" / "poisonous" / "obf1.jpg"
DEFAULT_BRIGHT = BASE_DIR / "mushroom_dataset" / "edible" / "cDKpL0ecOMWFfXZWEJU0VuMLU694AfKdLIQt601h.jpg"
FALLBACK_BRIGHT = BASE_DIR / "test_mushroom.jpg"


def get_exact_train_datagen():
    """Build the exact ImageDataGenerator used in train.py and train_species.py."""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.80, 1.25],          # ±15% to ±25% brightness
        horizontal_flip=True,
        vertical_flip=False,                    # Preserves upward orientation
        fill_mode="reflect",
        preprocessing_function=field_photometry_and_occlusion,
    )


def load_sample_tensor(path: Path) -> np.ndarray:
    """Load image, resize to (224, 224), and return batch array (1, 224, 224, 3) in [0, 255]."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")
    img_resized = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def generate_and_save_grid(
    image_tensor: np.ndarray,
    output_path: Path,
    title: str,
    datagen: tf.keras.preprocessing.image.ImageDataGenerator,
    num_samples: int = 16,
):
    """Passes tensor through datagen 16 times, unnormalizes, and plots a 4x4 grid."""
    print(f"Generating {num_samples} augmented variants for '{title}'...")

    # flow() returns batches scaled by rescale=1/255 and augmented
    generator = datagen.flow(image_tensor, batch_size=1, seed=None)

    augmented_images = []
    for _ in range(num_samples):
        batch = next(generator)  # shape (1, 224, 224, 3), range [0.0, 1.0]
        # Unnormalize back to 8-bit RGB range [0, 255]
        unnormalized = np.clip(batch[0] * 255.0, 0.0, 255.0).astype(np.uint8)
        augmented_images.append(unnormalized)

    # Plot 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(augmented_images[idx])
        ax.set_title(f"Aug #{idx + 1}", fontsize=10, pad=3)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved 4x4 grid to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Check mushroom data augmentation pipeline.")
    parser.add_argument("--sample1", type=Path, default=DEFAULT_PALE, help="Path to pale/white sample image")
    parser.add_argument("--sample2", type=Path, default=DEFAULT_BRIGHT, help="Path to bright colored sample image")
    parser.add_argument("--out1", type=Path, default=BASE_DIR / "grid_sample_1.png", help="Output file for sample 1 grid")
    parser.add_argument("--out2", type=Path, default=BASE_DIR / "grid_sample_2.png", help="Output file for sample 2 grid")
    args = parser.parse_args()

    # Fallback to test_mushroom.jpg if sample 2 default is not found
    sample2_path = args.sample2
    if not sample2_path.exists() and FALLBACK_BRIGHT.exists():
        sample2_path = FALLBACK_BRIGHT

    print("\n" + "=" * 60)
    print("MUSHROOM AUGMENTATION PIPELINE INSPECTION")
    print("=" * 60)
    print(f"Sample 1 (Pale/White):   {args.sample1}")
    print(f"Sample 2 (Bright Color): {sample2_path}")
    print(f"Output 1:                {args.out1}")
    print(f"Output 2:                {args.out2}\n")

    datagen = get_exact_train_datagen()

    # Process Sample 1 (Pale / White mushroom)
    tensor1 = load_sample_tensor(args.sample1)
    generate_and_save_grid(
        image_tensor=tensor1,
        output_path=args.out1,
        title=f"Sample 1 (Pale / White) — {args.sample1.name}",
        datagen=datagen,
        num_samples=16,
    )

    # Process Sample 2 (Brightly colored mushroom)
    tensor2 = load_sample_tensor(sample2_path)
    generate_and_save_grid(
        image_tensor=tensor2,
        output_path=args.out2,
        title=f"Sample 2 (Brightly Colored) — {sample2_path.name}",
        datagen=datagen,
        num_samples=16,
    )

    print("\n" + "=" * 60)
    print("Inspection complete! Review the two generated grids to check:")
    print("  1. Color fidelity: Cap/gill tones remain natural (no radical hue shifts)")
    print("  2. Cutout patches: p=0.5 subtle noise occlusions (2%-8% area, 1 patch)")
    print("  3. Border padding: 'reflect' mirror padding (no stretched edge-smearing stripes)")
    print("  4. Gaussian blur: Subtle softening simulating macro autofocus misses")
    print("  5. Brightness/contrast: Dappled sunlight vs deep shade variations")
    print("  6. Orientation: Upward gravity growth preserved (no vertical flips)")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

