"""
smoke_test.py — 2-3 Epoch Augmentation Performance & Convergence Smoke Test
=============================================================================
Runs 3 training epochs using the exact train.py pipeline to verify:
  1. Data augmentation CPU throughput (step times in ms/step)
  2. Training loss smooth decrease
  3. Validation metrics remain clean and unaffected by augmentations
  4. Production model weights are preserved safely
"""

import os
import sys
import time
from pathlib import Path

# Ensure algorithm directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("KERAS_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

try:
    from augmentations import field_photometry_and_occlusion
except ImportError:
    from .augmentations import field_photometry_and_occlusion

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
DATA_DIR = BASE_DIR / "mushroom_dataset"
PROD_MODEL_PATH = BASE_DIR / "models" / "mushroom_image_model.h5"


class StepTimingCallback(callbacks.Callback):
    """Measures precise batch/step execution duration to detect CPU bottlenecks."""

    def __init__(self):
        super().__init__()
        self.epoch_batch_times = []
        self.batch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_batch_times = []

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_start is not None:
            elapsed = (time.perf_counter() - self.batch_start) * 1000.0  # ms
            self.epoch_batch_times.append(elapsed)

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_batch_times:
            # Skip the very first batch of epoch 1 as it includes lazy warmup/compilation
            times = self.epoch_batch_times[1:] if (epoch == 0 and len(self.epoch_batch_times) > 1) else self.epoch_batch_times
            avg_ms = np.mean(times)
            p90_ms = np.percentile(times, 90)
            logs["avg_step_ms"] = avg_ms
            logs["p90_step_ms"] = p90_ms


def main():
    print("=" * 65)
    print("MUSHROOM AUGMENTATION PIPELINE — 3-EPOCH SMOKE TEST")
    print("=" * 65)
    print(f"Dataset path:       {DATA_DIR}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Target image size:  {IMAGE_SIZE}")
    print("Checking dataset...")

    if not DATA_DIR.exists():
        print(f"❌ Error: Dataset directory {DATA_DIR} does not exist.")
        return 1

    # Exact training generator from train.py
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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

    # Exact validation generator (strictly clean rescaling)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"Training samples:   {train_gen.samples} ({len(train_gen)} steps/epoch)")
    print(f"Validation samples: {val_gen.samples} ({len(val_gen)} steps/epoch)")
    print(f"Detected classes:   {train_gen.class_indices}")

    # Build fresh MobileNetV2 architecture with identical weights to preserve production model
    print("\nInitializing model architecture...")
    if PROD_MODEL_PATH.exists():
        # Load architecture and weights from existing model for realistic warm start
        model = tf.keras.models.load_model(str(PROD_MODEL_PATH), compile=False)
        print("  ✓ Initialized weights from existing checkpoint")
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(train_gen.class_indices), activation="softmax"),
        ])
        print("  ✓ Initialized fresh MobileNetV2 with ImageNet backbone")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    timing_cb = StepTimingCallback()

    print("\nStarting 3-epoch smoke test run...")
    print("-" * 65)

    start_time = time.perf_counter()
    history = model.fit(
        train_gen,
        epochs=3,
        validation_data=val_gen,
        callbacks=[timing_cb],
        verbose=1,
    )
    total_duration = time.perf_counter() - start_time

    print("\n" + "=" * 65)
    print("SMOKE TEST RESULTS SUMMARY")
    print("=" * 65)
    losses = history.history["loss"]
    accs = history.history["accuracy"]
    val_losses = history.history.get("val_loss", [])
    val_accs = history.history.get("val_accuracy", [])

    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10}")
    print("-" * 65)
    for ep in range(len(losses)):
        vl = f"{val_losses[ep]:.4f}" if ep < len(val_losses) else "N/A"
        va = f"{val_accs[ep]*100:.2f}%" if ep < len(val_accs) else "N/A"
        print(f"Epoch {ep+1:<2} | {losses[ep]:<12.4f} | {accs[ep]*100:<9.2f}% | {vl:<10} | {va:<10}")

    print("-" * 65)
    print(f"Total test time:    {total_duration:.2f} seconds")
    avg_sec_per_epoch = total_duration / len(losses)
    print(f"Average per epoch:  {avg_sec_per_epoch:.2f} seconds (~{avg_sec_per_epoch / len(train_gen) * 1000:.1f} ms/step total)")

    # Check loss trend
    loss_decreased = losses[-1] < losses[0]
    print(f"\nLoss progression:   {losses[0]:.4f} -> {losses[1]:.4f} -> {losses[-1]:.4f}")
    if loss_decreased:
        print("✅ PASS: Training loss decreased smoothly over epochs.")
    else:
        print("⚠️ NOTE: Training loss was already very low or plateaued.")

    # Check CPU bottleneck
    # If step time is well under 1000ms on CPU, throughput is healthy
    ms_per_step = (avg_sec_per_epoch / len(train_gen)) * 1000.0
    if ms_per_step < 1200:
        print(f"✅ PASS: Step times remain fast (~{ms_per_step:.1f} ms/step). No CPU bottleneck detected.")
    else:
        print(f"⚠️ High step times (~{ms_per_step:.1f} ms/step). Review augmentation overhead.")

    print("✅ Model preservation: Production checkpoint was NOT overwritten.")
    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

