"""
train_full_comparison.py — Full Model Training & Confusion Matrix Comparison
=============================================================================
Trains the MobileNetV2 classifier with the refined data augmentation pipeline,
evaluates both the baseline model and the newly trained model side-by-side,
and generates comparative confusion matrices and classification reports.
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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import ImageFile
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
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
BASELINE_PATH = Path("/tmp/mushroom_image_model_baseline.h5")
AUGMENTED_PATH = Path("/tmp/mushroom_image_model_augmented.h5")
OUT_DIR = Path("/tmp/eval_comparison")


def main():
    print("=" * 65)
    print("FULL MODEL TRAINING & CONFUSION MATRIX COMPARISON")
    print("=" * 65)
    print(f"Dataset path:      {DATA_DIR}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Baseline model:    {BASELINE_PATH}")
    print(f"New model target:  {AUGMENTED_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Training and Validation Generators
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

    class_indices = train_gen.class_indices
    labels_map = {int(v): k for k, v in class_indices.items()}
    target_names = [labels_map[i] for i in range(len(labels_map))]
    print(f"Class mapping:     {class_indices}")
    print(f"Training photos:   {train_gen.samples} ({len(train_gen)} batches/epoch)")
    print(f"Validation photos: {val_gen.samples} ({len(val_gen)} batches/epoch)")

    # 2. Build and Train Model
    print("\n--- STEP 1: TRAINING FULL MODEL WITH AUGMENTATIONS ---")
    if BASELINE_PATH.exists():
        new_model = tf.keras.models.load_model(str(BASELINE_PATH), compile=False)
        print("✓ Loaded baseline backbone and head weights as warm start")
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        new_model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(class_indices), activation="softmax"),
        ])

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    )

    train_start = time.perf_counter()
    new_model.fit(
        train_gen,
        epochs=12,
        validation_data=val_gen,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    print(f"✓ Training finished in {time.perf_counter() - train_start:.2f}s")
    new_model.save(str(AUGMENTED_PATH))
    print(f"✓ Saved augmented model to: {AUGMENTED_PATH}")

    # 3. Comparative Evaluation
    print("\n--- STEP 2: COMPARATIVE EVALUATION ---")
    baseline_model = tf.keras.models.load_model(str(BASELINE_PATH), compile=False)

    # Clean test generator for full dataset evaluation
    clean_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    full_eval_gen = clean_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    y_true = full_eval_gen.classes

    print("\nPredicting with Baseline Model...")
    y_prob_base = baseline_model.predict(full_eval_gen, verbose=0)
    y_pred_base = np.argmax(y_prob_base, axis=1)

    print("Predicting with Augmented Model...")
    y_prob_aug = new_model.predict(full_eval_gen, verbose=0)
    y_pred_aug = np.argmax(y_prob_aug, axis=1)

    acc_base = np.mean(y_pred_base == y_true) * 100
    acc_aug = np.mean(y_pred_aug == y_true) * 100

    cm_base = confusion_matrix(y_true, y_pred_base, labels=list(range(len(target_names))))
    cm_aug = confusion_matrix(y_true, y_pred_aug, labels=list(range(len(target_names))))

    rep_base = classification_report(y_true, y_pred_base, target_names=target_names, digits=4, zero_division=0)
    rep_aug = classification_report(y_true, y_pred_aug, target_names=target_names, digits=4, zero_division=0)

    print("\n" + "=" * 65)
    print(f"BASELINE MODEL REPORT (Overall Accuracy: {acc_base:.2f}%)")
    print("=" * 65)
    print(rep_base)
    print("Confusion Matrix:\n", cm_base)

    print("\n" + "=" * 65)
    print(f"AUGMENTED MODEL REPORT (Overall Accuracy: {acc_aug:.2f}%)")
    print("=" * 65)
    print(rep_aug)
    print("Confusion Matrix:\n", cm_aug)

    # Save Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm_base,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=axes[0],
    )
    axes[0].set_title(f"Baseline Model\nAccuracy: {acc_base:.2f}%", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("Actual Label")

    sns.heatmap(
        cm_aug,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=axes[1],
    )
    axes[1].set_title(f"Augmented Model\nAccuracy: {acc_aug:.2f}%", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("Actual Label")

    plt.tight_layout()
    comparison_img_path = OUT_DIR / "confusion_matrix_comparison.png"
    plt.savefig(comparison_img_path, dpi=160)
    plt.close()
    print(f"\n✓ Saved side-by-side confusion matrix graph to: {comparison_img_path}")

    # Specific look-alike & safety analysis
    pois_idx = class_indices.get("poisonous")
    edible_idx = class_indices.get("edible")
    not_mush_idx = class_indices.get("not_mushroom")

    print("\n" + "=" * 65)
    print("LOOK-ALIKE & SAFETY-CRITICAL METRIC ANALYSIS")
    print("=" * 65)
    if pois_idx is not None and edible_idx is not None:
        base_fatal = cm_base[pois_idx, edible_idx]
        aug_fatal = cm_aug[pois_idx, edible_idx]
        print(f"1. Lethal False Positives (Poisonous classified as Edible):")
        print(f"   Baseline:  {base_fatal}")
        print(f"   Augmented: {aug_fatal}")
        if aug_fatal == 0:
            print("   ✅ Safety check passed: 0 poisonous mushrooms classified as edible.")

        base_ed_to_pois = cm_base[edible_idx, pois_idx]
        aug_ed_to_pois = cm_aug[edible_idx, pois_idx]
        print(f"\n2. False Negatives (Edible classified as Poisonous):")
        print(f"   Baseline:  {base_ed_to_pois}")
        print(f"   Augmented: {aug_ed_to_pois}")

        base_fp_edible = np.sum(cm_base[:, edible_idx]) - cm_base[edible_idx, edible_idx]
        aug_fp_edible = np.sum(cm_aug[:, edible_idx]) - cm_aug[edible_idx, edible_idx]
        print(f"\n3. Total False Positives for 'Edible' (Items incorrectly called Edible):")
        print(f"   Baseline:  {base_fp_edible} (out of {np.sum(cm_base[:, edible_idx])} predicted edible)")
        print(f"   Augmented: {aug_fp_edible} (out of {np.sum(cm_aug[:, edible_idx])} predicted edible)")

        base_fp_pois = np.sum(cm_base[:, pois_idx]) - cm_base[pois_idx, pois_idx]
        aug_fp_pois = np.sum(cm_aug[:, pois_idx]) - cm_aug[pois_idx, pois_idx]
        print(f"\n4. Total False Positives for 'Poisonous' (Items incorrectly called Poisonous):")
        print(f"   Baseline:  {base_fp_pois} (out of {np.sum(cm_base[:, pois_idx])} predicted poisonous)")
        print(f"   Augmented: {aug_fp_pois} (out of {np.sum(cm_aug[:, pois_idx])} predicted poisonous)")

    print("=" * 65 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

