"""
evaluate.py - Mushroom Classification Model Evaluator
======================================================
Evaluates your trained MobileNetV2 classification model
against test or validation photos and computes accuracy,
confusion matrix, and safety metrics.

Usage:
    python evaluate.py
    python evaluate.py --model models/mushroom_image_model.h5 --data-dir mushroom_dataset
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# SETTINGS & DEFAULTS
# ─────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "mushroom_image_model.h5"
DEFAULT_DATA  = BASE_DIR / "mushroom_dataset"
DEFAULT_OUT   = BASE_DIR / "outputs"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32


def evaluate(model_path: Path, data_dir: Path, output_dir: Path):
    print("\n Loading model...")
    if not model_path.exists():
        print(f"❌ No trained model found at: {model_path}")
        print("   Please run train.py first.")
        return 1

    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"✅ Model loaded from: {model_path}\n")

    if not data_dir.exists():
        print(f"❌ Evaluation dataset not found at: {data_dir}")
        return 1

    print(f" Loading evaluation photos from: {data_dir}...")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = test_datagen.flow_from_directory(
        str(data_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # keep order for accurate comparison
    )

    if test_gen.samples == 0:
        print(f"❌ No evaluation images found in: {data_dir}")
        return 1

    class_indices = test_gen.class_indices
    # Map index to class label name (e.g. {0: 'edible', 1: 'not_mushroom', 2: 'poisonous'})
    labels_map = {int(v): k for k, v in class_indices.items()}
    target_names = [labels_map[i] for i in range(len(labels_map))]

    print(f" Found {test_gen.samples} photos across classes: {target_names}\n")

    print(" Running predictions...")
    y_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_gen.classes

    # Overall Accuracy
    accuracy = float(np.mean(y_pred == y_true) * 100)

    print("\n" + "=" * 55)
    print(f" EVALUATION RESULTS (Accuracy: {accuracy:.2f}%)")
    print("=" * 55)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    # Safety-critical check: Did the model classify any poisonous mushroom as edible?
    if "poisonous" in class_indices and "edible" in class_indices:
        pois_idx = class_indices["poisonous"]
        edible_idx = class_indices["edible"]
        lethal_mistakes = int(cm[pois_idx, edible_idx])
        print(" SAFETY AUDIT:")
        if lethal_mistakes == 0:
            print("  ✅ 0 poisonous mushrooms were classified as edible. (Safe)")
        else:
            print(f"  🚨 DANGER: {lethal_mistakes} poisonous mushroom(s) were misclassified as EDIBLE!")
            print("     Consider fine-tuning model thresholds or adjusting class weights.")
    print("=" * 55)

    # Save Confusion Matrix Graph
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "confusion_matrix.png"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f}%)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"\n📊 Confusion matrix graph saved to: {graph_path}")
    print(" Evaluation complete!\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Mushroom Model Evaluator")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to trained .h5 model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Path to save outputs")
    args = parser.parse_args()

    return evaluate(args.model, args.data_dir, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
