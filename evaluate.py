"""
evaluate.py - Mushroom Model Evaluator
========================================
Tests your trained model against your test photos
and shows you how accurate it is.

HOW TO RUN:
    python evaluate.py

BEFORE RUNNING:
    Make sure your test photos are inside:
    dataset/test/edible/        ← edible mushroom photos
    dataset/test/poisonous/     ← poisonous mushroom photos
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

MODEL_PATH  = "models/best_model.keras"
TEST_DIR    = "dataset/test"
OUTPUT_DIR  = "outputs"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
THRESHOLD   = 0.3             # same threshold used in predict.py


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("\n Loading model...")

if not os.path.exists(MODEL_PATH):
    print(" No trained model found!")
    print(f"   Expected at: {MODEL_PATH}")
    print("   Please run train.py first.")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print(" Model loaded!\n")


# ─────────────────────────────────────────────
# LOAD TEST PHOTOS
# ─────────────────────────────────────────────

print(" Loading test photos...")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False              # keep order for accurate comparison
)

print(f" Found {test_gen.samples} test photos\n")


# ─────────────────────────────────────────────
# RUN PREDICTIONS
# ─────────────────────────────────────────────

print(" Running predictions...")

y_probs  = model.predict(test_gen, verbose=1)        # raw probabilities
y_pred   = (y_probs > THRESHOLD).astype(int).flatten()  # apply threshold
y_true   = test_gen.classes                           # actual labels


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────

print("\n" + "="*50)
print(" EVALUATION RESULTS")
print("="*50)

report = classification_report(
    y_true, y_pred,
    target_names=["Edible", "Poisonous"]
)
print(report)

# Confusion matrix numbers
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  True  Edible    (correct edible):    {tn}")
print(f"  True  Poisonous (correct poisonous): {tp}")
print(f"  False Poisonous (edible called bad): {fp}  ← false alarm")
print(f"  False Edible    (poisonous called safe): {fn}  ← DANGEROUS")
print("="*50)

if fn == 0:
    print("\n No poisonous mushrooms were missed — model is safe!")
else:
    print(f"\n  {fn} poisonous mushroom(s) were incorrectly labeled as edible.")
    print("   Consider lowering the THRESHOLD value to catch more poisonous ones.")


# ─────────────────────────────────────────────
# SAVE CONFUSION MATRIX GRAPH
# ─────────────────────────────────────────────

print("\n Saving confusion matrix graph...")

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["Edible", "Poisonous"],
    yticklabels=["Edible", "Poisonous"]
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()

graph_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(graph_path)
print(f" Saved to {graph_path}")

print("\n Evaluation complete!")