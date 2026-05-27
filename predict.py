"""
predict.py - Mushroom Photo Predictor
======================================
Give it a mushroom photo and it will tell you
if it is EDIBLE or POISONOUS.

HOW TO RUN:
    python predict.py your_photo.jpg

EXAMPLE:
    python predict.py mushroom1.jpg
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

MODEL_PATH = "models/best_model.keras"  # path to your trained model
IMG_SIZE   = (224, 224)                 # must match what you trained with
THRESHOLD  = 0.3                        # below = edible, above = poisonous
                                        # 0.3 means we flag poisonous earlier
                                        # to be safe (less risky)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(" No trained model found!")
        print(f"   Expected at: {MODEL_PATH}")
        print("   Please run train.py first.")
        sys.exit(1)

    print(" Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded!\n")
    return model


# ─────────────────────────────────────────────
# PREDICT A SINGLE PHOTO
# ─────────────────────────────────────────────

def predict(img_path, model):
    # Check if photo exists
    if not os.path.exists(img_path):
        print(f" Photo not found: {img_path}")
        sys.exit(1)

    # Load and prepare the photo
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0       # normalize
    img_array = np.expand_dims(img_array, axis=0)     # add batch dimension

    # Run prediction
    probability = model.predict(img_array, verbose=0)[0][0]

    # Interpret result
    is_poisonous = probability > THRESHOLD
    label      = "  POISONOUS" if is_poisonous else " EDIBLE"
    confidence = probability if is_poisonous else 1 - probability

    # Print result
    print("=" * 40)
    print(f"  Photo     : {img_path}")
    print(f"  Result    : {label}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Raw Score : {probability:.4f}  (threshold: {THRESHOLD})")
    print("=" * 40)

    # Safety warning
    if is_poisonous:
        print("\n  WARNING: Do NOT eat this mushroom!")
        print("   Always confirm with a real expert before consuming any wild mushroom.")
    else:
        print("\n Looks edible — but always confirm with an expert before eating!")

    return label, confidence


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Get photo path from command line
    if len(sys.argv) < 2:
        print(" Please provide a photo path.")
        print("   Usage: python predict.py your_photo.jpg")
        sys.exit(1)

    photo_path = sys.argv[1]

    model = load_model()
    predict(photo_path, model)