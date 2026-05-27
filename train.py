"""
train.py - Mushroom Classifier Training Script
===============================================
This script trains a model to detect if a mushroom is
edible or poisonous using your own photos.

HOW TO RUN:
    python train.py

BEFORE RUNNING:
    Make sure your photos are inside:
    dataset/train/edible/       ← edible mushroom photos
    dataset/train/poisonous/    ← poisonous mushroom photos
    dataset/val/edible/
    dataset/val/poisonous/
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# SETTINGS — change these if needed
# ─────────────────────────────────────────────

DATASET_DIR  = "dataset"       # folder where your photos are
MODEL_DIR    = "models"        # folder where the trained model will be saved
OUTPUT_DIR   = "outputs"       # folder where graphs will be saved
IMG_SIZE     = (224, 224)      # size all photos will be resized to
BATCH_SIZE   = 32              # how many photos to process at once
EPOCHS       = 20              # how many times to go through all photos
THRESHOLD    = 0.3             # below this = edible, above = poisonous


# ─────────────────────────────────────────────
# STEP 1 — LOAD PHOTOS
# ─────────────────────────────────────────────

print("\n Loading photos...")

# Training photos — with augmentation (flipping, rotating, zooming)
# This helps the model learn even with fewer photos
train_datagen = ImageDataGenerator(
    rescale=1./255,            # normalize pixel values
    rotation_range=30,         # randomly rotate photos
    width_shift_range=0.2,     # randomly shift left/right
    height_shift_range=0.2,    # randomly shift up/down
    zoom_range=0.2,            # randomly zoom in/out
    horizontal_flip=True,      # randomly flip photos
    brightness_range=[0.7, 1.3] # randomly change brightness
)

# Validation photos — no augmentation, just normalize
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"        # 0 = edible, 1 = poisonous
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print(f" Found {train_gen.samples} training photos")
print(f" Found {val_gen.samples} validation photos")
print(f" Classes: {train_gen.class_indices}")  # shows edible=0, poisonous=1


# ─────────────────────────────────────────────
# STEP 2 — BUILD THE MODEL
# ─────────────────────────────────────────────

print("\n Building model...")

# Load EfficientNetB0 — already trained by Google on millions of photos
# We just add our mushroom classifier on top
base_model = EfficientNetB0(
    weights="imagenet",        # use Google's pretrained weights
    include_top=False,         # remove Google's classifier
    input_shape=(224, 224, 3)  # our photo size
)

# Freeze the base — don't change Google's weights yet
base_model.trainable = False

# Build our classifier on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),              # randomly turn off neurons to prevent overfitting
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # output: 0=edible, 1=poisonous
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),       # how many poisonous did we catch?
        tf.keras.metrics.Precision(name="precision")  # how accurate were our poisonous flags?
    ]
)

print(" Model built successfully")
model.summary()


# ─────────────────────────────────────────────
# STEP 3 — TRAIN PHASE 1
# Train only our new layers, keep base frozen
# ─────────────────────────────────────────────

print("\n  Training Phase 1 — Learning mushroom features...")

callbacks = [
    # Stop early if the model stops improving
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),

    # Save the best version of the model
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate if stuck
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)


# ─────────────────────────────────────────────
# STEP 4 — TRAIN PHASE 2
# Unfreeze the base and fine-tune together
# ─────────────────────────────────────────────

print("\n  Training Phase 2 — Fine-tuning...")

# Unfreeze last 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Use a much smaller learning rate to avoid ruining the base weights
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision")
    ]
)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)


# ─────────────────────────────────────────────
# STEP 5 — SAVE GRAPHS
# ─────────────────────────────────────────────

print("\n Saving training graphs...")

def plot_history(history, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy graph
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{title} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss graph
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{title} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"   Saved {filename}")

plot_history(history1, "Phase 1", "phase1_training.png")
plot_history(history2, "Phase 2", "phase2_training.png")


# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────

print("\n Training complete!")
print(f" Model saved to: {MODEL_DIR}/best_model.keras")
print(f" Graphs saved to: {OUTPUT_DIR}/")
print("\nNext step: Run predict.py to test your model on a photo!")
print("    python predict.py your_mushroom_photo.jpg")