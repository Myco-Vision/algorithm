import os
import sys
from pathlib import Path
import numpy as np
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

# Ensure algorithm directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from augmentations import field_photometry_and_occlusion
except ImportError:
    from .augmentations import field_photometry_and_occlusion

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
DATA_DIR = "mushroom_dataset"  # needs 'edible', 'poisonous', AND 'not_mushroom' subfolders
MODEL_DIR = "models"

print("Checking dataset structure...")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Could not find the dataset directory: '{DATA_DIR}'. Please create it and organize your photos.")

# --- STEP 1: DATA AUGMENTATION AND 80/20 SPLIT ---
print("Loading data (80% training with augmentations / 20% clean validation)...")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.80, 1.25],
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='reflect',
    preprocessing_function=field_photometry_and_occlusion,
    validation_split=VALIDATION_SPLIT,
)

# Clean rescaling only for validation so it matches real user uploads in predict.py
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
)

validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42,
)

class_indices = train_generator.class_indices
print(f"Class mapping detected: {class_indices}")
print(f"Training samples: {train_generator.samples} | Validation samples: {validation_generator.samples}")


# --- STEP 2: BUILD TRANSFER LEARNING MODEL ---
print("Building deep learning model using pre-trained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# --- STEP 3: TRAIN THE MODEL ---
print("Starting training loop...")
epochs = 20

has_val = validation_generator.samples > 0
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy' if has_val else 'accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator if has_val else None,
    callbacks=[early_stop],
)


# --- STEP 4: SAVE THE OUTPUTS ---
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "mushroom_image_model.h5")
model.save(model_path)

np.save(os.path.join(MODEL_DIR, "class_indices.npy"), class_indices)

best_train_acc = max(history.history.get("accuracy", [0.0])) * 100
best_val_acc = max(history.history.get("val_accuracy", [best_train_acc / 100])) * 100

print("\n" + "="*50)
print(f"Training Complete! Best weights saved to: {model_path}")
print(f"  Best Train Accuracy : {best_train_acc:.2f}%")
print(f"  Best Val Accuracy   : {best_val_acc:.2f}%")
print("="*50)
