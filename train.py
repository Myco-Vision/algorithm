import os
import numpy as np
from PIL import ImageFile
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

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
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='reflect',
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
num_classes = len(class_indices)
print(f"Class mapping detected: {class_indices}")
print(f"Training samples: {train_generator.samples} | Validation samples: {validation_generator.samples}")

# Compute balanced class weights to prevent any majority class bias
class_counts = np.bincount(train_generator.classes, minlength=num_classes)
total_train = float(train_generator.samples)
class_weights = {
    idx: float(total_train / (num_classes * max(count, 1)))
    for idx, count in enumerate(class_counts)
}
print(f"Balanced class weights: {class_weights}")


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
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# --- STEP 3: TRAIN THE MODEL (WARMUP + FINE-TUNING) ---
print("Starting Phase 1: classification head warmup...")
epochs_warmup = 12
has_val = validation_generator.samples > 0

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy' if has_val else 'accuracy',
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

history_warmup = model.fit(
    train_generator,
    epochs=epochs_warmup,
    validation_data=validation_generator if has_val else None,
    class_weight=class_weights,
    callbacks=[early_stop],
)

print("Starting Phase 2: fine-tuning top MobileNetV2 layers for texture & OOD discrimination...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

early_stop_ft = callbacks.EarlyStopping(
    monitor='val_accuracy' if has_val else 'accuracy',
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

history_ft = model.fit(
    train_generator,
    epochs=8,
    validation_data=validation_generator if has_val else None,
    class_weight=class_weights,
    callbacks=[early_stop_ft],
)


# --- STEP 4: SAVE THE OUTPUTS ---
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "mushroom_image_model.h5")
model.save(model_path)

np.save(os.path.join(MODEL_DIR, "class_indices.npy"), class_indices)

all_train_acc = history_warmup.history.get("accuracy", [0.0]) + history_ft.history.get("accuracy", [0.0])
all_val_acc = history_warmup.history.get("val_accuracy", []) + history_ft.history.get("val_accuracy", [])
best_train_acc = max(all_train_acc) * 100
best_val_acc = (max(all_val_acc) * 100) if all_val_acc else best_train_acc

print("\n" + "="*50)
print(f"Training Complete! Best weights saved to: {model_path}")
print(f"  Best Train Accuracy : {best_train_acc:.2f}%")
print(f"  Best Val Accuracy   : {best_val_acc:.2f}%")
print("="*50)
