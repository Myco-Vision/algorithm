import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
DATA_DIR = "mushroom_dataset"  # now needs 'edible', 'poisonous', AND 'not_mushroom' subfolders
MODEL_DIR = "models"

print("Checking dataset structure...")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Could not find the dataset directory: '{DATA_DIR}'. Please create it and organize your photos.")

# --- STEP 1: DATA AUGMENTATION AND LOADING ---
print("Loading data and applying image augmentations...")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_generator.class_indices
print(f"Class mapping detected: {class_indices}")


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

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)


# --- STEP 4: SAVE THE OUTPUTS ---
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "mushroom_image_model.h5")
model.save(model_path)

np.save(os.path.join(MODEL_DIR, "class_indices.npy"), class_indices)

print("\n" + "="*40)
print(f"Training Complete! Model saved to: {model_path}")
print("="*40)