import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4  # Small batch size optimized for small, custom datasets
DATA_DIR = "mushroom_dataset"  # Folder containing 'edible' and 'poisonous' subfolders
MODEL_DIR = "models"

print("Checking dataset structure...")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Could not find the dataset directory: '{DATA_DIR}'. Please create it and organize your photos.")

# --- STEP 1: DATA AUGMENTATION AND LOADING ---
print("Loading data and applying image augmentations...")

# Heavy augmentation helps your 8 images mimic a much larger dataset
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    # Removed validation_split to prevent 0-image crash
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# We use the same images for a basic sanity check since data is limited
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

class_indices = train_generator.class_indices
print(f"Class mapping detected: {class_indices}")


# --- STEP 2: BUILD TRANSFER LEARNING MODEL ---
print("Building deep learning model using pre-trained MobileNetV2...")

# Load pre-trained features (trained on millions of real-world images)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base layers so we do not lose the pre-trained knowledge
base_model.trainable = False

# Add your custom mushroom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Prevents overfitting on small image counts
    layers.Dense(1, activation='sigmoid')  # Outputs 0.0 (Edible) to 1.0 (Poisonous)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
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

# Save the trained deep learning network structure and weights
model_path = os.path.join(MODEL_DIR, "mushroom_image_model.h5")
model.save(model_path)

# Save the class index dictionary so your prediction script knows which number means what
np.save(os.path.join(MODEL_DIR, "class_indices.npy"), class_indices)

print("\n" + "="*40)
print(f"Training Complete! Model saved to: {model_path}")
print("="*40)
