import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Configuration
DATA_DIR = pathlib.Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32
EPOCHS = 10
MODEL_PATH = "traffic_sign_model.h5"
LABELS_PATH = "labels.txt"

def train_model():
    if not TRAIN_DIR.exists():
        print(f"Error: Training directory {TRAIN_DIR} not found. Run data_setup.py first.")
        return

    # Dynamic Class Detection
    print("Detecting classes...")
    # folders in train_dir assume to be class names
    class_names = sorted([item.name for item in TRAIN_DIR.iterdir() if item.is_dir()])
    num_classes = len(class_names)
    
    if num_classes == 0:
        print("No classes found. Ensure data/train contains subdirectories with images.")
        return

    print(f"Found {num_classes} classes: {class_names}")
    
    # Save labels for the app to use
    with open(LABELS_PATH, 'w') as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Saved labels to {LABELS_PATH}")

    # Load Data
    print("Loading data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=None,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int',
        class_names=class_names # Ensure order matches
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        validation_split=None,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int',
        class_names=class_names
    )

    # Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Model Architecture (Simple CNN)
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Reduce overfitting
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()

    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
