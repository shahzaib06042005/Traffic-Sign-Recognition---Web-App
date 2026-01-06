import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image

MODEL_PATH = "traffic_sign_model.h5"
LABELS_PATH = "labels.txt"
VAL_DIR = Path("data/val")
IMG_HEIGHT = 60
IMG_WIDTH = 60

def verify_model():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(LABELS_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        
    print(f"Classes: {classes}")
    
    # Test one image from each class
    print("\n--- Testing Sample Images ---")
    for class_name in classes:
        class_dir = VAL_DIR / class_name
        if not class_dir.exists():
            continue
            
        # Get first image
        img_path = list(class_dir.glob("*.png"))[0]
        
        # Preprocess
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = 100 * np.max(predictions[0])
        predicted_label = classes[predicted_idx]
        
        status = "[PASS]" if predicted_label == class_name else "[FAIL]"
        print(f"{status} | True: {class_name:<15} | Pred: {predicted_label:<15} | Conf: {confidence:.2f}%")
        
        if predicted_label != class_name:
            # Show top 3 for failures
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            print(f"    Top 3 for failure:")
            for idx in top_3_indices:
                print(f"    - {classes[idx]}: {100*predictions[0][idx]:.2f}%")

if __name__ == "__main__":
    verify_model()
