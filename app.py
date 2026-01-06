import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pathlib

# Configuration
MODEL_PATH = "traffic_sign_model.h5"
LABELS_PATH = "labels.txt"
IMG_HEIGHT = 32
IMG_WIDTH = 32

st.set_page_config(page_title="Traffic Sign Recognizer", layout="wide")

@st.cache_resource
def load_model_and_labels():
    # Load Model
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load Labels
    classes = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        classes = ["Unknown" for _ in range(100)] # Fallback
        
    return model, classes

model, classes = load_model_and_labels()

st.title("🚦 Traffic Sign Recognition")
st.markdown("Upload an image or use your webcam to identify traffic signs.")

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
else:
    # Sidebar
    st.sidebar.title("Options")
    mode = st.sidebar.radio("Input Source", ["Upload Image", "Webcam"])

    def predict_image(image):
        # Preprocess
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create batch axis

        # Predict
        predictions = model.predict(img_array)
        # model output is already softmax, so predictions[0] are probabilities
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = 100 * np.max(predictions[0])
        
        if confidence < 60:
            return "No Sign Detected", confidence
        
        return classes[predicted_class_idx], confidence

    input_image = None
    
    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", width=300)
            
    elif mode == "Webcam":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            input_image = Image.open(camera_image)
            st.image(input_image, caption="Webcam Capture", width=300)

    # Perform Inference if image exists
    if input_image is not None:
        st.divider()
        with st.spinner('Analyzing...'):
            try:
                label, confidence = predict_image(input_image)
                
                st.subheader(f"Prediction: **{label}**")
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.2f}%")
                
                if confidence < 50:
                    st.warning("⚠️ Low confidence. Make sure the sign is clearly visible.")
                else:
                    st.success("✅ Sign Recognized!")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # Instructions
    with st.expander("How to add more signs?"):
        st.markdown("""
        To train on new signs:
        1. Create a new folder in `data/train` with the sign name.
        2. Add images of the sign to that folder.
        3. Add a corresponding folder in `data/val` with validation images.
        4. Run `python train.py` to retrain the model.
        5. Restart this app.
        """)
