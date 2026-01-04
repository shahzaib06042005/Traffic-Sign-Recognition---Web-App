# Traffic Sign Recognition App

A machine learning application that recognizes traffic signs from images or webcam feed.
Built with **TensorFlow** and **Streamlit**.

## Features
- **Upload Image**: Drag and drop any traffic sign image.
- **Webcam Support**: Real-time capture from your device camera.
- **Extensible**: Easily add new signs to the dataset and retrain.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   Run the setup script to download and organize the dataset:
   ```bash
   python data_setup.py
   ```
   *Note: This downloads a subset of the GTSRB dataset.*

3. **Train Model**
   Train the CNN model on the data:
   ```bash
   python train.py
   ```
   This will generate `traffic_sign_model.h5` and `labels.txt`.

4. **Run App**
   Launch the web interface:
   ```bash
   streamlit run app.py
   ```

## Adding New Signs
To extend the model:
1. Create a new folder in `data/train/` (e.g., `data/train/Pedestrian_Crossing`).
2. Add images of the new sign to that folder.
3. Add a corresponding folder in `data/val/` with validation images.
4. Run `python train.py` again.
5. Restart the app.
