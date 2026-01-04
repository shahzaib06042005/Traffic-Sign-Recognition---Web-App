import os
import zipfile
import requests
import shutil
import random
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEMP_DIR = Path("temp_download")

# GTSRB Subset URL (Using a reliable source or direct link if available, 
# for now we will use a placeholder or a common tutorial link. 
# If this fails, we will instruct user to download manually)
# Using a small subset hosted on a public repo for demo purposes would be ideal.
# But for a real project, we might want the real deal. 
# Let's try downloading the GTSRB - German Traffic Sign Recognition Benchmark
# This specific URL is often used in tutorials.
DATASET_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"

# Selected Classes (Folder IDs in GTSRB)
# 14: Stop
# 17: No entry
# 1: Speed limit (30km/h)
# 13: Yield
# 33: Turn right ahead
SELECTED_CLASSES = {
    "14": "Stop",
    "17": "No_Entry",
    "1": "Speed_Limit_30",
    "13": "Yield",
    "33": "Turn_Right"
}

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping download.")
        return
    
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def setup_data():
    # Reuse zip if exists in TEMP_DIR
    zip_path = TEMP_DIR / "images.zip"
    
    if not zip_path.exists():
        TEMP_DIR.mkdir(exist_ok=True)
        try:
            download_file(DATASET_URL, zip_path)
        except Exception as e:
            print(f"Failed to download: {e}")
            return
    else:
        print("Found existing images.zip, skipping download.")

    # Clean existing data dir if we are identifying re-run
    if TRAIN_DIR.exists():
        print("Cleaning up previous partial run...")
        try:
            shutil.rmtree(DATA_DIR)
        except Exception as e:
            print(f"Warning: Could not delete data dir: {e}")

    try:
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
             # Extract to temp dir
            zip_ref.extractall(TEMP_DIR)
            
        # Organize
        print("Organizing files...")
        base_src_path = TEMP_DIR / "GTSRB/Final_Training/Images"
        
        for class_id, class_name in SELECTED_CLASSES.items():
            folder_name = class_id.zfill(5)
            src_folder = base_src_path / folder_name
            
            if not src_folder.exists():
                print(f"Warning: Source folder for {class_name} ({folder_name}) not found.")
                continue
                
            # Create dest folders
            train_dest = TRAIN_DIR / class_name
            val_dest = VAL_DIR / class_name
            train_dest.mkdir(parents=True, exist_ok=True)
            val_dest.mkdir(parents=True, exist_ok=True)
            
            # Use COPY instead of MOVE to avoid lock issues
            images = list(src_folder.glob("*.ppm"))
            random.shuffle(images)
            split_idx = int(len(images) * 0.8)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            from PIL import Image

            for img in train_imgs:
                try:
                    # Convert to PNG
                    im = Image.open(str(img))
                    dest_path = train_dest / (img.stem + ".png")
                    im.save(str(dest_path))
                except Exception as e:
                    print(f"Error processing {img}: {e}")
            
            for img in val_imgs:
                try:
                    # Convert to PNG
                    im = Image.open(str(img))
                    dest_path = val_dest / (img.stem + ".png")
                    im.save(str(dest_path))
                except Exception as e:
                     print(f"Error processing {img}: {e}")
                
            print(f"processed {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

        print("Data setup complete.")
        
    except Exception as e:
        print(f"Failed to setup data: {e}")
    finally:
        # Cleanup temp extracted files but KEEP zip for safety?
        # Or just try to remove everything and ignore errors
        pass

if __name__ == "__main__":
    setup_data()
