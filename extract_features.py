import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import random

# ─── Configuration ───────────────────────────────────────────────────
DATA_DIR = "Data"
FEATURES_PATH = os.path.join("Features", "features.csv")
IMAGE_SIZE = (128, 128)
LBP_POINTS = 8
LBP_RADIUS = 1
HIST_BINS = (8, 8, 8)

# ─── Feature Extraction Functions ────────────────────────────────────
def extract_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, HIST_BINS,
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    return np.mean(lbp), np.std(lbp)

def extract_laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_contrast(gray):
    return np.std(gray)

def extract_features_from_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, IMAGE_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = extract_histogram(image)
    lbp_mean, lbp_std = extract_lbp(gray)
    lap_var = extract_laplacian_variance(gray)
    contrast = extract_contrast(gray)

    return np.concatenate([hist, [lbp_mean, lbp_std, lap_var, contrast]])

# ─── Main Dataset Construction ───────────────────────────────────────
def build_dataset():
    rows = []
    labels = {"RealArt": 0, "AiArtData": 1}

    # Step 1: load file lists
    all_files = {}
    for label_name in labels:
        folder = os.path.join(DATA_DIR, label_name)
        files = [os.path.join(folder, f)
                 for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        all_files[label_name] = files

    # Step 2: balance dataset
    min_len = min(len(files) for files in all_files.values())
    print(f"⚖️ Balancing classes to {min_len} images each")

    # Step 3: process each class
    for label_name, label_value in labels.items():
        selected_files = random.sample(all_files[label_name], min_len)
        for file_path in tqdm(selected_files, desc=f"Processing {label_name}"):
            file_name = os.path.basename(file_path)
            try:
                features = extract_features_from_image(file_path)
                rows.append([file_name, label_value] + features.tolist())
            except Exception as e:
                print(f"❌ Skipped {file_name}: {e}")

    # Step 4: build dataframe and save
    columns = (
        ["file", "label"] +
        [f"hist_{i}" for i in range(np.prod(HIST_BINS))] +
        ["lbp_mean", "lbp_std", "lap_var", "contrast"]
    )
    df = pd.DataFrame(rows, columns=columns)
    os.makedirs("Features", exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"✅ Features saved to {FEATURES_PATH}")

# ─── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    build_dataset()
