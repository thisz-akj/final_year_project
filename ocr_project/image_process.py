import cv2
import numpy as np
import os
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = "/home/workstation/Documents/Azadec22b1109/ocr_project/raw_data_2"        # 👈 CHANGE THIS
OUTPUT_DIR = "/home/workstation/Documents/Azadec22b1109/ocr_project/processed_image2" # 👈 CHANGE THIS

IMAGE_SIZE = 224

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# =====================================================
# IMAGE PROCESSING
# =====================================================
def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # ---------------------------------
    # Resize very large images first
    # ---------------------------------
    h, w = img.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        img = cv2.resize(
            img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------
    # Light denoising
    # ---------------------------------
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # ---------------------------------
    # Sharpen strokes (unsharp mask)
    # ---------------------------------
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # ---------------------------------
    # Adaptive threshold → binarize
    # ---------------------------------
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Ensure black text on white
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # ---------------------------------
    # Remove tiny noise (safe)
    # ---------------------------------
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # ---------------------------------
    # Final resize
    # ---------------------------------
    binary = cv2.resize(
        binary, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return binary


# =====================================================
# PROCESS FOLDER
# =====================================================
def process_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        processed = preprocess_image(img_path)
        if processed is None:
            continue

        out_path = output_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), processed)

        print(f"[OK] {img_path.name} → {out_path.name}")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    process_folder(INPUT_DIR, OUTPUT_DIR)
    print("\n✅ Preprocessing complete.")
