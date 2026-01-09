#!/usr/bin/env python3
# =====================================================
# Delete all images inside class folders
# Keeps the folder structure intact
# =====================================================

import os
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
DATASET_DIR = "/home/workstation/Documents/Azadec22b1109/ocr_project/data"  # 👈 CHANGE THIS

VALID_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

# =====================================================
# DELETE IMAGES FUNCTION
# =====================================================
def delete_images_from_class_folders(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        print("❌ Dataset directory does not exist")
        return

    total_deleted = 0

    # Loop over class folders
    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue

        deleted_in_class = 0

        for file in class_dir.iterdir():
            if file.is_file() and file.suffix.lower() in VALID_EXTS:
                file.unlink()  # delete file
                deleted_in_class += 1
                total_deleted += 1

        print(f"[OK] {class_dir.name}: deleted {deleted_in_class} images")

    print(f"\n✅ Done. Total images deleted: {total_deleted}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    delete_images_from_class_folders(DATASET_DIR)
