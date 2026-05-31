# create_yolo_dataset_complete.py
import os
import shutil
from pathlib import Path
import random
import cv2
import numpy as np

# ---------------- CONFIG ----------------
base_path = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising")
dataset_root = base_path / "yolo_dataset"

# Source images - use your denoised images or original images
source_images = base_path / "outputs" / "denoised_pipeline_existing_metrics"

# Alternative source if the above doesn't have images
if not source_images.exists() or len(list(source_images.glob("*/*.*"))) == 0:
    source_images = base_path / "data" / "train" / "clean"
    
if not source_images.exists():
    print("❌ No source images found!")
    print(f"Checked: {base_path / 'outputs' / 'denoised_pipeline_existing_metrics'}")
    print(f"Checked: {base_path / 'data' / 'train' / 'clean'}")
    exit()

# Create dataset directories
for split in ['train', 'val', 'test']:
    (dataset_root / split / 'images').mkdir(parents=True, exist_ok=True)
    (dataset_root / split / 'labels').mkdir(parents=True, exist_ok=True)

print(f"✓ Dataset root: {dataset_root}")
print(f"✓ Source images: {source_images}")

# Collect all images
all_images = []
if source_images == base_path / "outputs" / "denoised_pipeline_existing_metrics":
    # Images are in subfolders by noise type
    for noise_folder in source_images.iterdir():
        if noise_folder.is_dir():
            for img_path in noise_folder.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Add noise type prefix to avoid name conflicts
                    new_name = f"{noise_folder.name}_{img_path.name}"
                    all_images.append((img_path, new_name))
else:
    # Images are directly in the folder
    for img_path in source_images.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            all_images.append((img_path, img_path.name))

print(f"✓ Found {len(all_images)} images")

if len(all_images) == 0:
    print("❌ No images found! Please check your source path.")
    exit()

# Split into train/val/test
random.seed(42)
random.shuffle(all_images)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_idx = int(len(all_images) * train_ratio)
val_idx = int(len(all_images) * (train_ratio + val_ratio))

train_images = all_images[:train_idx]
val_images = all_images[train_idx:val_idx]
test_images = all_images[val_idx:]

# Copy images
def copy_images(images, split):
    for img_path, new_name in images:
        dest_path = dataset_root / split / 'images' / new_name
        shutil.copy2(img_path, dest_path)
    print(f"  {split}: {len(images)} images")

print("\n📁 Copying images to dataset structure:")
copy_images(train_images, 'train')
copy_images(val_images, 'val')
copy_images(test_images, 'test')

print(f"\n✓ Dataset created successfully at: {dataset_root}")