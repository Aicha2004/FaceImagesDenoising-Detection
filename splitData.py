import random
import shutil
from pathlib import Path

# ---------------- CONFIG ----------------
raw_image_dir = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\archive\img_align_celeba\img_align_celeba")
output_dir = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data")

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ---------------- FUNCTIONS ----------------
def copy_images(image_list, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_name in image_list:
        shutil.copy(src_dir / img_name, dst_dir / img_name)

# ---------------- PROCESS ----------------
random.seed(42)

# List all image files
image_files = list(raw_image_dir.glob("*.jpg")) + list(raw_image_dir.glob("*.jpeg"))
image_files = [img.name for img in image_files]

# Shuffle images
random.shuffle(image_files)
total = len(image_files)

# Compute split indices
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_images = image_files[:train_end]
val_images = image_files[train_end:val_end]
test_images = image_files[val_end:]

# Copy images to train/val/test folders
copy_images(train_images, raw_image_dir, output_dir / "train")
copy_images(val_images, raw_image_dir, output_dir / "val")
copy_images(test_images, raw_image_dir, output_dir / "test")

print("Dataset split completed!")
print(f"Train: {len(train_images)} images")
print(f"Validation: {len(val_images)} images")
print(f"Test: {len(test_images)} images")