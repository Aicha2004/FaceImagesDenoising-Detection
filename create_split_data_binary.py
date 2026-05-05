import os
import random
import shutil

BASE_DIR = r"C:\Users\dell\Desktop\project"

SRC_DIR = os.path.join(BASE_DIR, "dataset", "images")
OUT_DIR = os.path.join(BASE_DIR, "split_data_binary")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, split, "clean"), exist_ok=True)

files = [
    f for f in os.listdir(SRC_DIR)
    if f.lower().endswith(valid_ext)
]

random.seed(42)
random.shuffle(files)

n = len(files)
train_end = int(TRAIN_RATIO * n)
val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

split_map = {
    "train": files[:train_end],
    "val": files[train_end:val_end],
    "test": files[val_end:]
}

for split, split_files in split_map.items():
    out_dir = os.path.join(OUT_DIR, split, "clean")

    for fname in split_files:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(out_dir, fname)

        if not os.path.exists(dst):
            shutil.copy(src, dst)

    print(split, len(split_files))

print("Created:", OUT_DIR)