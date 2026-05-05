import os
import shutil
import random

BASE_DIR = r"C:\Users\dell\Desktop\project"

SRC_DIR = os.path.join(BASE_DIR, "dataset", "images")
OUT_DIR = os.path.join(BASE_DIR, "split_original")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

files = [
    f for f in os.listdir(SRC_DIR)
    if f.lower().endswith(valid_ext)
]

random.shuffle(files)

n = len(files)
train_end = int(TRAIN_RATIO * n)
val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

split_files = {
    "train": files[:train_end],
    "val": files[train_end:val_end],
    "test": files[val_end:]
}

for split, names in split_files.items():
    out_dir = os.path.join(OUT_DIR, split)

    for fname in names:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(out_dir, fname)

        if not os.path.exists(dst):
            shutil.copy(src, dst)

    print(f"{split}: {len(names)} images")

print("Original clean dataset split completed.")