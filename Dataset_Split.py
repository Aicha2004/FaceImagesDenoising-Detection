import os, shutil
from sklearn.model_selection import train_test_split

src = "dataset/images"
files = os.listdir(src)

train, temp = train_test_split(files, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

def move(files, folder):
    os.makedirs(folder, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(folder, f))

move(train, "dataset/train/clean")
move(val, "dataset/val/clean")
move(test, "dataset/test/clean")