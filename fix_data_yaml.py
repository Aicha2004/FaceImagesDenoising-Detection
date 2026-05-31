# create_data_yaml_final.py
import yaml
from pathlib import Path

base_path = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising")
dataset_root = base_path / "yolo_dataset"
data_yaml_path = base_path / "data.yaml"

# Check if dataset exists
if not dataset_root.exists():
    print(f"❌ Dataset not found at {dataset_root}")
    print("Please run create_yolo_dataset_complete.py first!")
    exit()

# Count images in each split
train_images = len(list((dataset_root / "train" / "images").glob("*.*"))) if (dataset_root / "train" / "images").exists() else 0
val_images = len(list((dataset_root / "val" / "images").glob("*.*"))) if (dataset_root / "val" / "images").exists() else 0
test_images = len(list((dataset_root / "test" / "images").glob("*.*"))) if (dataset_root / "test" / "images").exists() else 0

print(f"Dataset statistics:")
print(f"  Train images: {train_images}")
print(f"  Val images: {val_images}")
print(f"  Test images: {test_images}")

if train_images == 0:
    print("❌ No training images found! Please create the dataset first.")
    exit()

# Create data.yaml
data_yaml = {
    'path': str(dataset_root.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 1,  # Number of classes (face only)
    'names': ['face']
}

# Save data.yaml
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\n✓ Created data.yaml at: {data_yaml_path}")
print(f"  Path: {data_yaml['path']}")
print(f"  Train: {data_yaml['train']}")
print(f"  Val: {data_yaml['val']}")
print(f"  Classes: {data_yaml['nc']} - {data_yaml['names']}")