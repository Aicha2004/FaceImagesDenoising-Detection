# create_data_yaml.py
import yaml
from pathlib import Path

base_path = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising")
dataset_root = base_path / "yolo_dataset"

data_yaml = {
    'path': str(dataset_root.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 1,  # Number of classes (just 'face' for now)
    'names': ['face']
}

# Save data.yaml
yaml_path = base_path / "data.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"✓ Created data.yaml at {yaml_path}")
print(f"  Path: {data_yaml['path']}")
print(f"  Train: {data_yaml['train']}")
print(f"  Val: {data_yaml['val']}")
print(f"  Classes: {data_yaml['nc']}")