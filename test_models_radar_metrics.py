import os
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================
# PATHS
# ==========================
BASE_DIR = r"C:\Users\dell\Desktop\project"

DATA_DIR = os.path.join(BASE_DIR, "split_data_binary", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models_binary")
RESULTS_DIR = os.path.join(BASE_DIR, "results_deep_models")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# SETTINGS
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

MODELS = {
    "deit_tiny": {
        "timm_name": "deit_tiny_patch16_224",
        "path": os.path.join(MODEL_DIR, "best_deit_tiny.pth")
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "path": os.path.join(MODEL_DIR, "best_efficientnet_b0.pth")
    },
    "resnet50": {
        "timm_name": "resnet50",
        "path": os.path.join(MODEL_DIR, "best_resnet50.pth")
    },
    "vit_small": {
        "timm_name": "vit_small_patch16_224",
        "path": os.path.join(MODEL_DIR, "best_vit_small.pth")
    },
    "vit_tiny": {
        "timm_name": "vit_tiny_patch16_224",
        "path": os.path.join(MODEL_DIR, "best_vit_tiny.pth")
    }
}

# ==========================
# DATASET
# ==========================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of test images:", len(test_dataset))

if num_classes != 2:
    raise ValueError("This script is for binary classification only: clean vs noise.")

if "noise" in class_names:
    positive_class = "noise"
elif "noisy" in class_names:
    positive_class = "noisy"
else:
    positive_class = class_names[1]

positive_index = class_names.index(positive_class)

print("Positive class:", positive_class)

# ==========================
# RADAR PLOT FUNCTION
# ==========================
def save_radar_plot(model_name, metrics_dict, save_path):
    radar_metrics = ["ACC", "BACC", "PRE", "REC", "F1", "MCC", "AUC"]
    values = [metrics_dict[m] * 100 for m in radar_metrics]

    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])

    ax.set_title(f"{model_name} Metrics Radar Plot", fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================
# TEST EACH MODEL
# ==========================
all_rows = []

for model_short_name, info in MODELS.items():

    print("\n=====================================")
    print("Testing model:", model_short_name)
    print("=====================================")

    model_path = info["path"]
    timm_name = info["timm_name"]

    if not os.path.exists(model_path):
        print("Missing model file:", model_path)
        continue

    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_score.extend(probs[:, positive_index])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    y_true_binary = (y_true == positive_index).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_index, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true_binary, y_score)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    row = {
        "model": model_short_name,
        "ACC": acc,
        "BACC": bacc,
        "PRE": pre,
        "REC": rec,
        "F1": f1,
        "AUC": auc,
        "MCC": mcc,
        "total_test_images": len(y_true)
    }

    all_rows.append(row)

    # save per-model metrics CSV
    per_model_df = pd.DataFrame([row])
    per_model_df.to_csv(
        os.path.join(RESULTS_DIR, f"{model_short_name}_metrics.csv"),
        index=False
    )

    # save confusion matrix CSV
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{class_names[0]}", f"true_{class_names[1]}"],
        columns=[f"pred_{class_names[0]}", f"pred_{class_names[1]}"]
    )

    cm_df.to_csv(
        os.path.join(RESULTS_DIR, f"{model_short_name}_confusion_matrix.csv")
    )

    # save classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    with open(os.path.join(RESULTS_DIR, f"{model_short_name}_classification_report.txt"), "w") as f:
        f.write(report)

    # save radar plot
    save_radar_plot(
        model_short_name,
        row,
        os.path.join(RESULTS_DIR, f"{model_short_name}_radar_metrics.png")
    )

    print(per_model_df)
    print("Saved radar plot:", f"{model_short_name}_radar_metrics.png")

# ==========================
# SAVE FINAL SUMMARY
# ==========================
summary_df = pd.DataFrame(all_rows)

summary_path = os.path.join(RESULTS_DIR, "all_models_metrics_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nFinal Summary:")
print(summary_df)

print("\nAll results saved in:")
print(RESULTS_DIR)