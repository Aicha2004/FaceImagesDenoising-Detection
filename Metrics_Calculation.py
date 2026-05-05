import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

BASE_DIR = r"C:\Users\dell\Desktop\project"
TEST_IMG_DIR = os.path.join(BASE_DIR, "yolo_dataset", "images", "test")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train4", "weights", "best.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

image_files = [
    f for f in os.listdir(TEST_IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

y_true = []
y_pred = []
y_score = []

for fname in image_files:
    img_path = os.path.join(TEST_IMG_DIR, fname)

    true_class = 0 if "_clean" in fname else 1
    y_true.append(true_class)

    results = model(img_path, verbose=False)
    r = results[0]

    if len(r.boxes) == 0:
        pred_class = 0
        score_noise = 0.0
    else:
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        best_idx = np.argmax(confs)
        pred_class = int(clss[best_idx])
        best_conf = float(confs[best_idx])

        score_noise = best_conf if pred_class == 1 else 1.0 - best_conf

    y_pred.append(pred_class)
    y_score.append(score_noise)

acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_score)
except ValueError:
    auc = 0.0

mcc = matthews_corrcoef(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

metrics_df = pd.DataFrame([{
    "ACC": acc,
    "PRE": pre,
    "REC": rec,
    "F1": f1,
    "AUC": auc,
    "MCC": mcc
}])

metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

cm_df = pd.DataFrame(
    cm,
    index=["clean", "noise"],
    columns=["clean", "noise"]
)
cm_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))

pred_df = pd.DataFrame({
    "filename": image_files,
    "y_true": y_true,
    "y_pred": y_pred,
    "y_score": y_score
})
pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

print(metrics_df)
print("Saved files:")
print(os.path.join(RESULTS_DIR, "metrics.csv"))
print(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))
print(os.path.join(RESULTS_DIR, "predictions.csv"))