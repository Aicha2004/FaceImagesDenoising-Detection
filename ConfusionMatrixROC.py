import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

BASE_DIR = r"C:\Users\dell\Desktop\project"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# load predictions
pred_path = os.path.join(RESULTS_DIR, "predictions.csv")

if not os.path.exists(pred_path):
    raise FileNotFoundError("Run Metrics_Calculation.py first to generate predictions.csv")

df = pd.read_csv(pred_path)

y_true = df["y_true"]
y_pred = df["y_pred"]
y_score = df["y_score"]

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["clean", "noise"],
    yticklabels=["clean", "noise"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# ---------- ROC Curve ----------
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

print("Saved:")
print(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
print(os.path.join(RESULTS_DIR, "roc_curve.png"))