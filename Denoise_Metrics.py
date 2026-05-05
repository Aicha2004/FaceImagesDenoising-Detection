import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ORIG_ROOT = "dataset/images"
DENOISED_ROOT = "denoised"
RESULTS_ROOT = "results"

os.makedirs(RESULTS_ROOT, exist_ok=True)

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# classification threshold using SSIM
# if SSIM is high -> predicted clean (0)
# if SSIM is low  -> predicted noise (1)
SSIM_THRESHOLD = 0.80

rows = []
summary_rows = []

filters = [d for d in os.listdir(DENOISED_ROOT) if os.path.isdir(os.path.join(DENOISED_ROOT, d))]

for filter_name in filters:
    filter_dir = os.path.join(DENOISED_ROOT, filter_name)
    noise_types = [d for d in os.listdir(filter_dir) if os.path.isdir(os.path.join(filter_dir, d))]

    y_true = []
    y_pred = []

    psnr_list = []
    ssim_list = []

    for noise_type in noise_types:
        denoise_dir = os.path.join(filter_dir, noise_type)
        files = [f for f in os.listdir(denoise_dir) if f.lower().endswith(VALID_EXT)]

        print(f"Processing filter={filter_name}, noise={noise_type}, images={len(files)}")

        for file in files:
            orig_path = os.path.join(ORIG_ROOT, file)
            den_path = os.path.join(denoise_dir, file)

            if not os.path.exists(orig_path):
                continue

            orig = cv2.imread(orig_path)
            den = cv2.imread(den_path)

            if orig is None or den is None:
                continue

            if orig.shape != den.shape:
                den = cv2.resize(den, (orig.shape[1], orig.shape[0]))

            orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            den_gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

            # image quality metrics
            psnr_val = peak_signal_noise_ratio(orig, den, data_range=255)
            ssim_val = structural_similarity(orig_gray, den_gray, data_range=255)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            # classification:
            # denoised image comes from noisy image -> true label = 1
            true_label = 1

            # predicted label using threshold
            pred_label = 0 if ssim_val >= SSIM_THRESHOLD else 1

            y_true.append(true_label)
            y_pred.append(pred_label)

            rows.append({
                "filter": filter_name,
                "noise_type": noise_type,
                "image": file,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "true_label": true_label,
                "pred_label": pred_label
            })

    # add clean images as class 0
    clean_files = [f for f in os.listdir(ORIG_ROOT) if f.lower().endswith(VALID_EXT)]

    for file in clean_files:
        orig_path = os.path.join(ORIG_ROOT, file)
        img = cv2.imread(orig_path)

        if img is None:
            continue

        # clean image compared with itself
        true_label = 0
        pred_label = 0

        y_true.append(true_label)
        y_pred.append(pred_label)

    # classification metrics
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # average image quality metrics
    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

    summary_rows.append({
        "filter": filter_name,
        "ACC": acc,
        "PRE": pre,
        "REC": rec,
        "F1": f1,
        "PSNR": avg_psnr,
        "SSIM": avg_ssim
    })

    # save confusion matrix for each filter
    cm_df = pd.DataFrame(cm, index=["clean", "noise"], columns=["pred_clean", "pred_noise"])
    cm_df.to_csv(os.path.join(RESULTS_ROOT, f"confusion_matrix_{filter_name}.csv"))

# save per-image results
details_df = pd.DataFrame(rows)
details_df.to_csv(os.path.join(RESULTS_ROOT, "all_metrics_per_image.csv"), index=False)

# save summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RESULTS_ROOT, "all_metrics_summary.csv"), index=False)

print("\nSummary:")
print(summary_df)

print("\nSaved:")
print(os.path.join(RESULTS_ROOT, "all_metrics_per_image.csv"))
print(os.path.join(RESULTS_ROOT, "all_metrics_summary.csv"))