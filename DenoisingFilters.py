import os
import cv2
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.signal import wiener

NOISY_ROOT = "noisy"
DENOISED_ROOT = "denoised"

os.makedirs(DENOISED_ROOT, exist_ok=True)

def adaptive_wiener_color(img):
    if len(img.shape) == 3:
        channels = []
        for c in range(3):
            ch = wiener(img[:, :, c], (5, 5))
            ch = np.clip(ch, 0, 255).astype(np.uint8)
            channels.append(ch)
        return cv2.merge(channels)
    else:
        out = wiener(img, (5, 5))
        return np.clip(out, 0, 255).astype(np.uint8)

def apply_filters(img):
    return {
        "gaussian_filter": cv2.GaussianBlur(img, (5, 5), 0),
        "median_filter": cv2.medianBlur(img, 5),
        "min_filter": minimum_filter(img, size=3).astype(np.uint8),
        "max_filter": maximum_filter(img, size=3).astype(np.uint8),
        "adaptive_filter": adaptive_wiener_color(img)
    }

noise_types = [d for d in os.listdir(NOISY_ROOT) if os.path.isdir(os.path.join(NOISY_ROOT, d))]

for noise_type in noise_types:
    in_dir = os.path.join(NOISY_ROOT, noise_type)
    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    print(f"\nProcessing noise type: {noise_type} | {len(files)} images")

    for idx, file in enumerate(files, start=1):
        img_path = os.path.join(in_dir, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipped unreadable image: {img_path}")
            continue

        filtered = apply_filters(img)

        for name, fimg in filtered.items():
            out_dir = os.path.join(DENOISED_ROOT, name, noise_type)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, file)

            if os.path.exists(out_path):
                continue

            cv2.imwrite(out_path, fimg)

        if idx % 50 == 0 or idx == len(files):
            print(f"{noise_type}: {idx}/{len(files)} done")

print("\nAll denoising finished.")