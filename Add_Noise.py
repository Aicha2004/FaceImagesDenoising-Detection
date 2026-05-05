import cv2
import numpy as np
import os

input_dir = "dataset/images"
output_root = "noisy"

def gaussian(img):
    noise = np.random.normal(0, 25, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def salt_pepper(img):
    out = img.copy()
    prob = 0.02
    rnd = np.random.rand(*img.shape[:2])
    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255
    return out

def speckle(img):
    noise = np.random.randn(*img.shape)
    return np.clip(img + img * noise, 0, 255).astype(np.uint8)

def poisson(img):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    return np.random.poisson(img * vals) / float(vals)

def motion(img):
    kernel = np.zeros((9,9))
    kernel[int((9-1)/2), :] = np.ones(9)
    kernel /= 9
    return cv2.filter2D(img, -1, kernel)

noise_funcs = {
    "gaussian": gaussian,
    "salt_pepper": salt_pepper,
    "speckle": speckle,
    "poisson": poisson,
    "motion": motion
}

for name, func in noise_funcs.items():
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, file))
        noisy = func(img)
        cv2.imwrite(os.path.join(out_dir, file), noisy)