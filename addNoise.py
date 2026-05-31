import numpy as np
import random
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================= CONFIG =================
test_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data\test")
clean_subfolder = test_folder / "clean"

output_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\test_noise")
output_folder.mkdir(parents=True, exist_ok=True)

# ================= HELPER FUNCTIONS =================
def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# ================= NOISE FUNCTIONS =================
def add_gaussian(img, sigma=25):
    img = img.astype(np.float32)
    noisy = img + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255)

def add_salt_pepper(img, prob=0.02):
    noisy = img.copy()

    salt = np.random.rand(img.shape[0], img.shape[1]) < prob
    pepper = np.random.rand(img.shape[0], img.shape[1]) < prob

    noisy[salt] = 255
    noisy[pepper] = 0

    return noisy

def add_speckle(img, intensity=0.2):
    img = img.astype(np.float32)
    noisy = img + img * np.random.randn(*img.shape) * intensity
    return np.clip(noisy, 0, 255)

def add_poisson(img):
    img = img.astype(np.float32)

    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))

    noisy = np.random.poisson(img * vals) / float(vals)

    return np.clip(noisy, 0, 255)

def add_motion_blur(img, k=15):
    img = img.astype(np.uint8)

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1
    kernel /= k

    return cv2.filter2D(img, -1, kernel)

def add_gaussian_blur(img, k=11):
    img = img.astype(np.uint8)
    return cv2.GaussianBlur(img, (k, k), 5)

def add_defocus_blur(img, k=15):
    img = img.astype(np.uint8)
    return cv2.GaussianBlur(img, (k, k), 10)

def add_zoom_blur(img):
    img = img.astype(np.uint8)

    h, w, _ = img.shape

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, 0, 1.2)

    return cv2.warpAffine(img, M, (w, h))

def add_fog(img, alpha=0.4):
    img = img.astype(np.float32)

    white = np.full(img.shape, 255, dtype=np.float32)

    foggy = cv2.addWeighted(
        img,
        1 - alpha,
        white,
        alpha,
        0,
        dtype=cv2.CV_32F
    )

    return np.clip(foggy, 0, 255)

def add_low_light(img, factor=0.35):
    img = img.astype(np.float32)
    dark = img * factor
    return np.clip(dark, 0, 255)

def add_jpeg(img, quality=15):
    img = img.astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    success, encoded = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    if not success:
        return img

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return decoded_rgb

def add_rain(img, drops=400):
    noisy = img.astype(np.uint8).copy()

    h, w, _ = noisy.shape

    for _ in range(drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        length = random.randint(10, 20)

        x2 = min(x + 2, w - 1)
        y2 = min(y + length, h - 1)

        cv2.line(
            noisy,
            (x, y),
            (x2, y2),
            (220, 220, 220),
            1
        )

    return noisy

def add_shadow(img):
    noisy = img.astype(np.float32).copy()

    h, w, _ = noisy.shape

    x1 = random.randint(0, w // 2)

    noisy[:, :x1] *= 0.4

    return np.clip(noisy, 0, 255)

def add_stripe(img, stripe_width=3, gap=20):
    noisy = img.astype(np.uint8).copy()

    h, w, _ = noisy.shape

    for i in range(0, w, gap):
        noisy[:, i:i + stripe_width] = 255

    return noisy

def add_sensor_noise(img, intensity=15):
    img = img.astype(np.float32)

    noisy = img + np.random.normal(0, intensity, img.shape)

    hot_pixels = np.random.rand(
        img.shape[0],
        img.shape[1]
    ) < 0.002

    mask = np.stack(
        [hot_pixels, hot_pixels, hot_pixels],
        axis=2
    )

    noisy[mask] = 255

    return np.clip(noisy, 0, 255)

def add_mixed(img):
    noisy = add_gaussian(img)

    noisy = add_salt_pepper(
        noisy.astype(np.uint8)
    )

    noisy = add_speckle(noisy)

    return np.clip(noisy, 0, 255)

# ================= NOISE DICTIONARY =================
NOISES = {
    "gaussian": add_gaussian,
    "salt_pepper": add_salt_pepper,
    "speckle": add_speckle,
    "poisson": add_poisson,
    "motion_blur": add_motion_blur,
    "gaussian_blur": add_gaussian_blur,
    "defocus_blur": add_defocus_blur,
    "zoom_blur": add_zoom_blur,
    "fog": add_fog,
    "low_light": add_low_light,
    "jpeg": add_jpeg,
    "rain": add_rain,
    "shadow": add_shadow,
    "stripe_noise": add_stripe,
    "sensor_noise": add_sensor_noise,
    "mixed": add_mixed
}

# ================= FIND IMAGES =================
supported_extensions = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
]

test_images = []

for ext in supported_extensions:
    test_images.extend(clean_subfolder.glob(f"*{ext}"))
    test_images.extend(clean_subfolder.glob(f"*{ext.upper()}"))

if len(test_images) == 0:
    raise FileNotFoundError(
        f"No images found in {clean_subfolder}"
    )

print(f"Found {len(test_images)} images")

# ================= APPLY NOISE =================
for noise_name, func in NOISES.items():

    out_dir = output_folder / noise_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(test_images, desc=noise_name):

        try:
            img = load_image(img_path)

            noisy = func(img)

            save_image(
                noisy,
                out_dir / img_path.name
            )

        except Exception as e:
            print(f"\nERROR in {noise_name} -> {img_path.name}")
            print(e)

print("\nDONE : ALL NOISES GENERATED SUCCESSFULLY")