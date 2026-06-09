import numpy as np
import random
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# ================= CONFIG =================
test_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data\test")
clean_subfolder = test_folder / "clean"

output_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\test_noise2")
output_folder.mkdir(parents=True, exist_ok=True)

# Number of parallel workers (adjust based on your CPU)
NUM_WORKERS = multiprocessing.cpu_count() // 2  # Use half your CPU cores
print(f"Using {NUM_WORKERS} parallel workers")

# ================= HELPER FUNCTIONS =================
def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# ================= OPTIMIZED NOISE FUNCTIONS =================
def add_gaussian(img, sigma=25):
    """Add Gaussian noise - vectorized"""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255)

def add_salt_pepper(img, prob=0.02):
    """Add salt and pepper - vectorized"""
    noisy = img.copy().astype(np.uint8)
    mask = np.random.random(img.shape[:2])
    noisy[mask < prob/2] = 0
    noisy[mask > 1 - prob/2] = 255
    return noisy

def add_speckle(img, intensity=0.2):
    """Add speckle noise - vectorized"""
    img = img.astype(np.float32)
    noise = np.random.randn(*img.shape) * intensity
    return np.clip(img + img * noise, 0, 255)

def add_poisson(img):
    """Add Poisson noise - vectorized"""
    img = img.astype(np.float32)
    vals = 2 ** np.ceil(np.log2(len(np.unique(img))))
    return np.clip(np.random.poisson(img * vals) / vals, 0, 255)

def add_motion_blur(img, k=15):
    """Add motion blur - optimized kernel"""
    img = img.astype(np.uint8)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

def add_gaussian_blur(img, k=11):
    """Add Gaussian blur - OpenCV optimized"""
    return cv2.GaussianBlur(img.astype(np.uint8), (k, k), 5)

def add_defocus_blur(img, k=15):
    """Add defocus blur"""
    return cv2.GaussianBlur(img.astype(np.uint8), (k, k), 10)

def add_zoom_blur(img):
    """Add zoom blur - simplified for speed"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, 1.15)
    return cv2.warpAffine(img.astype(np.uint8), M, (w, h))

def add_fog(img, alpha=0.4):
    """Add fog effect - vectorized"""
    img = img.astype(np.float32)
    foggy = img * (1 - alpha) + 255 * alpha
    return np.clip(foggy, 0, 255)

def add_low_light(img, factor=0.35):
    """Add low light - vectorized"""
    return np.clip(img.astype(np.float32) * factor, 0, 255)

def add_jpeg(img, quality=15):
    """Add JPEG compression"""
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

def add_rain(img, drops=300):
    """Add rain - reduced drops for speed"""
    noisy = img.astype(np.uint8).copy()
    h, w = noisy.shape[:2]
    for _ in range(drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        cv2.line(noisy, (x, y), (min(x + 2, w-1), min(y + 15, h-1)), (200, 200, 200), 1)
    return noisy

def add_shadow(img):
    """Add shadow - vectorized"""
    noisy = img.astype(np.float32).copy()
    h, w = noisy.shape[:2]
    x1 = random.randint(0, w // 3)
    noisy[:, :x1] *= 0.5
    return np.clip(noisy, 0, 255)

def add_stripe(img, stripe_width=3, gap=20):
    """Add stripe noise"""
    noisy = img.astype(np.uint8).copy()
    h, w = noisy.shape[:2]
    for i in range(0, w, gap):
        noisy[:, i:min(i+stripe_width, w)] = 255
    return noisy

def add_sensor_noise(img, intensity=15):
    """Add sensor noise"""
    img = img.astype(np.float32)
    noisy = img + np.random.normal(0, intensity, img.shape)
    hot_pixels = np.random.random(img.shape[:2]) < 0.001  # Reduced probability
    noisy[hot_pixels] = 255
    return np.clip(noisy, 0, 255)

def add_mixed(img):
    """Add mixed noise - simplified"""
    img = img.astype(np.float32)
    # Gaussian + Salt & Pepper only (faster)
    img = img + np.random.normal(0, 15, img.shape)
    mask = np.random.random(img.shape[:2])
    img[mask < 0.01] = 0
    img[mask > 0.99] = 255
    return np.clip(img, 0, 255)

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
supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

test_images = []
for ext in supported_extensions:
    test_images.extend(clean_subfolder.glob(f"*{ext}"))
    test_images.extend(clean_subfolder.glob(f"*{ext.upper()}"))

if len(test_images) == 0:
    raise FileNotFoundError(f"No images found in {clean_subfolder}")

print(f"Found {len(test_images)} images")
print(f"Generating {len(NOISES)} noise types")
print(f"Total images to generate: {len(test_images) * len(NOISES)}")

# ================= OPTIMIZED PROCESSING =================
def process_noise_type(noise_name, func):
    """Process a single noise type (for parallel execution)"""
    out_dir = output_folder / noise_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for img_path in test_images:
        try:
            img = load_image(img_path)
            noisy = func(img)
            save_image(noisy, out_dir / img_path.name)
            results.append(f"✓ {img_path.name}")
        except Exception as e:
            results.append(f"✗ {img_path.name}: {e}")
    return noise_name, results

# Sequential processing (most stable, still fast)
print("\n" + "="*60)
print("GENERATING NOISES")
print("="*60)

for noise_name, func in tqdm(NOISES.items(), desc="Noise types"):
    out_dir = output_folder / noise_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(test_images, desc=noise_name, leave=False):
        try:
            img = load_image(img_path)
            noisy = func(img)
            save_image(noisy, out_dir / img_path.name)
        except Exception as e:
            print(f"\nError on {noise_name}/{img_path.name}: {e}")

print("\n" + "="*60)
print("DONE: ALL NOISES GENERATED SUCCESSFULLY")
print("="*60)

# Optional: Print summary
print("\n📊 SUMMARY:")
for noise_name in NOISES.keys():
    out_dir = output_folder / noise_name
    count = len(list(out_dir.glob("*.jpg"))) + len(list(out_dir.glob("*.png")))
    print(f"  {noise_name}: {count} images")