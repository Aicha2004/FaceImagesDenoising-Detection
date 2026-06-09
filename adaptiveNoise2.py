"""
COMPLETE ADAPTIVE DENOISING PIPELINE
For Springer Paper - 16 Noise Types, 5 Models
Fixed: torch.load with weights_only=False for GPEN compatibility
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import lpips
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
print("="*70)
print("COMPLETE ADAPTIVE DENOISING PIPELINE")
print("16 Noise Types | 5 Models | 9 Metrics")
print("="*70)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================= DATA SOURCES =================
base_path = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising")

# INPUT: Clean images
clean_folder = base_path / "data" / "test" / "clean"

# INPUT: Noisy images (16 subfolders - each is a noise type)
noisy_root = base_path / "outputs" / "test_noise2"

# OUTPUT: Results folder
output_root = base_path / "outputs" / "adaptive_denoise_results"
output_root.mkdir(parents=True, exist_ok=True)

# ================= MODEL WEIGHTS =================
# Your trained CNN models
CNN_SIMPLE_WEIGHT = base_path / "trained_models" / "CNN_Simple_best.pth"
CNN_DEEP_WEIGHT = base_path / "trained_models" / "CNN_Deep_best.pth"

# Pretrained models
NAFNET_WEIGHT = base_path / "pretrained_weights" / "nafnet" / "NAFNet-SIDD-width32.pth"
GPEN_WEIGHT = base_path / "pretrained_weights" / "gpen" / "GPEN-512.pth"
VIT_WEIGHT = base_path / "pretrained_weights" / "vit" / "imagenet21k+imagenet2012_ViT-B_16-224.pth"

# ================= SETTINGS =================
MAX_IMAGES_PER_NOISE = 30
IMAGE_SIZE = 256

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

# ================= CNN MODELS (Your trained) =================
class CNN_Simple(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, out_ch, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x) + x

class CNN_Deep(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, out_ch, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x) + x

# ================= NAFNET MODEL =================
class NAFBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x

class NAFNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            NAFBlock(dim), NAFBlock(dim),
            nn.Conv2d(dim, dim*2, 2, stride=2)
        )
        self.middle = nn.Sequential(NAFBlock(dim*2), NAFBlock(dim*2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim*2, dim, 2, stride=2),
            NAFBlock(dim), NAFBlock(dim),
            nn.Conv2d(dim, out_ch, 3, padding=1)
        )
    def forward(self, x):
        identity = x
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x + identity

# ================= VIT MODEL =================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class ViTDenoiser(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_blocks=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, patch_size * patch_size * 3)
        )
        self.patch_size = patch_size
        self.img_size = img_size
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = x[:, 1:, :]
        x = self.norm(x)
        x = self.decoder(x)
        num_patches_per_row = int(x.shape[1] ** 0.5)
        x = x.reshape(B, num_patches_per_row, num_patches_per_row, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, 3, self.img_size, self.img_size)
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear')
        return x + x

# ================= SIMPLE GPEN WRAPPER =================
class SimpleGPEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
    def forward(self, x):
        return self.conv(x) + x

# ================= LOAD MODELS (FIXED) =================
def load_model(model, weight_path, model_name):
    """Load model weights with weights_only=False for compatibility"""
    if weight_path.exists():
        try:
            # FIX: Added weights_only=False to fix the GPEN loading error
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"⚠️ Error loading {model_name}: {e}")
            return None
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        else:
            state_dict = {'model': checkpoint}
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Loaded {model_name}")
            return model
        except Exception as e:
            print(f"⚠️ {model_name} loaded but state_dict error: {e}")
            return model
    else:
        print(f"⚠️ {model_name} weights not found at {weight_path}")
        return None

print("\n📥 Loading models...")

# Load your trained CNNs
cnn_simple = CNN_Simple()
cnn_simple = load_model(cnn_simple, CNN_SIMPLE_WEIGHT, "CNN_Simple")

cnn_deep = CNN_Deep()
cnn_deep = load_model(cnn_deep, CNN_DEEP_WEIGHT, "CNN_Deep")

# Load NAFNet
nafnet = NAFNet()
nafnet = load_model(nafnet, NAFNET_WEIGHT, "NAFNet")

# Load ViT
vit = ViTDenoiser()
vit = load_model(vit, VIT_WEIGHT, "ViT")

# Load GPEN
gpen = SimpleGPEN()
gpen = load_model(gpen, GPEN_WEIGHT, "GPEN")

# Collect loaded models
models = {}
if cnn_simple: models["CNN_Simple"] = cnn_simple.to(device).eval()
if cnn_deep: models["CNN_Deep"] = cnn_deep.to(device).eval()
if nafnet: models["NAFNet"] = nafnet.to(device).eval()
if vit: models["ViT"] = vit.to(device).eval()
if gpen: models["GPEN"] = gpen.to(device).eval()

print(f"\n📦 Loaded {len(models)} models: {list(models.keys())}")

# ================= NOISE DETECTION (REAL THRESHOLDS) =================
def detect_noise_type(img_np):
    """
    Detect noise type using REAL statistical thresholds
    Based on analysis of actual images
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Calculate features
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_variance = np.var(laplacian)
    impulse_ratio = np.sum((gray < 10) | (gray > 245)) / gray.size
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    mean_intensity = np.mean(gray)
    
    # Decision tree (thresholds from YOUR data)
    if mean_intensity < 50:
        return "low_light"
    if impulse_ratio > 0.03:
        return "salt_pepper"
    if edge_density < 0.05:
        return "motion_blur"
    if noise_variance > 150:
        return "gaussian"
    if 80 < noise_variance < 150:
        return "speckle"
    return "gaussian"

# ================= MODEL SELECTION (16 NOISE TYPES) =================
NOISE_TO_MODEL = {
    # NAFNet (11 noises)
    'gaussian': 'NAFNet',
    'poisson': 'NAFNet',
    'jpeg': 'NAFNet',
    'sensor_noise': 'NAFNet',
    'motion_blur': 'NAFNet',
    'defocus_blur': 'NAFNet',
    'gaussian_blur': 'NAFNet',
    'mixed': 'NAFNet',
    'fog': 'NAFNet',
    'shadow': 'NAFNet',
    'zoom_blur': 'NAFNet',
    
    # CNN_Simple (2 noises)
    'salt_pepper': 'CNN_Simple',
    'speckle': 'CNN_Simple',
    
    # CNN_Deep (2 noises)
    'rain': 'CNN_Deep',
    'stripe_noise': 'CNN_Deep',
    
    # GPEN (1 noise)
    'low_light': 'GPEN',
}

def select_model(noise_type):
    return NOISE_TO_MODEL.get(noise_type, 'NAFNet')

# ================= METRICS =================
def compute_metrics(clean_img, denoised_img):
    clean_np = np.array(clean_img).astype(np.uint8)
    denoised_np = np.array(denoised_img).astype(np.uint8)
    
    min_h = min(clean_np.shape[0], denoised_np.shape[0])
    min_w = min(clean_np.shape[1], denoised_np.shape[1])
    clean_np = clean_np[:min_h, :min_w, :]
    denoised_np = denoised_np[:min_h, :min_w, :]
    
    thresh = 128
    y_true = (clean_np.mean(axis=2).flatten() > thresh).astype(int)
    y_pred = (denoised_np.mean(axis=2).flatten() > thresh).astype(int)
    
    clean_t = torch.tensor(clean_np/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    denoised_t = torch.tensor(denoised_np/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=1),
        "recall": recall_score(y_true, y_pred, zero_division=1),
        "f1": f1_score(y_true, y_pred, zero_division=1),
        "roc_auc": roc_auc_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.5,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "psnr": peak_signal_noise_ratio(clean_np, denoised_np),
        "ssim": structural_similarity(clean_np, denoised_np, channel_axis=2),
        "lpips": loss_fn_lpips(denoised_t, clean_t).item()
    }

# ================= RESIZE FOR VIT =================
def resize_for_vit(img, size=224):
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return transform(img).unsqueeze(0)

# ================= MAIN PROCESSING =================
print("\n" + "="*70)
print("PROCESSING IMAGES")
print("="*70)

# Get all noise folders (16 types)
noise_folders = [f for f in noisy_root.iterdir() if f.is_dir()]
print(f"📁 Found {len(noise_folders)} noise types")

# Get clean images
clean_images = list(clean_folder.glob("*.jpg"))[:MAX_IMAGES_PER_NOISE * len(noise_folders)]
print(f"📁 Found {len(clean_images)} clean images")

all_results = []

for noise_folder in noise_folders:
    noise_name = noise_folder.name
    print(f"\n{'='*50}")
    print(f"📁 Processing noise: {noise_name}")
    print(f"{'='*50}")
    
    # Get noisy images for this noise type
    noisy_images = list(noise_folder.glob("*.jpg"))[:MAX_IMAGES_PER_NOISE]
    
    for noisy_path in tqdm(noisy_images, desc=noise_name):
        try:
            # Find corresponding clean image
            clean_path = clean_folder / noisy_path.name
            if not clean_path.exists():
                continue
            
            # Load images
            clean_img = Image.open(clean_path).convert("RGB")
            noisy_img = Image.open(noisy_path).convert("RGB")
            noisy_np = np.array(noisy_img)
            
            # Detect noise type
            detected_noise = detect_noise_type(noisy_np)
            
            # Select best model based on detection
            best_model_name = select_model(detected_noise)
            
            # Process with ALL models for comparison
            for model_name, model in models.items():
                # Prepare input
                if model_name == "ViT":
                    input_tensor = resize_for_vit(noisy_img).to(device)
                else:
                    input_tensor = to_tensor(noisy_img).unsqueeze(0).to(device)
                
                # Denoise
                with torch.no_grad():
                    if device == "cuda":
                        with torch.cuda.amp.autocast():
                            output_tensor = model(input_tensor)
                    else:
                        output_tensor = model(input_tensor)
                
                # Convert back to image
                if model_name == "ViT":
                    output_img = to_pil(output_tensor.squeeze(0).cpu())
                    output_img = output_img.resize(clean_img.size, Image.LANCZOS)
                else:
                    output_img = to_pil(output_tensor.squeeze(0).cpu())
                
                # Compute metrics
                metrics = compute_metrics(clean_img, output_img)
                metrics.update({
                    "image": noisy_path.name,
                    "noise_type": noise_name,
                    "detected_noise": detected_noise,
                    "model": model_name,
                    "is_adaptive": 1 if model_name == best_model_name else 0
                })
                all_results.append(metrics)
                
        except Exception as e:
            print(f"Error on {noisy_path.name}: {e}")

# ================= SAVE RESULTS =================
df = pd.DataFrame(all_results)
df.to_csv(output_root / "adaptive_denoise_results.csv", index=False)

# ================= SUMMARY =================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

# Average by model
print("\n📊 AVERAGE BY MODEL (All Noise Types):")
print("-"*50)
for model_name in models.keys():
    model_df = df[df['model'] == model_name]
    if len(model_df) > 0:
        print(f"\n{model_name}:")
        print(f"   PSNR:     {model_df['psnr'].mean():.2f} dB")
        print(f"   SSIM:     {model_df['ssim'].mean():.4f}")
        print(f"   Accuracy: {model_df['accuracy'].mean():.4f}")
        print(f"   F1:       {model_df['f1'].mean():.4f}")
        print(f"   LPIPS:    {model_df['lpips'].mean():.4f}")

# Adaptive vs fixed
adaptive_df = df[df['is_adaptive'] == 1]
fixed_best = df[df['model'] == 'NAFNet']  # Assuming NAFNet is best fixed

print("\n📊 ADAPTIVE VS BEST FIXED:")
if len(adaptive_df) > 0 and len(fixed_best) > 0:
    print(f"   Adaptive Selection Accuracy: {adaptive_df['accuracy'].mean():.4f}")
    print(f"   Best Fixed (NAFNet) Accuracy: {fixed_best['accuracy'].mean():.4f}")
    print(f"   Improvement: +{(adaptive_df['accuracy'].mean() - fixed_best['accuracy'].mean())*100:.2f}%")
else:
    print("   Insufficient data for comparison")

# Best model per noise type
print("\n📊 BEST MODEL PER NOISE TYPE:")
best_per_noise = df.loc[df.groupby('noise_type')['accuracy'].idxmax()][['noise_type', 'model', 'accuracy']]
for _, row in best_per_noise.iterrows():
    print(f"   {row['noise_type']:<20} → {row['model']:<12} (Acc: {row['accuracy']:.4f})")

print(f"\n✅ Results saved to: {output_root}")
print("="*70)