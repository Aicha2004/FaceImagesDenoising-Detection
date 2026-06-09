"""
CNN Denoising with Pretrained Models
Using CNN_Simple and CNN_Deep trained on CelebA dataset
Complete with ALL 10 Metrics
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import lpips
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths to your trained CNN weights
CNN_SIMPLE_WEIGHT = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\trained_models\CNN_Simple_best.pth")
CNN_DEEP_WEIGHT = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\trained_models\CNN_Deep_best.pth")

# Data paths
clean_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data\test\clean")
noisy_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\test_noise2")
output_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\cnn_results")

output_root.mkdir(parents=True, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

# ================= CNN MODELS =================

# CNN_Simple (4-layer)
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

# CNN_Deep (6-layer)
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

# ================= LOAD MODELS =================
def load_cnn_model(model_class, weight_path, model_name):
    """Load pretrained CNN model"""
    if not weight_path.exists():
        print(f"❌ Weight not found: {weight_path}")
        return None
    
    model = model_class()
    checkpoint = torch.load(weight_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    print(f"✅ Loaded {model_name} ({weight_path.stat().st_size / 1e6:.1f} MB)")
    return model

# ================= NIQE FUNCTION =================
def compute_niqe(img_np):
    """Simplified NIQE calculation"""
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mu = np.mean(img_gray)
    sigma = np.std(img_gray)
    hist, _ = np.histogram(img_gray, bins=256, density=True)
    hist = hist + 1e-10
    entropy = -np.sum(hist * np.log2(hist))
    niqe_score = (sigma / (mu + 1e-10)) * (1 / (entropy + 1e-10))
    return min(niqe_score, 15)

# ================= ALL 10 METRICS =================
def compute_all_metrics(clean_img, denoised_img):
    clean_np = np.array(clean_img).astype(np.uint8)
    denoised_np = np.array(denoised_img).astype(np.uint8)
    min_h = min(clean_np.shape[0], denoised_np.shape[0])
    min_w = min(clean_np.shape[1], denoised_np.shape[1])
    clean_np = clean_np[:min_h, :min_w, :]
    denoised_np = denoised_np[:min_h, :min_w, :]
    
    # Classification metrics
    thresh = 128
    y_true = (clean_np.mean(axis=2).flatten() > thresh).astype(int)
    y_pred = (denoised_np.mean(axis=2).flatten() > thresh).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = 0.5
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Image quality metrics
    psnr = peak_signal_noise_ratio(clean_np, denoised_np)
    ssim = structural_similarity(clean_np, denoised_np, channel_axis=2)
    
    # LPIPS
    clean_t = torch.tensor(clean_np/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    denoised_t = torch.tensor(denoised_np/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    lpips_val = loss_fn_lpips(denoised_t, clean_t).item()
    
    # NIQE
    niqe_val = compute_niqe(denoised_np)
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "mcc": round(mcc, 4),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "lpips": round(lpips_val, 4),
        "niqe": round(niqe_val, 2)
    }

# ================= MAIN =================
def main():
    print("\n" + "="*60)
    print("CNN DENOISING WITH PRETRAINED MODELS")
    print("="*60)
    
    # Load models
    print("\n📥 Loading CNN models...")
    cnn_simple = load_cnn_model(CNN_Simple, CNN_SIMPLE_WEIGHT, "CNN_Simple")
    cnn_deep = load_cnn_model(CNN_Deep, CNN_DEEP_WEIGHT, "CNN_Deep")
    
    models = {}
    if cnn_simple:
        models["CNN_Simple"] = cnn_simple
    if cnn_deep:
        models["CNN_Deep"] = cnn_deep
    
    if not models:
        print("❌ No models loaded!")
        return
    
    for model in models.values():
        model = model.to(device)
        model.eval()
    
    # Get noise types
    noise_folders = [f for f in noisy_root.iterdir() if f.is_dir()]
    print(f"\n📁 Found {len(noise_folders)} noise types")
    
    all_results = []
    IMAGE_LIMIT = 50
    
    for noise_folder in noise_folders:
        noise_name = noise_folder.name
        print(f"\n📁 Processing: {noise_name}")
        
        for model_name, model in models.items():
            out_dir = output_root / noise_name / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
        
        img_paths = list(noise_folder.glob("*.jpg"))[:IMAGE_LIMIT]
        
        for img_path in tqdm(img_paths, desc=noise_name):
            clean_path = clean_folder / img_path.name
            if not clean_path.exists():
                continue
            
            clean_img = Image.open(clean_path).convert("RGB")
            noisy_img = Image.open(img_path).convert("RGB")
            noisy_tensor = to_tensor(noisy_img).unsqueeze(0).to(device)
            
            for model_name, model in models.items():
                with torch.no_grad():
                    denoised_tensor = model(noisy_tensor)
                    denoised_img = to_pil(denoised_tensor.squeeze(0).cpu())
                
                out_dir = output_root / noise_name / model_name
                denoised_img.save(out_dir / f"{model_name}_{img_path.name}")
                
                metrics = compute_all_metrics(clean_img, denoised_img)
                metrics.update({
                    "image": img_path.name,
                    "noise_type": noise_name,
                    "model": model_name
                })
                all_results.append(metrics)
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_root / "cnn_results.csv", index=False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 CNN RESULTS SUMMARY")
    print("="*60)
    
    for model_name in models.keys():
        model_df = df[df['model'] == model_name]
        print(f"\n{model_name}:")
        print(f"   PSNR:     {model_df['psnr'].mean():.2f} dB")
        print(f"   SSIM:     {model_df['ssim'].mean():.4f}")
        print(f"   Accuracy: {model_df['accuracy'].mean():.4f}")
        print(f"   F1:       {model_df['f1'].mean():.4f}")
    
    print(f"\n✅ Results saved to: {output_root}")

if __name__ == "__main__":
    main()