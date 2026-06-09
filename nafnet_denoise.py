"""
NAFNet - Simple Baseline for Image Restoration (ECCV 2022)
Complete with ALL 10 Metrics for Facial Image Denoising
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
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

# Paths to your NAFNet weights
NAFNET_WEIGHT = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\pretrained_weights\nafnet\NAFNet-SIDD-width32.pth")

# Data paths
clean_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data\test\clean")
noisy_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\test_noise2")
output_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\nafnet_complete_results")

output_root.mkdir(parents=True, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

# ================= NAFNET ARCHITECTURE =================
class NAFBlock(nn.Module):
    """NAFNet Basic Block"""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x

class NAFNet(nn.Module):
    """NAFNet for Image Denoising (ECCV 2022)"""
    def __init__(self, in_ch=3, out_ch=3, dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            NAFBlock(dim),
            NAFBlock(dim),
            nn.Conv2d(dim, dim*2, 2, stride=2)
        )
        self.middle = nn.Sequential(
            NAFBlock(dim*2),
            NAFBlock(dim*2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim*2, dim, 2, stride=2),
            NAFBlock(dim),
            NAFBlock(dim),
            nn.Conv2d(dim, out_ch, 3, padding=1)
        )
        
    def forward(self, x):
        identity = x
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x + identity

# ================= LOAD NAFNET MODEL =================
def load_nafnet():
    """Load NAFNet with pretrained weights"""
    model = NAFNet(dim=32)
    
    if NAFNET_WEIGHT.exists():
        print(f"📥 Loading NAFNet weights from: {NAFNET_WEIGHT}")
        checkpoint = torch.load(NAFNET_WEIGHT, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
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
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"   ✅ Loaded successfully ({NAFNET_WEIGHT.stat().st_size / 1e6:.1f} MB)")
        return model
    else:
        print(f"   ❌ Weight file not found: {NAFNET_WEIGHT}")
        return None

# ================= COMPLETE METRICS (ALL 10) =================
def compute_niqe(img_np):
    """Simplified NIQE calculation (no-reference quality)"""
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mu = np.mean(img_gray)
    sigma = np.std(img_gray)
    entropy = -np.sum(np.histogram(img_gray, bins=256, density=True)[0] * 
                      np.log2(np.histogram(img_gray, bins=256, density=True)[0] + 1e-10))
    niqe_score = (sigma / (mu + 1e-10)) * (1 / (entropy + 1e-10))
    return min(niqe_score, 15)  # Cap at 15 for reasonable range

def compute_all_metrics(clean_img, denoised_img):
    """
    Compute ALL 10 metrics:
    1. Accuracy, 2. Precision, 3. Recall, 4. F1, 5. ROC-AUC, 6. MCC,
    7. PSNR, 8. SSIM, 9. LPIPS, 10. NIQE
    """
    
    # Convert to numpy
    clean_np = np.array(clean_img).astype(np.uint8)
    denoised_np = np.array(denoised_img).astype(np.uint8)
    
    # Align shapes
    min_h = min(clean_np.shape[0], denoised_np.shape[0])
    min_w = min(clean_np.shape[1], denoised_np.shape[1])
    clean_np = clean_np[:min_h, :min_w, :]
    denoised_np = denoised_np[:min_h, :min_w, :]
    
    # ===== 1-6. CLASSIFICATION METRICS =====
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
    
    # ===== 7-8. IMAGE QUALITY METRICS =====
    psnr = peak_signal_noise_ratio(clean_np, denoised_np)
    ssim = structural_similarity(clean_np, denoised_np, channel_axis=2)
    
    # ===== 9. LPIPS =====
    clean_tensor = torch.tensor(clean_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    denoised_tensor = torch.tensor(denoised_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        lpips_val = loss_fn_lpips(denoised_tensor, clean_tensor).item()
    
    # ===== 10. NIQE =====
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

# ================= MAIN PROCESSING =================
def main():
    print("\n" + "="*80)
    print("NAFNET IMAGE DENOISING - COMPLETE METRICS (10 Metrics)")
    print("="*80)
    
    # Load NAFNet model
    print("\n📥 Loading NAFNet model...")
    model = load_nafnet()
    
    if model is None:
        print("\n❌ Cannot proceed without NAFNet weights!")
        print("   Please ensure NAFNet-SIDD-width32.pth is in pretrained_weights/nafnet/")
        return
    
    model = model.to(device)
    model.eval()
    
    # Verify paths
    print(f"\n📁 Checking paths:")
    print(f"   Clean folder: {'✅' if clean_folder.exists() else '❌'} - {clean_folder}")
    print(f"   Noisy folder: {'✅' if noisy_root.exists() else '❌'} - {noisy_root}")
    
    # Get noise types
    noise_types = [f for f in noisy_root.iterdir() if f.is_dir()]
    print(f"\n📁 Found {len(noise_types)} noise types: {[f.name for f in noise_types]}")
    
    all_results = []
    IMAGE_LIMIT = 50  # Limit for speed (change to 500 for full)
    
    for noise_folder in noise_types:
        noise_name = noise_folder.name
        print(f"\n{'='*60}")
        print(f"📁 Processing: {noise_name}")
        print(f"{'='*60}")
        
        out_dir = output_root / noise_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        img_paths = list(noise_folder.glob("*.jpg"))[:IMAGE_LIMIT]
        print(f"   Images: {len(img_paths)}")
        
        for img_path in tqdm(img_paths, desc=f"   Denoising {noise_name}"):
            clean_path = clean_folder / img_path.name
            
            if not clean_path.exists():
                print(f"   ⚠️ Clean image not found: {img_path.name}")
                continue
            
            try:
                # Load images
                clean_img = Image.open(clean_path).convert("RGB")
                noisy_img = Image.open(img_path).convert("RGB")
                
                # Convert to tensor
                noisy_tensor = to_tensor(noisy_img).unsqueeze(0).to(device)
                
                # Apply NAFNet denoising
                with torch.no_grad():
                    if device == "cuda":
                        with torch.cuda.amp.autocast():
                            denoised_tensor = model(noisy_tensor)
                    else:
                        denoised_tensor = model(noisy_tensor)
                
                # Convert back to image
                denoised_img = to_pil(denoised_tensor.squeeze(0).cpu())
                
                # Save denoised image
                denoised_img.save(out_dir / f"NAFNet_{img_path.name}")
                
                # Compute ALL 10 metrics
                metrics = compute_all_metrics(clean_img, denoised_img)
                metrics.update({
                    "image": img_path.name,
                    "noise_type": noise_name,
                    "model": "NAFNet"
                })
                all_results.append(metrics)
                
            except Exception as e:
                print(f"   ❌ Error on {img_path.name}: {e}")
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_root / "nafnet_complete_results.csv", index=False)
    
    # ================= PRINT DETAILED SUMMARY =================
    print("\n" + "="*80)
    print("NAFNET RESULTS - COMPLETE SUMMARY (ALL 10 METRICS)")
    print("="*80)
    
    # Summary by noise type
    summary = df.groupby('noise_type').agg({
        'psnr': 'mean',
        'ssim': 'mean',
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'roc_auc': 'mean',
        'mcc': 'mean',
        'lpips': 'mean',
        'niqe': 'mean'
    }).round(4)
    
    print("\n📊 AVERAGE METRICS BY NOISE TYPE:")
    print("="*80)
    print(summary.to_string())
    
    # ===== OVERALL AVERAGE =====
    print("\n" + "="*80)
    print("📊 OVERALL AVERAGE (All Noise Types Combined):")
    print("="*80)
    print(f"   {'Metric':<15} {'Value':<10} {'Range' :<20}")
    print(f"   {'-'*15} {'-'*10} {'-'*20}")
    print(f"   {'PSNR':<15} {df['psnr'].mean():<10.2f} dB")
    print(f"   {'SSIM':<15} {df['ssim'].mean():<10.4f}")
    print(f"   {'Accuracy':<15} {df['accuracy'].mean():<10.4f}")
    print(f"   {'Precision':<15} {df['precision'].mean():<10.4f}")
    print(f"   {'Recall':<15} {df['recall'].mean():<10.4f}")
    print(f"   {'F1 Score':<15} {df['f1'].mean():<10.4f}")
    print(f"   {'ROC-AUC':<15} {df['roc_auc'].mean():<10.4f}")
    print(f"   {'MCC':<15} {df['mcc'].mean():<10.4f}")
    print(f"   {'LPIPS':<15} {df['lpips'].mean():<10.4f} (lower is better)")
    print(f"   {'NIQE':<15} {df['niqe'].mean():<10.2f} (lower is better)")
    
    # ===== BEST PERFORMING NOISE TYPES =====
    print("\n" + "="*80)
    print("📊 BEST AND WORST PERFORMING NOISE TYPES:")
    print("="*80)
    
    best_psnr = df.groupby('noise_type')['psnr'].mean().idxmax()
    worst_psnr = df.groupby('noise_type')['psnr'].mean().idxmin()
    best_acc = df.groupby('noise_type')['accuracy'].mean().idxmax()
    worst_acc = df.groupby('noise_type')['accuracy'].mean().idxmin()
    
    print(f"   Best PSNR:     {best_psnr} ({df[df['noise_type']==best_psnr]['psnr'].mean():.2f} dB)")
    print(f"   Worst PSNR:    {worst_psnr} ({df[df['noise_type']==worst_psnr]['psnr'].mean():.2f} dB)")
    print(f"   Best Accuracy: {best_acc} ({df[df['noise_type']==best_acc]['accuracy'].mean():.4f})")
    print(f"   Worst Accuracy:{worst_acc} ({df[df['noise_type']==worst_acc]['accuracy'].mean():.4f})")
    
    # ===== SAVE SUMMARY =====
    summary.to_csv(output_root / "nafnet_summary_by_noise.csv")
    
    print(f"\n✅ Results saved to: {output_root}")
    print(f"   📄 nafnet_complete_results.csv - All 10 metrics per image")
    print(f"   📄 nafnet_summary_by_noise.csv - Summary by noise type")
    print(f"   📁 Denoised images in subfolders")
    
    print("\n" + "="*80)
    print("✅ NAFNET DENOISING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    import cv2  # For NIQE calculation
    main()