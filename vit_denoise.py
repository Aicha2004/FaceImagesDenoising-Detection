"""
ViT - Vision Transformer for Image Denoising
IMPROVED VERSION - ALL 10 METRICS
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, PSNR, SSIM, LPIPS, NIQE
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance
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

# ===== AUTO-DETECT VIT WEIGHTS =====
vit_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\pretrained_weights\vit")
vit_files = list(vit_folder.glob("*.pth"))

if vit_files:
    VIT_WEIGHT = vit_files[0]
    print(f"✅ Found ViT weights: {VIT_WEIGHT.name} ({VIT_WEIGHT.stat().st_size / 1e6:.1f} MB)")
    
    checkpoint = torch.load(VIT_WEIGHT, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'cls_token' in checkpoint:
            embed_dim = checkpoint['cls_token'].shape[-1]
        elif 'model' in checkpoint and 'cls_token' in checkpoint['model']:
            embed_dim = checkpoint['model']['cls_token'].shape[-1]
        else:
            embed_dim = 768
    else:
        embed_dim = 768
    print(f"   Detected embedding dimension: {embed_dim}")
else:
    VIT_WEIGHT = None
    embed_dim = 768
    print("❌ No ViT weights found")

# Data paths
clean_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data\test\clean")
noisy_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\test_noise2")
output_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\vit_improved_results")

output_root.mkdir(parents=True, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# Initialize LPIPS
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

# ================= IMPROVED PREPROCESSING =================
def preprocess_image(img, noise_type):
    """Apply noise-specific preprocessing"""
    img_np = np.array(img).astype(np.float32)
    
    # Low light enhancement
    if noise_type == "low_light":
        gamma = 1.5
        img_np = np.power(img_np / 255.0, gamma) * 255.0
        img_np = np.clip(img_np, 0, 255)
        p2, p98 = np.percentile(img_np, (2, 98))
        img_np = np.clip((img_np - p2) / (p98 - p2) * 255, 0, 255)
    
    # Fog removal
    elif noise_type == "fog":
        img_np = cv2.convertScaleAbs(img_np, alpha=1.3, beta=10)
    
    # Stripe noise reduction
    elif noise_type == "stripe_noise":
        img_np = cv2.medianBlur(img_np.astype(np.uint8), 3)
    
    # Zoom blur sharpening
    elif noise_type == "zoom_blur":
        blurred = cv2.GaussianBlur(img_np.astype(np.uint8), (0, 0), 3.0)
        img_np = cv2.addWeighted(img_np.astype(np.uint8), 1.5, blurred, -0.5, 0)
    
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

# ================= VIT DENOISER =================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViTDenoiser(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_heads=12, num_blocks=12):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, patch_size * patch_size * 3),
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
        return x * 0.8 + x * 0.2

# ================= LOAD MODEL =================
def load_vit_model():
    if VIT_WEIGHT is None:
        return None
    
    model = ViTDenoiser(embed_dim=embed_dim, num_heads=12, num_blocks=12)
    
    print(f"📥 Loading ViT weights from: {VIT_WEIGHT}")
    try:
        checkpoint = torch.load(VIT_WEIGHT, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = {'model': checkpoint}
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"   ✅ Loaded ViT-Base")
        return model
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

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
    """Compute ALL 10 metrics"""
    
    clean_np = np.array(clean_img).astype(np.uint8)
    denoised_np = np.array(denoised_img).astype(np.uint8)
    
    min_h = min(clean_np.shape[0], denoised_np.shape[0])
    min_w = min(clean_np.shape[1], denoised_np.shape[1])
    clean_np = clean_np[:min_h, :min_w, :]
    denoised_np = denoised_np[:min_h, :min_w, :]
    
    # ===== CLASSIFICATION METRICS =====
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
    
    # ===== IMAGE QUALITY METRICS =====
    psnr = peak_signal_noise_ratio(clean_np, denoised_np)
    ssim = structural_similarity(clean_np, denoised_np, channel_axis=2)
    
    # ===== LPIPS =====
    clean_tensor = torch.tensor(clean_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    denoised_tensor = torch.tensor(denoised_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        lpips_val = loss_fn_lpips(denoised_tensor, clean_tensor).item()
    
    # ===== NIQE =====
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

# ================= RESIZE =================
def resize_to_224(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img)

# ================= MAIN =================
def main():
    print("\n" + "="*80)
    print("IMPROVED VIT - ALL 10 METRICS")
    print("Metrics: Acc, Prec, Rec, F1, AUC, MCC, PSNR, SSIM, LPIPS, NIQE")
    print("="*80)
    
    if VIT_WEIGHT is None:
        print("\n❌ No ViT weights found!")
        return
    
    model = load_vit_model()
    if model is None:
        return
    
    model = model.to(device)
    model.eval()
    
    noise_types = [f for f in noisy_root.iterdir() if f.is_dir()]
    print(f"\n📁 Found {len(noise_types)} noise types")
    
    all_results = []
    IMAGE_LIMIT = 30
    
    for noise_folder in noise_types:
        noise_name = noise_folder.name
        print(f"\n{'='*60}")
        print(f"📁 Processing: {noise_name}")
        
        out_dir = output_root / noise_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        img_paths = list(noise_folder.glob("*.jpg"))[:IMAGE_LIMIT]
        
        for img_path in tqdm(img_paths, desc=f"   {noise_name}"):
            clean_path = clean_folder / img_path.name
            if not clean_path.exists():
                continue
            
            try:
                clean_img = Image.open(clean_path).convert("RGB")
                noisy_img = Image.open(img_path).convert("RGB")
                
                # Apply preprocessing
                noisy_img = preprocess_image(noisy_img, noise_name)
                
                noisy_tensor = resize_to_224(noisy_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if device == "cuda":
                        with torch.cuda.amp.autocast():
                            denoised_tensor = model(noisy_tensor)
                    else:
                        denoised_tensor = model(noisy_tensor)
                
                denoised_img = to_pil(denoised_tensor.squeeze(0).cpu())
                denoised_img = denoised_img.resize(noisy_img.size, Image.LANCZOS)
                denoised_img.save(out_dir / f"ViT_{img_path.name}")
                
                metrics = compute_all_metrics(clean_img, denoised_img)
                metrics.update({
                    "image": img_path.name,
                    "noise_type": noise_name,
                    "model": "ViT_Improved"
                })
                all_results.append(metrics)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_root / "vit_improved_all_metrics.csv", index=False)
        
        # ===== PRINT ALL 10 METRICS TABLE =====
        print("\n" + "="*80)
        print("📊 ALL 10 METRICS BY NOISE TYPE")
        print("="*80)
        
        summary = df.groupby('noise_type').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'roc_auc': 'mean',
            'mcc': 'mean',
            'psnr': 'mean',
            'ssim': 'mean',
            'lpips': 'mean',
            'niqe': 'mean'
        }).round(4)
        
        # Reorder columns for better presentation
        summary = summary[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc', 'psnr', 'ssim', 'lpips', 'niqe']]
        
        print(summary.to_string())
        
        # ===== OVERALL AVERAGE =====
        print("\n" + "="*80)
        print("📊 OVERALL AVERAGE (All Noise Types)")
        print("="*80)
        print(f"   {'Metric':<15} {'Value':<10} {'Interpretation':<30}")
        print(f"   {'-'*15} {'-'*10} {'-'*30}")
        print(f"   {'Accuracy':<15} {df['accuracy'].mean():<10.4f} {'Higher is better':<30}")
        print(f"   {'Precision':<15} {df['precision'].mean():<10.4f} {'Higher is better':<30}")
        print(f"   {'Recall':<15} {df['recall'].mean():<10.4f} {'Higher is better':<30}")
        print(f"   {'F1 Score':<15} {df['f1'].mean():<10.4f} {'Higher is better':<30}")
        print(f"   {'ROC-AUC':<15} {df['roc_auc'].mean():<10.4f} {'Higher is better (0.5=random)':<30}")
        print(f"   {'MCC':<15} {df['mcc'].mean():<10.4f} {'+1 perfect, 0 random, -1 opposite':<30}")
        print(f"   {'PSNR (dB)':<15} {df['psnr'].mean():<10.2f} {'Higher is better':<30}")
        print(f"   {'SSIM':<15} {df['ssim'].mean():<10.4f} {'Higher is better (max 1.0)':<30}")
        print(f"   {'LPIPS':<15} {df['lpips'].mean():<10.4f} {'Lower is better (0=identical)':<30}")
        print(f"   {'NIQE':<15} {df['niqe'].mean():<10.2f} {'Lower is better (naturalness)':<30}")
        
        # ===== BEST AND WORST =====
        print("\n" + "="*80)
        print("📊 BEST AND WORST PERFORMING NOISE TYPES")
        print("="*80)
        
        best_acc = df.groupby('noise_type')['accuracy'].mean().idxmax()
        worst_acc = df.groupby('noise_type')['accuracy'].mean().idxmin()
        best_psnr = df.groupby('noise_type')['psnr'].mean().idxmax()
        worst_psnr = df.groupby('noise_type')['psnr'].mean().idxmin()
        best_ssim = df.groupby('noise_type')['ssim'].mean().idxmax()
        worst_ssim = df.groupby('noise_type')['ssim'].mean().idxmin()
        
        print(f"   Best Accuracy:  {best_acc} ({df[df['noise_type']==best_acc]['accuracy'].mean():.4f})")
        print(f"   Worst Accuracy: {worst_acc} ({df[df['noise_type']==worst_acc]['accuracy'].mean():.4f})")
        print(f"   Best PSNR:      {best_psnr} ({df[df['noise_type']==best_psnr]['psnr'].mean():.2f} dB)")
        print(f"   Worst PSNR:     {worst_psnr} ({df[df['noise_type']==worst_psnr]['psnr'].mean():.2f} dB)")
        print(f"   Best SSIM:      {best_ssim} ({df[df['noise_type']==best_ssim]['ssim'].mean():.4f})")
        print(f"   Worst SSIM:     {worst_ssim} ({df[df['noise_type']==worst_ssim]['ssim'].mean():.4f})")
        
        print(f"\n✅ Results saved to: {output_root}")
        print(f"   📄 vit_improved_all_metrics.csv")
    else:
        print("\n⚠️ No results generated.")

if __name__ == "__main__":
    main()