import torch
from torch import nn
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

# ---------------- CONFIG ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Folder with clean images
clean_folder = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\data\test\clean")

# Folder with pre-generated noisy images (your existing denoising output)
noisy_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\test_noise")

# Folder to save denoised results + metrics CSV
output_root = Path(r"C:\Users\DELL\Desktop\Projet_FaceDenoising\outputs\denoised_pipeline_existing_metrics")
output_root.mkdir(parents=True, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# ---------------- Tensor conversion ----------------
def pil_to_tensor(img):
    return to_tensor(img).unsqueeze(0).to(device)

def tensor_to_pil(x):
    return to_pil(x.squeeze(0).detach().cpu().clamp(0,1))

def pil_to_np(img):
    return np.array(img).astype(np.float32)

def np_to_pil(img_np):
    return Image.fromarray(np.clip(img_np,0,255).astype(np.uint8))

# ---------------- CNN Denoiser ----------------
class CNNDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,3,3,padding=1)
        )
    def forward(self,x): return self.net(x)

def load_dl_denoisers():
    return {"CNN": CNNDenoiser().to(device)}

# ---------------- Filters ----------------
def apply_filter(img_np, filter_name):
    img_np = img_np.astype(np.float32)
    if filter_name=="min": return cv2.erode(img_np.astype(np.uint8), np.ones((3,3),np.uint8))
    if filter_name=="max": return cv2.dilate(img_np.astype(np.uint8), np.ones((3,3),np.uint8))
    if filter_name=="median": return cv2.medianBlur(img_np.astype(np.uint8),3)
    if filter_name=="gaussian": return cv2.GaussianBlur(img_np.astype(np.uint8),(3,3),0)
    if filter_name=="bilateral": return cv2.bilateralFilter(img_np.astype(np.uint8),7,50,50)
    if filter_name=="edge_preserving": return cv2.edgePreservingFilter(img_np.astype(np.uint8), flags=1, sigma_s=60, sigma_r=0.4)
    if filter_name=="fog":
        white = np.full(img_np.shape, 255, dtype=np.float32)
        return np.clip(cv2.addWeighted(img_np,0.65,white,0.35,0),0,255).astype(np.uint8)
    return img_np

FILTERS = ["min","max","median","gaussian","bilateral","edge_preserving","fog","CNN"]

# ---------------- Metrics ----------------
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

def compute_metrics(clean_img, denoised_img):
    clean_np = pil_to_np(clean_img).astype(np.uint8)
    denoised_np = pil_to_np(denoised_img).astype(np.uint8)
    
    # Align shapes
    min_h = min(clean_np.shape[0], denoised_np.shape[0])
    min_w = min(clean_np.shape[1], denoised_np.shape[1])
    clean_np = clean_np[:min_h,:min_w,:]
    denoised_np = denoised_np[:min_h,:min_w,:]

    # Pixel-level metrics
    thresh = 128
    y_true = (clean_np.mean(axis=2).flatten() > thresh).astype(int)
    y_pred = (denoised_np.mean(axis=2).flatten() > thresh).astype(int)
    accuracy = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,zero_division=1)
    precision = precision_score(y_true,y_pred,zero_division=1)
    recall = recall_score(y_true,y_pred,zero_division=1)
    try: roc_auc = roc_auc_score(y_true,y_pred)
    except: roc_auc=0.5
    mcc = matthews_corrcoef(y_true,y_pred)

    # Image similarity metrics
    psnr_val = peak_signal_noise_ratio(clean_np, denoised_np)
    ssim_val = structural_similarity(clean_np, denoised_np, channel_axis=2)

    # LPIPS
    img_tensor = torch.tensor(denoised_np/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    clean_tensor = torch.tensor(clean_np/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
    lpips_val = loss_fn_lpips(img_tensor, clean_tensor).item()

    # NIQE placeholder
    niqe_val = np.nan

    return {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,
            "roc_auc":roc_auc,"mcc":mcc,"psnr":psnr_val,"ssim":ssim_val,"lpips":lpips_val,"niqe":niqe_val}

# ---------------- Processing ----------------
denoisers = load_dl_denoisers()
results = []

BATCH_SIZE = 16  # adjust based on GPU

for noise_folder in noisy_root.iterdir():
    if not noise_folder.is_dir():
        continue
    print(f"\nProcessing noise type: {noise_folder.name}")
    out_dir = output_root / noise_folder.name
    out_dir.mkdir(parents=True,exist_ok=True)

    img_paths = list(noise_folder.glob("*.jpg"))

    # CNN batch processing
    for i in tqdm(range(0,len(img_paths),BATCH_SIZE)):
        batch_paths = img_paths[i:i+BATCH_SIZE]
        batch_imgs = []
        batch_clean = []

        for img_path in batch_paths:
            try:
                clean_img = Image.open(clean_folder / img_path.name).convert("RGB")
            except FileNotFoundError:
                print(f"Clean image not found for {img_path.name}, skipping")
                continue
            noisy_img = Image.open(img_path).convert("RGB")
            batch_imgs.append(noisy_img)
            batch_clean.append(clean_img)

        if len(batch_imgs)==0: continue

        # CNN batch
        cnn_model = denoisers["CNN"]
        cnn_model.eval()
        tensors = torch.cat([pil_to_tensor(img) for img in batch_imgs],dim=0)
        with torch.no_grad():
            denoised_tensors = cnn_model(tensors)

        for j,img_path in enumerate(batch_paths):
            denoised_img = tensor_to_pil(denoised_tensors[j])
            save_path = out_dir / f"CNN_{img_path.name}"
            denoised_img.save(save_path)
            metrics = compute_metrics(batch_clean[j], denoised_img)
            metrics.update({"image":img_path.name,"filter":"CNN","noise_type":noise_folder.name})
            results.append(metrics)

        # Classical filters
        for j,img_path in enumerate(batch_paths):
            noisy_np = pil_to_np(batch_imgs[j])
            for filter_name in FILTERS:
                if filter_name=="CNN": continue
                denoised_np = apply_filter(noisy_np, filter_name)
                denoised_img = np_to_pil(denoised_np)
                save_path = out_dir / f"{filter_name}_{img_path.name}"
                denoised_img.save(save_path)
                metrics = compute_metrics(batch_clean[j],denoised_img)
                metrics.update({"image":img_path.name,"filter":filter_name,"noise_type":noise_folder.name})
                results.append(metrics)

# Save CSV
df = pd.DataFrame(results)
df.to_csv(output_root / "denoising_metrics_existing_no_NIQE.csv",index=False)
print("Denoising + full metrics pipeline complete.")