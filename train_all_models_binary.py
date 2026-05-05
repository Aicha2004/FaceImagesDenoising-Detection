import os
import copy
import torch
import timm
import pandas as pd
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================
# PATHS
# ==========================
BASE_DIR = r"C:\Users\dell\Desktop\project"

DATA_DIR = os.path.join(BASE_DIR, "split_data_binary")
MODEL_DIR = os.path.join(BASE_DIR, "models_binary")
RESULTS_DIR = os.path.join(BASE_DIR, "results_binary")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================
# SETTINGS
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ==========================
# MODELS
# ==========================
MODELS = {
    # ViT / Transformer models
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "deit_tiny": "deit_tiny_patch16_224",

    # CNN deep models
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
   
}

# ==========================
# DATA TRANSFORMS
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ==========================
# LOAD DATA
# ==========================
train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "val")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train folder not found: {train_path}")

if not os.path.exists(val_path):
    raise FileNotFoundError(f"Val folder not found: {val_path}")

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

print("Classes:", train_dataset.classes)

if len(train_dataset.classes) != 2:
    raise ValueError(
        f"ERROR: Expected 2 classes: clean and noise. "
        f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}"
    )

if set(train_dataset.classes) != set(["clean", "noise"]):
    raise ValueError(
        f"ERROR: folders must be named clean and noise. "
        f"Found: {train_dataset.classes}"
    )

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)
print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================
# TRAIN ALL MODELS
# ==========================
history_rows = []

for short_name, timm_name in MODELS.items():

    print("\n===================================")
    print(f"Training model: {short_name}")
    print(f"TIMM name: {timm_name}")
    print("===================================")

    try:
        model = timm.create_model(
            timm_name,
            pretrained=True,
            num_classes=num_classes
        ).to(DEVICE)
    except Exception as e:
        print(f"Could not create model {short_name}. Skipping.")
        print("Reason:", e)
        continue

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):

        # --------------------------
        # TRAIN
        # --------------------------
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --------------------------
        # VALIDATION
        # --------------------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(
            f"{short_name} | Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        history_rows.append({
            "model": short_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        # --------------------------
        # SAVE BEST
        # --------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

            save_path = os.path.join(MODEL_DIR, f"best_{short_name}.pth")
            torch.save(model.state_dict(), save_path)
            print("Saved best model:", save_path)

    # Save final best again
    model.load_state_dict(best_weights)
    save_path = os.path.join(MODEL_DIR, f"best_{short_name}.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Finished {short_name}. Best Val Acc: {best_val_acc:.4f}")

# ==========================
# SAVE TRAINING HISTORY
# ==========================
history_df = pd.DataFrame(history_rows)
history_df.to_csv(
    os.path.join(RESULTS_DIR, "training_history_all_models.csv"),
    index=False
)

print("\nAll training finished.")
print("Models saved in:", MODEL_DIR)
print("History saved in:", RESULTS_DIR)