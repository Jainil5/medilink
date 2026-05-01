import os, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ================= PATH OPTIMIZATION =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ================= CONFIG =================
MODEL_LIST = [
    "tf_efficientnet_b5",
    "convnext_base",
    "densenet169",
    "vit_base_patch16_224",
    "swin_small_patch4_window7_224"
]

IMAGE_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 20
NUM_CLASSES = 7
SEED = 42

DATASET_DIR = PROJECT_ROOT / "datasets" / "HAM10000"
CSV_PATH = str(DATASET_DIR / "HAM10000_metadata.csv")

IMG_DIRS = [
    str(DATASET_DIR / "HAM10000_images_part_1"),
    str(DATASET_DIR / "HAM10000_images_part_2"),
]

LABEL_MAP = {
    "akiec":0,"bcc":1,"bkl":2,
    "df":3,"mel":4,"nv":5,"vasc":6
}

OUTPUT_DIR = PROJECT_ROOT / "app" / "services" / "outputs"
MODELS_DIR = str(OUTPUT_DIR / "saved_models")
METRICS_DIR = str(OUTPUT_DIR / "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ================= SEED =================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)

# ================= AUGMENTATION =================
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.CoarseDropout(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ================= DATASET =================
class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dirs, transform):
        self.samples = []
        self.transform = transform

        for _, row in df.iterrows():
            img_id = row["image_id"]
            label = LABEL_MAP[row["dx"]]

            for d in img_dirs:
                path = os.path.join(d, img_id + ".jpg")
                if os.path.exists(path):
                    self.samples.append((path, label))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))

        augmented = self.transform(image=img)
        img = augmented["image"]

        return img, label

# ================= EVALUATION =================
@torch.no_grad()
def evaluate(model, loader, return_probs=False):
    model.eval()

    all_probs, all_preds, all_trues = [], [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(1)

        all_probs.append(probs.cpu())
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(y.cpu().numpy())

    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    if return_probs:
        all_probs = torch.cat(all_probs, dim=0).numpy()
        return metrics, all_probs, y_true

    return metrics

# ================= LR TUNING =================
def get_lr(name):
    if "efficientnet" in name:
        return 1e-4
    elif "vit" in name or "swin" in name:
        return 3e-5
    else:
        return 2e-4

# ================= TRAIN =================
def train_one_model(name, train_loader, val_loader):
    model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=get_lr(name), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader)

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        val_metrics = evaluate(model, val_loader)
        val_acc = val_metrics["accuracy"]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pt"))

# ================= MAIN =================
def main():
    df = pd.read_csv(CSV_PATH)

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["dx"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=SEED)

    train_loader = DataLoader(HAM10000Dataset(train_df, IMG_DIRS, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(HAM10000Dataset(val_df, IMG_DIRS, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(HAM10000Dataset(test_df, IMG_DIRS, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ===== TRAIN =====
    for model_name in MODEL_LIST:
        print(f"\nTraining {model_name}")
        train_one_model(model_name, train_loader, val_loader)

    # ===== EVALUATE =====
    results = []
    ensemble_probs = []

    weights = [0.3, 0.2, 0.2, 0.15, 0.15]

    for model_name in MODEL_LIST:
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{model_name}.pt")))
        
        metrics, probs, y_true = evaluate(model, test_loader, return_probs=True)

        # Save individual metrics
        pd.DataFrame([metrics]).to_csv(
            os.path.join(METRICS_DIR, f"{model_name}_metrics.csv"),
            index=False
        )

        row = {"model": model_name}
        row.update(metrics)
        results.append(row)

        ensemble_probs.append(probs)

    # ===== ENSEMBLE =====
    avg_probs = np.zeros_like(ensemble_probs[0])
    for w, p in zip(weights, ensemble_probs):
        avg_probs += w * p

    ensemble_preds = np.argmax(avg_probs, axis=1)

    ensemble_metrics = {
        "accuracy": accuracy_score(y_true, ensemble_preds),
        "precision_macro": precision_score(y_true, ensemble_preds, average="macro"),
        "recall_macro": recall_score(y_true, ensemble_preds, average="macro"),
        "f1_macro": f1_score(y_true, ensemble_preds, average="macro"),
        "precision_weighted": precision_score(y_true, ensemble_preds, average="weighted"),
        "recall_weighted": recall_score(y_true, ensemble_preds, average="weighted"),
        "f1_weighted": f1_score(y_true, ensemble_preds, average="weighted"),
    }

    pd.DataFrame([ensemble_metrics]).to_csv(
        os.path.join(METRICS_DIR, "ensemble_metrics.csv"),
        index=False
    )

    ensemble_row = {"model": "ensemble"}
    ensemble_row.update(ensemble_metrics)
    results.append(ensemble_row)

    # ===== FINAL REPORT =====
    df_report = pd.DataFrame(results)
    df_report.to_csv(os.path.join(METRICS_DIR, "final_report.csv"), index=False)

if __name__ == "__main__":
    main()