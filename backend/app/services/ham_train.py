import os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm


MODEL_LIST = [
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "resnet50"
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 2e-4
NUM_CLASSES = 7
SEED = 42

BASE_DIR = "datasets/HAM10000"
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

IMG_DIRS = [
    os.path.join(BASE_DIR, "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "HAM10000_images_part_2"),
]

LABEL_MAP = {
    "akiec":0,"bcc":1,"bkl":2,
    "df":3,"mel":4,"nv":5,"vasc":6
}

OUTPUT_DIR = "backend/services/outputs"
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
REPORT_PATH = os.path.join(OUTPUT_DIR, "model_evaluation_report.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(SEED)


train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

val_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])


class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dirs, transform):
        self.transform = transform
        self.samples = []

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
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


@torch.no_grad()
def evaluate(model, loader, return_probs=False):

    model.eval()

    all_probs = []
    all_preds = []
    all_trues = []

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


def train_one_model(model_name, train_loader, val_loader):

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_acc = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()
        running_loss = 0

        loop = tqdm(train_loader)

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader)
        val_acc = val_metrics["accuracy"]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(MODELS_DIR, f"{model_name}.pt")
            )


def main():

    df = pd.read_csv(CSV_PATH)

    train_df, temp_df = train_test_split(
        df, test_size=0.3,
        stratify=df["dx"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df["dx"],
        random_state=SEED
    )

    train_ds = HAM10000Dataset(train_df, IMG_DIRS, train_transform)
    val_ds   = HAM10000Dataset(val_df, IMG_DIRS, val_transform)
    test_ds  = HAM10000Dataset(test_df, IMG_DIRS, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    for model_name in MODEL_LIST:
        train_one_model(model_name, train_loader, val_loader)

    results = []
    ensemble_probs = []

    for model_name in MODEL_LIST:

        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        model.load_state_dict(
            torch.load(
                os.path.join(MODELS_DIR, f"{model_name}.pt"),
                map_location=DEVICE
            )
        )

        metrics, probs, y_true = evaluate(
            model,
            test_loader,
            return_probs=True
        )

        row = {"model": model_name}
        row.update(metrics)
        results.append(row)

        ensemble_probs.append(probs)

    avg_probs = np.mean(ensemble_probs, axis=0)
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

    ensemble_row = {"model": "ensemble"}
    ensemble_row.update(ensemble_metrics)
    results.append(ensemble_row)

    df_report = pd.DataFrame(results)
    df_report.to_csv(REPORT_PATH, index=False)

    # print(df_report)


if __name__ == "__main__":
    main()
