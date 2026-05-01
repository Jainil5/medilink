import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------- CONFIG ---------------- #
MODEL_LIST = [
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "resnet50"
]

NUM_CLASSES = 7
BATCH_SIZE = 32
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "datasets/HAM10000"
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

IMG_DIRS = [
    os.path.join(BASE_DIR, "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "HAM10000_images_part_2"),
]

MODELS_DIR = "backend/app/services/outputs/saved_models"
OUTPUT_CSV = "backend/app/services/outputs/model_metrics.csv"

LABEL_MAP = {
    "akiec":0,"bcc":1,"bkl":2,
    "df":3,"mel":4,"nv":5,"vasc":6
}

# ---------------- TRANSFORM ---------------- #
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# ---------------- DATASET ---------------- #
class HAMDataset(Dataset):
    def __init__(self, df):
        self.samples = []

        for _, row in df.iterrows():
            img_id = row["image_id"]
            label = LABEL_MAP[row["dx"]]

            for d in IMG_DIRS:
                path = os.path.join(d, img_id + ".jpg")
                if os.path.exists(path):
                    self.samples.append((path, label))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = transform(img)
        return img, label

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv(CSV_PATH)

# recreate SAME split
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["dx"], random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["dx"], random_state=SEED
)

test_dataset = HAMDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- EVALUATE FUNCTION ---------------- #
@torch.no_grad()
def evaluate(model):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        out = model(x)
        probs = torch.softmax(out, dim=1)

        preds = probs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.append(probs.cpu())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    all_probs = torch.cat(all_probs).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    return metrics, y_true, all_probs

# ---------------- MAIN ---------------- #
results = []
ensemble_probs = []

for model_name in MODEL_LIST:

    print(f"Evaluating {model_name}")

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f"{model_name}.pt"), map_location=DEVICE)
    )

    metrics, y_true, probs = evaluate(model)

    row = {"model": model_name}
    row.update(metrics)
    results.append(row)

    ensemble_probs.append(probs)

    # 🔥 Confusion Matrix
    print(f"\nConfusion Matrix for {model_name}:\n")
    print(confusion_matrix(y_true, np.argmax(probs, axis=1)))

# ---------------- ENSEMBLE ---------------- #
avg_probs = np.mean(ensemble_probs, axis=0)
ensemble_preds = np.argmax(avg_probs, axis=1)

ensemble_metrics = {
    "accuracy": accuracy_score(y_true, ensemble_preds),
    "precision_macro": precision_score(y_true, ensemble_preds, average="macro"),
    "recall_macro": recall_score(y_true, ensemble_preds, average="macro"),
    "f1_macro": f1_score(y_true, ensemble_preds, average="macro"),
}

results.append({"model": "ensemble", **ensemble_metrics})

# ---------------- SAVE ---------------- #
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Metrics saved to:", OUTPUT_CSV)