import os
import torch
import pandas as pd
import numpy as np
from PIL import Image

import timm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


MODEL_LIST = [
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "resnet50"
]

NUM_CLASSES = 7
IMAGE_SIZE = 224
BATCH_SIZE = 16

MODELS_DIR = "backend/services/outputs/saved_models"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

HAM_LABELS = {
    "akiec":0,"bcc":1,"bkl":2,"df":3,"mel":4,"nv":5,"vasc":6
}

IDX_TO_LABEL = {v:k for k,v in HAM_LABELS.items()}


DDI_TO_HAM = {
    "melanoma-in-situ": "mel",
    "squamous-cell-carcinoma-in-situ": "akiec",
    "basal-cell-carcinoma": "bcc",
    "benign-keratosis": "bkl",
    "dermatofibroma": "df",
    "vascular-lesion": "vasc",
    "mycosis-fungoides": "mel"
}

# ---------------- TRANSFORM ---------------- #

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# ---------------- DATASET ---------------- #

class DDIDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform):

        df = pd.read_csv(csv_path)

        # keep only diseases we can map
        df = df[df["disease"].isin(DDI_TO_HAM.keys())]

        self.samples = []

        for _, row in df.iterrows():

            img_path = os.path.join(image_dir, row["DDI_file"])

            ham_label = DDI_TO_HAM[row["disease"]]
            label_idx = HAM_LABELS[ham_label]

            self.samples.append((img_path, label_idx))

        self.transform = transform

        print(f"Loaded {len(self.samples)} DDI samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img, label

# ---------------- LOAD ENSEMBLE ---------------- #

def load_ensemble():

    models = []

    for name in MODEL_LIST:

        model = timm.create_model(
            name,
            pretrained=False,
            num_classes=NUM_CLASSES
        )

        weight_path = os.path.join(MODELS_DIR, f"{name}.pt")
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

        model.to(DEVICE)
        model.eval()

        models.append(model)

    print("✅ Ensemble loaded")

    return models

# ---------------- ENSEMBLE PREDICT ---------------- #

@torch.no_grad()
def ensemble_predict(models, images):

    probs = []

    for model in models:
        out = model(images)
        prob = torch.softmax(out, dim=1)
        probs.append(prob)

    avg_prob = torch.mean(torch.stack(probs), dim=0)

    preds = torch.argmax(avg_prob, dim=1)

    return preds

# ---------------- EVALUATION ---------------- #

def evaluate_ddi():

    ddi_csv = "datasets/ddidiversedermatologyimages/ddi_metadata.csv"
    ddi_images = "datasets/ddidiversedermatologyimages/images"

    dataset = DDIDataset(ddi_csv, ddi_images, test_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    models = load_ensemble()

    y_true, y_pred = [], []

    for x, y in loader:

        x = x.to(DEVICE)

        preds = ensemble_predict(models, x)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(y.numpy())

    print("\n📊 DDI ENSEMBLE RESULTS\n")

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    evaluate_ddi()
