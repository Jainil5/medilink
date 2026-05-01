import os
import sys
import numpy as np
import torch
import timm
from PIL import Image
import torchvision.transforms as T
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= PATH =================
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

XAI_MODEL_NAME = "resnet50"
IMAGE_SIZE = 224
NUM_CLASSES = 7

OUTPUT_DIR = PROJECT_ROOT / "app" / "services" / "outputs"
MODELS_DIR = str(OUTPUT_DIR / "saved_models")

os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_MAP = {"akiec":0, "bcc":1, "bkl":2, "df":3, "mel":4, "nv":5, "vasc":6}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

FULL_NAME_MAP = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions"
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ================= MODEL LOADER =================
def load_model(name):
    model_path = os.path.join(MODELS_DIR, f"{name}.pt")

    # ===== CASE 1: LOAD TRAINED MODEL =====
    if os.path.exists(model_path):
        print(f"✅ Loading trained weights for {name}")
        model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    # ===== CASE 2: AUTO DOWNLOAD =====
    else:
        print(f"⬇️ {name} not found. Downloading pretrained weights...")

        try:
            # Special case: ResNet50 (XAI only)
            if name == XAI_MODEL_NAME:
                model = timm.create_model(name, pretrained=True)
                print("⚠️ Using ImageNet pretrained ResNet50 ONLY for XAI (not trained on skin dataset)")

            else:
                model = timm.create_model(name, pretrained=True)

                # Replace classifier for 7 classes
                if hasattr(model, "reset_classifier"):
                    model.reset_classifier(NUM_CLASSES)

            # Save locally
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved {name} to {model_path}")

        except Exception as e:
            print(f"❌ Failed loading {name}: {e}")
            return None

    model.to(DEVICE)
    model.eval()
    return model

# ================= IMAGE =================
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")

    plain_img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    vis_img = plain_img.astype(np.float32) / 255.0

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    return tensor, plain_img, vis_img

# ================= ENSEMBLE =================
@torch.no_grad()
def ensemble_predict(img_tensor, models):
    if len(models) == 0:
        raise ValueError("❌ No models loaded")

    probs_all = []

    for model in models.values():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        probs_all.append(probs.cpu().numpy())

    ensemble_probs = np.mean(np.stack(probs_all, axis=0), axis=0)[0]

    pred_class = int(np.argmax(ensemble_probs))
    confidence = float(ensemble_probs[pred_class])

    return pred_class, confidence, ensemble_probs

# ================= XAI =================
def run_resnet_xai(img_tensor, plain_img, vis_img, model):
    target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(img_tensor)[0]

    heatmap_overlay = show_cam_on_image(vis_img, grayscale_cam, use_rgb=True)

    cam_norm = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    mask = (cam_norm > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bordered_img = plain_img.copy()
    cv2.drawContours(bordered_img, contours, -1, (255, 0, 0), 2)

    return heatmap_overlay, bordered_img

# ================= SAVE =================
def save_matplotlib_merge(original, bordered, heatmap, pred_name, confidence, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(bordered)
    axes[1].set_title("Boundary")
    axes[1].axis("off")

    axes[2].imshow(heatmap)
    axes[2].set_title("GradCAM (ResNet50 - ImageNet)")
    axes[2].axis("off")

    fig.suptitle(f"Prediction: {pred_name} | Confidence: {confidence:.4f}", fontsize=12)

    disclaimer = "Prediction from ensemble. XAI from pretrained ResNet50 (ImageNet). Consult dermatologist."
    fig.text(0.5, 0.02, disclaimer, ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ================= MAIN =================
def run_test(image_path):

    # Load ensemble models
    models = {}
    for name in MODEL_LIST:
        model = load_model(name)
        if model is not None:
            models[name] = model

    # Load ResNet50 for XAI
    resnet_xai_model = load_model(XAI_MODEL_NAME)

    if resnet_xai_model is None:
        raise ValueError("❌ ResNet50 failed to load")

    # Load image
    img_tensor, plain_img, vis_img = load_image(image_path)

    # Predict
    pred, conf, _ = ensemble_predict(img_tensor, models)

    label_key = INV_LABEL_MAP[pred]
    full_label = FULL_NAME_MAP.get(label_key, label_key)

    # XAI
    heatmap, bordered = run_resnet_xai(
        img_tensor, plain_img, vis_img, resnet_xai_model
    )

    # Save
    test_image_name = os.path.basename(image_path)
    output_path = PROJECT_ROOT / "app" / "services" / "test" / "output" / f"test_{test_image_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_matplotlib_merge(
        plain_img, bordered, heatmap, full_label, conf, str(output_path)
    )

    return {
        "prediction": full_label,
        "confidence": conf,
        "xai_image": str(output_path)
    }

# Example
# print(run_test("demo/indian_images/000012.png"))