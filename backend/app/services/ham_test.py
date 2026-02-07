import os
import numpy as np
import torch
import timm
from PIL import Image
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


MODEL_LIST = [
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "resnet50"
]

IMAGE_SIZE = 224
NUM_CLASSES = 7

MODELS_DIR = "services/outputs/saved_models"


LABEL_MAP = {
    "akiec":0,
    "bcc":1,
    "bkl":2,
    "df":3,
    "mel":4,
    "nv":5,
    "vasc":6
}

INV_LABEL_MAP = {v:k for k,v in LABEL_MAP.items()}


FULL_NAME_MAP = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesions"
}


DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        (0.485,0.456,0.406),
        (0.229,0.224,0.225)
    )
])


def load_model(name):
    model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_DIR, f"{name}.pt"),
            map_location=DEVICE
        )
    )
    model.to(DEVICE).eval()
    return model


models = {name: load_model(name) for name in MODEL_LIST}
resnet_model = models["resnet50"]


def load_image(path):

    if not os.path.exists(path):
        print("Image not found:", path)
        exit()

    img = Image.open(path).convert("RGB")

    plain_img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

    vis_img = plain_img.astype(np.float32) / 255.0

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    return tensor, plain_img, vis_img


@torch.no_grad()
def ensemble_predict(img_tensor):

    probs_all = []

    for model in models.values():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        probs_all.append(probs.cpu().numpy())

    probs_all = np.stack(probs_all, axis=0)

    ensemble_probs = np.mean(probs_all, axis=0)[0]

    pred_class = np.argmax(ensemble_probs)
    confidence = ensemble_probs[pred_class]

    return pred_class, confidence, ensemble_probs


def resnet_xai(img_tensor, plain_img, vis_img):

    target_layer = resnet_model.layer4[-1]

    cam = GradCAM(
        model=resnet_model,
        target_layers=[target_layer],
    )

    grayscale_cam = cam(img_tensor)[0]

    heatmap_overlay = show_cam_on_image(
        vis_img,
        grayscale_cam,
        use_rgb=True
    )

    cam_norm = (grayscale_cam - grayscale_cam.min()) / (
        grayscale_cam.max() - grayscale_cam.min() + 1e-8
    )

    threshold = 0.5
    mask = (cam_norm > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bordered_img = plain_img.copy()

    cv2.drawContours(
        bordered_img,
        contours,
        -1,
        (255, 0, 0),
        2
    )

    return heatmap_overlay, bordered_img


def get_full_name(abbr):
    return FULL_NAME_MAP.get(abbr, abbr)


def save_matplotlib_merge(original, bordered, heatmap, pred_name, confidence, save_path):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(bordered)
    axes[1].set_title("Boundary")
    axes[1].axis("off")

    axes[2].imshow(heatmap)
    axes[2].set_title("Heatmap")
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction by AI Model: {pred_name}   Confidence: {confidence:.4f}",
        fontsize=12,
        y=0.95
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.9])

    disclaimer_text = (
        "This AI model is intended to assist dermatologists and should not be used as a replacement "
        "for professional medical diagnosis or clinical judgment."
    )

    fig.text(
        0.5,
        0.04,
        disclaimer_text,
        ha="center",
        fontsize=10,
        wrap=True
    )

    plt.savefig(save_path, dpi=200)
    plt.close()



def run_test(image_path):

    img_tensor, plain_img, vis_img = load_image(image_path)

    pred, conf, probs = ensemble_predict(img_tensor)

    abbr_label = INV_LABEL_MAP[pred]
    full_label = get_full_name(abbr_label)

    # print("\nENSEMBLE RESULT")
    # print("Predicted disease:", full_label)
    # print("Confidence:", round(conf,4))

    # print("\nClass probabilities:")
    # for i, p in enumerate(probs):
    #     abbr = INV_LABEL_MAP[i]
    #     full = get_full_name(abbr)
    #     print(f"{full} : {round(p,4)}")

    heatmap, bordered = resnet_xai(
        img_tensor,
        plain_img,
        vis_img
    )

    save_matplotlib_merge(
        plain_img,
        bordered,
        heatmap,
        full_label,
        conf,
        "test/merged_result.png"
    )
    print(conf)

    return {
        "prediction": full_label,
        "confidence": float(conf),
        "xai_image": "/Users/jainil/Documents/development/medilink/backend/app/test/merged_result.png"
    }

# print(run_test(
#     "datasets/HAM10000/HAM10000_images_part_1/ISIC_0024350.jpg"
# ))
