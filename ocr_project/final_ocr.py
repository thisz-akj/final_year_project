import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
DATASET_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/dataset"
MODEL_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/tamil_brahmi_ocr_final_v2.pth"
IMAGE_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/Screenshot from 2026-01-09 17-46-34.png"

IMAGE_SIZE = 224
DOT_AREA_THRESHOLD = 350
LINE_THRESHOLD = 0.6
SPACE_FACTOR = 1.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# LOAD CLASS NAMES
# =====================================================
dataset = datasets.ImageFolder(DATASET_PATH)
class_names = dataset.classes
num_classes = len(class_names)

# =====================================================
# LOAD MODEL
# =====================================================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =====================================================
# TRANSFORM (MATCH TRAINING)
# =====================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =====================================================
# IMAGE PREPROCESS
# =====================================================
def preprocess_word(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return img, binary

# =====================================================
# CHARACTER SEGMENTATION (SIMPLE & STABLE)
# =====================================================
def segment_characters(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 200:  # ignore noise
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

# =====================================================
# PAD TO SQUARE (CRITICAL FIX)
# =====================================================
def pad_to_square(img, pad_value=255):
    h, w = img.shape
    size = max(h, w)

    padded = np.ones((size, size), dtype=np.uint8) * pad_value
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return padded

# =====================================================
# OCR WORD PIPELINE
# =====================================================
def ocr_word(img_path, visualize=True):
    original, binary = preprocess_word(img_path)
    boxes = segment_characters(binary)

    predictions = []
    confidences = []

    plt.figure(figsize=(12, 3))
    plot_idx = 1

    with torch.no_grad():
        for i, (x, y, w, h) in enumerate(boxes):
            char_img = original[y:y+h, x:x+w]

            # 🔑 normalize like training
            char_img = pad_to_square(char_img)
            pil_img = Image.fromarray(char_img)

            tensor = transform(pil_img).unsqueeze(0).to(device)

            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, dim=1)

            label = class_names[idx.item()]
            confidence = conf.item()

            predictions.append(label)
            confidences.append(confidence)

            if visualize:
                plt.subplot(1, len(boxes), plot_idx)
                plt.imshow(char_img, cmap="gray")
                plt.title(f"{label}\n{confidence*100:.1f}%")
                plt.axis("off")
                plot_idx += 1

    if visualize:
        plt.suptitle("Segmented Characters & Predictions", fontsize=14)
        plt.show()

    word = "".join(predictions)
    return word, predictions, confidences

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    word, chars, confs = ocr_word(IMAGE_PATH)

    print("\n--- OCR RESULT ---")
    print("Predicted word:", word)
    print("Characters     :", chars)
    print("Confidences    :", [round(c*100, 2) for c in confs])
