import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
DATASET_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/dataset"
MODEL_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/tamil_brahmi_ocr_final_v2.pth"
IMAGE_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/Screenshot from 2026-01-08 11-20-05.png"

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
# TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =====================================================
# PREPROCESS
# =====================================================
def preprocess_sentence(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return img, clean

# =====================================================
# COMPONENT FILTER
# =====================================================
def is_valid_component(x, y, w, h, area, binary):
    if w < 8 or h < 12:
        return False
    if area < 80:
        return False

    aspect = max(w, h) / (min(w, h) + 1e-5)
    if aspect > 8.0:
        return False

    roi = binary[y:y+h, x:x+w]
    ink = cv2.countNonZero(roi)
    density = ink / (w * h)
    if density < 0.05:
        return False

    return True

# =====================================================
# SEGMENTATION PIPELINE
# =====================================================
def get_components(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if is_valid_component(x, y, w, h, area, binary):
            comps.append((x, y, w, h, area))
    return comps


def split_mains_dots(components):
    mains, dots = [], []
    for x, y, w, h, area in components:
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if area < DOT_AREA_THRESHOLD and aspect < 1.8:
            dots.append((x, y, w, h))
        else:
            mains.append((x, y, w, h))
    return mains, dots


def merge_close_mains(mains):
    merged, used = [], [False] * len(mains)
    for i, (x1, y1, w1, h1) in enumerate(mains):
        if used[i]:
            continue
        X1, Y1, X2, Y2 = x1, y1, x1+w1, y1+h1
        for j, (x2, y2, w2, h2) in enumerate(mains):
            if i == j or used[j]:
                continue
            overlap_x = min(X2, x2+w2) - max(X1, x2)
            close_y = abs((y2+h2//2)-(y1+h1//2)) < max(h1, h2)*1.2
            if overlap_x > min(w1, w2)*0.3 and close_y:
                X1, Y1 = min(X1, x2), min(Y1, y2)
                X2, Y2 = max(X2, x2+w2), max(Y2, y2+h2)
                used[j] = True
        used[i] = True
        merged.append((X1, Y1, X2-X1, Y2-Y1))
    return merged


def merge_dots(mains, dots):
    merged = []
    for mx, my, mw, mh in mains:
        X1, Y1, X2, Y2 = mx, my, mx+mw, my+mh
        cx = mx + mw//2
        for dx, dy, dw, dh in dots:
            dcx = dx + dw//2
            dcy = dy + dh//2
            if abs(dcx-cx) < mw and (my-mh*0.6 < dcy < my+mh*1.6):
                X1, Y1 = min(X1, dx), min(Y1, dy)
                X2, Y2 = max(X2, dx+dw), max(Y2, dy+dh)
        merged.append((X1, Y1, X2-X1, Y2-Y1))
    return merged


def segment_characters(binary):
    comps = get_components(binary)
    mains, dots = split_mains_dots(comps)
    mains = merge_close_mains(mains)
    return merge_dots(mains, dots)

# =====================================================
# LINE GROUPING + SPACES
# =====================================================
def group_lines(boxes):
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []
    for box in boxes:
        x, y, w, h = box
        cy = y + h//2
        placed = False
        for line in lines:
            ly = line[0][1] + line[0][3]//2
            if abs(cy-ly) < h*LINE_THRESHOLD:
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])
    for line in lines:
        line.sort(key=lambda b: b[0])
    return lines


def insert_spaces(line_boxes, chars):
    if len(chars) == 1:
        return chars[0]
    gaps = []
    for i in range(1, len(line_boxes)):
        px, _, pw, _ = line_boxes[i-1]
        cx, _, _, _ = line_boxes[i]
        gaps.append(cx-(px+pw))
    thresh = np.median(gaps)*SPACE_FACTOR
    out = chars[0]
    for i in range(1, len(chars)):
        if gaps[i-1] > thresh:
            out += " "
        out += chars[i]
    return out

# =====================================================
# VISUAL OCR (NEW ADDITION)
# =====================================================
def visualize_predictions(img_path):
    original, binary = preprocess_sentence(img_path)
    boxes = segment_characters(binary)
    lines = group_lines(boxes)

    vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    final_lines = []
    char_idx = 0

    with torch.no_grad():
        for line in lines:
            preds = []
            for x, y, w, h in line:
                char_img = original[y:y+h, x:x+w]
                pil = Image.fromarray(char_img)
                tensor = transform(pil).unsqueeze(0).to(device)

                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                label = class_names[idx]
                conf = probs[0, idx].item()

                preds.append(label)

                cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(vis, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                plt.figure(figsize=(2,2))
                plt.title(f"{label} ({conf:.2f})")
                plt.imshow(char_img, cmap="gray")
                plt.axis("off")
                plt.show()

                char_idx += 1

            final_lines.append(insert_spaces(line, preds))

    plt.figure(figsize=(14,6))
    plt.title("OCR Predictions")
    plt.imshow(vis)
    plt.axis("off")
    plt.show()

    print("\nFINAL OCR OUTPUT:\n")
    print("\n".join(final_lines))


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    visualize_predictions(IMAGE_PATH)
