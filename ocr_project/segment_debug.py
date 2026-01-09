import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PREPROCESS
# =====================================================
def preprocess_sentence(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Upscale to preserve fine strokes
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold (keeps faint strokes)
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # VERY light cleanup (do NOT over-clean)
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return img, clean


# =====================================================
# CONNECTED COMPONENTS (ULTRA RELAXED)
# =====================================================
def get_components(binary):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    comps = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # 🔑 Only reject microscopic specks
        if area < 20:
            continue

        comps.append((x, y, w, h, area))

    return comps


# =====================================================
# SPLIT MAINS / DOTS (VERY SAFE)
# =====================================================
def split_mains_dots(components):
    mains, dots = [], []

    for x, y, w, h, area in components:
        aspect = max(w, h) / (min(w, h) + 1e-5)

        # True dots / diacritics
        if area < 350 and aspect < 2.5:
            dots.append((x, y, w, h))
        else:
            mains.append((x, y, w, h))

    return mains, dots


# =====================================================
# MERGE BROKEN MAIN STROKES (AGGRESSIVE)
# =====================================================
def merge_close_mains(mains):
    merged = []
    used = [False] * len(mains)

    for i, (x1, y1, w1, h1) in enumerate(mains):
        if used[i]:
            continue

        X1, Y1 = x1, y1
        X2, Y2 = x1 + w1, y1 + h1

        for j, (x2, y2, w2, h2) in enumerate(mains):
            if i == j or used[j]:
                continue

            # Horizontal overlap or near-touch
            overlap = min(X2, x2 + w2) - max(X1, x2)
            overlap_ok = overlap > -min(w1, w2) * 0.3

            # Vertical closeness
            close_y = abs(
                (y2 + h2 // 2) - (y1 + h1 // 2)
            ) < max(h1, h2) * 1.8

            if overlap_ok and close_y:
                X1 = min(X1, x2)
                Y1 = min(Y1, y2)
                X2 = max(X2, x2 + w2)
                Y2 = max(Y2, y2 + h2)
                used[j] = True

        used[i] = True
        merged.append((X1, Y1, X2 - X1, Y2 - Y1))

    return merged


# =====================================================
# MERGE DOTS INTO GLYPHS (VERY FORGIVING)
# =====================================================
def merge_dots(mains, dots):
    merged = []

    for mx, my, mw, mh in mains:
        X1, Y1 = mx, my
        X2, Y2 = mx + mw, my + mh
        cx = mx + mw // 2

        for dx, dy, dw, dh in dots:
            dcx = dx + dw // 2
            dcy = dy + dh // 2

            horizontal_ok = abs(dcx - cx) < mw * 1.4
            vertical_ok = (
                dcy > my - mh * 1.0 and
                dcy < my + mh * 2.2
            )

            if horizontal_ok and vertical_ok:
                X1 = min(X1, dx)
                Y1 = min(Y1, dy)
                X2 = max(X2, dx + dw)
                Y2 = max(Y2, dy + dh)

        merged.append((X1, Y1, X2 - X1, Y2 - Y1))

    return merged


# =====================================================
# LINE GROUPING (STABLE)
# =====================================================
def group_by_lines(boxes):
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []

    for box in boxes:
        x, y, w, h = box
        cy = y + h // 2
        placed = False

        for line in lines:
            ly = line[0][1] + line[0][3] // 2
            if abs(cy - ly) < h * 0.8:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    for line in lines:
        line.sort(key=lambda b: b[0])

    return lines


# =====================================================
# VISUALIZE
# =====================================================
def visualize(img_path):
    original, binary = preprocess_sentence(img_path)

    components = get_components(binary)
    mains, dots = split_mains_dots(components)
    mains = merge_close_mains(mains)
    boxes = merge_dots(mains, dots)
    lines = group_by_lines(boxes)

    vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    for li, line in enumerate(lines):
        for (x, y, w, h) in line:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Binary Image")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Final OCR Segmentation (MAX PRESERVATION)")
    plt.imshow(vis)
    plt.axis("off")

    plt.show()


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    image_path = "/home/workstation/Documents/Azadec22b1109/ocr_project/Screenshot from 2026-01-08 11-09-11.png"
    visualize(image_path)
