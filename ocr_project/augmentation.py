import cv2
import numpy as np
import os
import random
import sys

# =====================================================
# CONFIGURATION
# =====================================================
IMAGES_PER_CLASS = 250  # increase later (e.g. 300)
IMAGE_SIZE = 224

# =====================================================
# IMAGE LOADING
# =====================================================
def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return img


def resize_image(img):
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))


# =====================================================
# STROKE STYLE VARIATION (SAFE)
# =====================================================
def stroke_thickness_variation(img):
    kernel = np.ones((2, 2), np.uint8)
    choice = random.choice(["thin", "normal", "thick"])

    if choice == "thin":
        return cv2.erode(img, kernel, iterations=1)
    elif choice == "thick":
        return cv2.dilate(img, kernel, iterations=1)

    return img


# =====================================================
# ORGANIC CURVE WARP (FINAL VERSION)
# =====================================================
def organic_curve_warp(img):
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    amp_x = random.uniform(0.8, 1.8)
    amp_y = random.uniform(0.8, 1.8)

    freq_x = random.uniform(0.008, 0.015)
    freq_y = random.uniform(0.008, 0.015)

    phase_x = random.uniform(0, 2 * np.pi)
    phase_y = random.uniform(0, 2 * np.pi)

    dir_x = random.choice([-1, 1])
    dir_y = random.choice([-1, 1])

    dx = dir_x * amp_x * np.sin(2 * np.pi * y * freq_y + phase_y)
    dy = dir_y * amp_y * np.sin(2 * np.pi * x * freq_x + phase_x)

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# =====================================================
# GEOMETRIC VARIATION (SAFE)
# =====================================================
def small_rotation_and_shift(img):
    h, w = img.shape

    angle = random.uniform(-10, 10)
    tx = random.uniform(-10, 10)
    ty = random.uniform(-10, 10)

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    M[:, 2] += (tx, ty)

    return cv2.warpAffine(img, M, (w, h), borderValue=255)


def squeeze_transform(img):
    h, w = img.shape

    # Non-uniform scaling factors
    scale_x = random.uniform(0.85, 1.5)
    scale_y = random.uniform(0.85, 1.5)


    # Centered affine transform
    M = np.array([
        [scale_x, 0, (1 - scale_x) * w / 2],
        [0, scale_y, (1 - scale_y) * h / 2]
    ], dtype=np.float32)

    squeezed = cv2.warpAffine(
    img,
    M,
    (w, h),
    cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REPLICATE
)


    return squeezed

# =====================================================
# NOISE FUNCTIONS (NEW ADDITION ⭐)
# =====================================================
def salt_pepper_spots(img, amount=0.002):
    noisy = img.copy()
    h, w = img.shape
    num_spots = int(amount * h * w)

    for _ in range(num_spots // 2):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy[y, x] = 255

    for _ in range(num_spots // 2):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        noisy[y, x] = 0

    return noisy


def stone_grain_noise(img):
    grain = np.random.normal(0, 6, img.shape)
    noisy = img + grain
    return np.clip(noisy, 0, 255).astype(np.uint8)


# =====================================================
# PHOTOMETRIC VARIATION (LIGHT)
# =====================================================
def adjust_contrast_brightness(img):
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# =====================================================
# FULL AUGMENTATION PIPELINE
# =====================================================
def augment_image(base_img):
    img = base_img.copy()

    # -------------------------
    # Style & geometry
    # -------------------------
    if random.random() < 0.6:
        img = squeeze_transform(img)

    if random.random() < 0.8:
        img = stroke_thickness_variation(img)

    if random.random() < 0.85:
        img = organic_curve_warp(img)

    if random.random() < 0.7:
        img = small_rotation_and_shift(img)

    # -------------------------
    # Noise injection (multi-level)
    # -------------------------
    noise_chance = random.random()

    if noise_chance < 0.3:
        pass  # 50% images remain clean

    else:
        noise_level = random.choices(
            ["low", "medium", "high"],
            weights=[0.25, 0.35, 0.4]
        )[0]

        if noise_level == "low":
            img = salt_pepper_spots(img, amount=random.uniform(0.0005, 0.001))
            img = stone_grain_noise(img)

        elif noise_level == "medium":
            img = salt_pepper_spots(img, amount=random.uniform(0.0015, 0.003))
            img = stone_grain_noise(img)
            img = adjust_contrast_brightness(img)

        elif noise_level == "high":
            img = salt_pepper_spots(img, amount=random.uniform(0.005, 0.01))
            img = stone_grain_noise(img)
            img = adjust_contrast_brightness(img)

    # -------------------------
    # Final resize
    # -------------------------
    return resize_image(img)

# =====================================================
# DATASET AUGMENTATION
# =====================================================
def augment_dataset(dataset_dir):
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        files = sorted(os.listdir(class_path))
        if len(files) == 0:
            continue

        base_image_path = os.path.join(class_path, files[0])
        base_img = load_image(base_image_path)

        print(f"[INFO] Augmenting class: {class_name}")

        for i in range(IMAGES_PER_CLASS):
            aug_img = augment_image(base_img)
            save_path = os.path.join(class_path, f"{class_name}_n3_{i:03}.png")
            cv2.imwrite(save_path, aug_img)

        print(f"[DONE] {class_name}: {IMAGES_PER_CLASS} images created")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python augment_dataset.py <DATASET_PATH>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    print(f"[START] Using dataset: {dataset_path}")
    augment_dataset(dataset_path)
    print("[FINISHED] Dataset augmentation complete")
