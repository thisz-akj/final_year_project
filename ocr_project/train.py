import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm   # 🔑 progress bar

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 96
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 5         # Early stopping patience
SEED = 42

BEST_MODEL_PATH = "tamil_brahmi_ocr_best_v2.pth"
FINAL_MODEL_PATH = "tamil_brahmi_ocr_final_v2.pth"

# =====================================================
# DEVICE (GPU / CPU)
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)

# =====================================================
# TRANSFORMS (ENSURES CORRECT IMAGE SIZE)
# =====================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =====================================================
# LOAD DATASET
# =====================================================
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
num_classes = len(full_dataset.classes)

print(f"Total images: {len(full_dataset)}")
print(f"Number of classes: {num_classes}")

# Train / Validation Split (80 / 20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =====================================================
# MODEL (MobileNetV2)
# =====================================================
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Replace classifier for Tamil Brahmi characters
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Freeze backbone (initial training)
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

# =====================================================
# LOSS & OPTIMIZER
# =====================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# TRAINING WITH EARLY STOPPING
# =====================================================
best_val_acc = 0.0
epochs_without_improve = 0

print("\n🚀 Starting Training...\n")

for epoch in range(EPOCHS):
    start_time = time.time()

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]",
        leave=False
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]",
            leave=False
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    epoch_time = time.time() - start_time

    # ---------------- LOGGING ----------------
    print(
        f"Epoch [{epoch+1:03d}/{EPOCHS}] | "
        f"Time: {epoch_time:.1f}s | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # ---------------- EARLY STOPPING ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("   Best model saved")
    else:
        epochs_without_improve += 1
        print(f"   ⚠️ No improvement ({epochs_without_improve}/{PATIENCE})")

    if epochs_without_improve >= PATIENCE:
        print("\nEarly stopping triggered")
        break

# =====================================================
# SAVE FINAL MODEL
# =====================================================
torch.save(model.state_dict(), FINAL_MODEL_PATH)

print("\nTraining completed")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"Best model saved as: {BEST_MODEL_PATH}")
print(f"Final model saved as: {FINAL_MODEL_PATH}")
