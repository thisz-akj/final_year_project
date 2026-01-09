import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/tamil_brahmi_ocr_final_v2.pth"
DATASET_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/dataset"
IMAGE_PATH = "/home/workstation/Documents/Azadec22b1109/ocr_project/Screenshot from 2026-01-08 11-59-41.png"
IMAGE_SIZE = 224

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# LOAD CLASS NAMES (IMPORTANT)
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
# TRANSFORM (EXACTLY SAME AS TRAINING)
# =====================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =====================================================
# LOAD IMAGE (NO EXTRA PROCESSING)
# =====================================================
pil_image = Image.open(IMAGE_PATH).convert("L")
input_tensor = transform(pil_image).unsqueeze(0).to(device)

# =====================================================
# PREDICTION
# =====================================================
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probs, dim=1)

predicted_class = class_names[predicted_idx.item()]
confidence = confidence.item()

# =====================================================
# VISUALIZE IMAGE FED TO MODEL
# =====================================================
plt.figure(figsize=(4, 4))
plt.imshow(pil_image, cmap="gray")
plt.axis("off")
plt.title(
    f"Prediction: {predicted_class}\nConfidence: {confidence*100:.2f}%",
    fontsize=12
)
plt.show()

# =====================================================
# CONSOLE OUTPUT
# =====================================================
print("\nOCR TEST RESULT")
print("----------------------")
print("Image           :", IMAGE_PATH)
print("Predicted class :", predicted_class)
print("Confidence      :", round(confidence * 100, 2), "%")
