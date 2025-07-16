import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ─── Configuration ─────────────────────────────────────────────
DATA_DIR = 'Data'
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE}")

# ─── Define Transformations (for evaluation, no augmentation) ───
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ─── Load Dataset ──────────────────────────────────────────────
dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
print(f"Classes found: {class_names}")

# ─── Split into Train and Validation ───────────────────────────
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_ds = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ─── Define Model and Load Weights ─────────────────────────────
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_cnn_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ─── Inference on Validation Set ───────────────────────────────
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ─── Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_confusion_matrix.png")
plt.close()
print(" Confusion matrix saved as 'cnn_confusion_matrix.png'.")

# ─── Classification Report and Metrics ─────────────────────────
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\n CNN Classification Report:\n")
print(report)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1-Score: {f1:.4f}")

# ─── Bar Chart of Evaluation Metrics ───────────────────────────
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values(), color="cornflowerblue")
plt.ylim(0, 1)
plt.title("CNN Evaluation Metrics")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig("cnn_scores_bar.png")
plt.close()
print(" Metrics bar chart saved as 'cnn_scores_bar.png'.")
