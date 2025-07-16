import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import copy

# ─── Configuration ─────────────────────────────────────────────
DATA_DIR = 'Data'
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {DEVICE}")

# ─── Stronger Data Augmentation ────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
])

# ─── Load Dataset ──────────────────────────────────────────────
dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
print(f" Classes found: {class_names}")

# ─── Split into Train and Validation ───────────────────────────
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f" Training samples: {len(train_ds)}")
print(f" Validation samples: {len(val_ds)}")

# ─── Define Model ──────────────────────────────────────────────
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# Fine-tune ALL layers
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ─── Loss and Optimizer ────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ─── Early Stopping Setup ──────────────────────────────────────
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
patience = 10
trigger_times = 0

# ─── Training Loop ─────────────────────────────────────────────
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Check for improvement
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("️ Early stopping triggered!")
            break

    scheduler.step()

print(" Training Complete.")

# ─── Load Best Weights ─────────────────────────────────────────
model.load_state_dict(best_model_wts)

# ─── Save Model ────────────────────────────────────────────────
torch.save(model.state_dict(), 'best_cnn_model.pth')
print(" Best model saved as 'best_cnn_model.pth'.")

# ─── Final Evaluation on Validation Set ────────────────────────
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f" Validation Accuracy: {accuracy:.2f}%")
