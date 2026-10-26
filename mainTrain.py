import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from model import BrainTumorCNN, INPUT_SIZE

image_directory = 'datasets/'
MODEL_PATH = 'BrainTumor10Epochs.pt'
EPOCHS = 10
BATCH_SIZE = 16

torch.manual_seed(0)

dataset = []
labels = []

# Load both classes
for folder, label in [('no', 0), ('yes', 1)]:
    folder_path = os.path.join(image_directory, folder)

    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print("Could not load:", image_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

        dataset.append(image)
        labels.append(label)

# Convert to NumPy arrays
dataset = np.array(dataset, dtype=np.float32) / 255.0
labels = np.array(labels)

# Split into training, validation, and testing
x_train, x_test, y_train, y_test = train_test_split(
    dataset, labels,
    test_size=0.2,
    random_state=0,
    stratify=labels
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=0,
    stratify=y_train
)

print("Training images:", x_train.shape)
print("Validation images:", x_val.shape)
print("Testing images:", x_test.shape)


def make_loader(x, y, shuffle):
    # PyTorch expects (N, channels, height, width)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    y = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle)


train_loader = make_loader(x_train, y_train, shuffle=True)
val_loader = make_loader(x_val, y_val, shuffle=False)
test_loader = make_loader(x_test, y_test, shuffle=False)

# Build CNN model
model = BrainTumorCNN()
print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Integer labels work directly with CrossEntropyLoss (no one-hot needed)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            total_loss += loss_fn(outputs, targets).item() * len(targets)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += len(targets)

    return total_loss / total, correct / total


# Train
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += len(targets)

    val_loss, val_accuracy = evaluate(val_loader)
    print(f"Epoch {epoch}/{EPOCHS} - "
          f"loss: {total_loss / total:.4f} - accuracy: {correct / total:.4f} - "
          f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

# Test
test_loss, test_accuracy = evaluate(test_loader)

print("Test accuracy:", test_accuracy)

# Save
torch.save(model.state_dict(), MODEL_PATH)
