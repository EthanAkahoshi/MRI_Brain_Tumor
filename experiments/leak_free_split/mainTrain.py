# Experiment (unfinished): leak-free split that groups near-duplicate images so
# copies of the same scan never land in different splits, plus a 4-block CNN
# with batch norm (see model.py in this folder). Written for the original
# datasets/no and datasets/yes layout. Run from the project root with:
#   set PYTHONPATH=experiments\leak_free_split && python experiments\leak_free_split\mainTrain.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sklearn.model_selection import StratifiedGroupKFold

from model import BrainTumorCNN, INPUT_SIZE

image_directory = 'datasets/'
MODEL_PATH = 'BrainTumorCNN.pt'
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 16

# Images whose 64x64 grayscale versions differ by less than this (mean absolute
# pixel difference, also checked mirrored) count as copies of the same scan
DUPLICATE_THRESHOLD = 0.02

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


def group_near_duplicates(images):
    # The dataset contains augmented copies of the same scans. Group them so a
    # copy never ends up in a different split from its original.
    small = np.array([
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (64, 64))
        for image in images
    ])
    flat = torch.from_numpy(small.reshape(len(small), -1))
    mirrored = torch.from_numpy(small[:, :, ::-1].copy().reshape(len(small), -1))

    parent = list(range(len(images)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(flat), 256):
        chunk = flat[start:start + 256]
        distance = torch.minimum(
            torch.cdist(chunk, flat, p=1),
            torch.cdist(chunk, mirrored, p=1),
        ) / flat.shape[1]

        for i, j in (distance < DUPLICATE_THRESHOLD).nonzero().tolist():
            parent[find(start + i)] = find(j)

    return np.array([find(i) for i in range(len(images))])


groups = group_near_duplicates(dataset)
print(f"{len(dataset)} images in {len(np.unique(groups))} groups of near-duplicates")


def group_split(indices, n_splits):
    # Hold out ~1/n_splits of the groups, keeping class balance
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    keep, held_out = next(splitter.split(indices, labels[indices], groups[indices]))
    return indices[keep], indices[held_out]


# Split into training, validation, and testing (no group crosses splits)
train_idx, test_idx = group_split(np.arange(len(dataset)), n_splits=5)
train_idx, val_idx = group_split(train_idx, n_splits=5)

print("Training images:", len(train_idx))
print("Validation images:", len(val_idx))
print("Testing images:", len(test_idx))

# Random variations of the training images, regenerated every epoch
train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])


class ImageDataset(Dataset):
    def __init__(self, indices, transform=None):
        # PyTorch expects (N, channels, height, width)
        self.images = torch.from_numpy(dataset[indices]).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels[indices]).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[i]


train_loader = DataLoader(ImageDataset(train_idx, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ImageDataset(val_idx), batch_size=64)
test_loader = DataLoader(ImageDataset(test_idx), batch_size=64)

# Build CNN model
model = BrainTumorCNN()
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Integer labels work directly with CrossEntropyLoss (no one-hot needed)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


def evaluate(loader):
    model.eval()
    total_loss, predictions, targets_all = 0.0, [], []

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            total_loss += loss_fn(outputs, targets).item() * len(targets)
            predictions.append(outputs.argmax(dim=1))
            targets_all.append(targets)

    predictions = torch.cat(predictions)
    targets_all = torch.cat(targets_all)
    accuracy = (predictions == targets_all).float().mean().item()
    return total_loss / len(targets_all), accuracy, predictions, targets_all


# Train, keeping the weights from the epoch with the best validation accuracy
best_val_accuracy, best_epoch = 0.0, 0

for epoch in range(1, MAX_EPOCHS + 1):
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

    val_loss, val_accuracy, _, _ = evaluate(val_loader)
    scheduler.step(val_accuracy)

    improved = val_accuracy > best_val_accuracy
    if improved:
        best_val_accuracy, best_epoch = val_accuracy, epoch
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch}/{MAX_EPOCHS} - "
          f"loss: {total_loss / total:.4f} - accuracy: {correct / total:.4f} - "
          f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
          f"{' *' if improved else ''}", flush=True)

    if epoch - best_epoch >= EARLY_STOP_PATIENCE:
        print("Stopping early: no improvement for", EARLY_STOP_PATIENCE, "epochs")
        break

# Test the best checkpoint
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
test_loss, test_accuracy, predictions, targets = evaluate(test_loader)

true_pos = ((predictions == 1) & (targets == 1)).sum().item()
true_neg = ((predictions == 0) & (targets == 0)).sum().item()
false_pos = ((predictions == 1) & (targets == 0)).sum().item()
false_neg = ((predictions == 0) & (targets == 1)).sum().item()

print(f"Best epoch: {best_epoch} (val_accuracy {best_val_accuracy:.4f})")
print("Test accuracy:", test_accuracy)
print(f"Sensitivity (tumours caught): {true_pos / (true_pos + false_neg):.4f}")
print(f"Specificity (healthy cleared): {true_neg / (true_neg + false_pos):.4f}")
print(f"Confusion matrix: TN={true_neg} FP={false_pos} FN={false_neg} TP={true_pos}")
