import hashlib
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from sklearn.model_selection import train_test_split

from model import BrainTumorCNN, preprocess_legacy as preprocess  # legacy trainer for BrainTumorCNN

image_directory = 'datasets/'
# Held-out benchmark: never trained on. 'notumor' is class 0, every other folder is a tumour type
benchmark_directory = os.path.join(image_directory, 'Testing')
MODEL_PATH = 'BrainTumorBest.pt'
EPOCHS = 30
EARLY_STOP_PATIENCE = 8
BATCH_SIZE = 16
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

torch.manual_seed(0)


def image_files(folder):
    # Every image in the folder and its subfolders, in a fixed order
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, name)


def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_images(paths_and_labels, skip_hashes):
    # Load, preprocess and label images, skipping exact duplicates
    images, image_labels = [], []
    seen = set(skip_hashes)
    skipped = 0

    for image_path, label in paths_and_labels:
        digest = file_hash(image_path)
        if digest in seen:
            skipped += 1
            continue
        seen.add(digest)

        image = cv2.imread(image_path)
        if image is None:
            print("Could not load:", image_path)
            continue

        # Crop to the brain, resize, scale to 0-1 -> (channels, height, width)
        images.append(preprocess(image)[0].numpy())
        image_labels.append(label)

    return np.array(images, dtype=np.float32), np.array(image_labels), skipped


benchmark_files = [
    (path, 0 if os.path.basename(os.path.dirname(path)) == 'notumor' else 1)
    for path in image_files(benchmark_directory)
]
benchmark_hashes = {file_hash(path) for path, _ in benchmark_files}

# Load both classes, leaving out duplicates and anything in the benchmark
dataset, labels, skipped = load_images(
    [(path, label) for folder, label in [('no', 0), ('yes', 1)]
     for path in image_files(os.path.join(image_directory, folder))],
    skip_hashes=benchmark_hashes,
)
x_benchmark, y_benchmark, _ = load_images(benchmark_files, skip_hashes=set())

print(f"Loaded {len(dataset)} images ({(labels == 0).sum()} no tumor, {(labels == 1).sum()} tumor); "
      f"skipped {skipped} duplicates or benchmark images")
print(f"Benchmark images: {len(x_benchmark)}")

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

# Random variations of the training images, regenerated every epoch
train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
])


class ImageDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.images = torch.from_numpy(x)
        self.labels = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[i]


# Augmentation for training only; validation and test images stay unchanged
train_loader = DataLoader(ImageDataset(x_train, y_train, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ImageDataset(x_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(ImageDataset(x_test, y_test), batch_size=BATCH_SIZE)
benchmark_loader = DataLoader(ImageDataset(x_benchmark, y_benchmark), batch_size=BATCH_SIZE)

# Build CNN model
model = BrainTumorCNN()
print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# There are far more tumour images than healthy ones, so weight each class by
# the inverse of its frequency; otherwise the model can score well by
# answering "tumour" too often
class_counts = np.bincount(y_train, minlength=2)
class_weights = torch.tensor(len(y_train) / (2 * class_counts), dtype=torch.float32)
print("Class weights (no tumor, tumor):", class_weights.tolist())

# Integer labels work directly with CrossEntropyLoss (no one-hot needed)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    model.eval()
    total_loss, total = 0.0, 0
    true_pos = true_neg = false_pos = false_neg = 0

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total_loss += loss_fn(outputs, targets).item() * len(targets)
            total += len(targets)

            true_pos += ((predictions == 1) & (targets == 1)).sum().item()
            true_neg += ((predictions == 0) & (targets == 0)).sum().item()
            false_pos += ((predictions == 1) & (targets == 0)).sum().item()
            false_neg += ((predictions == 0) & (targets == 1)).sum().item()

    sensitivity = true_pos / max(true_pos + false_neg, 1)   # tumours caught
    specificity = true_neg / max(true_neg + false_pos, 1)   # healthy scans cleared
    return {
        'loss': total_loss / total,
        'accuracy': (true_pos + true_neg) / total,
        'balanced_accuracy': (sensitivity + specificity) / 2,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }


def report(name, metrics):
    print(f"{name}: accuracy {metrics['accuracy']:.4f} - "
          f"balanced accuracy {metrics['balanced_accuracy']:.4f} - "
          f"sensitivity {metrics['sensitivity']:.4f} - specificity {metrics['specificity']:.4f}")


# Train, saving the weights whenever validation balanced accuracy reaches a new best
best_val_score, best_epoch = 0.0, 0

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

    val = evaluate(val_loader)

    improved = val['balanced_accuracy'] > best_val_score
    if improved:
        best_val_score, best_epoch = val['balanced_accuracy'], epoch
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch}/{EPOCHS} - "
          f"loss: {total_loss / total:.4f} - accuracy: {correct / total:.4f} - "
          f"val_loss: {val['loss']:.4f} - val_accuracy: {val['accuracy']:.4f} - "
          f"val_balanced: {val['balanced_accuracy']:.4f}"
          f"{' (best, saved)' if improved else ''}", flush=True)

    if epoch - best_epoch >= EARLY_STOP_PATIENCE:
        print(f"Stopping early: no improvement for {EARLY_STOP_PATIENCE} epochs")
        break

# Test the best epoch, not the last one
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

print(f"Best epoch: {best_epoch} (val balanced accuracy {best_val_score:.4f})")
report("Test split", evaluate(test_loader))
report("Benchmark (datasets/Testing)", evaluate(benchmark_loader))
