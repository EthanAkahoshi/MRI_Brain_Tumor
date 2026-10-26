import sys
from pathlib import Path

import cv2
import torch

from model import BrainTumorCNN, CLASS_NAMES, preprocess

MODEL_PATH = Path(__file__).resolve().parent / 'BrainTumor10Epochs.pt'
DEFAULT_IMAGES = Path(r'Z:\GoogleDownloads\Brain\pred')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def load_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return preprocess(image)


def collect_images(paths):
    images = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            images.extend(sorted(
                p for p in path.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ))
        else:
            images.append(path)
    return images


model = BrainTumorCNN()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

image_paths = collect_images(sys.argv[1:] or [DEFAULT_IMAGES])
if not image_paths:
    sys.exit("No images found.")

with torch.no_grad():
    for image_path in image_paths:
        image = load_image(image_path)
        if image is None:
            print("Could not load:", image_path)
            continue

        probabilities = torch.softmax(model(image), dim=1)[0]
        predicted = int(probabilities.argmax())

        print(f"{image_path.name}: {CLASS_NAMES[predicted]} "
              f"({probabilities[predicted].item():.1%})")
