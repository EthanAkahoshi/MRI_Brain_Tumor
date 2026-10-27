import sys
from pathlib import Path

import cv2

from model import CLASS_NAMES, DECISION_THRESHOLD, load_model, preprocess, tumor_probability

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
            # Subfolders too (e.g. datasets/yes/glioma)
            images.extend(sorted(
                p for p in path.rglob('*')
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ))
        else:
            images.append(path)
    return images


model = load_model()

image_paths = collect_images(sys.argv[1:] or [DEFAULT_IMAGES])
if not image_paths:
    sys.exit("No images found.")

for image_path in image_paths:
    image = load_image(image_path)
    if image is None:
        print("Could not load:", image_path)
        continue

    p = tumor_probability(model, image)
    predicted = int(p >= DECISION_THRESHOLD)
    score = p if predicted else 1 - p

    print(f"{image_path.name}: {CLASS_NAMES[predicted]} ({score:.1%})")
