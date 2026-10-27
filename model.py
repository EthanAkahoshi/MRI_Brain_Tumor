from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 128
CLASS_NAMES = ['No tumor', 'Tumor']
BACKGROUND_THRESHOLD = 20

# Final model: source-balanced BN CNN with preprocess() below, picked on validation
# (experiments/engineer, run prepv2_srcbal_seed1_v2). Its validation threshold is 0.5,
# so "Tumor" exactly when P(tumor) >= 0.5, the same as taking the larger class score.
MODEL_PATH = Path(__file__).resolve().parent / 'BrainTumorFinal.pt'
DECISION_THRESHOLD = 0.5


def crop_to_brain(image_bgr):
    # Cut away the black border around the head, so the model can't use the
    # framing of the scan as a shortcut
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, BACKGROUND_THRESHOLD, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)

    if w == 0 or h == 0:
        # Nothing brighter than the background: keep the whole image
        return image_bgr
    return image_bgr[y:y + h, x:x + w]


def pad_to_square(image):
    # Centre on a black square. The web app pads its display image with this too,
    # which is what keeps the heatmap lined up with the scan.
    h, w = image.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side) + image.shape[2:], image.dtype)
    out[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = image
    return out


def preprocess(image_bgr):
    # Used by the final model, mainTest.py and the web app. Removes the source cues
    # the shortcut relied on: crop, pad to square (aspect ratio), grayscale (colour
    # overlays), and stretch the brain's 1st-99th percentile to 0-1 (brightness).
    square = pad_to_square(cv2.cvtColor(crop_to_brain(image_bgr), cv2.COLOR_BGR2GRAY))
    small = cv2.resize(square, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    brain = small[small > BACKGROUND_THRESHOLD]
    if brain.size:
        lo, hi = np.percentile(brain, [1, 99])
        out = np.clip((small - lo) / max(hi - lo, 1.0), 0, 1)
    else:
        out = small / 255.0
    out = np.round(out * 255) / 255  # uint8-exact, so the training cache matches inference
    return torch.from_numpy(np.repeat(out[None].astype(np.float32), 3, 0)).unsqueeze(0)


def preprocess_legacy(image_bgr):
    # Older pipeline (crop, BGR -> RGB, resize, scale to 0-1). Used by BrainTumorCNN
    # checkpoints, mainTrain.py and the earlier experiments; keep for reproducibility.
    image = cv2.cvtColor(crop_to_brain(image_bgr), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = image.astype(np.float32) / 255.0

    # (height, width, channels) -> (1, channels, height, width)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class BrainTumorBNCNN(nn.Module):
    # Final model: 5 conv blocks with batch norm and global average pooling
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, x):
        # Raw logits; softmax for probabilities
        return self.classifier(self.features(x))


def load_model(path=MODEL_PATH):
    model = BrainTumorBNCNN()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def tumor_probability(model, tensor):
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0, 1].item()


class BrainTumorCNN(nn.Module):
    # Legacy 3-block model (BrainTumorBest*.pt checkpoints, preprocess_legacy)
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 128 -> 126 -> 63 -> 61 -> 30 -> 28 -> 14
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

        nn.init.kaiming_uniform_(self.features[3].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.features[6].weight, nonlinearity='relu')

    def forward(self, x):
        # Returns raw scores (logits); apply softmax for probabilities
        return self.classifier(self.features(x))
