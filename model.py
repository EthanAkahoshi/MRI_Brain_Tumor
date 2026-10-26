import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 64
CLASS_NAMES = ['No tumor', 'Tumor']


def preprocess(image_bgr):
    # Same preprocessing as mainTrain.py: BGR -> RGB, resize, scale to 0-1
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = image.astype(np.float32) / 255.0

    # (height, width, channels) -> (1, channels, height, width)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


class BrainTumorCNN(nn.Module):
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

        # 64 -> 62 -> 31 -> 29 -> 14 -> 12 -> 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

        nn.init.kaiming_uniform_(self.features[3].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.features[6].weight, nonlinearity='relu')

    def forward(self, x):
        # Returns raw scores (logits); apply softmax for probabilities
        return self.classifier(self.features(x))
