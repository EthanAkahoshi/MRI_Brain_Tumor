import torch.nn as nn

INPUT_SIZE = 128
CLASS_NAMES = ['No tumor', 'Tumor']


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super().__init__()

        # 128 -> 64 -> 32 -> 16 -> 8
        self.features = nn.Sequential(
            conv_block(3, 32),
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
        # Returns raw scores (logits); apply softmax for probabilities
        return self.classifier(self.features(x))
