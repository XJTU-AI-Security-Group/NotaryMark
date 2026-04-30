import torch
import torch.nn as nn


class WatermarkDetector(nn.Module):
    """Binary detector that scores whether a latent contains a watermark signal."""

    def __init__(self, channels=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, w_pred):
        """Return a single watermark-presence logit for each latent input."""
        x = self.features(w_pred)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
