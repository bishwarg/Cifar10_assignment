
"""VGG6 model supporting multiple activation functions."""

import torch
import torch.nn as nn


def get_activation(name):
    activations = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "SiLU": nn.SiLU,
        "GELU": nn.GELU
    }

    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")

    return activations[name]()


class VGG6(nn.Module):

    def __init__(self, activation="ReLU", num_classes=10):
        super(VGG6, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            get_activation(activation),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
