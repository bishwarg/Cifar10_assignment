
"""Configuration settings for VGG6 on CIFAR-10."""

DEFAULTS = {
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 20,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    "seed": 42,
    "dropout": 0.3,
    "num_workers": 2
}
