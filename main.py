
"""Main entry point for VGG6 CIFAR-10 experiments."""

import argparse
import os
import torch
import torch.nn as nn

from src.config import DEFAULTS
from src.train import run_training
from src.data import make_loaders
from src.model import VGG6
from src.engine import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="VGG6 CIFAR-10 Experiment Runner"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "activation",
            "optimizer",
            "hyperparameter",
            "test"
        ],
        default="baseline"
    )

    parser.add_argument("--activation", default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained model checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = DEFAULTS.copy()

    for key in [
        "activation",
        "optimizer",
        "learning_rate",
        "batch_size",
        "epochs"
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.mode != "test":
        run_training(config)
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        raise ValueError(
            "Please provide the selected model checkpoint "
            "using --checkpoint."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    saved_config = checkpoint["config"]

    model = VGG6(
        activation=saved_config["activation"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    _, _, test_loader = make_loaders(
        batch_size=saved_config["batch_size"],
        seed=saved_config["seed"],
        num_workers=saved_config["num_workers"]
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
