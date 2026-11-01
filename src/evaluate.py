
"""Evaluate the selected VGG6 model on the CIFAR-10 test set."""

import argparse
import os
import torch
import torch.nn as nn

from src.data import make_loaders
from src.model import VGG6


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from a saved checkpoint."""

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Support both plain state dictionaries and checkpoint dictionaries.
    if isinstance(checkpoint, dict):
        state_dict = checkpoint

        for key in (
            "model_state_dict",
            "state_dict",
            "model"
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    else:
        state_dict = checkpoint

    # Handle models saved using DataParallel.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict)
    return model


def evaluate(model, test_loader, criterion, device):
    """Calculate test loss and top-1 accuracy."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VGG6 on CIFAR-10 test data."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_epwq34m8.pth"
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data"
    )
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Model: VGG6")
    print("Activation: GELU")
    print("Checkpoint:", args.checkpoint)

    # Load CIFAR-10 test data using the existing data pipeline.
    _, _, test_loader = make_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        num_workers=args.num_workers
    )

    # Initialize the selected VGG6 architecture.
    model = VGG6(
        activation="GELU",
        num_classes=10
    )

    model = load_checkpoint(
        model,
        args.checkpoint,
        device
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Final evaluation.
    test_loss, test_accuracy, correct, total = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n" + "=" * 50)
    print("FINAL CIFAR-10 TEST RESULTS")
    print("=" * 50)
    print(f"Test samples       : {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test loss          : {test_loss:.4f}")
    print(f"Test top-1 accuracy: {test_accuracy:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
