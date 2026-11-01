
"""Training pipeline for VGG6 on CIFAR-10 with W&B tracking."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import wandb

from src.config import DEFAULTS
from src.data import make_loaders
from src.model import VGG6
from src.engine import train_one_epoch, evaluate


CHECKPOINT_DIR = "/content/CS6886W_VGG6_Assignment1/checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(name, model, config):
    name = name.lower()
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    momentum = config["momentum"]

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )

    elif name == "nesterov":
        return torch.optim.SGD(
            model.parameters(), lr=lr,
            momentum=momentum, nesterov=True,
            weight_decay=weight_decay
        )

    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    elif name == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    elif name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )

    elif name == "nadam":
        return torch.optim.NAdam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def run_training(config=None):
    config = {**DEFAULTS, **(config or {})}

    set_seed(config["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    print("Configuration:", config)

    # Reuse an existing W&B agent run during sweeps.
    # Otherwise, create a normal W&B run.
    if wandb.run is None:
        run = wandb.init(
            project="VGG6-CIFAR10",
            config=config
        )
    else:
        run = wandb.run
        config = {**config, **dict(wandb.config)}

    try:
        train_loader, val_loader, test_loader = make_loaders(
            batch_size=config["batch_size"],
            seed=config["seed"],
            num_workers=config["num_workers"]
        )

        model = VGG6(
            activation=config["activation"]
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = get_optimizer(
            config["optimizer"], model, config
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        best_val_accuracy = 0.0
        best_epoch = 0

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        for epoch in range(config["epochs"]):

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device
            )

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # Log all metrics after every epoch.
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

            print(
                f"Epoch [{epoch + 1}/{config['epochs']}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch + 1

                # Save a separate checkpoint for every experiment.
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"best_model_{run.id}.pth"
                )

                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_accuracy": best_val_accuracy
                }, checkpoint_path)

        wandb.summary["best_val_accuracy"] = best_val_accuracy
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["checkpoint"] = (
            f"best_model_{run.id}.pth"
        )

        print("\nBest validation accuracy:", best_val_accuracy)
        print("Best epoch:", best_epoch)

        return model, history, test_loader, device

    finally:
        wandb.finish()
