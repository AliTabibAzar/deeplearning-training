import os
import re
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def parse_training_log(log_path="training.log"):
    """
    فایل لاگ رو می‌خونه و مقادیر loss و MAE رو استخراج می‌کنه
    """
    epochs = []
    train_losses = []
    val_losses = []
    val_maes = []
    
    pattern = r"epoch (\d+): train_loss=([\d.]+) \| val_loss=([\d.]+) \| val_mae=([\d.]+)"
    
    if not os.path.exists(log_path):
        logger.error("Log file not found: %s", log_path)
        return None
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
                val_maes.append(float(match.group(4)))
    
    if not epochs:
        logger.error("No training data found in log file")
        return None
    
    logger.info("Parsed %d epochs from log file", len(epochs))
    
    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes
    }


def plot_training_curves(data, save_path="training_curves.png"):
    if data is None:
        return
    
    epochs = data["epochs"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(epochs, data["train_losses"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, data["val_losses"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, data["val_maes"], "g-o", label="Val MAE", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (years)")
    axes[1].set_title("Validation Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info("Saved training curves to: %s", save_path)


def plot_from_log(log_path="training.log", save_path="training_curves.png"):
    data = parse_training_log(log_path)
    if data:
        plot_training_curves(data, save_path)
        return True
    return False